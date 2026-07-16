# Findings: vLLM 0.25.1 e2e — native deep_gemm mega vs flashinfer moe_ep

**Date:** 2026-07-15 · **Model:** DeepSeek-V4-Flash (4096 h / 2048 moe-inter /
256 experts / top-6 / 43 layers, fp4 experts + ue8m0-32 scales) ·
**Hardware:** 4x GB200 (one node), TP=4 + EP=4 · **Stack:** vLLM 0.25.1 wheel,
flashinfer branch `new_cutedsl_kernels` (0.6.15, editable), deep_gemm from the
fi-ep container, CuTe-DSL 4.6.1.

## Headline throughput — offline repeat matrix (2026-07-15 23:00, DEFINITIVE)

`bench_offline.py` / `run_offline_matrix.sh`: one engine boot per cell,
5 timed rounds after warmup, **prefix caching off** (with it on, repeat
rounds are 100% cache hits — observed fake 91k tok/s "prefill"), eager, 128
prompts, fixed token-id prompts (seed 0). fi wrapper at full native parity:
validated-once fast path + one shared symm workspace across all 43 layers.
Round spreads < 3% — engine-restart variance eliminated.

| cell (tok/s median) | native | fi_dg | fi_nvfp4 | fi_dg/nat | nvfp4/nat |
|---|---|---|---|---|---|
| prefill 1024in/1out | 31777 | 28350 | 25843 | **0.89x** | **0.81x** |
| decode 128in/256out | 1681 | 1421 | 1318 | **0.85x** | **0.78x** |

JSONs: `results/offline_20260715_221930_*.json`.

Interpretation:
- Even with dispatch fast path + shared workspace, **fi_dg trails native by
  11-15%** — the remaining gap is NOT per-call validation (removed) and NOT
  workspace duplication (removed). Prime suspect: staging. Native uses ONE
  fused `prepare_megamoe_inputs` kernel; fi's `stage_mega_moe_inputs` is
  `per_token_cast_to_fp8` (allocating) + 4 separate copies into the
  workspace = ~5-6x the launches+allocs per layer per step. Needs a profile
  to confirm (next step).
- **fi_nvfp4 is SLOWER than fi_dg** (0.91x of fi_dg at prefill) despite the
  kernel being 1.35-1.65x faster than dg at this tokens/step in the
  microbench. Prime suspect: the CuTeDSL frontend's launch-kwargs cache keys
  on tensor identity — in the microbench the same input tensors are reused
  every iteration (cache hits), but vLLM hands the layer FRESH
  hidden_states/topk tensors every step, forcing the full cute-tensor-view
  rebuild (from_dlpack x12) per layer per step. This is the same artifact
  documented in the 2026-07-14 microbench methodology fix, now appearing in
  production form. Fix direction: stage into fixed buffers before launch so
  cached launch kwargs stay valid.
- The MoE layer is only part of step time, so kernel-level wins/losses are
  diluted ~2-3x in these e2e numbers.

## nsys attribution — where the fi perf goes (2026-07-15 23:00)

Profiles: `results/nsys_20260715_225236_{native,fi_dg,fi_nvfp4}.nsys-rep`
(+ `_cuda_gpu_kern_sum.csv` / `_cuda_api_sum.csv`), prefill workload, 64
prompts, warmup + 2 rounds, CUDA-trace only. Caveat: nsys adds per-API-call
host overhead, so wall times under the profiler are biased *against*
launch-heavy backends — use it for kernel self-times and counts, not wall.

**1. The cutedsl nvfp4 mega kernel is NOT faster than dg at this model's
geometry.** GPU self-time per launch (same engine, same steps):

| mega kernel | avg / launch | instances |
|---|---|---|
| native `deep_gemm::sm100_fp8_fp4_mega_moe` | **1176 µs** | 8428 |
| fi_dg same kernel | 1297 µs (launch-skew-inflated, see below) | 8428 |
| fi_nvfp4 `Sm100MegaMoEKernel` (cutedsl) | **1464 µs** | 9632 |

The microbench's 1.35–1.65x nvfp4 win was at (hidden 7168, top-8); this
model is (hidden 4096, moe-inter 2048, 256E, **top-6**) — a geometry the
kernel sweep never covered, running on `default_knobs` profiles derived at
7168-hidden. At 4096-hidden the dg kernel's per-token work also halves. So
the kernel-level advantage itself did not transfer. Actions: run
`FI_MOE_EP_KNOBS=auto` (online autotune at this geometry), and add
(4096, 2048, 256, top6) to the kernel-repo tuning sweep.

**2. Staging kernel soup — the integration-side eater.** cudaLaunchKernel
calls for the identical workload:

| backend | cudaLaunchKernel | memcpyAsync | approx torch kernels per MoE-layer step |
|---|---|---|---|
| native | 100k | 128k | ~1 (fused `_prepare_megamoe_inputs_kernel`, 63.8 µs) |
| fi_dg | 330k (3.3x) | 185k | ~5-10 (`per_token_cast_to_fp8` reduce+elementwise+4 copies) |
| fi_nvfp4 | **946k (9.4x)** | 231k | **~98** (input-quant + staging torch soup: 97k elementwise, 63k copies, 54k binary, 44k index, 32k reduce ≈ 8.2 s GPU) |

fi_nvfp4 spends ~8.2 s of GPU time in small torch kernels vs native's 0.6 s
fused staging — plus the matching host-side launch cost and the cutedsl
launch-kwargs cache misses (fresh vLLM tensors each step) documented above.

**3. Launch-skew redistribution, not allreduce difference.** The TP
allreduce avg swings wildly (native 646 µs, fi_dg 327 µs, fi_nvfp4 770 µs at
identical counts) because whichever collective/mega kernel arrives last
absorbs the inter-rank skew as spin-wait. fi_dg's mega-kernel avg being 10%
above native's (same kernel, same inputs) is this skew, driven by its extra
host-side staging time — total GPU busy for fi_dg is actually LOWER than
native (30.3 vs 34.3 s) while wall is higher: fi_dg is host-gap-bound.

**Bottom line:** the e2e deficit decomposes into (a) kernel: nvfp4 cutedsl
needs retuning/sweeping at this model's geometry — it is currently ~25%
slower per launch than dg here; (b) integration: fi staging must become one
fused kernel writing the symm buffers directly (native parity) and the
cutedsl frontend needs fixed-address staging so its launch-kwargs cache
hits. Fixing (b) alone should roughly close fi_dg's 11-15% gap; fixing both
is required before fi_nvfp4 can win e2e.

## Superseded: cross-engine `vllm bench throughput` cells (eager, 128 prompts)

**Methodology notes discovered after the fact:** (1) these cells boot a
fresh engine per measurement — prefill-heavy cells showed ±35% cross-restart
variance; (2) the "prefill" cells actually generated ~128 output tokens per
prompt (the dataset ignores `--output-len 1`), so they are mixed workloads,
not pure prefill. Kept as raw data only — use the repeat matrix above.

CSV: `results/bench_20260715_205636.csv` (per-cell JSON alongside).
Identical prompt sets (seed 0) and forced identical output token counts
(`ignore_eos=True`) per cell.

**⚠ Run-to-run variance dominates single measurements.** The prefill cell
flipped between repeats (run 1: native 4221 / fi_dg 5864; run 2: native 6465 /
fi_dg 4862 — each side "won" by ~35% once), with identical prompts and token
counts. Cross-engine-restart variance (suspect: per-run NCCL algo selection
for the TP allreduces, JIT/cache warmth) is larger than any backend delta.
All conclusions below use medians over repeats; single-run tables are kept
only as raw data.

Raw cells so far (tok/s total):

| workload (in/out) | native r1 | fi_dg r1 | native r2 | fi_dg r2 |
|---|---|---|---|---|
| prefill 1024/1  | 4221 | 5864 | 6465 | 4862 |
| decode 128/256  | 6598 | 6163 | (running) | (running) |
| mixed 512/128   | 7186 | 6408 | (running) | (running) |

- **fi_dg = flashinfer moe_ep `deep_gemm_mega` kernel** — the *same*
  `deep_gemm.fp8_fp4_mega_moe` kernel as native, argument-identical launch,
  bit-identical input staging (proven, see below). Any true delta is
  layer/runtime plumbing only; current data says fi_dg ≈ native within
  (large) noise.
- **Both sides eager**: fi-path CUDA-graph compatibility untested; graphs-on
  native decode would be faster in production. Fair A/B, conservative
  absolutes.

## Correctness

Method: 8 fixed prompts, greedy 64 tokens, per-token logprobs
(`smoke_infer.py` / `compare_outputs.py`, dumps in `results/smoke_*.json`).

| comparison | greedy exact | mean \|dlogprob\| (common prefix) |
|---|---|---|
| native vs native (rerun) | **8/8** | 0.0000 |
| native vs fi_dg | 3/8 | 0.01–0.06 |
| fi_dg vs fi_dg (rerun) | 2/8 | 0.01–0.08 |

- Native is fully deterministic run-to-run.
- **fi_dg is NOT deterministic run-to-run**, with self-divergence the same
  magnitude as its divergence from native → fi_dg is at *statistical parity*
  with native (all outputs coherent and quality-equivalent), but some source
  of nondeterministic accumulation exists in the fi launch context. OPEN
  QUESTION — staging is bit-exact and kernel args identical, so suspicion is
  on kernel-internal scheduling sensitivity to context (SM occupancy from
  other engine work, stream state). Chase with nsys / single-layer dumps.
- Supporting probe: vLLM's fused `prepare_megamoe_inputs` and fi's
  `stage_mega_moe_inputs` produce **bit-identical** fp8 x / packed-ue8m0 sf /
  topk buffers on identical inputs.

## Integration bugs found (fixed in `patch_0251/`)

1. **Stale `LOCAL_RANK` in vLLM workers vs flashinfer device binding.**
   flashinfer's runtime/layer constructors call
   `torch.cuda.set_device(LOCAL_RANK or bootstrap.rank)`; vLLM workers carry a
   `LOCAL_RANK` that doesn't match the worker's assigned device, so weight
   transforms launched on the wrong GPU → `CUDA_ERROR_ILLEGAL_ADDRESS` in
   `deep_gemm.transform_sf_into_required_layout` at load. Fix: pin
   `os.environ["LOCAL_RANK"] = str(torch.cuda.current_device())` before
   bootstrap (fi_utils). Consider upstream: flashinfer should not rebind the
   device when one is already bound.
2. **`FleetParams.weights` API move** — branch moved weights to
   `MoEEpLayer(weights=...)`; old 0.24-era patch passed it inside FleetParams.

## Dependency matrix for vLLM 0.25.1 on the fi-ep container (aarch64)

All encoded in `setup_container.sh`; every bullet was a real breakage:

| package | vllm 0.25.1 pin | what we run | why |
|---|---|---|---|
| torch | 2.11.0 | 2.11.0+cu130 (venv) | vllm ops use stable-libtorch ABI → NGC torch 2.12 mismatch is fine |
| nvidia-cutlass-dsl | 4.5.2 | **4.6.1** | 4.5.2 compiles fi cutedsl mega kernels 34–54% slower (TUNING.md) |
| quack-kernels | >=0.3.3 (resolves old) | 0.6.1 | old quack breaks on dsl 4.6 (`cute.core.ThrMma` removed) |
| (vendored) vllm cute kernels | — | sed `cute.core.ThrMma`→`cute.ThrMma` | pure rename in dsl 4.6 |
| apache-tvm-ffi | 0.1.9 | **0.1.11** | dsl 4.6.1 needs `make_kwargs_wrapper(map_dataclass_to_tuple=)`; 0.1.12 breaks tilelang |
| tilelang | ==0.1.9 | 0.1.12 | 0.1.9 breaks on tvm-ffi >0.1.9 registry; 0.1.12 caps tvm-ffi at 0.1.11 |
| flashinfer-python | ==0.6.13 | branch 0.6.15 editable, `--no-deps` | pip 24.0 resolver also crashes on NGC dist metadata |

## Status / next steps

- [x] native + fi_dg e2e working, benchmarked (this doc)
- [ ] prefill repeat run (in flight)
- [ ] fi_nvfp4 smoke (in flight; checkpoint fp4 → bf16 dequant → nvfp4 requant
      path, expect larger logprob delta from double quantization)
- [ ] fi_dg nondeterminism root-cause
- [ ] CUDA-graph compatibility of fi path (then graphs-on sweep)
- [ ] serve-mode (DP4 TP1 EP) benchmark with TTFT/TPOT, matching the old
      `run_deepseek_v4_flash.sh` topology
