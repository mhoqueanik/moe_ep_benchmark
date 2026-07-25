# vLLM 0.25.1 e2e: DeepSeek-V4-Flash with native deep_gemm mega vs flashinfer moe_ep

End-to-end vLLM inference benchmark comparing:

- **native** — vLLM 0.25.1's built-in `deep_gemm_mega_moe` MoE backend
  (`DeepseekV4MegaMoEExperts`, `deep_gemm.fp8_fp4_mega_moe`)
- **fi_dg** — flashinfer `moe_ep` mega path with the `deep_gemm_mega` kernel
  (same deep_gemm kernel underneath, flashinfer runtime/layer on top)
- **fi_nvfp4** — flashinfer `moe_ep` CuTeDSL mega kernel (`nvfp4_cutedsl`;
  an NVFP4 checkpoint is consumed prequantized, while MXFP4 weights are
  dequantized to bf16 at load and requantized)

Model: DeepSeek-V4-Flash (hidden 4096, moe_inter 2048, 256 experts, top-6,
43 layers) from
`/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-ai_deepseek-v4-flash/hf/hf-6e76323_orig`.
4x GB200 (SM100), TP=4 + expert parallel (EP=4).

> Reproducing the vLLM PR specifically (the `flashinfer_moe_ep_mega_*`
> backend strings that replaced the `FI_MOE_EP=1` opt-in)? Start from
> **`RUNBOOK_VLLM_PR.md`** — patch, tier-1/tier-2 validation, expected
> outputs, and the known coverage gaps.

## 0. Layout

```
vllm_e2e/
├── patch_0251/            # fi integration for the vLLM 0.25.1 wheel
│   ├── model.py           # 0.25.1 deepseek_v4/nvidia/model.py + 5 fi hunks
│   ├── fi_utils.py        # flashinfer moe_ep glue (wrapper experts class)
│   ├── apply.sh / reset.sh
├── setup_container.sh     # venv + vllm 0.25.1 + flashinfer branch + patch
├── in_container.sh        # run a command in the held-node container
├── smoke_infer.py         # correctness: greedy gen + logprob dump
├── compare_outputs.py     # diff two smoke dumps
├── bench_throughput.sh    # vllm bench throughput sweep -> CSV
├── results/               # CSVs, JSONs, smoke dumps
└── logs/
```

## 1. Get a node + container

```bash
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
# 4h hold job (node with 4 GPUs)
JOBID=$(sbatch --parsable -A coreai_libraries_cudnn -p batch -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J coreai_libraries_cudnn-fi.vllm_e2e.hold \
    --output=$W/logs/hold_%j.log --wrap "sleep 14400")
# every command below goes through:
JOBID=$JOBID bash $W/in_container.sh '<command>'
```

`in_container.sh` uses the flashinfer-ep image
(`$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh`), mounts `$ROOT` and
`/lustre/share` (model, read-only), and reuses the named container `fivllm`
across calls.

## 2. One-time setup (per venv; survives container/node restarts)

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash setup_container.sh'
```

What it does (see the script):
1. venv (`--system-site-packages`) at `vllm_e2e/venv0251` on lustre
2. `pip install vllm==0.25.1` (wheel; brings its pinned torch)
3. editable flashinfer branch install (JIT, `BUILD_NIXL_EP=0`)
4. keeps vllm's own `nvidia-cutlass-dsl==4.5.2` pin — at 4.6.1 parity since
   the 2026-07-22 MR!27 mainloop WAR in the fi branch (see
   `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/TUNING.md`). `DSL_461=1`
   restores the pre-WAR force-upgrade to 4.6.1 (+ quack/tvm-ffi/tilelang
   compat chain) that runs 37-40 were validated on.
5. applies `patch_0251/` to the installed vllm; sanity import checks
   (every line should print PASS)

## 3. Backend selection (one moe_backend string per config)

Each config is a distinct `moe_backend`. The python entry points read it from
`MOE_BACKEND`; `vllm bench throughput` takes it as `--moe-backend`:

| config   | moe_backend |
|----------|-------------|
| native   | `deep_gemm_mega_moe` |
| fi_dg    | `flashinfer_moe_ep_mega_deep_gemm` |
| fi_nvfp4 | `flashinfer_moe_ep_mega_cutedsl` |

`FI_MOE_EP` / `FI_MOE_EP_MEGAKERNEL` are retired (runs up to 2026-07-23 used
them; see RUNS.md). The patched model now *errors* if either is still
exported, because a stale export would otherwise quietly produce native
numbers under an fi label. `patch_0251/apply.sh` also patches
`vllm/config/kernel.py`, since 0.25.1's `MoEBackend` literal would reject the
new strings before the model sees them.

Optional: `FI_MOE_EP_KNOBS=auto` (online autotune of cutedsl kernel knobs at
first forward) or a JSON dict of explicit knobs.

## 4. Correctness smoke

```bash
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && MOE_BACKEND=deep_gemm_mega_moe python smoke_infer.py --tag native --out results/smoke_native.json'
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && MOE_BACKEND=flashinfer_moe_ep_mega_deep_gemm python smoke_infer.py --tag fi_dg --out results/smoke_fi_dg.json'
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && python compare_outputs.py results/smoke_native.json results/smoke_fi_dg.json'
```

Measured (2026-07-15, see FINDINGS.md): native is bit-deterministic
run-to-run (8/8 exact). fi_dg is NOT (2/8 vs itself) and lands at
*statistical parity* with native (|dlogprob| 0.01-0.06/token), despite
bit-identical staging and argument-identical kernel launches — root cause of
the nondeterminism unresolved. fi_nvfp4 shows the expected larger
double-quantization delta (|dlogprob| 0.02-0.20/token, generations coherent).
Compare logprob divergence, not exact tokens, and always collect the
native-vs-native control first.

## 5. Throughput benchmark

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash bench_throughput.sh'
# knobs: BACKENDS="native fi_dg fi_nvfp4" WORKLOADS="prefill:1024:1 decode:128:256 mixed:512:128" NUM_PROMPTS=256 EAGER=0
# detached (survives the launching shell/session):
bash $W/launch_detached.sh $JOBID my_bench 'BACKENDS="fi_nvfp4" EAGER=1 bash bench_throughput.sh'
```

Results land in `results/bench_<stamp>.csv` (+ per-run JSON from
`vllm bench throughput --output-json`).

**Repeat every cell ≥3x and use medians.** Prefill-heavy cells showed ±35%
cross-engine-restart variance on identical inputs (decode/mixed cells were
stable within ~2%). `vllm bench throughput` fixes the prompt set (seed 0) and
forces output lengths (`ignore_eos=True`), so per-cell token counts are
identical across backends — the variance is engine-side (suspect NCCL algo
selection per run).

## 6. Results

See `FINDINGS.md`.

## 7. Known caveats

- vLLM 0.25.1 pins `nvidia-cutlass-dsl==4.5.2`; since the MR!27 WAR
  (2026-07-22) the setup keeps that pin by default (`DSL_461=1` for the old
  force-upgrade path). A venv built pre-WAR still contains 4.6.1 — `FRESH=1`
  to rebuild on 4.5.2 (the setup's DSL guard hard-fails otherwise).
  Native-4.5.2 e2e REVALIDATED (run 41: decode-1k fi_nvfp4 1.068x native,
  inside the runs-37-40 band). Only the fi cutedsl kernels consume it.
- Teardown-only cosmetic tracebacks on native 4.5.2: vLLM's
  `worker.shutdown()` imports CuMemAllocator, which trips over tilelang
  0.1.9's `libcudart_stub.so` (no `cudaDeviceReset`). Post-round,
  engine-exit path only — results unaffected (run 41). The 4.6.1 chain's
  tilelang 0.1.12 didn't have the stub issue.
- flashinfer branch = 0.6.15 vs vllm pin 0.6.13 (editable install overrides;
  resolver warning expected).
- cutedsl megakernels get the checkpoint's fp4/ue8m0-32 weights via a
  dequant-to-bf16 + requant (double quantization) — small accuracy delta vs
  native is expected and reported in FINDINGS.md.
- EPLB is not supported on the fi path (wrapper stubs the EPLB hooks).
- vLLM workers carry a stale `LOCAL_RANK`; flashinfer's internal
  `set_device(LOCAL_RANK)` then lands work on the wrong GPU (illegal memory
  access at weight load). `fi_utils.ensure_fi_moe_ep_runtime` pins LOCAL_RANK
  to the worker's current device before any fi construction — keep that if
  you refactor.
- The fi layer retains its canonical `MoEWeightPack` after preprocessing;
  the wrapper nulls it (`_mega_layer._weights = None`) — without this the
  cutedsl paths OOM (43 layers x ~3.2 GB of retained bf16 dequants).
- CUDA graphs (`EAGER=0`) are untested on the fi path — all published numbers
  are eager-mode both sides.
