# SKILL: verifying FlashInfer moe_ep and integrating it into a serving framework

Audience: someone outside the core team who wants to (A) independently
verify that `flashinfer.moe_ep` is correct and fast, and/or (B) integrate it
into a serving framework (SGLang, TRT-LLM, a custom engine) the way it is
already integrated into vLLM. This repo holds the verification harnesses:
the kernel microbenchmarks (repo root) and the full vLLM end-to-end record
(`vllm_e2e/` — runbook, experiment log, and the integration patch that is
the worked example for part B).

Sibling repo referenced below:

| repo | role |
|---|---|
| `flashinfer-2/flashinfer-moe_ep` | the library (`flashinfer/moe_ep/`), its tests, design docs |
| this repo (`moe_ep_benchmark`) | microbench + `vllm_e2e/` (RUNBOOK, RUNS, FINDINGS, COMM, integration patch) |

Hardware assumptions: GB200 / sm_100a, 4 GPUs for the standard multi-rank
suite. Software: vLLM 0.25.1 era, CuTe-DSL **4.6.1** (4.5.2 compiles the
same kernels 34–54% slower — pin it), `nvshmem4py-cu13`.

---

## A. Verifying fi moe_ep — the ladder

Run these in order; each step assumes the previous passed. Every claim we
publish has one of these behind it.

### A1. Unit + contract tests (flashinfer repo, CPU-light)

```bash
cd flashinfer-moe_ep
pytest tests/moe_ep/test_weight_pack_union.py \
       tests/moe_ep/test_mega_layer_validation.py \
       tests/moe_ep/test_layer_factory.py -q
```

What it proves: the API contracts hold — notably the `MoEWeightPack`
discriminated union (a pack with exactly one scale plane raises instead of
silently re-quantizing packed data).

### A2. Kernel vs torch oracle (single GPU)

```bash
pytest tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -q
pytest tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py -q
```

History says do not skip this: the two worst bugs so far (an N-major weight
layout that was 1.6x "faster" and wrong; a prefix-cache artifact showing
91k tok/s) both *looked like wins* until an oracle ran.

### A3. Multi-rank layer path (4 GPUs)

```bash
torchrun --nproc_per_node=4 -m pytest \
  tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py -q -m "gpu_4 and arch_blackwell"
# also: test_mega_cuda_graph_multirank.py (graph capture contract)
```

### A4. Kernel microbench vs deep_gemm (this repo, 4 GPUs)

```bash
MEGA_TIMING=e2e_pipelined CUDA_VISIBLE_DEVICES=0,1,2,3 python bench_moe_ep_mega.py \
  --world-size 4 --mega-backend nvfp4_cutedsl --tokens-per-rank 2048 \
  --num-experts 256 --top-k 8 --hidden 7168 --intermediate 2048
```

Expected (2026-07 numbers, e2e_pipelined p50): nvfp4 ~610-626us @2048 vs
deep_gemm ~820-845us; the advantage grows with tokens/rank (parity at 8,
~1.6-1.9x at 8192). The bench prints `acc_loss_pct` — a perf row without it
is not evidence. `run_sweep.sh` sweeps token counts; see the repo README
for the section layout.

### A5. End-to-end in vLLM (`vllm_e2e/`, the full recipe)

`vllm_e2e/RUNBOOK.md` is the reproduce guide (node + container + venv +
patch). The two gates before quoting any number:

1. **GSM8K oracle** (`vllm_e2e/eval_gsm8k.py`) on BOTH the backend under
   test and the native baseline — DSV4-Flash lands ~0.95-0.97 at 200
   questions; a backend outside that band is broken, not "slightly less
   accurate".
2. **Two-checkpoint policy**: nvfp4_cutedsl runs the NVFP4 checkpoint
   (prequantized weights path), deep_gemm runs the mx original — same base
   weights; `bench_offline.resolve_model` handles it and stamps the model
   path into every result JSON.

Current e2e reference points (RUNS.md runs 31-34, 4x GB200 TP4+EP4, CUDA
graphs): prefill 8k-chunks **1.18x native**; decode @1024-seq concurrency
parity-to-ahead with far tighter round-to-round spread (<1% vs native's
~15% swings); decode @128-seq 0.96x (small-batch reduce launch, known).
GSM8K: native 0.9650 / fi_nvfp4 0.9750.

Known variance traps (all hit us; all in `vllm_e2e/RUNS.md`):
fresh-engine-per-run prefill variance ±35% (boot once, repeat rounds —
`bench_offline.py` exists because of this), prefix caching fakes prefill
throughput (disable it), eager-mode decode is launch-bound for every
backend (compare under graphs only), and native decode at high concurrency
declines round-over-round (median several sessions before trusting it).

---

## B. Integrating into a serving framework (SGLang etc.)

### The surface

```python
from flashinfer.moe_ep import (
    MoEEpLayer, BootstrapConfig, FleetParams, MegaConfig,
    Nvfp4CutedslMegaMoeConfig, MoEWeightPack, MoEEpTensors,
)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size, rank, process_group, auto_bootstrap=False),
    fleet_params=FleetParams(num_experts, max_tokens_per_rank, token_hidden_size),
    weights=MoEWeightPack(w13, w2[, w13_scale, w2_scale]),
    backend=MegaConfig(megakernel=Nvfp4CutedslMegaMoeConfig(...),
                       preprocess_weights=True, quantize_input=True),
)
layer.warmup()                      # COLLECTIVE — all EP ranks together
y = layer(MoEEpTensors(hidden_states, topk_ids, topk_weights))
```

The worked example is the vLLM wrapper in this repo:
`vllm_e2e/patch_0251/fi_utils.py` (+ design docs in
`flashinfer-moe_ep/docs/design_docs/`, esp. `vllm_moe_ep_integration.md`
and `moe_ep_architecture.md`). Port its *decisions*, not its vLLM plumbing.

### The contract checklist (each item is a bug we found or prevented)

1. **Bootstrap / device binding.** Pass your framework's EP process group in
   `BootstrapConfig`. Pin `LOCAL_RANK` to the device your framework bound
   BEFORE building layers — a mismatched rebind runs weight transforms
   against another GPU's pointers (CUDA_ERROR_ILLEGAL_ADDRESS at load).
2. **Warmup before CUDA-graph capture, on ALL EP ranks.** The lazy paths
   (symmetric-heap alloc, `cute.compile`, module load) are collective and
   raise if they fire mid-capture. One full eager forward per rank
   (`layer.warmup()`) makes `forward` capturable.
3. **Weights.** `MoEWeightPack` is a discriminated union: bf16 canonical
   (backend quantizes at init) or pre-quantized packed fp4 + BOTH scale
   planes (consumed verbatim; NVFP4 recipe = e4m3 per-16). Passing exactly
   one scale raises by design. For NVFP4 checkpoints, map the per-tensor
   globals to `fc1_alpha`/`fc2_alpha` (per-expert) and leave activation
   quant dynamic — the wrapper's `nvfp4_prequant_pack_and_alphas` is the
   reference, including the e4m3-exact gate/up scale-fold rule. Release the
   source pack after init (multi-GB per layer otherwise).
4. **One shared workspace across layers.** Workspaces pool by geometry
   (`workspace_pool`); never allocate per-layer (43x symmetric memory).
5. **Steady-state host path.** `MoEEpMegaLayer.forward` re-validates every
   call (~100us/layer-call — a measured skew generator at 43 layers). After
   the first forward, go straight to `kernel.stage_inputs` +
   `kernel.compute` (the wrapper's "fast path"), and note:
   `compute(output=None)` (zero-copy view) is a **cutedsl-backend contract
   only** — deep_gemm needs a real output tensor.
6. **Tuning.** Never `knobs="auto"` in an engine (minutes of collective
   compiles at first forward). Tune offline (`python -m
   flashinfer.moe_ep.tune`, same EP world/geometry/buckets as production),
   ship the JSON knob cache, point `FLASHINFER_MOE_EP_KNOB_CACHE` at it;
   per-role caches (prefill-tuned vs decode-tuned engines) are the
   deployment pattern. Untuned geometries fall back to heuristics — fine to
   boot, leaves perf on the table.
7. **Feed the kernel big batches.** Its advantage grows with tokens/rank
   (parity at 8 -> 1.6-1.9x at 8192 in microbench). Framework-side this
   means: large prefill chunks (8k if memory allows — with a SPARSE CUDA
   graph capture-size list, or graph pools eat all KV memory), high decode
   concurrency, and capture sizes covering your real step shapes.
8. **Determinism.** Default config is bitwise-deterministic per launch
   (`in_kernel_fc2_reduce=False`). Engine-level output nondeterminism
   across runs is batch-formation timing, not the kernel — verify with
   shape logs before blaming the backend.
9. **Known gaps** (state them, don't discover them): single NVLink domain
   only (multi-node EP is a TODO), EPLB hooks stubbed, MTP/spec-decode
   interaction unvalidated (and NVFP4 checkpoints may keep mx-format MTP
   experts — recipe detection must be per-module before enabling it).

### Suggested SGLang porting order

1. Boot single-rank (`MEGA_NO_DIST=1`) with bf16 weights + `quantize_input=True`
   on a toy geometry; verify vs the A2 oracle.
2. 4-rank EP with your scheduler's process group; A3-equivalent parity test.
3. Weight loading from your checkpoint format (bf16 first, prequant NVFP4
   second); GSM8K gate vs your existing MoE path.
4. CUDA graphs (warmup contract), then the fast path, then offline tunes.
5. Only then benchmark — with the A5 variance traps in mind.

---

## Where to read more

* `flashinfer-moe_ep/flashinfer/moe_ep/kernel_src/cutedsl_megamoe/TUNING.md`
  — knob system, measured sweeps, benchmark methodology and its pitfalls.
* `vllm_e2e/RUNS.md` (this repo) — the full experiment log (every number
  above traces to a run entry); `FINDINGS.md` — analysis; `COMM.md` —
  measured communication patterns (latency-bound at intranode scale; EP8
  pre-sizing).
* `flashinfer-moe_ep/flashinfer/moe_ep/todo_*.md` — honest open-items list.
