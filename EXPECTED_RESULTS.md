# Expected Results — fi moe_ep vs Cursor Mixture-of-Kittens (MoK)

Reference numbers for the MXFP8 forward-only (inference) microbenchmark
comparison between FlashInfer's `mxfp8_cutedsl` mega backend and Cursor's
Mixture-of-Kittens megakernel. Rerun instructions: RUNBOOK.md §"MoK
comparison"; all launch scripts live in `mok_comparison/`.

## Environment of record (2026-08-04)

- Cluster: ptyche, 1 node, 4× GB200 (SM100, compute cap 10.0), NVLink
- Image: `flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh`
  (torch 2.12.0a0+nv26.05 / CUDA 13.2, nvcc 13.2, Python 3.12)
- fi checkout: `flashinfer-2/flashinfer-moe_ep` branch
  `sm90_implementation_vincent` @ `d9c749c3`
- MoK: https://github.com/cursor/mixture-of-kittens @ `8f90b74`
  ("Initial public release"), ThunderKittens submodule @ `1c3920d9`,
  built with `MOK_ARCH=SM100`, plus
  `mok_comparison/mok_sm100_capability_check.patch` (upstream hard-codes an
  SM103-only runtime check that contradicts its README's SM100 build path)
- Harness: `bench_moe_ep_mega.py` with the `kernel_src/sm100/` module alias
  (in this branch)

## Workload

2048 tokens/rank, 384 experts (96/rank), top-k 6, hidden 7168,
intermediate 3072, EP = world size = 4, bf16 activations in/out.

## Results

All rows below run the SAME workload: routed experts + a shared expert
(MoK fuses its shared expert into the megakernel; the fi rows include a
MoK-parity shared expert in the timed region via `MEGA_SHARED_EXPERT=1`).
The "+quant" fi rows additionally time the fused bf16→MXFP8 activation
quant + staging kernel (`MEGA_TIMED_QUANT=1`), matching MoK's timed scope
exactly: bf16 tokens in → bf16 combined output out, everything in between
on the clock. **Quote the bold full-scope rows against MoK.**

| Variant | Fwd latency p50 | Effective TFLOP/s |
|---|---|---|
| **fi + shared + timed quant (full MoK scope), tuned, `e2e_pipelined`** | **1.832 ms** | **~1034** |
| fi + shared + timed quant (full MoK scope), tuned, `e2e` | 1.942 ms | ~975 |
| fi + shared, staging excluded, tuned, `e2e_pipelined` | 1.782 ms | ~1063 |
| fi + shared, staging excluded, tuned, `e2e` | 1.867 ms | ~1014 |
| **MoK MXFP8 forward (routed + shared)** | **2.562 ms** | **739** |
| MoK BF16 forward (routed + shared) | 3.708 ms | 511 |

Headline (bold rows, identical scope, `e2e_pipelined` = the MoK-comparable
timing mode): **fi is ~1.40× faster than MoK** (1.832 ms vs 2.562 ms).
The fused quant+staging kernel costs fi ~50 µs; the staging-excluded rows
are kept only to show that delta. CSVs of record:
`results/bench_moklike_{sharedquant,shared}_{pipelined,e2e}_fi_mega.csv`.

Do not benchmark fi without `MEGA_SHARED_EXPERT=1 MEGA_TIMED_QUANT=1` when
comparing against MoK: anything less under-counts fi's work and inflates
the ratio.

### Timing modes (plain language)

`MEGA_TIMING` selects how iterations are launched — the timed work is the
same full FI forward either way:

- `e2e_pipelined` — steady-state: iterations enqueued back-to-back, no
  per-iteration barrier/sync (like a serving/training pipeline). **This is
  the mode comparable to MoK's harness**, which also times back-to-back
  forwards with per-iteration CUDA events.
- `e2e` — cold-start: each iteration launched from a global barrier +
  device sync on an idle GPU, so samples include collective start-up skew.
  Always ≥ `e2e_pipelined`; the difference isolates launch/collective
  overhead.
- `kernel` — bare pre-built kernel launch (tester parity); excludes the FI
  host wrapper. Not comparable to MoK; incompatible with
  `MEGA_SHARED_EXPERT`.

`MEGA_SHARED_EXPERT=1` adds a MoK-parity dense shared expert (SwiGLU MLP,
same intermediate size, bf16 cuBLAS) inside the timed region; the CSV
kernel column gains a `+shared` suffix. MoK fuses its shared expert into
the megakernel, while fi runs it as three sequential cuBLAS GEMMs after the
kernel — this costs fi ~185 µs and is, if anything, pessimistic for fi
(no overlap).

## Same-input output cross-check (2026-08-04): PASS

Both implementations were fed bit-identical inputs (tokens, top-k routing,
bf16 expert + shared-expert weights; `mok_comparison/xcheck_common.py`
deterministic protocol) and their MXFP8 forward outputs compared:

```
rank 0: rel-L2 1.559%   rank 1: 1.551%   rank 2: 1.553%   rank 3: 1.553%
overall: rel-L2 1.554%, max|diff| 0.07 -> PASS (tol 5%)
```

Bitwise equality is not expected — each side quantizes weights/activations
to MXFP8 with its own kernels and accumulates in a different order — so
agreement at ~1.5% rel-L2 (within MXFP8 noise, and consistent across
ranks) confirms the two compute the same function. Rerun:
`mok_comparison/run_shared_and_xcheck.sh`.

### FLOPs normalization

Both sides: `6·T·(topk+1)·H·I` = 1.894 TFLOP per rank-forward (routed +
shared expert). Effective TFLOP/s = that over the p50 latency.

## Why is MoK slower here? (2026-08-04 analysis)

From code inspection and targeted measurements — the ~1.40× gap is
structural at this scale, not a tuning artifact:

1. **MoK's forward is a training forward — it always builds the backward
   stash.** `mok/functional.py::forward` returns a `MoKForwardContext`
   holding quantized fc1 gate/up outputs, the post-SwiGLU hidden
   (re-quantized in *transposed* layout for the wgrad GEMMs), and the
   dispatched activations (also transposed+quantized) — roughly 250
   MB/rank of extra HBM writes plus in-kernel transpose-quantize stages
   per forward. There is no inference mode that skips this. fi's mega
   kernel writes only the combined output. In an inference comparison
   this is pure overhead for MoK.
2. **MoK's SM-resident communication stage is the bottleneck at EP=4**
   (measured, `mok_comparison/bench_mok_sweep_fwd.py`): MXFP8 forward vs
   `fwd_num_comm_sms` on 152-SM GB200s —
   4→9.39 ms, 8→5.07, 16→3.04, 24→2.57, **36→2.56 (default, best)**,
   52→2.62. Latency scales ~inversely with comm SMs until ~24: the
   forward is throughput-limited by its dedicated dispatch/combine copy
   engine, which needs 24–36 SMs (~25% of the chip) to keep pace,
   leaving ~116 SMs for the GEMMs. fi's cutedsl kernel instead returns
   combined tokens through the GEMM epilogue warps
   (`token_back_mode=epi_warps`, the tuner winner) — no whole-SM comm
   reservation.
3. **Determinism by construction.** MoK serializes combine additions in a
   fixed macrobatch order (`csrc/mok_megakernel.cuh:1410`) to guarantee
   bitwise reproducibility — a deliberate scheduling constraint. fi's
   default mxfp8 path is also deterministic in output but keeps dynamic
   (atomic-counter) load balancing for work assignment.
4. **Activation quantization** is NOT the gap: timing fi's fused
   bf16→MXFP8 quant+staging kernel (`MEGA_TIMED_QUANT=1`) costs only
   ~50 µs (1.782 → 1.832 ms). MoK additionally quantizes *transposed*
   activation copies for the backward — part of item 1, not a
   measurement-scope issue.
5. **Design point mismatch.** MoK is engineered for NVL72-scale EP
   (cross-rack NVLink, comm/compute overlap at configurable granularity);
   at single-node EP=4 that machinery is oversized. Sweeping its knobs at
   this scale (12 configs over `fwd_num_comm_sms` × `minibatch_size`)
   improves on the defaults by ≤1% (best 2.537 ms at comm SMs 24,
   minibatch 2048) — its README's NVL72/SM103 performance claims are a
   different regime and are not contradicted by this comparison.

### Caveats

1. **Activation quantization scope**: the fi harness lifts bf16→MXFP8 input
   staging out of the timed region; MoK quantizes activations inside the
   timed forward. A small part of the gap is timing scope, not kernel
   speed.
2. **Shared expert implementation**: fi's shared expert runs as three
   sequential bf16 cuBLAS GEMMs after the fused kernel (~185 µs); MoK
   overlaps its shared expert inside the megakernel. This is pessimistic
   for fi.
3. **Timing statistic**: fi reports p50 across 100 CUDA-event-timed iters
   (100 warmup); MoK reports median-across-iters of max-across-ranks over
   100 iters (500 warmup). Both steady-state, back-to-back launches;
   `e2e_pipelined` is the MoK-comparable fi mode.
4. **Accuracy metrics are not comparable across rows**: fi reports 6.36%
   rel-L2 vs its bf16 *dense* reference (routed only); MoK reports 2.6%
   relative error vs its own bf16 MoE reference. The same-input
   cross-check above is the apples-to-apples correctness signal.
5. **Knob tuning is a wash here**: the offline tuner's winner
   (`flag_batch=4, token_back_mode=epi_warps`, see
   `mok_comparison/moe_ep_knob_cache_moklike.json`) matches the built-in
   heuristic at this token bucket (≤0.6% difference). Expect tuning to
   matter more at other token counts.
6. MoK ran at its benchmark defaults (minibatch 4096, MXFP8 fwd comm
   SMs 36); its hyperparameters were **not** swept, mirroring fi's
   heuristic-default posture. MoK's README performance claims target
   NVL72-scale EP on SM103 — this single-node SM100 comparison does not
   contradict them.
