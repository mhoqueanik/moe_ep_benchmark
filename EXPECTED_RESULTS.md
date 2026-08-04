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

| Variant | Fwd latency p50 | Effective TFLOP/s |
|---|---|---|
| fi `mxfp8_cutedsl`, default knobs, `MEGA_TIMING=e2e_pipelined` | 1.599 ms | ~1016 |
| fi `mxfp8_cutedsl`, tuned knobs, `e2e_pipelined` | 1.597 ms | ~1016 |
| fi `mxfp8_cutedsl`, default knobs, `e2e` | 1.669 ms | ~973 |
| fi `mxfp8_cutedsl`, tuned knobs, `e2e` | 1.659 ms | ~979 |
| MoK MXFP8 forward | 2.562 ms | 739 |
| MoK BF16 forward | 3.708 ms | 511 |

Headline: **fi is ~1.6× faster in raw latency, ~1.37× after FLOPs
normalization** on this workload. CSVs of record:
`results/bench_moklike_{pipelined,e2e,tuned_pipelined,tuned_e2e}_fi_mega.csv`.

### FLOPs normalization

- fi (routed experts only): `6·T·topk·H·I` = 1.624 TFLOP per rank-forward
- MoK (routed + shared expert): `6·T·(topk+1)·H·I` = 1.894 TFLOP

### Caveats — read before quoting the raw 1.6×

1. **Shared expert**: MoK's forward also computes a shared expert (~14.3%
   extra FLOPs at top-k 6); fi computes routed experts only. The TFLOP/s
   column corrects for this; fi still leads ~1.37×.
2. **Activation quantization scope**: the fi harness lifts bf16→MXFP8 input
   staging out of the timed region; MoK quantizes activations inside the
   timed forward. Part of the remaining gap is timing scope, not kernel
   speed.
3. **Timing statistic**: fi reports p50 across 100 CUDA-event-timed iters
   (100 warmup); MoK reports median-across-iters of max-across-ranks over
   100 iters (500 warmup). Both steady-state, back-to-back launches;
   `e2e_pipelined` is the MoK-comparable fi mode.
4. **Accuracy metrics are not comparable across rows**: fi reports 6.36%
   rel-L2 vs its bf16 *dense* reference; MoK reports 2.6% relative error vs
   its own bf16 MoE reference.
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
