# RTX PRO 6000 single-GPU MegaMoE split results (2026-08-10)

Hardware: 1x RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs, 96GB).
Config: DSV4-flash (topk=6, hidden=7168, intermediate=6144, MXFP8xMXFP8,
gate_up_clamp=10, balanced routing), split_launch=green_graph,
comm_backend=p2p_direct, K1/K2/TX/RX = 120/68/0/0 (proportional scale of the
Pro 5000 72/38 split — NOT tuned for this die), 100 iters, p50 shown.
Baseline: hanyueh's MegaMoE EP8 on 8x RTX Pro 5000 (DSV4-flash, incl. top-k
reduce) — see reference_rtxpro5000_numbers.md.
TFLOP/s derived from baseline work: FLOP = ref_time x ref_TFLOP/s.

## A. EP1-proxy, 48 experts (job 1805442, slurm-1805442.out)
Per-GPU workload matched to one EP8 rank (48 local experts, same rows/expert),
minus dispatch/combine comm. The apples-to-apples kernel comparison.

| Tokens | p50 | TFLOP/s | ref EP8/GPU | ratio |
|---|---|---|---|---|
| 16 | 2,285.70 us | 2.1 | 1,397.79 us / 3.5 | 0.61x |
| 32 | 2,416.83 us | 4.0 | 1,436.34 us / 6.7 | 0.59x |
| 64 | 2,435.28 us | 7.9 | 1,577.98 us / 12.2 | 0.65x |
| 128 | 2,473.07 us | 15.6 | 1,712.43 us / 22.6 | 0.69x |
| 512 | 2,786.50 us | 55.5 | 2,123.14 us / 72.8 | 0.76x |
| 1,024 | 3,888.29 us | 79.5 | 3,121.98 us / 99.1 | 0.80x |
| 2,048 | 5,969.01 us | 103.6 | 4,674.06 us / 132.3 | 0.78x |
| 4,096 | 10,503.41 us | 117.8 | 8,622.56 us / 143.5 | 0.82x |
| 8,192 | 19,185.31 us | 128.9 | 16,772.03 us / 147.5 | 0.87x |
| 12,288 | 27,356.27 us | 135.6 | 24,718.35 us / 150.1 | 0.90x |
| 16,384 | 37,404.35 us | 132.3 | 32,763.55 us / 151.0 | 0.88x |
| 20,480 | 45,976.83 us | 134.5 | 40,939.44 us / 151.1 | 0.89x |
| 24,576 | 53,546.08 us | 138.6 | 49,117.60 us / 151.1 | 0.92x |

## B. EP1, all 384 experts local (job 1805367, slurm-1805367.out)
Weight-bandwidth-bound (~35GB weight bank streamed per iter -> ~18.7ms floor).
Not comparable to EP8 per-GPU; kept for reference.
Peak 120.4 TFLOP/s @ 24576 tokens (61,639.30 us).

## C. 8 ranks time-slicing one GPU (job 1804928) — see slurm-1804928.out
~88ms serialization floor; 62 aggregate TFLOP/s @4096 tokens. Correctness
datapoint only. 24576 tokens/rank OOMs (workspace 11.92 GiB/rank).

## D. K1/K2 split scan (job 1805994, slurm-1805994.out), EP1-proxy 48 experts
No in-repo tuner covers moe_sm120_mxfp8_split (tester --sweep is sm100
nvfp4-only); this is the manual offline-scan style the heuristic came from.
K1 must be a multiple of 8 on this part (green-context granularity; other
values get rounded and fail the partition check). p50, 50 iters:

| K1/K2 | 24576 tok | TF | 8192 tok | TF |
|---|---|---|---|---|
| 120/68 | 52,641 us | 141.0 | 19,368 us | 127.7 |
| **128/60** | **50,230 us** | **147.8** | **18,020 us** | **137.3** |
| 136/52 | 59,571 us | 124.6 | 21,894 us | 113.0 |
| 144/44 | 66,436 us | 111.7 | 24,140 us | 102.5 |
| 152/36 | 79,403 us | 93.5 | 28,703 us | 86.2 |

**Best split on 188 SMs: K1=128 / K2=60** (~68/32) -> 147.8 TF @24576 = 0.98x
of a Pro 5000 EP8 rank (151.1 TF). Clear minimum at 128; curve rises steeply
with more K1. Remaining ~2% gap + no comm overlap here, so kernels are
essentially at parity per-GPU; the 1.7x SM advantage does NOT translate --
suggests occupancy/schedule, clocks, or memory-bound phases cap it.

## Open items
- K1/K2 green-context split is untuned on 188 SMs; a sweep around 120/68 may
  close part of the 0.92x gap at large tokens.
- Even compute-matched, PRO 6000 lands below Pro 5000 EP8 per-GPU numbers
  despite 1.7x SMs on paper — worth checking clocks/power and whether the
  kernel's static schedule leaves the extra SMs idle.
- Multi-node EP over rtxpro6000 nodes is blocked by SLURM topology (each node
  is an isolated Level-0 switch; 25GbE RoCE only).
