# Mega-path low-tokens/rank breakdown (nvfp4_cutedsl vs deep_gemm_mega)

SLURM job 2359277+2359364; 1x8 B200, EP8 (DP8/TP1), 256 experts top-8, hidden 7168, inter 2048; DSL 4.5.2. clock64 phase timing (`+pt` rows) perturbs latency and is never perf-quotable; latency comparisons use the uninstrumented cells.

## 0. Findings (what to optimize next)

Full range 8..8192 tokens/rank confirms a two-regime picture with a crossover;
neither regime is "the GEMM is slow".

**Regime A — 8..256 tokens/rank (decode-like): a fixed per-launch tax.**
Steady-state kernel time is at dg parity through 128 tok/rank (0.97-1.04x).
The e2e deficit (~1.2x) is a roughly constant ~40-60 us per-launch cost dg
does not pay (dg's e2e-minus-kernel gap is ~8-25 us; cutedsl's stays ~50-87 us
even at 8k tokens): dispatch_barrier (~14 us) + the 3-barrier NVLink tail
quiesce (drain ~15-18 us + publish ~8 us), amplified by cross-rank arrival
skew on cold launches. This tax is why cutedsl loses ONLY at low tokens.
Targets:
  1. per-launch tail quiesce: fold/elide the NVLink drain+publish pair or
     overlap the tail with the next layer's dispatch;
  2. launch-path skew: shave/pre-arm the FI forward path ahead of the kernel
     (the kernel-mode thunk already shows what steady-state looks like).

**Crossover.** Kernel time crosses between 256 (1.20x) and 512 (0.94x);
e2e crosses between 512 (1.08x) and 1024 (0.90x). Above that, cutedsl wins
outright and keeps winning (0.64-0.66x = ~1.5-1.6x faster at 4k-8k).
The 128-256 bump (kernel 1.04x -> 1.20x) is the transition valley: tile
counts too small for cutedsl's pipeline, overhead not yet amortized.

**Regime B — per-token costs visible at scale (not a divergence).**
fc2-epilogue combine STG work grows linearly and settles at ~20% of kernel
(26 us @512 -> 410 us @8192); fc1 epilogue work is another ~17%;
dispatch_pull reaches ~30% but runs on dedicated warps overlapped with
compute. cutedsl still beats dg 1.56x at 8k despite these, so they are
secondary throughput targets, not the low-tok bottleneck:
  3. fc2 combine STG throughput (epi_warps peer-STG): MEGA_IKR=1, quantized
     combine wire, or wider store vectorization — worth ~up-to-20% of the
     kernel at prefill sizes;
  4. dispatch_pull pipelining at high token counts.

Everywhere, sched_publish "backpressure" (~50-55%) and epi_fc1_wait are
symptoms (consumers pacing on fc1 weight streaming / MMA), not independent
problems: mma_fc1 sits at the fc1 weight-streaming bandwidth floor
(~60-70 us at low tokens) that both backends share.

Instrumentation sanity: pt=1 adds ~0-7 us across 170-2000 us kernels; the
accuracy gate (pt=0 vs pt=1) is bit-identical (23.175% rel-L2 vs the
synthetic microbench reference, both jobs).

## 1. Latency: dg vs cutedsl

| tok/rank | dg e2e us | cutedsl e2e us | e2e ratio | dg kernel us | cutedsl kernel us | kernel ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 178.5 | 215.1 | 1.21x | 170.0 | 167.0 | 0.98x |
| 16 | 190.7 | 227.9 | 1.20x | 180.2 | 179.2 | 0.99x |
| 32 | 216.5 | 227.5 | 1.05x | 184.3 | 179.2 | 0.97x |
| 64 | 193.4 | 230.0 | 1.19x | 184.4 | 181.2 | 0.98x |
| 128 | 199.7 | 244.1 | 1.22x | 188.4 | 195.5 | 1.04x |
| 256 | 222.4 | 301.8 | 1.36x | 206.9 | 248.9 | 1.20x |
| 512 | 297.8 | 322.7 | 1.08x | 280.7 | 263.2 | 0.94x |
| 1024 | 479.0 | 429.0 | 0.90x | 460.8 | 371.7 | 0.81x |
| 2048 | 813.1 | 633.8 | 0.78x | 788.5 | 566.4 | 0.72x |
| 4096 | 1649.1 | 1080.3 | 0.66x | 1594.3 | 1025.5 | 0.64x |
| 8192 | 3138.8 | 2067.0 | 0.66x | 3107.8 | 1980.4 | 0.64x |

## 2. cutedsl in-kernel phase breakdown (clock64, MEGA_TIMING=kernel)

Cycles are per-CTA means over active CTAs; `mean` averages the 8 ranks, `worst rank` is the rank with the largest value (skew signal). `% kern` is vs that rank-mean kernel_total. Approx us scales cycles by the measured p50 of the instrumented run.

### tokens/rank = 8

measured p50 (instrumented): 174.0 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3584 | 1.3 | 2.3 | 2.5 |  |
| dispatch_barrier | 21797 | 8.2 | 14.2 | 16.5 |  |
| dispatch_pull | 6238 | 2.3 | 4.1 | 4.2 |  |
| dispatch_total | 31620 | 11.8 | 20.6 | 22.9 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 226342 | 84.7 | 147.4 | 149.2 |  |
| sched_publish | 141821 | 53.1 | 92.4 | 93.8 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 35270 | 13.2 | 23.0 | 25.3 | idle waiting for scheduled work |
| tma_a_loop_total | 251223 | 94.0 | 163.6 | 164.9 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 35385 | 13.3 | 23.1 | 25.4 | idle waiting for scheduled work |
| tma_b_fc1_wait | 12978 | 4.9 | 8.4 | 9.7 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 358 | 0.1 | 0.2 | 0.4 | fc1->fc2 handoff spin |
| tma_b_loop_total | 251311 | 94.1 | 163.7 | 165.0 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 137287 | 51.4 | 89.4 | 91.2 | idle waiting for scheduled work |
| mma_fc1 | 79039 | 29.6 | 51.5 | 52.2 |  |
| mma_fc2 | 28475 | 10.7 | 18.5 | 19.2 |  |
| mma_loop_total | 246824 | 92.4 | 160.8 | 162.0 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 34391 | 12.9 | 22.4 | 24.8 | idle waiting for scheduled work |
| epi_fc1_wait | 129229 | 48.4 | 84.2 | 85.5 |  |
| epi_fc1 | 145027 | 54.3 | 94.5 | 96.5 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 22767 | 8.5 | 14.8 | 15.9 |  |
| epi_fc2 | 48156 | 18.0 | 31.4 | 32.5 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 726 | 0.3 | 0.5 | 0.5 |  |
| epi_flag | 21556 | 8.1 | 14.0 | 15.1 |  |
| epi_loop_total | 251542 | 94.1 | 163.8 | 165.1 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 23199 | 8.8 | 15.2 | 24.7 |  |
| tail_shared_reset | 177 | 0.1 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 11778 | 4.4 | 7.7 | 8.2 |  |
| tail_local_reset | 138 | 0.1 | 0.1 | 0.1 |  |
| **whole kernel** | | | | | |
| kernel_total | 267156 | 100.0 | 174.0 | 175.1 |  |

### tokens/rank = 16

measured p50 (instrumented): 184.4 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3390 | 1.2 | 2.2 | 2.2 |  |
| dispatch_barrier | 22494 | 7.7 | 14.2 | 16.3 |  |
| dispatch_pull | 7655 | 2.6 | 4.8 | 5.3 |  |
| dispatch_total | 33540 | 11.5 | 21.1 | 23.1 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 251113 | 85.8 | 158.2 | 159.5 |  |
| sched_publish | 159519 | 54.5 | 100.5 | 103.2 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 36018 | 12.3 | 22.7 | 24.8 | idle waiting for scheduled work |
| tma_a_loop_total | 276492 | 94.5 | 174.2 | 175.4 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 36131 | 12.3 | 22.8 | 24.9 | idle waiting for scheduled work |
| tma_b_fc1_wait | 13850 | 4.7 | 8.7 | 9.3 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 351 | 0.1 | 0.2 | 0.2 | fc1->fc2 handoff spin |
| tma_b_loop_total | 276580 | 94.5 | 174.2 | 175.4 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 150117 | 51.3 | 94.6 | 96.2 | idle waiting for scheduled work |
| mma_fc1 | 87297 | 29.8 | 55.0 | 55.3 |  |
| mma_fc2 | 32430 | 11.1 | 20.4 | 20.8 |  |
| mma_loop_total | 272002 | 93.0 | 171.4 | 172.6 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 35109 | 12.0 | 22.1 | 24.3 | idle waiting for scheduled work |
| epi_fc1_wait | 140419 | 48.0 | 88.5 | 89.9 |  |
| epi_fc1 | 157879 | 54.0 | 99.5 | 100.3 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 25285 | 8.7 | 15.9 | 17.2 |  |
| epi_fc2 | 55576 | 19.0 | 35.0 | 36.3 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 822 | 0.3 | 0.5 | 0.5 |  |
| epi_flag | 25559 | 8.7 | 16.1 | 17.7 |  |
| epi_loop_total | 276813 | 94.6 | 174.4 | 175.6 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 19880 | 6.8 | 12.6 | 19.9 |  |
| tail_shared_reset | 177 | 0.1 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 11606 | 4.0 | 7.3 | 7.7 |  |
| tail_local_reset | 139 | 0.0 | 0.1 | 0.1 |  |
| **whole kernel** | | | | | |
| kernel_total | 292604 | 100.0 | 184.3 | 185.4 |  |

### tokens/rank = 32

measured p50 (instrumented): 185.4 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3472 | 1.2 | 2.2 | 2.3 |  |
| dispatch_barrier | 22605 | 7.6 | 14.0 | 16.2 |  |
| dispatch_pull | 10519 | 3.5 | 6.5 | 7.0 |  |
| dispatch_total | 36596 | 12.2 | 22.7 | 24.6 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 257094 | 86.0 | 159.4 | 160.8 |  |
| sched_publish | 164866 | 55.2 | 102.2 | 103.9 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 36315 | 12.1 | 22.5 | 24.7 | idle waiting for scheduled work |
| tma_a_loop_total | 282624 | 94.6 | 175.2 | 177.3 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 36452 | 12.2 | 22.6 | 24.8 | idle waiting for scheduled work |
| tma_b_fc1_wait | 14490 | 4.8 | 9.0 | 9.5 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 356 | 0.1 | 0.2 | 0.2 | fc1->fc2 handoff spin |
| tma_b_loop_total | 282711 | 94.6 | 175.3 | 177.4 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 153275 | 51.3 | 95.0 | 97.1 | idle waiting for scheduled work |
| mma_fc1 | 89174 | 29.8 | 55.3 | 56.0 |  |
| mma_fc2 | 33470 | 11.2 | 20.8 | 20.9 |  |
| mma_loop_total | 278136 | 93.1 | 172.4 | 174.5 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 35378 | 11.8 | 21.9 | 24.1 | idle waiting for scheduled work |
| epi_fc1_wait | 143337 | 48.0 | 88.9 | 90.2 |  |
| epi_fc1 | 160872 | 53.8 | 99.7 | 100.6 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 24843 | 8.3 | 15.4 | 17.0 |  |
| epi_fc2 | 57222 | 19.1 | 35.5 | 36.1 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 829 | 0.3 | 0.5 | 0.6 |  |
| epi_flag | 26755 | 8.9 | 16.6 | 17.8 |  |
| epi_loop_total | 282988 | 94.7 | 175.5 | 177.6 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 16708 | 5.6 | 10.4 | 12.7 |  |
| tail_shared_reset | 177 | 0.1 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 11872 | 4.0 | 7.4 | 8.2 |  |
| tail_local_reset | 140 | 0.0 | 0.1 | 0.1 |  |
| **whole kernel** | | | | | |
| kernel_total | 298867 | 100.0 | 185.3 | 186.0 |  |

### tokens/rank = 64

measured p50 (instrumented): 187.4 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3492 | 1.2 | 2.1 | 2.3 |  |
| dispatch_barrier | 23707 | 7.9 | 14.7 | 16.4 |  |
| dispatch_pull | 16590 | 5.5 | 10.3 | 11.1 |  |
| dispatch_total | 43789 | 14.5 | 27.1 | 28.6 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 257534 | 85.5 | 159.6 | 161.7 |  |
| sched_publish | 163851 | 54.4 | 101.5 | 104.7 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 37592 | 12.5 | 23.3 | 25.2 | idle waiting for scheduled work |
| tma_a_loop_total | 284106 | 94.3 | 176.1 | 177.2 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 37771 | 12.5 | 23.4 | 25.3 | idle waiting for scheduled work |
| tma_b_fc1_wait | 14714 | 4.9 | 9.1 | 10.0 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 346 | 0.1 | 0.2 | 0.2 | fc1->fc2 handoff spin |
| tma_b_loop_total | 284194 | 94.4 | 176.1 | 177.2 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 154709 | 51.4 | 95.9 | 97.1 | idle waiting for scheduled work |
| mma_fc1 | 89283 | 29.7 | 55.3 | 56.0 |  |
| mma_fc2 | 33499 | 11.1 | 20.8 | 21.1 |  |
| mma_loop_total | 279778 | 92.9 | 173.4 | 174.5 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 36601 | 12.1 | 22.7 | 24.6 | idle waiting for scheduled work |
| epi_fc1_wait | 142828 | 47.4 | 88.5 | 89.6 |  |
| epi_fc1 | 160204 | 53.2 | 99.3 | 100.3 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 22383 | 7.4 | 13.9 | 15.2 |  |
| epi_fc2 | 55847 | 18.5 | 34.6 | 35.2 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 848 | 0.3 | 0.5 | 0.6 |  |
| epi_flag | 29048 | 9.7 | 18.0 | 19.0 |  |
| epi_loop_total | 284569 | 94.5 | 176.3 | 177.4 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 16440 | 5.5 | 10.2 | 12.3 |  |
| tail_shared_reset | 177 | 0.1 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 11670 | 3.9 | 7.2 | 7.8 |  |
| tail_local_reset | 264 | 0.1 | 0.2 | 0.2 |  |
| **whole kernel** | | | | | |
| kernel_total | 301162 | 100.0 | 186.6 | 187.4 |  |

### tokens/rank = 128

measured p50 (instrumented): 199.8 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3501 | 1.1 | 2.2 | 2.3 |  |
| dispatch_barrier | 21970 | 6.8 | 13.6 | 15.5 |  |
| dispatch_pull | 23888 | 7.4 | 14.8 | 15.1 |  |
| dispatch_total | 49359 | 15.3 | 30.6 | 32.6 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 274796 | 85.1 | 170.5 | 172.6 |  |
| sched_publish | 164927 | 51.1 | 102.3 | 105.8 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 38433 | 11.9 | 23.8 | 26.1 | idle waiting for scheduled work |
| tma_a_loop_total | 299046 | 92.6 | 185.6 | 187.3 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 38502 | 11.9 | 23.9 | 26.1 | idle waiting for scheduled work |
| tma_b_fc1_wait | 24460 | 7.6 | 15.2 | 15.7 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 338 | 0.1 | 0.2 | 0.2 | fc1->fc2 handoff spin |
| tma_b_loop_total | 299135 | 92.6 | 185.6 | 187.3 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 160417 | 49.7 | 99.5 | 101.0 | idle waiting for scheduled work |
| mma_fc1 | 94106 | 29.1 | 58.4 | 59.2 |  |
| mma_fc2 | 36920 | 11.4 | 22.9 | 24.0 |  |
| mma_loop_total | 294131 | 91.1 | 182.5 | 183.6 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 35587 | 11.0 | 22.1 | 24.0 | idle waiting for scheduled work |
| epi_fc1_wait | 152625 | 47.3 | 94.7 | 96.3 |  |
| epi_fc1 | 170386 | 52.8 | 105.7 | 106.7 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 14830 | 4.6 | 9.2 | 11.0 |  |
| epi_fc2 | 61746 | 19.1 | 38.3 | 41.0 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 904 | 0.3 | 0.6 | 0.6 |  |
| epi_flag | 29530 | 9.2 | 18.3 | 19.7 |  |
| epi_loop_total | 300188 | 93.0 | 186.2 | 188.2 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 15971 | 5.0 | 9.9 | 11.5 |  |
| tail_shared_reset | 177 | 0.1 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 11667 | 3.6 | 7.2 | 7.8 |  |
| tail_local_reset | 397 | 0.1 | 0.2 | 0.4 |  |
| **whole kernel** | | | | | |
| kernel_total | 322923 | 100.0 | 200.3 | 201.6 |  |

### tokens/rank = 256

measured p50 (instrumented): 255.1 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 4240 | 1.0 | 2.6 | 2.9 |  |
| dispatch_barrier | 22066 | 5.3 | 13.6 | 14.7 |  |
| dispatch_pull | 38252 | 9.2 | 23.5 | 24.1 |  |
| dispatch_total | 64559 | 15.6 | 39.7 | 40.8 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 362475 | 87.3 | 222.7 | 225.6 |  |
| sched_publish | 229238 | 55.2 | 140.8 | 143.4 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 42953 | 10.4 | 26.4 | 27.2 | idle waiting for scheduled work |
| tma_a_loop_total | 378939 | 91.3 | 232.8 | 234.5 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 42505 | 10.2 | 26.1 | 27.1 | idle waiting for scheduled work |
| tma_b_fc1_wait | 41069 | 9.9 | 25.2 | 26.1 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 336 | 0.1 | 0.2 | 0.2 | fc1->fc2 handoff spin |
| tma_b_loop_total | 379223 | 91.4 | 233.0 | 234.6 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 203323 | 49.0 | 124.9 | 126.2 | idle waiting for scheduled work |
| mma_fc1 | 101219 | 24.4 | 62.2 | 63.0 |  |
| mma_fc2 | 66284 | 16.0 | 40.7 | 41.8 |  |
| mma_loop_total | 378384 | 91.2 | 232.5 | 234.6 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 39927 | 9.6 | 24.5 | 25.4 | idle waiting for scheduled work |
| epi_fc1_wait | 163076 | 39.3 | 100.2 | 102.0 |  |
| epi_fc1 | 185832 | 44.8 | 114.2 | 116.3 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 2724 | 0.7 | 1.7 | 2.3 |  |
| epi_fc2 | 120126 | 28.9 | 73.8 | 78.9 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 877 | 0.2 | 0.5 | 0.6 |  |
| epi_flag | 39898 | 9.6 | 24.5 | 25.3 |  |
| epi_loop_total | 388716 | 93.6 | 238.8 | 241.6 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 24296 | 5.9 | 14.9 | 17.4 |  |
| tail_shared_reset | 177 | 0.0 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 12272 | 3.0 | 7.5 | 8.3 |  |
| tail_local_reset | 524 | 0.1 | 0.3 | 0.4 |  |
| **whole kernel** | | | | | |
| kernel_total | 415082 | 100.0 | 255.1 | 255.1 |  |

### tokens/rank = 512

measured p50 (instrumented): 263.2 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3857 | 0.9 | 2.5 | 2.6 |  |
| dispatch_barrier | 23544 | 5.7 | 15.1 | 17.2 |  |
| dispatch_pull | 71214 | 17.3 | 45.6 | 47.5 |  |
| dispatch_total | 98615 | 24.0 | 63.1 | 66.1 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 325987 | 79.3 | 208.7 | 210.4 |  |
| sched_publish | 208694 | 50.7 | 133.6 | 137.3 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 41025 | 10.0 | 26.2 | 28.1 | idle waiting for scheduled work |
| tma_a_loop_total | 351329 | 85.4 | 224.9 | 228.6 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 41297 | 10.0 | 26.4 | 28.3 | idle waiting for scheduled work |
| tma_b_fc1_wait | 43180 | 10.5 | 27.6 | 28.9 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 17815 | 4.3 | 11.4 | 17.1 | fc1->fc2 handoff spin |
| tma_b_loop_total | 351380 | 85.4 | 224.9 | 228.6 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 188971 | 45.9 | 121.0 | 123.8 | idle waiting for scheduled work |
| mma_fc1 | 112356 | 27.3 | 71.9 | 74.1 |  |
| mma_fc2 | 44292 | 10.8 | 28.4 | 31.0 |  |
| mma_loop_total | 349050 | 84.9 | 223.4 | 227.1 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 35701 | 8.7 | 22.9 | 24.8 | idle waiting for scheduled work |
| epi_fc1_wait | 165238 | 40.2 | 105.8 | 111.0 |  |
| epi_fc1 | 202492 | 49.2 | 129.6 | 135.1 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 22854 | 5.5 | 14.6 | 19.7 |  |
| epi_fc2 | 61322 | 14.9 | 39.2 | 43.0 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 1735 | 0.4 | 1.1 | 1.3 |  |
| epi_flag | 52182 | 12.7 | 33.4 | 38.1 |  |
| epi_loop_total | 356308 | 86.7 | 228.1 | 231.0 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 23642 | 5.8 | 15.1 | 19.9 |  |
| tail_shared_reset | 165 | 0.0 | 0.1 | 0.1 |  |
| tail_nvlink_publish | 11636 | 2.8 | 7.5 | 8.2 |  |
| tail_local_reset | 146 | 0.0 | 0.1 | 0.1 |  |
| **whole kernel** | | | | | |
| kernel_total | 411238 | 100.0 | 263.2 | 263.3 |  |

### tokens/rank = 1024

measured p50 (instrumented): 371.8 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 3508 | 0.6 | 2.3 | 2.6 |  |
| dispatch_barrier | 25797 | 4.5 | 17.0 | 18.1 |  |
| dispatch_pull | 136998 | 24.2 | 90.1 | 91.8 |  |
| dispatch_total | 166303 | 29.4 | 109.4 | 110.8 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 7 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 380529 | 67.3 | 250.3 | 256.9 |  |
| sched_publish | 266254 | 47.1 | 175.2 | 180.1 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 42863 | 7.6 | 28.2 | 29.4 | idle waiting for scheduled work |
| tma_a_loop_total | 406369 | 71.9 | 267.4 | 275.2 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 42381 | 7.5 | 27.9 | 29.0 | idle waiting for scheduled work |
| tma_b_fc1_wait | 43105 | 7.6 | 28.4 | 29.9 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 16425 | 2.9 | 10.8 | 17.7 | fc1->fc2 handoff spin |
| tma_b_loop_total | 406460 | 71.9 | 267.4 | 275.2 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 217095 | 38.4 | 142.8 | 147.8 | idle waiting for scheduled work |
| mma_fc1 | 123349 | 21.8 | 81.2 | 83.4 |  |
| mma_fc2 | 57733 | 10.2 | 38.0 | 40.4 |  |
| mma_loop_total | 401925 | 71.1 | 264.4 | 272.7 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 39769 | 7.0 | 26.2 | 27.6 | idle waiting for scheduled work |
| epi_fc1_wait | 170031 | 30.1 | 111.8 | 114.2 |  |
| epi_fc1 | 233356 | 41.3 | 153.5 | 157.0 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 28330 | 5.0 | 18.6 | 23.6 |  |
| epi_fc2 | 97091 | 17.2 | 63.9 | 68.3 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 2499 | 0.5 | 1.6 | 1.8 |  |
| epi_flag | 36114 | 6.4 | 23.7 | 24.5 |  |
| epi_loop_total | 412250 | 72.9 | 271.2 | 278.8 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 27111 | 4.8 | 17.9 | 22.1 |  |
| tail_shared_reset | 201 | 0.0 | 0.1 | 0.2 |  |
| tail_nvlink_publish | 12558 | 2.2 | 8.2 | 8.8 |  |
| tail_local_reset | 428 | 0.1 | 0.3 | 0.3 |  |
| **whole kernel** | | | | | |
| kernel_total | 565390 | 100.0 | 371.9 | 372.8 |  |

### tokens/rank = 2048

measured p50 (instrumented): 570.3 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 5287 | 0.7 | 3.8 | 4.1 |  |
| dispatch_barrier | 37482 | 4.7 | 26.7 | 34.3 |  |
| dispatch_pull | 220625 | 27.8 | 158.1 | 163.5 |  |
| dispatch_total | 263394 | 33.1 | 188.7 | 201.4 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 8 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 612947 | 77.0 | 439.2 | 454.2 |  |
| sched_publish | 442134 | 55.6 | 316.8 | 326.7 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 58967 | 7.4 | 42.1 | 49.6 | idle waiting for scheduled work |
| tma_a_loop_total | 650642 | 81.8 | 466.1 | 478.6 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 57772 | 7.2 | 41.3 | 48.6 | idle waiting for scheduled work |
| tma_b_fc1_wait | 76414 | 9.6 | 54.8 | 56.1 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 18435 | 2.3 | 13.3 | 21.8 | fc1->fc2 handoff spin |
| tma_b_loop_total | 650735 | 81.8 | 466.2 | 478.7 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 346655 | 43.5 | 248.3 | 258.4 | idle waiting for scheduled work |
| mma_fc1 | 194791 | 24.5 | 139.6 | 143.7 |  |
| mma_fc2 | 100343 | 12.6 | 71.9 | 74.8 |  |
| mma_loop_total | 647058 | 81.3 | 463.6 | 476.7 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 52600 | 6.6 | 37.6 | 44.7 | idle waiting for scheduled work |
| epi_fc1_wait | 226030 | 28.4 | 161.9 | 166.3 |  |
| epi_fc1 | 343424 | 43.1 | 246.1 | 253.3 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 29387 | 3.7 | 21.1 | 26.8 |  |
| epi_fc2 | 163140 | 20.5 | 117.0 | 122.9 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 6248 | 0.8 | 4.5 | 4.9 |  |
| epi_flag | 91695 | 11.5 | 65.7 | 70.0 |  |
| epi_loop_total | 662434 | 83.2 | 474.6 | 485.4 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 25240 | 3.2 | 18.1 | 24.2 |  |
| tail_shared_reset | 566 | 0.1 | 0.4 | 0.5 |  |
| tail_nvlink_publish | 11579 | 1.4 | 8.3 | 8.7 |  |
| tail_local_reset | 707 | 0.1 | 0.5 | 0.8 |  |
| **whole kernel** | | | | | |
| kernel_total | 795730 | 100.0 | 570.2 | 570.4 |  |

### tokens/rank = 4096

measured p50 (instrumented): 1029.2 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 5313 | 0.4 | 3.9 | 4.2 |  |
| dispatch_barrier | 81903 | 5.7 | 58.7 | 70.8 |  |
| dispatch_pull | 403068 | 28.4 | 292.9 | 316.6 |  |
| dispatch_total | 490285 | 34.5 | 355.6 | 371.0 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 8 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 1000975 | 70.5 | 727.8 | 807.9 |  |
| sched_publish | 730196 | 51.5 | 531.0 | 588.5 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 113562 | 7.9 | 81.8 | 93.4 | idle waiting for scheduled work |
| tma_a_loop_total | 1083173 | 76.2 | 786.8 | 826.0 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 111278 | 7.8 | 80.1 | 91.6 | idle waiting for scheduled work |
| tma_b_fc1_wait | 75155 | 5.3 | 54.6 | 56.5 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 16902 | 1.2 | 12.1 | 31.7 | fc1->fc2 handoff spin |
| tma_b_loop_total | 1083288 | 76.3 | 786.8 | 826.1 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 586538 | 41.3 | 425.7 | 438.2 | idle waiting for scheduled work |
| mma_fc1 | 293086 | 20.7 | 213.1 | 233.6 |  |
| mma_fc2 | 189897 | 13.4 | 138.1 | 155.1 |  |
| mma_loop_total | 1077611 | 75.9 | 782.8 | 823.4 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 101746 | 7.1 | 73.2 | 85.0 | idle waiting for scheduled work |
| epi_fc1_wait | 282427 | 19.9 | 205.2 | 220.6 |  |
| epi_fc1 | 502366 | 35.4 | 365.3 | 403.7 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 37858 | 2.6 | 27.4 | 43.7 |  |
| epi_fc2 | 306056 | 21.6 | 222.6 | 252.2 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 12662 | 0.9 | 9.2 | 10.9 |  |
| epi_flag | 163832 | 11.5 | 119.0 | 127.5 |  |
| epi_loop_total | 1095792 | 77.2 | 795.9 | 835.4 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 71992 | 5.0 | 51.6 | 67.7 |  |
| tail_shared_reset | 489 | 0.0 | 0.4 | 0.5 |  |
| tail_nvlink_publish | 11082 | 0.8 | 8.1 | 8.6 |  |
| tail_local_reset | 524 | 0.0 | 0.4 | 0.6 |  |
| **whole kernel** | | | | | |
| kernel_total | 1421720 | 100.0 | 1031.7 | 1042.9 |  |

### tokens/rank = 8192

measured p50 (instrumented): 1994.8 us; active CTAs/rank: [148]

| phase | mean cyc | mean % kern | mean ~us | worst-rank ~us | note |
|---|---:|---:|---:|---:|---|
| **dispatch warps (w8-11)** | | | | | |
| dispatch_prep | 9859 | 0.4 | 7.5 | 8.6 |  |
| dispatch_barrier | 36434 | 1.4 | 27.6 | 33.7 |  |
| dispatch_pull | 776472 | 29.6 | 590.0 | 608.4 |  |
| dispatch_total | 822764 | 31.3 | 625.1 | 641.3 |  |
| **sched warp (w7)** | | | | | |
| sched_pre_init_wait | 8 | 0.0 | 0.0 | 0.0 | cross-rank dispatch arrival, seen by compute |
| sched_loop_total | 1830830 | 69.7 | 1391.1 | 1437.9 |  |
| sched_publish | 1379972 | 52.6 | 1048.5 | 1079.2 | backpressure: consumers not draining tiles |
| **TMA-A warp (w5)** | | | | | |
| tma_a_consume_work | 92562 | 3.5 | 70.2 | 75.4 | idle waiting for scheduled work |
| tma_a_loop_total | 1872330 | 71.3 | 1422.6 | 1459.8 |  |
| **TMA-B warp (w6)** | | | | | |
| tma_b_consume_work | 90215 | 3.4 | 68.5 | 73.5 | idle waiting for scheduled work |
| tma_b_fc1_wait | 73148 | 2.8 | 55.6 | 56.7 | dispatch->fc1 token arrival spin |
| tma_b_fc2_wait | 3152 | 0.1 | 2.4 | 2.5 | fc1->fc2 handoff spin |
| tma_b_loop_total | 1872411 | 71.3 | 1422.6 | 1459.8 |  |
| **MMA warp (w4)** | | | | | |
| mma_consume_work | 971677 | 37.0 | 738.2 | 752.4 | idle waiting for scheduled work |
| mma_fc1 | 524458 | 20.0 | 398.5 | 411.0 |  |
| mma_fc2 | 359903 | 13.7 | 273.5 | 283.4 |  |
| mma_loop_total | 1869652 | 71.2 | 1420.5 | 1457.3 |  |
| **epilogue warps (w0-3)** | | | | | |
| epi_consume_work | 71090 | 2.7 | 54.0 | 59.0 | idle waiting for scheduled work |
| epi_fc1_wait | 435706 | 16.6 | 331.1 | 347.0 |  |
| epi_fc1 | 867469 | 33.0 | 659.1 | 683.5 | whole fc1 epi call (work = epi_fc1 - epi_fc1_wait) |
| epi_fc2_wait | 48482 | 1.9 | 36.9 | 39.6 |  |
| epi_fc2 | 578239 | 22.0 | 439.4 | 456.4 | whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait) |
| epi_drain_barrier | 25549 | 1.0 | 19.4 | 20.4 |  |
| epi_flag | 322362 | 12.3 | 244.9 | 250.0 |  |
| epi_loop_total | 1881441 | 71.7 | 1429.5 | 1466.8 |  |
| **kernel tail** | | | | | |
| tail_rendezvous | 8 | 0.0 | 0.0 | 0.0 |  |
| tail_nvlink_drain | 36824 | 1.4 | 27.9 | 48.0 |  |
| tail_shared_reset | 540 | 0.0 | 0.4 | 0.5 |  |
| tail_nvlink_publish | 11451 | 0.4 | 8.7 | 9.2 |  |
| tail_local_reset | 514 | 0.0 | 0.4 | 0.5 |  |
| **whole kernel** | | | | | |
| kernel_total | 2626138 | 100.0 | 1995.0 | 2001.9 |  |

