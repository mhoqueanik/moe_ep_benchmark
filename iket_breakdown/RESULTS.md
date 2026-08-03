# Mega-path low-tokens/rank breakdown (nvfp4_cutedsl vs deep_gemm_mega)

SLURM job 2359277; 1x8 B200, EP8 (DP8/TP1), 256 experts top-8, hidden 7168, inter 2048; DSL 4.5.2. clock64 phase timing (`+pt` rows) perturbs latency and is never perf-quotable; latency comparisons use the uninstrumented cells.

## 0. Findings (what to optimize next)

Two distinct regimes; neither is "the GEMM is slow".

**Regime A — 8..64 tokens/rank (decode-like): the kernel is NOT the problem.**
Steady-state kernel time is at dg parity (0.97-0.99x). The whole e2e deficit
(~1.2x) is a roughly constant ~40-48 us per-launch cost that dg does not pay
(dg's e2e-minus-kernel gap is ~8-10 us). The in-kernel data shows why the
cutedsl launch is so sensitive to cold starts: every launch runs
dispatch_barrier (~14 us) plus a 3-barrier NVLink tail quiesce
(drain ~15-18 us + publish ~8 us), and a barrier-cold launch adds cross-rank
arrival skew on top of each of those synchronization points. Targets, in
order:
  1. per-launch tail quiesce: fold/elide the NVLink drain+publish pair
     (sense-reversal currently needs 3 barriers/launch), or overlap the tail
     with the next layer's dispatch;
  2. launch-path skew: the FI forward path ahead of the kernel (arg prep /
     workspace reset / output copy) staggers rank arrival into
     dispatch_barrier — shave host work or pre-arm like the kernel-mode
     thunk does.

**Regime B — 128..256 tokens/rank: a real kernel gap opens (1.04x -> 1.20x).**
The divergence is in the fc2 epilogue combine path, not the mainloops:
epi_fc2 WORK (call minus wait) grows 29 us @128 -> ~72 us @256 (28.9% of the
kernel), while mma_fc1 (~60 us) sits at the fc1 weight-streaming bandwidth
floor both backends share. dispatch_pull (23.5 us) and the dispatch->fc1
arrival spin (tma_b_fc1_wait, 25 us) also scale up. Targets:
  3. fc2 epilogue combine STG throughput (epi_warps peer-STG path) — this is
     where cutedsl loses to dg's combine at higher token counts; candidates:
     MEGA_IKR=1 (in-kernel REDG reduce), quantized combine wire, or wider
     store vectorization;
  4. dispatch_pull / fc1 arrival pipelining at higher token counts.

Everywhere, sched_publish "backpressure" (~51-55%) and epi_fc1_wait (~40-48%)
are symptoms of the same two causes above (consumers waiting on fc1 weight
streaming; epilogue waiting on MMA), not independent problems.

Instrumentation sanity: pt=1 adds ~4-6 us to a ~170-250 us kernel; the
accuracy gate (pt=0 vs pt=1, tokens/rank=64) is bit-identical at 23.175%
rel-L2 on the synthetic microbench reference.

## 1. Latency: dg vs cutedsl

| tok/rank | dg e2e us | cutedsl e2e us | e2e ratio | dg kernel us | cutedsl kernel us | kernel ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 178.5 | 215.1 | 1.21x | 170.0 | 167.0 | 0.98x |
| 16 | 190.7 | 227.9 | 1.20x | 180.2 | 179.2 | 0.99x |
| 32 | 216.5 | 227.5 | 1.05x | 184.3 | 179.2 | 0.97x |
| 64 | 193.4 | 230.0 | 1.19x | 184.4 | 181.2 | 0.98x |
| 128 | 199.7 | 244.1 | 1.22x | 188.4 | 195.5 | 1.04x |
| 256 | 222.4 | 301.8 | 1.36x | 206.9 | 248.9 | 1.20x |

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

