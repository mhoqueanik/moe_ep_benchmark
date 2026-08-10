# Expected results: DSV4 split kernel, single RTX PRO 6000 (sm_120)

Recorded 2026-08-10. Use these as the regression reference when re-running the
single-GPU DSV4 sweeps on an `rtxpro6000` node (2u2g-emr-046x/047x, RTX PRO
6000 Blackwell Server Edition, 188 SMs, 96 GB).

Config: `moe_sm120_mxfp8_split.mega_runner`, world=1 (torchrun standalone,
nproc=1), DSV4-flash shape (topk=6, hidden=7168, intermediate=6144,
MXFP8xMXFP8, `--gate_up_clamp 10`, balanced routing),
`--split_launch green_graph --comm_backend p2p_direct
--enable_static_expert_shape`, `MEGA_HEURISTIC_ALLOW_ANY_SMS=1`,
`NVSHMEM_SYMMETRIC_SIZE=8G`, 100 perf iters / 10 warmup, CUDA-event p50.
Job scripts: `/home/mhoqueanik/sm120_rtx/perf_logs/sm120_perf_job_rtxpro6000_split_ep1*.sh`
(cluster jobs 1805442, 1805367, 1805994).

**SM split: use K1=128 / K2=60** (tuned, see k1k2_scan_kernel.csv). K1 must be
a multiple of 8 (green-context partition granularity) and K1+K2 must equal 188.
The tables below were measured at the pre-tune split K1=120/K2=68 except where
noted; expect ~4-7% faster at 128/60 for >=8192 tokens.

## A. EP1-proxy, 48 experts (`--num_total_experts 48`) — primary reference

One EP8 rank's compute (48 local experts, same rows/expert), no comm.
Tolerance: p50 within ~±5% run-to-run (p90/p10 spread grows with tokens).

| tokens_per_rank | expected p50 (K1=120/68) | tuned p50 (K1=128/60) |
|---|---|---|
| 16 | 2,286 us | — |
| 32 | 2,417 us | — |
| 64 | 2,435 us | — |
| 128 | 2,473 us | — |
| 512 | 2,787 us | — |
| 1,024 | 3,888 us | — |
| 2,048 | 5,969 us | — |
| 4,096 | 10,503 us | — |
| 8,192 | 19,185 us | 18,020 us |
| 12,288 | 27,356 us | — |
| 16,384 | 37,404 us | — |
| 20,480 | 45,977 us | — |
| 24,576 | 53,546 us | 50,230 us (147.8 TFLOP/s) |

Sanity anchor: tuned 24,576-token p50 of ~50.2 ms = 0.98x of a Pro 5000 EP8
rank (49,117.60 us / 151.101 TFLOP/s, hanyueh's baseline in
`reference_rtxpro5000_numbers.md`). A p50 above ~56 ms (>10% regression) or a
"K2 partition differs from heuristic" error (wrong/rounded SM split) means
something changed.

## B. EP1, all 384 experts local (`--num_total_experts 384`), K1=120/68

Weight-bandwidth-bound regime (full ~35 GB weight bank streamed per iter);
expect a flat ~18.7 ms floor from 512 through 2,048 tokens.

| tokens_per_rank | expected p50 |
|---|---|
| 16 | 3,632 us |
| 512 | 18,743 us |
| 2,048 | 18,696 us |
| 8,192 | 28,556 us |
| 24,576 | 61,639 us |

(Full 13-point sweeps for both configs in `ep1_proxy48_kernel.csv` /
`ep1_384exp_kernel.csv`.)

## Known failure modes on this setup

- `/home/scratch.mhoqueanik_gpu` is NOT mounted on rtxpro6000 nodes — stage
  the repo to NFS home and build the venv on node-local /tmp.
- `NVSHMEM_HEAP_KIND=SYSMEM` fails ("invalid argument") on discrete x86 GPUs;
  use the default VIDMEM heap.
- `--num_tokens_per_rank 24576` with world=8 rank-sharing OOMs (11.92 GiB
  workspace per rank); irrelevant for world=1.
- The stock heuristic hard-codes 110 SMs (RTX Pro 5000); without
  `MEGA_HEURISTIC_ALLOW_ANY_SMS=1` + explicit `--k1_sms/--k2_sms/--tx_sms/--rx_sms`
  it refuses to run on this die.
