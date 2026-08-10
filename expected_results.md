# Expected results — 1x RTX PRO 6000, `moe_sm120_mxfp8_split` (DSV4)

Every number here was measured 2026-08-10 on one **single-GPU RTX PRO 6000**
node of the `rtxpro6000` SLURM partition. If your run lands outside the
tolerances in §2, check §5 — every failure mode listed there actually
happened while bringing this up.

Raw CSVs + full write-up: `results/sm120/dsv4_split_rtxpro6000/`.
Baseline being compared against: hanyueh's 8x RTX Pro 5000 EP8/DP2EP4 tables
(`results/sm120/dsv4_split_rtxpro6000/reference_rtxpro5000_numbers.md`).

---

## 1. Environment

| item | value |
|---|---|
| node | `2u2g-emr-0462..0477` / `2u2g-gen-0773-0774` (SLURM partition `rtxpro6000`, 1 GPU/node) |
| GPU | RTX PRO 6000 Blackwell **Server Edition**, sm_120, **188 SMs**, 96 GB (94.97 usable) |
| host | x86_64 (Emerald Rapids), 251 G RAM, 126 G /dev/shm, driver 595.84.01 |
| software | fresh venv from `ci/setup_venv.sh` + `ci/requirements.txt`: torch 2.13.0+cu130, nvidia-cutlass-dsl[cu13] 4.6.0, nvshmem4py-cu13, python 3.12 |
| kernel repo | `cutedsl_megamoe` @ `45f5e56` (branch `hanyueh/sm120-mxfp8-split`) **+ the local changes in §4** |

**Filesystem trap:** these nodes do NOT mount `/home/scratch.mhoqueanik_gpu`
(the job dies instantly with no log). Stage the repo source (~6 MB, no .git)
to NFS home — staged copy at `/home/mhoqueanik/sm120_rtx/cutedsl_megamoe` —
and build the venv on node-local `/tmp` (home quota is 5 G).

## 2. Topology and inputs

**Topology: world=1.** One torchrun rank, one GPU, no NVSHMEM peer traffic, no
dispatch/combine cost, no NCCL. This is deliberately NOT the reference's
topology (8x RTX Pro 5000, one GPU per EP rank, PCIe P2P): this cluster has no
multi-GPU sm_120 node, and multi-node is impossible here — SLURM registers
every rtxpro6000 node as an isolated Level-0 switch with no parent
(multi-node `sbatch` fails with "Requested topology configuration is not
available"), and the fabric is a single 25 GbE RoCE port anyway.

**Inputs (DSV4-flash shape).** topk=6, hidden=7168, intermediate=6144,
MXFP8xMXFP8 (e4m3, block-32 scales), bf16 fc2 output, balanced routing,
`gate_up_clamp 10`, seed 1234 (runner default), 10 warmup / 100 timed iters,
CUDA-event p50.

Two expert-count variants, and reading them right matters:

- **`--num_total_experts 48` ("EP1-proxy") — the primary reference.** One
  GPU doing exactly one EP8 rank's compute: 48 local experts, same
  rows-per-expert as the reference at equal tokens/rank, ~4.3 G weights.
  Directly comparable to the per-GPU EP8 baseline, minus comm.
- **`--num_total_experts 384` (true EP1) — secondary.** All experts local:
  streams the full ~35 G weight bank every iteration and is
  weight-bandwidth-bound (flat ~18.7 ms floor from 512-2048 tokens). Do not
  compare its TFLOP/s against the EP8 baseline.

## 3. Command

`sbatch /home/mhoqueanik/sm120_rtx/perf_logs/sm120_perf_job_rtxpro6000_split_ep1_48exp.sh`
(48-expert sweep; `..._ep1.sh` is the 384-expert variant, `sm120_tune_k1k2_rtxpro6000.sh`
the SM-split scan). The core invocation:

```bash
export MEGA_STRICT_TORCH_REF=0 MEGA_SPLIT_INCLUDE_K3=1
export MEGA_HEURISTIC_ALLOW_ANY_SMS=1          # see §4
export NVSHMEM_SYMMETRIC_SIZE=8G NVSHMEM_DEBUG=ERROR
# NO NVSHMEM_HEAP_KIND=SYSMEM here — see §5

torchrun --standalone --nproc_per_node=1 \
  -m moe_sm120_mxfp8_split.mega_runner \
  --data_parallel_size 1 --tensor_parallel_size 1 \
  --num_tokens_per_rank <T> --num_topk 6 --num_total_experts 48 \
  --hidden 7168 --intermediate 6144 --fc2_output_dtype bfloat16 \
  --route_distribution balanced --gate_up_clamp 10 \
  --cluster_shape_mnk 1,1,1 --enable_static_expert_shape \
  --split_launch green_graph --comm_backend p2p_direct \
  --k1_sms 128 --k2_sms 60 --tx_sms 0 --rx_sms 0 \
  --perf_run --skip_ref_check --use_cuda_events \
  --perf_warmup 10 --perf_iters 100
```

**SM split: `--k1_sms 128 --k2_sms 60`** is the tuned point for 188 SMs
(scan in `k1k2_scan_kernel.csv`; the stock heuristic's 72/38 is a 110-SM
value). K1 must be a **multiple of 8** — the green-context API rounds other
values and the runner's partition check then aborts — and K1+K2 must equal 188
(`tx/rx = 0` under `p2p_direct`).

## 4. Changes needed on top of upstream `45f5e56`

The upstream split runner hard-codes RTX Pro 5000 (110 SMs) in three places
and cannot run on this die without the following (all uncommitted in
`/home/scratch.mhoqueanik_gpu/cutedsl_megamoe_sm120/cutedsl_megamoe`, mirrored
in the staged copy):

1. **`heuristic.py`** — `MegaMoEHeuristicInput.validate()` rejects
   `num_sms != 110`; now bypassable with `MEGA_HEURISTIC_ALLOW_ANY_SMS=1`
   (explicit `--k1_sms/--k2_sms/--tx_sms/--rx_sms` then required; the
   total-coverage check still enforces they sum to the device SM count).
2. **`mega_runner.py`** — passes the real `multi_processor_count` into the
   heuristic input (was: implicit 110), and `MegaMoETester.__init__`'s
   `total_sms != 110` assert now checks against the queried device SM count.
3. Not needed for world=1 but present for rank-sharing/multi-node work:
   the `MEGA_SINGLE_GPU_GLOO=1` shim extended to `expert_dp.py` (device-fold,
   gloo EP/TP groups, CPU UID broadcast, CPU-staged all_gather) and
   `gpu_topology.py` (unique-GPU check keyed on (hostname, pci_bus_id)).

On an actual RTX Pro 5000 all of these are behavior-neutral.

## 5. Failure modes actually hit

- **Job dies in ~18 s, no log:** output path or repo on
  `/home/scratch.mhoqueanik_gpu` — not mounted here (§1).
- **`NVSHMEM ... mem_heap.cpp: cuda failed with invalid argument`:**
  `NVSHMEM_HEAP_KIND=SYSMEM` set. That's a GB10-ism (coherent unified
  memory); discrete x86 GPUs need the default VIDMEM heap.
- **`CUDA K2 partition differs from heuristic: actual=X, expected=Y`:**
  K1 not a multiple of 8, or K1+K2 != 188.
- **`the current heuristic is specialized for RTX Pro 5000` /
  `config must cover all 188 device SMs`:** missing
  `MEGA_HEURISTIC_ALLOW_ANY_SMS=1` or the §4 patches.
- **OOM at 24576 tokens with world=8 rank-sharing:** `local_workspace` is
  11.92 GiB/rank. Irrelevant at world=1; fundamental at world=8 on one card.

---

## 6. Expected numbers

### 6.1 EP1-proxy, 48 experts — primary (p50, ±5% run-to-run)

Measured at K1=120/68; tuned K1=128/60 column where measured. TFLOP/s uses the
baseline's FLOP accounting (ref time x ref TFLOP/s at equal tokens).

| tokens/rank | p50 (120/68) | tuned p50 (128/60) | vs Pro 5000 EP8 rank |
|---|---|---|---|
| 16 | 2,286 us | — | 0.61x |
| 32 | 2,417 us | — | 0.59x |
| 64 | 2,435 us | — | 0.65x |
| 128 | 2,473 us | — | 0.69x |
| 512 | 2,787 us | — | 0.76x |
| 1,024 | 3,888 us | — | 0.80x |
| 2,048 | 5,969 us | — | 0.78x |
| 4,096 | 10,503 us | — | 0.82x |
| 8,192 | 19,185 us | **18,020 us** | 0.87x → 0.93x |
| 12,288 | 27,356 us | — | 0.90x |
| 16,384 | 37,404 us | — | 0.88x |
| 20,480 | 45,977 us | — | 0.89x |
| 24,576 | 53,546 us | **50,230 us** (147.8 TF) | 0.92x → **0.98x** |

**Sanity anchor:** tuned 24,576-token p50 ≈ 50.2 ms = 0.98x of a Pro 5000 EP8
rank (49,117.60 us / 151.101 TFLOP/s). Above ~56 ms is a >10% regression.

**The headline finding, not a bug in the run:** even tuned, 188 SMs only
*match* a 110-SM Pro 5000 rank per-GPU (and this run has no comm competing).
The extra 78 SMs do not translate at this operating point — profile
occupancy/schedule/clocks before trusting PRO 6000 projections.

### 6.2 K1/K2 scan (48 experts) — shape of the curve

| K1/K2 | 24,576 tok p50 | 8,192 tok p50 |
|---|---|---|
| 120/68 | 52,641 us | 19,368 us |
| **128/60** | **50,230 us** | **18,020 us** |
| 136/52 | 59,571 us | 21,894 us |
| 144/44 | 66,436 us | 24,140 us |
| 152/36 | 79,403 us | 28,703 us |

Sharp minimum at 128; giving K1 more SMs degrades fast. (50 iters; scan p50s
run ~2-5% above the 100-iter sweep at the same split — compare within the scan.)

### 6.3 EP1, 384 experts — bandwidth-bound anchors (p50)

16 tok: 3,632 us · 512: 18,743 us · 2,048: 18,696 us (the ~18.7 ms floor) ·
8,192: 28,556 us · 24,576: 61,639 us (120.4 TF).
