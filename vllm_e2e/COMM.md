# Communication pattern — DSV4-Flash vLLM e2e (TP4 + EP4, 4x GB200)

Companion to FINDINGS.md / RUNS.md. Goal: a per-channel picture of what
actually crosses the links during real engine steps, to mine perf ideas.
Status: analytic model below is filled in; measured sections are
placeholders until the instrumented runs land (see the plan at the end).

## Why host-side comm logs cannot tell this story

* The mega kernel's dispatch/combine is **device-side** NVSHMEM/NVLink
  (TMA pulls + STGs on peer mappings issued from inside one kernel).
  `NVSHMEM_DEBUG=INFO` covers bootstrap only — there are no host calls to
  log. The ground truth is the kernel's own routing (topk_ids / recv
  counts) plus NVLink hardware counters.
* vLLM's TP allreduce **bypasses NCCL below a size threshold** (custom
  `cross_device_reduce_*` kernels). `NCCL_DEBUG=INFO
  NCCL_DEBUG_SUBSYS=INIT,COLL` still earns its keep for topology + the
  true-NCCL collective inventory, but it sees the smallest slice.

## Analytic wire model (balanced routing expectation)

Geometry: hidden 4096, top-6 of 256 experts, 64 experts/rank, 4 ranks,
43 MoE layers. NVFP4 dispatch wire; bf16 combine wire (production config).

Per-token routing expectation (uniform): a token's 6 experts hit a given
rank with p = 1 - C(192,6)/C(256,6) ~= 0.824 → ~3.29 distinct ranks per
token, of which ~2.47 are remote. Remote topk cells per token:
6 x (3/4) = 4.5.

Per-token payloads:

| channel | bytes per unit |
|---|---|
| dispatch (NVFP4) | 2048 (fp4 data) + 256 (e4m3 SF) + 12 (meta) ~= 2316 B per (token, remote rank) — token payload dedups per rank |
| combine (bf16)   | 4096 x 2 = 8192 B per (token, remote topk cell) — one fc2 row per cell, reduced on the destination |

Per-rank egress, per MoE layer:

| workload | dispatch | combine (bf16) | combine/dispatch ratio |
|---|---|---|---|
| prefill step (4096 tok/rank) | 4096 x 2.47 x 2316 B ~= **23 MB** | 4096 x 4.5 x 8192 B ~= **151 MB** | **6.4x** |
| decode step (128 tok/rank)   | 128 x 2.47 x 2316 B ~= 0.73 MB | 128 x 4.5 x 8192 B ~= 4.7 MB | 6.4x |

x43 layers per step: prefill moves ~1.0 GB dispatch + **~6.5 GB combine**
per rank per step; at the measured ~87 ms fi GPU/step that is ~75 GB/s
sustained combine egress (bursts higher inside each kernel's combine
phase) vs ~12 GB/s dispatch. Decode: ~31 MB + ~202 MB per step.

TP allreduce (per layer, 2x: attn out + ffn out): tokens x 4096 x 2 B
each — prefill ~34 MB/AR (NCCL territory), decode ~1 MB/AR (custom AR).

**What the model already says:**

1. **Combine dominates wire bytes ~6.4x** (bf16 x topk fan-out vs fp4 x
   rank-dedup). This is why quantized combine wires won the 7168/top-8
   microbench — and why they still lost e2e at DSV4 (run 24): at this
   geometry the sustained rate is far below NVLink capacity, so cutting
   bytes buys little; the cost was the forced token-back mode. Re-check
   wires only where combine actually saturates a link (bigger hidden,
   bigger topk, or cross-node EP8 where per-pair bandwidth is lower).
2. Neither channel approaches GB200 NVLink capacity at TP4+EP4 intranode
   → the mega kernel's comm is **latency/skew-bound, not bandwidth-bound**
   here. Consistent with run 27 (in-kernel skew absorption) and run 30
   (lockstep graphs recovered the time without moving fewer bytes).
   Overlap (drop-and-go combine) attacks latency; wire quant attacks
   bandwidth — this model says prioritize the former at this geometry.
3. Dispatch is cheap. Ideas that add dispatch-side redundancy (e.g.
   replicating hot experts EPLB-style) have ~23 MB/layer of headroom
   before dispatch even matches today's combine bytes.

Caveat: balanced-routing expectation. Run 25 measured 14-28x per-expert
load skew; rank-pair aggregation smooths much of it (64 experts/rank),
but the recv-matrix dump below will quantify the real pair imbalance.

## Measured: NVLink hardware counters (2026-07-19, run-33 regimes)

`nvidia-smi nvlink -gt d` deltas over exactly the timed rounds
(bench_offline NVLINK_COUNTERS=1; warmup/boot excluded). ALL NVLink
traffic (mega dispatch/combine + TP allreduce + custom AR):

| cell (graphs) | tok/s | total egress/GPU | avg egress/GPU |
|---|---|---|---|
| native prefill-8k | 45,804 | 2688 GB | 155 GB/s |
| fi_nvfp4 prefill-8k | 53,459 | 2408 GB | 163 GB/s |
| native decode-1k | 28,318 | 2769 GB | 99 GB/s |
| fi_nvfp4 decode-1k | 27,998 | 2484 GB | 88 GB/s |

Findings:
* **~18% of GB200 NVLink capacity at the most intense regime** — the
  latency-bound conclusion from the model is now measured, not inferred.
  Byte-cutting (wire quant) has no lever here; overlap does.
* **fi moves ~10% fewer total bytes than native for identical work** (both
  workloads) — the fp4 dispatch wire vs native's fp8, combine equal (bf16).
* fi's higher *rate* at prefill is just finishing sooner (fewer bytes /
  less time).

## Measured: routing matrices (2026-07-19)

Send-side (src rank 0 -> dst rank) token/cell counts from the extended
FI_MOE_EP_LOAD_STATS hook, eager runs, summed over layers and steps:

* **cells/tokens dedup ratio = 1.83 in BOTH workloads** — the analytic
  expectation (6 topk / 3.29 hit-ranks = 1.82) measured exactly; the wire
  model above is validated.
* **Rank-pair traffic is nearly balanced: max/mean = 1.046 (prefill) /
  1.032 (decode)** despite the 14-28x per-expert skew (64 experts/rank
  aggregate it away). Mild self-affinity (dst=self ~4% above mean).
  Consequences: EPLB's value is straggler *time inside the kernel*
  (per-expert), not link congestion; and per-layer knob grouping stays
  dead (run 26) — the links never see the imbalance.

## Measured: NCCL topology

NCCL INIT (native prefill boot): 32 channels rank-pair P2P/CUMEM, NVLS
multicast available (24 channels), all-ring PXN 0 GDR 1 — standard
single-node NVL topology; nothing exotic. True-NCCL collective inventory
and custom-AR counts deferred to an nsys pass if ever needed — the
counter+matrix data above already answer the sizing questions.

## Ideas ledger (updated as measurements land)

Post-measurement conclusions (2026-07-19):

* **CLOSED: byte-cutting at DSV4 TP4+EP4.** 18% measured utilization —
  quantized wires / compression cannot pay here (run 24's tie now
  explained by counters). Latency/overlap is the only comm frontier at
  this scale: drop-and-go combine, kernel-tail reduce, lockstep graphs.
* **USE: the validated model as an EP8 pre-sizing calculator.** Dedup
  1.83 measured == 1.82 predicted -> trust the model. EP8 exercise
  (256 experts / 8 ranks = 32/rank, top-6, hidden 4096, done 2026-07-19):
  P(rank hit) = 1 - C(224,6)/C(256,6) ~= 0.551 -> ~3.86 remote token
  copies (+56% dispatch bytes vs EP4) and 6 x 7/8 = 5.25 remote combine
  cells (+17%). Per-GPU egress at 8k chunks rises from ~18% to roughly
  22-25% of NVLink — **same-clique NVL72 EP8 stays latency-bound; wire
  quant stays dead there too.** It re-enters only across a lower-bandwidth
  boundary (IB / cross-clique), where combine's 6.4x share makes it the
  target. Also note: 7 peers to straggle instead of 3 -> the skew/overlap
  problem AMPLIFIES at EP8; drop-and-go combine matters more, not less.
* **EPLB scope sharpened:** links balanced to 5% -> EPLB is purely a
  compute-straggler play (25-35% per-layer spread), not congestion; and
  the bandwidth headroom makes hot-expert REPLICATION variants wire-free
  at this scale — wider design space than rebalance-only.
* **Instrument permanently:** NVLINK_COUNTERS=1 + the routing-matrix hook
  cost ~nothing — include in standard cell recipes as a comm-regression
  tripwire (traffic anomalies surface even when tok/s looks plausible).
* MTP (decode batch amplification) — parked; superseded for now by
  concurrency-matched workloads (run 33), which achieve the same
  amortization without engine changes.
