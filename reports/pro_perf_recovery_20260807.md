# Pro fi_cutedsl perf recovery campaign — 2026-08-07

Goal: recover the July pre-SP advantage (fi_cutedsl 1.317x native at Pro
prefill-8k) on the current SP stack, which sits at ~1.09x. Method: recreate
the pre-SP MoE workload (high tokens/rank) via big batches, nsys kernel
attribution, knob/graph/memory tuning, iterate. Nothing pushed; the only
tree change is the local uncommitted `FI_COMBINE_DTYPE` lever in
`vllm/utils/flashinfer_moe_ep.py` (XXX, do not commit).

## Verdict

The July 1.317x is **structurally unrecoverable on this stack at equal
workloads**; the honest recovered number on Pro prefill is **~1.09x
best-vs-best** (1.13x vs a naively-configured native). Three independent
walls, each measured:

1. **SP removed the slice fi was winning.** July: native ran the MoE block
   full-batch on every rank + all-reduce (8x redundant at TP8); fi's 2x-faster
   MoE+transport ate a huge time slice. Now both backends run SP (RS+AG,
   sharded MoE): the MoE mega kernel is ~40% of step time and the SP
   collectives (~20%) are *identical code on both sides*.
2. **KV ceiling forbids the pre-SP tokens/rank.** 8192 tok/rank under SP
   needs 65536-token batches; Pro profile-run OOMs at 64k even at util 0.90
   (both backends), and at 32k capacity only ~46k KV tokens remain. Max
   reachable is 4096 tok/rank.
3. **Amdahl at 4096 tok/rank.** nsys (nsys_pro/, kernel_breakdown.py): fi's
   cutedsl mega is genuinely ~2x faster than native's deep_gemm mega at equal
   tokens/rank (443us vs 858us at 1024 tok/rank, fastest-rank = pure compute),
   but 40% × 2x ≈ 1.15x e2e ceiling, minus launch-skew spin (the fi persistent
   kernel absorbs rank skew in-kernel: waiting ranks show 1405us for the same
   launch) and graph-padding effects. Measured 1.09x is at that ceiling.

## Final Pro prefill table (clean memory config, 3-round medians, tok/s)

Config: GPU_MEM_UTIL=0.95, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0,
MAX_MODEL_LEN=2048, prefill:1024:1, 384 prompts. fi = nvfp4 combine + tuned
knob cache (knob_cache_pro_nvfp4_pre.json, entries at 8k/16k/32k/64k).

| cell | native | fi_cutedsl | ratio |
|---|---|---|---|
| 16k capacity, dense ladder | 41020 | 44826 | 1.093x |
| 32k capacity, capture<=8192 | 41925 | **45559** (rerun 45457) | 1.087x |
| 32k capacity, dense ladder | 40275 | 43464 | 1.079x |
| 32k capacity, sparse ladder (2048,8192,32768) | 34113 | 37348 | (both padding-hurt) |

Champion config: **fi_cutedsl, 32k batched tokens, capture capped at 8192**
= 45559 tok/s, +11% over the best native cell and +2% over fi's own 16k.
Both backends prefer big prefill steps *uncaptured* — graph padding to the
next capture size costs more than eager launch overhead.

## Methodology traps found (worth keeping)

- **KV starvation invalidates big-batch cells silently.** At util 0.90 +
  graph profiling, Pro pre32k held 14,143 KV tokens (3.45x concurrency): the
  scheduler never formed 32k batches and "pre32k" numbers were ~half of real.
  Always check `GPU KV cache size` / `Maximum concurrency` per cell.
- **Sparse capture ladders tax big-batch prefill ~15-18%** (steps pad to the
  next size) — for BOTH backends.
- **--standalone torchrun rendezvous times out in-container** (c10d resolves
  the node FQDN); use default static rendezvous with explicit
  `--master_addr=127.0.0.1 --master_port=<unique>` per tune.

## FI-team items

1. **ILLEGAL_INSTRUCTION under memory pressure**: fi_cutedsl + nvfp4 combine
   + 32768-token graph crashed deterministically (2/2) mid-steady-state in
   the KV-starved config (util 0.90, 14k KV free); the same cell is clean at
   util 0.95. 16k/24k graphs clean even when starved. Smells like workspace/
   graph-pool collision, not a size-2^15 kernel bug (full 32k graph ran clean
   3 rounds with headroom). Repro: job_pro_bb3_discrim.sh nvfp4_graphs cell.
2. bf16 combine staging cannot fit 32k capacity on Pro (init "No available
   memory") — quantized combine wire is mandatory there.
3. Prior items stand: residual ~0.04 Pro nvfp4-compute acc gap; decode
   non-monotonicity; dummy-load crash on main; nsys-vs-fi-graphs deadlock.

## Accuracy

- 200q requant nvfp4-combine (bb2): 0.855 (trunc 16) vs 0.885 bf16 band.
- 500q requant nvfp4-combine (job 2372181): **0.842** (421/500, trunc 37)
  vs requant bf16 0.866, native 0.904. Two independent samples both put
  nvfp4 combine ~2.4-3.0% below bf16 combine on Pro (~1.5 sigma each,
  consistent direction) — treat as a real, small acc cost of the nvfp4
  wire on Pro. Note nvfp4 combine is *mandatory* at 32k capacity (bf16
  staging doesn't fit); at <=16k capacity bf16 combine remains available
  and the ship decision (bf16 default) stands.

## Where real headroom would come from (not vllm-config reachable)

The traces show both backends paying the same NCCL SP ReduceScatter+AllGather
(~20% of step). fi's symm-mem dispatch/combine could absorb the SP resharding
(dispatch directly from the sharded layout), which is the one visible lever
that grows the fi-differentiated slice. Kernel/API work for the FI team.

## Artifacts

- Jobs: 2372065 (tunes+bb2), 2372056 (nsys), 2372102 (bb3), 2372123 (bb4),
  2372152 (bb5), 2372155 (bb6), 2372157 (bb7), 2372162 (bb8), 2372179 (acc).
- Traces + analyzer: nsys_pro/{native,ficd}_pre8k.sqlite, kernel_breakdown.py.
- Results: pro/bb*.json, pro/bb5_*.json; knob cache: knob_cache_pro_nvfp4_pre.json.
