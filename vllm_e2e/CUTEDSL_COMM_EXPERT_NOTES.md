# CuTeDSL MegaMoE comm pattern — findings + questions for the kernel team

2026-07-20. All numbers measured on 4x GB200, TP4+EP4 intranode, DSV4-Flash
(hidden 4096, top-6 of 256 experts, 43 MoE layers). Sources: `COMM.md`,
`RUNS.md` runs 22-33, `kernel_src/cutedsl_megamoe/TUNING.md`,
`flashinfer/moe_ep/todo_multinode.md`.

---

## 1. The kernel in one picture (our read — please correct)

One fused kernel per MoE layer. All comm is direct NVLink load/store on
peer-mapped symmetric-heap addresses — no host API on the datapath, no RDMA.

```
 rank 0                                rank 1..3
┌───────────────────────────┐
│ DISPATCH                  │  fp4 payload + e4m3 SF + 12B meta
│  store token → peer pool ─┼──────────────►  recv pool [world x tokens]
│  red.add.release.sys flag─┼──────────────►  fc1_ready_counter
├───────────────────────────┤
│ FC1 / FC2 (grouped GEMM)  │  TMA-B loads SPIN on fc1_ready  ◄── waits here
├───────────────────────────┤
│ COMBINE (token-back)      │  one bf16 fc2 row per (token, topk cell)
│  cp.reduce.async.bulk ────┼──────────────►  reduce-add at source rank
│  fc2_done flags           │
└───────────────────────────┘
 sync fabric: nvlink_barrier_signal/counter, flag_batch / epi_flag_batch
 (how many flags per release), token-back scheduler counter
```

Peer addressing — one constant delta per peer, computed once at init:

```
 my heap:    [ base ................ buf ]      buf = base + off
 peer heap:  [ base' ............... buf' ]     buf' = buf + delta[peer]
                                                 (same off — collective
                                                  allocation order must match)
 delta[16] passed BY VALUE (128B struct) → ranks > 16 = untested IR path
```

Variants: `in_kernel_fc2_reduce` (ikr) replaces the combine epilogue with
cross-rank REDG atomics (nondeterministic order); quantized combine wires
(fp4/fp8) exist but force `reuse_dispatch_warps` token-back and no ikr.

## 2. What crosses the wire (validated model)

Measured dedup 1.83 remote copies/token vs 1.82 predicted → model is exact.

```
 per token:   DISPATCH  ~2.5 remote ranks x 2.3 KB   (fp4, dedup per rank)
              COMBINE   4.5 remote cells x 8.2 KB    (bf16, per topk cell)

 per rank / prefill step (4096 tok x 43 layers):
   dispatch  ~1.0 GB   ██
   combine   ~6.5 GB   █████████████        ← combine is 6.4x of the bytes
```

## 3. The five findings that matter

**F1 — Latency-bound, not bandwidth-bound.** NVLink counters over timed
rounds: **~18% of GB200 capacity** at the hottest regime. Byte-cutting is
dead at this scale — the fp4 combine wire that won -16..19% in the 7168/top-8
microbench LOST e2e here (run 24: the forced token-back mode cost more than
4x fewer bytes bought).

**F2 — The real cost is in-kernel waiting on stragglers.** Per-layer
decomposition (run 27, eager prefill):

```
 eager, per layer (~us):        fi mega             deep_gemm
 rank arrives early ──►  ┌────────────────┐   ┌────────────────┐
                         │ WAIT ~850      │   │ WAIT ~0        │
                         │ (in-kernel,    │   │ (combine is    │
                         │  all 43 layers)│   │  arrival-order │
                         │ WORK 467-1038  │   │  independent;  │
                         └────────────────┘   │  skew hides in │
                                              │  TP allreduce) │
                                              └────────────────┘
 CUDA-graph lockstep replay:
                         ┌──────────┐
                         │ WAIT ~0  │  → fi mega 668us vs native 872us
                         │ WORK     │    (1.3x), e2e prefill 1.14x native
                         └──────────┘    — same bytes, zero kernel changes
```

**F3 — Links stay balanced even when experts don't.** 14-28x per-expert
load skew aggregates to max/mean **1.046** rank-pair traffic (64
experts/rank). So EPLB is purely a compute-straggler play, and hot-expert
REPLICATION is wire-free (dispatch has ~23 MB/layer headroom).

**F4 — Schedule knobs can't fix stragglers.** Skew-aware sweeps (run 26):
static vs atomic_counter scheduling ties within noise at real skew — layer
time is bounded by the straggler expert's actual WORK, which ordering
cannot move.

**F5 — Small-batch weak spots.** ikr loses at small token counts even when
tuned for them (wins only >=2048 at 7168/top-8). And the explicit TopkReduce
tail costs +32us/decode step — the mega kernel itself already beats dg at
decode shapes (124 vs 136us); the tail is the remaining 3%.

**Scale-out preview** (validated model, EP8 same clique): +56% dispatch /
+17% combine bytes → ~22-25% utilization, still latency-bound; but 7 peers
to straggle instead of 3 → **the waiting problem amplifies, not the
bandwidth problem**. Wire quant only re-enters across IB / cross-clique.

## 4. Questions (ranked by measured value here)

**A. Combine overlap — the ~850us/layer eager lever (F2)**
1. Where exactly does the kernel bind to peer arrival — per-expert-tile
   fc1_ready spins, or a bulk phase barrier? Could combine emit per-tile as
   FC2 tiles finish ("drop-and-go"), so a fast rank streams out while a slow
   rank still computes?
2. Could dispatch→FC1 tolerate partial arrival (start on ready expert rows,
   revisit) instead of spin-until-complete?
3. 43 identical layers back-to-back: can layer L+1's dispatch overlap layer
   L's combine tail (persistent kernel / fused pair)?

**B. Small-batch combine (F5)**
4. Feasibility of a **deterministic** in-kernel tail reduce for small token
   counts (fixed per-token reduction order), replacing the separate
   TopkReduce launch without ikr's atomic cost?
5. What drives ikr's small-token penalty — REDG latency with too few tokens
   to pipeline, or destination-line contention? Would single-writer
   owner-rank accumulation (no atomics) work at decode shapes?
6. **NVLS/multimem:** NCCL negotiates NVLS multicast on this fabric. Could
   combine use `multimem.ld_reduce` (switch-side reduction, one fetch)
   instead of P-way unicast fan-in — and does the symmetric heap support
   multimem mappings today?

**C. Scale-out: EP8 / NVL72 / multinode**
7. What breaks first past 16 ranks — the by-val delta struct, flag arrays,
   or recv-pool memory (world x tokens)? Any hierarchical (clique-local +
   cross-clique) dispatch design that avoids linear recv-pool scaling?
8. On MNNVL fabric, is one-delta-per-peer valid for every heap sub-region,
   or do fabric mappings need per-region deltas? (Our multinode guard allows
   MNNVL behind an env flag, unvalidated.)
9. What latency multiplier should we budget for `red.add.release.sys` +
   flag protocol over fabric vs intranode — do flag_batch sweet spots shift
   enough to need fabric-specific tune profiles?
10. For IB multinode: NVSHMEM device puts (IBGDA) inside this kernel, or a
    proxy/staging redesign? Which parts of the current kernel survive?

**D. Stragglers + design couplings**
11. Since scheduling can't help (F4): is tile-level work stealing across
    experts feasible in the current scheduler, or is expert-granularity
    EPLB/replication the only lever? Does the kernel admit "one expert
    served by two ranks, token set split" without a new routing preprocess?
12. Why do quantized combine wires hard-require `reuse_dispatch_warps`?
    Relaxing it would make wire choice orthogonal to schedule — design
    constraint or implementation shortcut? (This coupling decided run 24.)

**E. Toolchain**
13. cutlass-dsl 4.5.x emits 34-54% slower code than 4.6.1 for these kernels
    (vLLM pins 4.5.2). Known lowering regression we can work around in
    kernel source? **ANSWERED 2026-07-22: yes — cutedsl_megamoe MR!27 peels
    the last k-tile out of the fc12 MMA-consumer mainloop under ==4.5.2 and
    restores full 4.6.1 parity (mxfp8 never regressed; only the nvfp4
    swap-AB mainloop was affected).**
14. `cute.compile` of one mega-kernel candidate takes **~12 min at
    256-expert geometries** (measured 2026-07-20; scales with experts/rank,
    not hidden). This makes on-hardware knob tuning nearly unusable. Is the
    cost in DSL lowering or ptxas, is it parallelizable/cacheable on your
    side, and is a pre-compiled tactic library (cubin-style, like the
    TRT-LLM MoE backends ship) on the roadmap?

## 5. One-line asks

* Now: drop-and-go combine (A1) + deterministic tail reduce (B4) — together
  they close both measured e2e gaps (eager prefill wait, decode 3%).
* Next: NVLS multimem combine feasibility (B6) — biggest structural win.
* Gate for NVL72/MNNVL bring-up: C7-C9.
