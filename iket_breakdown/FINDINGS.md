# nvfp4_cutedsl mega kernel vs deep_gemm_mega — kernel-only analysis, with IKET markers

Data: SLURM jobs 2359277 + 2359364 (sweep 8..8192 tok/rank) and 2359949 (gap
discrimination). 1×8 B200, EP8, 256 experts top-8, hidden 7168, inter 2048, DSL 4.5.2.
All numbers in this section are **MEGA_TIMING=kernel** (steady-state, back-to-back
launches — no launch/barrier-cold effects). Marker anchors are on the flashinfer
`iket_analysis` branch; clock64 slots mirror the marker names
(`src/src/phase_timing.py`). Full tables: [RESULTS.md](RESULTS.md).

## Kernel-only headline

| tok/rank | 8 | 16 | 32 | 64 | 128 | 192 | 256 | 384 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ratio cutedsl ÷ dg | 0.98 | 0.99 | 0.97 | 0.98 | 1.04 | 1.10 | **1.20** | 1.09 | 0.94 | 0.81 | 0.72 | 0.64 | 0.64 |

The kernel is at dg parity ≤128 tok/rank and wins outright ≥512 (1.56× faster
at 8k). The only kernel-level slowdown was a hump peaked at 256 tok/rank
(+21%) — now resolved as a knob-bucket artifact (next section); after tuning,
what remains everywhere is the serial barrier tail (~39 µs/launch). (dg's "kernel" numbers still include FI wrapper overhead —
no bare-launch thunk for dg — so cutedsl's true relative position is slightly
better than shown everywhere.)

## The 256-tok hump: resolved — a knob-bucket artifact (jobs 2359949, 2360287)

Short version: the hump is a bad default in the 256-max-token knob bucket, and
tuning removes ~40% of it immediately; what tuning cannot remove is the serial
barrier tail, analyzed below.

- The 256 bucket defaults to `epi_warps` token-back, whose combine STG
  (`fc2_store_combine`, `epilogue_refactor.py:2349`) misbehaves at that pool
  geometry: 69 µs at 256 vs 29 µs at 384 (non-monotonic ⇒ config artifact).
- Autotune's winning change is `token_back_mode → standalone_warps` (combine
  push on dedicated warps w12-15): 248.8 → 232.4 µs. Pinning those knobs
  reproduces it (232.5 µs, job 2360287), and the phase table confirms the
  mechanism: `epi_fc2` drops 71 → 33 µs.
- `MEGA_IKR=1` is ruled out (worse at every point, +12–16%).
- **Fix:** adopt `standalone_warps` in the knob table's 256 bucket and re-sweep
  the bucket. Discrimination table: [results/job2359949_kernelgap_extract.txt](results/job2359949_kernelgap_extract.txt).

## The remaining kernel gap: the serial tail (job 2360287)

With tuned knobs at 256 tok/rank the kernel is 232.5 µs vs dg 205.9 (+12.9%),
and the phase table shows exactly where that lives:

| span (tuned 256) | µs |
|---|---:|
| compute pipeline (sched/TMA/MMA/epi loop totals all end) | ~190–195 |
| kernel_total | 234.4 |
| → serial tail after compute | **~40** |

The compute pipeline alone finishes at ~195 µs — *below* dg's total. The whole
residual is the serial tail that runs after the last fc2 tile:
token-back drain + `Tail.NvlinkDrain` (16.5 µs, `token_comm.py:1865`) +
`Tail.NvlinkPublish` (7.3 µs, `:1931`), plus `Dispatch_Barrier` (15.1 µs,
`:1613`) at the front — ≈ 39 µs of serialized cross-rank synchronization per
launch that dg's kernel does not pay at this size. The same fixed costs set
the absolute floor at 8–128 tok/rank (where cutedsl still matches dg only
because dg carries other overheads).

**Kernel-only action items, in order:**
1. Knob table: `standalone_warps` for the 256 bucket (+ re-sweep) — banked,
   ~16 µs.
2. Tail quiesce: fold/elide the NVLink drain+publish pair, or overlap the tail
   with the next layer's dispatch — worth ~24 µs/launch at every token count.
3. Dispatch count-exchange: overlap the ~14 µs barrier with input staging.
4. Do NOT pursue IKR in this regime.

## Phase flow through the megakernel

One launch = one MoE layer; a token is *pulled* to the rank owning its experts,
run through fc1 → SwiGLU → fc2 there, and *pushed* back home. Stations in
causal order (marker names in backticks; steps 3-8 run concurrently on their
own warps):

1. **Prologue** — `pipeline_init_wait`: cluster rendezvous, pipelines armed.
2. **Dispatch counts** — `Dispatch_Prep`: scan local top-k, count tokens per
   expert; `Dispatch_Barrier`: all ranks exchange counts (first cross-rank
   sync).
3. **Token pull** — `Dispatch_Pull`: copy each assigned token's NVFP4 data +
   SFs + weight from its home rank into the local pool; bump `fc1_ready`.
4. **Scheduler** — `Sched_PreInit_Wait` (≈0) then tile generation +
   `sched_publish` into the consumer queue (full queue = backpressure).
5. **TMA loads** — `tma_weight_fc1/fc2`: stream expert weights (the low-token
   floor); `tma_token_fc1` + `tma_token_fc1_wait`: load tokens once step 3
   delivered them; `tma_token_fc2_wait`: spin on fc1 output readiness.
6. **MMA** — `mma_fc1` (gate+up), `mma_fc2` (down-proj); `mma_acquire` for
   slot handoffs.
7. **fc1 epilogue** — `fc1_epi_wait`/`fc1_epi`: SwiGLU + NVFP4 requant +
   store, signal `fc1_done` (releases fc2's B-side load).
8. **fc2 epilogue + combine** — `fc2_epi_wait`/`fc2_epi`: scale and write the
   result back to the token's home rank. epi_warps mode: the cross-rank store
   is `fc2_store_combine` here; standalone/reuse modes: local store + push by
   dedicated warps (`token_back_push`). `epi_flag` publishes done counters.
9. **Kernel tail** — `Tail.Rendezvous`, then `Tail.NvlinkDrain` (all pushes
   landed), `Tail.SharedReset`, `Tail.NvlinkPublish`, `Tail.LocalReset` —
   serial fixed cost, the residual-gap culprit.

## Fixed in-kernel floor (absolute decode latency, both-backends-relative)

These are steady-state kernel costs on every launch — they don't explain a gap
vs dg at ≤128 (parity holds), but they are the biggest absolute levers at tiny
token counts:

- **Kernel-tail NVLink quiesce ~23–26 µs** — `Tail.NvlinkDrain` (~15–18 µs,
  `src/src/token_comm.py:1865`) + `Tail.NvlinkPublish` (~8 µs,
  `token_comm.py:1931`); 15% of an 8-tok kernel. Fold/elide the pair or overlap
  with the next layer's dispatch.
- **Dispatch count-exchange barrier ~14 µs** — `Dispatch_Barrier`
  (`token_comm.py:1613`).
- **fc1 weight-streaming floor ~60–70 µs** — `mma_fc1` (`kernel_fc12.py:2324`),
  fed by `tma_weight_fc1` (`kernel_fc12.py:1733`): every expert's full weights
  stream regardless of token count. Both backends pay it; it is why kernel
  time barely moves from 8 → 128 tok/rank.

## Measured and exonerated (symptoms, not causes)

- `sched_publish` ~50–55% (`kernel_fc12.py:1663`) and `epi_fc1_wait` ~40–48%
  (`epilogue_refactor.py:1467`): consumers pacing on the fc1 weight-stream /
  MMA — mirrors of `mma_fc1`, not independent problems.
- `Sched_PreInit_Wait` ≈ 0 (`token_comm.py:577`): dispatch arrival is not what
  the compute side waits on.
- `dispatch_pull` grows to ~30% at 8k (`token_comm.py:1643`) but runs on
  dedicated warps overlapped with compute.

## Deferred: e2e / barrier-cold behavior (not kernel)

Parked per current focus; evidence kept for when we return to the e2e microbench.

- e2e ratios (barrier-cold full FI forward): 1.19–1.36× at 8–256 tok/rank,
  crossing below 1.0 between 512 and 1024 — entirely explained by a constant
  ~40–60 µs/launch cost dg does not pay (dg e2e−kernel ≈ 8–25 µs, cutedsl
  ≈ 48–87 µs, roughly constant through 8k tokens).
- Direct proof of the mechanism (job 2359944, e2e mode + phase timing): at 8
  tok/rank, `Dispatch_Barrier` inflates **14 µs → 164 µs (75% of the kernel)**
  on rank 0 — per-rank host-side launch jitter is absorbed at the first
  cross-rank sync point, so the slowest rank's launch delay becomes everyone's
  latency. Fix direction: slim/pre-arm the per-call host path (thunk-style),
  reduce sync points.

## Sanity

Instrumentation (`MEGA_PHASE_TIMING=1`) adds ~0–7 µs on 170–2000 µs kernels;
accuracy gate pt=0 vs pt=1 bit-identical (23.175% rel-L2, synthetic microbench
reference). IKET markers no-op on public DSL wheels (dialect is internal-tot
only); the clock64 fallback uses the same phase boundaries, so a future
`run-iket` trace is directly comparable. Autotune cells used an isolated knob
cache (`FLASHINFER_MOE_EP_KNOB_CACHE`) — the default cache is untouched.
