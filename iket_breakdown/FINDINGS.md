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

The kernel is at dg parity ≤128 tok/rank and wins outright ≥512 (1.56× faster at
8k). The only kernel-level slowdown is a **hump peaked exactly at 256 tok/rank**
(+21%), not a trend. (dg's "kernel" numbers still include FI wrapper overhead —
no bare-launch thunk for dg — so cutedsl's true relative position is slightly
better than shown everywhere.)

## The 256-tok hump, discriminated (job 2359949)

| tok/rank | dg | cutedsl default | + `MEGA_IKR=1` | + `MEGA_KNOBS=auto` |
|---:|---:|---:|---:|---:|
| 128 | 188.4 | 195.5 (+3.8%) | 226.3 (worse) | 195.5 (no change) |
| 192 | 199.6 | 220.1 (+10.3%) | 252.8 (worse) | — |
| 256 | 205.9 | 248.8 (+20.8%) | 277.6 (worse) | **232.4 (+12.9%)** |
| 384 | 238.5 | 261.1 (+9.5%) | 267.3 (worse) | 255.0 (+6.9%) |
| 512 | 278.5 | 263.2 (−5.5%) | 269.2 | — |

- **In-flight REDG combine (`MEGA_IKR=1`) is ruled out — it is *worse* at every
  point in the window** (+12–16%): its per-fc2-tile cross-rank atomic-adds cost
  more than the staged combine here (`epi_fc2` grows 71.0 → 113.5 µs at 256).
- **~40% of the hump is a knob-table gap.** Online autotune recovers 16.4 µs at
  256; the winning change in the 256-max-token bucket is
  `token_back_mode: epi_warps → standalone_warps` (combine push moved off the
  epilogue warps to a dedicated warp group), tiler unchanged (256×128×256).
- **The smoking gun is `epi_fc2` (combine store) being anomalous exactly in the
  256 bucket:** work = 69 µs (27% of kernel) at 256 but only **29 µs at 384** —
  the cost is non-monotonic in tokens, so it is a *configuration* artifact of
  the epi-warps token-back at that pool geometry, not a fundamental per-token
  cost. Markers: `fc2_store_combine` (`epilogue_refactor.py:2349`) with the
  wait split via `fc2_epi_wait` (`epilogue_refactor.py:2185,2199`).
- **Residual after tuning: +12.9% at 256.** Phase table at the tuned point not
  yet captured (autotune cell ran without `+pt`); next probe: rerun 256 with
  the winning knobs pinned via `MEGA_KNOBS` + `MEGA_PHASE_TIMING=1`.

**Action items, kernel-only:**
1. Fix the knob table's 256-max-token bucket (adopt `standalone_warps`
   token-back; re-tune the bucket) — recovers ~7 µs of the ~43 µs gap
   immediately, more once the bucket is properly swept.
2. Profile the epi-warps combine STG pattern at 128–384 pool geometries
   (`fc2_store_combine` marker) — why it degrades only there.
3. Do NOT pursue IKR for this regime.

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
