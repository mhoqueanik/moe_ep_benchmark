# Why nvfp4_cutedsl loses to deep_gemm_mega at low tokens/rank — exact causes, with IKET markers

Data: SLURM jobs 2359277 + 2359364 (1×8 B200, EP8, 256 experts top-8, hidden 7168, inter 2048, DSL 4.5.2).
Each cause points at the IKET marker that measures it (same names as the clock64 `PT` slots in
`src/src/phase_timing.py`); file:line anchors are on the flashinfer `iket_analysis` branch.
Full tables: [RESULTS.md](RESULTS.md).

**Headline:** the GEMMs are never the problem. The kernel is at dg parity ≤128 tok/rank
(0.97–1.04×); cutedsl loses e2e (~1.2×) purely on a **constant ~40–60 µs per-launch tax** that
dg does not pay. That tax stops mattering above ~512–1024 tok/rank, where cutedsl wins outright
(1.56× faster at 8k).

## Primary causes (the low-tok/rank tax, ~40–60 µs every launch)

- **Kernel-tail NVLink quiesce — ~23–26 µs/launch, pure fixed cost.**
  Every launch ends with a 3-barrier sense-reversing NVLink sequence plus counter resets.
  - Markers: `Tail.NvlinkDrain` (~15–18 µs) — `src/src/token_comm.py:1865`;
    `Tail.NvlinkPublish` (~8 µs) — `token_comm.py:1931`
    (plus `Tail.Rendezvous`/`Tail.SharedReset`/`Tail.LocalReset`, all <1 µs).
  - At 8 tok/rank that is **15% of the whole kernel**; dg has no equivalent tail of this size.
  - Fix direction: fold/elide the drain+publish pair, or overlap the tail with the next
    layer's dispatch.

- **Dispatch count-exchange barrier — ~14 µs/launch, fixed cost.**
  The all-rank exchange of expert send counts before any token moves.
  - Marker: `Dispatch_Barrier` — `token_comm.py:1613` (slot `dispatch_barrier`, 8.2% of kernel
    at 8 tok/rank).

- **Cold-launch arrival skew amplifies both barriers — the ~35–50 µs e2e-vs-kernel delta.**
  Steady-state (kernel mode, back-to-back thunk launches): cutedsl = dg. Barrier-cold e2e:
  cutedsl pays +48 µs vs its own kernel time, dg only +8–10 µs — and the gap persists at 8k
  tokens (+87 vs +31 µs). A kernel with 4 cross-rank sync points inherits the slowest rank's
  launch delay at every one of them.
  - Evidence markers: `Dispatch_Barrier` and `Tail.*` under e2e vs kernel timing mode;
    host side, the FI forward path (arg prep / workspace reset / output copy) ahead of the
    kernel is what staggers rank arrival — the kernel-mode thunk shows the ceiling.
  - Fix direction: pre-arm/slim the per-call host path (extend the thunk approach to the
    serving path), reduce in-kernel cross-rank sync points.

## Secondary causes (per-token costs, visible at ≥128 tok/rank; NOT a divergence)

- **fc2 epilogue combine store (cross-rank peer STG) — linear, settles at ~20% of kernel.**
  Work = `epi_fc2` − `epi_fc2_wait`: 29 µs @128 → 72 µs @256 → 410 µs @8192. This is what
  turns the 256-tok transition point into a visible 1.20× kernel gap before amortization wins.
  - Markers: `fc2_store_combine` — `src/moe_nvfp4_swapab/epilogue_refactor.py:2349`;
    wait split via `fc2_epi_wait` — `epilogue_refactor.py:2185,2199`.
  - Fix direction: `MEGA_IKR=1` (in-kernel REDG reduce), quantized combine wire, wider stores.

- **dispatch_pull grows to ~30% at 8k** (`Dispatch_Pull` — `token_comm.py:1643`) but runs on
  dedicated warps overlapped with compute; secondary pipelining target only.

## Measured but exonerated (symptoms, not causes)

- **`sched_publish` "backpressure" ~50–55% everywhere** (`kernel_fc12.py:1663`) and
  **`epi_fc1_wait` ~40–48%** (`epilogue_refactor.py:1467`): consumers pacing on fc1.
- **`mma_fc1` ~60–70 µs at low tokens** (`kernel_fc12.py:2324`, fed by `tma_weight_fc1`
  `kernel_fc12.py:1733`): the fc1 weight-streaming bandwidth floor — every expert's full
  weights stream regardless of token count, and **both backends pay it equally** (hence
  kernel parity at low tokens).
- **`Sched_PreInit_Wait` ≈ 0** (`token_comm.py:577`): dispatch arrival is NOT what the compute
  side waits on — the counts are ready before the sched warp asks.

## Sanity

- Instrumentation overhead: ~0–7 µs on 170–2000 µs kernels; accuracy gate pt=0 vs pt=1
  bit-identical (23.175% rel-L2 vs synthetic reference).
- IKET markers no-op on public DSL wheels (dialect is internal-tot-only); the numbers above
  come from the clock64 fallback (`MEGA_PHASE_TIMING=1`) using the same phase boundaries, so
  a future `run-iket` trace is directly comparable.
