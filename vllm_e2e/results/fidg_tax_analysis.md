# fi_dg decode "tax" — nsysnode d1 trace analysis (2026-07-20)

Traces: `results/nsysnode_{native,fidg,fi}_d1.sqlite` (nsys node-mode, 4 GPUs,
workload `decode:128:64`, 1024 prompts, max_num_batched_tokens 4096,
max_num_seqs 1024, FULL decode graphs, 83 capture sizes).
Question: why is fi_dg ~4.9% slower than native at decode when both launch the
IDENTICAL `sm100_fp8_fp4_mega_moe_impl` kernel.

## 0. Data integrity — read this first (method req. 5)

The requested "steady-state timed decode region" **does not exist in any of
the three traces**. Two independent failures, verified, not papered over:

1. **All three runs crashed during round 1** (the timed round). Logs
   (`logs/nsysnode_*_d1.log`) show round 0 (warmup) completing
   (native 21099.6 / fidg 19380.9 / fi 19756.6 total tok/s), then round 1
   stalling mid-generate (native: progress frozen at 398/1024 prompts) and a
   NCCL watchdog exception killing the run.
2. **CUPTI event caps truncated collection** ("Not all CUDA events might have
   been collected", ~3.5–3.7M events/worker in DIAGNOSTIC_EVENT):
   - native: recording blackout 165.2–168.3 s; final 27 decode steps recorded
     at ~50% event sampling (22/43 megas, 21/43 attention, half of everything,
     uniformly); recording hard-stops at 169.57 s.
   - **fidg: recording dies at 190.2 s, DURING round-0 prefill (step ~24/32).
     The fi_dg trace contains ZERO real decode steps.** All 1161 of its
     in-graph megas are 27×43 lockstep warmup replays.
   - fi: 27 real decode steps recorded at ~30% sampling (13/43 megas).

Consequently the "43 launches/step" check: **holds exactly (43/43, verified
via 43 distinct graphNodeIds per decode graph and 43-mega eager prefill
groups) everywhere recording was complete**; the sampled tails show 22/43
(native) and 13/43 (fi) — a uniform CUPTI drop artifact, since every kernel
name in those windows is down by the same factor (window busy 18.03 ms
recorded vs 36.5 ms expected for native).

What IS comparable, and what the numbers below use:

- **Lockstep block**: cudagraph 11228 (the bs-1024 FULL decode graph — same
  graphId in all three traces) replayed 27× back-to-back at the end of engine
  init, all 43 megas recorded, all 4 devices, ranks in lockstep. Steps 3–26
  used (a 58–60 ms hiccup follows replay 1 in every trace).
- **Round-0 prefill**: eager 4096-token chunked-prefill steps (43 megas each,
  exactly): native steps 1–17, fidg steps 0–23, fi steps 0–21.
- **Round-0 real decode tails**: native 27 steps, fi 27 steps (sampled);
  they match the lockstep block to <0.1% (native cadence 34.56 vs 34.54 ms,
  mega 310 vs 307 us; fi 32.94 vs 32.95 ms), which validates using the
  lockstep block as a proxy for fi_dg's unrecorded decode.

## 1. Segmentation sanity (method req. 1)

Inter-mega-launch gap histogram on device 0 is bimodal per phase, not
globally: decode graph replays have intra-step mega gaps of 400–500 us and
step-boundary gaps ~2.3 ms; eager prefill steps have intra-step gaps of
2–5 ms (attention/dense work between megas) and boundary gaps ~10 ms.
Steps were therefore segmented per phase (graph replays grouped by
graphNodeId cycle; eager steps by >8 ms gaps), then verified to contain
exactly 43 megas. Cross-device alignment by time overlap is exact (per-step
start offsets across devices < 3 us in-graph, < 1 ms eager).

## 2. In-graph decode steps: fi_dg is FASTER than native (answers 4b)

Per-step averages, lockstep replays 3–26, identical on all 4 devices
(device spread < 0.1%):

| metric (per step) | native | fi_dg | fi_nvfp4 |
|---|---|---|---|
| step wall (start→start) | **34.537 ms** | **33.842 ms (−2.0%)** | **32.952 ms (−4.6%)** |
| wall min–max over 23 steps | 34.425–34.630 | 33.750–33.978 | 32.902–33.058 |
| mega span (first start→last end) | 32.261 ms | 31.584 ms | 30.672 ms |
| Σ mega durations (43×) | 13.190 ms | 13.198 ms (+0.06%) | 11.031 ms |
| mega mean | 306.7 us | 306.9 us | 256.5 us |
| Σ allreduce (87× two_shot) | 3.222 ms | 3.182 ms | 3.135 ms |
| device idle inside step | 1.22 ms | 1.20 ms | 1.11 ms |
| in-graph fraction | 43/43 megas in graph | 43/43 | 43/43 |

Real-decode tails confirm: native 34.56 ms/step, mega 310 us, allreduce
41.2 us avg (vs 37.0 lockstep — mild real skew); fi 32.94 ms/step, mega
257 us, topk_reduce 33.6 us ×13 recorded.

Full per-kernel diff of the in-graph step, fi_dg − native (only |Δ|>30 us/step):

| kernel | native | fi_dg | Δ/step |
|---|---|---|---|
| `_prepare_megamoe_inputs_kernel` | 43× / 792 us | absent | **−792 us** |
| `mxfp8_quant_and_process…DataPreprocess` | absent | 43× / 132 us | **+132 us** |
| `two_shot_all_reduce_kernel_inplace` | 3222 us | 3182 us | −40 us |
| `sm100_fp8_fp4_mega_moe_impl` | 13177 us | 13211 us | +34 us |
| everything else | | | ±<25 us |
| **total busy** | 36.37 ms | 35.72 ms | **−610 us** |

So inside the decode graph, fi_dg **replaces** native's 18.4 us/layer input
staging kernel with flashinfer's fused 3.1 us/layer DataPreprocess and is a
net **0.70 ms/step (2.0%) faster**. Mega, allreduce, attention, idle — all
identical. Cross-device skew in-graph is nil for both (per-layer start spread
med 1.6 us, p90 2.7–2.8 us; per-layer duration max−min med 1.8–2.0 us).

**The 4.9% decode deficit cannot come from graph-replayed decode steps.**

## 3. Where the tax actually lives: the round's eager prefill phase (4c)

The d1 "decode" metric is `output_tokens / round_wall`, and the round wall
includes prefilling 1024×128 prompt tokens = 32 eager 4096-token chunks
before the 64 graph decode steps. Round-0 walls (from logged tok/s):
native 9.32 s, fidg 10.14 s (**+0.83 s, −8.1%**), fi 9.95 s.

Eager prefill steps (43 megas each, per-step means):

| metric | native (17 steps) | fi_dg (24 steps) | fi_nvfp4 (22 steps) |
|---|---|---|---|
| step wall | 172.3 ms | 182.6 ms (**+10.3 ms**) | 189.9 ms |
| Σ mega | 54.7 ms | 65.3 ms (**+10.6 ms**) | 31.7 ms |
| non-mega time | 117.6 ms | 117.3 ms (equal) | 158.2 ms |
| mega mean | 1273 us | 1519 us (+19%) | 737 us |
| mega grid | 152 | 152 | 2 clusters/? (grid 2) |

Cross-device decomposition per layer (min duration across ranks = WORK,
max−min duration = absorbed WAIT, start spread = SKEW):

| | native | fi_dg | fi_nvfp4 |
|---|---|---|---|
| WORK (min-dur) mean | 911 us | **909 us (identical)** | 736 us |
| SKEW (start spread) med / p90 | 361 / 388 us | **610 / 649 us (1.7×)** | 693 / 736 us |
| WAIT (dur max−min) med | 361 us | 610 us | 692 us |

WAIT ≡ SKEW to the microsecond, and mega_mean − WORK ≡ SKEW
(native 1273−911=362; fidg 1519−909=610). I.e. **the collective mega kernel
absorbs whatever inter-rank launch skew exists at its entry as in-kernel
spin-wait, one-for-one — for the dg kernel too, not just the fi kernel**
(refines run 27, which measured native WAIT≈0 only because native's launch
skew is what sets the floor). fi_dg does identical work; its eager host path
simply delivers the launches 1.7× more skewed across ranks, and the whole
+10 ms/prefill-step is that skew × 43 layers (43 × 249 us ≈ 10.7 ms ≈
observed +10.3 ms; non-mega time is byte-equal).

Round-0 attribution for fi_dg vs native (+0.826 s total):

| component | Δ |
|---|---|
| (i) eager prefill skew-absorption, 32 chunks × +10.3 ms | **+0.33 s** |
| (ii) longer mega durations in-graph | +0.002 s (nil) |
| (iii) allreduce | −0.003 s (nil) |
| (iv) extra discrete kernels in-graph | **−0.042 s** (DataPreprocess beats `_prepare_megamoe_inputs`) |
| (v) inter-kernel idle in step | ~0 (1.20 vs 1.22 ms/step) |
| decode graph steps, 64 × −0.70 ms | **−0.045 s** |
| accounted | **+0.28 s** |
| unobservable residual (native blackout 165.2–168.3 s covers its prefill steps 18–32 + decode entry; fidg recording dead after prefill step 24; host/sampler segments) | +0.55 s |

The residual cannot be attributed from these traces (see §0); by phase
timing it sits in the prefill→decode transition and host-side segments, not
in graph replays (both tails replay at lockstep speed). Note the residual is
the same *kind* of time as (i): eager/host-path, not GPU graph work.

### Answers to 4a
In every observable decode window, 100% of steps are FULL-graph replays of
graph 11228 (native 27/27 recorded steps, fi 27/27; zero eager mega launches
inside decode windows). No out-of-graph decode steps were observed for any
backend, so the "eager decode steps" hypothesis is unsupported — and for
fi_dg untestable here (0 decode steps recorded). At fixed bs=1024 with a
1024 capture size, decode should never fall off the graph.

## 4. fi_nvfp4 vs fi_dg (4d)

Same picture in kind, different arithmetic:

- **In-graph decode**: fi_nvfp4 is the fastest step wall (32.95 ms, −4.6% vs
  native): its mega does −2.15 ms/step less work (11.03 vs 13.19 ms) but pays
  **+1.45 ms/step TopkReduce** (`topk_reduce…` 43× 33.8 us) plus +0.12 ms
  nvfp4 DataPreprocess; net −1.28 ms busy vs native.
- **Eager prefill**: differs in kind, not just TopkReduce: skew is higher
  still (693 vs fidg 610 vs native 361 us) AND non-mega time balloons
  (158 vs 117 ms/step — staging + TopkReduce + the fi host path), while mega
  WORK is far lower (736 us). fi wall 189.9 ms/step is worst despite the
  fastest kernel.
- Real decode tail confirms fi at 32.94 ms/step with TopkReduce present.

## 5. Bottom line

- The premise "fi_dg is slower at decode because of something in the decode
  step" is **falsified** for graph-replayed steps: with the identical mega
  kernel, fi_dg's in-graph decode step is 0.70 ms (2.0%) FASTER than native
  (fused DataPreprocess staging), mega/allreduce/idle identical, skew nil.
- fi_dg's measured e2e "decode" deficit is a **round-composition artifact**:
  the metric's denominator includes the eager prompt-prefill ramp, where
  fi_dg's host launch path generates 1.7× native's inter-rank skew and the
  collective dg mega absorbs it as in-kernel spin (+19% mega duration, work
  identical) → +10.3 ms per 4096-token chunk, ≈ +0.33 s/round of the +0.83 s
  round-0 gap; graph decode gives −0.05 s back; ~0.55 s residual lives in
  trace-blackout host segments of the same eager kind.
- Fix direction (consistent with runs 27/32): lean/lockstep fi host path in
  eager mode, or capture the prefill-sized batches (MAX_CAPTURE path) so the
  skew never reaches the kernel. The decode graphs need no fixing.
- **Re-profile with `--cuda-event-buffer-size` raised / shorter window /
  `cudaProfilerStart` gating**: these traces lost the timed rounds to CUPTI
  event caps and a round-1 NCCL watchdog crash (worth its own investigation:
  all three backends hung in round 1 under nsys node-mode).

## Appendix: region map (device-0 trace seconds)

| region | native | fidg | fi |
|---|---|---|---|
| lockstep 27× graph-11228 replays | 159.90–160.85 | 183.52–184.46 | 194.67–195.59 |
| remaining graph captures | 160.85–161.82 | ~184.5–185.3 | ~195.6–196.5 |
| round-0 eager prefill (recorded) | 161.9–165.2 (steps 1–17) | 185.4–190.2 (steps 0–23) | 196.6–201.0 (steps 0–21) |
| recording blackout | 165.2–168.3 | 172–175 (partial), none later | 201.5–204.0 |
| round-0 decode (recorded, sampled) | 168.63–169.57 (27 steps, 22/43 megas) | **none** | 204.09–204.99 (27 steps, 13/43 megas) |
| CUDA recording ends | 169.57 | 190.2 | 205.0 |
