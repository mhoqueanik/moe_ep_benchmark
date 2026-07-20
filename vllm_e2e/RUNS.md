# vLLM e2e run log — DeepSeek-V4-Flash, native dg vs flashinfer moe_ep

All vLLM 0.25.1 end-to-end runs (2026-07-15), one place. Analysis lives in
`FINDINGS.md`, reproduce steps in `RUNBOOK.md`. The flashinfer kernel-level
sweeps stay in the flashinfer repo's `cutedsl_megamoe/TUNING.md` (unchanged —
that document is kernel-microbench scope; this one is vLLM e2e scope).

Setup for every run: DeepSeek-V4-Flash (hidden 4096 / moe-inter 2048 /
256 experts / top-6 / 43 MoE layers, fp4 experts + ue8m0-32 scales),
4x GB200, TP=4 + EP=4, eager mode, kv fp8, block 256,
max_num_batched_tokens 4096. Backends: **native** = vLLM `deep_gemm_mega_moe`
unmodified; **fi_dg** = flashinfer moe_ep, `deep_gemm_mega` kernel;
**fi_nvfp4** = flashinfer moe_ep, `nvfp4_cutedsl` kernel (checkpoint fp4
dequantized to bf16 at load, requantized nvfp4).

## Where the GPU time goes (nsys, prefill 1024-tok prompts, per-backend 100%)

Buckets from `results/nsys_20260715_225236_*_cuda_gpu_kern_sum.csv` (GPU
kernel self-time; identical workload, 64 prompts x warmup+2 rounds).
`staging+small` = MoE staging/quant + small torch kernels
(elementwise/copy/reduce/index).

![GPU time attribution: native vs fi_dg vs fi_nvfp4](results/gpu_time_attribution.png)

(Regenerate with `python make_time_attribution_chart.py`; "other" folds
mhc/norm 1.5s + rope/kv ~1.3s + dense/sampler/misc.)

How to read it (details in FINDINGS.md "nsys attribution"):

- **MoE mega kernel**: per-launch avg native dg 1176 µs / fi_dg 1297 µs
  (same kernel — delta is inter-rank launch skew) / fi_nvfp4 cutedsl
  **1464 µs** → the cutedsl kernel is ~25% slower than dg at THIS geometry
  (4096-hidden/top-6; the microbench win was at 7168-hidden/top-8).
- **staging+small** grows 1.7s → 3.4s → 11.7s across native → fi_dg →
  fi_nvfp4: native stages with ONE fused kernel; fi_dg with
  `per_token_cast_to_fp8`+copies; fi_nvfp4 with ~98 small torch kernels per
  layer-step. cudaLaunchKernel counts: 100k / 330k / 946k.
- **allreduce** swings are skew absorption (whoever arrives last spins), not
  real collective cost — treat as slack, not work.
- fi_dg has LESS total GPU work than native yet lower e2e throughput →
  host-gap-bound (the staging launch storm stalls the GPU).

## Definitive throughput (offline repeat matrix, prefix caching OFF)

`bench_offline.py`: one engine boot per cell, 5 timed rounds after warmup,
round spread < 3%. fi wrapper with fast path + shared workspace.
JSONs: `results/offline_20260715_221930_*.json`.

| tok/s (median) | native | fi_dg | fi_nvfp4 (default knobs) | fi_nvfp4 (`KNOBS=auto`) |
|---|---|---|---|---|
| prefill 1024/1 | 31777 | 28350 (0.89x) | 25843 (0.81x) | 22348 (0.70x) |
| decode 128/256 | 1681 | 1421 (0.85x) | 1318 (0.78x) | 1154 (0.69x) |

**knobs=auto paradox:** the autotuner's winner (ikr + mma 256x256 +
flag_batch 8 + standalone_warps) measured **710 µs** in its own harness vs
1464 µs default / 1176 µs dg — a 2x kernel-level win — yet e2e got ~13%
WORSE on both workloads. The tuner's metric (synchronized collective
launches, median of max-across-ranks) does not transfer to the pipelined
engine; suspects: ikr's cross-rank atomics under real skew, and a possible
interaction with the shared-workspace patch binding buffers before the
knob switch (auto+sharedws correctness smoke: run 15). Also: auto re-tunes
per encountered shape (24 cute.compiles each, 576+ candidate timings in the
decode engine) — unusable in-engine; production should pin an
e2e-validated knob dict via `FI_MOE_EP_KNOBS='{...}'`.

## Correctness (greedy 64 tok x 8 prompts, per-token logprobs)

| comparison | exact | mean \|dlogprob\| |
|---|---|---|
| native vs native rerun | 8/8 | 0.0000 |
| native vs fi_dg | 3/8 | 0.01-0.06 |
| fi_dg vs fi_dg rerun | 2/8 | 0.01-0.08 (nondeterministic — open) |
| native vs fi_dg (fast-path+sharedws wrapper) | 3/8 | ~0.04 (unchanged) |
| native vs fi_nvfp4 | 1/8 | 0.02-0.20 (double-quant, expected) |

## Chronological run log

| # | run | node/job | config | result | artifact |
|---|---|---|---|---|---|
| 1 | smoke native (3 attempts) | 2387842 | FI=0 | dep-stack fixes (quack/dsl/tvm-ffi/tilelang), then PASS | logs/smoke_native.log |
| 2 | smoke fi_dg (3 attempts) | 2387842 | fi deep_gemm_mega | LOCAL_RANK device bug found+fixed, then PASS | logs/smoke_fi_dg.log |
| 3 | native rerun control | 2387842 | FI=0 | 8/8 bit-exact | results/smoke_native2.json |
| 4 | fi_dg rerun control | 2387842 | fi dg | 2/8 vs itself — nondeterminism found | results/smoke_fi_dg2.json |
| 5 | vllm-bench sweep r1 | 2387842 | 3 workloads x native,fi_dg | superseded (restart variance; output-len quirk) | results/bench_20260715_205636.csv |
| 6 | vllm-bench repeats r2-r4 | both | prefill/decode/mixed | prefill ±35% variance discovered | results/bench_20260715_21*.csv |
| 7 | smoke fi_nvfp4 (2 attempts) | 2388721 | fi nvfp4_cutedsl | weight-pack retention OOM found+fixed, then PASS | logs/smoke_fi_nvfp4.log |
| 8 | vllm-bench fi_nvfp4 | 2388721 | 3 workloads | superseded (one run raced a duplicate engine) | results/bench_20260715_2149*.csv |
| 9 | offline matrix v1 | 2388721 | fast-path wrapper | DISCARDED — prefix-cache hits faked 91k tok/s | logs/offline_matrix.log |
| 10 | offline matrix v2 | 2388721 | + prefix caching off, shared workspace | DEFINITIVE table above | results/offline_20260715_221930_*.json |
| 11 | smoke fi_dg fast-path | 2389111 | optimized wrapper | correctness unchanged | results/smoke_fi_dg_fast.json |
| 12 | nsys profile matrix | 2389111 | 3 backends, prefill | attribution chart above | results/nsys_20260715_225236_* |
| 13 | offline fi_nvfp4 knobs=auto | 2389111 | FI_MOE_EP_KNOBS=auto | kernel 710us win, e2e 13% LOSS (see paradox note) | results/offline_20260715_231848_fi_nvfp4_*.json |
| 14 | 4.5.2 DSL sensitivity (microbench, same day) | — | kernel-level, TUNING.md | 4.6.1 = perf floor (34-54% slower on 4.5.2) | flashinfer TUNING.md |
| 15 | smoke fi_nvfp4 knobs=auto | 2389111 | auto+sharedws correctness | SANE: \|dlp\| 0.03-0.15 vs native (same band as default nvfp4); auto e2e loss is a real perf effect, not corruption | results/smoke_fi_nvfp4_auto.json |
| 16 | offline matrix + nsys, upstream fixes @888383f5 | 2393880 | fi branch: fused DataPreprocess staging, workspace pool, weight release, knob cache (heuristic fallback, no DSV4 entry yet) | fi_nvfp4 prefill 27001 (+4.7% vs run 12's 25.8k, 0.85x native) decode 1348 (+2.1%, 0.82x, now >= fi_dg); fi_dg 27446/1329 (flat, decode spread 1044-1459); native control stable 31653/1653. nsys: fi_nvfp4 launches 946k->784k (-17%), GPU busy 51.5->48.5s, fused stage kernel n=9804 @6us replaces per-step torch soup, mega avg 1464->1333us (skew absorption down; knobs unchanged). fi_dg trace byte-identical counts (untouched, as expected) | results/offline_20260716_155158_*.json, results/nsys_20260716_162044_* |

| 17 | DSV4 offline tune + tuned re-bench | 2395263 | python -m flashinfer.moe_ep.tune (nvfp4, 4096/2048/256/top6, EP4, bucket 4096, 12 deterministic candidates) -> knob cache results/knob_cache_dsv4.json; bench with FLASHINFER_MOE_EP_KNOB_CACHE set | Tune winner 588.7us harness (256x128 tile, fb4, standalone_warps, NO ikr) vs 1464us heuristic. E2E TRANSFERRED (unlike run-13 ikr paradox): fi_nvfp4 prefill 28204 (+4.5% vs run 16, 0.88x native 31887), decode 1415 (+5.0%, 0.82x native 1717), spreads ultra-tight (28.1-28.3k / 1411-1421 — deterministic winner). nsys: GPU busy 48.5->46.8s, mega avg 1333->1296us in-engine (skew-wait dominated; harness delta shows up as e2e + busy-time win, not kernel-avg), allreduce avg 771->730us | results/offline_20260716_19*/20*_*.json, results/nsys_20260716_201058_* |

| 18 | DP4/TP1+EP4 matrix (no TP allreduce), tuned fp4 vs native | 2397027 | bench_offline_dp.py (multi-proc offline DP: one LLM/rank, VLLM_DP_* env, no CUDA_VISIBLE_DEVICES, fresh master port per run) | prefill: native 41830 (+32% vs its TP4), fi_nvfp4 35392 (+25%, ratio 0.85x UNCHANGED from TP4 -> residual prefill gap is MoE-path glue, not allreduce skew). decode: native COLLAPSES to 971 (from 1653 TP4) while fi_nvfp4 holds 1337 -> **fp4 1.38x native decode under DP-attention topology**. nsys (prefill): mega kernel TOTALS near-equal (fi 32.6s vs native 31.7s — kernel-level parity in-engine; per-launch avgs 1897 vs 1585us are both skew-inflated under DP, fi more so), GPU busy 49.7 vs 39.3s, launches 776k vs 79k -> the whole prefill gap is the wrapper/glue launch soup + host gaps | results/dp4_20260716_234202_*.json, results/nsys_dp4_20260716_234202_* |

| 19 | nondeterminism root cause CLOSED | 2396002/2397066 | layer probes (flashinfer bda8af5b) + smoke rerun pairs + FI_MOE_EP_SHAPE_LOG schedule diff | (1) fi_dg layer bit-exact under repeats, ~100ms rotating rank skew (dg combine is arrival-order FIXED), and shifted input addresses. (2) 07-15 premise inverted on fresh node: fi_dg 8/8 self-exact, NATIVE 3/8 — nondeterminism is engine-level, backend-independent. (3) MECHANISM: on a diverging native pair, per-MoE-call batch-shape schedules differ on all ranks from step 5 (1-token vs 8-token batch) — vLLM scheduler batch composition is timing-dependent across runs; shapes → split/rounding deltas → greedy flips. Determinism claims need N rerun pairs + shape-log control | results/shape_det_*, results/shape_det2_*, flashinfer tests/moe_ep/test_moe_ep_deep_gemm_skew_determinism.py |

| 20 | patch simplification + CUDA graphs + launch attribution CORRECTION | 2399251 | fi_utils.py: _SHARED_WORKSPACE + _weights=None workarounds removed (upstream pool/release); smoke eager PASS; **ENFORCE_EAGER=0 smoke PASS for fi_nvfp4** (coherent generations — upstream warmup contract + capture guards suffice, no wrapper changes). Glue nsys trio (fi-fused/fi-torch/native, 16 prompts): the big launch-count deltas (88k/44k/22k elementwise etc.) are IDENTICAL across 16- and 64-prompt runs → they are MODEL-LOAD weight-preprocess kernels (bf16 dequant→requant, 43 layers x per-expert torch quant), NOT hot path. Hot-path staging is clean (fused cell has 0 extra _pack_fp4/step; torch cell +1/step). CORRECTION to runs 16/18 narratives: steady-state fi GPU work ≈ native; the residual 0.85x prefill gap is host-side per-layer-call overhead — exactly what EAGER=0 absorbs | logs/step4*, results/nsys_glue_095834_*, results/step4_smoke_graphs.json |

| 21 | ENFORCE_EAGER=0 bench (CUDA graphs, TP4+EP4, tuned cache) | 2399251 | first-ever graphs-mode numbers; both backends capture cleanly | **decode transforms: native 10763 (6.5x its eager 1653), fi_nvfp4 10101 (7.1x its eager 1415) -> fi = 0.94x native** — eager decode was launch-bound for every backend and all prior decode comparisons were eager artifacts. Prefill: native 31442 (~eager), fi 26251 (below its eager 28204 — graph-mode padding interaction, prefill best config remains eager for fi). Best-known production config: graphs decode + (for now) eager prefill; fi fp4 overall 0.94x/0.89x of native at decode/prefill in their best modes | results/graphs_20260717_102219_*.json |

| 22 | ikr A/B + launch cuts + steady-state gap attribution | 2399251 | flashinfer ea26808b (zero-copy output view, memoized tail mask, caller-owned ikr precedence); wrapper FI_MOE_EP_IKR | GAP ATTRIBUTION (windowed nsys, 95% GPU busy under graphs): decode gap = explicit TopkReduce +32us/step (fp4 mega kernel itself BEATS dg at decode, 124 vs 136us); prefill gap = mega kernel +14% at chunked-prefill shapes; host exonerated. ikr A/B: LOSES both — decode-graphs 4328 (vs 10101 non-ikr, known small-token ikr penalty x untuned combo), prefill 25708 (vs 28204); smoke sane (\|dlp\| 0.04-0.06). Wrapper default = explicit reduce; FI_MOE_EP_IKR=1 opt-in. Decode lever now = ikr-aware retune at decode buckets or kernel-side small-batch ikr | results/nsys_gap_103821_*, results/ikr_20260717_110801_*, analyze_gap_nsys.py |

| 23 | decode-targeted retune + graph-safe memo (fi bd5e4dfd/c41c5e3e) | 2399251 | tune --live-tokens 256 --allow-nondeterministic (24 cand) -> knob_cache_dsv4_decode.json; two memo bugs found+fixed by the bench itself (multi-size graph replay under-fill; view-slicing on capture-touched buffers) — regression tests added | Sweep: NON-ikr wins even at decode shapes (256x128/fb8/epi_warps, 290us) — ikr small-token penalty is inherent; run-22 ikr e2e numbers were memo-bug contaminated but the conclusion stands. FINAL decode-graphs: native 10763 / fi pretuned-cache 10207 (0.95x, confirms launch cuts safe) / **fi decode-tuned cache 10413 (0.97x, +2%)**. Per-role knob caches = real deployment pattern (prefill-tuned vs decode-tuned via FLASHINFER_MOE_EP_KNOB_CACHE per engine role). Remaining 3% decode gap = TopkReduce; needs kernel-side small-batch in-flight reduce (trtllm import todo) | results/final_120250_*.json, results/knob_cache_dsv4_decode.json |

| 24 | quantized combine wire (FI_MOE_EP_COMBINE=nvfp4) A/B | 2399251 | wrapper combine_dtype plumb + wire-valid tune (4 cand, winner 256x256/fb8/reuse_dispatch_warps) -> knob_cache_dsv4_wire_nvfp4.json | Wire does NOT transfer at DSV4: prefill-eager 25827 vs 28204 bf16 (-8%; wire-forced reuse_dispatch_warps costs more than the 4x combine-traffic saving at this geometry — the microbench win was at 7168/top-8), decode-graphs 10437 ~= dectuned bf16 10413 (tie, worse accuracy: smoke \|dlp\| up to 0.16). VERDICT: bf16 wire stays production at DSV4; wires worth re-checking only at higher-combine-traffic geometries | results/wire_131339_*.json |

| 25 | per-layer signal analysis (cold-run stats + nsys phases) | 2400175 | FI_MOE_EP_LOAD_STATS wrapper hook (per-layer expert-load skew JSON) + launch-index-mod-43 phase analysis of existing gap traces | Per-layer mega-kernel time spread 35% prefill / 25% decode, rank-consistent slow layers. Expert skew severe everywhere (max/mean load 14-28x by layer; layers 39/17/33/4 worst). Aligned correlation time-vs-skew: Spearman ~0.39 both workloads (same best rotation offset 20 on independent traces) — skew is a real but partial driver (~15% of variance); the rest is dynamic per-step skew. RECOMMENDATION: EPLB support in the fi wrapper first (dynamic, no memory cost, parity requirement — currently stubbed while native supports it; note all runs 1-24 had EPLB off both sides, comparison fair); K-group per-layer knobs second (tooling ready; costs one compile + one multi-GB workspace per group) | results/layer_load_stats.rank*.json, analyze via RUNS.md run-25 scripts |

| 26 | layer-wide tuning strategy: skew-aware sweeps (fi 546b6078) | 2400223 | tune --skew (power-law routing at measured ratios) + --sweep schedule (load_balance x group_hint over cache-winner base); 4 sweeps: decode/prefill @ skew18, decode @ skew14 vs skew28 (group discrimination) | **VERDICT: per-layer knob grouping NOT justified at DSV4.** static-vs-atomic_counter top candidates differ 0.4-2.5us (<1%, noise ties — apparent winner flips across skew levels are statistical); top-candidate time barely moves 14x->28x skew (296->302us). The schedule axes are skew-INSENSITIVE: per-launch time under skew is bounded by the straggler expert's actual work, which no scheduling knob removes. Existing per-role global configs CONFIRMED optimal under production-like routing (robustness result). The 25-35% per-layer spread is load imbalance itself -> EPLB (dynamic rebalancing) or kernel-side work stealing, not knobs | logs/skew_tune_*.log, results/knob_cache_*skew*.json |

| 27 | per-layer work/wait/skew decomposition (prefill eager nsys) | 2400223 | analyze_layer_nsys.py: k-th mega launch matched across ranks; min-duration=WORK, excess=absorbed WAIT, start spread=SKEW | STRUCTURAL FINDING: native dg kernel durations are arrival-INDEPENDENT (WAIT ~0.3us on every launch, all ranks equal) — its 362us skew is absorbed by the TP allreduce it runs anyway (overlapped sync, zero marginal cost). fi nvfp4 kernel absorbs the FULL arrival skew in-kernel (WAIT median 850us ~= SKEW 864us, uniform across all 43 layers — a flat per-layer tax, not layer-concentrated), and fi regenerates 2.4x native's skew between launches (864 vs 362us). Per-layer WORK varies 467-1038us with routing skew (worst phase-layers ~10/26/40 match the load-stats most-skewed list). fi kernel-work superiority stands where skew is controlled: harness 588 vs ~1176us, decode-graphs 124 vs 136us. PREFILL LOSS = skew regeneration (fi host path variance) x in-kernel absorption (no overlap). Fixes: lean wrapper host path; kernel-side async/overlapped combine (dg-style); graphs (proven at decode). fi_dg control cell lost to job timeout (rep exists, unexported) | results/nsys_layer_145829_*, analyze_layer_nsys.py |

| 28 | host-path pinpoint + cached launch thunk (fi 58761040) | 2400561/2400641 | host_overhead_probe.py (enqueue-only per-component us) + backend thunk via shim make_launch_thunk | PINPOINT: per-layer-call host cost ~100us = ~2us vLLM wrapper (innocent) + 16us/DSL-launch (torch-op parity, exonerated) + ~70us loop-invariant Python in nvfp4_mega_moe()/frontend.run() (re-validation, clamp re-resolve, 12-field inputs rebuild, launch-cache re-key). NOT in MoEEpMegaLayer (bypassed). FIX: backend caches make_launch_thunk per (workspace, weights, compiled-session, STREAM) — stream in key is load-bearing (thunk kwargs bind stream at build; capture must build a capture-stream thunk or the kernel launch ESCAPES the graph — caught by the multi-size interleave test, replays reusing stale output). compute() host cost 6.3x->3.0x raw-launch. E2E: prefill 28204->28593 (+1.4%, 0.90x native), decode-graphs 10413->10491 (+0.7%, 0.975x). fp4 cumulative from 07-15 baseline: prefill +10.8% | results/thunk_*.json, tmp host_overhead_probe.py |

| 29 | prefill-sized CUDA graphs (max_cudagraph_capture_size=4096) | 2400641 | MAX_CAPTURE knob in bench_offline/smoke_infer -> compilation_config | **HEADLINE: fi fp4 prefill 49402 tok/s (+73% over its eager 28593) vs native 43395 (+37%) = fi 1.14x NATIVE** — prefill steps captured+replayed in lockstep kill the inter-rank wait for both backends, and fp4's 1.8x kernel-work advantage finally surfaces e2e, right in the predicted band. Rounds ultra-tight (49.37-49.59k). The run-27 waiting problem is fixed by DEPLOYMENT CONFIG (one vLLM knob), no kernel change needed for prefill; kernel-side drop-and-go remains as further upside for eager/unaligned regimes. Production config now: graphs with 4096 capture for prefill (1.14x), graphs decode (0.975x). Sanity smoke under 4096-capture PASSED (coherent, |dlp| 0.03-0.20 = established fp4-vs-native band, 1/8 exact) | results/cap4k_*.json |

| 30 | cap4k steady-state nsys (what limits >1.14x) | 2400771/2400863 | analyze_cap4k_window.py (time-window = one timed round from bench JSON; launch-count windows break on capture-warmup noise) | Native: GPU 95.5% busy, mega 871.6us (43% of round), allreduce COLLAPSED 675->102us (lockstep killed the skew it used to absorb — run-27 story confirmed from native side). fi: mega 668.6us at engine chunk shapes = wait GONE (654us work estimate confirmed), **1.30x faster than native mega at blended shapes** (1.8x at clean 4096); allreduce equal. CAVEAT: fi trace under-expands DSL kernels inside full-step captured graphs (busy% and some kernels missing) — per-kernel avgs sound, utilization split not. What limits more speedup: mega still the largest single item (~50% of traced busy) -> further kernel-work gains still pay; TopkReduce fusion; attention band is shared with native. Decode levers remain concurrency/MTP + kernel reduce fusion | results/nsys_cap4k_*, analyze_cap4k_window.py |

| 31 | two-checkpoint policy live: NVFP4 ckpt via prequant path + GSM8K gate | 2407737 | MoEWeightPack union (flashinfer, tests+oracles+microbench green, no perf delta); fi_utils prequant glue: NVFP4 ckpt (nvidia_deepseek-v4-flash-nvfp4, cast from same mxfp4 base) loads via PrequantizedMoEWeights — e4m3-per-16 scales verbatim, weight_scale_2 -> per-expert fc1/fc2_alpha staged per-forward via MoEEpTensors (shared workspace preserved), gate/up fold e4m3-exact-guarded, input_scale unused (dynamic act quant) | Smoke PASS first try (coherent). **GSM8K gate: native (mx ckpt) 0.9650, fi_nvfp4 (nvfp4 ckpt) 0.9750** (200q, 0 invalid) — double-quant caveat RESOLVED; fi now runs single-quant checkpoint numerics dg cannot consume. Perf cells reproduce run 29/23 within noise: prefill-cap4k native 43124 / fi 49304 (**1.14x**), decode-graphs native 10777 / fi 10381 (0.96x, TopkReduce gap unchanged — see todo_kernel_tail_topk_reduce.md). Prequant load also removes the boot-time dequant->requant soup | results/twockpt_*.json, results/gsm8k_*.json, results/smoke_fi_nvfp4_prequant.json, results/union_refactor_20260719_fi_mega.csv |

| 32 | fi_dg control row (first fi_dg cells since run 16) + fast-path output=None fix | 2407737/2408070 | BUG FOUND: the wrapper's zero-copy fast path (`compute(output=None)`, ea26808b-era) is a cutedsl-backend contract — deep_gemm's backend feeds `output` to the pybind arg0 → TypeError at warmup, then NCCL-destroy teardown deadlock (presents as a startup hang; the run-31-era grep-filtered logs hid it). fi_dg was silently broken in-wrapper since the zero-copy change. FIX: fast ctx carries a zero_copy flag (kernel_name != deep_gemm_mega); dg path allocates a bf16 output per call | fi_dg (mx ckpt): GSM8K **0.9650 (193/200 — identical count to native)**; prefill-cap4k **44546 (1.03x native!)**; decode-graphs 10722 (0.995x); eager prefill 29019 (best fi_dg eager ever). VERDICT: under lockstep graphs the fi integration layer costs NOTHING (slightly ahead at prefill — fused DataPreprocess staging beats native's) — the run-16 era 11-15% fi_dg deficit was host/eager overhead, now fully absorbed. Decomposition: fi_nvfp4 prefill 1.14x = fi layer 1.03x x kernel ~1.11x; fi_nvfp4 decode 0.96x is pure kernel+TopkReduce (fi layer 0.995x exonerated) | results/twockpt_fi_dg_*.json, results/gsm8k_fi_dg.json |

| 33 | workload-matched regimes (8k prefill chunks, 1k decode concurrency) | 2408070 | bench_offline grew --max-num-seqs / --gpu-memory-utilization / CAPTURE_SIZES (sparse capture list — the dense every-16 list at MAX_CAPTURE=8192 estimated 310 GiB of graph pools and drove fi's KV negative; 6 sparse sizes fix both backends, same config each side). Two-checkpoint policy; fi on 4096-bucket knobs (no 8192 tune yet — headroom) | **decode @1024-seq concurrency FLIPS: fi 18747 vs native 16601 output tok/s = 1.13x** (was 0.96x at 128-seq; fixed per-step costs incl. TopkReduce amortize 8x — the decode "gap" was a batch-size artifact). **prefill @8k chunks: fi 53839 vs native 46027 = 1.17x** (both gain over 4k chunks; fi gains more, per the microbench batch curve). fi rounds ultra-tight both cells; native decode-1k spread 23.9-27.6k. Deployment guidance: serve DSV4+fi with the largest chunk/concurrency memory allows | results/regime_*_sparse.json, results/regime_*_decode1k.json |

| 34 | retuned regimes + 3-backend table + decode-1k CORRECTION | 2408070 | fresh tunes: 8192 bucket (963.5us winner, 256x128/fb4/standalone_warps -> knob_cache_dsv4_8k.json) and decode-1k (350.5us @1024 live, reuse_dispatch_warps -> knob_cache_dsv4_dec1k.json); reran all 3 backends both regimes same-session | prefill-8k: native 45701 / fi_dg 47534 (1.04x) / **fi_nvfp4 53962 (1.18x)** — 8192-bucket tune worth +0.2% over borrowed 4096 knobs. decode-1k output tok/s: fi_nvfp4 **18765, <1% spread across 6 rounds/2 sessions**; fi_dg 18094 stable; native 15924-19435 with WITHIN-SESSION round-over-round decline in both sessions. **CORRECTION to run 33: the 1.13x decode claim used native's weak session; vs native's best fi is 0.98x. Honest claim: decode-1k = parity (0.96-1.13x band) with fi far more stable** — reproducibility is the differentiator; native's decode-1k decline pattern unexplained (KV pool? scheduler?) and worth its own investigation before any decode-1k number is quoted as native's true median | results/retuned_*.json, results/knob_cache_dsv4_8k.json, results/knob_cache_dsv4_dec1k.json |

| 35 | nsys pass at run-33 regimes: band refresh NOT publishable, TopkReduce confirmation extracted | 2408656 | 4 traced cells (native/fi x prefill-8k/decode-1k); decode cells timed out mid-round-1 under nsys overhead (~15M events) | Windowed band derivation at these regimes FAILED sanity (17% busy where run 30 measured 95%; AR 60ms/step vs known ~4ms): captured-graph kernel under-expansion (half of expected mega launches absent from CUPTI table) + nsys host overhead re-inflating skew-absorbing collectives. gpu_time_attribution.png therefore keeps the run-27/30 bands (provenance stated in-figure) with annotations updated to run-31/33 e2e numbers. What IS trustworthy: per-launch self-times — **TopkReduce 28us @128 tok (~18% of MoE-path GPU time) / 34.7us @1024 (~9%) / 68.6us @8192 (~7%), sub-linear = launch-bound; fused staging 3-5us (free)**. These are the confirmation numbers for the kernel-tail-reduce HANDOFF to the kernel team (flashinfer todo_kernel_tail_topk_reduce.md has the full implementation package). A future honest band refresh needs the run-30 methodology (longer un-killed rounds + graph-expansion workarounds) | results/nsysband_*, this analysis in-line |

| 36 | decode-win attempt: 2048-seq + endurance + live-2048 tune — VERDICT: parity, win needs kernel work | 2408878 | d2k (2048 prompts, max_num_seqs 2048, capture 2048, 6 rounds), d1k endurance (8 rounds), then tune --live-tokens 2048 (winner standalone_warps 411us harness) + rerun | decode-2k output tok/s: native 20596 / fi 20104 (**0.976x**), both <1% spread. d1k endurance: native 19024 STABLE across 8 rounds — **the run-33/34 native decline+weak session did NOT reproduce; run-33's 1.13x is conclusively a native-outlier artifact; true decode = parity 0.98x at both 1k and 2k concurrency** (fi's differentiator is stability, not a lead). Microbench 1.35x @2048 does not transfer to decode e2e (attention/misc scale with seqs too; TopkReduce grows with tokens; both backends gain from concurrency, native slightly more). live-2048 tuned rerun REGRESSED (19751 vs 20104 on dec1k knobs — tuner-harness/pipelined-engine transfer gap again, cf. run 13); dec1k cache remains best for 2k. **Decode-win path: kernel-tail topk reduce (kernel team) ~+3-4% e2e -> ~1.01-1.02x; or MTP (parked).** Production claims should be: prefill 1.18x, decode parity with superior stability | results/d2k_*.json, d1k_endur_*.json, knob_cache_dsv4_dec2k.json |

| 37 | decode DEFICIT RESOLVED: capture the prefill ramp -> fi wins decode | 2409803 | Root cause chain (traces + agent analysis, results/fidg_tax_analysis.md): decode cells' MAX_CAPTURE=1024 left the round's 32x 4096-token prefill chunks EAGER; in eager, fi's host path generates 1.7x native's inter-rank launch skew (610 vs 361 us/layer) which the collective mega kernel absorbs as spin (+19% duration, identical work); the decode metric divides by round wall -> phantom "decode deficit". In-graph decode steps were ALWAYS fi-favored (native 34.54 / fi_dg 33.84 / fi_nvfp4 32.95 ms wall — same-kernel fi_dg -2%, staging -660us). FIX = config: decode cells now capture up to 4096 (sparse list), all backends | decode-1k output tok/s (5 rounds, <1% spread): **native 21049 / fi_dg 21540 (1.023x) / fi_nvfp4 22452 (1.067x)** — fi_dg strictly better as expected from identical kernel; fi_nvfp4 = the decode win. Production config: capture ALL recurring step shapes incl. chunk sizes. Residual hardening TODO (uncapturable shapes): port the launch-thunk + persistent output buffer to the deep_gemm backend; fold padding mask into staging. Trace-infra lesson: nsys node-mode hits CUPTI event caps + NCCL watchdog on ALL backends — future traces need cudaProfilerStart-gated windows | results/dcap_*.json, results/fidg_tax_analysis.md |

| 38 | knob A/B under full capture: tuner transfer RESTORED, decode best now 1.074x | 2409803 | rerun the run-36-rejected standalone_warps winner (knob_cache_dsv4_dec2k.json) at decode-1k full-capture vs the dec1k cache | **fi 22614 output tok/s (1.074x native 21049)** vs 22452 on dec1k knobs (+0.7%) — the harness winner wins e2e once the eager ramp is captured. CONCLUSION: run-36's "tuner does not transfer at decode" was eager-ramp contamination; with production capture coverage the offline tune workflow is validated for BOTH phases (tune-and-ship, no e2e revalidation needed). Production decode cache = knob_cache_dsv4_dec2k.json. Headline table now: prefill 1.18x / decode 1.07x / GSM8K 0.975 | results/dcap_fi_dec2kknobs.json |

| 39 | gpu_time_attribution REFRESHED at production regimes | 2410136 (chart gen only) | bands derived from EXISTING node-mode traces (no new tracing needed): prefill-8k from nsysnode_{native,fi}_p8 anchored windows; decode-1k from the bs-1024 graph-template (graphId 11228) kernel composition per replay. Gated-capture attempt abandoned: cudaProfilerStart in the bench PARENT gates nothing — vllm kernels run in WORKER processes (12MB trace, zero kernels); in-worker gating would need vllm plumbing | Sanity: prefill band wall ratio 174.7/147.6 = 1.18x matches e2e 1.18x exactly; decode busy 35.4 vs 34.4ms consistent with 1.07x; shared bands byte-equal across backends both panels. fi leads BOTH phases on totals; only giveback = topk-reduce band (1.6ms/step decode — the kernel-team handoff item, visible in-figure). Chart: results/gpu_time_attribution.png (suptitle carries provenance). Tracing recipe lessons consolidated: node-mode + name-bucketing + 43-mega anchor; gate windows are broken for multiproc vllm; prefer graph-template filtering for decode composition | results/gpu_time_attribution.png, make_time_attribution_chart.py |

## Open items / next-run plan

1. **[next run] DP4/TP1+EP topology** (old `run_deepseek_v4_flash.sh` serve
   config): eliminates the per-layer TP allreduce entirely (21-36% of GPU
   time in the nsys profile, plus its skew noise) — attention goes
   data-parallel, cross-rank traffic collapses into the mega kernel's own
   dispatch/combine. Run all backends under `vllm serve` + `vllm bench
   serve` for TTFT/TPOT too.
2. ~~Wire the cutedsl fused staging+quant kernel~~ DONE upstream
   (75221c88 nvfp4/mxfp8, e78a9ef0 deep_gemm). Run-20 correction: the
   remaining launch-count gap vs native is LOAD-TIME weight preprocessing,
   not hot path; residual prefill gap is host per-layer-call overhead →
   EAGER=0 (now working) is the lever.
3a. EPLB support in the fi wrapper (run-25 recommendation): mirror native's
   eplb_map_to_physical_and_record + expert-weight export hooks; attacks the
   measured 25-35% per-layer straggler spread dynamically.
3. cutedsl kernel retune at (4096, 2048, 256, top6): knobs=auto (run 13)
   restored the kernel win (710us vs dg 1176us) but LOST e2e at prefill —
   tuner-harness vs pipelined-e2e mismatch; needs kernel-repo sweep at this
   geometry + e2e-validated profiles, and a correctness smoke for the
   auto+shared-workspace combination.
4. ~~Fuse fi_dg staging~~ DONE (flashinfer e78a9ef0: byte-identical recipe via DataPreprocess mxfp8_e4m3 + 16B-alignment fallback).
5. ~~fi_dg run-to-run nondeterminism~~ CLOSED (run 19): engine-level batch-formation timing, backend-independent; not an fi bug.
6. ~~CUDA-graph compatibility~~ DONE: layer path upstream (964d3017,
   single-rank + 2-rank lockstep tests green) AND in-engine — run-20
   ENFORCE_EAGER=0 fi_nvfp4 smoke passes with no wrapper warmup changes
   (vLLM's dummy-run warmup compiles everything before capture).
7. Simplify patch_0251/fi_utils.py against upstream 888383f5: drop
   `_SHARED_WORKSPACE` (workspace pool upstream), drop `_weights = None`
   (released upstream); knob cache can pin DSV4 winners once item 3's
   e2e-validated sweep exists.
8. **[2026-07-19] Two-checkpoint policy + GSM8K oracle** (plumbing landed,
   first run pending): fi_nvfp4 runs the NVFP4 cast
   (`nvidia_deepseek-v4-flash-nvfp4/hf/hf-48bfe38_orig`, same base weights —
   see its cast_mxfp4_to_nvfp4.log) via the backend's prequantized-weights
   branch; native/fi_dg keep the mx original. `bench_offline.resolve_model`
   now picks the checkpoint per backend (override: --model / MODEL /
   MODEL_NVFP4) and records it in every result JSON. `eval_gsm8k.py` is the
   cross-checkpoint fairness gate — run it on BOTH backends (expect ~0.95;
   `--min-acc` to hard-gate) before quoting any perf row. Blocked on the
   wrapper weight-loading glue (flashinfer
   `todo_nvfp4_prequant_checkpoint.md`: MoEWeightPack prequant + global-scale
   -> fc1/fc2_alpha/input_norm_const mapping; gate/up scale folds must be
   e4m3-exact or fail loudly — no lossy rescale, apples-to-apples rule).
   Expected effect: boot time + accuracy positioning, NOT step time (per-step
   staging is the same fused bf16->nvfp4 kernel either way).
