# Expected results — 1x4 SM103 (GB300), FlashInfer `moe_ep`

Every number here was measured on one **4x GB300** node (Blackwell Ultra,
sm_103) from this branch's own scripts, on 2026-08-03. This is the 1x4 port of
[expected_results.md](expected_results.md) — same cells, same methodology,
half the world size. **Do not compare absolutes across the two files**: EP4
holds 64 of Flash's 256 experts per rank where EP8 holds 32, so
tokens-per-expert doubles at a given tokens/rank and both the kernel crossover
and the engine throughput move. The fi-vs-native ratios within one file are
the claim.

**Configuration.** vLLM 0.25.1 (wheel + `vllm_e2e/patch_0251/`), flashinfer
branch `4_5_2-perf-fix` @ `adfa4749`, nvidia-cutlass-dsl **4.5.2** (vLLM
0.25.1's own pin, asserted by the same guards as the 1x8 branch — 4.5.2
compiles and runs cleanly on sm_103). Image
`flashinfer-ep-pt2605-mega_moe_ep.sqsh` (aarch64, built 2026-07-23 on a GB300
node). Checkpoints: the cluster mirror copies of the pinned revisions —
`deepseek-ai_deepseek-v4-flash/hf/hf-6e76323_orig` (mx) and
`nvidia_deepseek-v4-flash-nvfp4/hf/hf-48bfe38_orig` (NVFP4) — verified against
the §1.3 schema checks before any run.

**EP4 throughout, but the two levels reach it differently**, exactly as on
the 1x8 branch: the e2e sweep (§1) runs **TP4 + EP4, DP1**; the kernel
microbenchmark (§2) runs **DP4 + EP4, TP1**. The backend names, checkpoint
policy and what a ratio means are unchanged — see the header of
[expected_results.md](expected_results.md).

V4-Pro is **not** measured end-to-end at 1x4 (its geometry is in the §2
kernel tables only), and serving mode is not ported.

---

## 1. vLLM e2e — DeepSeek-V4-Flash, EP4/TP4

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_ep4.sh` (~1 h 5 min). Job
**2567468**, node theia0062. fi_cutedsl runs the NVFP4 checkpoint with
`results/knob_cache_ep4.json` (tuned at world 4 on GB300, job 2567354).

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 47412 | 49145 (1.037x) | 56432 (**1.190x**) |
| decode-1k | 32313 | 32936 (1.019x) | 34396 (**1.064x**) |
| 100K ISL / 1K | 35575 | 36389 (1.023x) | 38632 (**1.086x**) |
| 32K ISL / 32 | 43615 | 44846 (1.028x) | 50360 (**1.155x**) |

Latency on the interactivity cells (`REQUIRE_LATENCY=1`), fi_cutedsl vs
native: TTFT 35.2s vs 40.4s at 100K, 10.6s vs 12.4s at 32K; ITL p50 44.7ms vs
47.1ms and 159.0ms vs 184.9ms.

The pattern matches the 1x8 B200 sweep cell for cell: fi_dg at wrapper parity
(1.02-1.04x) everywhere, fi_cutedsl's win largest on the prefill-heavy cells
and smallest on decode-1k — decode runs below the kernel crossover (§2). If
fi_dg is far from parity, read expected_results.md §5.1 before believing
anything.

Raw JSONs: `vllm_e2e/results/sweep_ep4_<cell>_<backend>.json` (twelve files).

## 2. Kernel microbenchmark — no vLLM, no checkpoints

`bash model_shapes/submit_jobs.sh` (defaults to GPUS=4 / `results_ep4/` on
this branch). `e2e_pipelined` p50 microseconds per rank, CuteDSL variants'
speedup vs `deep_gemm_mega` in brackets; knobs resolve to the tier-3 heuristic
(no cache), like the 1x8 tables. Rendered from
`model_shapes/results_ep4/` — the same tables live in
[model_shapes/RESULTS_EP4.md](model_shapes/RESULTS_EP4.md).

Jobs: deepseek_v4_flash **2567353**+2567501, deepseek_v3 2567420+2567497,
kimi_k2_6 2567421+2567498, gpt_oss_120b 2567422+2567499, qwen3_5_397b
2567423+2567500, deepseek_v4_pro 2567424+2567502 (second ID = the 1024/4096
token points; `make_tables.py` merges the CSVs). Noise check
(`p50/min > 1.5x`, RUNBOOK §5): clean on every CSV.

**`deepseek_v4_flash`** — hidden 4096, inter 2048, 256 experts, top-6 — the geometry the §1 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 129.0 | 142.3 (0.91x) | 148.0 (0.87x) | 147.2 (0.88x) | 150.5 (0.86x) |
| 64 | 177.2 | 190.7 (0.93x) | 206.7 (0.86x) | 197.6 (0.90x) | 199.8 (0.89x) |
| 512 | 197.5 | 231.1 (0.85x) | 229.3 (0.86x) | 216.1 (0.91x) | 220.2 (0.90x) |
| 1024 | 229.4 | 259.9 (0.88x) | 260.1 (0.88x) | 233.4 (0.98x) | 241.7 (0.95x) |
| 2048 | 364.5 | 331.5 (1.10x) | 334.8 (1.09x) | 275.5 (1.32x) | 291.8 (1.25x) |
| 4096 | 610.3 | 533.5 (1.14x) | 531.4 (1.15x) | 398.3 (1.53x) | 442.3 (1.38x) |
| 8192 | 1123.3 | 988.2 (1.14x) | 976.9 (1.15x) | 698.0 (1.61x) | 803.8 (1.40x) |

**`deepseek_v4_pro`** — hidden 7168, inter 3072, 384 experts, top-6.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 290.8 | 306.2 (0.95x) | 310.3 (0.94x) | 310.5 (0.94x) | 312.3 (0.93x) |
| 64 | 549.6 | 571.4 (0.96x) | 581.0 (0.95x) | 573.4 (0.96x) | 577.0 (0.95x) |
| 512 | 609.3 | 632.8 (0.96x) | 642.6 (0.95x) | 610.4 (1.00x) | 617.5 (0.99x) |
| 1024 | 644.1 | 678.6 (0.95x) | 678.2 (0.95x) | 645.1 (1.00x) | 653.6 (0.99x) |
| 2048 | 824.3 | 762.8 (1.08x) | 758.8 (1.09x) | 693.2 (1.19x) | 711.7 (1.16x) |
| 4096 | 1457.3 | 994.3 (1.47x) | 1003.8 (1.45x) | 892.1 (1.63x) | 925.7 (1.57x) |
| 8192 | 2587.6 | 1593.9 (1.62x) | 1635.5 (1.58x) | 1459.7 (1.77x) | 1501.4 (1.72x) |

**`deepseek_v3`** — hidden 7168, inter 2048, 256 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 215.0 | 223.2 (0.96x) | 235.3 (0.91x) | 229.4 (0.94x) | 229.5 (0.94x) |
| 64 | 290.8 | 289.1 (1.01x) | 317.2 (0.92x) | 306.2 (0.95x) | 309.3 (0.94x) |
| 512 | 327.7 | 361.5 (0.91x) | 366.1 (0.90x) | 330.9 (0.99x) | 337.9 (0.97x) |
| 1024 | 440.3 | 425.3 (1.04x) | 429.0 (1.03x) | 357.4 (1.23x) | 373.7 (1.18x) |
| 2048 | 757.2 | 571.6 (1.32x) | 575.5 (1.32x) | 478.2 (1.58x) | 510.9 (1.48x) |
| 4096 | 1331.2 | 907.8 (1.47x) | 922.6 (1.44x) | 769.0 (1.73x) | 812.0 (1.64x) |
| 8192 | 2653.2 | 1731.6 (1.53x) | 1790.0 (1.48x) | 1359.3 (1.95x) | 1432.1 (1.85x) |

**`kimi_k2_6`** — hidden 7168, inter 2048, 384 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 247.9 | 257.0 (0.96x) | 277.5 (0.89x) | 263.2 (0.94x) | 265.2 (0.93x) |
| 64 | 413.7 | 408.6 (1.01x) | 445.2 (0.93x) | 424.0 (0.98x) | 425.3 (0.97x) |
| 512 | 461.8 | 472.6 (0.98x) | 476.2 (0.97x) | 449.0 (1.03x) | 454.6 (1.02x) |
| 1024 | 518.1 | 535.5 (0.97x) | 535.6 (0.97x) | 481.2 (1.08x) | 490.5 (1.06x) |
| 2048 | 743.5 | 658.4 (1.13x) | 650.2 (1.14x) | 545.8 (1.36x) | 580.6 (1.28x) |
| 4096 | 1365.5 | 971.6 (1.41x) | 988.0 (1.38x) | 838.7 (1.63x) | 879.0 (1.55x) |
| 8192 | 2700.8 | 1762.2 (1.53x) | 1809.4 (1.49x) | 1409.5 (1.92x) | 1523.2 (1.77x) |

**`qwen3_5_397b`** — hidden 4096, inter 1024, 512 experts, top-10.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 129.0 | 144.4 (0.89x) | 165.2 (0.78x) | 156.7 (0.82x) | 160.8 (0.80x) |
| 64 | 195.5 | 212.0 (0.92x) | 264.5 (0.74x) | 239.3 (0.82x) | 236.5 (0.83x) |
| 512 | 231.4 | 269.3 (0.86x) | 263.2 (0.88x) | 238.6 (0.97x) | 241.6 (0.96x) |
| 1024 | 294.9 | 302.1 (0.98x) | 300.0 (0.98x) | 267.3 (1.10x) | 281.6 (1.05x) |
| 2048 | 492.5 | 449.5 (1.10x) | 455.4 (1.08x) | 349.2 (1.41x) | 375.8 (1.31x) |
| 4096 | 917.5 | 781.3 (1.17x) | 794.1 (1.16x) | 560.1 (1.64x) | 650.2 (1.41x) |
| 8192 | 1759.2 | 1463.3 (1.20x) | 1502.2 (1.17x) | 1037.3 (1.70x) | 1203.2 (1.46x) |

**`gpt_oss_120b`** — hidden 2880, inter 2880, 128 experts, top-4 — `dg` is
`—` throughout: `deep_gemm_mega` requires hidden and intermediate divisible
by 128, and 2880 is not, so there is no baseline column (expected, not a
failed run).

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | — | 109.6 | 112.8 | 114.8 | 115.7 |
| 64 | — | 126.0 | 132.7 | 134.1 | 136.2 |
| 512 | — | 148.5 | 148.5 | 146.4 | 150.5 |
| 1024 | — | 165.9 | 168.0 | 165.8 | 160.8 |
| 2048 | — | 236.5 | 236.5 | 222.2 | 227.3 |
| 4096 | — | 349.4 | 348.2 | 302.1 | 312.3 |
| 8192 | — | 633.8 | 619.2 | 490.5 | 549.9 |

**The crossover moved up vs EP8, as predicted by RUNBOOK §2b item 5.** At EP4
each rank holds twice the experts, so tokens-per-expert halves at a given
tokens/rank and the DeepGEMM↔CuteDSL crossover sits between **1024 and 2048
tok/rank** on most shapes (vs 512-1024 at EP8); `deepseek_v3` crosses at
~1024. V4-Flash remains the least favourable geometry with a baseline —
plain `nvfp4 bf16` tops out at 1.14x (vs 1.20x at EP8 on B200), and
`+combine_nvfp4` reaches 1.61x at 8192. That decode-1k runs far below the
crossover is why §1's decode cell gains least (1.064x).

## 3. Accuracy gate — GSM8K, both checkpoints

Manual §4b cells on the hold node (job **2567354**), TP=4, 200 questions,
greedy:

| model | native | fi_dg | fi_cutedsl (NVFP4 cast) | delta |
|---|---|---|---|---|
| Flash | 0.965 (193/200) | 0.965 (193/200) | **0.975 (195/200)** | +0.010 |

Each result JSON (`vllm_e2e/results/gsm8k_ep4_*.json`) records the checkpoint
actually loaded: mx `hf-6e76323_orig` for native/fi_dg, NVFP4 `hf-48bfe38_orig`
for fi_cutedsl — the gate was armed. +0.010 on 200 questions is 2 questions,
i.e. noise; fi_cutedsl clears `--min-acc 0.93`.

Tier 2 smoke (same hold job): tier-1 config checks 15/15 (arch floor
validated against the live sm_103 device); the `[fi_moe_ep] ep_rank=` banner
appeared once per rank with `world=4` — four lines — in both fi logs and zero
times in the native log. fi_dg vs native was **8/8 exact-match** on GB300
(same DeepGEMM kernel through the wrapper); fi_cutedsl vs native 1/8 exact,
mean |dlogprob| 0.036-0.062 — inside the 1x8 branch's 0.016-0.13
cross-checkpoint band.

---

## 4. Provenance — SLURM job IDs

All 2026-08-03, partition `gb300`, this branch's scripts:

| what | job(s) | node |
|---|---|---|
| e2e sweep §1 (4 cells x 3 backends) | 2567468 | theia0062 |
| hold job: venv (FRESH=1), EP4 knob tune, tier 1-3 | 2567354 | theia0064 |
| micro deepseek_v4_flash | 2567353 + 2567501 | theia0284 / — |
| micro deepseek_v3 | 2567420 + 2567497 | theia0059 / theia0065 |
| micro kimi_k2_6 | 2567421 + 2567498 | theia0061 / — |
| micro gpt_oss_120b | 2567422 + 2567499 | theia0058 / — |
| micro qwen3_5_397b | 2567423 + 2567500 | theia0132 / — |
| micro deepseek_v4_pro | 2567424 + 2567502 | theia0133 / — |

(The `+` job is the 1024/4096 gap-fill at the same world size; `make_tables`
merges the CSVs.)

## 5. Tolerances and failure modes

The 1x8 branch's tolerances carry over: ratios ±0.02x between sessions,
absolutes move more with node/thermal state, all three backends of a cell in
one session, GSM8K ±0.02 is agreement. The two plausible-but-wrong failure
modes — `MAX_CAPTURE` without `CAPTURE_SIZES`, and exporting `MODEL` around
the gate — are documented in [expected_results.md](expected_results.md) §5
and apply here unchanged; every shipped EP4 cell pins `CAPTURE_SIZES` and
passes `--model` per backend.

One 1x4-specific hazard: the shipped `knob_cache_ep8.json` is keyed
`world_size: 8` / `NVIDIA B200`, so pointing a GB300 EP4 run at it silently
falls through to the heuristic (the cache never borrows a neighbour's knobs)
— fi_cutedsl would be understated, not wrong. The EP4 cells use
`knob_cache_ep4.json`, tuned at world 4 on GB300.
