# Expected results — 1x8 SM100, FlashInfer `moe_ep`

Every number here was measured from a scratch clone on one **8x B200** node.
If your run lands outside the tolerances below, something is different — §5
lists the three ways that has actually happened.

**Configuration (this branch, `v_0_26`).** vLLM 0.26 development tree —
the PR branch `fi-moe-ep-v4` @ `9dbb4c7e0` **built from source** (RUNBOOK
§1.2b) — flashinfer `moe_ep-respect-caller-device` @ `e4d7c1b3`,
nvidia-cutlass-dsl **4.6.1**. §1, §2, §3 and §4 carry numbers measured on
this stack (2026-08-05..07); §2b (serving) has not been re-measured and
still carries the July 0.25.1 recording (vLLM 0.25.1 wheel +
`vllm_e2e/patch_0251/`, flashinfer `4_5_2-perf-fix` @ `1ee41bcd`,
cutlass-dsl 4.5.2 — the branch `vllm_repro_8_gpu_v2` documents that stack
in full).

**EP8 throughout, but the two levels reach it differently.** The e2e sweeps
(§1, §2, §4) run **TP8 + EP8, DP1** — one engine sharded eight ways. The kernel
microbenchmark (§3) runs **DP8 + EP8, TP1** — one process per GPU via
`torch.multiprocessing`, no tensor sharding, since there is no model to shard.
Expert parallelism is 8 in both, which is the axis under test; do not read the
two levels as the same parallel configuration in every respect.

e2e also: kv fp8, block 256, prefix caching off, round 0 discarded as warmup,
median of the timed rounds (3 for every cell except the 100K one, which runs 2).

**The three backends, and what a ratio between them means.**

| name | `moe_backend` string | what it is |
|---|---|---|
| `native` | `deep_gemm_mega_moe` | vLLM's own DeepGEMM MegaMoE. The baseline every speedup below is relative to. |
| `fi_dg` | `flashinfer_moe_ep_mega_deep_gemm` | FlashInfer `moe_ep` DeepGEMM MegaMoE — **the same kernel as native**, reached through different glue: the `moe_ep` wrapper rather than a torch op. So `fi_dg` vs `native` isolates integration overhead, not kernel work, and ~1.00x is the expected answer. |
| `fi_cutedsl` | `flashinfer_moe_ep_mega_cutedsl` | FlashInfer `moe_ep` NVFP4 CuteDSL MegaMoE — a **different kernel**, and the one the work is actually about. Its speedup is the result. |

Read the two columns differently: `fi_dg` at ~1.00x says the wrapper costs
nothing (and, at 0.44x, said something was badly wrong — §5.1). `fi_cutedsl`
above 1.0x is the kernel win — 1.05-1.09x tuned on this stack (§1b, §2),
down from July's 1.19-1.32x for the structural reason §2 explains.

**Two checkpoints, deliberately.** native and fi_dg run the mx original;
fi_cutedsl runs the NVFP4 cast of the same base weights, because that is the
format its kernel consumes. That makes every throughput ratio
cross-checkpoint, which is why §4 exists and is not optional. On Pro the
prequant NVFP4 checkpoint turned out to be the dominant accuracy cost —
§4 separates checkpoint from kernel with a requant-at-load discriminator.

---

## 1. vLLM e2e — DeepSeek-V4-Flash, EP8

Measured 2026-08-06 on the configuration above (job 2368118; result JSONs
in `vllm_e2e/results/sweep_ep8pr_*.json`). Requires the sequence-parallel
fix `aa0317318` — without it the fi backends run the MoE block full-batch
on every rank and land at 0.42-0.65x. This table is the **default-config**
sweep: fi_cutedsl on heuristic knobs and the bf16 combine wire (the ship
default); §1b below is the tuned reading.

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 95238 | 95019 (0.998x) | 94310 (0.990x) |
| decode-1k | 47564 | 47444 (0.997x) | 45510 (0.957x) |
| 100K ISL / 1K | 54225 | 54218 (1.000x) | 50956 (0.940x) |
| 32K ISL / 32 | 75046 | 74976 (0.999x) | 74320 (0.990x) |

Native is 1.6-2.4x the July 0.25.1 recording (sequence-parallel MoE, fp8
sparse MLA attention, fused allreduce_rms); fi_dg tracks it at parity.
An earlier staging experiment (`756a6dd07`) made fi_dg bitwise-identical
to native, but an A/B showed FI's own CuTeDSL DataPreprocess staging is
faster (fi_dg 1.003-1.008x vs 0.999-1.002x), so it was reverted
(`9dbb4c7e0`) — fi_dg now matches native on 6 of 8 smoke prompts exactly,
not 8 of 8, and no doc should claim bitwise equality. fi_cutedsl's
sub-1.0x column here is the untuned reading: under sequence parallelism
the MoE sees tokens/rank 8x smaller, which puts these cells at or near
the §3 DeepGEMM↔CuteDSL crossover (512-1024 tok/rank), and the default
knob cache predates the sharded sizes.

The July 0.25.1+patch numbers this table replaced: native 38986/30845/
29632/35700 tok/s with fi_dg ~1.02x and fi_cutedsl 1.06-1.20x.

## 1b. Flash, fi_cutedsl tuned — nvfp4 combine + retuned knobs

Jobs 2370725 + 2370799 (2026-08-06); result JSONs and the knob caches in
`vllm_e2e/results/retune_20260806/`. Knobs tuned per capacity/live-token
operating point into `knob_cache_flash_nvfp4_{pre,dec}.json`; combine wire
switched to nvfp4. Both levers are follow-up-PR material — the shipped
default remains bf16 combine + heuristic fallback — so read this table as
the kernel's tuned potential on the 0.26 stack, not as what a stock build
produces:

| cell | tok/rank at MoE | native tok/s | fi_cutedsl (nvfp4 combine, tuned) |
|---|---|---|---|
| prefill-8k | 1024 | 94972 | 100141 (**1.054x**) |
| prefill-16k | 2048 | 103105 | 109130 (**1.058x**) |
| decode-1k | 128 | 49049 | 47407 (0.967x) |
| decode-2k | 256 | 53481 | 56374 (**1.054x**) |

Prefill recovers a 5-6% win over native (vs 0.99x untuned); decode-2k
flips positive with decode-shape knobs. decode-1k's 128 tok/rank sits
well below the §3 crossover and stays a small loss — expected, not a
regression. Accuracy under both levers is gated in §4 (Flash 500q: fi
nvfp4 combine 0.960 vs native 0.956).

## 2. vLLM e2e — DeepSeek-V4-Pro, EP8

Measured 2026-08-06 (job 2370724) on the configuration above, same
tuned-fi_cutedsl setup as §1b (nvfp4 combine + `knob_cache_pro_nvfp4_*`);
result JSONs in `vllm_e2e/results/retune_20260806/pro/`. The 100K-ISL and
32K-ISL cells and the fi_dg perf rows have **not** been re-measured on
the 0.26 stack — fi_dg's wrapper parity is established on Flash (§1) and
its Pro accuracy in §4.

| cell | tok/rank at MoE | native tok/s | fi_cutedsl (nvfp4 combine, tuned) |
|---|---|---|---|
| prefill-8k | 1024 | 39120 | 42432 (**1.085x**) |
| prefill-16k | 2048 | 40967 | 44754 (**1.092x**) |
| decode-1k | 128 | 20573 | 21924 (**1.066x**) |
| decode-2k | 256 | 19473 | 17938 (0.921x) |

decode-2k is a genuine tuned loss at 256 tok/rank; oddly non-monotonic
against decode-1k's win at 128 (Flash shows the mirror image — loses at
128, wins at 256). Unexplained, tracked as an FI-team follow-up.

**Do not compare this table to July's 1.19-1.32x and read a regression.**
The July win was structural, not kernel: pre-SP native ran the MoE block
full-batch on every rank (8x redundant at TP8) and fi's in-kernel
dispatch/combine ate that entire slice. Sequence parallelism removed it
for both backends. The 2026-08-07 recovery campaign
([reports/pro_perf_recovery_20260807.md](reports/pro_perf_recovery_20260807.md))
measured the walls: the kernel is still ~2x native's mega kernel at equal
tokens/rank (nsys, 443us vs 858us at 1024 tok/rank), but the MoE is now
~40% of step time (Amdahl ceiling ~1.15x) and Pro's KV cache caps batches
at 4096 tok/rank. Best measured big-batch cell: 32k-token capacity with
capture capped at 8192 — native 41925, fi_cutedsl **45559** (1.087x),
the highest Pro fi absolute of the campaign. July's 1.317x is
structurally unreachable on this stack at equal workloads; ~1.09x
best-vs-best is the honest ceiling.

The July 0.25.1+patch numbers this section replaced: native
15240/12897/12223/14117 tok/s (prefill-8k/decode-1k/100K/32K), fi_dg
1.019-1.026x, fi_cutedsl 1.192-1.317x.

## 2b. vLLM serving mode — server + client, both models

> **Not re-measured on the 0.26 stack.** Everything in this section is
> the July 0.25.1 recording, kept because what it establishes — that the
> serving harness reproduces the offline *ratios* to ±0.022x — is a
> property of the harness, not of the kernel stack. Expect current
> serving ratios to track §1/§1b/§2's offline columns the same way; the
> absolute tok/s below belong to the old stack and old (pre-SP) native.

`sbatch vllm_e2e/job_vllm_serving_sweep_ep8.sh` (Flash, ~2.5 h) and
`job_vllm_serving_sweep_pro.sh` (Pro, ~3.7 h); RUNBOOK §3g. One process runs
`vllm serve --moe-backend <be>` per (cell, backend), a second runs
`vllm bench serve` against it. The cells are §1/§2's four workloads verbatim
— same lengths, request counts and per-cell engine settings — so each row
below corresponds 1:1 to an offline row above. Round 0 is discarded as
warmup and each cell reports the **median of the timed client rounds**
(3, except 100K's 2): the native decode baseline drifts round-over-round in
serving just as it does offline (measured: single warm rounds of 26786 vs
28997 tok/s on two nodes — an 8% swing, larger than the fi-vs-native
effect, which is why single-round serving numbers are quoted nowhere in
this file).

Headline is **total token throughput** (input+output, client-measured over
HTTP), the same headline as §1/§2; ratio vs native:

**DeepSeek-V4-Flash** (job 2345223; ctx32k from rerun 2345501):

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 38074 | 39280 (1.032x) | 45643 (**1.199x**) |
| decode-1k | 29419 | 29958 (1.018x) | 30861 (**1.049x**) |
| 100K ISL / 1K | 29457 | 30059 (1.020x) | 32765 (**1.112x**) |
| 32K ISL / 32 | 35661 | 36518 (1.024x) | 41932 (**1.176x**) |

Flash serving lands within 1-3% of the offline absolutes and the fi_cutedsl
ratios match the offline column to ±0.012x on every cell (1.199 vs 1.195,
1.049 vs 1.061, 1.112 vs 1.111, 1.176 vs 1.177) — the two harnesses agree.
The ctx32k row was measured twice, in separate sessions on separate nodes
(2345223 then 2345501): 35583/36463/41888 vs 35661/36518/41932 — ratios
reproduce to 0.001x. Interactivity latency, fi_cutedsl vs native: TTFT p50
42.5 s vs 49.7 s at 100K and 12.9 s vs 15.2 s at 32K; ITL p99 236 ms vs
272 ms and p50 193 ms vs 228 ms.

**DeepSeek-V4-Pro** (job 2345224; ctx32k from rerun 2345502):

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 15216 | 15609 (1.026x) | 19860 (**1.305x**) |
| decode-1k | 12408 | 12619 (1.017x) | 14518 (**1.170x**) |
| 100K ISL / 1K | 12190 | 12417 (1.019x) | 14909 (**1.223x**) |
| 32K ISL / 32 | 13976 | 14290 (1.022x) | 18052 (**1.292x**) |

Pro agrees with §2 the same way Flash agrees with §1: fi_cutedsl ratios
within ±0.022x of the offline column on every cell (1.305 vs 1.317, 1.170
vs 1.192, 1.223 vs 1.231, 1.292 vs 1.293), fi_dg at wrapper parity
(1.02-1.03x) everywhere, and the fi_cutedsl win growing with model size —
over HTTP exactly as in-process. Like Flash, the ctx32k row was measured in
two sessions (2345224 then 2345502) and its ratios reproduce to 0.001x.
Interactivity latency, fi_cutedsl vs native: TTFT p50 95.5 s vs 122.4 s at
100K and 29.9 s vs 38.9 s at 32K; decode-1k ITL p50 73.8 ms vs 83.5 ms.

**Serving absolutes sit below the offline cells, by design.** The client
measures over HTTP, including tokenize/detokenize, streaming and scheduling
gaps between requests; the offline harness times `llm.generate()`
in-process. Compare serving to serving; the claim that carries across both
harnesses is the fi-vs-native *ratio*, and even that with the caveat that
the serving baseline is noisier — quote the medians and check the printed
min..max spread before reading anything into a small delta.

## 3. Kernel microbenchmark — no vLLM, no checkpoints

`GPUS=8 ./run.sh` for the ad-hoc sweep, or
`model_shapes/submit_jobs.sh` for the shape table; the CSVs behind these
tables are in `model_shapes/results_ep8_20260805_dsl461/` (the superseded
DSL-4.5.2 run remains in `model_shapes/results_ep8/`).

The tables below were re-measured 2026-08-05 on the current stack —
flashinfer `moe_ep-respect-caller-device` @ `e4d7c1b3` (post-restructure
main + flashinfer#4348), cutlass-dsl **4.6.1** (jobs 2367043-2367048).
Every cell landed within the §6 tolerances of the 4.5.2 numbers they
replace, so the two stacks are interchangeable at kernel level.

`e2e_pipelined` p50 microseconds per rank, with each CuteDSL variant's speedup
against `deep_gemm_mega` in brackets — higher is better, >1.00x means CuteDSL is
ahead. All six shapes in `model_shapes/shapes.tsv`, EP8:

**`deepseek_v4_flash`** — hidden 4096, inter 2048, 256 experts, top-6 — the geometry the §1 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 110.6 | 119.8 (0.92x) | 128.0 (0.86x) | 125.8 (0.88x) | 128.0 (0.86x) |
| 64 | 126.9 | 134.1 (0.95x) | 146.5 (0.87x) | 144.4 (0.88x) | 146.3 (0.87x) |
| 512 | 155.7 | 191.5 (0.81x) | 193.4 (0.81x) | 168.9 (0.92x) | 173.0 (0.90x) |
| 1024 | 237.7 | 234.5 (1.01x) | 240.4 (0.99x) | 197.7 (1.20x) | 209.9 (1.13x) |
| 2048 | 382.5 | 340.8 (1.12x) | 338.9 (1.13x) | 277.4 (1.38x) | 293.9 (1.30x) |
| 4096 | 686.6 | 588.7 (1.17x) | 584.6 (1.17x) | 431.1 (1.59x) | 484.3 (1.42x) |
| 8192 | 1328.1 | 1119.3 (1.19x) | 1108.9 (1.20x) | 787.5 (1.69x) | 912.3 (1.46x) |

**`deepseek_v4_pro`** — hidden 7168, inter 3072, 384 experts, top-6 — the geometry the §2 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 260.1 | 261.1 (1.00x) | 267.2 (0.97x) | 267.2 (0.97x) | 263.2 (0.99x) |
| 64 | 327.8 | 335.0 (0.98x) | 349.1 (0.94x) | 343.1 (0.96x) | 347.2 (0.94x) |
| 512 | 376.9 | 396.3 (0.95x) | 399.3 (0.94x) | 377.9 (1.00x) | 382.0 (0.99x) |
| 1024 | 491.6 | 443.5 (1.11x) | 447.3 (1.10x) | 421.0 (1.17x) | 428.5 (1.15x) |
| 2048 | 908.4 | 625.6 (1.45x) | 636.0 (1.43x) | 575.0 (1.58x) | 592.9 (1.53x) |
| 4096 | 1595.9 | 1029.1 (1.55x) | 1039.3 (1.54x) | 935.9 (1.71x) | 970.2 (1.64x) |
| 8192 | 3151.4 | 1849.4 (1.70x) | 1927.7 (1.63x) | 1685.4 (1.87x) | 1841.1 (1.71x) |

**`deepseek_v3`** — hidden 7168, inter 2048, 256 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 170.9 | 171.0 (1.00x) | 185.3 (0.92x) | 183.3 (0.93x) | 183.3 (0.93x) |
| 64 | 184.4 | 183.4 (1.01x) | 207.9 (0.89x) | 201.8 (0.91x) | 205.7 (0.90x) |
| 512 | 280.6 | 267.8 (1.05x) | 273.4 (1.03x) | 242.8 (1.16x) | 252.3 (1.11x) |
| 1024 | 467.4 | 376.3 (1.24x) | 383.9 (1.22x) | 314.3 (1.49x) | 328.6 (1.42x) |
| 2048 | 803.3 | 582.6 (1.38x) | 601.1 (1.34x) | 488.5 (1.64x) | 519.1 (1.55x) |
| 4096 | 1566.7 | 1060.9 (1.48x) | 1077.2 (1.45x) | 837.1 (1.87x) | 905.8 (1.73x) |
| 8192 | 3144.1 | 2066.4 (1.52x) | 2128.3 (1.48x) | 1577.0 (1.99x) | 1742.3 (1.80x) |

**`kimi_k2_6`** — hidden 7168, inter 2048, 384 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 206.9 | 210.0 (0.99x) | 226.3 (0.91x) | 218.0 (0.95x) | 218.1 (0.95x) |
| 64 | 253.9 | 246.8 (1.03x) | 271.4 (0.94x) | 257.0 (0.99x) | 257.1 (0.99x) |
| 512 | 313.4 | 320.6 (0.98x) | 321.1 (0.98x) | 291.8 (1.07x) | 301.0 (1.04x) |
| 1024 | 450.7 | 408.6 (1.10x) | 412.7 (1.09x) | 340.9 (1.32x) | 354.8 (1.27x) |
| 2048 | 816.7 | 621.6 (1.31x) | 644.0 (1.27x) | 527.4 (1.55x) | 558.1 (1.46x) |
| 4096 | 1626.6 | 1089.1 (1.49x) | 1072.1 (1.52x) | 866.2 (1.88x) | 900.7 (1.81x) |
| 8192 | 3136.6 | 2110.4 (1.49x) | 2147.2 (1.46x) | 1643.5 (1.91x) | 1762.2 (1.78x) |

**`qwen3_5_397b`** — hidden 4096, inter 1024, 512 experts, top-10.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 112.7 | 123.9 (0.91x) | 150.4 (0.75x) | 138.3 (0.81x) | 140.3 (0.80x) |
| 64 | 130.6 | 142.4 (0.92x) | 180.7 (0.72x) | 162.8 (0.80x) | 166.8 (0.78x) |
| 512 | 194.6 | 212.0 (0.92x) | 216.1 (0.90x) | 179.3 (1.09x) | 191.4 (1.02x) |
| 1024 | 311.2 | 302.1 (1.03x) | 316.2 (0.98x) | 240.7 (1.29x) | 261.2 (1.19x) |
| 2048 | 546.8 | 474.0 (1.15x) | 482.3 (1.13x) | 354.7 (1.54x) | 392.1 (1.39x) |
| 4096 | 1023.0 | 871.3 (1.17x) | 893.9 (1.14x) | 611.3 (1.67x) | 709.1 (1.44x) |
| 8192 | 1992.7 | 1649.6 (1.21x) | 1707.1 (1.17x) | 1133.0 (1.76x) | 1307.7 (1.52x) |

**`gpt_oss_120b`** — hidden 2880, inter 2880, 128 experts, top-4 — `dg` is
`—` throughout: `deep_gemm_mega` requires hidden and intermediate both
divisible by 128, and 2880 is not, so there is no baseline to divide by.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | — | 93.2 | 97.2 | 97.3 | 97.3 |
| 64 | — | 95.1 | 99.4 | 101.3 | 101.4 |
| 512 | — | 132.2 | 136.3 | 128.9 | 130.1 |
| 1024 | — | 175.1 | 179.2 | 168.0 | 173.1 |
| 2048 | — | 246.8 | 248.9 | 222.2 | 230.4 |
| 4096 | — | 386.1 | 388.1 | 332.9 | 347.7 |
| 8192 | — | 709.6 | 709.7 | 549.4 | 619.5 |

**The crossover sits between 512 and 1024 tok/rank on every shape.** Below it
`deep_gemm_mega` wins; above it the CuteDSL variants pull away, and the
quantized-combine wires (`+combine_nvfp4`, `+combine_mxfp8`) extend the lead
further at large batches.

**V4-Flash is the least favourable geometry of the five that have a baseline.**
The shape the §1 e2e sweep uses tops out at 1.20x on plain `nvfp4_bf16`, where
V4-Pro reaches 1.64x and `deepseek_v3` 1.59x. So §1's 1.06-1.20x end-to-end is
a conservative reading of the kernel, and §2's larger Pro gains follow the
kernel rather than any integration difference.

`gpt_oss_120b`'s empty `dg` column is expected, not a failed run: the harness
attempts `deep_gemm_mega`, hits its `hidden % 128 == 0 && intermediate % 128 == 0`
assertion once per token count, logs it and carries on with the CuteDSL
variants.

`acc_loss_pct` in the CSVs is a synthetic-input reconstruction error (20.6% for
`deep_gemm_mega`, 23.1-24.9% for the CuteDSL variants), **not** a model-quality
number. Model quality is §4.

## 4. Accuracy gate — GSM8K, both checkpoints

Re-measured 2026-08-05..06 on the 0.26 stack, at 200 questions (the
gate's default, `sbatch vllm_e2e/job_gsm8k_flash_pro.sh`, ~35 min) and at
500 questions where 200q granularity (0.005/question) could not separate
the hypotheses.

**Flash — passes cleanly, both combine wires** (500q, job 2370839;
fi_cutedsl on the prequant NVFP4 cast, truncations 0-1 everywhere):

| backend | 200q | 500q |
|---|---|---|
| native | 0.965 | 0.956 |
| fi_cutedsl, bf16 combine | 0.965 | 0.966 |
| fi_cutedsl, nvfp4 combine | — | 0.960 |

fi ≥ native on both wires; the §1b perf claims carry no accuracy
asterisk on Flash.

**Pro — the prequant NVFP4 checkpoint is the culprit, not the kernel.**
Native on the 0.26 stack scores 0.895 at 200q / 0.904 at 500q (job
2370516/2370584) — the recorded 0.880-0.890 band held, so any fi drop is
real. fi_cutedsl on the prequant `deepseek-v4-pro-nvfp4` checkpoint drops
hard: 0.806 at 500q bf16 combine (0.784 nvfp4 combine), with 2-5x more
truncations. Two discriminators (job 2370724, Phase C) isolate the cause:

| Pro configuration | 200q | 500q |
|---|---|---|
| native, mx original | 0.895 | 0.904 |
| fi_dg, mx original | 0.895 | — |
| fi_cutedsl, **requant-at-load** from mx original, bf16 combine | 0.885 | 0.866 |
| fi_cutedsl, prequant NVFP4 ckpt, bf16 combine | — | 0.806 |
| fi_cutedsl, requant-at-load, nvfp4 combine | 0.855 | 0.842 |

Three verdicts, in order of size:

1. **Checkpoint.** fi_dg on the mx original matches native exactly
   (179/200), and requant-at-load recovers most of the prequant drop
   (0.806 → 0.866) — regenerate the Pro NVFP4 checkpoint before quoting
   any accuracy number from it.
2. **Residual kernel gap, Pro-specific.** Requant bf16-combine still
   sits ~0.04 below native at 500q (2.8σ — not noise). Flash shows no
   such gap (fi ≥ native on the *prequant* cast), so it is tied to Pro's
   scale/shape, not to nvfp4-vs-fp8 generically. Reported to the FI team.
3. **nvfp4 combine costs ~2.4-3.0% on Pro** (two consistent samples),
   ~0 on Flash — which is why bf16 combine ships as the default and the
   nvfp4 wire stays a tuning lever (§1b). Note it becomes mandatory at
   32k-token capacity on Pro, where bf16 combine staging does not fit.

**Do not read Pro's ~0.90 native as a regression** — all backends on the
mx original agree, and it is not a truncation artifact (raising
`--max-tokens` 512→2048 moves accuracy <0.005 while a handful of
completions never terminate at any budget). `--min-acc 0.93` is
calibrated for Flash; for Pro it will fail and that failure is expected.

---

## 5. The three ways this has actually gone wrong

Both produced *plausible wrong numbers* rather than errors, which is why they
are documented rather than merely fixed.

### 5.1 A cell that sets `MAX_CAPTURE` without pinning `CAPTURE_SIZES`

The dense default capture ladder makes vLLM's CUDA-graph memory profiler
reserve **~48 GiB/GPU** for the flashinfer backends against a real capture cost
of ~6 GiB — the same ~6 GiB it estimates correctly for native. The phantom
reservation is taken out of the KV cache. Measured by running the V4-Pro
decode cell unpinned on purpose — these rows are not in `results/`, since the
shipped cells all pin `CAPTURE_SIZES`:

| backend | KV available | KV tokens | resident seqs | tok/s |
|---|---|---|---|---|
| native | 48.91 GiB | 95,979 | 1024 of 1024 | 13268 |
| fi_dg | 7.28 GiB | 14,286 | **189** | 5969 (0.45x) |
| fi_cutedsl | 0.07 GiB | — | engine will not start | OOM |

**The tell is a backend that is fast per step and slow overall.** fi_dg's ITL
was *better* than native (42.6 vs 94.4 ms) precisely because its batches were
5x smaller — it reads as a good kernel on a starved engine. Check
`Available KV cache memory` and the scheduler's `Running:`/`Waiting:` counts
before blaming a kernel.

Severity scales with how little KV headroom the model leaves: Pro EP8 held 189
of 1024 requested sequences, Flash EP8 520 — the bigger the weights, the harder
it bites, and on a small enough model it hides entirely. All shipped cells pin
`CAPTURE_SIZES`; doing so costs native ~3% to batch padding, which is already
reflected in §1 and §2.

### 5.2 Exporting `MODEL` around the GSM8K gate

`resolve_model` ranks `--model` > `$MODEL` > per-backend default. A `MODEL=`
in the environment therefore sends *every* backend to that checkpoint,
including fi_cutedsl — the gate then compares the mx weights against
themselves, scores a comfortable pass, and validates nothing — the fi_cutedsl
row carries the mx checkpoint under an NVFP4 label.

`job_gsm8k_flash_pro.sh` passes `--model` per cell and unsets `MODEL` inside
the container. **Check the `model` field in each result JSON** — it records
what was actually loaded, and the job's summary flags any fi_cutedsl row that
did not run an nvfp4 checkpoint.

### 5.3 A big-batch cell whose batches never form

Raising `MAX_BATCHED_TOKENS` only raises a *ceiling*; the scheduler fills
batches from resident sequences, and those are capped by KV cache. On Pro
at 32k capacity with `gpu_memory_utilization=0.90` + graph profiling, only
14,143 KV tokens survived — 3.45x concurrency — so the "pre32k" cell
silently ran ~4k-token steps and reported roughly half the real number
for both backends. The starved config also crashed fi_cutedsl+nvfp4-combine
deterministically under a 32768-token graph (clean with headroom — an
FI-team item). Fix: `GPU_MEM_UTIL=0.95`,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, `MAX_MODEL_LEN` no larger
than the workload needs. **Check `GPU KV cache size` / `Maximum
concurrency` in the engine log for every big-batch cell** before believing
its tok/s; details in
[reports/pro_perf_recovery_20260807.md](reports/pro_perf_recovery_20260807.md).

---

## 6. Tolerances

Ratios are stable to about ±0.02x between sessions; absolute throughput moves
more with node and thermal state. Run all three backends of a cell **in one
session** — native's decode drifts round-over-round, so cross-session ratios
are not trustworthy. GSM8K on 200 questions has a granularity of 0.005, so
treat anything inside ±0.02 as agreement.

The ±0.02x above is measured, not assumed: repeating the whole set on a second
pass reproduced it to within **0.5% on absolute throughput and 0.008x on every
ratio**. Every table was produced by this branch's own scripts against a freshly
built venv.
