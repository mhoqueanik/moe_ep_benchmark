# Expected results — 1x8 SM100, FlashInfer `moe_ep`

Every number here was measured from a scratch clone on one **8x B200** node.
If your run lands outside the tolerances below, something is different — §5
lists the two ways that has actually happened.

**Configuration.** vLLM 0.25.1 (wheel + `vllm_e2e/patch_0251/`), flashinfer
branch `4_5_2-perf-fix` @ `1ee41bcd`, nvidia-cutlass-dsl **4.5.2** (vLLM
0.25.1's own pin).

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

Read the two columns differently: `fi_dg` at 1.02x says the wrapper costs
nothing (and, at 0.44x, said something was badly wrong — §5.1). `fi_cutedsl`
at 1.32x is the kernel win.

**Two checkpoints, deliberately.** native and fi_dg run the mx original;
fi_cutedsl runs the NVFP4 cast of the same base weights, because that is the
format its kernel consumes. That makes every throughput ratio
cross-checkpoint, which is why §4 exists and is not optional.

---

## 1. vLLM e2e — DeepSeek-V4-Flash, EP8

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` (~1 h).

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 38986 | 40225 (1.032x) | 46584 (**1.195x**) |
| decode-1k | 30845 | 31494 (1.021x) | 32741 (**1.061x**) |
| 100K ISL / 1K | 29632 | 30230 (1.020x) | 32913 (**1.111x**) |
| 32K ISL / 32 | 35700 | 36597 (1.025x) | 42028 (**1.177x**) |

Latency on the interactivity cells (`REQUIRE_LATENCY=1`), fi_cutedsl vs native:
TTFT 42.2s vs 49.3s at 100K, 12.8s vs 15.1s at 32K; ITL p50 51.9ms vs 56.1ms
and 191.2ms vs 226.5ms.

## 2. vLLM e2e — DeepSeek-V4-Pro, EP8

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` (~2 h).

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 15240 | 15630 (1.026x) | 20074 (**1.317x**) |
| decode-1k | 12897 | 13157 (1.020x) | 15368 (**1.192x**) |
| 100K ISL / 1K | 12223 | 12453 (1.019x) | 15053 (**1.231x**) |
| 32K ISL / 32 | 14117 | 14435 (1.023x) | 18250 (**1.293x**) |

Latency, fi_cutedsl vs native: TTFT 95.0s vs 122.3s at 100K and 29.5s vs 38.4s
at 32K; ITL p50 111.8ms vs 134.3ms and 441.3ms vs 573.9ms.

**The fi_cutedsl win grows with model size** — 1.19-1.32x on Pro against
1.06-1.20x on Flash, on identical cells. fi_dg is at parity (1.02x) everywhere,
on both models. If you see fi_dg far from 1.02x, read §5.1 before believing it.

## 2b. vLLM serving mode — server + client, both models

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

`sbatch vllm_e2e/job_gsm8k_flash_pro.sh` (~35 min, both models, both at TP8).

| model | native | fi_dg | fi_cutedsl (NVFP4 cast) | delta |
|---|---|---|---|---|
| Flash | 0.965 | 0.965 | **0.965** | +0.000 |
| Pro | 0.880 | 0.880 | **0.890** | +0.010 |

**The delta is the number that gates a perf claim.** Because fi_cutedsl runs a
different checkpoint, its throughput is only comparable if its accuracy is —
±0.010 on 200 questions is 2 questions, i.e. noise. Both models pass.

**Do not read Pro's 0.880 as a regression.** All three backends agree exactly
(176/176/178 correct; Flash is 193/193/193), so it is a property of the model
and this eval, not of
`moe_ep`. It is not a truncation artifact either: raising `--max-tokens`
512 -> 1024 -> 2048 moves accuracy 0.8800 -> 0.8750 -> 0.8750 while truncated
completions only fall 15 -> 14 -> 13, i.e. a handful never terminate at any
budget. `--min-acc 0.93` is calibrated for Flash; for Pro it will fail and that
failure is expected.

---

## 5. The two ways this has actually gone wrong

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
