# Expected results — 1x8 SM100, FlashInfer `moe_ep`

Every number here was measured from a scratch clone on **2026-07-25**, on one
**8x B200** node. If your run lands outside the tolerances below, something is
different — §5 lists the two ways that has actually happened.

**Configuration.** vLLM 0.25.1 (wheel + `vllm_e2e/patch_0251/`), flashinfer
branch `4_5_2-perf-fix` @ `1ee41bcd`, nvidia-cutlass-dsl **4.5.2** (vLLM
0.25.1's own pin), TP8 + EP8, DP1, kv fp8, block 256,
prefix caching off, round 0 discarded as warmup, median of 3 timed rounds.

**The three backends, and what a ratio between them means.**

| name | `moe_backend` string | what it is |
|---|---|---|
| `native` | `deep_gemm_mega_moe` | vLLM's own DeepGEMM MegaMoE. The baseline every speedup below is relative to. |
| `fi_dg` | `flashinfer_moe_ep_mega_deep_gemm` | FlashInfer `moe_ep` DeepGEMM MegaMoE — **the same kernel as native**, reached through different glue: the `moe_ep` wrapper rather than a torch op. So `fi_dg` vs `native` isolates integration overhead, not kernel work, and ~1.00x is the expected answer. |
| `fi_cutedsl` | `flashinfer_moe_ep_mega_cutedsl` | FlashInfer `moe_ep` NVFP4 CuteDSL MegaMoE — a **different kernel**, and the one the work is actually about. Its speedup is the result. |

Read the two columns differently: `fi_dg` at 1.02x says the wrapper costs
nothing (and, at 0.44x, said something was badly wrong — §5.1). `fi_cutedsl`
at 1.31x is the kernel win.

**Two checkpoints, deliberately.** native and fi_dg run the mx original;
fi_cutedsl runs the NVFP4 cast of the same base weights, because that is the
format its kernel consumes. That makes every throughput ratio
cross-checkpoint, which is why §4 exists and is not optional.

---

## 1. vLLM e2e — DeepSeek-V4-Flash, EP8

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` (~1 h). Job **2337646**.

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

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` (~2 h). Job **2337637**.

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 15240 | 15630 (1.026x) | 20074 (**1.317x**) |
| decode-1k | 12897 | 13157 (1.020x) | 15368 (**1.192x**) |
| 100K ISL / 1K | 12223 | 12453 (1.019x) | 15053 (**1.231x**) |
| 32K ISL / 32 | 14117 | 14435 (1.023x) | 18250 (**1.293x**) |

Latency, fi_cutedsl vs native: TTFT 95.0s vs 122.3s at 100K and 29.5s vs 38.4s
at 32K; ITL p50 111.8ms vs 134.3ms and 441.3ms vs 573.9ms.

**The fi_cutedsl win grows with model size** — 1.19-1.31x on Pro against
1.06-1.20x on Flash, on identical cells. fi_dg is at parity (1.02x) everywhere,
on both models. If you see fi_dg far from 1.02x, read §5.1 before believing it.

## 3. Kernel microbenchmark — no vLLM, no checkpoints

`GPUS=8 ./run.sh` for the ad-hoc sweep, or
`model_shapes/submit_jobs.sh` for the shape table. Job **2337617**;
`model_shapes/results_ep8/model_shapes_20260725_154623_deepseek_v4_flash.csv`.

`e2e_pipelined` p50 microseconds per rank, with each CuteDSL variant's speedup
against `deep_gemm_mega` in brackets — higher is better, >1.00x means CuteDSL is
ahead. All six shapes in `model_shapes/shapes.tsv`, EP8:

**`deepseek_v4_flash`** — hidden 4096, inter 2048, 256 experts, top-6 — the geometry the §1 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 108.5 | 119.8 (0.91x) | 128.0 (0.85x) | 126.0 (0.86x) | 128.0 (0.85x) |
| 64 | 124.9 | 132.2 (0.94x) | 146.4 (0.85x) | 146.4 (0.85x) | 144.4 (0.86x) |
| 512 | 154.7 | 189.4 (0.82x) | 192.0 (0.81x) | 168.9 (0.92x) | 173.1 (0.89x) |
| 1024 | 233.5 | 232.4 (1.00x) | 237.0 (0.99x) | 197.5 (1.18x) | 205.9 (1.13x) |
| 2048 | 379.1 | 334.8 (1.13x) | 334.8 (1.13x) | 273.4 (1.39x) | 287.7 (1.32x) |
| 4096 | 680.0 | 578.5 (1.18x) | 574.4 (1.18x) | 422.9 (1.61x) | 472.0 (1.44x) |
| 8192 | 1320.4 | 1104.8 (1.20x) | 1091.7 (1.21x) | 772.2 (1.71x) | 887.3 (1.49x) |

**`deepseek_v4_pro`** — hidden 7168, inter 3072, 384 experts, top-6 — the geometry the §2 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 260.1 | 261.1 (1.00x) | 267.3 (0.97x) | 267.2 (0.97x) | 268.6 (0.97x) |
| 64 | 327.7 | 334.7 (0.98x) | 349.1 (0.94x) | 345.0 (0.95x) | 347.1 (0.94x) |
| 512 | 376.9 | 394.3 (0.96x) | 398.3 (0.95x) | 377.8 (1.00x) | 382.0 (0.99x) |
| 1024 | 492.1 | 441.3 (1.12x) | 444.4 (1.11x) | 418.8 (1.18x) | 426.9 (1.15x) |
| 2048 | 899.1 | 626.2 (1.44x) | 664.0 (1.35x) | 572.4 (1.57x) | 586.7 (1.53x) |
| 4096 | 1591.8 | 1023.0 (1.56x) | 1036.2 (1.54x) | 941.1 (1.69x) | 962.0 (1.65x) |
| 8192 | 3144.2 | 1919.0 (1.64x) | 1945.0 (1.62x) | 1716.7 (1.83x) | 1727.9 (1.82x) |

**`deepseek_v3`** — hidden 7168, inter 2048, 256 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 170.8 | 171.0 (1.00x) | 185.3 (0.92x) | 183.2 (0.93x) | 183.3 (0.93x) |
| 64 | 184.4 | 183.3 (1.01x) | 207.9 (0.89x) | 203.7 (0.91x) | 205.9 (0.90x) |
| 512 | 282.6 | 267.3 (1.06x) | 271.2 (1.04x) | 242.8 (1.16x) | 253.0 (1.12x) |
| 1024 | 465.0 | 375.8 (1.24x) | 384.0 (1.21x) | 314.4 (1.48x) | 326.6 (1.42x) |
| 2048 | 809.4 | 576.5 (1.40x) | 598.9 (1.35x) | 490.5 (1.65x) | 517.1 (1.57x) |
| 4096 | 1604.6 | 1045.5 (1.53x) | 1061.9 (1.51x) | 860.1 (1.87x) | 892.9 (1.80x) |
| 8192 | 3236.3 | 2031.6 (1.59x) | 2092.1 (1.55x) | 1576.0 (2.05x) | 1707.0 (1.90x) |

**`kimi_k2_6`** — hidden 7168, inter 2048, 384 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 206.8 | 209.9 (0.99x) | 226.3 (0.91x) | 216.1 (0.96x) | 218.0 (0.95x) |
| 64 | 253.9 | 245.3 (1.04x) | 271.3 (0.94x) | 257.0 (0.99x) | 257.0 (0.99x) |
| 512 | 315.4 | 320.5 (0.98x) | 322.0 (0.98x) | 291.8 (1.08x) | 302.2 (1.04x) |
| 1024 | 450.7 | 408.4 (1.10x) | 410.6 (1.10x) | 342.9 (1.31x) | 355.3 (1.27x) |
| 2048 | 825.3 | 619.5 (1.33x) | 639.9 (1.29x) | 533.6 (1.55x) | 563.1 (1.47x) |
| 4096 | 1664.0 | 1048.5 (1.59x) | 1070.0 (1.56x) | 875.5 (1.90x) | 901.7 (1.85x) |
| 8192 | 3128.4 | 2045.9 (1.53x) | 2101.2 (1.49x) | 1628.6 (1.92x) | 1759.7 (1.78x) |

**`qwen3_5_397b`** — hidden 4096, inter 1024, 512 experts, top-10.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 112.7 | 125.8 (0.90x) | 150.6 (0.75x) | 138.2 (0.82x) | 141.7 (0.80x) |
| 64 | 131.1 | 142.5 (0.92x) | 181.2 (0.72x) | 162.8 (0.81x) | 166.8 (0.79x) |
| 512 | 194.7 | 209.9 (0.93x) | 214.9 (0.91x) | 179.2 (1.09x) | 189.4 (1.03x) |
| 1024 | 309.2 | 298.0 (1.04x) | 310.3 (1.00x) | 240.6 (1.29x) | 259.2 (1.19x) |
| 2048 | 549.0 | 465.9 (1.18x) | 474.1 (1.16x) | 351.1 (1.56x) | 384.0 (1.43x) |
| 4096 | 1032.2 | 855.1 (1.21x) | 877.6 (1.18x) | 592.9 (1.74x) | 686.0 (1.50x) |
| 8192 | 2022.4 | 1619.0 (1.25x) | 1678.3 (1.21x) | 1098.8 (1.84x) | 1270.8 (1.59x) |

**`gpt_oss_120b`** — hidden 2880, inter 2880, 128 experts, top-4 — `dg` is
`—` throughout: `deep_gemm_mega` requires hidden and intermediate both
divisible by 128, and 2880 is not, so there is no baseline to divide by.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | — | 93.2 | 95.4 | 97.2 | 97.4 |
| 64 | — | 95.2 | 99.4 | 101.3 | 101.4 |
| 512 | — | 132.2 | 136.2 | 127.9 | 132.0 |
| 1024 | — | 173.1 | 177.2 | 165.0 | 169.0 |
| 2048 | — | 240.7 | 244.8 | 222.1 | 226.4 |
| 4096 | — | 379.9 | 383.8 | 329.6 | 339.1 |
| 8192 | — | 697.4 | 697.2 | 541.7 | 607.3 |

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
Job **2337638**; token-budget probe 2337550.

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
reservation is taken out of the KV cache. Measured on Pro EP8 (job 2337473):

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
it bites, and on a small enough model it hides entirely. All shipped cells now
pin `CAPTURE_SIZES`; pinning costs native ~3% to batch padding, so **decode-1k
numbers from before 2026-07-25 are not comparable with these.**

### 5.2 Exporting `MODEL` around the GSM8K gate

`resolve_model` ranks `--model` > `$MODEL` > per-backend default. A `MODEL=`
in the environment therefore sends *every* backend to that checkpoint,
including fi_cutedsl — the gate then compares the mx weights against
themselves, scores a comfortable pass, and validates nothing. This happened
(job 2337127): its `fi_nvfp4` result recorded `model=...hf-6e76323_orig`.

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

Provenance: every table above is the **verification pass of 2026-07-25
evening**, run from this branch's own scripts against a freshly rebuilt venv —
microbenchmark **2337617**, Flash e2e **2337646**, Pro e2e **2337637**, GSM8K
**2337638**. It reproduced the original measurement pass (2337199 / 2337204 /
2337438 / 2337487 / 2337476) to within **0.5% on absolute throughput and
0.008x on every ratio**, which is where the ±0.02x tolerance above comes from.
The diagnostic jobs behind §5 are 2337473 (root cause), 2337549 (Flash fix) and
2337550 (token budget). The full chronological log lives on the `vllm-pr`
branch in `vllm_e2e/RUNS.md` (runs 43-50).
