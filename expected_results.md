# Expected results — 1x8 SM100, FlashInfer `moe_ep`

Every number here was measured from a scratch clone on **2026-07-25**, on one
**8x B200** node. If your run lands outside the tolerances below, something is
different — §5 lists the two ways that has actually happened.

**Configuration.** vLLM 0.25.1 (wheel + `vllm_e2e/patch_0251/`), flashinfer
branch `4_5_2-perf-fix` @ `1ee41bcd`, nvidia-cutlass-dsl **4.5.2** (pinned — the
CuteDSL codegen is 34-54% slower before it), TP8 + EP8, DP1, kv fp8, block 256,
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

DSV4-Flash geometry (hidden 4096, inter 2048, 256 experts, top-6). `e2e_pipelined`
p50 microseconds per rank, with the same numbers as speedup vs
`deep_gemm_mega` — higher is better, >1.00x means the CuteDSL kernel is ahead:

| tok/rank | deep_gemm_mega | nvfp4_cutedsl | +combine_mxfp8 | +combine_nvfp4 |
|---|---|---|---|---|
| 8 | 108.5 µs | 119.8 (0.91x) | 128.0 (0.85x) | 126.0 (0.86x) |
| 64 | 124.9 µs | 132.2 (0.94x) | 144.4 (0.86x) | 146.4 (0.85x) |
| 512 | 154.7 µs | 189.4 (0.82x) | 173.1 (0.89x) | 168.9 (0.92x) |
| 1024 | 233.5 µs | 232.4 (**1.00x**) | 205.9 (1.13x) | 197.5 (1.18x) |
| 2048 | 379.1 µs | 334.8 (1.13x) | 287.7 (1.32x) | 273.4 (1.39x) |
| 4096 | 680.0 µs | 578.5 (1.18x) | 472.0 (1.44x) | 422.9 (1.61x) |
| 8192 | 1320.4 µs | 1104.8 (1.20x) | 887.3 (1.49x) | **772.2 (1.71x)** |

The crossover is near 1024 tokens/rank: below it deep_gemm_mega wins, above it
the cutedsl kernels pull away, reaching 1.70x at 8192. This is the kernel-level
shape of the e2e prefill win in §1-2 — large batches are where it pays.

`acc_loss_pct` in that CSV is a synthetic-input reconstruction error (20.6% for
deep_gemm_mega, 23.1-24.9% for the cutedsl variants), **not** a model-quality
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
