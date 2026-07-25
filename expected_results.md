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

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` (~1 h). Job 2337204;
decode-1k re-measured in 2337549 after the fix in §5.1.

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 39012 | 39824 (1.021x) | 46774 (**1.199x**) |
| decode-1k | 30724 | 31517 (1.026x) | 32695 (**1.064x**) |
| 100K ISL / 1K | 29642 | 30265 (1.021x) | 32940 (**1.111x**) |
| 32K ISL / 32 | 35714 | 36604 (1.025x) | 42078 (**1.178x**) |

Latency on the interactivity cells (`REQUIRE_LATENCY=1`), fi_cutedsl vs native:
TTFT 42.1s vs 49.3s at 100K, 12.8s vs 15.1s at 32K; ITL p50 51.8ms vs 56.1ms
and 190.9ms vs 226.4ms.

## 2. vLLM e2e — DeepSeek-V4-Pro, EP8

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` (~2 h). Job 2337438;
decode-1k from 2337487.

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 15178 | 15666 (1.032x) | 19873 (**1.309x**) |
| decode-1k | 12872 | 13143 (1.021x) | 15327 (**1.191x**) |
| 100K ISL / 1K | 12174 | 12404 (1.019x) | 14995 (**1.232x**) |
| 32K ISL / 32 | 14051 | 14365 (1.022x) | 18162 (**1.293x**) |

**The fi_cutedsl win grows with model size** — 1.19-1.31x on Pro against
1.06-1.20x on Flash, on identical cells. fi_dg is at parity (1.02x) everywhere,
on both models. If you see fi_dg far from 1.02x, read §5.1 before believing it.

## 3. Kernel microbenchmark — no vLLM, no checkpoints

`GPUS=8 ./run.sh` for the ad-hoc sweep, or
`model_shapes/submit_jobs.sh` for the shape table. Job 2337199;
`model_shapes/results_ep8/model_shapes_20260725_045752_deepseek_v4_flash.csv`.

DSV4-Flash geometry (hidden 4096, inter 2048, 256 experts, top-6). `e2e_pipelined`
p50 microseconds per rank, with the same numbers as speedup vs
`deep_gemm_mega` — higher is better, >1.00x means the CuteDSL kernel is ahead:

| tok/rank | deep_gemm_mega | nvfp4_cutedsl | +combine_mxfp8 | +combine_nvfp4 |
|---|---|---|---|---|
| 8 | 108.6 µs | 121.8 (0.89x) | 128.1 (0.85x) | 128.0 (0.85x) |
| 64 | 125.0 µs | 134.2 (0.93x) | 146.4 (0.85x) | 146.5 (0.85x) |
| 512 | 155.7 µs | 191.5 (0.81x) | 175.1 (0.89x) | 169.1 (0.92x) |
| 1024 | 237.7 µs | 232.4 (**1.02x**) | 207.9 (1.14x) | 197.5 (1.20x) |
| 2048 | 381.0 µs | 336.7 (1.13x) | 289.8 (1.31x) | 275.4 (1.38x) |
| 4096 | 693.4 µs | 578.6 (1.20x) | 471.9 (1.47x) | 425.0 (1.63x) |
| 8192 | 1321.4 µs | 1100.8 (1.20x) | 880.7 (1.50x) | **779.2 (1.70x)** |

The crossover is near 1024 tokens/rank: below it deep_gemm_mega wins, above it
the cutedsl kernels pull away, reaching 1.70x at 8192. This is the kernel-level
shape of the e2e prefill win in §1-2 — large batches are where it pays.

`acc_loss_pct` in that CSV is a synthetic-input reconstruction error (20.6% for
deep_gemm_mega, 23.1-24.9% for the cutedsl variants), **not** a model-quality
number. Model quality is §4.

## 4. Accuracy gate — GSM8K, both checkpoints

`sbatch vllm_e2e/job_gsm8k_flash_pro.sh` (~35 min, both models). Job 2337476;
token-budget probe 2337550.

| model | native | fi_dg | fi_cutedsl (NVFP4 cast) | delta |
|---|---|---|---|---|
| Flash | 0.960 | 0.960 | **0.970** | +0.010 |
| Pro | 0.880 | 0.880 | **0.890** | +0.010 |

**The delta is the number that gates a perf claim.** Because fi_cutedsl runs a
different checkpoint, its throughput is only comparable if its accuracy is —
±0.010 on 200 questions is 2 questions, i.e. noise. Both models pass.

**Do not read Pro's 0.880 as a regression.** All three backends agree exactly
(176/176/178 correct), so it is a property of the model and this eval, not of
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

Severity scales with how little headroom you have: Pro EP8 held 189 sequences,
Flash EP8 520, Flash EP4 the full 1280 — at EP4 the bug was latent, not absent.
All shipped cells now pin `CAPTURE_SIZES`; pinning costs native ~3% to batch
padding, so **decode-1k numbers from before 2026-07-25 are not comparable with
these.**

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

Provenance for every number: jobs 2337199, 2337204, 2337438, 2337473, 2337476,
2337487, 2337549, 2337550. The full chronological log lives on the `vllm-pr`
branch in `vllm_e2e/RUNS.md` (runs 43-50).
