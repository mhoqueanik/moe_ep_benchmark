# moe_ep benchmarks — 1x8 SM100 reproduction

Reproduction branch (`vllm_repro_8_gpu`): the FlashInfer `moe_ep` mega-MoE
path on one 8-GPU Blackwell node, at two levels, plus the vLLM patch and the
numbers to check yourself against. Nothing else — development history lives
elsewhere in the repo.

**Start with [expected_results.md](expected_results.md)** — the numbers, the
tolerances, and the two failure modes that produce plausible-but-wrong results.

## What reproduces

| | what | how | ~time |
|---|---|---|---|
| Kernel microbenchmark | cutedsl vs deep_gemm_mega at DSV4 shapes. No vLLM, no checkpoints. | [RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §2 | 20 min |
| vLLM e2e, Flash | 4 cells x 3 backends, EP8 | `vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` | 1 h |
| vLLM e2e, Pro | same cells, V4-Pro | `vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` | 2 h |
| Accuracy gate | GSM8K, both models, both checkpoints | `vllm_e2e/job_gsm8k_flash_pro.sh` | 35 min |

Setup — container, venv, patch, checkpoints — is
[RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §1. It is the single runbook, in four
sections: prep (§1), microbenchmark (§2), e2e throughput (§3), accuracy (§4).

## Layout

```
expected_results.md        the numbers, tolerances, failure modes
RUNBOOK_REPRO.md           the runbook: §1 prep, §2 micro, §3 e2e, §4 accuracy
run.sh, run_sweep.sh       microbenchmark launchers   (GPUS=8 => EP8)
bench_moe_ep_*.py          microbenchmark bodies
plot*.py, tests/          chart rendering, dense-reference correctness test
model_shapes/              per-shape kernel sweep + results_ep8/ (the CSV §2 cites)
vllm_e2e/
  patch_0251/              the vLLM patch (apply.sh / reset.sh)
  bench_offline.py         the e2e throughput harness
  eval_gsm8k.py            accuracy gate (records the checkpoint it loaded)
  smoke_infer.py           routing smoke + logprobs
  compare_outputs.py       logprob diff between two smoke runs
  test_backend_registration.py   tier-1 config checks, no model
  job_*.sh                 the three SLURM jobs above
  setup/                   checkpoint downloads (4) + knob retune (2)
  results/                 only the JSONs expected_results.md cites
```

## Benchmark levels and timing methodology

Two levels, two meanings of "benchmark":

* **Microbenchmark** (`run.sh`, `bench_moe_ep_*.py`, `model_shapes/`): the MoE
  layer alone — synthetic tokens, deterministic per-rank weights, no vLLM, no
  checkpoints. One process per GPU (DP8/EP8/TP1). Answers "what does one
  forward of the mega path cost at this geometry."
* **vLLM e2e** (`vllm_e2e/`): the same kernels inside a real engine
  (TP8/EP8/DP1), measured as request throughput. Answers "does the kernel win
  survive the serving stack."

Within the microbenchmark, `MEGA_TIMING` picks the timed region:

* **`e2e`** (default) — the full FI forward path (arg prep, workspace reset,
  kernel, sync, output copy), every iteration launched from a global barrier
  with an idle GPU. Cold-start latency: what a single isolated call costs.
* **`e2e_pipelined`** — the same full forward, but iterations enqueued
  back-to-back with no per-iteration barrier or sync. Steady-state latency,
  like consecutive layers in a serving pipeline. This is the methodology
  behind the `model_shapes/` tables. `e2e` minus `e2e_pipelined` isolates the
  barrier-cold collective start skew.
* **`kernel`** — a prebuilt bare kernel launch, back-to-back, 300 MB L2 flush
  outside the event window. Parity with the kernel repo's own tester
  (`cutedsl_megamoe tester/solver.py perf_run`), for comparing against
  kernel-development numbers.

All three are timed with **CUDA events** (median over 50 iters, 20 warmup),
not `torch.profiler`/CUPTI kernel time. That is a deliberate choice, not an
oversight:

* An event pair brackets the stream and counts everything between — including
  the time the megakernel spends spin-waiting on peer ranks at its in-kernel
  symmetric-memory rendezvous. That wait *is* the latency a serving engine
  experiences; a per-kernel device-time sum answers a different question
  (kernel residency) and treats the gaps between a multi-kernel backend's
  launches as free, which is unfair to compare against a fused kernel.
* Profiler instrumentation is not skew-neutral. CUPTI's per-launch
  interception and buffer management add a *variable, per-rank* delay on the
  launch path; the megakernel's duration depends on the relative arrival of
  all 8 ranks, so that jitter is transduced into longer kernels on every rank
  — the tool inflates the quantity it measures. Measured on
  deepseek_v4_pro @ 512 tok/rank (job 2340556): profiler-summed kernel time
  p50 = 541–570 us against 377 us by CUDA events, with the profiler run's
  *minimum* (372 us) matching the event numbers — the gap is
  instrumentation-induced arrival skew, not compute.
* Events are the methodology everyone else already uses for this kernel: the
  kernel repo's tester times per-stage CUDA events the same way
  (`MEGA_TIMING=kernel` mirrors it). The cudnn-frontend SDPA training
  benchmark's profiler-based procedure is sound for single-GPU attention
  kernels, where residency ≈ wall time; it does not transfer to a
  communication-fused multi-GPU kernel.

## Things that will bite you

* **`FI_MOE_EP` is a hard error.** Any non-empty `FI_MOE_EP` or
  `FI_MOE_EP_MEGAKERNEL` aborts at startup — including `FI_MOE_EP=0`, and
  including on the native backend. Backend selection is one `moe_backend`
  string.
* **Do not unpin cutlass-dsl, and do not pair it with a different flashinfer
  branch.** 4.5.2 is vLLM 0.25.1's own pin and the runtime `4_5_2-perf-fix` is
  validated against; the two move together. The CuteDSL codegen is
  version-sensitive — on 4.5.2 *without* that branch's MR!27 mainloop WAR the
  kernels ran 34-54% slower, which is what the 4.6.1 compatibility chain used
  to work around. With the WAR, 4.5.2 matches the old 4.6.1 stack to within
  0.7% and 4.6.1 is unnecessary. `model_shapes/job_payload.sh` asserts the
  version because it does not use the venv.
* **The two levels reach EP8 differently.** The microbenchmark (`run.sh`,
  `model_shapes/`) has no model to shard, so it runs one process per GPU:
  world size = DP = EP, and `GPUS=8` means DP8/EP8/TP1. The vLLM e2e sweeps
  shard one engine instead: **TP8/EP8/DP1**. Expert parallelism is 8 in both —
  that is the axis under test — but they are not the same configuration, and a
  different world size is a different measurement either way.
* **`make_tables.py` ignores the `gpus` column** — it keys on
  `(geometry, tokens/rank, variant)`, so results from two world sizes in one
  directory silently overwrite each other. Use a separate `OUT_DIR` per EP
  size (this branch ships `model_shapes/results_ep8/`).
* **Tier-1 config checks are not sufficient.** They pass even when the run
  quietly executes the native path. The proof a backend string reached a
  kernel is the `[fi_moe_ep] ep_rank=…` banner — one line per rank in the fi
  log, and **absent** from the native log.
* **`smoke_infer.py` does not resolve the checkpoint per backend** the way
  `bench_offline.py` and `eval_gsm8k.py` do. Pass `--model $MODEL_NVFP4`
  explicitly for the cutedsl backend, or you will smoke-test it on the mx
  weights.
* **Never export `MODEL` around `eval_gsm8k.py`** — it overrides the
  per-backend checkpoint and silently disarms the gate. See
  expected_results.md §5.2.

## Provenance

Measured on one 1x8 B200 node, from this branch's own scripts. The numbers
and their tolerances are in [expected_results.md](expected_results.md).
