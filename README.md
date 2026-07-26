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
