# moe_ep benchmarks — 1x8 SM100 reproduction

Reproduction-only branch (`vllm_repro_8_gpu`): the FlashInfer `moe_ep`
mega-MoE path on one 8-GPU Blackwell node, at two levels, plus the vLLM patch
and the numbers to check yourself against. Everything not needed to reproduce
lives on `vllm-pr`.

**Start with [expected_results.md](expected_results.md)** — the numbers, the
tolerances, and the two failure modes that produce plausible-but-wrong results.

## Can you run this?

Not from a public checkout. Three of the inputs are not obtainable outside
NVIDIA, so be honest with yourself about this list before budgeting time:

| you need | where it comes from | outside NVIDIA? |
|---|---|---|
| 8x SM100 GPUs, one node, SLURM + pyxis/enroot | your cluster | yes, if you have the hardware |
| `flashinfer-moe_ep` @ `4_5_2-perf-fix` | a fork, see RUNBOOK_REPRO §1a | access-dependent |
| mx checkpoints (V4-Flash, V4-Pro) | `/lustre/share/coreai_dlalgo_ci/artifacts/model/` | **no — internal mirror** |
| NVFP4 casts | HF `nvidia/DeepSeek-V4-*-NVFP4`, via `vllm_e2e/setup/` | yes |
| GSM8K | internal parquet, else auto-downloads from the OpenAI jsonl | yes (fallback) |

The mx checkpoints are the hard blocker: they are the baseline every ratio is
measured against, and `bench_offline.py` compiles their mirror path in as
`DEFAULT_MODEL`. With your own copy of the same weights, point `MODEL` /
`MODEL_NVFP4` (or `--model`) at it and everything else works — but read
expected_results.md §5.2 first, because setting `MODEL` is exactly what
disarms the accuracy gate.

Also cluster-specific and worth overriding: the `#SBATCH --account` /
`--partition` lines in `vllm_e2e/job_*.sh` (pass `sbatch -A <acct> -p <part>`),
and `ROOT`, which every script takes from the environment.

## What reproduces

| | what | how | ~time |
|---|---|---|---|
| Kernel microbenchmark | cutedsl vs deep_gemm_mega at DSV4 shapes. No vLLM, no checkpoints. | [RUNBOOK.md](RUNBOOK.md), `GPUS=8 ./run.sh` | 20 min |
| vLLM e2e, Flash | 4 cells x 3 backends, EP8 | `vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` | 1 h |
| vLLM e2e, Pro | same cells, V4-Pro | `vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` | 2 h |
| Accuracy gate | GSM8K, both models, both checkpoints | `vllm_e2e/job_gsm8k_flash_pro.sh` | 35 min |

Setup — container, venv, patch, checkpoints — is
[vllm_e2e/RUNBOOK_1x8.md](vllm_e2e/RUNBOOK_1x8.md), which defers to
[RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §1-3 for the from-scratch build.

## Layout

```
expected_results.md        the numbers, tolerances, failure modes
RUNBOOK.md                 microbenchmark runbook
RUNBOOK_REPRO.md           from-scratch e2e build + run (setup is §1-3)
run.sh, run_sweep.sh       microbenchmark launchers   (GPUS=8 => EP8)
bench_moe_ep_*.py          microbenchmark bodies
model_shapes/              per-shape kernel table + its EP8 result
vllm_e2e/
  patch_0251/              the vLLM patch (apply.sh / reset.sh)
  bench_offline.py         the e2e throughput harness
  eval_gsm8k.py            accuracy gate (records the checkpoint it loaded)
  smoke_infer.py           routing smoke + logprobs
  compare_outputs.py       logprob diff between two smoke runs
  test_backend_registration.py   tier-1 config checks, no model
  job_*.sh                 the three SLURM jobs above
  setup/                   checkpoint download + knob tuning
  results/                 only the JSONs expected_results.md cites
```

## Things that will bite you

* **`FI_MOE_EP` is a hard error.** Any non-empty `FI_MOE_EP` or
  `FI_MOE_EP_MEGAKERNEL` aborts at startup — including `FI_MOE_EP=0`, and
  including on the native backend. Backend selection is one `moe_backend`
  string.
* **Do not unpin cutlass-dsl.** 4.5.2 exactly; the CuteDSL codegen is 34-54%
  slower before it, which makes an unpinned sweep unattributable.
  `model_shapes/job_payload.sh` asserts it because it does not use the venv.
* **World size = DP = EP** (`run.sh`). `GPUS=8` is EP8; a different world size
  is a different measurement, not a better estimate of the same one.
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

Measured 2026-07-25 on 1x8 B200. Jobs 2337199, 2337204, 2337438, 2337473,
2337476, 2337487, 2337549, 2337550. Full run log and the analysis history are
on `vllm-pr` (`vllm_e2e/RUNS.md`, runs 43-50).
