# CLAUDE.md

Benchmarks for the FlashInfer `moe_ep` mega-MoE path on Blackwell (sm_100), at
two levels: a kernel microbenchmark (no vLLM, no checkpoints) and a vLLM 0.25.1
end-to-end benchmark on DeepSeek-V4-Flash.

[README.md](README.md) is the map — §1 microbenchmark, §2 e2e, §3 scripts.
[RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) is the from-scratch reproduction path and
the most current document in the repo; prefer it over the older runbooks where
they disagree.

## Orientation

* Two levels, two directories: microbenchmark at the top level
  (`run.sh`, `bench_moe_ep_*.py`, `model_shapes/`), e2e in `vllm_e2e/`.
* Numbers are provenanced by SLURM job ID. When quoting or updating a number,
  carry the job ID with it; `vllm_e2e/RUNS.md` is the chronological log.
* The measured configuration is 1x4 GB200, EP4, vLLM 0.25.1, cutlass-dsl 4.5.2.
  Anything else is a different measurement, not a better estimate of the same
  one.

## Invariants worth knowing before changing anything

**`FI_MOE_EP` is a hard error.** Any non-empty `FI_MOE_EP` or
`FI_MOE_EP_MEGAKERNEL` aborts at startup — including `FI_MOE_EP=0` and
including on the native backend. The `main` branch still uses that old opt-in;
`vllm-pr` is the branch everything current lives on.

**Do not unpin cutlass-dsl.** The CuteDSL codegen is version-sensitive (34-54%
slower pre-4.5.2), so an unpinned upgrade makes a sweep unattributable. The venv
gets 4.5.2 from vLLM's own pin; `model_shapes/job_payload.sh` installs and
asserts it explicitly because it does not use the venv. The container image
pins 4.5.0 and is *not* the source of truth.

**World size = DP = EP** (`run.sh:33`). `GPUS=8` is EP8, a different regime from
the recorded EP4 numbers.

**`make_tables.py` ignores the `gpus` column.** It keys cells on
`(geometry, tokens/rank, variant)`, so CSVs from different world sizes in one
directory silently overwrite each other, and `RESULTS.md` is tracked. Use a
separate `OUT_DIR` per EP size.

**`results/archive_pre20260715_broken_nvfp4_layout/`** is deliberately retained
bad data (broken nvfp4 weight layout — faster but numerically wrong). Never
merge it into current tables.

**Benchmarks are timing-sensitive.** Do not edit kernel sources or harness
scripts while a sweep is in flight. Round 0 of every cell is a warmup and is
excluded from medians — do not remove it; prefix caching is off for the same
reason.

## Verifying an e2e change

Tiers, cheapest first: `test_backend_registration.py` (~1 min, no model) ->
`smoke_infer.py` + `compare_outputs.py` (~12 min) -> `bench_offline.py` cells ->
`eval_gsm8k.py` as the cross-checkpoint accuracy gate.

Tier 1 is not sufficient: it passes even when the run quietly executes the
native path. The proof a backend string reached a kernel is the
`[fi_moe_ep] ep_rank=…` banner, one line per rank in the fi log and **absent**
from the native log.

`smoke_infer.py` does not resolve the checkpoint per backend the way
`bench_offline.py` and `eval_gsm8k.py` do — pass `--model $MODEL_NVFP4`
explicitly for the cutedsl backend.

## Conventions

* Shell scripts take `ROOT`/`OUT_DIR`/`MODEL` style overrides from the
  environment rather than hardcoding paths; keep it that way when adding one.
* `vllm_e2e/orchestrate_*.sh` are one-off investigation drivers kept as
  provenance, not routine paths. Read the header before reusing one.
* Job scripts and SLURM logs live outside the repo in `$ROOT/logs_fi/`. Promote
  one into the repo only when a runbook tells people to run it.
