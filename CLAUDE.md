# CLAUDE.md

Reproduction-only branch (`vllm_repro_8_gpu`) for the FlashInfer `moe_ep`
mega-MoE path on Blackwell (sm_100), on **one 8-GPU node**, at two levels: a
kernel microbenchmark (no vLLM, no checkpoints) and a vLLM 0.25.1 end-to-end
benchmark on DeepSeek-V4-Flash and V4-Pro.

[expected_results.md](expected_results.md) is the source of truth for numbers
and for the two known ways this produces plausible-but-wrong results.
[README.md](README.md) is the repo map. Setup is
[RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) — the single runbook, four sections:
prep (§1), microbenchmark (§2), e2e throughput (§3), accuracy (§4).

Analysis history, one-off investigation drivers, EP4 material and the
chronological run log live on the `vllm-pr` branch. Do not port them here —
this branch is deliberately only what a reproduction needs.

## Orientation

* Two levels, two directories: microbenchmark at the top level
  (`run.sh`, `bench_moe_ep_*.py`, `model_shapes/`), e2e in `vllm_e2e/`.
* Numbers are provenanced by SLURM job ID. When quoting or updating a number,
  carry the job ID with it; `expected_results.md` §6 lists them.
* The measured configuration is 1x8 B200, EP8, vLLM 0.25.1, cutlass-dsl 4.5.2.
  Anything else is a different measurement, not a better estimate of the same
  one.

## Invariants worth knowing before changing anything

**`FI_MOE_EP` is a hard error.** Any non-empty `FI_MOE_EP` or
`FI_MOE_EP_MEGAKERNEL` aborts at startup — including `FI_MOE_EP=0` and
including on the native backend. The `main` branch still uses that old opt-in;
backend selection here is one `moe_backend` string.

**Do not unpin cutlass-dsl.** 4.5.2 is vLLM 0.25.1's own pin and the runtime
the flashinfer `4_5_2-perf-fix` branch is validated against — DSL version and fi
branch move together. The codegen is version-sensitive: on 4.5.2 *without* that
branch's MR!27 mainloop WAR the kernels ran 34-54% slower, which is what the
old 4.6.1 compatibility chain existed to avoid; with the WAR, 4.5.2 matches the
4.6.1 stack within 0.7% (vllm-pr RUNS.md runs 41-42) and 4.6.1 is unnecessary.
The venv gets 4.5.2 from vLLM's pin; `model_shapes/job_payload.sh` installs and
asserts it explicitly because it does not use the venv. The container image
pins 4.5.0 and is *not* the source of truth.

**World size = DP = EP** (`run.sh:33`). `GPUS=8` is EP8, which is what every
number here was measured at.

**`make_tables.py` ignores the `gpus` column.** It keys cells on
`(geometry, tokens/rank, variant)`, so CSVs from different world sizes in one
directory silently overwrite each other. Use a separate `OUT_DIR` per EP size —
this branch ships `model_shapes/results_ep8/`.

**Every cell that sets `MAX_CAPTURE` must pin `CAPTURE_SIZES`.** Without it,
vLLM's CUDA-graph memory profiler reserves ~48 GiB/GPU for the flashinfer
backends against a real cost of ~6 GiB, and the difference comes out of the KV
cache — the engine then runs a fraction of the requested sequences and the
backend looks fast-per-step but slow overall. `dec1k` shipped without it until
2026-07-25. See expected_results.md §5.1.

**Never export `MODEL` around `eval_gsm8k.py`.** `resolve_model` ranks it above
the per-backend NVFP4 default, so it sends every backend to the same checkpoint
and the cross-checkpoint gate silently validates nothing. Pass `--model` per
cell. See expected_results.md §5.2.

**Benchmarks are timing-sensitive.** Do not edit kernel sources or harness
scripts while a sweep is in flight. Round 0 of every cell is a warmup and is
excluded from medians — do not remove it; prefix caching is off for the same
reason. Run all three backends of a cell in one session.

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

`--min-acc 0.93` is calibrated for DSV4-Flash. V4-Pro scores ~0.88 on all three
backends and will fail that gate; the number that actually gates a perf claim
is the native-vs-cutedsl delta, not the absolute.

## Conventions

* Shell scripts take `ROOT`/`OUT_DIR`/`MODEL` style overrides from the
  environment rather than hardcoding paths; keep it that way when adding one.
  `vllm_e2e/setup/*.sh` require `ROOT` explicitly.
* Job scripts belong in the repo only when a runbook tells people to run one.
  Keep one-off investigation drivers on `vllm-pr`.
