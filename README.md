# MoE-EP Benchmark

Benchmarks for the FlashInfer `moe_ep` mega-MoE expert path on Blackwell
(sm_100), at two levels:

* **Microbenchmark** — drives the kernels directly, one process per GPU
  (`torch.multiprocessing`), mirroring DP=N + EP (TP=1). No vLLM, no
  checkpoints. Isolates kernel work from integration overhead.
* **vLLM end-to-end** — the same backends behind a patched vLLM 0.25.1 serving
  DeepSeek-V4-Flash, measuring what the kernel win is actually worth.

## Where to start

| If you want to… | Go to |
|---|---|
| Reproduce the recorded numbers from scratch, on any machine | [RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) |
| Verify `fi moe_ep`, or integrate it into a serving framework (SGLang, TRT-LLM, …) | [SKILL.md](SKILL.md) — the verification ladder and the framework-integration contract checklist |
| Just run the microbenchmark on a node you already have | §1 below, or [RUNBOOK.md](RUNBOOK.md) |
| Just run the e2e benchmark | §2 below |
| Add a new mega-kernel backend | `docs/design_docs/moe_ep_runbook.md` in the flashinfer checkout |

## Repository map

```
run.sh, run_sweep.sh          microbenchmark drivers (§1)
bench_moe_ep_*.py             the three microbenchmark sections
model_shapes/                 per-model-geometry sweep -> RESULTS.md (§1)
results/                      microbenchmark CSVs and logs
RUNBOOK.md                    microbenchmark: "just run it"
RUNBOOK_REPRO.md              from-scratch reproduction, both levels
SKILL.md                      verification ladder + integration checklist
vllm_e2e/                     everything end-to-end (§2)
tests/                        harness self-tests
```

---

# 1. Microbenchmark

## Sections

| Section | Script | Backends |
|---------|--------|----------|
| `vllm_split` | `bench_moe_ep_nonmega.py` | DeepEP dispatch/combine + local compute (`deepgemm`, `trtllm`) |
| `vllm_mega` | `bench_moe_ep_vllm_mega.py` | vLLM fused mega MoE (`vllm_deep_gemm_mega`) |
| `fi_mega` | `bench_moe_ep_mega.py` | FlashInfer fused mega (`deep_gemm_mega`, `mxfp8_cutedsl`, `nvfp4_cutedsl`) |

## Two ways to run it

**Ad-hoc, one geometry** — `run.sh` / `run_sweep.sh`, driven by the environment
variables below. Good for investigating a single shape.

**Per-model geometries, batch** — `model_shapes/`, which sweeps the MoE
geometries of real models (`shapes.tsv`: DeepSeek V3/V4, Kimi K2.6, gpt-oss,
Qwen3.5) across five FlashInfer variants and renders
[model_shapes/RESULTS.md](model_shapes/RESULTS.md). Submits its own SLURM job
per shape:

```bash
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh
python model_shapes/make_tables.py model_shapes/results/model_shapes_*.csv
```

`GPUS` sets the world size, and world size = DP = EP — so `GPUS=8` is EP8, a
different regime from the recorded EP4 numbers, not a refresh of them. Write it
to a separate `OUT_DIR`: `make_tables.py` keys cells on
`(geometry, tokens/rank, variant)` and ignores the CSV's `gpus` column, so
mixing EP sizes in one directory silently overwrites cells. See
[RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §4b.

## Output files

Results land in `results/` (override with `OUT_DIR`).

**`run.sh`** — one row per variant at a fixed `TOKENS`:

```
results/bench_<stamp>_vllm_split.{log,csv}
results/bench_<stamp>_vllm_mega.{log,csv}
results/bench_<stamp>_fi_mega.{log,csv}
```

**`run_sweep.sh`** — same variants swept over sequence length; rows append per section:

```
results/sweep_<stamp>_vllm_split.{log,csv}
results/sweep_<stamp>_vllm_mega.{log,csv}
results/sweep_<stamp>_fi_mega.{log,csv}
```

`results/archive_pre20260715_broken_nvfp4_layout/` holds pre-2026-07-15 data
measured against a broken nvfp4 weight layout — kept deliberately, with a README
explaining the bug. Do not merge it into current tables.

## `run.sh`

Run from the repo root (or adjust paths).

```bash
# Full suite (all three sections)
./moe_ep_benchmark/run.sh

# One section only
SECTION=vllm_split ./moe_ep_benchmark/run.sh
SECTION=vllm_mega  ./moe_ep_benchmark/run.sh
SECTION=fi_mega    ./moe_ep_benchmark/run.sh

# Decode path (DeepEP low-latency all2all)
ALGO=ll ./moe_ep_benchmark/run.sh

# Prefill path (DeepEP high-throughput; default)
ALGO=ht ./moe_ep_benchmark/run.sh

# Include input activation quant in split-path timing (HT only)
EXCLUDE_QUANT=0 ./moe_ep_benchmark/run.sh

# Split path only: skip vllm_mega when running all sections
VLLM_MEGA=0 ./moe_ep_benchmark/run.sh

# Split path only: single expert backend
EXPERTS_LIST=deepgemm ./moe_ep_benchmark/run.sh
EXPERTS_LIST=trtllm   ./moe_ep_benchmark/run.sh

# FI mega only: subset of backends
MEGA_LIST=deep_gemm_mega ./moe_ep_benchmark/run.sh
MEGA_LIST="mxfp8_cutedsl nvfp4_cutedsl" SECTION=fi_mega ./moe_ep_benchmark/run.sh

# Problem size / hardware
GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./moe_ep_benchmark/run.sh
TOKENS=64 NUM_EXPERTS=256 TOPK=8 HIDDEN=7168 INTER=2048 ./moe_ep_benchmark/run.sh

# Timing
WARMUP=20 ITERS=50 ./moe_ep_benchmark/run.sh

# Python interpreters (vLLM vs FlashInfer envs)
PYTHON=python FI_PYTHON=python ./moe_ep_benchmark/run.sh

# Custom output directory
OUT_DIR=/tmp/moe_bench ./moe_ep_benchmark/run.sh
```

**Tip:** To compare local expert GEMMs under the same all-to-all, use `ALGO=ht` so DeepGEMM and TRT-LLM both run on DeepEP high-throughput.

## `run_sweep.sh`

Sweeps `TOKENS` (tokens per rank) and reuses all `run.sh` knobs. Default points are powers of two from 1 to 8192.

```bash
# Full sweep (all sections, default seq lengths)
./moe_ep_benchmark/run_sweep.sh

# One section only
SECTION=vllm_split ./moe_ep_benchmark/run_sweep.sh
SECTION=vllm_mega  ./moe_ep_benchmark/run_sweep.sh
SECTION=fi_mega    ./moe_ep_benchmark/run_sweep.sh

# Explicit sweep points
SEQ_LENS="1 8 64 512 4096" ./moe_ep_benchmark/run_sweep.sh

# Auto-generated powers of two in a range (default: 1 .. 8192)
SEQ_LEN_MIN=1 SEQ_LEN_MAX=4096 ./moe_ep_benchmark/run_sweep.sh

# Combined with run.sh options
ALGO=ht GPUS=4 EXCLUDE_QUANT=0 ./moe_ep_benchmark/run_sweep.sh
SECTION=fi_mega MEGA_LIST=deep_gemm_mega SEQ_LENS="8 128 1024" ./moe_ep_benchmark/run_sweep.sh
VLLM_MEGA=0 SEQ_LENS="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192" ./moe_ep_benchmark/run_sweep.sh
```

## Environment reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SECTION` | `all` | `vllm_split`, `vllm_mega`, `fi_mega`, or `all` |
| `GPUS` | `4` | World size (= DP = EP) |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPU device list |
| `ALGO` | `ht` | `ht` (prefill) or `ll` (decode); split path only |
| `TOKENS` | `8` | Tokens per rank (`run.sh` only; sweep overrides) |
| `NUM_EXPERTS` | `256` | Total experts |
| `TOPK` | `8` | Top-k routing |
| `HIDDEN` | `7168` | Hidden size |
| `INTER` | `2048` | Expert intermediate size |
| `WARMUP` | `20` | Warmup iterations |
| `ITERS` | `50` | Timed iterations |
| `EXPERTS_LIST` | `deepgemm trtllm` | Split-path expert backends |
| `MEGA_LIST` | `deep_gemm_mega mxfp8_cutedsl nvfp4_cutedsl` | FI mega backends |
| `MEGA_KNOBS` | *(unset)* | Cutedsl kernel knobs: `auto` = online autotune at first forward; a JSON dict (e.g. `'{"flag_batch": 4}'`) = explicit override; unset = shim heuristic |
| `MEGA_TIMING` | `e2e` | Timed region: `e2e` = full FI forward from a barrier-cold start (arg prep + kernel + output copy); `e2e_pipelined` = same full forward but iters back-to-back enqueued (steady-state serving shape); `kernel` = tester-parity bare kernel launch (cutedsl backends; matches `cutedsl_megamoe tester perf_run`) |
| `MEGA_IKR` | `0` | `1` = `in_kernel_fc2_reduce` (in-flight REDG top-k combine) on the cutedsl backends; CSV `compute_kernel` gains a `+ikr` suffix. In `kernel` mode the thunk includes the required per-launch `output.zero_()` |
| `MEGA_COMBINE_DTYPE` | `bf16` | Cross-rank combine wire format for `nvfp4_cutedsl`: `mxfp8` (`32e4m3xe8m0`, 2x less combine traffic) or `nvfp4` (`16e2m1xbf16`, 4x less); CSV suffix `+combine_<fmt>`. Mutually exclusive with `MEGA_IKR=1` |
| `MEGA_ACC` | `1` | `0` skips the accuracy-loss pass. When on, one un-timed forward is compared against an fp32 dense-MoE ground truth over all experts (peer weights regenerated from seeds); the global rel-L2 percentage is printed separately from the latency numbers and lands in the CSV `acc_loss_pct` column |
| `VLLM_MEGA` | `1` | Include `vllm_mega` when `SECTION=all` |
| `EXCLUDE_QUANT` | `1` | Exclude input quant from split-path timing |
| `PYTHON` | `python` | Interpreter for vLLM scripts |
| `FI_PYTHON` | `$PYTHON` | Interpreter for FlashInfer mega script |
| `OUT_DIR` | `moe_ep_benchmark/results` | Output directory |
| `SEQ_LENS` | *(auto)* | Space-separated sweep points (`run_sweep.sh`) |
| `SEQ_LEN_MIN` | `1` | Sweep range start (`run_sweep.sh`) |
| `SEQ_LEN_MAX` | `8192` | Sweep range end (`run_sweep.sh`) |

---

# 2. vLLM end-to-end

Everything lives in [`vllm_e2e/`](vllm_e2e/). vLLM 0.25.1 is patched in place
(`patch_0251/`) to register two FlashInfer backends alongside the native one:

| `--moe-backend` | megakernel | consumes |
|---|---|---|
| `deep_gemm_mega_moe` | — (native vLLM) | MXFP4 |
| `flashinfer_moe_ep_mega_deep_gemm` | `deep_gemm_mega` | MXFP4 verbatim |
| `flashinfer_moe_ep_mega_cutedsl` | `nvfp4_cutedsl` | NVFP4 prequantized |

## To reproduce the recorded numbers

**Start here: [RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §5.** It is the from-scratch
path — clone, build the container image, fetch both checkpoints at pinned
revisions, set up the venv, then run the tiers below. Expected numbers and the
SLURM job IDs they came from are in §5d.

The whole sweep in one command, from `vllm_e2e/`:

```bash
cd vllm_e2e && sbatch job_vllm_pr_runbook_sweep.sh
```

That is the script that produced job 2441711 — every number in §5d. It grabs its
own exclusive node, re-applies the patch, runs all four cells against all three
backends, and prints a summary table. It reads `ROOT`, `IMG`, `ROUNDS`, `MODEL`,
`MODEL_NVFP4` and `EXTRA_MOUNTS` from the environment, so another checkout needs
no edit.

## To run pieces of it

All of these go through the container wrapper on a held node —
`JOBID=<id> bash in_container.sh '<command>'`. Tiers, cheapest first:

| Tier | Command | Cost | Catches |
|---|---|---|---|
| 1 | `python test_backend_registration.py` | ~1 min, no model | a backend registered in one file but not the other |
| 2 | `python smoke_infer.py …` + `compare_outputs.py` | ~12 min, 4 GPUs | the backend string not actually reaching a kernel |
| 3 | `python bench_offline.py --workload …` | minutes per cell | throughput / TTFT / ITL |
| gate | `python eval_gsm8k.py …` | ~10 min | cross-checkpoint accuracy fairness |

Tier 1 is not sufficient on its own — it passes even if the run quietly executes
the native path. The proof is the `[fi_moe_ep] ep_rank=…` banner, one line per
rank, in the fi log and absent from the native one.

## Documents in `vllm_e2e/`

| File | What it is |
|---|---|
| `RUNBOOK_VLLM_PR.md` | the PR in detail: what the patch changes, per-tier validation, coverage gaps |
| `RUNBOOK.md` | the e2e benchmark itself — backends, workloads, knobs |
| `RUNBOOK_8GPU_SM100.md` | self-contained 8-GPU SM100 procedure. **Written, not executed** — everything measured so far is 4x GB200 |
| `RUNS.md` | chronological log of every run, with what each one settled |
| `FINDINGS.md` | conclusions that outlived the run that produced them |
| `COMM.md`, `CUTEDSL_COMM_EXPERT_NOTES.md` | communication-path design notes |
| `results/fidg_tax_analysis.md` | where the fi_dg-vs-native overhead goes |

## Patch mechanics

`patch_0251/apply.sh` is idempotent: it copies the PR's `model.py` in, installs
`vllm/utils/flashinfer_moe_ep.py`, inserts the two backend strings into the
`MoEBackend` literal in `vllm/config/kernel.py` (required — `KernelConfig`
rejects unknown values before the model is constructed), and fails if anything
still imports the pre-move `fi_utils`. `patch_0251/reset.sh` restores the
pristine vLLM files.

---

# 3. Scripts and the job archive

## Core scripts

| Script | Purpose |
|---|---|
| `run.sh`, `run_sweep.sh` | microbenchmark drivers (§1) |
| `model_shapes/submit_jobs.sh` | one SLURM job per model geometry |
| `model_shapes/job_payload.sh` | its in-container payload: editable install, DSL pin, guard, sweep |
| `model_shapes/run_model_shapes.sh` | the sweep itself, five variants per shape |
| `vllm_e2e/in_container.sh` | run one command in the container on a held node — the entry point for everything e2e |
| `vllm_e2e/setup_container.sh` | one-time venv: vLLM 0.25.1 + editable flashinfer + patch |
| `vllm_e2e/job_vllm_pr_runbook_sweep.sh` | the full four-cell, three-backend sweep (§2) |
| `vllm_e2e/patch_0251/apply.sh`, `reset.sh` | apply / revert the vLLM patch |
| `vllm_e2e/launch_detached.sh` | run a bench that survives the parent shell dying |

## One-off investigation drivers

`vllm_e2e/orchestrate_*.sh` (14 scripts) each drove one specific investigation
and are kept as the record of how a given result was produced. They are not part
of any routine path and no document references them; read the header comment
before reusing one. Examples: `orchestrate_gap_nsys.sh` (steady-state gap
attribution via nsys), `orchestrate_decode_retune.sh` (24-candidate decode
tuning sweep), `orchestrate_combine_wire.sh` (quantized combine-wire
experiment), `orchestrate_shape_determinism.sh` (fi_dg batch-shape
nondeterminism).

Also in this category: `bench_throughput.sh`, `run_offline_matrix.sh`,
`profile_matrix.sh`, and the `analyze_*.py` nsys post-processors.

## `logs_fi/` — the job archive

**Not in this repo.** It sits next to the checkouts at `$ROOT/logs_fi/` and is
machine-local, ~10 MB: roughly 125 SLURM logs plus the 23 job scripts that
produced them. It is the provenance for the recorded numbers — notably
`vllm_pr_sweep_2441711.log` (every e2e figure) and
`vllm_pr_backends_2439811.log` (the 8/8-exact smoke).

The job scripts there fall into three groups:

* `job_moe_ep_452_*.sh` — microbenchmark and test sweeps at the pinned DSL 4.5.2
* `job_moe_ep_dsl*_bench.sh` — DSL-version comparisons (4.5.3, 4.6.0, 4.7.0.dev)
* `job_vllm_*.sh`, `run_*_20260714.sh` — e2e cells and dated investigation runs

Only `job_vllm_pr_runbook_sweep.sh` has been promoted into the repo, because it
is the one the runbooks tell you to run. The rest are historical; if you need to
re-run one, copy it in and de-hardcode `ROOT` the same way.
