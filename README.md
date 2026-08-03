# moe_ep benchmarks — 1x4 SM103 (GB300) reproduction

Reproduction branch (`vllm_repro_4_gpu_sm103`): the FlashInfer `moe_ep`
mega-MoE path on one 4-GPU Blackwell Ultra (GB300, sm_103) node, at two
levels, plus the vLLM patch and the numbers to check yourself against.
Ported from the 1x8 B200 branch (`vllm_repro_8_gpu_v2`); world size is 4
everywhere here, and a different world size is a different measurement.

**Start with [expected_results_1x4_sm103.md](expected_results_1x4_sm103.md)**
— this branch's numbers. [expected_results.md](expected_results.md) is the
1x8 B200 reference, with the tolerances and the two failure modes that
produce plausible-but-wrong results (which apply here unchanged).

## What reproduces

| | what | how | ~time |
|---|---|---|---|
| Kernel microbenchmark | cutedsl vs deep_gemm_mega at DSV4 shapes. No vLLM, no checkpoints. EP4. | [RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §2 | 20 min |
| vLLM e2e, Flash | 4 cells x 3 backends, TP4/EP4 | `vllm_e2e/job_vllm_pr_runbook_sweep_ep4.sh` | ~1.5 h |
| Accuracy gate | GSM8K, Flash, both checkpoints, TP4 | RUNBOOK §4b one-liners on the hold node | 35 min |

The `*_ep8.sh` / `*_pro.sh` / serving job scripts inherited from the 1x8
branch are **not ported** — they assume an 8-GPU node and the EP8 knob cache.
V4-Pro and serving mode are unmeasured at 1x4.

Setup — container, venv, patch, checkpoints — is
[RUNBOOK_REPRO.md](RUNBOOK_REPRO.md) §1. It is the single runbook, in four
sections: prep (§1), microbenchmark (§2), e2e throughput (§3), accuracy (§4).

## Layout

```
expected_results.md        the numbers, tolerances, failure modes
RUNBOOK_REPRO.md           the runbook: §1 prep, §2 micro, §3 e2e, §4 accuracy
run.sh, run_sweep.sh       microbenchmark launchers   (GPUS=4 => EP4)
bench_moe_ep_*.py          microbenchmark bodies
plot*.py, tests/          chart rendering, dense-reference correctness test
model_shapes/              per-shape kernel sweep + results_ep4/ (GB300; results_ep8/ = B200 reference)
vllm_e2e/
  patch_0251/              the vLLM patch (apply.sh / reset.sh)
  bench_offline.py         the e2e throughput harness (offline, in-process)
  serving_payload.sh       serving-mode harness: vllm serve + vllm bench serve
  eval_gsm8k.py            accuracy gate (records the checkpoint it loaded)
  smoke_infer.py           routing smoke + logprobs
  compare_outputs.py       logprob diff between two smoke runs
  test_backend_registration.py   tier-1 config checks, no model
  job_*.sh                 the five SLURM jobs above
  setup/                   checkpoint downloads (4) + knob retune (2)
  results/                 only the JSONs expected_results.md cites
```

## Running the benchmarks

Setup first — container image, venv, vLLM patch, checkpoints — is
RUNBOOK_REPRO.md §1; nothing below works without it. All submit scripts take
`ROOT`/`ACCOUNT`/`PARTITION`/`IMG` overrides from the environment.

**Microbenchmark** (no vLLM, no checkpoints; RUNBOOK §2):

```bash
# full model-shape sweep: one SLURM job per shape in shapes.tsv (~20 min each,
# parallel nodes), CSVs land in model_shapes/results_ep4/ (override OUT_DIR)
ACCOUNT=<account> PARTITION=<partition> IMG=<flashinfer-ep image> \
    bash model_shapes/submit_jobs.sh

# turn the CSVs into the per-shape markdown tables — works standalone on the
# shipped results_ep4/ CSVs, no GPU needed. One table per shape:
#   | tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
# (p50 us; fp4-family cells carry a speedup-vs-dg ratio)
python model_shapes/make_tables.py \
    model_shapes/results_ep4/model_shapes_*.csv -o RESULTS.md

# one cell by hand, inside the container on a 4-GPU node
MEGA_LIST=nvfp4_cutedsl TOKENS=512 SECTION=fi_mega bash run.sh
SECTION=fi_mega bash run_sweep.sh          # token sweep, one backend list
```

Resubmit a single shape to fill gaps (`SHAPE_LIST=qwen3_5_397b bash
model_shapes/submit_jobs.sh` — compiles are cached, later CSVs win). Two
transient failure modes seen in practice, both fixed by resubmitting the
shape: a pip network error installing cutlass-dsl (the DSL guard aborts the
job), and an editable-install race when many jobs start at once
(`ModuleNotFoundError: flashinfer`, every cell warns and the CSV is empty).

**vLLM e2e** (RUNBOOK §3-§4; needs the §1 venv, patch and checkpoints):

```bash
cd vllm_e2e
sbatch -A <account> -p <partition> job_vllm_pr_runbook_sweep_ep4.sh   # Flash, 4 cells x 3 backends, TP4/EP4, ~1.5 h
# accuracy gate: RUNBOOK §4a/§4b one-liners with TP=4 on the hold node
```

(The `*_ep8.sh` / `*_pro.sh` / serving jobs are the unported 1x8 scripts —
see "What reproduces".)

The two `serving` jobs are the server+client counterpart of the offline
sweeps, on the **same four workloads**: one process runs
`vllm serve --moe-backend <be>` per (cell, backend), a second drives it with
`vllm bench serve` (random dataset, fixed lengths, round 0 discarded, median
of the timed rounds). RUNBOOK §3g.

Run all three backends of a cell in one session, and check the
`[fi_moe_ep] ep_rank=…` banner before believing any fi number (see below).

## Tuning

The cutedsl kernel picks its schedule (mma tile, cluster shape, token-back
mode, …) through three tiers, highest priority first:

1. **Explicit knobs** — `MEGA_KNOBS='{"mma_tiler_mnk": [256,128,256], ...}'`
   (microbenchmark only; lists become tuples).
2. **Knob cache** — a JSON file keyed by (device, dtype, world size, geometry,
   combine wire, token bucket), pointed to by `FLASHINFER_MOE_EP_KNOB_CACHE`
   (default `~/.cache/flashinfer/moe_ep_knob_cache.json`; `0` disables). An
   untuned geometry falls through — the cache never borrows a neighbour's
   knobs.
3. **Built-in heuristic** — token-count-bucketed profiles hardcoded in the
   flashinfer checkout at
   `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/tuner.py`
   (`default_knobs`: <512 / 512-1023 / 1024-2047 / >=2048 for nvfp4, two
   buckets for mxfp8). Editing the heuristic means editing that file, not
   this repo.

Who uses what: the **microbenchmark tables resolve tier 3** (the sweep sets no
cache and no `MEGA_KNOBS`) — deliberately, so they measure what an untuned
user gets. The **e2e sweeps resolve tier 2** from caches this branch ships:
`vllm_e2e/results/knob_cache_ep8.json` (Flash) and `knob_cache_pro_ep8.json`
(Pro). Mind the gap when comparing levels: at deepseek_v4_pro @ 512 tok/rank
the heuristic runs ~394 us where the tuned schedule reaches ~377 us — parity
with deep_gemm (jobs 2340421/2340445).

To retune:

```bash
# e2e caches (only needed if your geometry or world size differs from EP8 —
# synthetic weights, no checkpoint, ~10 min each; see RUNBOOK §3a)
ROOT=... IMG=... JOBID=<hold job> bash vllm_e2e/setup/tune_knobs_flash_ep8.sh
ROOT=... IMG=... JOBID=<hold job> bash vllm_e2e/setup/tune_knobs_pro_ep8.sh

# microbenchmark, online: autotune at the first forward, winner kept for the
# session and recorded to the knob cache by rank 0
MEGA_KNOBS=auto SECTION=fi_mega bash run.sh

# microbenchmark, offline sweep for one shape/token point (SLURM, one node):
# run_tune512.sh = curated sweep; _schedule / _custom = wider searches
SHAPE_NAME=deepseek_v4_pro TOKENS=512 bash model_shapes/submit_tune512.sh
DRIVER=run_tune512_custom.sh bash model_shapes/submit_tune512.sh
```

Tune-job winners land in `model_shapes/results_tune512/knob_cache_*.json`;
nothing consumes them unless you point `FLASHINFER_MOE_EP_KNOB_CACHE` at one.
A sweep run against a cache is a *different measurement* than the published
heuristic-resolved tables — keep it in a separate `OUT_DIR`.

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

### The workloads, exactly

#### Microbenchmark (`model_shapes/` tables, expected_results §3)

One problem = one MoE layer forward at a fixed **tokens/rank** — the number
of tokens each of the 8 DP ranks feeds into expert dispatch. No batch/seq
distinction exists at this level; tokens/rank is the whole problem size.

| parameter | value |
|---|---|
| parallelism | DP8 / EP8 / TP1 — one process per GPU, world size = EP |
| geometry | per shape from `model_shapes/shapes.tsv` (hidden / moe_inter / experts / top-k), e.g. V4-Flash 4096/2048/256/top-6 |
| tokens/rank sweep | 8, 64, 512, 1024, 2048, 4096, 8192 (`SEQ_LENS`) |
| activations | `randn` bf16, per-rank seed |
| routing | top-k over `randn` scores, per-rank seed — uniform expert load, no capacity dropping |
| weights | synthetic, deterministically seeded per expert (no checkpoint) |
| knobs | tier-3 heuristic (`MEGA_KNOBS` and knob cache unset) |

Timing: per tokens/rank point, 20 warmup + 50 timed iterations; each
iteration is one full FI forward bracketed by a CUDA event pair on the
stream; the tables report the **p50 across the 50 iterations** of the
`e2e_pipelined` region (iterations enqueued back-to-back, no per-iteration
barrier — see the region definitions above). `e2e_us_min` is kept alongside
p50 in the CSV as the noise tell (expected_results §5).

#### Offline e2e (`bench_offline.py`, expected_results §1–§2)

One engine boot per (cell, backend); `llm.generate()` over a fixed prompt
set, repeated. Common to all cells: TP8/EP8/DP1, kv-cache fp8, block 256,
prefix caching **off**, CUDA graphs on (`ENFORCE_EAGER=0`), greedy sampling
(`temperature=0`), `ignore_eos`, prompts = random token ids at seed 0 (same
prompts every round and backend).

| cell | ISL:OSL | requests | concurrency cap | max batched tokens | capture sizes (max) | max-model-len | timed rounds |
|---|---|---|---|---|---|---|---|
| prefill-8k | 1024:1 | 256 | engine default | 8192 | 256,2048,4096,8192 (8192) | 4096 | 3 |
| decode-1k | 128:256 | 1024 | `max_num_seqs=1024` | 4096 | 256,1024,2048,4096 (4096) | 4096 | 3 |
| 100K ISL / 1K | 100000:1024 | 32 | `max_num_seqs=32` | 8192 | 32,256,2048,8192 (8192) | 102400 | 2 |
| 32K ISL / 32 | 32768:32 | 32 | `max_num_seqs=32` | 8192 | 32,256,2048,8192 (8192) | 33792 | 3 |

(The two interactivity cells also set `gpu_memory_utilization=0.93` and
`REQUIRE_LATENCY=1`.)

Timing: each round is one `llm.generate()` over the full prompt set,
bracketed by `time.perf_counter()` wall clock — no profiler, no
instrumentation inside the engine. **Total tok/s = (requests x ISL + output
tokens) / elapsed.** Round 0 is a warmup, kept in the JSON but excluded from
the headline; the headline is the **median across the timed rounds**.
TTFT/ITL come from vLLM's own per-request `RequestOutput.metrics` (engine
monotonic timestamps; ITL = (last_token_ts - first_token_ts)/(n-1)):
percentiles are taken over all requests within a round, then the headline
takes the median across rounds of each percentile.

#### Serving e2e (`serving_payload.sh`, expected_results §2b)

Same four cells, same per-cell engine settings — but the engine runs behind
`vllm serve --moe-backend <be>` (one boot per cell x backend) and the
measurement is `vllm bench serve` over HTTP `/v1/completions` from a second
process on the same node. Client side: `--dataset-name random` with
`--random-range-ratio 0` (fixed lengths), seed 0, `--ignore-eos`,
`--max-concurrency` = the request count (all in flight, like the offline
scheduler sees), request rate inf.

| cell | ISL:OSL | requests = concurrency | server flags beyond the common set |
|---|---|---|---|
| prefill-8k | 1024:1 | 256 | as offline pre8k row above |
| decode-1k | 128:256 | 1024 | as offline dec1k row above |
| 100K ISL / 1K | 100000:1024 | 32 | as offline lc100k row above |
| 32K ISL / 32 | 32768:32 | 32 | as offline ctx32k row above |

Timing: entirely client-side wall clock — the duration from first request
sent to last response finished, with per-request TTFT and ITL measured on
the streaming HTTP responses. So serving numbers additionally include
tokenize/detokenize, HTTP and scheduling gaps, which is the point of the
level. Round 0 (a full client pass) is discarded as warmup; the headline is
the **median total token throughput across the timed rounds** (3 per cell,
2 for the 100K cell), with min..max spread printed — required because the
native decode baseline drifts run-over-run by more than the fi-vs-native
delta (expected_results §2b).

Before any round counts, the payload checks routing per (cell, backend):
eight `[fi_moe_ep] ep_rank=` banner lines in the fi server logs, zero in the
native log.

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
