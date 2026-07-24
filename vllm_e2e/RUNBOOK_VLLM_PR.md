# Running the vLLM PR by hand: `flashinfer_moe_ep_mega_*` backends

How to reproduce, by hand, the vLLM change that selects the FlashInfer
`moe_ep` expert path with a `moe_backend` string instead of the old
`FI_MOE_EP=1` environment opt-in.

This is the PR-specific companion to `RUNBOOK.md` (which documents the
general vLLM e2e benchmark setup). Read section 0 here, then either follow
this file straight through, or use `RUNBOOK.md` for anything not
PR-specific.

Everything below was executed on 2026-07-24, jobs 2439803 and 2439811, on one
4xGB200 node. Expected outputs are quoted from those runs.

---

## 0. What the PR is, and where the code lives

Three registered MoE backends replace `FI_MOE_EP=1` + `FI_MOE_EP_MEGAKERNEL`:

| config | `moe_backend` string | megakernel | needs |
|---|---|---|---|
| fi_dg | `flashinfer_moe_ep_mega_deep_gemm_sm100` | `deep_gemm_mega` | torch.distributed |
| fi_nvfp4 | `flashinfer_moe_ep_mega_cutedsl_sm100_nvfp4` | `nvfp4_cutedsl` | + NVSHMEM |
| fi_mxfp8 | `flashinfer_moe_ep_mega_cutedsl_sm100_mxfp8` | `mxfp8_cutedsl` | + NVSHMEM |

All three are SM100-only, require expert parallel, and are DeepSeek-V4 only.
The native `deep_gemm_mega_moe` backend is unchanged.

Three repos are involved (all under `ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik`):

| repo | branch | role |
|---|---|---|
| `moe_ep_benchmark` | **`vllm-pr`** | this runbook, the harness, and `patch_0251/` |
| `vllm-fi-moe-ep` | `fi-moe-ep-v4` | the actual vLLM PR commits |
| `flashinfer-2/flashinfer-moe_ep` | `4_5_2-perf-fix` | `flashinfer.moe_ep` runtime + kernels |

`moe_ep_benchmark` `main` deliberately still uses the old `FI_MOE_EP=1`
mechanism, because no released vLLM knows the new backend strings. Use
`main` for normal benchmarking; use `vllm-pr` only for this.

**Two ways to run the PR code.** This runbook covers (a), which is what was
validated:

* **(a) Patch an installed vLLM 0.25.1** — `patch_0251/apply.sh` copies the
  PR's `fi_utils.py`/`model.py` over the installed wheel and registers the
  backend strings in `vllm/config/kernel.py`. Fast, no vLLM build.
* **(b) Build the `vllm-fi-moe-ep` branch from source** — the real PR tree,
  but it sits on a much newer vLLM `main` than 0.25.1 and needs a full source
  build. Not covered here and not validated. Note `patch_0251/fi_utils.py` is
  byte-identical to the PR's copy; `patch_0251/model.py` is a 0.25.1 port and
  differs from the PR's by ~90 lines of unrelated upstream drift.

---

## 1. Get a node and a container

```bash
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e

# 4h hold job on one node (4x GB200)
JOBID=$(sbatch --parsable -A coreai_libraries_cudnn -p batch -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J coreai_libraries_cudnn-fi.vllm_pr.hold \
    --output=$W/logs/hold_%j.log --wrap "sleep 14400")
echo "hold job $JOBID"

# every command below runs through this wrapper
JOBID=$JOBID bash $W/in_container.sh '<command>'
```

`in_container.sh` uses `$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh`,
mounts `$ROOT` read-write and `/lustre/share` read-only (checkpoints), and
sets `FLASHINFER_WORKSPACE_BASE` to a lustre path so the JIT cache survives
the job. Reuses a named container across calls.

## 2. Check out the branch

```bash
git -C $ROOT/moe_ep_benchmark switch vllm-pr
git -C $ROOT/moe_ep_benchmark log --oneline -1     # expect 5d1ed33 or later
```

The flashinfer repo needs `moe_ep` present; `4_5_2-perf-fix` or any later
branch carrying `flashinfer/moe_ep/` works.

## 3. One-time venv setup

Skip if `$W/venv0251` already exists and `RUNBOOK.md` §2 was followed.

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash setup_container.sh'
```

Installs the `vllm==0.25.1` wheel plus the editable flashinfer branch into a
persistent venv on lustre. Ends with a sanity block and a hard
`nvidia-cutlass-dsl` version guard (4.5.2 by default). `FRESH=1` rebuilds
from scratch — needed if the venv predates the 2026-07-22 MR!27 WAR and still
carries DSL 4.6.1.

## 4. Apply the patch

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && bash patch_0251/apply.sh'
```

Expected:

```
kernel.py: registered flashinfer_moe_ep_mega_deep_gemm_sm100, flashinfer_moe_ep_mega_cutedsl_sm100_nvfp4, flashinfer_moe_ep_mega_cutedsl_sm100_mxfp8
patched: .../vllm/models/deepseek_v4/nvidia (backup: model.py.orig)
patched: .../vllm/config/kernel.py (backup: kernel.py.orig)
```

Idempotent — re-running prints `kernel.py: backends already registered
(no-op)`. Pristine files are kept as `model.py.orig` and `kernel.py.orig`.

Registering the strings in `vllm/config/kernel.py` is **required, not
cosmetic**: `KernelConfig` validates `moe_backend` against the `MoEBackend`
literal and rejects an unknown value before the model is ever constructed.
The registration is an in-place insertion into that literal rather than a
whole-file copy, so it does not pin the rest of the config module to the
vLLM version this patch was written against. If vLLM restructures the
literal, `apply.sh` fails loudly and points at itself.

## 5. Tier 1 — config checks (~1 min, no model)

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && python test_backend_registration.py'
```

Covers the `MoEBackend` registration, `KernelConfig` accept/reject/normalise,
the `FI_MOE_EP_BACKENDS` table (spec round-trip, NVSHMEM only for cutedsl,
mega-vs-fi predicates), and the three rejections in
`validate_fi_moe_ep_config` — retired env vars, EPLB, arch floor.

Expected tail:

```
14/14 checks passed
ALL CHECKS PASSED
```

Two `Failed to import Triton kernels ... triton_kernels.matmul_ogs` ERROR
lines are pre-existing container noise, not failures. `VERBOSE=1` prints
tracebacks for any check that does fail.

**Run this whenever you touch backend selection.** It catches the most
likely rot — a backend registered in one file but not the other — in a
minute instead of the ~10 a real smoke costs.

## 6. Tier 2 — end-to-end smoke (~12 min, 4 GPUs)

Tier 1 alone is **not** sufficient: it passes even if `use_fi_mega_moe`
silently stays false and the run quietly executes the native path. So prove
the backend string actually reached the kernel.

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  ENFORCE_EAGER=1 MOE_BACKEND=deep_gemm_mega_moe \
  python smoke_infer.py --tag native --out results/pr_native.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  ENFORCE_EAGER=1 MOE_BACKEND=flashinfer_moe_ep_mega_deep_gemm_sm100 \
  python smoke_infer.py --tag fi_dg --out results/pr_fi_dg.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python compare_outputs.py results/pr_native.json results/pr_fi_dg.json'
```

**Check the bootstrap banner in the fi run's log** — this is the actual proof
the string resolved to a kernel, on every EP rank:

```
[fi_moe_ep] ep_rank=0 world=4 cuda.current_device=0 megakernel=deep_gemm_mega
[fi_moe_ep] ep_rank=1 world=4 cuda.current_device=1 megakernel=deep_gemm_mega
... (one per rank)
```

The **native** run must print no `[fi_moe_ep]` line at all. If it does, the
predicate is mis-routing and the comparison below is meaningless.

Measured 2026-07-24 (job 2439811):

| comparison | exact | mean \|dlogprob\| | recorded band (RUNS.md) |
|---|---|---|---|
| fi_dg vs native | **8/8** | 0.0000 | 3/8, 0.01-0.06 |
| fi_nvfp4 vs native | 1/8 | 0.016-0.13 | 1/8, 0.02-0.20 |

`fi_dg` came out bit-exact against native — better than the July-15 record,
consistent with the run-32 zero-copy fix holding under eager. `fi_nvfp4`
diverges as expected: it dequantizes the mx checkpoint to bf16 and requantizes
to NVFP4, so double quantization moves logprobs slightly.

### Doing all of it in one job

`$ROOT/logs_fi/job_vllm_pr_backend_strings.sh` runs setup, patch, tier 1, and
(with `SMOKE=1`) tier 2 with the banner assertions wired in as hard failures:

```bash
sbatch logs_fi/job_vllm_pr_backend_strings.sh                      # tier 1 only, ~1 min
SMOKE=1 sbatch --time=03:00:00 logs_fi/job_vllm_pr_backend_strings.sh   # + tier 2, ~12 min
```

## 7. Throughput

Unchanged from `RUNBOOK.md` §5 except that a config is now one backend
string. The python entry points read `MOE_BACKEND`; `vllm bench throughput`
takes `--moe-backend`.

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl_sm100_nvfp4 \
  python bench_offline.py --tag fi_nvfp4 --workload decode:128:256 \
    --rounds 3 --out results/pr_bench_fi_nvfp4.json'
```

The label-to-backend mapping lives in one `case` statement per driver script
(`bench_throughput.sh`, `run_offline_matrix.sh`, `profile_matrix.sh`, the
`orchestrate_*.sh` family), so `BACKENDS="native fi_dg fi_nvfp4"` still works
as before.

For headline numbers, follow `RUNBOOK.md` §5 and RUNS.md runs 37-40: capture
**all** recurring step shapes (`MAX_CAPTURE=4096`), or fi decode looks
falsely slow because eager prefill chunks leak into decode rounds.

Both headline regimes were re-measured through the backend strings in one
session (job 2440327), median total tok/s over 3 rounds:

| regime | native | fi_dg | fi_nvfp4 | prior |
|---|---|---|---|---|
| decode-1k (capture 4096, `dec2k` knobs) | 32258 | 32896 (1.020x) | **34522 (1.070x)** | 1.074x |
| prefill-8k (capture 8192, `8k` knobs) | 45779 | 47452 (1.037x) | **53806 (1.175x)** | 1.181x |

Every cell is within 2.2% of its pre-switch value, so the **1.18x prefill /
1.07x decode** headline holds on the backend-string path. Exact cells:

```bash
# decode-1k
ENFORCE_EAGER=0 MAX_CAPTURE=4096 MAX_NUM_SEQS=1024 \
MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl_sm100_nvfp4 \
FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_dsv4_dec2k.json \
python bench_offline.py --tag fi_dec --workload decode:128:256 \
  --num-prompts 1024 --rounds 3 --out results/fi_dec.json

# prefill-8k -- the sparse CAPTURE_SIZES list is mandatory here: the dense
# default made vllm estimate 310 GiB of graph pool and drove KV negative.
ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 \
CAPTURE_SIZES=256,2048,4096,8192 \
MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl_sm100_nvfp4 \
FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_dsv4_8k.json \
python bench_offline.py --tag fi_pre --workload prefill:1024:1 \
  --num-prompts 256 --rounds 3 --out results/fi_pre.json
```

Run the three backends of a regime in **one session**. Run 34 found native's
decode-1k drifts round-over-round within a session, so cross-session ratios
are not trustworthy.

## 8. Things that will bite you

**`FI_MOE_EP` is now a hard error.** Any non-empty `FI_MOE_EP` or
`FI_MOE_EP_MEGAKERNEL` in the environment aborts at startup, including
`FI_MOE_EP=0`, and including when the backend is native. This is deliberate:
under the old mechanism a stale export silently changed which path ran, so
leftover exports could quietly produce native numbers labelled "fi". Unset
them. Old shells, job scripts, and `main`-branch harness scripts all set
them.

**EPLB is rejected, not ignored.** `--enable-eplb` with any of these backends
raises at startup. The FlashInfer experts neither apply the
logical-to-physical expert map nor report per-expert load, so a rebalance
would move weights without moving routing. Use `deep_gemm_mega_moe` if you
need EPLB. (Because the combination is rejected, the cooperative-launch
`NCCL_MAX_CTAS` override in `eplb_utils.py` stays keyed on
`deep_gemm_mega_moe` and needs no change.)

**The venv keeps whatever was applied last.** `apply.sh` writes into the
installed wheel, so after running this the venv holds the vllm-pr sources. To
go back to `main`'s workflow, switch the branch and re-apply:

```bash
git -C $ROOT/moe_ep_benchmark switch main
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && bash patch_0251/apply.sh'
```

That restores `fi_utils.py`/`model.py`. The three extra `kernel.py` literal
entries stay behind and are harmless — `main` never selects them. For a full
revert, copy `kernel.py.orig` back. Forgetting the re-apply is loud, not
silent: `main`'s scripts set `FI_MOE_EP=1`, which the new code rejects.

**DeepSeek-V3.2 still uses `FI_MOE_EP=1`.** `patch_v32/` keeps the stock
FusedMoE factory, whose quant oracles reject a `flashinfer_moe_ep_*` backend,
so V3.2 gates inside its own patched model and names its megakernel directly.
`orchestrate_v32.sh` is intentionally unconverted.

**Teardown tracebacks are cosmetic.** On DSL 4.5.2, `worker.shutdown()`
imports `CuMemAllocator`, which trips over tilelang's `libcudart_stub.so`
missing `cudaDeviceReset`. They appear after results are written. Harmless.

## 9. Coverage

### Checked

Jobs 2439803 / 2439811 (registration + eager smokes) and 2440327 (prequant,
mxfp8, graph-mode throughput), all 4xGB200, vLLM 0.25.1, cutlass-dsl 4.5.2.

| what | result |
|---|---|
| config registration + guards | 14/14 (§5) |
| per-rank kernel resolution | every EP rank bootstraps the named megakernel; native bootstraps none |
| fi_dg vs native, eager | 8/8 bit-exact, \|dlp\| 0.0000 |
| fi_nvfp4 (mx ckpt, dequant path) vs native | 1/8 exact, \|dlp\| 0.016-0.13 |
| fi_nvfp4 (NVFP4 ckpt, **prequant** path) vs native | 2/8 exact, \|dlp\| 0.013-0.077 — cross-checkpoint |
| **fi_mxfp8** vs native | 1/8 exact, \|dlp\| 0.021-0.062 |
| NVFP4 ckpt + a deep_gemm backend | rejected at startup, naming the nvfp4 backend |
| graph-mode throughput, both regimes | within 2.2% of pre-switch (§7) |

The prequant comparison is **cross-checkpoint** (NVFP4 cast vs mx original),
so a wider band than the same-checkpoint rows is expected and is not evidence
of a bug — `eval_gsm8k.py` is the cross-checkpoint fairness gate, and it has
not been re-run since the switch.

Note `bench_offline.py`'s two-checkpoint policy routes `fi_nvfp4` to the
NVFP4 cast automatically, so every historical `fi_nvfp4` *throughput* number
(including the 1.18x) was always measured on the prequant path. Only the
smoke path needed `MODEL=` set by hand.

### Still open

* **No multi-node run.** Single node, TP4+EP4 only.
* **GSM8K not re-run** since the switch (the cross-checkpoint accuracy gate).
* **Option (b) — building the real PR branch from source — is unverified.**
  Everything above patches a 0.25.1 wheel instead.
