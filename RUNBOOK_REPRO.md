# MoE-EP: reproducing the whole thing from scratch

Manual steps for the FlashInfer `moe_ep` mega-MoE work on DeepSeek-V4-Flash:
clone, fetch checkpoints, build, kernel microbenchmark, vLLM e2e. Commands are
copied verbatim from the scripts that produced the recorded numbers.

Provenance of every expected number below: jobs **2441404** (microbenchmark)
and **2441711** (e2e sweep), 2026-07-24, one 4xGB200 node, vLLM 0.25.1,
cutlass-dsl 4.5.2.

> **On this branch (`vllm_repro_8_gpu`) the measured configuration is 1x8, not
> the 1x4 this document was originally written against.** §1–§3 (build,
> container, venv, patch, checkpoints) are world-size independent and are what
> `vllm_e2e/RUNBOOK_1x8.md` defers to. For the runs themselves follow that
> document, and check yourself against `expected_results.md` — the EP4 sweep
> script and the EP4 numbers are not carried here.

Companions, paths relative to this repo unless noted:
`vllm_e2e/RUNBOOK_1x8.md` (the executed 8-GPU procedure),
`expected_results.md` (the numbers), `RUNBOOK.md` (microbenchmark), and — in
the flashinfer checkout — `docs/design_docs/moe_ep_runbook.md`, which owns the
container recipe (§1c) and the guide to adding a new mega-kernel backend.
The chronological run log (`vllm_e2e/RUNS.md`) lives on the `vllm-pr` branch.

---

## 0. Prerequisites

* One node with 4x GB200 (sm_100). Nothing here is multi-node — TP4+EP4 only.
* SLURM with pyxis/enroot, account `coreai_libraries_cudnn`.
* Container image: `$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh`.
  Ships torch 2.12, deep_gemm, triton, nvshmem, cutlass. Does **not** ship vLLM.
  You build it yourself in §1c — the recipe lives in repo (2), so it cannot be
  built before the clone, and it needs a SLURM allocation.
* ~350 GB of disk for the two checkpoints (149 GB + 157 GB), plus room for the
  venv and the JIT cache.

```bash
export ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
export W=$ROOT/moe_ep_benchmark/vllm_e2e
export IMG=$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh
```

`ROOT` may be any scratch dir you own, but **keep the directory layout below it
exactly as shown** — including the `flashinfer-2/` parent. Several scripts
hardcode these paths; see §6 for the full list of what to patch when `ROOT`
differs.

---

## 1. Clone the repos and build the image

### 1a. The two repos you need

```bash
mkdir -p $ROOT/flashinfer-2

# (1) harness, runbooks, and the vLLM patch          -- branch vllm-pr
git clone https://github.com/mhoqueanik/moe_ep_benchmark.git $ROOT/moe_ep_benchmark
git -C $ROOT/moe_ep_benchmark switch vllm-pr

# (2) flashinfer kernels + moe_ep runtime            -- branch 4_5_2-perf-fix
git clone https://github.com/mhoqueanik/flashinfer-moe_ep.git \
    $ROOT/flashinfer-2/flashinfer-moe_ep
git -C $ROOT/flashinfer-2/flashinfer-moe_ep switch 4_5_2-perf-fix
git -C $ROOT/flashinfer-2/flashinfer-moe_ep submodule update --init --recursive
```

Flashinfer's 4 submodules (cccl, cutlass, nixl, spdlog) are required — both the
image build and the editable install compile against them.

Branch discipline: `moe_ep_benchmark` `main` deliberately still uses the old
`FI_MOE_EP=1` opt-in, because no released vLLM knows the new backend strings.
Use `vllm-pr` for everything in this file.

### 1b. Optional third repo — the PR, for reading only

```bash
# branch fi-moe-ep-v4 -- REFERENCE ONLY, never built, nothing here imports it
git clone https://github.com/mhoqueanik/vllm.git $ROOT/vllm-fi-moe-ep
git -C $ROOT/vllm-fi-moe-ep switch fi-moe-ep-v4
```

Nothing in `moe_ep_benchmark` references this path — no script, no import. Skip
it unless you want to diff the port. The PR sits on a much newer vLLM `main`
than 0.25.1, so the validated path patches a 0.25.1 wheel instead (§3d);
building it from source is untried. `patch_0251/flashinfer_moe_ep.py` is
byte-identical to the PR's copy, so repo (1) already gives you that file;
`patch_0251/model.py` is a 0.25.1 port and differs by ~90 lines of unrelated
upstream drift.

Commits the recorded numbers were taken at:

| repo | branch | commit |
|---|---|---|
| `moe_ep_benchmark` | `vllm-pr` | `36d3add` |
| `flashinfer-2/flashinfer-moe_ep` | `4_5_2-perf-fix` | `1ee41bcd` |
| `vllm-fi-moe-ep` (optional) | `fi-moe-ep-v4` | `c019433` |

### 1c. Build the container image

Needs a SLURM allocation. Easiest is to hold the §3a node first and come back
here — the ordering works because nothing in §3 runs before the image exists.
The build script comes from repo (2), which is why this cannot precede §1a.

```bash
RW=$ROOT/flashinfer-2/flashinfer-moe_ep
srun --overlap --jobid="$JOBID" -N1 \
  --container-image=nvcr.io/nvidia/pytorch:26.05-py3 \
  --container-save=$IMG \
  --container-mounts=$RW:/host/flashinfer \
  bash -lc 'bash /host/flashinfer/docker/install/build_flashinfer_ep_pytorch.sh'
```

**Mount at `/host/flashinfer`, not `/host`.** The build script defaults
`FI_SRC=/host/flashinfer` and does `cd "$FI_SRC"; pip install -e .`, so the repo
*root* (which holds `pyproject.toml`) must land at `/host/flashinfer`. Mounting
`$RW:/host` instead makes `FI_SRC` resolve to `$RW/flashinfer` — the package
directory, which has no `pyproject.toml` — and the editable install fails.
(Verified 2026-07-25 on a from-scratch build.)

(Upstream writes `--jobid="$SLURM_JOB_ID"` without `--overlap`, which is right
only from *inside* a batch script. From a login shell against the §3a hold job
you want `$JOBID` and `--overlap`, as above.)

**Build it on the same CPU architecture as the nodes you will run on.** The
image is a full userspace, not just CUDA payload: the one in §0 was built on
GB200 and is aarch64, so it cannot be copied to an x86_64 cluster. The recipe
itself is arch-agnostic — it resolves wheels at build time — so the identical
command produces a working image on either. See §4b for the check.

Upstream source: `docs/design_docs/moe_ep_runbook.md` §"Create the container" in
repo (2). It saves to the **undated** `flashinfer-ep-pt2605-mega_moe_ep.sqsh`;
`--container-save=$IMG` above writes the dated name §0 pins instead. Same
recipe — the date is just a rebuild stamp. If `$ROOT` already holds several
`.sqsh` files, `-20260712` is the one the numbers were taken on;
`-new-cutedsl` is a **later** sibling and is *not* it, so do not pick by mtime.

Build flags are tri-state (unset = on, best-effort): `BUILD_NIXL_EP=0` skips the
NIXL-EP meson build, `BUILD_NIXL_EP=1` makes its missing build deps a hard
error, `BUILD_NVEP=0` turns both backends off.

> **The image is not the DSL source of truth.** `build_flashinfer_ep_pytorch.sh`
> pins `nvidia-cutlass-dsl[cu13]==4.5.0`, but every recorded number is on
> **4.5.2**. The two paths reach it differently: §3 gets it for free because
> `vllm==0.25.1` pins 4.5.2 and the venv shadows the image, while §4 does *not*
> use the venv, so its payload must install and assert 4.5.2 explicitly over the
> image's 4.5.0. Never read the DSL version off the image — §3f and the §4
> payload guard are the checks.

---

## 2. Fetch the two checkpoints

Two checkpoints, same base weights. Each backend runs the format its kernel
consumes natively, so fi_cutedsl skips a dequant/requant:

| backend | checkpoint | size |
|---|---|---|
| native, fi_dg | mx-format original | 149 GB |
| fi_cutedsl | NVFP4 cast | 157 GB |

Run this **on a host with outbound network**, not inside the container and not
on a compute node — nothing here needs a GPU, and compute nodes are commonly
walled off from the Hub. It is also the one long-running step you want going
before you hold a node in §3a, since 305 GB dominates the whole setup.

```bash
export CKPT=$ROOT/checkpoints
export HF_HOME=$ROOT/.cache/huggingface
mkdir -p $CKPT

python -m pip install -U "huggingface_hub[cli,hf_transfer]"

# HF_HUB_ENABLE_HF_TRANSFER is DEPRECATED and ignored on huggingface_hub >=1.x:
# the client now defaults to the Xet high-performance transfer, which is
# CPU-heavy. On a shared login node with an arbiter/cgroup reaper that got the
# process SIGKILLed (exit 137) at ~18GB with 300+GB RAM free -- i.e. NOT an OOM.
# Disable Xet so the pull uses low-CPU plain-HTTPS range downloads (resumable):
export HF_HUB_DISABLE_XET=1

# Both repos are public and ungated as of 2026-07-25, so no licence click.
# hf auth login raises rate limits, but is NOT strictly required: verified
# 2026-07-25 that an anonymous, Xet-disabled pull of the 46-shard NVFP4 repo
# completed without a 429. Log in if you do hit "We had to rate limit your IP".
hf auth login   # optional; skip to try anonymous first

# Full 40-char commits pin the exact trees §5d was measured on. Do NOT drop
# --revision and do NOT resolve to main -- see the warning below.

# (a) mx-format original -- native and fi_dg
hf download deepseek-ai/DeepSeek-V4-Flash \
    --revision 6e763230a9d263eca2023f1d4a5ce1bfe126cf48 \
    --local-dir $CKPT/deepseek-v4-flash

# (b) NVFP4 cast of the same base weights -- fi_cutedsl
hf download nvidia/DeepSeek-V4-Flash-NVFP4 \
    --revision 48bfe38c62be14e8d82f9e3be12fe5d30a2e38c8 \
    --local-dir $CKPT/deepseek-v4-flash-nvfp4
```

On huggingface_hub older than 0.34 the command is `huggingface-cli download`
with the same arguments. `hf download` resumes, so re-run it after an
interruption rather than starting over.

Point the harness at them — otherwise it uses the cluster-local mirror paths
compiled into `bench_offline.py`, which will not exist on another machine:

```bash
export MODEL=$CKPT/deepseek-v4-flash                  # native, fi_dg
export MODEL_NVFP4=$CKPT/deepseek-v4-flash-nvfp4      # fi_cutedsl
```

`bench_offline.py` picks between them from `MOE_BACKEND` automatically, so you
never pass `--model`. Priority is `--model` > `MODEL` env > per-backend default.

Sanity-check before spending a node on it:

```bash
python -c "
import json, pathlib
for p in ('$MODEL', '$MODEL_NVFP4'):
    d = pathlib.Path(p)
    cfg = json.load(open(d / 'config.json'))
    n = len(list(d.glob('model-*.safetensors')))
    print(f'{d.name}: {cfg[\"model_type\"]} {cfg[\"num_hidden_layers\"]}L '
          f'{cfg[\"n_routed_experts\"]}E top-{cfg[\"num_experts_per_tok\"]}, {n} shards')
"
```

Expect `deepseek_v4 43L 256E top-6` and **46 shards** for both. The NVFP4 copy
additionally carries `hf_quant_config.json` (producer `modelopt`, version
`dsv4-nvfp4-experts`, per-expert `w1`/`w2`/`w3` marked `NVFP4` at
`awq_block_size` 16) and a `cast_mxfp4_to_nvfp4.log` recording the cast
per tensor. If `hf_quant_config.json` is missing you have the wrong repo, and
fi_cutedsl will silently fall back to the dequant path.

That check passes on HEAD too, so confirm the revisions separately — the
metadata schemas below are the tell:

```bash
python -c "
import json
q = json.load(open('$MODEL_NVFP4/hf_quant_config.json'))['quantization']
k = next(iter(q['quantized_layers']))
mx = json.load(open('$MODEL/config.json'))
assert q['quant_algo'] is None, 'NVFP4 ckpt is at HEAD, not 48bfe38'
assert k.count('.') == 5, 'NVFP4 ckpt is at HEAD, not 48bfe38 (per-layer keys)'
assert 'expert_dtype' not in mx, 'mx ckpt is at HEAD, not 6e76323'
print('both checkpoints are at the pinned revisions')
"
```

> **Do not download HEAD.** Both repos moved on after the pinned commits. The
> safetensors are byte-identical either way — only metadata changed — but the
> metadata is what the loaders read:
>
> | file | pinned (what §5d ran) | HEAD |
> |---|---|---|
> | NVFP4 `hf_quant_config.json` | `quant_algo: null`, per-expert-tensor keys (`layers.0.ffn.experts.0.w1`), `awq_block_size: 16` | `quant_algo: "MIXED_PRECISION"`, per-layer keys (`layers.0.ffn.experts`), `group_size: 16` |
> | mx `config.json` | no `expert_dtype` | `expert_dtype: "fp4"` |
>
> The NVFP4 rewrite landed in `7fc18be` (2026-06-10), two commits past the
> pin. If fi_cutedsl's loader keys on the per-expert entries or on
> `awq_block_size`, HEAD gives you the silent dequant-path fallback *with*
> `hf_quant_config.json` present — which the shard/config check above will not
> catch.

Both repo IDs were confirmed against huggingface.co on 2026-07-25: public,
ungated, 46 shards, 156.7 GiB (NVFP4) and 148.6 GiB (mx). The lowercase
spellings redirect to the canonical casing, so either form downloads the same
tree.

> **The CI mirror's NVFP4 copy is now off-pin — do not use it for fi_cutedsl.**
> `bench_offline.py:29`'s `DEFAULT_MODEL_NVFP4` points at
> `nvidia_deepseek-v4-flash-nvfp4/hf/hf-48bfe38_orig`, but as of 2026-07-25 that
> path no longer exists on `/lustre/share/coreai_dlalgo_ci`; the mirror advanced
> to `hf-e3cd60e_orig`, whose `hf_quant_config.json` is the **post-rewrite
> schema** (`quant_algo: "MIXED_PRECISION"`, per-layer keys, `group_size`) — the
> exact silent-dequant-fallback case above. So the "unset MODEL_NVFP4 falls back
> to the compiled-in mirror default" path is broken: you must download the
> pinned `48bfe38` NVFP4 (above) and pass `MODEL_NVFP4` explicitly. The **mx**
> mirror, by contrast, IS still at the pin (`deepseek-ai_deepseek-v4-flash/hf/
> hf-6e76323_orig`, 46 shards), so `DEFAULT_MODEL` for native/fi_dg is fine. The
> same is true of the V4-Pro mirror: mx `hf-0366e4e_orig` is pinned, but both
> NVFP4-Pro revisions (`hf-1449d1e`, `hf-d6acf0c`) are post-rewrite.

If you cannot reach the Hub, the fallback is to copy the 157 GB directory from
a cluster that has it. Regenerating the cast is not an option here — no repo in
this tree carries an mxfp4→NVFP4 script, and `cast_mxfp4_to_nvfp4.log` records
the result (33792 expert tensors across 46 shards, 100% lossless) but not the
tool or its invocation. Running fi_cutedsl on the mx checkpoint
(`MODEL_NVFP4=$MODEL`) takes the dequant→requant path instead: it runs, and is
what the pre-2026-07-19 setup did, **but it will not reproduce §5d** — those
numbers were all measured on the prequantized path.

---

## 3. Build the environment

### 3a. Hold a node

```bash
JOBID=$(sbatch --parsable -A coreai_libraries_cudnn -p batch -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J coreai_libraries_cudnn-fi.vllm_pr.hold \
    --output=$W/logs/hold_%j.log --wrap "sleep 14400")
echo "hold job $JOBID"
```

### 3b. Every later command goes through the container wrapper

```bash
JOBID=$JOBID bash $W/in_container.sh '<command>'
```

`in_container.sh` runs `srun --overlap --jobid=$JOBID` into `$IMG` under the
container name `fivllm`, mounts `$ROOT` read-write and `/lustre/share`
read-only, and exports:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=$ROOT/.cache/huggingface
export PIP_CACHE_DIR=$ROOT/.cache/pip
export FLASHINFER_WORKSPACE_BASE=$ROOT/.cache/flashinfer-root-ws
```

`FLASHINFER_WORKSPACE_BASE` is load-bearing. The container runs as root, so
without it the JIT cache lands in `/root/.cache` inside the overlay, dies with
the hold job, and every new job repays the full nvcc/`cute.compile` cost — over
30 minutes for the trtllm moe module alone (observed 2026-07-21).

If your checkpoints live outside `$ROOT`, add them to `--container-mounts` or
`in_container.sh` will not see them.

Those four are the **only** variables the wrapper sets. Everything else —
`CKPT`, `MODEL`, `MODEL_NVFP4` — reaches the container solely through srun's
default `--export=ALL`, i.e. from whatever shell you type the command in. So
re-export §2's block in any new shell before using the hold job. Skipping it is
not silent-but-wrong, it just fails to find the model: §5c's
`export MODEL=$CKPT/deepseek-v4-flash` expands to `/deepseek-v4-flash` when
`CKPT` is unset. This bites most often the day *after* setup, when the 4h hold
job is still alive but your terminal is not.

### 3c. One command

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash setup_container.sh'
```

`FRESH=1` wipes and rebuilds — needed if the venv predates the 2026-07-22 MR!27
WAR and still carries DSL 4.6.1. Steps 3d-3f are what it does; run them by hand
only if you are debugging the setup.

### 3d. What that actually runs

```bash
export PIP_CACHE_DIR=$ROOT/.cache/pip
export PIP_CONSTRAINT=""
python3 -m venv --system-site-packages $W/venv0251
source $W/venv0251/bin/activate

# venv ships pip 24.0, whose resolver crashes (TypeError ... NoneType) on the
# NGC dist metadata visible through system-site-packages. Upgrade first.
python -m pip install -q --upgrade pip

# vLLM 0.25.1 + dep closure (torch 2.11 etc. land in the venv). vllm's compiled
# ops use the stable libtorch ABI, so the torch minor need not match the image.
python -m pip install vllm==0.25.1

# flashinfer branch, editable/JIT -- replaces the flashinfer-python wheel vllm
# just pulled in. --no-deps: closure already satisfied, and a full resolve
# trips over the NGC system-site metadata.
python -m pip uninstall -y -q flashinfer-python || true
BUILD_NIXL_EP=0 python -m pip install --no-build-isolation --no-deps \
    -e $ROOT/flashinfer-2/flashinfer-moe_ep

# patch the installed vLLM with the fi moe_ep integration
bash $W/patch_0251/apply.sh
```

**No cubin download is needed, and there is no step for one.** Both backends in
this PR are JIT-compiled from `flashinfer/moe_ep/kernel_src/cutedsl_megamoe`
through CuteDSL; nothing under `flashinfer/moe_ep/` reads a cubin. The
~25-minute `flashinfer --download-cubin` fetch pulls `TRTLLM_GEN_FMHA/GEMM/BMM`
artifacts for the trtllm-gen split path, which this work never selects.
Verified: the working cache holds 0 `.cubin` files and only ~8 MB of JIT
`cached_ops`, and the full sweep still runs. If you later add a split-path or
`fused_moe` baseline column, or move attention to a trtllm-gen backend, that
fetch becomes a hard prerequisite — raise `FLASHINFER_CUBIN_DOWNLOAD_THREADS`
(default 4) to cut the wall time.

Because the kernels are generated from source rather than shipped prebuilt, the
DSL version is part of the measurement, not an install detail — hence the pin
and the guards below.

### 3e. What `patch_0251/apply.sh` does

Idempotent; re-running prints `kernel.py: backends already registered (no-op)`.

1. Backs up `model.py` / `kernel.py` as `*.orig`, copies the PR's `model.py`
   into `vllm/models/deepseek_v4/nvidia/`.
2. Installs the helpers at `vllm/utils/flashinfer_moe_ep.py` and deletes the
   pre-move `nvidia/fi_utils.py` so a stale copy cannot shadow it.
3. Inserts the two backend strings into the `MoEBackend` literal in
   `vllm/config/kernel.py`. **Required, not cosmetic** — `KernelConfig`
   validates `moe_backend` against that literal and rejects unknown values
   before the model is ever constructed. Done as an in-place insertion, not a
   whole-file copy, so it does not roll the rest of that config module back to
   whatever vLLM version the patch was snapshotted from.
4. Drops stale bytecode, then greps the patched tree and **fails** if anything
   still imports `deepseek_v4.nvidia.fi_utils`. That guard exists because a
   function-local import survived the module move once; since it sat in the
   *native* experts' `forward()`, all three fi columns passed and it surfaced
   only 45 minutes into a sweep as a missing baseline (job 2441415).

Expected:

```
kernel.py: registered flashinfer_moe_ep_mega_deep_gemm, flashinfer_moe_ep_mega_cutedsl
patched: .../vllm/models/deepseek_v4/nvidia (backup: model.py.orig)
patched: .../vllm/config/kernel.py (backup: kernel.py.orig)
```

### 3f. Verify

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && python - <<PY
import importlib
from importlib.metadata import version
for m in ("torch", "vllm", "flashinfer", "flashinfer.moe_ep",
          "vllm.utils.flashinfer_moe_ep", "cutlass", "nvshmem.core"):
    print(f"OK {m}: {importlib.import_module(m).__file__}")
assert version("nvidia-cutlass-dsl") == "4.5.2", version("nvidia-cutlass-dsl")
print("GUARD PASS: cutlass-dsl 4.5.2")
PY'
```

`flashinfer.__file__` must resolve under `$ROOT/flashinfer-2`, not to a
site-packages wheel.

> The version of `setup_container.sh` at commit `36d3add` still probes the
> **pre-move** path `vllm.models.deepseek_v4.nvidia.fi_utils` in its own sanity
> block, so it prints a permanent, misleading `FAIL vllm fi patch`. It does not
> stop the setup (that block only prints; the DSL guard is the only hard gate).
> Use the check above instead.

---

## 4. Kernel microbenchmark (~16 min)

Drives the FlashInfer kernels directly, no vLLM, so it isolates kernel work
from integration overhead. Submits its own SLURM job — it does not use the hold
job from §3a, and it installs into the container overlay rather than the venv.

**§2 is not a prerequisite for this section.** The geometries come from
`model_shapes/shapes.tsv` (hidden / inter / experts / top-k) and the weights are
synthetic, so nothing here reads a checkpoint. If the microbenchmark is all you
want, you need §1, §3a's node, and this section — skip the 305 GB download.

```bash
cd $ROOT/moe_ep_benchmark
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh

# NOTE: model_shapes/results/ is TRACKED and a fresh clone already contains
# committed CSVs from prior runs. The glob below merges them all (make_tables
# keys on (geometry, tokens/rank, variant) and later files win), so on a
# from-scratch checkout render ONLY your run's CSV to compare against §4a:
python model_shapes/make_tables.py \
    model_shapes/results/model_shapes_<your_stamp>_deepseek_v4_flash.csv \
    -o /tmp/micro_scratch_RESULTS.md
# (the glob form is for accumulating cells at a FIXED world size once the dir is
#  yours; it silently mixes runs otherwise.)
python model_shapes/make_tables.py model_shapes/results/model_shapes_*.csv
```

Its in-container payload (`model_shapes/job_payload.sh`) — note the defaults are
env-overridable, which is what makes §4b possible without editing anything:

```bash
cd $ROOT/flashinfer-2/flashinfer-moe_ep
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e .
DSL_VERSION="${DSL_VERSION:-4.5.2}"
python -m pip install "nvidia-cutlass-dsl[cu13]==${DSL_VERSION}"
python -c "from importlib.metadata import version; v=version('nvidia-cutlass-dsl'); \
assert v=='${DSL_VERSION}', f'DSL {v} != ${DSL_VERSION}'; print(f'GUARD PASS: cutlass-dsl {v}')"
GPUS="${GPUS:-4}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    bash "$BENCH/model_shapes/run_model_shapes.sh"
```

This is the step that needs §6 if `ROOT` differs — both `submit_jobs.sh` and
`job_payload.sh` hardcode the original path, and the payload asserts on it.

Do **not** unpin the DSL. The CuteDSL codegen is version-sensitive enough
(34-54% slower pre-4.5.2) that an unpinned `--upgrade` makes a sweep
unattributable. `DSL_VERSION` overrides.

### 4a. Expected numbers — EP4, 4x GB200

Expected — `e2e_pipelined` p50 us at the DeepSeek-V4-Flash MoE geometry (4096
hidden / 2048 inter / 256 experts / top-6), EP4, job 2441404:

| tokens/rank | 8 | 64 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| `deep_gemm_mega` | 128.0 | 175.1 | 201.7 | 240.6 | 389.7 | 718.4 | 1246.2 |
| `nvfp4_cutedsl` | 141.3 | 188.6 | 229.4 | 257.0 | 341.0 | 564.2 | 1037.6 |
| `+ikr` | 148.6 | 202.8 | 228.4 | 260.1 | 339.4 | 558.1 | 1024.7 |
| `+combine_mxfp8` | 150.1 | 196.1 | 218.9 | 246.8 | 310.2 | 480.2 | 858.0 |
| `+combine_nvfp4` | 145.4 | 194.6 | 214.6 | 236.5 | 293.8 | 447.5 | 769.0 |
| **nvfp4 vs dg** | 0.91x | 0.93x | 0.88x | 0.94x | 1.14x | 1.27x | 1.20x |

DeepGEMM wins below ~1024 tokens/rank, CuteDSL above — which is why the e2e
decode cells gain less than the prefill ones. The quantized combine wires look
strong here (`combine_nvfp4` is 1.62x DeepGEMM at 8192) but did **not** transfer
e2e at this geometry, so they stay off by default; read RUNS.md run 24 before
enabling them.

### 4b. Porting to another system, and to EP8

Nothing below needs a source edit beyond §6's `sed`. In order of what actually
blocks you:

**1. The container image is architecture-bound.** The `.sqsh` in §0 was built on
the GB200 nodes and is **aarch64** — `unsquashfs -l` shows an
`aarch64-linux-gnu` userspace throughout. On an x86_64 target it will not run,
and the failure is not obviously an arch problem. Rebuild it on the target with
§1c; the build script is arch-agnostic (it resolves wheels at build time), so
the same command works on either. Check what you have:

```bash
unsquashfs -l $IMG | grep -m1 -o 'aarch64\|x86_64'
```

**2. SM100 is required, not preferred.** Every mega kernel is Blackwell-only and
the arch is validated against the live device, so `deep_gemm_mega` and
`nvfp4_cutedsl` both refuse to run on H100/H200 — this sweep has no meaning
there. B200 and GB200 NVL are both fine.

**3. Ask for GPUs explicitly if your scheduler needs it.** `submit_jobs.sh`
requests `-N1 --ntasks-per-node=1` and **no** `--gres`/`--gpus-per-node`, which
works only because `-p batch` here hands out whole nodes. Elsewhere that lands
you a 0-GPU allocation. Add the flag alongside the §6 account/partition swap.

**4. Set the world size by environment, not by editing.** `job_payload.sh`
defaults are `${GPUS:-4}` and `${CUDA_VISIBLE_DEVICES:-0,1,2,3}`, and
`submit_jobs.sh` submits with `--export=ALL`, so both ride through:

```bash
cd $ROOT/moe_ep_benchmark
GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh
```

Set **both**. `GPUS` alone leaves the device list at the 4-GPU default on any
cluster that does not populate `CUDA_VISIBLE_DEVICES` itself; conversely, if you
do request `--gres`, SLURM sets `CUDA_VISIBLE_DEVICES` in the job environment
and that wins over the exported value — harmless when it lists all 8, wrong if
it lists fewer than `GPUS`. `world size = DP = EP` (`run.sh:33`), so `GPUS=8`
*is* EP8.

**5. EP8 is a new measurement, not a refresh of §4a.** World size sets the
expert split, so at 8-way each rank holds 32 of DeepSeek-V4-Flash's 256 experts
instead of 64. Tokens-per-expert halves at a given tokens/rank, which is exactly
the axis the DeepGEMM↔CuteDSL crossover sits on — expect it to move off ~1024.
Record the result as its own table; it does not supersede §4a. Divisibility is
fine for every row of `shapes.tsv` at 8-way (`num_experts % world == 0` is
asserted at `bench_moe_ep_mega.py:353`; 128/256/384/512 all divide by 8).

**6. Leave `MEGA_KNOBS` unset.** Empty means the shim's token-count heuristic,
which is what job 2441404 used; `MEGA_KNOBS=auto` instead runs an online
autotune sweep and keeps the winner for the session. Turning that on in the same
run that changes EP size moves two variables at once. Tune as a follow-up, not
as part of the port.

**7. `gpt_oss_120b` will come back partly empty, by design.** Its 2880/2880
geometry is not `%128`, so `deep_gemm_mega` rejects it and only the fp4 variants
produce rows; the cutedsl kernels are tail-safe down to `%64`. `run_variant`
prints `[warn] … failed (continuing)`, so the gap in the table is expected
rather than a broken run.

**8. Write EP8 results to a separate directory — this one silently corrupts
§4a.** The CSVs do record a `gpus` column, but `make_tables.py` keys each cell
on `(geometry, tokens_per_rank, variant)` only and never reads it
(`make_tables.py:68`). Merging an EP8 CSV with the EP4 ones therefore overwrites
matching cells rather than separating them — later file wins, exactly as
`submit_jobs.sh` advertises for filling gaps at a *fixed* world size. §4's own
render command globs the whole directory, so the default path walks straight
into it:

```bash
python model_shapes/make_tables.py model_shapes/results/model_shapes_*.csv   # <-- merges EP4 + EP8
```

The output would look like §4a and contain EP8 numbers, with nothing in
`RESULTS.md` recording the difference — and `RESULTS.md` is a tracked file, so
the corruption commits. `run_model_shapes.sh` honours `OUT_DIR`, which rides
through `--export=ALL`, so keep the two apart at the source:

```bash
OUT_DIR=$ROOT/moe_ep_benchmark/model_shapes/results_ep8 \
GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh

python model_shapes/make_tables.py model_shapes/results_ep8/model_shapes_*.csv \
    -o model_shapes/RESULTS_EP8.md
```

Sanity-check before rendering — the column is there, so use it:

```bash
cut -d, -f10 model_shapes/results_ep8/model_shapes_*.csv | sort -u   # expect: gpus, 8
```

---

## 5. vLLM e2e

### 5a. Tier 1 — config checks (~1 min, no model)

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python test_backend_registration.py'
```

Expected tail: `15/15 checks passed` / `ALL CHECKS PASSED`. The 15th is a
flashinfer-version gate, added after the 14-check runs logged in `RUNS.md`.
Two `Failed to import Triton kernels ...
triton_kernels.matmul_ogs` ERROR lines are pre-existing container noise.
`VERBOSE=1` prints tracebacks for real failures.

Run this whenever you touch backend selection — it catches the likeliest rot, a
backend registered in one file but not the other, in a minute instead of the
~10 a smoke costs.

### 5b. Tier 2 — correctness smoke (~12 min, 4 GPUs)

Tier 1 is **not** sufficient: it passes even if `use_fi_mega_moe` silently stays
false and the run quietly executes the native path.

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  ENFORCE_EAGER=1 MOE_BACKEND=deep_gemm_mega_moe \
  python smoke_infer.py --tag native --out results/pr_native.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  ENFORCE_EAGER=1 MOE_BACKEND=flashinfer_moe_ep_mega_deep_gemm \
  python smoke_infer.py --tag fi_dg --out results/pr_fi_dg.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python compare_outputs.py results/pr_native.json results/pr_fi_dg.json'
```

**Check the bootstrap banner in the fi log** — this, not the diff, is the proof
the backend string reached a kernel on every EP rank:

```
[fi_moe_ep] ep_rank=0 world=4 cuda.current_device=0 megakernel=deep_gemm_mega
[fi_moe_ep] ep_rank=1 world=4 cuda.current_device=1 megakernel=deep_gemm_mega
... one per rank
```

The **native** run must print no `[fi_moe_ep]` line at all. If it does, the
predicate is mis-routing and the comparison means nothing.

Expected (job 2439811): fi_dg vs native **8/8 exact**, mean |dlogprob| 0.0000.
fi_cutedsl vs native 1/8 exact, 0.016-0.13 — it diverges by construction
(double quantization), and that comparison is cross-checkpoint, so a wider band
is expected and is not a bug.

> **The fi_dg 8/8-exact figure is build-specific — do not treat it as a gate.**
> On a from-scratch build 2026-07-25 (B200, DSL 4.5.2), fi_dg vs native came in
> at **1/8 exact, mean |dlogprob| ≈ 0.02–0.06**: near-identical, but one flipped
> logit early in a greedy decode diverges the rest of that sequence. native and
> fi_dg are separate kernel implementations, so bit-exactness is not guaranteed
> across builds/hardware. The real correctness gate is GSM8K (below), where a
> properly-armed run (job 2337476) scored native **0.960**, fi_dg **0.960**
> (identical), fi_cutedsl **0.970** on the NVFP4 cast — all in band. Use the
> logprob smoke to confirm *routing* (the `[fi_moe_ep]` banner), not to demand
> bit-exact generations.
>
> The fi_cutedsl **0.955** previously recorded here came from the 03:42 run
> that had `MODEL` exported and so scored the mx checkpoint under an
> `fi_nvfp4` tag — see the warning in §5b.

To produce that fi_cutedsl row, add a third smoke. **`smoke_infer.py` does not
resolve the checkpoint per backend** the way `bench_offline.py` and
`eval_gsm8k.py` do — its `--model` defaults to `$MODEL` regardless of
`MOE_BACKEND`, so the NVFP4 path must be passed explicitly or you will quietly
benchmark fi_cutedsl on the mx checkpoint:

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  ENFORCE_EAGER=1 MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl \
  python smoke_infer.py --tag fi_cutedsl --model $MODEL_NVFP4 \
    --out results/pr_fi_cutedsl.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python compare_outputs.py results/pr_native.json results/pr_fi_cutedsl.json'
```

Because that pair is cross-checkpoint, the logprob delta is not a pass/fail
signal — `eval_gsm8k.py` is. It boots one engine per backend and resolves the
checkpoint per backend; both must land in the same band (~0.95 for DSV4-Flash)
before any §5d ratio is an apples-to-apples claim.

> **Do not export `MODEL` around this gate — it silently disarms it.**
> `resolve_model` ranks `--model` > `$MODEL` > per-backend default, so a
> `MODEL=<mx path>` in the environment sends *every* backend to the mx
> checkpoint, including `fi_cutedsl`. The gate then compares the mx weights
> against themselves, scores a comfortable pass, and tests nothing. This is not
> hypothetical: the 2026-07-25 03:42 run (job 2337127) did exactly that — its
> `gsm8k_fi_nvfp4.json` records `model=...hf-6e76323_orig`, the mx original —
> so the NVFP4 checkpoints behind every fi_cutedsl throughput number went
> unvalidated until job 2337476. Pass `--model` explicitly and check the
> `model` field the eval records in each result JSON.

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python eval_gsm8k.py --tag native --model $MODEL_MX \
    --out results/gsm8k_native.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl \
  python eval_gsm8k.py --tag fi_nvfp4 --model $MODEL_NVFP4 --min-acc 0.93 \
    --out results/gsm8k_fi_nvfp4.json'
```

`--min-acc` exits 2 below threshold. The eval also records `truncated` — how
many completions hit `--max-tokens` — because a chain cut off mid-reasoning
still ends in *a* number and so scores as a confident wrong answer, not as
unparseable. A low accuracy with a high `truncated` is a token-budget problem,
not a model problem; `--min-acc 0.93` is calibrated for DSV4-Flash and is not
automatically the right threshold for a model that reasons longer.

### 5c. Throughput — the four headline cells

Run the three backends of a regime in **one session**: native's decode-1k drifts
round-over-round, so cross-session ratios are not trustworthy.

Everything below runs **inside the container on the held node**, with `$W` as
the working directory — `bench_offline.py` and `results/` are referenced
relatively, exactly as the sweep job does it. Either paste it into an
interactive shell there, or save it as `cells.sh` and pipe it in:

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash -s' < cells.sh
```

Common setup:

```bash
cd $W
source venv0251/bin/activate
bash patch_0251/apply.sh
# DO NOT export MODEL. resolve_model() (bench_offline.py:42) returns $MODEL for
# EVERY backend when it is set, so `export MODEL=$CKPT/deepseek-v4-flash` forces
# fi_cutedsl onto the mx checkpoint too — the dequant→requant path — and the
# fi_cutedsl column silently STOPS reproducing §5d (measured ~1.15x instead of
# ~1.18x at prefill-8k; verified 2026-07-25). Leave MODEL unset so native/fi_dg
# use DEFAULT_MODEL (the pinned mx mirror) and only fi_cutedsl consults
# MODEL_NVFP4. Set MODEL_NVFP4 explicitly (the compiled-in default is off-pin,
# see §2):
export MODEL_NVFP4=$CKPT/deepseek-v4-flash-nvfp4
export FI_MOE_EP_SKIP_VERSION_CHECK=1   # the 0.6.15 venv is below the new
                                        # flashinfer floor: documented
                                        # pre-release escape hatch

DG=flashinfer_moe_ep_mega_deep_gemm
CUTEDSL=flashinfer_moe_ep_mega_cutedsl

# One cell = one workload against all three backends, in one session.
# Verbatim from job_vllm_pr_runbook_sweep_ep8.sh. The knob cache is fi_cutedsl-only.
cell() {
    local name=$1; shift
    local envs=$1; shift
    for be in deep_gemm_mega_moe $DG $CUTEDSL; do
        local short=native
        [[ $be == "$DG" ]] && short=fi_dg
        [[ $be == "$CUTEDSL" ]] && short=fi_cutedsl
        local cache=''
        [[ $short == fi_cutedsl ]] && \
            cache=FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_dsv4_8k.json
        echo "--- $name / $short ---"
        env $envs MOE_BACKEND=$be $cache \
            python bench_offline.py --tag sw_${name}_${short} "$@" \
            --out results/sweep_${name}_${short}.json
    done
}
```

The four cells:

```bash
# prefill-8k (headline). The sparse CAPTURE_SIZES list is mandatory: the dense
# default made vllm estimate 310 GiB of graph pool and drove KV negative.
cell pre8k 'ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 CAPTURE_SIZES=256,2048,4096,8192' \
    --workload prefill:1024:1 --num-prompts 256 --rounds 3

# decode-1k (headline). CAPTURE_SIZES is mandatory here for the same reason --
# see the warning below; this cell shipped without it until 2026-07-25.
cell dec1k 'ENFORCE_EAGER=0 MAX_CAPTURE=4096 MAX_NUM_SEQS=1024 CAPTURE_SIZES=256,1024,2048,4096' \
    --workload decode:128:256 --num-prompts 1024 --rounds 3

# 100K ISL / 1K OSL @ 32 concurrent (interactivity)
cell lc100k 'ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 CAPTURE_SIZES=32,256,2048,8192 MAX_NUM_SEQS=32 MAX_MODEL_LEN=102400 GPU_MEM_UTIL=0.93 REQUIRE_LATENCY=1' \
    --workload longctx:100000:1024 --num-prompts 32 --rounds 2

# 32K ISL / 32 OSL @ 32 concurrent (interactivity)
cell ctx32k 'ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 CAPTURE_SIZES=32,256,2048,8192 MAX_NUM_SEQS=32 MAX_MODEL_LEN=33792 GPU_MEM_UTIL=0.93 REQUIRE_LATENCY=1' \
    --workload longctx:32768:32 --num-prompts 32 --rounds 3
```

> **Every cell that sets `MAX_CAPTURE` must also pin `CAPTURE_SIZES`, and the
> penalty for forgetting lands on the flashinfer backends only.** `dec1k` was
> the one cell that did not, and it silently produced garbage for months.
> Measured on V4-Pro EP8 (jobs 2337473 / 2337487): with the dense default
> ladder, vLLM's CUDA-graph memory profiler reserved **~48 GiB/GPU** for both
> flashinfer backends against a real capture cost of ~6 GiB — the same ~6 GiB
> it estimates correctly for native. The phantom reservation comes straight out
> of the KV cache:
>
> | backend | KV available | KV tokens | sequences resident | tok/s |
> |---|---|---|---|---|
> | native | 48.91 GiB | 95,979 | 1024 of 1024 | 13268 |
> | fi_dg | 7.28 GiB | 14,286 | **189** of 1024 | 5969 (0.45x) |
> | fi_cutedsl | 0.07 GiB | — | engine would not start | OOM |
>
> The tell is a backend that looks *fast per step and slow overall*: fi_dg's
> ITL was **better** than native (42.6 vs 94.4 ms) precisely because its
> batches were 5x smaller. If you see that shape, check
> `Available KV cache memory` and the `Running:`/`Waiting:` counts before
> blaming the kernel. Pinning restores fi_dg to 1.02x and fi_cutedsl to 1.19x,
> and costs native ~3% to batch padding — so **dec1k numbers recorded before
> 2026-07-25 are not comparable with ones recorded after.**

Results land in `$W/results/sweep_<cell>_<backend>.json` — twelve files.

All twelve at once (~1 h, submits its own exclusive node, prints a summary table
and re-applies the patch itself):

```bash
cd $W && sbatch job_vllm_pr_runbook_sweep_ep8.sh          # %j log lands here
```

This is the script that produced job 2441711 — every number in §5d. It lived in
a scratch dir until 2026-07-25; it now ships in the repo alongside
`bench_offline.py`, with the cells byte-identical to that job. It reads `ROOT`,
`IMG`, `ROUNDS`, `MODEL`, `MODEL_NVFP4`, and `EXTRA_MOUNTS` from the
environment, so a different checkout needs no edit:

```bash
# Pass MODEL_NVFP4 but NOT MODEL (see the resolve_model warning above): with
# MODEL set, the sweep forwards it and fi_cutedsl loads the mx dequant path.
cd $W && ROOT=$ROOT MODEL_NVFP4=$MODEL_NVFP4 \
    sbatch -A <account> -p <partition> job_vllm_pr_runbook_sweep_ep8.sh
```

`--rounds N` runs **N+1** passes: round 0 is a warmup, kept in the JSON as
`"warmup": true` and excluded from the median. Do not remove it — the warmup
round came in slower than the median in all twelve cells of job 2441711, by up
to 3.1% on decode-1k, which is *larger* than the 2.2% fi_dg-vs-native effect
that cell is measuring. Prefix caching is off for the same reason: rounds reuse
prompts, so with it on every post-warmup round is a 100% cache hit and prefill
measures nothing (once produced a fake 91k tok/s).

### 5d. Expected numbers

Median total tok/s, job 2441711:

| cell | native | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k `prefill:1024:1` x256 | 45816 | 47593 (1.039x) | **54132 (1.182x)** |
| decode-1k `decode:128:256` x1024 | 32191 | 32893 (1.022x) | **34435 (1.070x)** |
| 100K ISL / 1K OSL, 32 conc | 34553 | 35322 (1.022x) | **37491 (1.085x)** |
| 32K ISL / 32 OSL, 32 conc | 42235 | 43425 (1.028x) | **48430 (1.147x)** |

Latency, TTFT in seconds and ITL in milliseconds:

| cell | | native | fi_dg | fi_cutedsl |
|---|---|---|---|---|
| 100K / 1K | TTFT p50 | 41.7 | 40.6 | 36.6 |
| | ITL p50 / p99 | 48.5 / 83.7 | 47.5 / 81.8 | 45.8 / 76.0 |
| 32K / 32 | TTFT p50 | 12.8 | 12.4 | 11.1 |
| | ITL p50 / p99 | 190.9 / 191.1 | 185.6 / 185.9 | 165.5 / 165.6 |

Treat these as a band, not a target — rounds land within ~1%, but the native
decode-1k baseline drifts between sessions.

> **These numbers are on GB200; B200 lands slightly lower.** §5d is 1x4 **GB200**
> (Grace CPU + NVLink-C2C). A from-scratch rerun on 1x4 **B200** (2026-07-25,
> job 2337172) reproduced the *shape* — fi_cutedsl 1.151x prefill-8k, 1.051x
> decode-1k, 1.080x at 100K, 1.137x at 32K, with the same "advantage shrinks
> with context" trend — but absolute tok/s and ratios each sit a hair under the
> GB200 table (native prefill 44656 vs 45816; fi_cutedsl 1.151x vs 1.182x),
> because the Grace-side dispatch/attention work is on a discrete host instead.
> Both are Blackwell sm_100 and valid; just don't compare a B200 run cell-for-
> cell against the GB200 targets.

**The advantage shrinks as context grows**: 1.182x at prefill-8k, 1.147x at 32K,
1.085x at 100K. Attention takes a larger share of every step at long context, so
a fixed MoE-kernel win buys proportionally less end to end. The microbenchmark
shows the same crossover at ~1024 tokens/rank. Expect the trend, not a single
number.

**32K/32 is not a decode measurement.** With 32 output tokens its ITL p50 and
p99 match to one decimal (190.9 / 191.1) — the cell is prefill bound and ITL is
just the steady rate. Raise OSL if you want it to say something about decode.

---

## 6. Running with a different ROOT

`in_container.sh`, `setup_container.sh` and `bench_offline.py` all take
overrides (`ROOT`, `REPO`, `VENV`, `MODEL`, `MODEL_NVFP4`), so §2, §3 and §5
need no edits. The microbenchmark in §4 does — these three lines hardcode the
original path:

| file | line | what |
|---|---|---|
| `model_shapes/submit_jobs.sh` | 12 | `ROOT=/lustre/fsw/.../mhoqueanik` |
| `model_shapes/job_payload.sh` | 7 | `ROOT=/lustre/fsw/.../mhoqueanik` |
| `model_shapes/job_payload.sh` | 28 | `assert m.__file__.startswith(".../flashinfer-2")` |

```bash
OLD=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
sed -i "s#$OLD#$ROOT#g" $ROOT/moe_ep_benchmark/model_shapes/submit_jobs.sh \
                        $ROOT/moe_ep_benchmark/model_shapes/job_payload.sh
```

Also swap `-A coreai_libraries_cudnn` / `-p batch` for your account and
partition in `submit_jobs.sh`, `job_payload.sh`, and the §3a hold job.
`job_vllm_pr_runbook_sweep_ep8.sh` needs no edit for either — it takes `ROOT` from
the environment, and `sbatch -A … -p …` on the command line overrides its
`#SBATCH` lines.

The line-28 assert is why the `flashinfer-2/` parent directory has to stay:
it checks the resolved `flashinfer.__file__` prefix, so renaming the checkout
makes the microbenchmark fail after the install rather than before it.

---

## 7. Things that will bite you

**`FI_MOE_EP` is now a hard error.** Any non-empty `FI_MOE_EP` or
`FI_MOE_EP_MEGAKERNEL` aborts at startup — including `FI_MOE_EP=0`, and
including when the backend is native. Deliberate: under the old mechanism a
stale export silently changed which path ran, so leftovers could produce native
numbers labelled "fi". Old shells, job scripts, and every `main`-branch harness
script set them.

**EPLB is rejected, not ignored.** `--enable-eplb` with any fi backend raises at
startup. The FlashInfer experts neither apply the logical-to-physical expert map
nor report per-expert load, so a rebalance would move weights without moving
routing. Use `deep_gemm_mega_moe` if you need EPLB.

**Capture all recurring step shapes** (`MAX_CAPTURE=4096` for decode). Otherwise
eager prefill chunks leak into decode rounds and fi decode looks falsely slow.

**The venv keeps whatever was applied last.** `apply.sh` writes into the
installed wheel. To go back to `main`'s workflow, switch the branch and
re-apply. Forgetting is loud, not silent: `main`'s scripts set `FI_MOE_EP=1`,
which the new code rejects. For a full revert, copy `kernel.py.orig` back.

**Teardown tracebacks are cosmetic.** On DSL 4.5.2 `worker.shutdown()` imports
`CuMemAllocator`, which trips over tilelang's `libcudart_stub.so` missing
`cudaDeviceReset`. They appear after results are written.

## 8. Not covered

* No multi-node run. Single node, TP4+EP4 only.
* ~~GSM8K not re-run since the backend-string switch~~ **DONE 2026-07-25, job
  2337476** — and it caught that the gate had been disarmed by an exported
  `MODEL` (§5b). Flash: native 0.960 / fi_dg 0.960 / fi_cutedsl 0.970 on the
  NVFP4 cast. V4-Pro scores 0.880 / 0.880 / 0.890 — consistent across all three
  backends, so not a moe_ep issue, but below the 0.93 gate; whether that is the
  512-token cap truncating a longer reasoner or a real deficit is open (job
  2337550).
* Building the real PR branch from source is unverified; everything here
  patches a 0.25.1 wheel.
