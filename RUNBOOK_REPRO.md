# MoE-EP: reproducing the whole thing from scratch

Manual steps for the FlashInfer `moe_ep` mega-MoE work on DeepSeek-V4-Flash
and V4-Pro: clone, build the image, fetch checkpoints, build the venv, then the
kernel microbenchmark, the vLLM e2e sweeps and the accuracy gate. Commands are
copied from the scripts that produced the recorded numbers.

The numbers you should get are in [expected_results.md](expected_results.md),
measured on one 1x8 B200 node with vLLM 0.25.1, flashinfer `4_5_2-perf-fix`
@ `1ee41bcd` and cutlass-dsl 4.5.2.

This is the only runbook you need. The one other document worth knowing about
is in the flashinfer checkout — `docs/design_docs/moe_ep_runbook.md`, which
owns the container recipe (§1.2c) and the guide to adding a new mega-kernel
backend.

**Four sections.** [§1 Prep](#1-prep) — clone, image, checkpoints, venv, patch.
[§2 Kernel microbenchmark](#2-kernel-microbenchmark-16-min) — no vLLM, no
checkpoints. [§3 vLLM e2e — throughput](#3-vllm-e2e--throughput) — the four
headline cells, both models. [§4 Accuracy](#4-accuracy) — correctness smoke and
the GSM8K cross-checkpoint gate. Then §5 gotchas and §6 what is not covered.
Expected numbers for §2–§4 live in [expected_results.md](expected_results.md).

---

## 1. Prep

### 1.1. Prerequisites

* One node with **8x SM100** (cc 10.0) on NVLink — B200 or GB200 NVL8.
  Nothing here is multi-node. The measured configuration is TP8+EP8 throughout;
  `expected_results.md` is what you check against.
* **CUDA 12 or 13.** `moe_ep` is CUDA-major agnostic: the image build and the
  DSL install both derive the `cuXX` wheel suffix from `torch.version.cuda`
  (override with `CUDA_MAJOR=<n>`), and the build enforces the NCCL floor
  (>= 2.30.7) for whichever major it picks. The recorded numbers are on 13.
* SLURM with pyxis/enroot, and an account and partition you can submit to.
  Every `sbatch`/`srun` below shows `-A <account> -p <partition>` — substitute
  yours; nothing in the repo depends on a particular one.
* A container image, which you build in §1.2c — it is not downloadable. It
  ships torch 2.12, deep_gemm, triton, nvshmem and cutlass, and does **not**
  ship vLLM. The recipe lives in repo (2), so it cannot be built before the
  clone, and it needs a SLURM allocation.
* Disk under `$ROOT`: **~350 GB for Flash** — checkpoints 323 GB (mx 149 +
  NVFP4 174) plus ~25 GB for the container image, venv and pip cache, which all
  live there too. **Another 1.66 TB if you also want V4-Pro** (mx 806 + NVFP4
  851), which is optional. §2 (the kernel microbenchmark) reads no checkpoint,
  so it needs only the ~25 GB.

Set these three once; every command below is written against them.

```bash
export ROOT=/path/to/your/scratch          # ~350 GB, or ~2 TB with V4-Pro
export W=$ROOT/moe_ep_benchmark/vllm_e2e   # derived; do not change
export IMG=$ROOT/flashinfer-ep.sqsh        # the image §1.2c writes; any name
```

`ROOT` is yours to choose, and the layout below it is only what the scripts
default to — notably `$ROOT/flashinfer-2/flashinfer-moe_ep`. Every path is an
environment override, so no file needs editing; a different arrangement costs
you a `REPO=`. §1.5 has the details, including the account and partition you do
have to supply.

---

### 1.2. Clone the repos and build the image

#### 1.2a. The two repos you need

```bash
mkdir -p $ROOT/flashinfer-2

# (1) harness, runbook, and the vLLM patch     -- branch vllm_repro_8_gpu
git clone https://github.com/mhoqueanik/moe_ep_benchmark.git $ROOT/moe_ep_benchmark
git -C $ROOT/moe_ep_benchmark switch vllm_repro_8_gpu

# (2) flashinfer kernels + moe_ep runtime            -- branch 4_5_2-perf-fix
git clone https://github.com/mhoqueanik/flashinfer-moe_ep.git \
    $ROOT/flashinfer-2/flashinfer-moe_ep
git -C $ROOT/flashinfer-2/flashinfer-moe_ep switch 4_5_2-perf-fix
git -C $ROOT/flashinfer-2/flashinfer-moe_ep submodule update --init --recursive
```

Flashinfer's 4 submodules (cccl, cutlass, nixl, spdlog) are required — both the
image build and the editable install compile against them.

`vllm_repro_8_gpu` is the reproduction branch and is what this file documents:
1x8 only, one set of results. Other branches of this repo carry development
history and are not needed here — clone the branch above and everything in this
runbook applies.

#### 1.2b. Optional third repo — the PR, for reading only

```bash
# branch fi-moe-ep-v4 -- REFERENCE ONLY, never built, nothing here imports it
git clone https://github.com/mhoqueanik/vllm.git $ROOT/vllm-fi-moe-ep
git -C $ROOT/vllm-fi-moe-ep switch fi-moe-ep-v4
```

Nothing in `moe_ep_benchmark` references this path — no script, no import. Skip
it unless you want to diff the port. The PR sits on a much newer vLLM `main`
than 0.25.1, so the validated path patches a 0.25.1 wheel instead (§1.4e);
building it from source is untried. `patch_0251/flashinfer_moe_ep.py` is
byte-identical to the PR's copy, so repo (1) already gives you that file;
`patch_0251/model.py` is a 0.25.1 port and differs by ~90 lines of unrelated
upstream drift.

Commits the recorded numbers were taken at:

| repo | branch | commit |
|---|---|---|
| `moe_ep_benchmark` | `vllm_repro_8_gpu` | `5810ef1` or later |
| `flashinfer-2/flashinfer-moe_ep` | `4_5_2-perf-fix` | `1ee41bcd` |
| `vllm-fi-moe-ep` (optional) | `fi-moe-ep-v4` | `c019433` |

> The job scripts echo `git log --oneline -1` at startup, which reports the last
> *commit* rather than the working tree. The verification logs therefore say
> `399cb3c` even though they ran the scripts as of `5810ef1` — the edits were
> uncommitted when the jobs were submitted. If you are matching a log against
> the repo, trust the script contents, not that line.

#### 1.2c. Build the container image

Needs a SLURM allocation. Easiest is to hold the §1.4a node first and come back
here — the ordering works because nothing in §1.4 runs before the image exists.
The build script comes from repo (2), which is why this cannot precede §1.2a.

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

(Upstream writes `--jobid="$SLURM_JOB_ID"` without `--overlap`, which is right
only from *inside* a batch script. From a login shell against the §1.4a hold job
you want `$JOBID` and `--overlap`, as above.)

**Build it on the same CPU architecture as the nodes you will run on.** The
image is a full userspace, not just a CUDA payload, so an image built on
aarch64 (GB200) will not run on x86_64 (B200 hosts) or vice versa. The recipe
above is arch-agnostic — it resolves wheels at build time — so the identical
command produces a working image on either; just run it on the target. This
only becomes a problem if you copy a `.sqsh` in from somewhere else, and §2b
has the one-line check for that case.

Upstream source: `docs/design_docs/moe_ep_runbook.md` §"Create the container"
in repo (2). Left to itself it saves as `flashinfer-ep-pt2605-mega_moe_ep.sqsh`;
`--container-save=$IMG` above writes wherever you pointed `IMG`. Same recipe
either way — the filename carries no meaning, so if you keep several images
around, name them so you can tell which is which rather than relying on mtime.

Build flags are tri-state (unset = on, best-effort): `BUILD_NIXL_EP=0` skips the
NIXL-EP meson build, `BUILD_NIXL_EP=1` makes its missing build deps a hard
error, `BUILD_NVEP=0` turns both backends off.

> **The image is not the DSL source of truth.** `build_flashinfer_ep_pytorch.sh`
> pins `nvidia-cutlass-dsl[${CU}]==4.5.0`, but every recorded number is on
> **4.5.2**. The two paths reach it differently: §1.4 gets it for free because
> `vllm==0.25.1` pins 4.5.2 and the venv shadows the image, while §2 does *not*
> use the venv, so its payload must install and assert 4.5.2 explicitly over the
> image's 4.5.0. Never read the DSL version off the image — §1.4f and the §2
> payload guard are the checks.

---

### 1.3. Fetch the checkpoints

Two per model, same base weights. Each backend runs the format its kernel
consumes natively, so fi_cutedsl skips a dequant/requant:

| model | backend | repo | revision | size |
|---|---|---|---|---|
| Flash | native, fi_dg | `deepseek-ai/DeepSeek-V4-Flash` | `6e763230…` | 149 GB |
| Flash | fi_cutedsl | `nvidia/DeepSeek-V4-Flash-NVFP4` | `48bfe38c…` | 174 GB |
| Pro *(optional)* | native, fi_dg | `deepseek-ai/DeepSeek-V4-Pro` | `0366e4e0…` | 806 GB |
| Pro *(optional)* | fi_cutedsl | `nvidia/DeepSeek-V4-Pro-NVFP4` | `9e7e88ee…` | 851 GB |

All four are public and ungated. **Flash alone (323 GB)
is enough for §2, §3's Flash sweep and §4's Flash rows** — Pro is optional and
adds 1.66 TB.

Run this **on a host with outbound network**, not inside the container and not
on a compute node — nothing here needs a GPU, and compute nodes are commonly
walled off from the Hub. It is also the one long-running step you want going
before you hold a node in §1.4a: at 323 GB for Flash — 2.0 TB if you also take
Pro — it dominates the whole setup.

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

# All four repos are public and ungated, so there is no licence click.
# hf auth login raises rate limits but is not required -- an anonymous,
# Xet-disabled pull works. Log in if you hit "We had to rate limit your IP".
hf auth login   # optional; skip to try anonymous first

# The revisions pin the exact trees the recorded numbers were measured on.
# Do NOT drop --revision and do NOT resolve to main -- see the warning below.

# (a) mx-format original -- native and fi_dg
hf download deepseek-ai/DeepSeek-V4-Flash \
    --revision 6e763230a9d263eca2023f1d4a5ce1bfe126cf48 \
    --local-dir $CKPT/deepseek-v4-flash

# (b) NVFP4 cast of the same base weights -- fi_cutedsl
hf download nvidia/DeepSeek-V4-Flash-NVFP4 \
    --revision 48bfe38c62be14e8d82f9e3be12fe5d30a2e38c8 \
    --local-dir $CKPT/deepseek-v4-flash-nvfp4
```

**DeepSeek-V4-Pro — optional, 1.66 TB.** Skip it unless you want the §3e
throughput sweep and the Pro half of §4; §2 and everything Flash work without
it. Same two-format policy, same pinning rule:

```bash
# (c) mx-format original -- native and fi_dg          [optional]
hf download deepseek-ai/DeepSeek-V4-Pro \
    --revision 0366e4e064385807ea86b088a5c6c878ff23343b \
    --local-dir $CKPT/deepseek-v4-pro

# (d) NVFP4 cast -- fi_cutedsl                        [optional]
hf download nvidia/DeepSeek-V4-Pro-NVFP4 \
    --revision 9e7e88ee2a2677a2c4d2bc6c18d0e328769b555e \
    --local-dir $CKPT/deepseek-v4-pro-nvfp4
```

> **`--revision` is not optional on either NVFP4 repo.** Both have since
> published newer revisions whose `hf_quant_config.json` uses a different
> schema (`quant_algo: "MIXED_PRECISION"` with per-layer keys, instead of
> `null` with per-expert keys). fi_cutedsl loads those without complaining and
> silently takes the dequant fallback, so you get numbers that look plausible
> and mean nothing. `48bfe38c` and `9e7e88ee` are the newest revisions on each
> that still carry the prequantized schema.
> `vllm_e2e/setup/dl_nvfp4_{flash,pro}.sh` resolve that for you; if you paste
> the `hf download` commands by hand, keep `--revision`.

`vllm_e2e/setup/` wraps all four pulls if you would rather not paste:
`dl_mx_originals.sh [flash|pro|both]`, `dl_nvfp4_flash.sh`, `dl_nvfp4_pro.sh`.

On huggingface_hub older than 0.34 the command is `huggingface-cli download`
with the same arguments. `hf download` resumes, so re-run it after an
interruption rather than starting over.

Point the harness at them with the **per-model** variables the job scripts
consume. Otherwise it falls back to the cluster-local mirror paths compiled
into `bench_offline.py`, which do not exist on another machine:

```bash
export MODEL_MX_FLASH=$CKPT/deepseek-v4-flash
export MODEL_NVFP4_FLASH=$CKPT/deepseek-v4-flash-nvfp4
export MODEL_MX_PRO=$CKPT/deepseek-v4-pro                  # optional, §3e / §4
export MODEL_NVFP4_PRO=$CKPT/deepseek-v4-pro-nvfp4         # optional
```

> **Do not export `MODEL`.** `resolve_model` ranks `--model` > `$MODEL` >
> per-backend default, so a `MODEL=` in the environment sends *every* backend
> to that one checkpoint — including fi_cutedsl, which then silently runs the
> mx dequant path and reports meaningless numbers. All three job scripts pass
> `--model` per cell for exactly this reason, and unset `MODEL` inside the
> container. The same mistake disarmed the accuracy gate once; see §4b.

Sanity-check before spending a node on it:

```bash
python -c "
import json, pathlib
for p in ('$MODEL_MX_FLASH', '$MODEL_NVFP4_FLASH'):
    d = pathlib.Path(p)
    cfg = json.load(open(d / 'config.json'))
    n = len(list(d.glob('model-*.safetensors')))
    print(f'{d.name}: {cfg[\"model_type\"]} {cfg[\"num_hidden_layers\"]}L '
          f'{cfg[\"n_routed_experts\"]}E top-{cfg[\"num_experts_per_tok\"]}, {n} shards')
"
```

Expect `deepseek_v4 43L 256E top-6` and **46 shards** for both Flash copies.
Re-run it against `$MODEL_MX_PRO` / `$MODEL_NVFP4_PRO` if you took Pro, where
the answer is `deepseek_v4 61L 384E top-6` and **64 shards**. The NVFP4 copy
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
q = json.load(open('$MODEL_NVFP4_FLASH/hf_quant_config.json'))['quantization']
k = next(iter(q['quantized_layers']))
mx = json.load(open('$MODEL_MX_FLASH/config.json'))
assert q['quant_algo'] is None, 'NVFP4 ckpt is at HEAD, not 48bfe38'
assert k.count('.') == 5, 'NVFP4 ckpt is at HEAD, not 48bfe38 (per-layer keys)'
assert 'expert_dtype' not in mx, 'mx ckpt is at HEAD, not 6e76323'
print('both checkpoints are at the pinned revisions')
"
```

> **Do not download the latest revision.** All four repos moved on after the
> pinned commits. The
> safetensors are byte-identical either way — only metadata changed — but the
> metadata is what the loaders read:
>
> | file | pinned (what §3d ran) | HEAD |
> |---|---|---|
> | NVFP4 `hf_quant_config.json` | `quant_algo: null`, per-expert-tensor keys (`layers.0.ffn.experts.0.w1`), `awq_block_size: 16` | `quant_algo: "MIXED_PRECISION"`, per-layer keys (`layers.0.ffn.experts`), `group_size: 16` |
> | mx `config.json` | no `expert_dtype` | `expert_dtype: "fp4"` |
>
> If you took Pro, run the same two checks against `$MODEL_NVFP4_PRO` and
> `$MODEL_MX_PRO` — the assertions are identical, only the expected revisions
> differ (`9e7e88ee` and `0366e4e0`). Its newer revisions carry the same
> rewritten schema, so skipping this is the same silent failure.
>
> If fi_cutedsl's loader keys on the per-expert entries or on
> `awq_block_size`, HEAD gives you the silent dequant-path fallback *with*
> `hf_quant_config.json` present — which the shard/config check above will not
> catch.

Both Flash repos are 46 shards, 156.7 GiB (NVFP4) and 148.6 GiB (mx) as the Hub
reports them — the table above quotes on-disk `du`, which is a little larger.
The lowercase
spellings redirect to the canonical casing, so either form downloads the same
tree.

> **Ignore the paths compiled into `bench_offline.py`.** Its `DEFAULT_MODEL`
> and `DEFAULT_MODEL_NVFP4` point at a mirror on the machine these numbers were
> measured on, and that mirror has since drifted off-pin — its NVFP4 copy now
> carries the post-rewrite schema, i.e. exactly the silent-dequant case above.
> Those defaults exist only as a convenience there; on any other machine set
> the four `MODEL_*` variables and they are never consulted. The job scripts
> require them and fail at submit time if they are missing.

If you cannot reach the Hub, the fallback is to copy the NVFP4 directory from
a cluster that has it. Regenerating the cast is not an option here — no repo in
this tree carries an mxfp4→NVFP4 script, and `cast_mxfp4_to_nvfp4.log` records
the result (33792 expert tensors across 46 shards, 100% lossless) but not the
tool or its invocation. Running fi_cutedsl on the mx checkpoint
(`MODEL_NVFP4_FLASH=$MODEL_MX_FLASH`) takes the dequant→requant path instead:
it runs, **but it will not reproduce §3d** — those numbers were all measured on
the prequantized path.

---

### 1.4. Build the environment

#### 1.4a. Hold a node

```bash
mkdir -p $W/logs          # sbatch fails if --output's directory does not exist

JOBID=$(sbatch --parsable -A <account> -p <partition> -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J moe_ep.hold \
    --output=$W/logs/hold_%j.log --wrap "sleep 14400")
echo "hold job $JOBID"
```

Four hours is the wall clock everything in §1.4 and §4 runs against; §2 and §3
submit their own jobs and do not use it.

#### 1.4b. Every later command goes through the container wrapper

```bash
JOBID=$JOBID bash $W/in_container.sh '<command>'
```

`in_container.sh` runs `srun --overlap --jobid=$JOBID` into `$IMG` under the
container name `fivllm`, mounts `$ROOT` read-write (plus `/lustre/share`
read-only *if that path exists* — it is a cluster-local checkpoint mirror, and
`EXTRA_MOUNTS` adds anything else), and exports:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=$ROOT/.cache/huggingface
export PIP_CACHE_DIR=$ROOT/.cache/pip
export FLASHINFER_WORKSPACE_BASE=$ROOT/.cache/flashinfer-root-ws
```

`FLASHINFER_WORKSPACE_BASE` is load-bearing. The container runs as root, so
without it the JIT cache lands in `/root/.cache` inside the overlay, dies with
the hold job, and every new job repays the full nvcc/`cute.compile` cost — over
30 minutes for the trtllm moe module alone.

If your checkpoints live outside `$ROOT`, pass them via `EXTRA_MOUNTS` —
e.g. `EXTRA_MOUNTS=/data/ckpt:/data/ckpt:ro` — or the container will not see
them. No need to edit `in_container.sh`.

Those four are the **only** variables the wrapper sets. Everything else —
`CKPT` and the four `MODEL_*` paths — reaches the container solely through
srun's default `--export=ALL`, i.e. from whatever shell you type the command
in. So re-export §1.3's block in any new shell before using the hold job.
Skipping it is not silent-but-wrong, it just fails to find the model:
`MODEL_MX_FLASH=$CKPT/deepseek-v4-flash` expands to `/deepseek-v4-flash` when
`CKPT` is unset, and the job scripts' guards reject it at submit time. This
bites most often the day *after* setup, when the 4h hold job is still alive but
your terminal is not.

#### 1.4c. Build it

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash setup_container.sh'

# rebuilding over an existing venv? wipe it instead:
JOBID=$JOBID bash $W/in_container.sh 'FRESH=1 bash setup_container.sh'
```

`FRESH=1` wipes the venv first; use it whenever one is already there and you
are unsure of its provenance, since an older venv carrying DSL 4.6.1 will fail
every cell on the guard in §2. §1.4d and §1.4e describe what the script does and
§1.4f checks it landed — run those by hand only if you are debugging setup.

#### 1.4d. What that actually runs

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

**No cubin download is needed, and there is no step for one.** Both backends
are JIT-compiled from `flashinfer/moe_ep/kernel_src/cutedsl_megamoe`
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

#### 1.4e. What `patch_0251/apply.sh` does

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
   still imports `deepseek_v4.nvidia.fi_utils`. A function-local import of the
   pre-move path can hide in the *native* experts' `forward()`, where every fi
   column still passes and only the baseline breaks — an hour into a sweep,
   as a missing native column rather than an import error.

Expected:

```
kernel.py: registered flashinfer_moe_ep_mega_deep_gemm, flashinfer_moe_ep_mega_cutedsl
patched: .../vllm/models/deepseek_v4/nvidia (backup: model.py.orig)
patched: .../vllm/config/kernel.py (backup: kernel.py.orig)
```

#### 1.4f. Verify

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

`flashinfer.__file__` must resolve under your flashinfer checkout (`$REPO`,
by default `$ROOT/flashinfer-2/flashinfer-moe_ep`), not to a site-packages
wheel. If it points at site-packages the editable install did not take, and
you would be benchmarking a released flashinfer instead of the branch.

---

### 1.5. Running somewhere else

**No file needs editing.** Every script reads its paths from the environment:
`ROOT`, and where relevant `REPO`, `VENV`, `IMG`, `W`, and the four `MODEL_*`
checkpoint paths. That includes the microbenchmark — `submit_jobs.sh` and
`job_payload.sh` both take `ROOT`, and the import-path assert compares against
`$REPO` rather than a baked-in prefix.

What you do have to supply is a SLURM account and partition, since the defaults
are this cluster's:

```bash
# microbenchmark (§2)
ACCOUNT=<account> PARTITION=<partition> bash model_shapes/submit_jobs.sh

# e2e and accuracy (§3, §4) -- the CLI overrides the #SBATCH lines
sbatch -A <account> -p <partition> job_vllm_pr_runbook_sweep_ep8.sh
```

If you clone flashinfer anywhere other than the default
`$ROOT/flashinfer-2/flashinfer-moe_ep`, export `REPO` to point at it.
`job_payload.sh` asserts that the resolved
`flashinfer.__file__` sits under `$REPO` — that check is what stops a stray
wheel-installed flashinfer being benchmarked instead of your branch, and it
follows `REPO` wherever you put it.

---

## 2. Kernel microbenchmark (~25 min per shape)

Drives the FlashInfer kernels directly, no vLLM, so it isolates kernel work
from integration overhead. Submits its own SLURM job — it does not use the hold
job from §1.4a, and it installs into the container overlay rather than the venv.

**Only §1.2 is a prerequisite.** The geometries come from
`model_shapes/shapes.tsv` (hidden / inter / experts / top-k) and the weights are
synthetic, so nothing here reads a checkpoint, and the payload installs into the
container overlay rather than the venv. So if the microbenchmark is all you
want: clone and build the image (§1.2), then run this — skip §1.3 and §1.4
entirely. Building the image does need an allocation, and §1.4a is one way to
get one.

```bash
cd $ROOT/moe_ep_benchmark
ACCOUNT=<account> PARTITION=<partition> \
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh

# Omitting SHAPE_LIST runs every row of shapes.tsv -- six shapes, one job each,
# submitted in parallel. Output lands in model_shapes/results_ep8/, which
# already holds the committed CSV. make_tables keys on (geometry, tokens/rank, variant)
# and IGNORES the gpus column, so later files win and a glob silently mixes
# runs -- render only your own CSV to compare against expected_results.md §3:
python model_shapes/make_tables.py \
    model_shapes/results_ep8/model_shapes_<your_stamp>_deepseek_v4_flash.csv \
    -o /tmp/micro_scratch_RESULTS.md
# (the glob form is for accumulating cells at a FIXED world size, once the
#  directory holds only your runs.)
```

Its in-container payload (`model_shapes/job_payload.sh`) — note the defaults are
env-overridable, which is what makes §2b possible without editing anything:

```bash
cd $ROOT/flashinfer-2/flashinfer-moe_ep
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e .
DSL_VERSION="${DSL_VERSION:-4.5.2}"
CU="cu${CUDA_MAJOR:-$(python -c 'import torch; v=torch.version.cuda or ""; print(v.split(".")[0])')}"
python -m pip install "nvidia-cutlass-dsl[$CU]==${DSL_VERSION}"
python -c "from importlib.metadata import version; v=version('nvidia-cutlass-dsl'); \
assert v=='${DSL_VERSION}', f'DSL {v} != ${DSL_VERSION}'; print(f'GUARD PASS: cutlass-dsl {v}')"
GPUS="${GPUS:-8}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    bash "$BENCH/model_shapes/run_model_shapes.sh"
```

Both scripts take `ROOT`, `REPO` and `ACCOUNT`/`PARTITION` from the
environment, so a different location needs no edit — see §1.5.

Do **not** unpin the DSL. 4.5.2 is vLLM 0.25.1's own pin and what the
flashinfer `4_5_2-perf-fix` branch is validated against — the two move together,
and the codegen is version-sensitive enough that an unpinned `--upgrade` makes a
sweep unattributable. `DSL_VERSION` overrides the version this payload installs,
and `CUDA_MAJOR` the `cuXX` wheel suffix if you do not want it derived from
torch.

> **The same pin is enforced again in §3 and §4, against the venv.** This
> section installs the DSL itself, but `bench_offline.py` and `eval_gsm8k.py`
> call `assert_expected_dsl()`, which aborts unless the venv's
> `nvidia-cutlass-dsl` equals `EXPECT_DSL` (default `4.5.2`). An older venv
> carrying 4.6.1 therefore fails every e2e cell before it loads a model — that
> is the guard working, and the fix is `FRESH=1 setup_container.sh` (§1.4c),
> not `EXPECT_DSL=4.6.1`. Set `EXPECT_DSL` only when deliberately benching
> another runtime, or `EXPECT_DSL=""` to disable the check.

The geometries come from `model_shapes/shapes.tsv`, whose MoE shapes mirror the
cudnn-frontend SDPA training benchmark's model list (MoE-capable models only):

| name | hidden | moe_inter | experts | top-k |
|---|---|---|---|---|
| deepseek_v3 | 7168 | 2048 | 256 | 8 |
| kimi_k2_6 | 7168 | 2048 | 384 | 8 |
| gpt_oss_120b | 2880 | 2880 | 128 | 4 |
| qwen3_5_397b | 4096 | 1024 | 512 | 10 |
| **deepseek_v4_flash** | 4096 | 2048 | 256 | 6 |
| deepseek_v4_pro | 7168 | 3072 | 384 | 6 |

`SHAPE_LIST` selects rows; omitting it runs all six. Weights are synthetic, so
no checkpoint is read. All six are recorded in
[expected_results.md](expected_results.md) §3a.

### 2a. Expected numbers

[expected_results.md](expected_results.md) §3 has one table per shape — p50
microseconds for every variant with its speedup against `deep_gemm_mega` —
generated from the CSVs in `model_shapes/results_ep8/`.

The DeepGEMM↔CuteDSL crossover sits between 512 and 1024 tokens/rank on every
geometry: below it `deep_gemm_mega` wins, above it the CuteDSL variants pull
away. On DeepSeek-V4-Flash that reaches 1.20x at 8192 for plain `nvfp4 bf16`
and 1.71x for `+combine_nvfp4`. That crossover is why the e2e decode cells gain
less than the prefill ones — decode runs far below it.

Two things worth knowing before you compare your run: **V4-Flash is the least
favourable of the five shapes that have a baseline**, so the e2e sweeps in §3
sit on the geometry where the kernel wins least; and `deep_gemm_mega` cannot run
`gpt_oss_120b` at all, so its `dg` column is empty by construction — see §2b
item 7.

### 2b. Porting to another system, or another world size

Nothing below needs a source edit — see §1.5. In order of what actually
blocks you:

**1. The container image is architecture-bound.** It carries a full userspace,
so an aarch64 image (built on GB200) will not run on x86_64 (B200 hosts) or the
reverse, and the failure does not look like an arch problem. If you built it in
§1.2c on the machine you are running on, this cannot bite you. It bites when a
`.sqsh` is copied between clusters — check before you debug anything else:

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
you a 0-GPU allocation. Add the flag alongside the §1.5 account/partition swap.

**4. Set the world size by environment, not by editing.** `job_payload.sh`
defaults to `${GPUS:-8}` and `${CUDA_VISIBLE_DEVICES:-0,1,...,7}` on this
branch, and `submit_jobs.sh` submits with `--export=ALL`, so an override rides
through. To reproduce §2a you need nothing; to run a different world size:

```bash
cd $ROOT/moe_ep_benchmark
GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh
```

Set **both**. `GPUS` alone leaves the device list at the 8-GPU default on any
cluster that does not populate `CUDA_VISIBLE_DEVICES` itself; conversely, if you
do request `--gres`, SLURM sets `CUDA_VISIBLE_DEVICES` in the job environment
and that wins over the exported value — harmless when it lists all 8, wrong if
it lists fewer than `GPUS`. `world size = DP = EP` (`run.sh:33`), so `GPUS=8`
*is* EP8.

**5. A different world size is a different measurement.** §2a is EP8, and
world size sets the expert split — at 8-way each rank holds 32 of
DeepSeek-V4-Flash's 256 experts, at 4-way it holds 64. Tokens-per-expert
therefore halves between them at a given tokens/rank, which is exactly the axis
the DeepGEMM↔CuteDSL crossover sits on, so the crossover moves. If you run
another world size, record it as its own table rather than merging it into the
EP8 one. Divisibility is fine for every row of `shapes.tsv` at 8-way
(`num_experts % world == 0` is asserted at `bench_moe_ep_mega.py:353`;
128/256/384/512 all divide by 8).

**6. Leave `MEGA_KNOBS` unset.** Empty means the shim's token-count heuristic,
which is what the recorded numbers used; `MEGA_KNOBS=auto` instead runs an online
autotune sweep and keeps the winner for the session. Turning that on in the same
run that changes EP size moves two variables at once. Tune as a follow-up, not
as part of the port.

**7. `gpt_oss_120b` will come back partly empty, by design.** Its 2880/2880
geometry is not `%128`, so `deep_gemm_mega` rejects it and only the fp4 variants
produce rows; the cutedsl kernels are tail-safe down to `%64`. `run_variant`
prints `[warn] … failed (continuing)`, so the gap in the table is expected
rather than a broken run.

**8. One directory per world size — mixing them silently corrupts the
table.** The CSVs do record a `gpus` column, but `make_tables.py` keys each cell
on `(geometry, tokens_per_rank, variant)` only and never reads it
(`make_tables.py:68`). Merging CSVs from two world sizes therefore overwrites
matching cells rather than separating them — later file wins, exactly as
`submit_jobs.sh` advertises for filling gaps at a *fixed* world size. §2's
render command globs the whole directory, so the default path walks straight
into it if you ever add a second world size:

```bash
python model_shapes/make_tables.py model_shapes/results_ep8/model_shapes_*.csv
# This branch ships only results_ep8/. The hazard above is why: pointing this
# at a directory holding two world sizes silently overwrites cells.
```

The rendered table would look exactly like §2a while silently containing cells
from two different world sizes, and nothing in the output records which is
which. `run_model_shapes.sh` honours `OUT_DIR` (defaulting to `results_ep8/`
on this branch) and it rides through `--export=ALL`, so keep world sizes apart
at the source:

```bash
OUT_DIR=$ROOT/moe_ep_benchmark/model_shapes/results_ep8 \
GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SHAPE_LIST="deepseek_v4_flash" SEQ_LENS="8 64 512 1024 2048 4096 8192" \
    bash model_shapes/submit_jobs.sh

python model_shapes/make_tables.py model_shapes/results_ep8/model_shapes_*.csv \
    -o /tmp/micro_ep8_RESULTS.md      # expected_results.md §3 is the reference
```

Sanity-check before rendering — the column is there, so use it:

```bash
cut -d, -f10 model_shapes/results_ep8/model_shapes_*.csv | sort -u   # expect: gpus, 8
```

---

### 2c. The ad-hoc launcher (`run.sh`)

`model_shapes/submit_jobs.sh` above sweeps the shapes in `shapes.tsv` under
SLURM. `run.sh` is the interactive alternative: one process per GPU via
`torch.multiprocessing`, mirroring DP=N + EP with TP=1, for poking at a single
geometry. It defaults to `GPUS=8` / all eight devices on this branch.

```bash
# subset of fi_mega backends
MEGA_LIST="mxfp8_cutedsl nvfp4_cutedsl" SECTION=fi_mega bash run.sh
# a single backend
MEGA_LIST=deep_gemm_mega SECTION=fi_mega bash run.sh
# problem size
TOKENS=64 SECTION=fi_mega bash run.sh
# token sweep
SECTION=fi_mega bash run_sweep.sh          # or: SEQ_LENS="1 8 64 512 4096"
# cutedsl kernel knobs: online autotune, vs the pinned cache
MEGA_KNOBS=auto SECTION=fi_mega bash run.sh
# timed region: bare kernel launch vs full FI forward (default e2e)
MEGA_TIMING=kernel SECTION=fi_mega bash run.sh
```

fi_mega backends: `deep_gemm_mega | mxfp8_cutedsl | nvfp4_cutedsl`. The two
`vllm_*` sections (`bench_moe_ep_vllm_mega.py`, `bench_moe_ep_nonmega.py`) are
comparison baselines needing `vllm==0.20.0`, which the image does not ship —
skip them unless you want the split-path comparison.

The five columns in §2a's tables are `nvfp4_cutedsl` under two extra knobs,
which you can set here to reproduce one variant on its own:

| column | backend | knobs |
|---|---|---|
| `dg` | `deep_gemm_mega` | — |
| `nvfp4 bf16` | `nvfp4_cutedsl` | `MEGA_IKR=0 MEGA_COMBINE_DTYPE=bf16` |
| `+ikr` | `nvfp4_cutedsl` | `MEGA_IKR=1` (in-kernel fc2 reduce) |
| `+combine_mxfp8` | `nvfp4_cutedsl` | `MEGA_COMBINE_DTYPE=mxfp8` |
| `+combine_nvfp4` | `nvfp4_cutedsl` | `MEGA_COMBINE_DTYPE=nvfp4` |

`run_model_shapes.sh` sets these per variant; `run.sh` does not, so pass them
yourself if you are chasing a single column.

> **Comparing against the kernel repo's tester** (`cutedsl_megamoe -m
> tester.tester --mode Perf`): match BOTH the geometry and the timed region.
> The tester's problems use (hidden, inter, experts, topk) = (4096, 2048, 256,
> 6) or (7168, 3072, 384, 6), neither of which is `run.sh`'s default, and its
> timed region is a bare prebuilt kernel launch — no arg rebuild, reset, sync
> or output copy. Use `HIDDEN/INTER/NUM_EXPERTS/TOPK` plus `MEGA_TIMING=kernel`
> for an apples-to-apples run.

## 3. vLLM e2e — throughput

### 3a. The fi_cutedsl knob cache

The cutedsl kernel picks its tile/cluster schedule from a knob cache keyed by
geometry and world size. This branch ships both caches the sweeps use, so you
can go straight to §3b:

| cache | geometry | used by |
|---|---|---|
| `results/knob_cache_ep8.json` | 4096/2048/256/top-6 | Flash EP8 sweep |
| `results/knob_cache_pro_ep8.json` | 7168/3072/384/top-6 | V4-Pro EP8 sweep |

Retune only if your geometry or world size differs — the winning tile moves
with tokens-per-expert, and at EP8 each rank holds 32 of 256 experts rather
than 64. Synthetic weights, ~10 min per geometry, via
`vllm_e2e/setup/tune_knobs_{flash,pro}_ep8.sh`, which wrap:

```bash
FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_ep8.json \
torchrun --nproc_per_node=8 -m flashinfer.moe_ep.tune --dtype nvfp4 \
    --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 --max-tokens 8192
```

It is not cosmetic: on the EP8 cache fi_cutedsl prefill-8k reaches ~1.20x
(§3d); on a cache tuned for another geometry or world size it is understated.

### 3b. Tier 1 — config checks (~1 min, no model)

First rung of the ladder in §4; run it before spending a node on §3c.

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python test_backend_registration.py'
```

Expected tail: `15/15 checks passed` / `ALL CHECKS PASSED`. Two
`Failed to import Triton kernels ...
triton_kernels.matmul_ogs` ERROR lines are pre-existing container noise.
`VERBOSE=1` prints tracebacks for real failures.

Run this whenever you touch backend selection — it catches the likeliest rot, a
backend registered in one file but not the other, in a minute instead of the
~10 a smoke costs.

### 3c. The four headline cells

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
# The two Flash checkpoints from §1.3. cell() below passes whichever one the
# backend needs via --model, which is the only form resolve_model ranks above
# the environment.
export MODEL_MX_FLASH=$CKPT/deepseek-v4-flash
export MODEL_NVFP4_FLASH=$CKPT/deepseek-v4-flash-nvfp4
# DO NOT export MODEL. resolve_model() returns $MODEL for EVERY backend when it
# is set, which would drag fi_cutedsl onto the mx checkpoint — the
# dequant→requant path — and its column would quietly stop reproducing §3d.
export FI_MOE_EP_SKIP_VERSION_CHECK=1   # the 0.6.15 venv is below the new
                                        # flashinfer floor: documented
                                        # pre-release escape hatch

DG=flashinfer_moe_ep_mega_deep_gemm
CUTEDSL=flashinfer_moe_ep_mega_cutedsl

# One cell = one workload against all three backends, in one session. Same
# cells as job_vllm_pr_runbook_sweep_ep8.sh, minus its output filtering.
# The knob cache is fi_cutedsl-only.
cell() {
    local name=$1; shift
    local envs=$1; shift
    for be in deep_gemm_mega_moe $DG $CUTEDSL; do
        local short=native
        [[ $be == "$DG" ]] && short=fi_dg
        [[ $be == "$CUTEDSL" ]] && short=fi_cutedsl
        # --model per backend, never the MODEL env (see the resolve_model
        # warning in §1.3): MODEL outranks the per-backend NVFP4 default, so
        # setting it would drag fi_cutedsl onto the mx dequant path too.
        local model=$MODEL_MX_FLASH
        local cache=''
        if [[ $short == fi_cutedsl ]]; then
            model=$MODEL_NVFP4_FLASH
            cache=FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_ep8.json
        fi
        echo "--- $name / $short (model=$(basename $model)) ---"
        env $envs MOE_BACKEND=$be $cache \
            python bench_offline.py --model "$model" \
            --tag sw_ep8_${name}_${short} "$@" \
            --out results/sweep_ep8_${name}_${short}.json
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
# see the warning below.
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
> penalty for forgetting lands on the flashinfer backends only.** Measured on
> V4-Pro EP8: with the dense default ladder, vLLM's CUDA-graph memory profiler
> reserved **~48 GiB/GPU** for both
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
> and costs native ~3% to batch padding.

Results land in `$W/results/sweep_ep8_<cell>_<backend>.json` — twelve files.

All twelve at once (~1 h, submits its own exclusive node, prints a summary table
and re-applies the patch itself):

```bash
cd $W && sbatch -A <account> -p <partition> job_vllm_pr_runbook_sweep_ep8.sh
```

That inherits `ROOT` and the two Flash checkpoint paths from your shell (§1.1,
§1.3) via sbatch's `--export=ALL`. Being explicit is equivalent:

```bash
cd $W && ROOT=$ROOT IMG=$IMG \
    MODEL_MX_FLASH=$MODEL_MX_FLASH MODEL_NVFP4_FLASH=$MODEL_NVFP4_FLASH \
    sbatch -A <account> -p <partition> job_vllm_pr_runbook_sweep_ep8.sh
```

Both checkpoint variables are **required** — the script checks them before
requesting the allocation and exits with a readable message if either is unset
or is not a directory, rather than discovering it an hour in. It passes
`--model` per cell rather than relying on the `MODEL` environment variable,
which is what keeps fi_cutedsl on the NVFP4 cast (see the `resolve_model`
warning above). `ROUNDS` and `EXTRA_MOUNTS` are the other overrides.

`--rounds N` runs **N+1** passes: round 0 is a warmup, kept in the JSON as
`"warmup": true` and excluded from the median. Do not remove it — the warmup
round came in slower than the median in all twelve cells, by up
to 3.1% on decode-1k, which is *larger* than the 2.2% fi_dg-vs-native effect
that cell is measuring. Prefix caching is off for the same reason: rounds reuse
prompts, so with it on every post-warmup round is a 100% cache hit and prefill
measures nothing (once produced a fake 91k tok/s).

### 3d. Expected numbers

TP8+EP8, both models, in [expected_results.md](expected_results.md) §1-2 —
kept in one place so there is a single set of numbers to check against, with
tolerances and the two known failure modes.

Treat them as a band, not a target: ratios are stable to about ±0.02x between
sessions, absolute throughput moves more with node and thermal state, and the
native decode-1k baseline drifts round-over-round — which is why all three
backends of a cell must run in one session.

> **The recorded numbers are on B200.** A GB200 node (Grace CPU +
> NVLink-C2C) lands slightly differently, because the dispatch and attention
> host work sits on the Grace side rather than a discrete host. Both are
> Blackwell sm_100 and both are valid — just don't compare cell-for-cell
> across the two.

### 3e. The V4-Pro sweep

Everything above is DeepSeek-V4-Flash. The V4-Pro sweep completes the
throughput picture; the accuracy gate for both models is §4.

Requires the optional Pro checkpoints from §1.3. Same four cells, ~2 h, its own
exclusive 8-GPU node; native/fi_dg on the Pro mx checkpoint, fi_cutedsl on the
pinned Pro NVFP4, knob cache `knob_cache_pro_ep8.json`.

```bash
cd $W && sbatch -A <account> -p <partition> job_vllm_pr_runbook_sweep_pro.sh
```

`MODEL_MX_PRO` and `MODEL_NVFP4_PRO` come from your shell via `--export=ALL`,
exactly as the Flash sweep takes its two. The script checks both before
requesting the allocation and exits with a readable message if either is unset
or is not a directory, rather than discovering it an hour in. Output is
`results/sweep_pro_<cell>_<backend>.json` — twelve files.

The Pro NVFP4 checkpoint is 45 GiB *larger* than its mx one (850.4 vs
805.3 GiB), which is why fi_cutedsl is the first thing to run out of KV cache
if a cell is misconfigured — see §5.

## 4. Accuracy

Three verification tiers, cheapest first. Each one catches something the one
before it cannot:

| tier | what it proves | cost | where |
|---|---|---|---|
| **1** — config checks | both backend strings are registered and validate, and the retired `FI_MOE_EP` env vars are rejected | ~1 min, no model | §3b |
| **2** — correctness smoke | the backend string actually reached a kernel on every EP rank, and generations are sane | ~12 min | §4a |
| **3** — GSM8K gate | the NVFP4 checkpoint fi_cutedsl runs scores the same as the mx one native runs | ~35 min | §4b |

The throughput cells (§3c, §3e) are the measurement, not a check — they will
happily produce numbers from a mis-routed run, which is what tiers 1 and 2 are
for.

**Tier 1 is necessary but not sufficient.** It never builds a model, so it
passes even when `use_fi_mega_moe` silently stays false and the run executes the
native path — three identical columns labelled as three backends. Tier 2 is what
rules that out.

**Tier 3 is the only one that gates a number.** Because fi_cutedsl runs a
different checkpoint from native, its throughput is only comparable if its
accuracy is; tiers 1 and 2 say nothing about that.

### 4a. Tier 2 — correctness smoke (~12 min)

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
[fi_moe_ep] ep_rank=0 world=8 cuda.current_device=0 megakernel=deep_gemm_mega
[fi_moe_ep] ep_rank=1 world=8 cuda.current_device=1 megakernel=deep_gemm_mega
... one per rank, eight in total, world=8 on every line
```

`world` must equal your EP size — eight here. A line with `world=4` means the
run is not the configuration you think it is.

The **native** run must print no `[fi_moe_ep]` line at all. If it does, the
predicate is mis-routing and the comparison means nothing.

`compare_outputs.py` reports how many of the 8 prompts matched exactly and the
mean |dlogprob|. Typical: fi_dg vs native around **1/8 exact, mean |dlogprob|
0.02–0.06**; fi_cutedsl vs native **1/8 exact, 0.016–0.13**, wider because it is
cross-checkpoint and doubly quantized.

> **Do not treat exact-match counts as a gate.** native and fi_dg are separate
> kernel implementations, so bit-exactness is not guaranteed across builds or
> hardware — and one flipped logit early in a greedy decode diverges the whole
> rest of that sequence, which turns a near-identical run into a low
> exact-match count. What this tier proves is *routing*: the `[fi_moe_ep]`
> banner above, present on every fi rank and absent from native. Numerical
> correctness is tier 3, where Flash scores 0.965 on all three backends.

To produce that fi_cutedsl row, add a third smoke. **`smoke_infer.py` does not
resolve the checkpoint per backend** the way `bench_offline.py` and
`eval_gsm8k.py` do — its `--model` defaults to `$MODEL` regardless of
`MOE_BACKEND`, so the NVFP4 path must be passed explicitly or you will quietly
benchmark fi_cutedsl on the mx checkpoint:

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  ENFORCE_EAGER=1 MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl \
  python smoke_infer.py --tag fi_cutedsl --model $MODEL_NVFP4_FLASH \
    --out results/pr_fi_cutedsl.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python compare_outputs.py results/pr_native.json results/pr_fi_cutedsl.json'
```

### 4b. Tier 3 — GSM8K, the cross-checkpoint gate

The native-vs-fi_cutedsl smoke above compares two *different* checkpoints, so
its logprob delta is not a pass/fail signal — `eval_gsm8k.py` is. It boots one
engine per backend, scores 200 GSM8K questions, and records which checkpoint it
loaded. native and fi_cutedsl must land in the same band before any §3d ratio
is an apples-to-apples claim.

> **Do not export `MODEL` around this gate — it silently disarms it.**
> `resolve_model` ranks `--model` > `$MODEL` > per-backend default, so a
> `MODEL=<mx path>` in the environment sends *every* backend to the mx
> checkpoint, including `fi_cutedsl`. The gate then compares the mx weights
> against themselves, scores a comfortable pass, and tests nothing — the
> fi_cutedsl row would then carry the mx checkpoint under an NVFP4 label. Pass
> `--model` explicitly and check the `model` field the eval records in each
> result JSON.

```bash
JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  python eval_gsm8k.py --tag native --model $MODEL_MX_FLASH \
    --out results/gsm8k_native.json'

JOBID=$JOBID bash $W/in_container.sh 'source venv0251/bin/activate && \
  MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl \
  python eval_gsm8k.py --tag fi_cutedsl --model $MODEL_NVFP4_FLASH --min-acc 0.93 \
    --out results/gsm8k_fi_cutedsl.json'
```

`--min-acc` exits 2 below threshold. The eval also records `truncated` — how
many completions hit `--max-tokens` — because a chain cut off mid-reasoning
still ends in *a* number and so scores as a confident wrong answer, not as
unparseable. A low accuracy with a high `truncated` is a token-budget problem,
not a model problem; `--min-acc 0.93` is calibrated for DSV4-Flash and is not
automatically the right threshold for a model that reasons longer.

Both models and both checkpoints in one job (~35 min, its own 8-GPU node):

```bash
cd $W && sbatch -A <account> -p <partition> job_gsm8k_flash_pro.sh
```

All four `MODEL_*` paths from §1.3 reach it through `--export=ALL`, and it
checks them at submit time. It needs the optional Pro pair too, so if you
skipped those, run the two Flash cells by hand as above instead.

It runs six cells — native / fi_dg / fi_cutedsl for each model, all at TP8 —
and its summary prints the checkpoint each row actually loaded, flagging any
fi_cutedsl row that did not run an nvfp4 path. Expected numbers:
[expected_results.md](expected_results.md) §4.

## 5. Things that will bite you

**`FI_MOE_EP` is now a hard error.** Any non-empty `FI_MOE_EP` or
`FI_MOE_EP_MEGAKERNEL` aborts at startup — including `FI_MOE_EP=0`, and
including when the backend is native. Deliberate: under the old mechanism a
stale export silently changed which path ran, so leftovers could produce native
numbers labelled "fi". Old shells and older harness scripts set them, so clear
your environment if you have used an earlier version of this tooling.

**EPLB is rejected, not ignored.** `--enable-eplb` with any fi backend raises at
startup. The FlashInfer experts neither apply the logical-to-physical expert map
nor report per-expert load, so a rebalance would move weights without moving
routing. Use `deep_gemm_mega_moe` if you need EPLB.

**Capture all recurring step shapes** (`MAX_CAPTURE=4096` for decode). Otherwise
eager prefill chunks leak into decode rounds and fi decode looks falsely slow.

**But never `MAX_CAPTURE` without `CAPTURE_SIZES`.** The dense default capture
ladder makes vLLM's cudagraph memory profiler reserve ~48 GiB/GPU for the
flashinfer backends against a real cost of ~6 GiB, and the difference comes
out of the KV cache. The engine then holds a fraction of the sequences you
asked for, so the backend reads *fast per step and slow overall* — better ITL
than native, far worse throughput — and fi_cutedsl, whose NVFP4 weights are the
largest, is the first to fail outright with "No available memory for the cache
blocks". Every cell in §3c pins it. Full mechanism and the measured numbers:
[expected_results.md](expected_results.md) §5.1.

**A failed cell reads `MISSING`, not a wrong number.** Each sweep stamps a
start time and its summary only accepts result JSONs written after it, so a cell
that dies leaves `MISSING` in the table rather than silently reprinting the
committed result from a previous run. If you see `MISSING`, the per-cell output
above it has the traceback. A job can exit 0 with cells missing — read the
summary, not the exit code.

**A noisy node produces plausible-but-wrong microbenchmark numbers.** The
harness reports p50 over the timed iterations, and interference inflates it
without touching `e2e_us_min`. Compare the two columns before trusting a CSV:

```bash
python -c "
import csv,sys
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        p50, mn = float(r['e2e_us_p50']), float(r['e2e_us_min'])
        if mn and p50/mn > 1.5:
            print('%s %s tok=%s p50=%.1f min=%.1f (%.1fx)' %
                  (f.split('/')[-1], r['compute_kernel'], r['tokens_per_rank'], p50, mn, p50/mn))
" model_shapes/results_ep8/*.csv
```

A clean run prints nothing. Anything above ~1.5x is contaminated, and it does
not look like an error — it looks like a kernel that got slower at one size,
which is exactly the shape of a real finding. Rerun the shape on another node
rather than reasoning about the number.

**The venv keeps whatever was applied last.** `apply.sh` writes into the
installed wheel, so the venv reflects the last patch applied to it rather than
whatever you have checked out. For a full revert, copy `kernel.py.orig` and
`model.py.orig` back over the installed files.

**Teardown tracebacks are cosmetic.** On DSL 4.5.2 `worker.shutdown()` imports
`CuMemAllocator`, which trips over tilelang's `libcudart_stub.so` missing
`cudaDeviceReset`. They appear after results are written.

## 6. Not covered

* No multi-node run. Single node, TP8+EP8.
* Building the vLLM PR from source is unverified; everything here patches a
  0.25.1 wheel (§1.4e).
* `--min-acc 0.93` is calibrated for DSV4-Flash. **V4-Pro scores ~0.88 on all
  three backends and will fail that gate**, and it is not a token-budget
  artifact — raising `--max-tokens` to 2048 does not recover it. Since all
  three backends agree, it is a property of the model and this eval, not of
  `moe_ep`; what gates a perf claim is the native-vs-fi_cutedsl delta, not the
  absolute.
* Only EP8 is measured here. The microbenchmark and both sweeps run at world
  size 8 throughout; a different world size is a different measurement (§2b
  item 5), and no EP4 numbers are carried on this branch.
* The `vllm_*` microbenchmark sections (split-path baselines) are not run —
  they need `vllm==0.20.0`, which the image does not ship (§2c).
