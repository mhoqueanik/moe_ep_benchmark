# MoE-EP microbenchmark runbook (SLURM + pyxis container)

How to run the `moe_ep_benchmark/` microbenchmarks against a checkout of
`flashinfer-2/flashinfer-moe_ep` on a Blackwell (sm_100 / GB200) node.

See `README.md` for the full knob reference. This runbook is the "just run it"
recipe plus the container/install steps that aren't obvious.

---

## 0. What runs where

| Section | Script | Needs |
|---------|--------|-------|
| `fi_mega`    | `bench_moe_ep_mega.py`      | FlashInfer branch only (deep_gemm, nvshmem, cutedsl) |
| `vllm_mega`  | `bench_moe_ep_vllm_mega.py` | **`vllm==0.20.0`** + deep_gemm `fp8_fp4_mega_moe` |
| `vllm_split` | `bench_moe_ep_nonmega.py`   | **`vllm==0.20.0`** + DeepEP |

`fi_mega` is the section that exercises the new cutedsl kernels and needs no
vLLM. The two `vllm_*` sections are comparison baselines and require installing
`vllm==0.20.0` (not present in the base image) — skip them unless you need the
baseline.

---

## 1. Environment

```bash
export ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
export IMG=$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh
export REPO=$ROOT/flashinfer-2/flashinfer-moe_ep      # flashinfer checkout under test
```

The image already provides torch 2.12, deep_gemm, triton, nvshmem, cutlass,
cuda.bindings — verified importable. It does **not** ship vLLM.

---

## 2. Run the `fi_mega` benchmark (one command)

Editable-install the branch inside the container, then run `fi_mega`. The
editable install replaces the image's `flashinfer-python` so imports resolve to
`$REPO` (verify with `flashinfer.__file__`). The install lives in the container
overlay and is **not persistent** — rerun it each fresh container.

```bash
srun -A coreai_libraries_cudnn -p batch -N 1 --ntasks-per-node=1 --time=02:00:00 \
  --job-name=coreai_libraries_cudnn-fi.bench \
  --container-image="$IMG" \
  --container-mounts="$ROOT:$ROOT" \
  --container-workdir="$REPO" \
  bash -lc '
    set -uo pipefail
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    # 1) editable install of the branch (runbook §2a install; no build isolation)
    PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e .
    # 1b) upgrade the CuTe-DSL runtime the cutedsl mega kernels need (cu13
    #     build; >=4.5.2 — 4.5.2 is at parity since the MR!27 WAR, 2026-07-22)
    python -m pip install --upgrade nvidia-cutlass-dsl[cu13]
    # 2) sanity: flashinfer must resolve to the checkout, not the image copy
    python -c "import flashinfer; print(flashinfer.__file__)"
    # 3) run the benchmark
    SECTION=fi_mega GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
      bash '"$ROOT"'/moe_ep_benchmark/run.sh
  ' 2>&1 | tee $ROOT/logs_fi/bench_fi_mega_$(date +%Y%m%d_%H%M%S).log
```

Notes:
- Alternative to the editable install: skip step 1 and set
  `PYTHONPATH="$REPO:${PYTHONPATH:-}"` so imports pick up the checkout. Produces
  the same numbers, but the editable install is the canonical path (per the
  compat runbook `runbook_compat_mega_moe_integration.md` §2a).
- `run.sh` uses absolute `$HERE` paths, so cwd doesn't matter; `bench_common`
  resolves via the script dir automatically.

> **CRITICAL — the image's baked flashinfer points at `/host/flashinfer`.**
> The base image has an editable install whose `Editable project location` is
> `/host/flashinfer`, a path we do **not** mount (we mount `$REPO`). So in a
> plain container run with no `pip install -e .` and no `PYTHONPATH`,
> `import flashinfer` **fails outright** (`ModuleNotFoundError`) — there is no
> stale/wrong flashinfer that silently shadows the branch. flashinfer can only
> come from `$REPO`, via either step 1 (editable reinstall re-points it to
> `$REPO`) or the `PYTHONPATH` alternative above. The editable install is **not
> persistent** across fresh containers — redo step 1 each time.

### Verify the mega path really comes from `$REPO`

Run from `/tmp` (so cwd can't shadow) and confirm every module prints a path
under `$REPO`:

```bash
cd /tmp && python - <<'PY'
import importlib
REPO="/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer-2/flashinfer-moe_ep"
for name in ["flashinfer","flashinfer.moe_ep","flashinfer.moe_ep.layer",
             "flashinfer.moe_ep.core.kernel.registry",
             "flashinfer.moe_ep.kernel_src.cutedsl_megamoe",
             "flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.staging",
             "flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging",
             "flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.staging"]:
    f = importlib.import_module(name).__file__
    print(("OK  " if f.startswith(REPO) else "!!! "), name, "->", f)
PY
python -m pip show flashinfer-python | grep -Ei "Version|Editable"
```

`deep_gemm` is a separate dist-package under
`/usr/local/lib/python3.12/dist-packages/deep_gemm` — that's expected (external
dependency, not part of the flashinfer checkout).

---

## 3. Backend / geometry knobs

Default geometry: 256 experts, top-8, hidden 7168, inter 2048, 8 tokens/rank,
4 GPUs (DP=EP=4, TP=1). Override via env (see `README.md`):

```bash
# subset of fi_mega backends
MEGA_LIST="mxfp8_cutedsl nvfp4_cutedsl" SECTION=fi_mega bash run.sh
# a single backend
MEGA_LIST=deep_gemm_mega SECTION=fi_mega bash run.sh
# problem size / hardware
TOKENS=64 GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SECTION=fi_mega bash run.sh
# token sweep
SECTION=fi_mega bash run_sweep.sh          # or: SEQ_LENS="1 8 64 512 4096"
# cutedsl kernel knobs: online autotune / explicit dict (see README)
MEGA_KNOBS=auto SECTION=fi_mega bash run.sh
# timed region: tester-parity bare kernel launch vs full FI forward (default e2e)
MEGA_TIMING=kernel SECTION=fi_mega bash run.sh
```

fi_mega backends: `deep_gemm_mega | mxfp8_cutedsl | nvfp4_cutedsl`.

> **Comparing against the kernel repo's tester** (`latest_cutedsl_kernel/cutedsl_megamoe`
> `-m tester.tester --mode Perf`): match BOTH the geometry and the timed region.
> The tester's problems (`nvfp4_perf.jsonl`) use (hidden, inter, experts, topk) =
> (4096, 2048, 256, 6) or (7168, 3072, 384, 6) — neither equals this benchmark's
> default (7168, 2048, 256, top-8) — and its timed region is a bare prebuilt
> kernel launch (no arg rebuild / reset / sync / output copy, iters enqueued
> back-to-back). Use `HIDDEN/INTER/NUM_EXPERTS/TOPK` + `MEGA_TIMING=kernel`
> for an apples-to-apples run.

---

## 4. vLLM baseline sections (optional, heavier)

`vllm_mega` / `vllm_split` need `vllm==0.20.0`. Inside the container:

```bash
pip install vllm==0.20.0            # vllm_mega
# vllm_split additionally needs DeepEP built/available
SECTION=vllm_mega  bash run.sh
SECTION=vllm_split bash run.sh
```

The `vllm==0.20.0` (+ DeepEP) install is non-trivial and may need extra
dependency pinning; treat as a separate task from `fi_mega`.

---

## 5. Output

Results land in `moe_ep_benchmark/results/` (override with `OUT_DIR`):

```
results/bench_<stamp>_fi_mega.{log,csv}     # run.sh
results/sweep_<stamp>_fi_mega.{log,csv}     # run_sweep.sh
```

CSV columns: `path,algo,comm_backend,compute_kernel,quant_timed,weight_dtype,
input_dtype,act_compute_dtype,tokens_per_rank,gpus,num_experts,top_k,hidden,
inter,e2e_us_p50,e2e_us_min,e2e_us_max,tok_s`.

Plot with `plot.py`.

---

## 6. Branch API note (why the port was needed)

On `new_cutedsl_kernels`, the cutedsl mega frontend symbols
(`get_symm_buffer_for_mega_moe`, `get_symm_buffer_for_mxfp8_mega_moe`,
`make_dummy_epilogue_params`) moved:

- OLD: `flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend`
- NEW: `flashinfer.moe_ep.kernel_src.cutedsl_megamoe`  (top-level package export)

Signatures are unchanged; the per-backend `.staging` modules did NOT move.
`bench_moe_ep_mega.py` was already updated for this. If a future branch moves
them again, copy the import pattern from the passing multirank tests under
`tests/moe_ep/test_moe_ep_*_cutedsl_mega_multirank.py`.

---

## 7. Reference results (2026-07-13, new_cutedsl_kernels, 4× GB200)

fi_mega, default geometry, compute-only timing (staging + weight prep excluded):

| backend          | dtype                  | p50 (µs) | tok/s   |
|------------------|------------------------|----------|---------|
| `deep_gemm_mega` | fp4_int8_block32       | 238      | 134,264 |
| `nvfp4_cutedsl`  | nvfp4_block16          | 625      | 51,195  |
| `mxfp8_cutedsl`  | mxfp8_e4m3_block32     | 686      | 46,672  |
