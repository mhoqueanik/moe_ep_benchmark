#!/bin/bash
# In-container payload for one model-shape sweep job (see submit_jobs.sh).
# Mirrors RUNBOOK.md §2: editable-install the branch, upgrade CuTe-DSL,
# sanity-check the import path, then run the shape sweep.
set -uo pipefail

ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
REPO="${REPO:-$ROOT/flashinfer-2/flashinfer-moe_ep}"
BENCH="${BENCH:-$ROOT/moe_ep_benchmark}"

export FLASHINFER_DISABLE_VERSION_CHECK=1

cd "$REPO"
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . \
    2>&1 | tail -2
python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]" 2>&1 | tail -2

python - <<'PY'
import flashinfer, flashinfer.moe_ep
for m in (flashinfer, flashinfer.moe_ep):
    print("import check:", m.__name__, "->", m.__file__)
    assert m.__file__.startswith("/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer-2"), m.__file__
PY

GPUS="${GPUS:-4}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    bash "$BENCH/model_shapes/run_model_shapes.sh"
