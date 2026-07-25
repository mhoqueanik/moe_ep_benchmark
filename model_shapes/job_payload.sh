#!/bin/bash
# In-container payload for one model-shape sweep job (see submit_jobs.sh).
# Mirrors RUNBOOK_REPRO.md §2: editable-install the branch, upgrade CuTe-DSL,
# sanity-check the import path, then run the shape sweep.
set -uo pipefail

ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
REPO="${REPO:-$ROOT/flashinfer-2/flashinfer-moe_ep}"
BENCH="${BENCH:-$ROOT/moe_ep_benchmark}"

export FLASHINFER_DISABLE_VERSION_CHECK=1

cd "$REPO"
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . \
    2>&1 | tail -2
# Pin the DSL: the cutedsl kernels' codegen is version-sensitive (pre-4.5.2
# compiled them 34-54% slower), so an unpinned --upgrade makes a sweep
# unattributable. 4.5.2 is vLLM 0.25.1's own pin and the e2e provenance.
DSL_VERSION="${DSL_VERSION:-4.5.2}"
python -m pip install "nvidia-cutlass-dsl[cu13]==${DSL_VERSION}" 2>&1 | tail -2
python -c "from importlib.metadata import version; v=version('nvidia-cutlass-dsl'); \
assert v=='${DSL_VERSION}', f'DSL {v} != ${DSL_VERSION}'; print(f'GUARD PASS: cutlass-dsl {v}')" || exit 1

python - <<'PY'
import flashinfer, flashinfer.moe_ep
for m in (flashinfer, flashinfer.moe_ep):
    print("import check:", m.__name__, "->", m.__file__)
    assert m.__file__.startswith("/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer-2"), m.__file__
PY

GPUS="${GPUS:-8}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    bash "$BENCH/model_shapes/run_model_shapes.sh"
