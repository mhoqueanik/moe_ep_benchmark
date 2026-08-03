#!/bin/bash
# In-container payload for one model-shape sweep job (see submit_jobs.sh).
# Mirrors RUNBOOK_REPRO.md §2: editable-install the branch, upgrade CuTe-DSL,
# sanity-check the import path, then run the shape sweep.
set -uo pipefail

ROOT="${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}"
REPO="${REPO:-$ROOT/flashinfer-moe_ep}"
BENCH="${BENCH:-$ROOT/moe_ep_benchmark}"

export FLASHINFER_DISABLE_VERSION_CHECK=1

cd "$REPO"
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . \
    2>&1 | tail -2
# Pin the DSL: 4.5.2 is vLLM 0.25.1's own pin and what the flashinfer
# 4_5_2-perf-fix branch is validated against -- the two move together. An
# unpinned --upgrade makes a sweep unattributable.
DSL_VERSION="${DSL_VERSION:-4.5.2}"
# Derive the cuXX extra from torch rather than hardcoding cu13, matching
# build_flashinfer_ep_pytorch.sh: moe_ep itself is CUDA-major agnostic and
# works on 12 and 13. Override with CUDA_MAJOR=<n>.
CU="cu${CUDA_MAJOR:-$(python -c 'import torch; v=torch.version.cuda or ""; print(v.split(".")[0])')}"
python -m pip install "nvidia-cutlass-dsl[$CU]==${DSL_VERSION}" 2>&1 | tail -2
python -c "from importlib.metadata import version; v=version('nvidia-cutlass-dsl'); \
assert v=='${DSL_VERSION}', f'DSL {v} != ${DSL_VERSION}'; print(f'GUARD PASS: cutlass-dsl {v}')" || exit 1

# Guards against picking up a wheel-installed flashinfer instead of the editable
# checkout. Compares against $REPO rather than a baked-in path, so it travels.
FI_PARENT="$REPO" python - <<'PY'
import os
import flashinfer, flashinfer.moe_ep
want = os.environ["FI_PARENT"]
for m in (flashinfer, flashinfer.moe_ep):
    print("import check:", m.__name__, "->", m.__file__)
    assert m.__file__.startswith(want), "%s is not under %s" % (m.__file__, want)
PY

GPUS="${GPUS:-4}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    bash "$BENCH/model_shapes/run_model_shapes.sh"
