#!/bin/bash
# One-time (per venv) setup for the vLLM 0.25.1 e2e benchmark inside the
# flashinfer-ep container ($IMG in RUNBOOK.md). Creates a persistent venv on
# lustre so container restarts don't repeat the install.
#
#   bash setup_container.sh            # create/refresh venv + patch vllm
#   FRESH=1 bash setup_container.sh    # wipe venv and start over
set -uo pipefail

ROOT=${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}
REPO=${REPO:-$ROOT/flashinfer-2/flashinfer-moe_ep}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-$HERE/venv0251}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$ROOT/.cache/pip}
export PIP_CONSTRAINT=""

[[ "${FRESH:-0}" == "1" ]] && rm -rf "$VENV"

if [[ ! -f "$VENV/bin/activate" ]]; then
    python3 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

set -x
# 1) vLLM 0.25.1 wheel + its dep closure (torch 2.11 etc. land in the venv).
python -m pip install vllm==0.25.1

# 2) flashinfer branch (editable, JIT) — overrides the flashinfer-python pin.
BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e "$REPO"

# 3) CuTe-DSL runtime >=4.6.1: vllm pins 4.5.2, which compiles the cutedsl
#    mega kernels 34-54% slower (TUNING.md "CuTe-DSL runtime sensitivity").
#    Native deep_gemm paths don't touch cutlass-dsl, so the upgrade only
#    affects the fi cutedsl kernels (where it is required).
python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]>=4.6.1"

# 4) Patch the installed vLLM with the fi moe_ep integration.
bash "$HERE/patch_0251/apply.sh"
set +x

echo "==== sanity ===="
python - <<'PY'
import importlib, traceback
def check(label, fn):
    try:
        print(f"PASS {label}: {fn()}")
    except Exception as e:
        print(f"FAIL {label}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)

import os
check("torch", lambda: __import__("torch").__version__)
check("torch.cuda built", lambda: __import__("torch").version.cuda)
check("vllm", lambda: __import__("vllm").__version__)
check("vllm._C", lambda: bool(importlib.import_module("vllm._C")) or "loaded")
check("deep_gemm", lambda: importlib.import_module("deep_gemm").__file__)
check("deep_gemm.fp8_fp4_mega_moe", lambda: bool(getattr(importlib.import_module("deep_gemm"), "fp8_fp4_mega_moe")) and "present")
check("flashinfer", lambda: importlib.import_module("flashinfer").__file__)
check("flashinfer.moe_ep", lambda: importlib.import_module("flashinfer.moe_ep").__file__)
check("moe_ep runtime helpers", lambda: bool(importlib.import_module("flashinfer.moe_ep").bootstrap_moe_ep_runtime) and "present")
check("cutlass dsl", lambda: importlib.import_module("cutlass").__version__)
check("nvshmem4py", lambda: importlib.import_module("nvshmem.core").__name__)
check("vllm fi patch", lambda: importlib.import_module("vllm.models.deepseek_v4.nvidia.fi_utils").__name__)
PY
echo "==== setup done ===="
