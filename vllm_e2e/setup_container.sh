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
# 0) venv ships pip 24.0 whose resolver crashes (TypeError ... NoneType) on the
#    NGC dist metadata visible through system-site-packages; upgrade first.
python -m pip install -q --upgrade pip

# 1) vLLM 0.25.1 wheel + its dep closure (torch 2.11 etc. land in the venv).
#    vllm's compiled ops use the stable libtorch ABI (_C_stable_libtorch), so
#    the exact torch minor doesn't have to match the container's NGC torch.
python -m pip install vllm==0.25.1

# 2) flashinfer branch (editable, JIT) — replaces the flashinfer-python==0.6.13
#    wheel vllm just pulled in. --no-deps: the dep closure is already satisfied
#    and the full resolve trips over NGC system-site metadata.
python -m pip uninstall -y -q flashinfer-python || true
BUILD_NIXL_EP=0 python -m pip install --no-build-isolation --no-deps -e "$REPO"

# 3) CuTe-DSL runtime: vllm pins nvidia-cutlass-dsl==4.5.2, and since the
#    MR!27 mainloop WAR (fi branch 4_5_2-perf-fix, 2026-07-22) the cutedsl
#    mega kernels run at full 4.6.1 parity on 4.5.2 (TUNING.md "CuTe-DSL
#    runtime sensitivity" follow-up) — so the default is now to KEEP vllm's
#    own pin, which drops the entire 4.6.1 compat chain (quack 0.6.1,
#    tvm-ffi 0.1.11, tilelang 0.1.12, ThrMma rename patch) below.
#    NOTE: a venv built pre-WAR already contains 4.6.1 and a plain re-run
#    won't downgrade it — use FRESH=1 to rebuild on vllm's 4.5.2. The e2e
#    stack on native 4.5.2 has not yet been revalidated end-to-end (runs
#    37-40 used 4.6.1); DSL_461=1 restores the validated pre-WAR path:
if [[ "${DSL_461:-0}" == "1" ]]; then
    # vllm's DSV4 sparse-MLA indexer imports quack, and the quack version vllm
    # resolves (0.3.x) breaks on dsl>=4.6 (cute.core.ThrMma removed) — upgrade
    # quack to 0.6.1 (imports fine on 4.6.1 despite its ==4.6.0 metadata pin),
    # then force every dsl component to 4.6.1.
    python -m pip install --upgrade "quack-kernels==0.6.1"
    python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]==4.6.1"
    # dsl 4.6.1's tvm_ffi_provider needs make_kwargs_wrapper(map_dataclass_to_tuple=),
    # absent from the apache-tvm-ffi 0.1.9 wheel vllm resolves; tilelang (vllm's mhc
    # kernels) breaks on tvm-ffi 0.1.12's new registry. The intersection that keeps
    # dsl 4.6.1 AND tilelang alive: tvm-ffi 0.1.11 + tilelang 0.1.12 (vllm pins
    # tilelang==0.1.9 — resolver warning expected).
    python -m pip install --upgrade "apache-tvm-ffi==0.1.11" "tilelang==0.1.12"

    # 3b) vLLM's vendored CuTe kernels (vllm_flash_attn/cute, third_party/
    #     fmha_sm100) still reference cute.core.ThrMma, which 4.6 moved to
    #     cutlass.cute.ThrMma (cute/atom.py). Pure rename — patch in place.
    #     (Not needed on 4.5.2, where cute.core.ThrMma still exists.)
    VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
    grep -rl "cute\.core\.ThrMma" --include="*.py" "$VLLM_DIR" 2>/dev/null | \
        while read -r f; do sed -i "s/cute\.core\.ThrMma/cute.ThrMma/g" "$f"; done
    find "$VLLM_DIR/vllm_flash_attn" "$VLLM_DIR/third_party" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi

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
check("vllm custom ops", lambda: bool(importlib.import_module("vllm._custom_ops")) and "loaded")
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
