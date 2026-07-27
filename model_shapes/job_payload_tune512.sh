#!/bin/bash
# In-container payload for the tune512 job (see submit_tune512.sh).
# Same prep as job_payload.sh with two differences, both prompted by the
# shipped-sweep logs (jobs 2337xxx/2338xxx):
#   * the editable-install output is kept (tail -5) instead of hidden — in
#     those logs the install silently failed to take precedence and the
#     import-path guard's failure was non-fatal;
#   * the guard here asserts the pinned COMMIT of whatever checkout
#     flashinfer actually imports (fatal), rather than its path. Two
#     checkouts of 4_5_2-perf-fix exist under $ROOT; identical bytes at the
#     pin are what makes a number attributable, not which clone they came
#     from.
set -uo pipefail

ROOT="${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}"
REPO="${REPO:-$ROOT/flashinfer-2/flashinfer-moe_ep}"
BENCH="${BENCH:-$ROOT/moe_ep_benchmark}"
EXPECTED_FI_COMMIT="${EXPECTED_FI_COMMIT:-1ee41bcd}"

export FLASHINFER_DISABLE_VERSION_CHECK=1

cd "$REPO"
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . \
    2>&1 | tail -5
# Pin the DSL: 4.5.2 is vLLM 0.25.1's own pin and what the flashinfer
# 4_5_2-perf-fix branch is validated against -- the two move together.
DSL_VERSION="${DSL_VERSION:-4.5.2}"
CU="cu${CUDA_MAJOR:-$(python -c 'import torch; v=torch.version.cuda or ""; print(v.split(".")[0])')}"
python -m pip install "nvidia-cutlass-dsl[$CU]==${DSL_VERSION}" 2>&1 | tail -2
python -c "from importlib.metadata import version; v=version('nvidia-cutlass-dsl'); \
assert v=='${DSL_VERSION}', f'DSL {v} != ${DSL_VERSION}'; print(f'GUARD PASS: cutlass-dsl {v}')" || exit 1

# Fatal commit guard on the checkout the interpreter ACTUALLY imports.
EXPECTED_FI_COMMIT="$EXPECTED_FI_COMMIT" python - <<'PY' || exit 1
import os, subprocess
import flashinfer, flashinfer.moe_ep

want = os.environ["EXPECTED_FI_COMMIT"]
top = os.path.dirname(os.path.dirname(flashinfer.__file__))
# root-in-container inspecting a user-owned repo trips git's ownership check
env = dict(os.environ, GIT_CONFIG_COUNT="1",
           GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*")
head = subprocess.check_output(
    ["git", "-C", top, "rev-parse", "HEAD"], text=True, env=env).strip()
# untracked files (e.g. the editable install's egg-info) do not matter for
# attribution; tracked modifications do.
dirty = subprocess.check_output(
    ["git", "-C", top, "status", "--porcelain", "--untracked-files=no"],
    text=True, env=env).strip()
print(f"import check: flashinfer -> {flashinfer.__file__}")
print(f"import check: git HEAD {head[:12]} ({'dirty' if dirty else 'clean'}) at {top}")
assert head.startswith(want), f"flashinfer HEAD {head[:12]} != pinned {want}"
assert not dirty, f"flashinfer checkout at {top} has local modifications:\n{dirty}"
print("GUARD PASS: flashinfer commit")
PY

# DRIVER selects the in-container driver (run_tune512.sh or
# run_tune512_schedule.sh); both take the same SHAPE_NAME/TOKENS/OUT_DIR env.
GPUS="${GPUS:-8}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    bash "$BENCH/model_shapes/${DRIVER:-run_tune512.sh}"
