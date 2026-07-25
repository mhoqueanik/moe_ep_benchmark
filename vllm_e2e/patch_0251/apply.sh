#!/bin/bash
# Apply the fi moe_ep integration to an installed vLLM 0.25.1.
# Idempotent: backs up pristine files as *.orig on first run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
VER="$(python3 -c 'import vllm; print(vllm.__version__)')"
if [[ "$VER" != 0.25.1* ]]; then
    echo "WARNING: patch was ported for vLLM 0.25.1, found $VER" >&2
fi

DST="$VLLM_DIR/models/deepseek_v4/nvidia"
[[ -f "$DST/model.py.orig" ]] || cp "$DST/model.py" "$DST/model.py.orig"
cp "$HERE/model.py" "$DST/model.py"
# The helpers moved to vllm/utils/ alongside flashinfer.py and deep_gemm.py.
cp "$HERE/flashinfer_moe_ep.py" "$VLLM_DIR/utils/flashinfer_moe_ep.py"
# Drop the pre-move copy so a stale one cannot shadow the new location.
rm -f "$DST/fi_utils.py"

# The flashinfer path is selected by backend string, and KernelConfig rejects
# any moe_backend outside the MoEBackend Literal before the model ever sees
# it -- so both flashinfer_moe_ep_mega_* names have to be registered in the
# installed config too, not just handled in the model.
#
# Done as an in-place insertion rather than shipping a whole kernel.py: that
# file is core config and changes between vLLM releases, so a full-file copy
# would silently roll the rest of it back to whatever version this patch was
# snapshotted from. The only thing needed here is two lines in one Literal.
CFG="$VLLM_DIR/config"
[[ -f "$CFG/kernel.py.orig" ]] || cp "$CFG/kernel.py" "$CFG/kernel.py.orig"
python3 - "$CFG/kernel.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
src = path.read_text()
backends = (
    "flashinfer_moe_ep_mega_deep_gemm",
    "flashinfer_moe_ep_mega_cutedsl",
)
missing = [b for b in backends if f'"{b}"' not in src]
if not missing:
    print("kernel.py: backends already registered (no-op)")
    raise SystemExit(0)

# Insert at the end of the MoEBackend literal specifically. Anchoring on a
# member name is not enough -- e.g. "flashinfer_b12x" appears in both
# MoEBackend and LinearBackend, so a member-based anchor is ambiguous.
marker = "MoEBackend = Literal["
start = src.find(marker)
if start == -1:
    sys.exit(f"kernel.py: {marker!r} not found -- update patch_0251/apply.sh")
end = src.find("\n]", start)
if end == -1:
    sys.exit(f"kernel.py: unterminated {marker!r} -- update patch_0251/apply.sh")
if '"auto"' not in src[start:end]:
    sys.exit("kernel.py: MoEBackend literal looks wrong (no 'auto' member)")

path.write_text(
    src[: end + 1] + "".join(f'    "{b}",\n' for b in missing) + src[end + 1 :]
)
print("kernel.py: registered " + ", ".join(missing))
PY

# Drop stale bytecode so the patched sources are what actually imports.
find "$DST" "$CFG" "$VLLM_DIR/utils" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "patched: $DST (backup: model.py.orig)"
echo "patched: $CFG/kernel.py (backup: kernel.py.orig)"
