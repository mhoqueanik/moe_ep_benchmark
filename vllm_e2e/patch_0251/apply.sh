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
cp "$HERE/fi_utils.py" "$DST/fi_utils.py"
# Drop stale bytecode so the patched sources are what actually imports.
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "patched: $DST (backup: model.py.orig)"
