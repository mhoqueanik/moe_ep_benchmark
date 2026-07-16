#!/bin/bash
# Restore the pristine vLLM deepseek_v4 nvidia model.py and drop fi_utils.py.
set -euo pipefail
VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
DST="$VLLM_DIR/models/deepseek_v4/nvidia"
[[ -f "$DST/model.py.orig" ]] && cp "$DST/model.py.orig" "$DST/model.py"
rm -f "$DST/fi_utils.py"
find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "reset: $DST"
