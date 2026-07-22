#!/bin/bash
# Apply the DeepSeek V3.2 fi moe_ep patch into the installed vllm wheel.
# Mirrors patch_0251/apply.sh: copies the patched model.py + fi_experts.py
# into venv0251's vllm/models/deepseek_v32/nvidia/ (model.py.orig backup
# must already exist or is created from the pristine file).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VL=${VL:-$HERE/../venv0251/lib/python3.12/site-packages/vllm}
DST=$VL/models/deepseek_v32/nvidia
[ -f $DST/model.py.orig ] || cp $DST/model.py $DST/model.py.orig
cp $HERE/fi_experts.py $DST/fi_experts.py
cp $HERE/model.py $DST/model.py
echo "applied patch_v32 -> $DST"
