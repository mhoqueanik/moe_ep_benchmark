#!/bin/bash
# Resume the interrupted V4-Pro NVFP4 download. The first run
# (scratch_runbook_dl_pro_nvfp4.sh) died at 2026-07-25 06:29 with 49/64 shards
# and no model.safetensors.index.json.
#
# Pins the revision recorded by that run rather than re-resolving it: a newer
# good commit would hand us shards that do not match the 49 already on disk.
set -uo pipefail
export ROOT=${ROOT:?set ROOT to your checkout root}
export HF_HOME=$ROOT/.cache/huggingface
export HF_HUB_DISABLE_XET=1
CKPT=$ROOT/checkpoints
source $ROOT/hfdl_venv/bin/activate
REPO=nvidia/DeepSeek-V4-Pro-NVFP4
REV=$(cat "$CKPT/.pro_nvfp4_rev")
[[ -z "$REV" ]] && { echo "no pinned revision in $CKPT/.pro_nvfp4_rev"; exit 3; }
echo "resuming $REPO @ $REV"

hf download "$REPO" --revision "$REV" --local-dir "$CKPT/deepseek-v4-pro-nvfp4"
echo "=== hf download exit: $? ==="

D=$CKPT/deepseek-v4-pro-nvfp4
echo "shards: $(ls $D/model-*-of-00064.safetensors 2>/dev/null | wc -l) / 64"
ls -l $D/model.safetensors.index.json 2>&1 | tail -1
