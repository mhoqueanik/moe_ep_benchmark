#!/bin/bash
# Resume an interrupted V4-Pro NVFP4 download (dl_nvfp4_pro.sh). At 851 GB the
# pull is long enough that a killed process or a dropped connection is likely;
# hf download resumes, so re-running is safe.
#
# Uses the revision dl_nvfp4_pro.sh recorded in checkpoints/.pro_nvfp4_rev
# rather than re-resolving it. Re-resolving could pick a newer commit whose
# shards do not match the ones already on disk, leaving a mixed checkpoint.
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
