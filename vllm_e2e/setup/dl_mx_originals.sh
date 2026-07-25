#!/bin/bash
# The mx-format originals that native and fi_dg run -- the denominator of every
# ratio in expected_results.md. Public and ungated; the /lustre/share CI mirror
# is only a local cache of these same trees.
#
# The NVFP4 casts are separate: dl_nvfp4_flash.sh / dl_nvfp4_pro.sh.
#
# PIN THE REVISIONS. Both repos have advanced past what was measured, and for
# the NVFP4 side resolving to main is actively wrong (see dl_nvfp4_pro.sh) --
# so the habit of passing --revision is worth keeping here too, where it merely
# means "the tree the numbers came from" rather than "the tree that works".
set -uo pipefail
export ROOT=${ROOT:?set ROOT to your checkout root (the dir holding checkpoints/)}
export HF_HOME=$ROOT/.cache/huggingface
# huggingface_hub >=1.x ignores HF_HUB_ENABLE_HF_TRANSFER and defaults to Xet,
# which is CPU-heavy and gets reaped on shared login nodes (SIGKILL/137 at
# ~18GB with hundreds of GB free -- not an OOM). Plain HTTPS is resumable.
export HF_HUB_DISABLE_XET=1
CKPT=$ROOT/checkpoints
mkdir -p "$CKPT"

WHICH=${1:-both}   # flash | pro | both

FLASH_REV=6e763230a9d263eca2023f1d4a5ce1bfe126cf48   # 46 shards, 148.6 GiB
PRO_REV=0366e4e                                       # 64 shards, ~805 GiB

if [[ $WHICH == flash || $WHICH == both ]]; then
    echo "=== DeepSeek-V4-Flash mx @ $FLASH_REV ==="
    hf download deepseek-ai/DeepSeek-V4-Flash --revision "$FLASH_REV" \
        --local-dir "$CKPT/deepseek-v4-flash"
    echo "=== exit $? ==="
fi

if [[ $WHICH == pro || $WHICH == both ]]; then
    echo "=== DeepSeek-V4-Pro mx @ $PRO_REV ==="
    hf download deepseek-ai/DeepSeek-V4-Pro --revision "$PRO_REV" \
        --local-dir "$CKPT/deepseek-v4-pro"
    echo "=== exit $? ==="
fi

echo
echo "Point the harness at them (do NOT export MODEL around eval_gsm8k.py --"
echo "it overrides the per-backend NVFP4 default and disarms the gate):"
echo "  export MODEL_MX_FLASH=$CKPT/deepseek-v4-flash"
echo "  export MODEL_MX_PRO=$CKPT/deepseek-v4-pro"
for d in "$CKPT/deepseek-v4-flash" "$CKPT/deepseek-v4-pro"; do
    [[ -d $d ]] && echo "  $(basename $d): $(ls $d/model-*.safetensors 2>/dev/null | wc -l) shards, index $([[ -f $d/model.safetensors.index.json ]] && echo present || echo MISSING)"
done
