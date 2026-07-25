#!/bin/bash
# From-scratch runbook §2: fetch the PINNED NVFP4 checkpoint from HF Hub.
# Attempts anonymous download first (repo is public/ungated; hf auth login in the
# runbook only dodges the shared-egress rate limit). If it 429s, we fall back to
# asking the user for a token.
set -uo pipefail
export ROOT=${ROOT:?set ROOT to your checkout root (the dir holding moe_ep_benchmark/, the container image and checkpoints/)}
export HF_HOME=$ROOT/.cache/huggingface
export CKPT=$ROOT/checkpoints
mkdir -p "$CKPT"

VENV=$ROOT/hfdl_venv
if [[ ! -f "$VENV/bin/activate" ]]; then
  /usr/bin/python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -U "huggingface_hub[cli,hf_transfer]"
# The Xet high-performance path is CPU-heavy and got SIGKILLed by the login-node
# arbiter at ~18GB. Disable it: plain HTTPS range downloads are low-CPU. Resumable.
export HF_HUB_DISABLE_XET=1
export HF_XET_HIGH_PERFORMANCE=0

echo "=== hf version ==="; hf version 2>&1 || huggingface-cli version 2>&1

echo "=== downloading nvidia/DeepSeek-V4-Flash-NVFP4 @ 48bfe38 (pinned) ==="
hf download nvidia/DeepSeek-V4-Flash-NVFP4 \
    --revision 48bfe38c62be14e8d82f9e3be12fe5d30a2e38c8 \
    --local-dir "$CKPT/deepseek-v4-flash-nvfp4"
rc=$?
echo "=== hf download exit code: $rc ==="
exit $rc
