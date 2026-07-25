#!/bin/bash
# V4-Pro NVFP4: download the latest PRE-REWRITE (prequantized, quant_algo=None,
# per-expert keys) revision, analogous to Flash's 48bfe38. The CI mirror only
# has post-rewrite copies (dequant-fallback risk).
set -uo pipefail
export ROOT=${ROOT:?set ROOT to your checkout root (the dir holding moe_ep_benchmark/, the container image and checkpoints/)}
export HF_HOME=$ROOT/.cache/huggingface
export HF_HUB_DISABLE_XET=1
CKPT=$ROOT/checkpoints
source $ROOT/hfdl_venv/bin/activate
REPO=nvidia/DeepSeek-V4-Pro-NVFP4

# Resolve the full sha of the newest commit whose hf_quant_config.json is still
# the good prequantized schema (quant_algo is None, per-expert-tensor keys).
REV=$(python - <<'PY'
from huggingface_hub import list_repo_commits, hf_hub_download
import json
repo="nvidia/DeepSeek-V4-Pro-NVFP4"
best=None
for c in list_repo_commits(repo):          # newest-first
    try:
        p=hf_hub_download(repo,"hf_quant_config.json",revision=c.commit_id)
        q=json.load(open(p))["quantization"]
        k=next(iter(q["quantized_layers"]))
        if q.get("quant_algo") is None and k.count(".")==5:
            best=c.commit_id; break         # newest good one
    except Exception:
        pass
print(best or "")
PY
)
echo "resolved good NVFP4-Pro revision: $REV"
[[ -z "$REV" ]] && { echo "no good revision found"; exit 3; }
echo "$REV" > $ROOT/checkpoints/.pro_nvfp4_rev

hf download "$REPO" --revision "$REV" --local-dir "$CKPT/deepseek-v4-pro-nvfp4"
echo "=== hf download exit: $? ==="
