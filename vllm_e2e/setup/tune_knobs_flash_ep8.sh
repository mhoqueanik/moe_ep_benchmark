#!/bin/bash
# EP8 knob retune for DeepSeek-V4-Flash (RUNBOOK_REPRO.md §3a). The branch
# already ships the result as results/knob_cache_ep8.json, so this only needs
# rerunning if your geometry or world size differs -- at EP8 each rank holds
# 32 of 256 experts, which changes the winning tiles.
#
#   ROOT=... IMG=... JOBID=<hold job> bash tune_knobs_flash_ep8.sh
#
# Synthetic weights, no checkpoint required. ~10 min.
set -uo pipefail
export ROOT=${ROOT:?set ROOT to your checkout root}
export IMG=${IMG:?set IMG to the container image built in §1.2c}
export JOBID=${JOBID:?set JOBID to the §1.4a hold job id}
export W=$ROOT/moe_ep_benchmark/vllm_e2e

bash "$W/in_container.sh" "
set -uo pipefail
source venv0251/bin/activate
export FI_MOE_EP_SKIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_ep8.json
echo '##### Flash EP8 tune (nvfp4, 8192 bucket, world=8) #####'
torchrun --nproc_per_node=8 -m flashinfer.moe_ep.tune --dtype nvfp4 \
    --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 \
    --max-tokens 8192
echo '##### Flash EP8 TUNE DONE (rc='\$?') #####'
ls -l $W/results/knob_cache_ep8.json 2>&1
"
