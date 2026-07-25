#!/bin/bash
# EP8 knob retune (RUNBOOK_8GPU_SM100 §7). The shipped knob cache is EP4-tuned;
# at EP8 each rank holds 32/256 experts, changing the winning tiles. Writes
# knob_cache_ep8.json. NB: the 8gpu doc's `torchrun -np 8` is wrong; tune.py's
# own docstring uses --nproc_per_node.
set -uo pipefail
export ROOT=${ROOT:?set ROOT to your checkout root (the dir holding moe_ep_benchmark/, the container image and checkpoints/)}
export IMG=$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh
export W=$ROOT/moe_ep_benchmark/vllm_e2e
export JOBID=$(cat "$ROOT/holdjob.id")

bash "$W/in_container.sh" "
set -uo pipefail
source venv0251/bin/activate
export FI_MOE_EP_SKIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_ep8.json
echo '##### EP8 tune (nvfp4, 8192 bucket, world=8) #####'
torchrun --nproc_per_node=8 -m flashinfer.moe_ep.tune --dtype nvfp4 \
    --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 \
    --max-tokens 8192
echo '##### EP8 TUNE DONE (rc='\$?') #####'
ls -l $W/results/knob_cache_ep8.json 2>&1
"
