#!/bin/bash
# Skew-aware tuning ladder:
#  1) schedule sweep @ decode shapes (live=256) under measured skew 18x
#  2) schedule sweep @ prefill shapes (live=4096) under skew 18x
#  3) group discrimination: decode-shape sweeps @ skew 14x vs 28x
# Each sweep prints its winner; caches: *_skew variants (separate files).
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }
until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
sweep() { # name cache base_cache live skew
  local name=$1 cache=$2 basecache=$3 live=$4 skew=$5
  echo "=== sweep $name (live=$live skew=$skew) ==="
  rm -f "$cache"
  run "source venv0251/bin/activate && \
    FLASHINFER_MOE_EP_KNOB_CACHE=$basecache python -c \"
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import lookup_knobs
import json
k = lookup_knobs(dtype='nvfp4', world_size=4, hidden=4096, intermediate=4096,
                 num_experts=256, topk=6, max_tokens=4096)
print(json.dumps(k, default=list))\" > /tmp/base_knobs.json && \
    FLASHINFER_MOE_EP_KNOB_CACHE=$cache \
    torchrun --standalone --nproc_per_node=4 -m flashinfer.moe_ep.tune \
      --dtype nvfp4 --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 \
      --max-tokens 4096 --live-tokens $live --skew $skew \
      --sweep schedule --base-knobs \"\$(cat /tmp/base_knobs.json)\"" \
    > "$W/logs/skew_tune_$name.log" 2>&1 || rc=1
  grep -E "winner" "$W/logs/skew_tune_$name.log" | tail -1
}

sweep dec_skew18 "$W/results/knob_cache_dsv4_dec_skew.json" "$W/results/knob_cache_dsv4_decode.json" 256 18
sweep pre_skew18 "$W/results/knob_cache_dsv4_pre_skew.json" "$W/results/knob_cache_dsv4.json"        4096 18
sweep dec_skew14 "$W/results/knob_cache_grp_balanced.json"  "$W/results/knob_cache_dsv4_decode.json" 256 14
sweep dec_skew28 "$W/results/knob_cache_grp_skewed.json"    "$W/results/knob_cache_dsv4_decode.json" 256 28

echo "$rc" > "$W/logs/skew_tune.done"
exit $rc
