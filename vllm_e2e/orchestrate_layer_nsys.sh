#!/bin/bash
# Per-layer perf-loss localization: prefill-eager nsys for
#   fi_nvfp4 (tuned) | native dg | fi_dg (same-kernel control)
# sqlite exports for the work/wait/skew per-layer decomposition.
# Markers: logs/layer_nsys.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%H%M%S)
CACHE=$W/results/knob_cache_dsv4.json
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
for cell in \
  "fi_nvfp4:FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FI_MOE_EP_IKR=0" \
  "native:FI_MOE_EP=0" \
  "fi_dg:FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega"; do
  name=${cell%%:*}; envs=${cell#*:}
  rep="$W/results/nsys_layer_${STAMP}_${name}"
  echo "=== nsys $name (prefill eager) ==="
  run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
    env $envs FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
      --force-overwrite=true -o $rep \
      python bench_offline.py --tag lyr_$name --workload prefill:1024:1 \
        --num-prompts 64 --rounds 2 --out $W/results/lyr_${STAMP}_${name}.json && \
    nsys export --type sqlite --force-overwrite=true -o $rep.sqlite $rep.nsys-rep >/dev/null 2>&1" \
    > "$W/logs/layer_nsys_${name}.log" 2>&1 || { echo "$name FAILED"; rc=1; }
  grep -E "median" "$W/logs/layer_nsys_${name}.log" | tail -1
done
echo "$rc" > "$W/logs/layer_nsys.done"
exit $rc
