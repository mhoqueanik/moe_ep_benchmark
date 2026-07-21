#!/bin/bash
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%H%M%S)
FIENV="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FI_MOE_EP_IKR=0 ENFORCE_EAGER=0"
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }
until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
rc=0
for cell in "pretuned:$W/results/knob_cache_dsv4.json" "dectuned:$W/results/knob_cache_dsv4_decode.json"; do
  name=${cell%%:*}; cache=${cell#*:}
  echo "=== $name decode-graphs ==="
  run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
    env $FIENV FLASHINFER_MOE_EP_KNOB_CACHE=$cache \
    python bench_offline.py --tag ${name}_final --workload decode:128:256 \
      --rounds 5 --out results/final_${STAMP}_${name}.json" \
    > "$W/logs/decode_final_${name}.log" 2>&1 || rc=1
  grep -E "median" "$W/logs/decode_final_${name}.log" | tail -1
done
echo "$rc" > "$W/logs/decode_final.done"
exit $rc
