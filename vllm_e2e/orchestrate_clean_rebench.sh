#!/bin/bash
# Clean decode-graphs re-bench after the tail-mask memo fix:
#   A) prefill-tuned cache (regression check vs run-21 10101)
#   B) decode-tuned cache (the retune question, non-ikr winner)
# plus prefill-eager with the launch cuts (vs run-17 28204).
# Markers: logs/clean_rebench.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%Y%m%d_%H%M%S)
FIENV="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FI_MOE_EP_IKR=0"
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
for cell in \
  "pretuned_dec:$W/results/knob_cache_dsv4.json:ENFORCE_EAGER=0:decode:128:256" \
  "dectuned_dec:$W/results/knob_cache_dsv4_decode.json:ENFORCE_EAGER=0:decode:128:256" \
  "pretuned_pre:$W/results/knob_cache_dsv4.json:ENFORCE_EAGER=1:prefill:1024:1"; do
  IFS=: read -r name cache eager wname ilen olen <<< "$cell"
  echo "=== $name ($wname, $eager) ==="
  run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
    env $FIENV FLASHINFER_MOE_EP_KNOB_CACHE=$cache $eager \
    python bench_offline.py --tag $name --workload $wname:$ilen:$olen \
      --rounds 5 --out results/clean_${STAMP}_${name}.json" \
    > "$W/logs/clean_rebench_${name}.log" 2>&1 || rc=1
  grep -E "median" "$W/logs/clean_rebench_${name}.log" | tail -1
done
echo "$rc" > "$W/logs/clean_rebench.done"
exit $rc
