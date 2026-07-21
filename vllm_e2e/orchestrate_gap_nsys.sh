#!/bin/bash
# Steady-state gap attribution: nsys {native, fi_nvfp4} x {decode graphs,
# prefill eager}, node-level graph tracing, sqlite exports for time-windowed
# analysis (the timed rounds live at the tail of each trace).
# Markers: logs/gap_nsys.{log,done}
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
  "native_dec_g:FI_MOE_EP=0:ENFORCE_EAGER=0:decode:128:256" \
  "fi_dec_g:FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl:ENFORCE_EAGER=0:decode:128:256" \
  "native_pre_e:FI_MOE_EP=0:ENFORCE_EAGER=1:prefill:1024:1" \
  "fi_pre_e:FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl:ENFORCE_EAGER=1:prefill:1024:1"; do
  IFS=: read -r name fienv eager wname ilen olen <<< "$cell"
  rep="$W/results/nsys_gap_${STAMP}_${name}"
  echo "=== $name ==="
  run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
    env $fienv $eager FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
      --cuda-graph-trace=node --force-overwrite=true -o $rep \
      python bench_offline.py --tag $name --workload $wname:$ilen:$olen \
        --num-prompts 64 --rounds 2 --out $W/results/gap_${STAMP}_${name}.json && \
    nsys export --type sqlite --force-overwrite=true -o $rep.sqlite $rep.nsys-rep >/dev/null 2>&1" \
    > "$W/logs/gap_nsys_${name}.log" 2>&1 || { echo "$name FAILED"; rc=1; }
  grep -E "bench_offline.*median" "$W/logs/gap_nsys_${name}.log" | tail -1
done
echo "$rc" > "$W/logs/gap_nsys.done"
exit $rc
