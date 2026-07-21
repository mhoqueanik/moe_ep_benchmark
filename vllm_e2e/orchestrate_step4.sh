#!/bin/bash
# Step-4 bundle on the hold node:
#  1) fi_nvfp4 smoke with the SIMPLIFIED patch (workarounds removed) — regression
#  2) fi_nvfp4 smoke with enforce_eager=0 — CUDA graphs attempt
#  3) glue attribution: short nsys cells (fi fused / fi torch-staging / native)
# Markers: logs/step4.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%H%M%S)
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"
FIENV="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_dsv4.json"

rc=0
echo "=== 1: simplified-patch smoke (eager) ==="
run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
  env $FIENV python smoke_infer.py --tag simp_eager \
    --out results/step4_smoke_simp.json" \
  > "$W/logs/step4_smoke_simp.log" 2>&1 && echo "smoke PASS" || { echo "smoke FAIL"; rc=1; }
tail -3 "$W/logs/step4_smoke_simp.log" | head -2

echo "=== 2: EAGER=0 (CUDA graphs) smoke ==="
run "source venv0251/bin/activate && \
  env $FIENV ENFORCE_EAGER=0 python smoke_infer.py --tag fi_graphs \
    --out results/step4_smoke_graphs.json" \
  > "$W/logs/step4_smoke_graphs.log" 2>&1 && echo "graphs smoke PASS" || { echo "graphs smoke FAIL"; rc=1; }
grep -E "capture|graph|Error" "$W/logs/step4_smoke_graphs.log" | tail -4

echo "=== 3: glue attribution nsys (short cells) ==="
for cell in "fifused:$FIENV" "fitorch:$FIENV FLASHINFER_MEGA_FUSED_STAGE=0" "native:FI_MOE_EP=0"; do
  name=${cell%%:*}; envs=${cell#*:}
  rep="$W/results/nsys_glue_${STAMP}_${name}"
  echo "--- $name ---"
  run "source venv0251/bin/activate && \
    env $envs nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
      --force-overwrite=true -o $rep \
      python bench_offline.py --tag glue_$name --workload prefill:1024:1 \
        --num-prompts 16 --rounds 1 --out $W/results/glue_${STAMP}_${name}.json && \
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o $rep $rep.nsys-rep >/dev/null 2>&1" \
    > "$W/logs/step4_glue_${name}.log" 2>&1 || { echo "$name FAILED"; rc=1; }
done

echo "=== step4 exit: $rc ==="
echo "$rc" > "$W/logs/step4.done"
exit $rc
