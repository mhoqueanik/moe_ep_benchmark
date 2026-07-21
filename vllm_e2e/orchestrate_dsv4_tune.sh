#!/bin/bash
# Detached orchestrator: wait for hold node -> DSV4 offline tune -> re-bench
# (native + fi_nvfp4) with the tuned knob cache -> nsys fi_nvfp4.
# Markers: logs/dsv4_pipeline.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
CACHE=$W/results/knob_cache_dsv4.json

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

echo "=== step 1: offline tune (nvfp4, 4096/2048/256/top6, bucket 4096, EP4) ==="
JOBID=$JOBID bash "$W/in_container.sh" "source venv0251/bin/activate && \
  FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE \
  torchrun --standalone --nproc_per_node=4 -m flashinfer.moe_ep.tune \
    --dtype nvfp4 --hidden 4096 --intermediate 2048 \
    --num-experts 256 --topk 6 --max-tokens 4096"
rc=$?
echo "=== tune exit: $rc ==="
if [ "$rc" != "0" ]; then echo "$rc" > "$W/logs/dsv4_pipeline.done"; exit $rc; fi
cat "$CACHE" || true

echo "=== step 2: re-bench native + fi_nvfp4 with tuned cache ==="
JOBID=$JOBID bash "$W/in_container.sh" "source venv0251/bin/activate && \
  export FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE && \
  BACKENDS=\"native fi_nvfp4\" bash run_offline_matrix.sh && \
  BACKENDS=\"fi_nvfp4\" bash profile_matrix.sh"
rc=$?
echo "=== bench exit: $rc ==="
echo "$rc" > "$W/logs/dsv4_pipeline.done"
exit $rc
