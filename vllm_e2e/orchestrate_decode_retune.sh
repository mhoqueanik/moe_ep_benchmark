#!/bin/bash
# Decode-targeted retune: full 24-candidate sweep (ikr allowed) at
# live-tokens=256 on the engine's 4096 bucket -> separate cache file, then
# decode-graphs bench with that cache under both FI_MOE_EP_IKR settings.
# Baselines: fi 10101 (prefill-tuned cache, no ikr), native 10763.
# Markers: logs/decode_retune.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%Y%m%d_%H%M%S)
CACHE_DEC=$W/results/knob_cache_dsv4_decode.json
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
echo "=== decode-targeted sweep (24 candidates, live=256, bucket=4096) ==="
rm -f "$CACHE_DEC"
run "source venv0251/bin/activate && \
  FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE_DEC \
  torchrun --standalone --nproc_per_node=4 -m flashinfer.moe_ep.tune \
    --dtype nvfp4 --hidden 4096 --intermediate 2048 \
    --num-experts 256 --topk 6 --max-tokens 4096 --live-tokens 256 \
    --allow-nondeterministic" \
  > "$W/logs/decode_retune_sweep.log" 2>&1 || rc=1
grep -E "winner|us median" "$W/logs/decode_retune_sweep.log" | tail -3
cat "$CACHE_DEC" 2>/dev/null | head -5 || true

for ikr in 0 1; do
  echo "=== bench decode-graphs, decode-tuned cache, FI_MOE_EP_IKR=$ikr ==="
  run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
    env FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FI_MOE_EP_IKR=$ikr \
    FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE_DEC ENFORCE_EAGER=0 \
    python bench_offline.py --tag fi_dectune_ikr$ikr --workload decode:128:256 \
      --rounds 5 --out results/dectune_${STAMP}_ikr$ikr.json" \
    > "$W/logs/decode_retune_bench_ikr$ikr.log" 2>&1 || rc=1
  grep -E "median" "$W/logs/decode_retune_bench_ikr$ikr.log" | tail -1
done

echo "$rc" > "$W/logs/decode_retune.done"
exit $rc
