#!/bin/bash
# ikr payoff bench: fi_nvfp4 with in-kernel topk reduce (wrapper default now)
# + launch cuts, vs the run-21 graphs-mode numbers. Decode EAGER=0 (the
# production config) + prefill eager. Correctness smoke first (tolerance
# comparison domain — ikr is order-nondeterministic by design).
# Markers: logs/ikr_bench.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%Y%m%d_%H%M%S)
CACHE=$W/results/knob_cache_dsv4.json
FIENV="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE"
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
echo "=== smoke fi_nvfp4+ikr (correctness) ==="
run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
  env $FIENV python smoke_infer.py --tag fi_ikr --out results/smoke_fi_ikr_$STAMP.json && \
  python compare_outputs.py results/smoke_native.json results/smoke_fi_ikr_$STAMP.json" \
  > "$W/logs/ikr_smoke.log" 2>&1 || rc=1
grep -E "dlogprob|exact|SANE|mean" "$W/logs/ikr_smoke.log" | tail -4

echo "=== bench fi_nvfp4+ikr decode (EAGER=0) ==="
run "source venv0251/bin/activate && \
  env $FIENV ENFORCE_EAGER=0 python bench_offline.py --tag fi_ikr_g --workload decode:128:256 \
    --rounds 5 --out results/ikr_${STAMP}_decode_g.json" \
  > "$W/logs/ikr_decode.log" 2>&1 || rc=1
grep -E "median" "$W/logs/ikr_decode.log" | tail -1

echo "=== bench fi_nvfp4+ikr prefill (eager) ==="
run "source venv0251/bin/activate && \
  env $FIENV python bench_offline.py --tag fi_ikr_e --workload prefill:1024:1 \
    --rounds 5 --out results/ikr_${STAMP}_prefill_e.json" \
  > "$W/logs/ikr_prefill.log" 2>&1 || rc=1
grep -E "median" "$W/logs/ikr_prefill.log" | tail -1

echo "$rc" > "$W/logs/ikr_bench.done"
exit $rc
