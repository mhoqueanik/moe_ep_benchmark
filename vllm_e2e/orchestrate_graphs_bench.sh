#!/bin/bash
# EAGER=0 (CUDA graphs) bench: native vs fi_nvfp4 (tuned cache), TP4+EP4.
# Quantifies how much of fi's host-overhead prefill gap graphs recover.
# Markers: logs/graphs_bench.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%Y%m%d_%H%M%S)
CACHE=$W/results/knob_cache_dsv4.json
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
for backend in native fi_nvfp4; do
  case $backend in
    native)   ENVS="FI_MOE_EP=0" ;;
    fi_nvfp4) ENVS="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl" ;;
  esac
  for wl in prefill:1024:1 decode:128:256; do
    name=${wl%%:*}
    echo "=== bench $backend / $wl (EAGER=0) ==="
    run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
      env $ENVS FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE ENFORCE_EAGER=0 \
      python bench_offline.py --tag ${backend}_g --workload $wl --rounds 5 \
        --out results/graphs_${STAMP}_${backend}_${name}.json" \
      > "$W/logs/graphs_cell_${STAMP}_${backend}_${name}.log" 2>&1 || rc=1
    grep -E "bench_offline|Error" "$W/logs/graphs_cell_${STAMP}_${backend}_${name}.log" | tail -2
  done
done

echo "=== medians (EAGER=0, TP4) ==="
python3 - "$W/results" "$STAMP" <<'PY'
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/graphs_{sys.argv[2]}_*.json")):
    d = json.load(open(f))
    r = [x["total_tok_per_s"] for x in d["rounds"] if not x["warmup"]]
    print(f"{d['tag']:>11} {d['workload']:>10}: median {d['median_total_tok_per_s']:8.1f} "
          f"min {min(r):8.1f} max {max(r):8.1f} tok/s")
PY
echo "$rc" > "$W/logs/graphs_bench.done"
exit $rc
