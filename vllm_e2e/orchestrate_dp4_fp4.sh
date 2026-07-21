#!/bin/bash
# DP4/TP1+EP4 matrix via bench_offline_dp.py (multi-process offline DP):
# native vs fi_nvfp4 (tuned knob cache), no TP allreduce.
# bench (prefill + decode, 5 rounds) then nsys on both backends at prefill.
# Markers: logs/dp4_pipeline.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
CACHE=$W/results/knob_cache_dsv4.json
STAMP=${DP4_STAMP:-$(date +%Y%m%d_%H%M%S)}

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
    if [ -s "$W/results/dp4_${STAMP}_${backend}_${name}.json" ]; then
      echo "=== skip $backend / $wl (already done) ==="
      continue
    fi
    echo "=== bench $backend / $wl (DP4/TP1) ==="
    run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
      env $ENVS FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE \
      python bench_offline_dp.py --tag ${backend}_dp4 --workload $wl --rounds 5 --dp 4 \
        --out results/dp4_${STAMP}_${backend}_${name}.json" \
      > "$W/logs/dp4_cell_${STAMP}_${backend}_${name}.log" 2>&1 || rc=1
    grep -E "bench_offline_dp|died during startup|AssertionError|RuntimeError" \
      "$W/logs/dp4_cell_${STAMP}_${backend}_${name}.log" | tail -4
  done
done

echo "=== medians (DP4/TP1) ==="
python3 - "$W/results" "$STAMP" <<'PY'
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/dp4_{sys.argv[2]}_*.json")):
    d = json.load(open(f))
    r = [x["total_tok_per_s"] for x in d["rounds"] if not x["warmup"]]
    print(f"{d['tag']:>14} {d['workload']:>10}: median {d['median_total_tok_per_s']:8.1f} "
          f"min {min(r):8.1f} max {max(r):8.1f} tok/s")
PY

for backend in native fi_nvfp4; do
  case $backend in
    native)   ENVS="FI_MOE_EP=0" ;;
    fi_nvfp4) ENVS="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl" ;;
  esac
  echo "=== nsys $backend (DP4/TP1 prefill) ==="
  rep="$W/results/nsys_dp4_${STAMP}_${backend}"
  run "source venv0251/bin/activate && \
    env $ENVS FLASHINFER_MOE_EP_KNOB_CACHE=$CACHE \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
      --force-overwrite=true -o $rep \
      python bench_offline_dp.py --tag prof_${backend}_dp4 --workload prefill:1024:1 \
        --num-prompts 64 --rounds 2 --dp 4 \
        --out $W/results/prof_dp4_${STAMP}_${backend}.json && \
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o $rep $rep.nsys-rep >/dev/null 2>&1; \
    nsys stats --report cuda_api_sum --format csv --force-export=true -o $rep $rep.nsys-rep >/dev/null 2>&1" \
    2>&1 | tail -3 || rc=1
done

echo "=== dp4 pipeline exit: $rc ==="
echo "$rc" > "$W/logs/dp4_pipeline.done"
exit $rc
