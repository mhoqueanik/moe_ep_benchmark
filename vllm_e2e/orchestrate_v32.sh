#!/bin/bash
# DeepSeek V3.2 e2e: fi_nvfp4 smoke + parity vs native, then the graphs bench
# matrix (native vs fi_nvfp4, prefill + decode, TP4+EP4).
#
#   bash orchestrate_v32.sh <hold jobid>       # usually via launch_detached.sh
#
# Prereqs: patch_v32/apply.sh applied (fi_experts.py + patched v32 model.py),
# native smoke logs/smoke_v32_native.done == 0 (this script waits for it).
# Both backends use the SAME nvfp4 checkpoint (fp8 orig doesn't fit a node).
# Markers: logs/v32_bench.{log,done}; results: results/v32_${STAMP}_*.json
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%Y%m%d_%H%M%S)
MODEL_V32=/lustre/share/coreai_dlalgo_ci/artifacts/model/nvidia_deepseek-v3.2-nvfp4/hf/hf-7c0f62c_orig
# TOKENIZER_MODE=auto resolves to deepseek_v32; MOE_BACKEND=auto keeps the
# stock monolithic modelopt path (the fi swap is FI_MOE_EP env-gated inside
# the patched model, NOT via the moe_backend string as on V4).
COMMON="TOKENIZER_MODE=auto MOE_BACKEND=auto MODEL=$MODEL_V32"
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

# 1. Wait for the already-launched native smoke (JIT compile bound).
while [ ! -f "$W/logs/smoke_v32_native.done" ]; do sleep 60; done
if [ "$(cat "$W/logs/smoke_v32_native.done")" != "0" ]; then
  echo "native smoke FAILED — aborting (see logs/smoke_v32_native.log)"
  echo 1 > "$W/logs/v32_bench.done"; exit 1
fi
echo "=== native smoke ok ==="

# 2. fi_nvfp4 smoke (first run pays the cutedsl mega compile, ~10-15 min).
echo "=== fi_nvfp4 smoke ==="
run "source venv0251/bin/activate && bash patch_v32/apply.sh >/dev/null && \
  env $COMMON FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl \
  python smoke_infer.py --tag v32_fi_nvfp4 --out results/smoke_v32_fi_nvfp4.json" \
  > "$W/logs/smoke_v32_fi_nvfp4.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "fi smoke FAILED rc=$rc (see logs/smoke_v32_fi_nvfp4.log)"
  echo 1 > "$W/logs/v32_bench.done"; exit 1
fi

# 3. Parity: token exact-match + logprob divergence vs native.
echo "=== parity native vs fi_nvfp4 ==="
run "source venv0251/bin/activate && \
  python compare_outputs.py results/smoke_v32_native.json results/smoke_v32_fi_nvfp4.json" \
  | tee "$W/logs/v32_parity.log"

# 4. Graphs bench matrix (run-29-proven config: sparse capture incl. 4096
#    prefill steps; decode capture covered by the default sizes in the list).
rc=0
for backend in native fi_nvfp4; do
  case $backend in
    native)   ENVS="FI_MOE_EP=0" ;;
    fi_nvfp4) ENVS="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl" ;;
  esac
  for wl in prefill:1024:1 decode:128:256; do
    name=${wl%%:*}
    echo "=== bench $backend / $wl (graphs, MAX_CAPTURE=4096) ==="
    run "source venv0251/bin/activate && bash patch_v32/apply.sh >/dev/null && \
      env $COMMON $ENVS ENFORCE_EAGER=0 MAX_CAPTURE=4096 CAPTURE_SIZES=8,16,32,64,128,256,512,1024,2048,4096 \
      python bench_offline.py --tag v32_${backend} --workload $wl --rounds 5 \
        --out results/v32_${STAMP}_${backend}_${name}.json" \
      > "$W/logs/v32_cell_${STAMP}_${backend}_${name}.log" 2>&1 || rc=1
    grep -E "bench_offline|median|Error" "$W/logs/v32_cell_${STAMP}_${backend}_${name}.log" | tail -2
  done
done

echo "=== medians (graphs, TP4+EP4, nvfp4 ckpt) ==="
python3 - "$W/results" "$STAMP" <<'PY'
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/v32_{sys.argv[2]}_*.json")):
    d = json.load(open(f))
    r = [x["total_tok_per_s"] for x in d["rounds"] if not x["warmup"]]
    print(f"{d['tag']:>12} {d['workload']:>14}: median {d['median_total_tok_per_s']:9.1f} "
          f"min {min(r):9.1f} max {max(r):9.1f} tok/s")
PY
echo "$rc" > "$W/logs/v32_bench.done"
exit $rc
