#!/bin/bash
# nsys profile per backend: confirm mega-kernel GPU time fi-vs-native inside
# the live engine, and attribute where the e2e budget goes (launch counts,
# staging kernels, host gaps). CUDA-trace only, child processes (vLLM
# workers) are traced automatically.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/venv0251/bin/activate"
bash "$HERE/patch_0251/apply.sh"

BACKENDS=${BACKENDS:-"native fi_dg fi_nvfp4"}
WORKLOAD=${WORKLOAD:-prefill:1024:1}
NUM_PROMPTS=${NUM_PROMPTS:-64}
ROUNDS=${ROUNDS:-2}
STAMP=$(date +%Y%m%d_%H%M%S)

backend_env() {
    case "$1" in
        native)   echo "FI_MOE_EP=0" ;;
        fi_dg)    echo "FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega" ;;
        fi_nvfp4) echo "FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl" ;;
        *) echo "unknown backend $1" >&2; return 1 ;;
    esac
}

for backend in $BACKENDS; do
    envs=$(backend_env "$backend") || exit 1
    rep="$HERE/results/nsys_${STAMP}_${backend}"
    echo "=== profiling $backend -> $rep.nsys-rep"
    env $envs nsys profile \
        --trace=cuda --sample=none --cpuctxsw=none --stats=false \
        --force-overwrite=true -o "$rep" \
        python "$HERE/bench_offline.py" \
            --tag "prof_$backend" --workload "$WORKLOAD" \
            --num-prompts "$NUM_PROMPTS" --rounds "$ROUNDS" \
            --out "$HERE/results/prof_${STAMP}_${backend}.json" \
        2>&1 | tail -5
    for report in cuda_gpu_kern_sum cuda_api_sum; do
        nsys stats --report $report --format csv --force-export=true \
            -o "$rep" "$rep.nsys-rep" > /dev/null 2>&1 || true
    done
    ls -la "$rep"* 2>/dev/null | head -5
done
echo "=== profiling done: results/nsys_${STAMP}_*"
