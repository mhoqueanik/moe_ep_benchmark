#!/bin/bash
# Within-engine repeat matrix: backends x workloads via bench_offline.py.
# Applies the current patch first (fast-path wrapper), then one engine boot
# per cell, 5 timed rounds each.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/venv0251/bin/activate"
bash "$HERE/patch_0251/apply.sh"

BACKENDS=${BACKENDS:-"native fi_dg fi_nvfp4"}
WORKLOADS=${WORKLOADS:-"prefill:1024:1 decode:128:256"}
ROUNDS=${ROUNDS:-5}
STAMP=$(date +%Y%m%d_%H%M%S)

backend_env() {
    case "$1" in
        native)   echo "FI_MOE_EP=0" ;;
        fi_dg)    echo "FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega" ;;
        fi_nvfp4) echo "FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl" ;;
        fi_mxfp8) echo "FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=mxfp8_cutedsl" ;;
        *) echo "unknown backend $1" >&2; return 1 ;;
    esac
}

for backend in $BACKENDS; do
    envs=$(backend_env "$backend") || exit 1
    for wl in $WORKLOADS; do
        name=${wl%%:*}
        out="$HERE/results/offline_${STAMP}_${backend}_${name}.json"
        log="$HERE/logs/offline_${STAMP}_${backend}_${name}.log"
        echo "=== $backend / $wl -> $out"
        env $envs python "$HERE/bench_offline.py" \
            --tag "$backend" --workload "$wl" --rounds "$ROUNDS" \
            --out "$out" 2>&1 | tee "$log" | grep -E "bench_offline"
    done
done

echo "=== medians ==="
python - "$HERE/results" "$STAMP" <<'PY'
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/offline_{sys.argv[2]}_*.json")):
    d = json.load(open(f))
    r = [x["total_tok_per_s"] for x in d["rounds"] if not x["warmup"]]
    print(f"{d['tag']:>9} {d['workload']:>8}: median {d['median_total_tok_per_s']:8.1f} "
          f"min {min(r):8.1f} max {max(r):8.1f} tok/s")
PY
