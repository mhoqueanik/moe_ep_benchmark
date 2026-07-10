#!/bin/bash
# Sweep tokens/rank (per-rank seq_len) from 1 to 8k and run the MoE-EP suite
# at each point.  Each subsection (vllm_split, vllm_mega, fi_mega) appends to
# its own suffixed sweep CSV/log.
#
# Default sweep points are powers of two: 1 2 4 8 ... 8192.
# Override with a space-separated list:
#   SEQ_LENS="1 8 64 512 4096" ./moe_ep_benchmark/run_sweep.sh
#
# Run one subsection only:
#   SECTION=fi_mega ./moe_ep_benchmark/run_sweep.sh
#
# Other knobs match run.sh (ALGO, EXPERTS_LIST, MEGA_LIST, GPUS, ...).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse run.sh env defaults and benchmark helpers without executing a single run.
# shellcheck source=run.sh
source "$HERE/run.sh"

SEQ_LEN_MIN="${SEQ_LEN_MIN:-1}"
SEQ_LEN_MAX="${SEQ_LEN_MAX:-8192}"

if [ -n "${SEQ_LENS:-}" ]; then
    # shellcheck disable=SC2206
    SWEEP_LENS=($SEQ_LENS)
else
    SWEEP_LENS=()
    n="$SEQ_LEN_MIN"
    if [ "$n" -lt 1 ]; then
        n=1
    fi
    while [ "$n" -le "$SEQ_LEN_MAX" ]; do
        SWEEP_LENS+=("$n")
        n=$((n * 2))
    done
    if [ "${SWEEP_LENS[-1]}" -lt "$SEQ_LEN_MAX" ]; then
        SWEEP_LENS+=("$SEQ_LEN_MAX")
    fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="sweep"

run_sweep_section () {
    local sec="$1"
    local log="${OUT_DIR}/${RUN_PREFIX}_${STAMP}_${sec}.log"
    CSV="${OUT_DIR}/${RUN_PREFIX}_${STAMP}_${sec}.csv"

    {
        echo "################################################################"
        echo "  SWEEP SECTION ${sec}"
        echo "################################################################"
        echo "sweep log: ${log}"
        echo "sweep csv: ${CSV}"
        echo "sweep points (tokens/rank): ${SWEEP_LENS[*]}"
        echo "algo=${ALGO}  gpus=${GPUS}  exclude_quant=${EXCLUDE_QUANT}"
        echo ""

        for TOKENS in "${SWEEP_LENS[@]}"; do
            echo ""
            echo "============================================================"
            echo "  SWEEP section=${sec}  tokens/rank=${TOKENS}"
            echo "============================================================"

            case "$sec" in
                vllm_split)
                    run_vllm_split_bench \
                        || echo "[warn] sweep section=${sec} tokens/rank=${TOKENS} had failures (continuing)"
                    ;;
                vllm_mega)
                    run_vllm_mega_bench \
                        || echo "[warn] sweep section=${sec} tokens/rank=${TOKENS} had failures (continuing)"
                    ;;
                fi_mega)
                    run_fi_mega_bench \
                        || echo "[warn] sweep section=${sec} tokens/rank=${TOKENS} had failures (continuing)"
                    ;;
            esac
        done

        echo ""
        echo "=== sweep summary (${CSV}) ==="
        cat "$CSV" 2>/dev/null || true
    } 2>&1 | tee "$log"
}

for sec in $(sections_to_run); do
    run_sweep_section "$sec" \
        || echo "[warn] sweep section ${sec} had failures (continuing)"
done
