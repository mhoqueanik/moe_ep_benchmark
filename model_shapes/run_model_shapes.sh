#!/bin/bash
# Model-shape microbenchmark sweep for the fi mega MoE-EP path.
#
# Shapes are the MoE geometries of real models (shapes.tsv), analogous to the
# model list of the cudnn-frontend SDPA training benchmark. For every shape we
# run five variants at each tokens/rank point:
#
#   fi_dg          : deep_gemm_mega
#   fi_fp4         : nvfp4_cutedsl
#   fi_ikr         : nvfp4_cutedsl + MEGA_IKR=1 (in-kernel fc2 reduce)
#   fi_combine_fp8 : nvfp4_cutedsl + MEGA_COMBINE_DTYPE=mxfp8
#   fi_combine_fp4 : nvfp4_cutedsl + MEGA_COMBINE_DTYPE=nvfp4
#
# All rows land in ONE csv (geometry columns + compute_kernel suffix identify
# each cell); model_shapes/make_tables.py turns it into RESULTS.md.
#
# Usage (inside the flashinfer-ep container on a 4-GPU node):
#   bash model_shapes/run_model_shapes.sh
#   SHAPES="deepseek_v3 gpt_oss_120b" VARIANTS="fi_dg fi_fp4" \
#       SEQ_LENS="8 2048" bash model_shapes/run_model_shapes.sh
#
# Knobs inherited from run.sh: GPUS, WARMUP, ITERS, MEGA_KNOBS, MEGA_TIMING
# (default here: e2e_pipelined, the methodology of the 2026-07-15 tables).
set -uo pipefail

MS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$MS_HERE/results}"

# Definitions only (run.sh is guarded by a BASH_SOURCE check).
# shellcheck source=../run.sh
source "$MS_HERE/../run.sh"

SEQ_LENS="${SEQ_LENS:-8 64 512 2048 8192}"
VARIANTS="${VARIANTS:-fi_dg fi_fp4 fi_ikr fi_combine_fp8 fi_combine_fp4}"
SHAPES="${SHAPES:-}"   # names from shapes.tsv; empty = all
export MEGA_TIMING="${MEGA_TIMING:-e2e_pipelined}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CSV="${OUT_DIR}/model_shapes_${STAMP}.csv"
LOG="${OUT_DIR}/model_shapes_${STAMP}.log"

shape_selected () {
    local name="$1"
    [ -z "$SHAPES" ] && return 0
    local s
    for s in $SHAPES; do
        [ "$s" = "$name" ] && return 0
    done
    return 1
}

run_variant () {
    local variant="$1" backend
    case "$variant" in
        fi_dg)          backend=deep_gemm_mega; export MEGA_IKR=0 MEGA_COMBINE_DTYPE=bf16 ;;
        fi_fp4)         backend=nvfp4_cutedsl;  export MEGA_IKR=0 MEGA_COMBINE_DTYPE=bf16 ;;
        fi_ikr)         backend=nvfp4_cutedsl;  export MEGA_IKR=1 MEGA_COMBINE_DTYPE=bf16 ;;
        fi_combine_fp8) backend=nvfp4_cutedsl;  export MEGA_IKR=0 MEGA_COMBINE_DTYPE=mxfp8 ;;
        fi_combine_fp4) backend=nvfp4_cutedsl;  export MEGA_IKR=0 MEGA_COMBINE_DTYPE=nvfp4 ;;
        *) echo "[error] unknown variant: $variant"; return 1 ;;
    esac
    run_mega "$backend" \
        || echo "[warn] shape=${SHAPE_NAME} variant=${variant} tokens/rank=${TOKENS} failed (continuing)"
}

{
    echo "################################################################"
    echo "  MODEL-SHAPE SWEEP  (fi mega variants)"
    echo "################################################################"
    echo "csv: ${CSV}"
    echo "variants: ${VARIANTS}"
    echo "tokens/rank points: ${SEQ_LENS}"
    echo "timing=${MEGA_TIMING}  gpus=${GPUS}  warmup=${WARMUP}  iters=${ITERS}"
    echo ""

    while IFS=$'\t' read -r SHAPE_NAME HIDDEN INTER NUM_EXPERTS TOPK _comment; do
        case "$SHAPE_NAME" in ''|'#'*) continue ;; esac
        shape_selected "$SHAPE_NAME" || continue

        echo ""
        echo "================================================================"
        echo "  SHAPE ${SHAPE_NAME}: hidden=${HIDDEN} inter=${INTER}" \
             "experts=${NUM_EXPERTS} topk=${TOPK}"
        echo "================================================================"

        for variant in $VARIANTS; do
            for TOKENS in $SEQ_LENS; do
                echo ""
                echo ">>> shape=${SHAPE_NAME} variant=${variant} tokens/rank=${TOKENS}"
                run_variant "$variant"
            done
        done
    done < "$MS_HERE/shapes.tsv"

    echo ""
    echo "=== raw csv (${CSV}) ==="
    cat "$CSV" 2>/dev/null || true
} 2>&1 | tee "$LOG"
