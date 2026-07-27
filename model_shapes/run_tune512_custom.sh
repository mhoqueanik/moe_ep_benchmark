#!/bin/bash
# Custom-space knob sweep (kernel-dev grid + curated set) for nvfp4_cutedsl
# at one (shape, tokens/rank) cell — third round after run_tune512.sh (tile /
# token-back / ikr) and run_tune512_schedule.sh (load_balance x group_hint).
#
# The space is the kernel dev's tester-style --use_knob grid, crossed and
# is_valid-pruned by tune_custom.py, UNIONED with the shim's curated 24
# (which it happens to contain): 2 tiles x gh {512,256} x fb {1,2,4,8} x
# efb {(1,4),(2,4)} x 3 token-backs x ikr {off,on} = 192 candidates. New
# axes vs earlier rounds: fb {1,2}, efb (1,4), gh 256 in the main sweep.
#
#   S : collective sweep over the union space (ranked table in the log,
#       winner into a LOCAL cache file)
#   D : dg + winner re-paired in one session with the bench's e2e_pipelined
#       methodology (the tuner's wall-clock numbers are only a ranking)
#
# Usage (inside the flashinfer-ep container, 8 GPUs):
#   bash model_shapes/run_tune512_custom.sh
#   USE_KNOBS="mma_tiler_mnk=256,64,256 flag_batch=16 ..." \
#       bash model_shapes/run_tune512_custom.sh     # swap in a bigger grid
set -uo pipefail

MS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$MS_HERE/results_tune512}"
mkdir -p "$OUT_DIR"

SHAPE_NAME="${SHAPE_NAME:-deepseek_v4_pro}"
TOKENS="${TOKENS:-512}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Kernel-dev suggested grid (2026-07-27), tester --use_knob syntax with '='.
DEFAULT_USE_KNOBS="
mma_tiler_mnk=256,256,256
mma_tiler_mnk=256,128,256
cluster_shape_mnk=2,1,1
group_hint=512
group_hint=256
flag_batch=8
flag_batch=1
flag_batch=2
flag_batch=4
epi_flag_batch=1,4
epi_flag_batch=2,4
token_back_mode=epi_warps
token_back_mode=reuse_dispatch_warps
token_back_mode=standalone_warps
load_balance_mode=atomic_counter
in_kernel_fc2_reduce=false
in_kernel_fc2_reduce=true
"
USE_KNOBS="${USE_KNOBS:-$DEFAULT_USE_KNOBS}"

row="$(awk -F'\t' -v s="$SHAPE_NAME" '$1==s {print $2, $3, $4, $5}' "$MS_HERE/shapes.tsv")"
[ -n "$row" ] || { echo "[error] shape ${SHAPE_NAME} not in shapes.tsv"; exit 1; }
read -r HIDDEN INTER NUM_EXPERTS TOPK <<<"$row"
export HIDDEN INTER NUM_EXPERTS TOPK TOKENS STAMP

# Definitions only (run.sh is guarded by a BASH_SOURCE check).
# shellcheck source=../run.sh
source "$MS_HERE/../run.sh"

export MEGA_TIMING="${MEGA_TIMING:-e2e_pipelined}"
export MEGA_IKR=0 MEGA_COMBINE_DTYPE=bf16
LOCAL_CACHE="$OUT_DIR/knob_cache_custom_${SHAPE_NAME}_t${TOKENS}_${STAMP}.json"
LOG="$OUT_DIR/tune_custom_${SHAPE_NAME}_t${TOKENS}_${STAMP}.log"

knob_args=()
for tok in $USE_KNOBS; do
    knob_args+=(--use-knob "$tok")
done

cell () {
    local name="$1" backend="$2"
    CSV="$OUT_DIR/cell_${STAMP}_${name}.csv"
    echo ""
    echo ">>> CELL ${name}: backend=${backend}"
    echo "    MEGA_KNOBS=${MEGA_KNOBS:-<default>}"
    echo "    knob_cache=${FLASHINFER_MOE_EP_KNOB_CACHE:-<default>}"
    run_mega "$backend" || echo "[warn] cell ${name} failed (continuing)"
}

{
    echo "################################################################"
    echo "  CUSTOM-SPACE KNOB SWEEP  ${SHAPE_NAME} @ ${TOKENS} tokens/rank"
    echo "################################################################"
    echo "geometry: hidden=${HIDDEN} inter=${INTER} experts=${NUM_EXPERTS} topk=${TOPK}"
    echo "grid tokens:" $USE_KNOBS
    echo "out_dir=${OUT_DIR}  stamp=${STAMP}  gpus=${GPUS}"

    echo ""
    echo "---- S: collective sweep over grid + curated union ----"
    FLASHINFER_MOE_EP_KNOB_CACHE="$LOCAL_CACHE" \
        torchrun --nproc_per_node="$GPUS" "$MS_HERE/tune_custom.py" \
        --hidden "$HIDDEN" --intermediate "$INTER" \
        --num-experts "$NUM_EXPERTS" --topk "$TOPK" \
        --max-tokens "$TOKENS" \
        "${knob_args[@]}" \
        || { echo "[error] custom sweep failed"; exit 1; }

    echo ""
    echo "---- D: confirm winner vs dg, one session, bench methodology ----"
    export FLASHINFER_MOE_EP_KNOB_CACHE="$LOCAL_CACHE" MEGA_KNOBS=""
    cell custom_dg_confirm deep_gemm_mega
    cell custom_fp4_confirm nvfp4_cutedsl

    echo ""
    echo "=== confirm cells (p50 us) ==="
    awk -F, -v s="cell_${STAMP}_" 'FNR>1 {f=FILENAME; sub(".*/" s, "", f); sub(/\.csv$/, "", f);
        printf "%-40s %-22s p50=%8s  min=%8s\n", f, $4, $15, $16}' \
        "$OUT_DIR"/cell_"${STAMP}"_*.csv 2>/dev/null
    echo ""
    echo "winner cache: ${LOCAL_CACHE}"
    cat "$LOCAL_CACHE" 2>/dev/null || echo "(no cache written)"
} 2>&1 | tee "$LOG"
