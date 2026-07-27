#!/bin/bash
# Schedule-knob sweep for the nvfp4_cutedsl mega path at one
# (shape, tokens/rank) cell — the follow-up to run_tune512.sh.
#
# run_tune512.sh (job 2340314) swept tile / flag_batch / token-back / ikr;
# its winner (mma (256,128,256), fb4, standalone_warps, no ikr, 390.2 us)
# still holds load_balance_mode / group_hint at the _SWEEP_BASE values.
# This driver sweeps exactly those remaining axes with the offline tuner's
# `--sweep schedule` mode (flashinfer/moe_ep/tune.py): 8 candidates =
# {atomic_counter, static} x group_hint {None, 128, 256, 512}, pinned on
# the phase-B winner via --base-knobs.
#
# Two passes:
#   S1 uniform : the tuner's default near-uniform routing — the SAME routing
#                the microbenchmark cells use, so this winner is the one a
#                table cell can validate. Recorded in LOCAL_CACHE.
#   S2 skewed  : --skew 18 (the DSV4-measured per-launch max/mean load ratio
#                per tune.py's help) — the schedule axes are the
#                skew-sensitive ones, and uniform routing cannot
#                discriminate them. Production-facing information only; the
#                bench cannot validate it (its routing is near-uniform), so
#                it lands in a SEPARATE cache file and is not confirmed.
#   D  confirm : dg + the S1 winner (cache lookup) back-to-back in one
#                session with the bench's e2e_pipelined methodology — the
#                tuner's wall-clock ranking is only a ranking.
#
# Usage (inside the flashinfer-ep container, 8 GPUs):
#   bash model_shapes/run_tune512_schedule.sh
#   SKEW=25 BASE_KNOBS='{...}' bash model_shapes/run_tune512_schedule.sh
set -uo pipefail

MS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$MS_HERE/results_tune512}"
mkdir -p "$OUT_DIR"

SHAPE_NAME="${SHAPE_NAME:-deepseek_v4_pro}"
TOKENS="${TOKENS:-512}"
SKEW="${SKEW:-18}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Phase-B winner of job 2340314 (results_tune512/knob_cache_*_115240*.json).
# (A separate variable: a brace-laden JSON literal inside ${VAR:-...} would
# be cut at its first '}'.)
DEFAULT_BASE_KNOBS='{"cluster_shape_mnk": [2, 1, 1], "group_hint": 512, "epi_flag_batch": [2, 4], "load_balance_mode": "atomic_counter", "mma_tiler_mnk": [256, 128, 256], "flag_batch": 4, "token_back_mode": "standalone_warps", "in_kernel_fc2_reduce": false}'
BASE_KNOBS="${BASE_KNOBS:-$DEFAULT_BASE_KNOBS}"

row="$(awk -F'\t' -v s="$SHAPE_NAME" '$1==s {print $2, $3, $4, $5}' "$MS_HERE/shapes.tsv")"
[ -n "$row" ] || { echo "[error] shape ${SHAPE_NAME} not in shapes.tsv"; exit 1; }
read -r HIDDEN INTER NUM_EXPERTS TOPK <<<"$row"
export HIDDEN INTER NUM_EXPERTS TOPK TOKENS STAMP

# Definitions only (run.sh is guarded by a BASH_SOURCE check).
# shellcheck source=../run.sh
source "$MS_HERE/../run.sh"

export MEGA_TIMING="${MEGA_TIMING:-e2e_pipelined}"
export MEGA_IKR=0 MEGA_COMBINE_DTYPE=bf16
LOCAL_CACHE="$OUT_DIR/knob_cache_schedule_${SHAPE_NAME}_t${TOKENS}_${STAMP}.json"
SKEW_CACHE="$OUT_DIR/knob_cache_schedule_skew${SKEW}_${SHAPE_NAME}_t${TOKENS}_${STAMP}.json"
LOG="$OUT_DIR/tune_schedule_${SHAPE_NAME}_t${TOKENS}_${STAMP}.log"

# tune.py takes the model post-SwiGLU width (--intermediate == the
# *MegaMoeConfig.intermediate_size convention == shapes.tsv moe_inter).
tune_schedule () {
    torchrun --nproc_per_node="$GPUS" -m flashinfer.moe_ep.tune \
        --dtype nvfp4 \
        --hidden "$HIDDEN" --intermediate "$INTER" \
        --num-experts "$NUM_EXPERTS" --topk "$TOPK" \
        --max-tokens "$TOKENS" \
        --sweep schedule --base-knobs "$BASE_KNOBS" \
        "$@"
}

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
    echo "  SCHEDULE-KNOB SWEEP  ${SHAPE_NAME} @ ${TOKENS} tokens/rank"
    echo "################################################################"
    echo "geometry: hidden=${HIDDEN} inter=${INTER} experts=${NUM_EXPERTS} topk=${TOPK}"
    echo "base knobs: ${BASE_KNOBS}"
    echo "out_dir=${OUT_DIR}  stamp=${STAMP}  gpus=${GPUS}"

    echo ""
    echo "---- S1: schedule sweep, near-uniform routing (bench-comparable) ----"
    FLASHINFER_MOE_EP_KNOB_CACHE="$LOCAL_CACHE" tune_schedule \
        || { echo "[error] uniform schedule sweep failed"; exit 1; }

    echo ""
    echo "---- S2: schedule sweep, skew=${SKEW} (production-facing, informational) ----"
    FLASHINFER_MOE_EP_KNOB_CACHE="$SKEW_CACHE" tune_schedule --skew "$SKEW" \
        || echo "[warn] skewed schedule sweep failed (continuing)"

    echo ""
    echo "---- D: confirm S1 winner vs dg, one session, bench methodology ----"
    export FLASHINFER_MOE_EP_KNOB_CACHE="$LOCAL_CACHE" MEGA_KNOBS=""
    cell sched_dg_confirm deep_gemm_mega
    cell sched_fp4_confirm nvfp4_cutedsl

    echo ""
    echo "=== confirm cells (p50 us) ==="
    awk -F, -v s="cell_${STAMP}_" 'FNR>1 {f=FILENAME; sub(".*/" s, "", f); sub(/\.csv$/, "", f);
        printf "%-40s %-22s p50=%8s  min=%8s\n", f, $4, $15, $16}' \
        "$OUT_DIR"/cell_"${STAMP}"_*.csv 2>/dev/null
    echo ""
    echo "S1 winner cache: ${LOCAL_CACHE}"
    cat "$LOCAL_CACHE" 2>/dev/null || echo "(no cache written)"
    echo ""
    echo "S2 (skew ${SKEW}) winner cache: ${SKEW_CACHE}"
    cat "$SKEW_CACHE" 2>/dev/null || echo "(no cache written)"
} 2>&1 | tee "$LOG"
