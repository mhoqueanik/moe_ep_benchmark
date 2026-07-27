#!/bin/bash
# Per-shape knob tuning for the nvfp4_cutedsl mega path at one
# (shape, tokens/rank) cell — default deepseek_v4_pro @ 512 tokens/rank.
#
# Motivation: expected_results.md §3 shows plain nvfp4 bf16 *behind*
# deep_gemm_mega at 512 tok/rank on deepseek_v4_pro (394.3 vs 376.9 us,
# 0.96x). The built-in knob heuristic (flashinfer shim/tuner.py) was derived
# on a different geometry — 256 experts, top-8, inter 2048, EP4 GB200 — and
# no knob cache existed when the shipped table was measured, so the 512 row
# ran the generic _MID_TOKEN_KNOBS profile. This driver measures whether
# per-shape tuning closes the gap, using the same cell methodology as the
# shipped table (e2e_pipelined, warmup 20 / iters 50, all cells in one job):
#
#   A. baseline : dg + fp4 on the stock heuristic (knob cache disabled) —
#                 must reproduce the table row before tuning means anything.
#   B. autotune : MEGA_KNOBS=auto, the shim's curated collective sweep
#                 (24 candidates incl. in_kernel_fc2_reduce). Ranked table
#                 lands in the log; the winner is recorded in a LOCAL knob
#                 cache under OUT_DIR (never ~/.cache — a global cache would
#                 silently change every later default-knob run) and the
#                 winner's e2e_pipelined p50 lands in the cell CSV.
#   C. extra    : explicit MEGA_KNOBS cells for mma tiles the curated sweep
#                 never tries. At 512 tok/rank x top-6 / 384 experts the
#                 mean expert load is 512*8*6/384 = 64 tokens, so the
#                 narrow-token tile (N=64) and the 1-CTA tiles (M=128) are
#                 exactly the untested corner.
#   D. confirm  : rerun dg + the overall-best fp4 cell back-to-back in one
#                 session for the clean ratio (expected_results.md §6:
#                 cross-session ratios are not trustworthy).
#
# NOTE: phase B sweeps in_kernel_fc2_reduce, so its winner can be an ikr
# candidate — that corresponds to the table's "+ikr" column, not the plain
# "nvfp4 bf16" one, and is nondeterministic in accumulation order. The
# ranked log always shows the best non-ikr candidate too; read both.
#
# Usage (inside the flashinfer-ep container, 8 GPUs):
#   bash model_shapes/run_tune512.sh
#   SHAPE_NAME=deepseek_v4_flash TOKENS=512 bash model_shapes/run_tune512.sh
set -uo pipefail

MS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$MS_HERE/results_tune512}"
mkdir -p "$OUT_DIR"

SHAPE_NAME="${SHAPE_NAME:-deepseek_v4_pro}"
TOKENS="${TOKENS:-512}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Geometry from shapes.tsv — same source of truth as the shape sweep.
row="$(awk -F'\t' -v s="$SHAPE_NAME" '$1==s {print $2, $3, $4, $5}' "$MS_HERE/shapes.tsv")"
[ -n "$row" ] || { echo "[error] shape ${SHAPE_NAME} not in shapes.tsv"; exit 1; }
read -r HIDDEN INTER NUM_EXPERTS TOPK <<<"$row"
export HIDDEN INTER NUM_EXPERTS TOPK TOKENS STAMP

# Definitions only (run.sh is guarded by a BASH_SOURCE check).
# shellcheck source=../run.sh
source "$MS_HERE/../run.sh"

export MEGA_TIMING="${MEGA_TIMING:-e2e_pipelined}"
export MEGA_IKR=0 MEGA_COMBINE_DTYPE=bf16
LOCAL_CACHE="$OUT_DIR/knob_cache_${SHAPE_NAME}_t${TOKENS}_${STAMP}.json"
LOG="$OUT_DIR/tune_${SHAPE_NAME}_t${TOKENS}_${STAMP}.log"

# One CSV per cell: the bench CSV does not record MEGA_KNOBS, so the file
# name is the cell identity; explicit-knob cells get a .knobs.json sidecar.
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
    echo "  PER-SHAPE KNOB TUNE  ${SHAPE_NAME} @ ${TOKENS} tokens/rank"
    echo "################################################################"
    echo "geometry: hidden=${HIDDEN} inter=${INTER} experts=${NUM_EXPERTS} topk=${TOPK}"
    echo "timing=${MEGA_TIMING}  gpus=${GPUS}  warmup=${WARMUP}  iters=${ITERS}"
    echo "out_dir=${OUT_DIR}  stamp=${STAMP}"

    # ---- A: stock-heuristic baseline (what the shipped table measured) ----
    export FLASHINFER_MOE_EP_KNOB_CACHE=0 MEGA_KNOBS=""
    cell a_dg deep_gemm_mega
    cell a_fp4_heuristic nvfp4_cutedsl

    # ---- B: curated collective autotune, winner into the LOCAL cache ----
    export FLASHINFER_MOE_EP_KNOB_CACHE="$LOCAL_CACHE" MEGA_KNOBS=auto
    cell b_fp4_autotune nvfp4_cutedsl

    # ---- C: tiles outside the curated candidate set ----
    export FLASHINFER_MOE_EP_KNOB_CACHE=0
    base='"cluster_shape_mnk": [2, 1, 1], "group_hint": 512, "flag_batch": 4,
          "epi_flag_batch": [2, 4], "load_balance_mode": "atomic_counter"'
    i=0
    for tile in '[256, 64, 256]' '[128, 128, 256]' '[128, 256, 256]'; do
        for tb in reuse_dispatch_warps epi_warps; do
            i=$((i + 1))
            name="c${i}_tile$(echo "$tile" | tr -d '[] ' | tr , x)_${tb%%_*}"
            export MEGA_KNOBS="{\"mma_tiler_mnk\": ${tile}, \"token_back_mode\": \"${tb}\", ${base}}"
            echo "$MEGA_KNOBS" > "$OUT_DIR/cell_${STAMP}_${name}.knobs.json"
            cell "$name" nvfp4_cutedsl
        done
    done

    # ---- D: confirm — dg + overall-best fp4 cell in ONE session ----
    best="$(awk -F, 'FNR>1 && $4 != "deep_gemm_mega" {print $15, FILENAME}' \
        "$OUT_DIR"/cell_"${STAMP}"_*.csv 2>/dev/null | sort -n | head -1)"
    if [ -z "$best" ]; then
        echo "[error] no fp4 cell produced a CSV; nothing to confirm"; exit 1
    fi
    bestfile="${best#* }"
    bestname="$(basename "$bestfile" .csv)"; bestname="${bestname#cell_${STAMP}_}"
    echo ""
    echo ">>> best fp4 cell: ${bestname} (p50 ${best%% *} us)"
    case "$bestname" in
        b_*) # autotune winner: resolve via the local cache (knobs unset)
             export FLASHINFER_MOE_EP_KNOB_CACHE="$LOCAL_CACHE" MEGA_KNOBS="" ;;
        c*)  export FLASHINFER_MOE_EP_KNOB_CACHE=0 \
                    MEGA_KNOBS="$(cat "$OUT_DIR/cell_${STAMP}_${bestname}.knobs.json")" ;;
        *)   # heuristic already best: tuning bought nothing; clean re-pair anyway
             export FLASHINFER_MOE_EP_KNOB_CACHE=0 MEGA_KNOBS="" ;;
    esac
    cell d_dg_confirm deep_gemm_mega
    cell d_fp4_confirm nvfp4_cutedsl

    echo ""
    echo "=== all cells (p50 us) ==="
    awk -F, -v s="cell_${STAMP}_" 'FNR>1 {f=FILENAME; sub(".*/" s, "", f); sub(/\.csv$/, "", f);
        printf "%-40s %-22s p50=%8s  min=%8s\n", f, $4, $15, $16}' \
        "$OUT_DIR"/cell_"${STAMP}"_*.csv 2>/dev/null
    echo ""
    echo "knob cache (autotune winner): ${LOCAL_CACHE}"
    cat "$LOCAL_CACHE" 2>/dev/null || echo "(no cache written)"
} 2>&1 | tee "$LOG"
