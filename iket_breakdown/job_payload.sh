#!/bin/bash
# In-container payload for the low-tokens/rank mega-path breakdown run
# (iket_analysis branch).  Per tokens/rank in $TOKENS_LIST, runs on 1x8 B200:
#   deep_gemm_mega  : MEGA_TIMING=e2e + kernel          (baseline)
#   nvfp4_cutedsl   : MEGA_TIMING=e2e + kernel          (comparison)
#   nvfp4_cutedsl   : MEGA_TIMING=kernel + MEGA_PHASE_TIMING=1 (breakdown;
#                     labeled "+pt", not perf-quotable)
# CSVs land in $OUT_DIR; PT_CSV lines live in the job log.
set -uo pipefail

ROOT="${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}"
REPO="${REPO:-$ROOT/flashinfer-2/flashinfer-moe_ep}"
BENCH="${BENCH:-$ROOT/moe_ep_benchmark/.claude/worktrees/iket_analysis}"
OUT_DIR="${OUT_DIR:-$BENCH/iket_breakdown/results}"
TOKENS_LIST="${TOKENS_LIST:-8 16 32 64 128 256}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
TOPK="${TOPK:-8}"
HIDDEN="${HIDDEN:-7168}"
INTER="${INTER:-2048}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-50}"

export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "$OUT_DIR"

cd "$REPO"
echo "== flashinfer: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . 2>&1 | tail -2
CU="cu$(python -c 'import torch; v=torch.version.cuda or ""; print(v.split(".")[0])')"
python -m pip install "nvidia-cutlass-dsl[$CU]==4.5.2" 2>&1 | tail -1
python -c "from importlib.metadata import version; v=version('nvidia-cutlass-dsl'); \
assert v=='4.5.2', v; print('GUARD PASS: cutlass-dsl', v)" || exit 1

cd "$BENCH"
echo "== bench: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

run_cell() {
    local backend=$1 timing=$2 pt=$3 tokens=$4
    local csv="$OUT_DIR/mega_${timing}$( [ "$pt" = 1 ] && echo _pt ).csv"
    echo "== cell: backend=$backend timing=$timing pt=$pt tokens/rank=$tokens"
    MEGA_TIMING=$timing MEGA_PHASE_TIMING=$pt MEGA_ACC="${MEGA_ACC:-0}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bench_moe_ep_mega.py \
        --world-size 8 --mega-backend "$backend" \
        --tokens-per-rank "$tokens" --num-experts "$NUM_EXPERTS" \
        --top-k "$TOPK" --hidden "$HIDDEN" --intermediate "$INTER" \
        --warmup "$WARMUP" --iters "$ITERS" --out-csv "$csv" \
        || echo "CELL_FAILED backend=$backend timing=$timing pt=$pt tokens=$tokens"
}

for tokens in $TOKENS_LIST; do
    run_cell deep_gemm_mega  e2e    0 "$tokens"
    run_cell deep_gemm_mega  kernel 0 "$tokens"
    run_cell nvfp4_cutedsl   e2e    0 "$tokens"
    run_cell nvfp4_cutedsl   kernel 0 "$tokens"
    run_cell nvfp4_cutedsl   kernel 1 "$tokens"
done

# Instrumentation-corruption gate: the accuracy-loss number must match
# between pt=0 and pt=1 at the same cell (phase timing only writes its own
# workspace region; any drift here means it does not).
echo "== acc gate (pt off vs on, tokens/rank=64)"
MEGA_ACC=1 run_cell nvfp4_cutedsl e2e 0 64
MEGA_ACC=1 run_cell nvfp4_cutedsl e2e 1 64
echo "== SWEEP DONE"
