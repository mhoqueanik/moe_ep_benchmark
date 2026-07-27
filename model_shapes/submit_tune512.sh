#!/bin/bash
# Submit the per-shape knob-tune job (one node, 4h) — see run_tune512.sh
# for what it runs. Defaults to deepseek_v4_pro @ 512 tokens/rank.
#
#   bash model_shapes/submit_tune512.sh
#   SHAPE_NAME=deepseek_v4_flash bash model_shapes/submit_tune512.sh
set -uo pipefail

ROOT="${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}"
ACCOUNT="${ACCOUNT:-coreai_libraries_cudnn}"
PARTITION="${PARTITION:-batch}"
IMG="${IMG:-$ROOT/scratch_runbook/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh}"
BENCH=$ROOT/moe_ep_benchmark
REPO=$ROOT/flashinfer-2/flashinfer-moe_ep
MS=$BENCH/model_shapes

SHAPE_NAME="${SHAPE_NAME:-deepseek_v4_pro}"
TOKENS="${TOKENS:-512}"
DRIVER="${DRIVER:-run_tune512.sh}"
OUT_DIR="${OUT_DIR:-$MS/results_tune512}"
mkdir -p "$OUT_DIR"

tag="${DRIVER#run_tune512}"; tag="${tag%.sh}"; tag="${tag#_}"   # '' | 'schedule'
stamp="$(date +%Y%m%d_%H%M%S)_${SHAPE_NAME}_t${TOKENS}${tag:+_$tag}"
jobid=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J "coreai_libraries_cudnn-fi.tune512${tag:+.$tag}.${SHAPE_NAME}" \
    --output="$OUT_DIR/slurm_tune512${tag:+_$tag}_${SHAPE_NAME}_%j.log" \
    --wrap "srun --container-image='$IMG' \
        --container-mounts='$ROOT:$ROOT' \
        --container-workdir='$REPO' \
        bash -lc 'SHAPE_NAME=\"$SHAPE_NAME\" TOKENS=\"$TOKENS\" STAMP=\"$stamp\" OUT_DIR=\"$OUT_DIR\" DRIVER=\"$DRIVER\" bash $MS/job_payload_tune512.sh'")
echo "submitted tune512 ${SHAPE_NAME}@${TOKENS}: job ${jobid} (stamp ${stamp})"
