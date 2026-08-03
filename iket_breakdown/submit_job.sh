#!/bin/bash
# Submit the low-tokens/rank mega-path breakdown job (1 node, 8 GPUs, 4h).
#   IMG=<sqsh> bash submit_job.sh
set -uo pipefail

ROOT="${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}"
ACCOUNT="${ACCOUNT:-coreai_libraries_cudnn}"
PARTITION="${PARTITION:-batch}"
IMG="${IMG:-$ROOT/scratch_runbook/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh}"
BENCH="${BENCH:-$ROOT/moe_ep_benchmark/.claude/worktrees/iket_analysis}"
OUT_DIR="${OUT_DIR:-$BENCH/iket_breakdown/results}"
mkdir -p "$OUT_DIR"

jobid=$(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION" -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J "coreai_libraries_cudnn-fi.iket_breakdown" \
    --output="$OUT_DIR/slurm_%j.log" \
    --export=ALL,OUT_DIR="$OUT_DIR",TOKENS_LIST="${TOKENS_LIST:-8 16 32 64 128 256}" \
    --wrap "srun --container-image='$IMG' \
        --container-mounts='$ROOT:$ROOT' \
        --container-workdir='$BENCH' \
        bash $BENCH/iket_breakdown/job_payload.sh")
echo "submitted iket_breakdown: job $jobid (log $OUT_DIR/slurm_${jobid}.log)"
