#!/bin/bash
# Submit one 4h SLURM job per model shape (parallel nodes; the CuTe-DSL
# compile cache under ~/.cache/flashinfer is shared and file-locked, so
# concurrent jobs are safe). Resubmit a shape to fill missing cells — compiles
# are cached, make_tables.py merges CSVs (later files win).
#
#   bash submit_jobs.sh                        # all shapes in shapes.tsv
#   SHAPE_LIST="qwen3_5_397b" bash submit_jobs.sh
#   VARIANTS="fi_combine_fp8 fi_combine_fp4" SHAPE_LIST=... bash submit_jobs.sh
set -uo pipefail

ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
IMG="${IMG:-$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh}"
BENCH=$ROOT/moe_ep_benchmark
REPO=$ROOT/flashinfer-2/flashinfer-moe_ep
MS=$BENCH/model_shapes

SHAPE_LIST="${SHAPE_LIST:-$(awk -F'\t' '!/^[[:space:]]*#/ && NF {print $1}' "$MS/shapes.tsv")}"

for shape in $SHAPE_LIST; do
    stamp="$(date +%Y%m%d_%H%M%S)_${shape}"
    jobid=$(sbatch --parsable -A coreai_libraries_cudnn -p batch -N1 \
        --ntasks-per-node=1 --time=04:00:00 \
        -J "coreai_libraries_cudnn-fi.mshape.${shape}" \
        --output="$MS/results/slurm_${shape}_%j.log" \
        --export=ALL,SHAPES="$shape",STAMP="$stamp",VARIANTS="${VARIANTS:-}",SEQ_LENS="${SEQ_LENS:-}" \
        --wrap "srun --container-image='$IMG' \
            --container-mounts='$ROOT:$ROOT' \
            --container-workdir='$REPO' \
            bash -lc 'SHAPES=\"$shape\" STAMP=\"$stamp\" ${VARIANTS:+VARIANTS=\"$VARIANTS\"} ${SEQ_LENS:+SEQ_LENS=\"$SEQ_LENS\"} bash $MS/job_payload.sh'")
    echo "submitted ${shape}: job ${jobid} (csv model_shapes_${stamp}.csv)"
done
