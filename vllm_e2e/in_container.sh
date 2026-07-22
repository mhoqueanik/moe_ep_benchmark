#!/bin/bash
# Run a command inside the benchmark container on the held node.
#   JOBID=<slurm job id> bash in_container.sh '<shell command>'
# Reuses the named container across calls (overlay persists for the job's life).
set -euo pipefail

ROOT=${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}
IMG=${IMG:-$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh}
W=${W:-$ROOT/moe_ep_benchmark/vllm_e2e}
JOBID=${JOBID:?set JOBID to the hold job id}

exec srun --overlap --jobid="$JOBID" --ntasks=1 \
  --container-image="$IMG" \
  --container-name=fivllm \
  --container-mounts="$ROOT:$ROOT,/lustre/share:/lustre/share:ro" \
  --container-workdir="$W" \
  bash -lc "
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export HF_HOME=$ROOT/.cache/huggingface
    export PIP_CACHE_DIR=$ROOT/.cache/pip
    # Container runs as root: without this the flashinfer JIT cache lands in
    # /root/.cache (container overlay, dies with the hold job) and every new
    # job pays the full nvcc/cute.compile cost again (~30+ min for the trtllm
    # moe module alone, observed 2026-07-21).
    export FLASHINFER_WORKSPACE_BASE=$ROOT/.cache/flashinfer-root-ws
    $1
  "
