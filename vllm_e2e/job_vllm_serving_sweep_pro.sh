#!/bin/bash
#SBATCH --job-name=deepseekv4pro.vllm_serving_sweep
#SBATCH --account=coreai_libraries_cudnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --output=vllm_serving_sweep_pro_%j.log
#
# DeepSeek-V4-Pro serving-mode sweep at EP8/TP8 — identical cells to
# job_vllm_serving_sweep_ep8.sh (see its header for the four offline-mirror
# workloads and the invariants), on the Pro checkpoint pair and the Pro knob
# cache. Needs the optional 1.66 TB Pro checkpoints from RUNBOOK §1.3.
# 4 h wall (partition max): 12 server boots at Pro weight-load times
# dominate, ~3.7 h total -- if the tail is cut, rerun the missing cells via
# CELLS=... (each cell is self-contained).
#
# Usage:
#   cd <repo>/vllm_e2e && sbatch job_vllm_serving_sweep_pro.sh
#
# Overrides: as the Flash script, with MODEL_MX_PRO / MODEL_NVFP4_PRO required.
set -uo pipefail

# NB: do not "improve" this into a BASH_SOURCE-derived path -- sbatch copies the
# script to a spool dir, so the script's own location is not the checkout.
ROOT=${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}
W=$ROOT/moe_ep_benchmark/vllm_e2e
IMG=${IMG:-$ROOT/flashinfer-ep.sqsh}
MOUNTS="$ROOT:$ROOT,/lustre/share:/lustre/share:ro"
[[ -n "${EXTRA_MOUNTS:-}" ]] && MOUNTS="$MOUNTS,$EXTRA_MOUNTS"

FWD=""
for v in MODEL_MX_PRO MODEL_NVFP4_PRO; do
    [[ -n "${!v:-}" ]] || { echo "$v is unset -- see RUNBOOK_REPRO.md §1.3"; exit 2; }
    [[ -d "${!v}" ]]   || { echo "$v=${!v} is not a directory"; exit 2; }
    FWD+="export $v='${!v}'; "
done
for v in CELLS ROUNDS ROUNDS_LC100K PORT HEALTH_TIMEOUT_S; do
    [[ -n "${!v:-}" ]] && FWD+="export $v='${!v}'; "
done

srun --ntasks=1 \
  --container-image="$IMG" \
  --container-name=fivllm_serving \
  --container-mounts="$MOUNTS" \
  --container-workdir="$W" \
  bash -lc "
set -uo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=$ROOT/.cache/huggingface
export PIP_CACHE_DIR=$ROOT/.cache/pip
export FLASHINFER_WORKSPACE_BASE=$ROOT/.cache/flashinfer-root-ws
$FWD

echo '=== node ==='; hostname; nvidia-smi -L | head -4
echo '=== harness ==='; git -C $ROOT/moe_ep_benchmark log --oneline -1
echo '=== flashinfer ==='; git -C $ROOT/flashinfer-2/flashinfer-moe_ep log --oneline -1

export SERVE_MX=\$MODEL_MX_PRO
export SERVE_NVFP4=\$MODEL_NVFP4_PRO
export SERVE_PREFIX=pro
export SERVE_KNOB_CACHE=$W/results/knob_cache_pro_ep8.json
bash serving_payload.sh
"
rc=$?
echo "=== job exit rc=$rc ==="
exit $rc
