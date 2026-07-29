#!/bin/bash
#SBATCH --job-name=deepseekv4flash.vllm_serving_sweep_ep8
#SBATCH --account=coreai_libraries_cudnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --output=vllm_serving_sweep_ep8_%j.log
#
# DeepSeek-V4-Flash serving-mode sweep at EP8/TP8 on one 8-GPU SM100 node —
# the server+client counterpart of job_vllm_pr_runbook_sweep_ep8.sh. One
# process launches `vllm serve --moe-backend <be>`, a second drives it with
# `vllm bench serve` (random dataset, fixed lengths). The cells are the
# offline sweep's four workloads, verbatim, so the rows correspond 1:1:
#
#   pre8k   1024 in /    1 out, 256 requests                  [headline]
#   dec1k    128 in /  256 out, 1024 requests                 [headline]
#   lc100k 100000 in / 1024 out, 32 requests @ conc 32        [interactivity]
#   ctx32k  32768 in /   32 out, 32 requests @ conc 32        [interactivity]
#
# One server boot per (cell, backend) — 12 boots — because the per-cell
# engine settings are part of the cell definition. Same invariants as the
# offline sweep: TP8/EP8/DP1, sparse capture ladder, prefix caching off,
# fi_cutedsl on the NVFP4 checkpoint with the EP8 knob cache, [fi_moe_ep]
# banner check per backend, round 0 discarded as warmup, median of the timed
# rounds. The mechanics live in serving_payload.sh; this script is the
# allocation and the environment.
#
# Usage:
#   cd <repo>/vllm_e2e && sbatch job_vllm_serving_sweep_ep8.sh
#
# Overrides (all optional; sbatch exports the submitting env by default):
#   ROOT=/my/scratch          checkout root; default below
#   IMG=/path/to.sqsh         container image
#   MODEL_MX_FLASH=... MODEL_NVFP4_FLASH=...   required; see RUNBOOK §1.3
#   CELLS='pre8k dec1k lc100k ctx32k' ROUNDS=3 ROUNDS_LC100K=2 PORT=30000
#   EXTRA_MOUNTS=a:a,b:b      appended to --container-mounts
#   sbatch -A <acct> -p <part> ...   CLI flags override the #SBATCH lines above
set -uo pipefail

# NB: do not "improve" this into a BASH_SOURCE-derived path -- sbatch copies the
# script to a spool dir, so the script's own location is not the checkout.
ROOT=${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}
W=$ROOT/moe_ep_benchmark/vllm_e2e
IMG=${IMG:-$ROOT/flashinfer-ep.sqsh}
MOUNTS="$ROOT:$ROOT,/lustre/share:/lustre/share:ro"
[[ -n "${EXTRA_MOUNTS:-}" ]] && MOUNTS="$MOUNTS,$EXTRA_MOUNTS"

# Required, not optional: the payload passes the checkpoint per backend. Fail
# here rather than an hour into the allocation.
FWD=""
for v in MODEL_MX_FLASH MODEL_NVFP4_FLASH; do
    [[ -n "${!v:-}" ]] || { echo "$v is unset -- see RUNBOOK_REPRO.md §1.3"; exit 2; }
    [[ -d "${!v}" ]]   || { echo "$v=${!v} is not a directory"; exit 2; }
    FWD+="export $v='${!v}'; "
done
# Workload overrides ride through only when set, so the payload defaults stay
# the single source of truth.
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

export SERVE_MX=\$MODEL_MX_FLASH
export SERVE_NVFP4=\$MODEL_NVFP4_FLASH
export SERVE_PREFIX=ep8
export SERVE_KNOB_CACHE=$W/results/knob_cache_ep8.json
bash serving_payload.sh
"
rc=$?
echo "=== job exit rc=$rc ==="
exit $rc
