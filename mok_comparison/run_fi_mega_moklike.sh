#!/bin/bash
# fi moe_ep (MegaMoE mxfp8_cutedsl) forward benchmark on the MoK comparison
# workload: 2048 tok/rank, 384 experts, topk 6, H=7168, I=3072, EP=4.
set -uo pipefail

ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
REPO=$ROOT/flashinfer-2/flashinfer-moe_ep
BENCH=$ROOT/moe_ep_benchmark
cd "$REPO"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== node: $(hostname); GPUs:"; nvidia-smi -L
echo "=== branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"

PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . || exit 1
python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]" || exit 1
python -c "import flashinfer; print('flashinfer ->', flashinfer.__file__)"
python -c "import cutlass; print('cutlass-dsl', getattr(cutlass, '__version__', '?'))"

export SECTION=fi_mega GPUS=4
export TOKENS=2048 NUM_EXPERTS=384 TOPK=6 HIDDEN=7168 INTER=3072
export WARMUP=100 ITERS=100
export MEGA_LIST=mxfp8_cutedsl

echo "=== fi_mega mxfp8_cutedsl MEGA_TIMING=e2e_pipelined"
MEGA_TIMING=e2e_pipelined STAMP=moklike_pipelined bash "$BENCH/run.sh"

echo "=== fi_mega mxfp8_cutedsl MEGA_TIMING=e2e"
MEGA_TIMING=e2e STAMP=moklike_e2e bash "$BENCH/run.sh"

echo "=== newest CSVs:"
ls -t "$BENCH/results" | head -6
echo "=== DONE ==="
