#!/bin/bash
# Offline-tune the fi mxfp8_cutedsl mega knobs at the MoK comparison geometry,
# then rerun the forward benchmark with the tuned knob cache.
set -uo pipefail

ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
REPO=$ROOT/flashinfer-2/flashinfer-moe_ep
BENCH=$ROOT/moe_ep_benchmark
cd "$REPO"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASHINFER_MOE_EP_KNOB_CACHE=$ROOT/logs_mok/moe_ep_knob_cache_moklike.json

echo "=== node: $(hostname); GPUs:"; nvidia-smi -L
echo "=== branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"

PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . || exit 1
python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]" || exit 1
python -c "import flashinfer; print('flashinfer ->', flashinfer.__file__)"
python -c "import cutlass; print('cutlass-dsl', getattr(cutlass, '__version__', '?'))"

echo "=== tune: mxfp8_e4m3 H=7168 I=3072 E=384 topk=6 max-tokens=2048 EP=4"
time torchrun --standalone --nproc_per_node=4 -m flashinfer.moe_ep.tune \
    --dtype mxfp8_e4m3 --hidden 7168 --intermediate 3072 \
    --num-experts 384 --topk 6 --max-tokens 2048 || exit 1

echo "=== knob cache after tune:"
cat "$FLASHINFER_MOE_EP_KNOB_CACHE"

export SECTION=fi_mega GPUS=4
export TOKENS=2048 NUM_EXPERTS=384 TOPK=6 HIDDEN=7168 INTER=3072
export WARMUP=100 ITERS=100
export MEGA_LIST=mxfp8_cutedsl

echo "=== fi_mega mxfp8_cutedsl TUNED +SHARED EXPERT MEGA_TIMING=e2e_pipelined"
MEGA_SHARED_EXPERT=1 MEGA_TIMING=e2e_pipelined STAMP=moklike_tuned_pipelined bash "$BENCH/run.sh"

echo "=== fi_mega mxfp8_cutedsl TUNED +SHARED EXPERT MEGA_TIMING=e2e"
MEGA_SHARED_EXPERT=1 MEGA_TIMING=e2e STAMP=moklike_tuned_e2e bash "$BENCH/run.sh"

echo "=== newest CSVs:"
ls -t "$BENCH/results" | head -6
echo "=== DONE ==="
