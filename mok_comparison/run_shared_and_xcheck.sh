#!/bin/bash
# 1) fi mxfp8_cutedsl benchmark WITH a MoK-parity shared expert in the timed
#    region (MEGA_SHARED_EXPERT=1), tuned knob cache, both timing modes.
# 2) fi-vs-MoK same-input cross-check: both forwards on identical
#    deterministic inputs, outputs compared (rel-L2).
# Requires: MoK built in-tree at $ROOT/mixture-of-kittens (run_mok_fwd_bench.sh
# does that) and the fi checkout at $REPO.
set -uo pipefail

ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
REPO=$ROOT/flashinfer-2/flashinfer-moe_ep
BENCH=$ROOT/moe_ep_benchmark
MOK=$ROOT/mixture-of-kittens
XDIR=$ROOT/logs_mok/xcheck
mkdir -p "$XDIR"
cp "$BENCH/mok_comparison/xcheck_common.py" "$XDIR/"
cp "$BENCH/mok_comparison/xcheck_mok.py" "$XDIR/"
cd "$REPO"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASHINFER_MOE_EP_KNOB_CACHE=$ROOT/logs_mok/moe_ep_knob_cache_moklike.json

echo "=== node: $(hostname); GPUs:"; nvidia-smi -L

PIP_CONSTRAINT="" BUILD_NIXL_EP=0 python -m pip install --no-build-isolation -e . || exit 1
python -m pip install --upgrade "nvidia-cutlass-dsl[cu13]" || exit 1
python -c "import flashinfer; print('flashinfer ->', flashinfer.__file__)"

export SECTION=fi_mega GPUS=4
export TOKENS=2048 NUM_EXPERTS=384 TOPK=6 HIDDEN=7168 INTER=3072
export WARMUP=100 ITERS=100
export MEGA_LIST=mxfp8_cutedsl

echo "=== fi_mega +shared +timed-quant (full MoK scope), tuned, MEGA_TIMING=e2e_pipelined"
MEGA_SHARED_EXPERT=1 MEGA_TIMED_QUANT=1 MEGA_TIMING=e2e_pipelined STAMP=moklike_sharedquant_pipelined bash "$BENCH/run.sh"

echo "=== fi_mega +shared +timed-quant (full MoK scope), tuned, MEGA_TIMING=e2e"
MEGA_SHARED_EXPERT=1 MEGA_TIMED_QUANT=1 MEGA_TIMING=e2e STAMP=moklike_sharedquant_e2e bash "$BENCH/run.sh"

echo "=== xcheck: fi side (same-input protocol, output dumped)"
MEGA_XCHECK_DIR=$XDIR MEGA_SHARED_EXPERT=1 MEGA_ACC=0 MEGA_TIMING=e2e \
  WARMUP=5 ITERS=10 STAMP=xcheck bash "$BENCH/run.sh" || exit 1

echo "=== xcheck: MoK side"
cd "$MOK"
MEGA_XCHECK_DIR=$XDIR TOKENS=$TOKENS NUM_EXPERTS=$NUM_EXPERTS TOPK=$TOPK \
  HIDDEN=$HIDDEN INTER=$INTER \
  PYTHONPATH="$MOK:$XDIR" torchrun --standalone --nproc-per-node=4 "$XDIR/xcheck_mok.py" || exit 1

echo "=== xcheck: compare"
python "$BENCH/mok_comparison/xcheck_compare.py" "$XDIR" 4 5.0
echo "XCHECK_EXIT=$?"
echo "=== DONE ==="
