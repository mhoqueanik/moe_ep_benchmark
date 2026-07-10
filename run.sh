#!/bin/bash
# Standalone MoE-EP benchmark launcher -- single node, N GPUs.
# Runs three subsections (select with SECTION= or run all by default):
#   vllm_split : vLLM split-path (dispatch/combine + local compute) via bench_moe_ep_nonmega.py
#   vllm_mega  : vLLM fused mega path via bench_moe_ep_vllm_mega.py
#   fi_mega    : FlashInfer fused mega backends via bench_moe_ep_mega.py
# Each subsection writes its own suffixed .log and .csv.
# One process is spawned per GPU inside each run (torch.multiprocessing),
# so each variant is a single launch. Mirrors DP=N + EP (TP=1).
#
# Split path (vLLM, dispatch/combine + local compute):
#   algo=ll : DeepEP low-latency all2all    + DeepGEMM batched (masked) grouped GEMM
#   algo=ht : DeepEP high-throughput all2all + DeepGEMM contiguous grouped GEMM
#   trtllm  : DeepEP high-throughput all2all + TRT-LLM-gen fp8 block-scale GEMM (HT-only)
#
# Mega path (fused dispatch+GEMM+combine):
#   vLLM : vllm_deep_gemm_mega (DeepseekV4MegaMoEExperts / deep_gemm.fp8_fp4_mega_moe)
#   FI   : deep_gemm_mega | mxfp8_cutedsl | nvfp4_cutedsl
#
#   ALGO=ll ./moe_ep_benchmark/run.sh
#   SECTION=vllm_split ./moe_ep_benchmark/run.sh
#   EXCLUDE_QUANT=0 ./moe_ep_benchmark/run.sh   # include input quant on split HT path
#   MEGA_LIST="" VLLM_MEGA=0 ./moe_ep_benchmark/run.sh
#
# NOTE: to compare ONLY the local expert GEMM under an IDENTICAL all-to-all
# (DeepEP-HT), set ALGO=ht so both variants use HT.
set -uo pipefail

# --- vLLM venv (non-mega); FI_PYTHON defaults to PYTHON if unset ---
: "${PYTHON:=python}"
: "${FI_PYTHON:=$PYTHON}"

GPUS="${GPUS:-4}"                       # world size = DP = EP
DEVS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ALGO="${ALGO:-ht}"                      # ht (prefill) | ll (decode); ht enables exclude-quant on split
TOKENS="${TOKENS:-8}"                   # tokens per rank
NUM_EXPERTS="${NUM_EXPERTS:-256}"
TOPK="${TOPK:-8}"
HIDDEN="${HIDDEN:-7168}"
INTER="${INTER:-2048}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-50}"
EXPERTS_LIST="${EXPERTS_LIST:-deepgemm trtllm}"   # local MoE compute variants to sweep
MEGA_LIST="${MEGA_LIST:-deep_gemm_mega mxfp8_cutedsl nvfp4_cutedsl}"
VLLM_MEGA="${VLLM_MEGA:-1}"                       # 1 = include vllm_mega subsection when running all
EXCLUDE_QUANT="${EXCLUDE_QUANT:-1}"               # 1 = lift input act-quant out of timing (HT)
# SECTION: vllm_split | vllm_mega | fi_mega | all (default)
SECTION="${SECTION:-all}"

QUANT_FLAG=(--exclude-quant)
[ "$EXCLUDE_QUANT" = "0" ] && QUANT_FLAG=(--no-exclude-quant)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/results}"
mkdir -p "$OUT_DIR"

# Set by run_section / run_sweep; bench_${STAMP}_${section}.{log,csv} by default.
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_PREFIX="${RUN_PREFIX:-bench}"
CSV=""

run_one () {
    local experts="$1" algo="$2"; shift 2
    echo ""
    echo ">>> RUN path=split  experts=${experts}  algo=${algo}  tokens/rank=${TOKENS}  gpus=${GPUS}"
    CUDA_VISIBLE_DEVICES="$DEVS" "$PYTHON" "$HERE/bench_moe_ep_nonmega.py" \
        --world-size "$GPUS" \
        --algorithm "$algo" \
        --experts-backend "$experts" \
        --tokens-per-rank "$TOKENS" \
        --num-experts "$NUM_EXPERTS" \
        --top-k "$TOPK" \
        --hidden "$HIDDEN" \
        --intermediate "$INTER" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --out-csv "$CSV" \
        "${QUANT_FLAG[@]}" \
        "$@"
}

run_mega () {
    local backend="$1"; shift
    echo ""
    echo ">>> RUN path=mega  backend=${backend}  tokens/rank=${TOKENS}  gpus=${GPUS}"
    CUDA_VISIBLE_DEVICES="$DEVS" "$FI_PYTHON" "$HERE/bench_moe_ep_mega.py" \
        --world-size "$GPUS" \
        --mega-backend "$backend" \
        --tokens-per-rank "$TOKENS" \
        --num-experts "$NUM_EXPERTS" \
        --top-k "$TOPK" \
        --hidden "$HIDDEN" \
        --intermediate "$INTER" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --out-csv "$CSV" \
        "$@"
}

run_vllm_mega () {
    echo ""
    echo ">>> RUN path=vllm_mega  backend=vllm_deep_gemm_mega  tokens/rank=${TOKENS}  gpus=${GPUS}"
    CUDA_VISIBLE_DEVICES="$DEVS" "$PYTHON" "$HERE/bench_moe_ep_vllm_mega.py" \
        --world-size "$GPUS" \
        --tokens-per-rank "$TOKENS" \
        --num-experts "$NUM_EXPERTS" \
        --top-k "$TOPK" \
        --hidden "$HIDDEN" \
        --intermediate "$INTER" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --out-csv "$CSV" \
        "$@"
}

run_vllm_split_bench () {
    for experts in $EXPERTS_LIST; do
        algo="$ALGO"
        if [ "$experts" = "trtllm" ] && [ "$algo" = "ll" ]; then
            echo "[note] trtllm-fp8 experts are HT-only (Standard layout); forcing algo=ht"
            algo="ht"
        fi
        run_one "$experts" "$algo" "$@" \
            || echo "[warn] variant experts=${experts} algo=${algo} failed (continuing)"
    done
}

run_vllm_mega_bench () {
    run_vllm_mega "$@" \
        || echo "[warn] vllm mega variant failed (continuing)"
}

run_fi_mega_bench () {
    for backend in $MEGA_LIST; do
        run_mega "$backend" "$@" \
            || echo "[warn] mega variant backend=${backend} failed (continuing)"
    done
}

sections_to_run () {
    if [ "$SECTION" != "all" ]; then
        echo "$SECTION"
        return
    fi
    echo vllm_split
    if [ "$VLLM_MEGA" = "1" ]; then
        echo vllm_mega
    fi
    echo fi_mega
}

run_section () {
    local sec="$1"; shift
    local log="${OUT_DIR}/${RUN_PREFIX}_${STAMP}_${sec}.log"
    CSV="${OUT_DIR}/${RUN_PREFIX}_${STAMP}_${sec}.csv"

    {
        echo "============================================================"
        echo "  SECTION ${sec}"
        echo "============================================================"
        echo "log: ${log}"
        echo "csv: ${CSV}"
        echo "tokens/rank=${TOKENS}  algo=${ALGO}  gpus=${GPUS}  exclude_quant=${EXCLUDE_QUANT}"
        echo ""

        case "$sec" in
            vllm_split)
                run_vllm_split_bench "$@"
                ;;
            vllm_mega)
                run_vllm_mega_bench "$@"
                ;;
            fi_mega)
                run_fi_mega_bench "$@"
                ;;
            *)
                echo "[error] unknown section: ${sec} (expected vllm_split, vllm_mega, fi_mega, or all)"
                return 1
                ;;
        esac

        echo ""
        echo "=== summary (${CSV}) ==="
        cat "$CSV" 2>/dev/null || true
    } 2>&1 | tee "$log"
}

run_bench_suite () {
    local sec
    for sec in $(sections_to_run); do
        run_section "$sec" "$@" \
            || echo "[warn] section ${sec} had failures (continuing)"
    done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_bench_suite "$@"
fi
