#!/bin/bash
# Offline e2e throughput benchmark: vLLM native deep_gemm_mega_moe vs the
# flashinfer moe_ep mega backends, DeepSeek-V4-Flash, TP=4 + EP.
#
# Runs `vllm bench throughput` once per (backend, workload) cell and appends a
# CSV row parsed from its summary line.
#
# Env knobs:
#   BACKENDS   default "native fi_dg fi_nvfp4"
#   WORKLOADS  default "prefill:1024:1 decode:128:256 mixed:512:128"
#              (name:input_len:output_len)
#   NUM_PROMPTS default 256
#   EAGER      default 0  (1 -> --enforce-eager on all runs)
#   OUT_CSV    default results/bench_<stamp>.csv
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/venv0251/bin/activate"

MODEL=${MODEL:-/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-ai_deepseek-v4-flash/hf/hf-6e76323_orig}
BACKENDS=${BACKENDS:-"native fi_dg fi_nvfp4"}
WORKLOADS=${WORKLOADS:-"prefill:1024:1 decode:128:256 mixed:512:128"}
NUM_PROMPTS=${NUM_PROMPTS:-256}
EAGER=${EAGER:-0}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=${OUT_CSV:-$HERE/results/bench_${STAMP}.csv}
LOG_DIR=$HERE/logs

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR"
echo "backend,workload,input_len,output_len,num_prompts,eager,requests_per_s,total_tok_per_s,output_tok_per_s" >> "$OUT_CSV"

# `vllm bench throughput` takes the backend on the command line, so this
# emits the bare moe_backend string rather than an env assignment.
backend_moe() {
    case "$1" in
        native)   echo "deep_gemm_mega_moe" ;;
        fi_dg)    echo "flashinfer_moe_ep_mega_deep_gemm_sm100" ;;
        fi_nvfp4) echo "flashinfer_moe_ep_mega_cutedsl_sm100_nvfp4" ;;
        fi_mxfp8) echo "flashinfer_moe_ep_mega_cutedsl_sm100_mxfp8" ;;
        *) echo "unknown backend $1" >&2; return 1 ;;
    esac
}

for backend in $BACKENDS; do
    moe_backend=$(backend_moe "$backend") || exit 1
    for wl in $WORKLOADS; do
        IFS=: read -r name ilen olen <<<"$wl"
        log="$LOG_DIR/bench_${STAMP}_${backend}_${name}.log"
        echo "=== $backend / $name (in=$ilen out=$olen n=$NUM_PROMPTS) -> $log"
        extra=()
        [[ "$EAGER" == "1" ]] && extra+=(--enforce-eager)
        vllm bench throughput \
            --model "$MODEL" --trust-remote-code --tokenizer-mode deepseek_v4 \
            --tensor-parallel-size 4 --enable-expert-parallel \
            --moe-backend "$moe_backend" \
            --kv-cache-dtype fp8 --block-size 256 \
            --max-model-len 4096 --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
            --input-len "$ilen" --output-len "$olen" --num-prompts "$NUM_PROMPTS" \
            --output-json "$HERE/results/bench_${STAMP}_${backend}_${name}.json" \
            "${extra[@]}" 2>&1 | tee "$log"
        # summary line: "Throughput: X requests/s, Y total tokens/s, Z output tokens/s"
        line=$(grep -Eo "Throughput: [0-9.]+ requests/s, [0-9.]+ total tokens/s, [0-9.]+ output tokens/s" "$log" | tail -1)
        rps=$(echo "$line" | grep -Eo "[0-9.]+ requests" | grep -Eo "[0-9.]+")
        tts=$(echo "$line" | grep -Eo "[0-9.]+ total"    | grep -Eo "[0-9.]+")
        ots=$(echo "$line" | grep -Eo "[0-9.]+ output"   | grep -Eo "[0-9.]+")
        echo "$backend,$name,$ilen,$olen,$NUM_PROMPTS,$EAGER,${rps:-NA},${tts:-NA},${ots:-NA}" >> "$OUT_CSV"
    done
done
echo "wrote $OUT_CSV"
cat "$OUT_CSV"
