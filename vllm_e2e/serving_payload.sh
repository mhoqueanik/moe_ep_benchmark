#!/bin/bash
# Serving-mode e2e sweep payload — runs INSIDE the container, $W as cwd.
#
# The offline sweep (bench_offline.py) times llm.generate() in-process. This
# is the same comparison run the way a deployment sees it: one process serves
# an OpenAI endpoint (`vllm serve --moe-backend <be>`), a second drives it
# (`vllm bench serve`, random dataset, fixed lengths) at three concurrencies.
# Backend selection is a server CLI flag here, not the MOE_BACKEND env the
# offline harness uses — that flag is the demonstration.
#
# Driven by job_vllm_serving_sweep_{ep8,pro}.sh; runnable by hand on a held
# node too (RUNBOOK_REPRO.md §3g):
#
#   SERVE_MX=$CKPT/deepseek-v4-flash SERVE_NVFP4=$CKPT/deepseek-v4-flash-nvfp4 \
#   SERVE_PREFIX=ep8 SERVE_KNOB_CACHE=$W/results/knob_cache_ep8.json \
#     JOBID=$JOBID bash $W/in_container.sh 'bash serving_payload.sh'
#
# Inputs (env):
#   SERVE_MX          mx checkpoint — native + fi_dg           [required]
#   SERVE_NVFP4       NVFP4 checkpoint — fi_cutedsl            [required]
#   SERVE_PREFIX      output filename infix, ep8|pro           [required]
#   SERVE_KNOB_CACHE  fi_cutedsl knob cache; empty = heuristic
#   ISL/OSL           random dataset lengths     (default 8 / 1024)
#   CONCS             concurrency sweep          (default "32 128 1024")
#   PROMPTS_PER_CONC  num-prompts = this * C     (default 5)
#   ROUNDS            timed client rounds/cell   (default 3; median reported)
#   PORT              server port                (default 30000)
#   HEALTH_TIMEOUT_S  server-boot budget         (default 2400)
#
# Output: results/serving_${SERVE_PREFIX}_c${C}_${backend}_r${round}.json,
# ROUNDS files per cell; the summary reports the per-cell median of
# output_throughput. Server logs in
# logs/serving_${SERVE_PREFIX}_${backend}_<jobid>.log.
#
# ROUNDS exists for the same reason bench_offline repeats rounds in one
# engine: the native decode baseline drifts run-over-run. Measured at conc
# 1024 on Flash, two single rounds on two nodes put native at 26786 and
# 28997 tok/s (8%) under an identical protocol — larger than the fi-vs-native
# effect itself, so a single-round ratio is not trustworthy.
set -uo pipefail

: "${SERVE_MX:?see header}" "${SERVE_NVFP4:?see header}" "${SERVE_PREFIX:?see header}"
SERVE_KNOB_CACHE=${SERVE_KNOB_CACHE:-}
ISL=${ISL:-8}
OSL=${OSL:-1024}
CONCS=${CONCS:-32 128 1024}
PROMPTS_PER_CONC=${PROMPTS_PER_CONC:-5}
ROUNDS=${ROUNDS:-3}
PORT=${PORT:-30000}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-2400}
JOBTAG=${SLURM_JOB_ID:-manual}

source venv0251/bin/activate || exit 1
bash patch_0251/apply.sh || exit 1

echo; echo '########## TIER 1'
python test_backend_registration.py || { echo "tier1 FAILED"; exit 1; }

mkdir -p logs results
# Stale-result guard, same as the offline sweeps: the summary only accepts
# JSONs written after this point, so a dead cell reads MISSING instead of
# reprinting a committed result.
RUN_T0=$(date +%s); export RUN_T0

# max_model_len must cover ISL+OSL; keep a block of headroom for specials.
MAX_LEN=$(( (ISL + OSL + 511) / 256 * 256 ))

wait_gpu_idle() {
    # The next server cannot boot while the previous one still holds the GPUs.
    local i
    for i in $(seq 1 36); do
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && return 0
        sleep 5
    done
    echo "[serving] GPUs still busy after 180s — escalating to pkill -9" >&2
    pkill -9 -f 'vllm serve' 2>/dev/null
    pkill -9 -f 'VLLM::' 2>/dev/null
    sleep 20
    return 0
}

stop_server() {
    local spid=$1
    kill "$spid" 2>/dev/null
    wait "$spid" 2>/dev/null
    wait_gpu_idle
}

bench_client() {
    # bench_client <C> <num_prompts> <model> [save-args...]
    local C=$1 n=$2 model=$3; shift 3
    vllm bench serve \
        --backend vllm --host 127.0.0.1 --port "$PORT" \
        --model "$model" --tokenizer-mode deepseek_v4 \
        --dataset-name random \
        --random-input-len "$ISL" --random-output-len "$OSL" \
        --random-range-ratio 0 --seed 0 \
        --num-prompts "$n" --max-concurrency "$C" \
        --ignore-eos --disable-tqdm "$@"
}

serve_bench() {
    local short=$1 be=$2 model=$3
    local slog=logs/serving_${SERVE_PREFIX}_${short}_${JOBTAG}.log
    local -a senv=(FI_MOE_EP_SKIP_VERSION_CHECK=1)
    [[ $short == fi_cutedsl && -n $SERVE_KNOB_CACHE ]] && \
        senv+=("FLASHINFER_MOE_EP_KNOB_CACHE=$SERVE_KNOB_CACHE")

    echo; echo "=== $SERVE_PREFIX / $short: vllm serve --moe-backend $be (model=$(basename "$model")) ==="
    echo "    server log: $slog"
    # Same invariants as the offline dec1k cell: capture the decode step
    # shapes (sparse CAPTURE_SIZES ladder — never the dense default, see
    # expected_results.md §5.1) and no prefix caching (random prompts, but
    # keep the runs honest by construction).
    env "${senv[@]}" vllm serve "$model" \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --moe-backend "$be" \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --max-model-len "$MAX_LEN" \
        --max-num-seqs 1024 \
        --max-num-batched-tokens 4096 \
        --no-enable-prefix-caching \
        --compilation-config '{"max_cudagraph_capture_size": 4096, "cudagraph_capture_sizes": [256, 1024, 2048, 4096]}' \
        --host 127.0.0.1 --port "$PORT" \
        >"$slog" 2>&1 &
    local spid=$!

    local up=0 i
    for i in $(seq 1 $((HEALTH_TIMEOUT_S / 5))); do
        if ! kill -0 "$spid" 2>/dev/null; then
            echo "[serving] $short: server died during startup; tail of $slog:"
            tail -25 "$slog"
            wait_gpu_idle
            return 1
        fi
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        sleep 5
    done
    if [[ $up -ne 1 ]]; then
        echo "[serving] $short: no /health after ${HEALTH_TIMEOUT_S}s; tail of $slog:"
        tail -25 "$slog"
        stop_server "$spid"
        return 1
    fi

    # Routing proof, same rule as §4a: the fi backends must print the
    # [fi_moe_ep] banner once per EP rank; native must print none. A serving
    # cell that fails this is mis-routed and its numbers mean nothing.
    local banners
    banners=$(grep -c 'fi_moe_ep] ep_rank=' "$slog" || true)
    if [[ $short == native && $banners -ne 0 ]]; then
        echo "[serving] FAIL: native printed $banners [fi_moe_ep] banners"
        stop_server "$spid"; return 1
    elif [[ $short != native && $banners -ne 8 ]]; then
        echo "[serving] FAIL: $short printed $banners/8 [fi_moe_ep] banners"
        stop_server "$spid"; return 1
    fi
    echo "[serving] $short: server healthy, banner check passed ($banners/8 fi ranks)"

    # Warmup pass, discarded — the offline harness's round 0. First requests
    # pay residual JIT/allocation costs the timed runs must not include.
    echo "[serving] $short: warmup (64 prompts @ conc 32, discarded)"
    bench_client 32 64 "$model" >/dev/null 2>&1

    local C r rc=0
    for C in $CONCS; do
        local n=$((PROMPTS_PER_CONC * C))
        for r in $(seq 1 "$ROUNDS"); do
            local out=serving_${SERVE_PREFIX}_c${C}_${short}_r${r}.json
            echo; echo "--- $SERVE_PREFIX / $short / conc $C round $r/$ROUNDS ($n prompts, ${ISL}in/${OSL}out) ---"
            bench_client "$C" "$n" "$model" \
                --save-result --result-dir results --result-filename "$out" \
                || { echo "[serving] $short conc $C round $r FAILED"; rc=1; }
        done
    done

    stop_server "$spid"
    return $rc
}

serve_bench native     deep_gemm_mega_moe               "$SERVE_MX"    || echo "[serving] native FAILED (continuing)"
serve_bench fi_dg      flashinfer_moe_ep_mega_deep_gemm "$SERVE_MX"    || echo "[serving] fi_dg FAILED (continuing)"
serve_bench fi_cutedsl flashinfer_moe_ep_mega_cutedsl   "$SERVE_NVFP4" || echo "[serving] fi_cutedsl FAILED (continuing)"

echo; echo '########## SUMMARY'
SERVE_PREFIX=$SERVE_PREFIX CONCS=$CONCS ROUNDS=$ROUNDS python - <<'PY'
import json, os

RUN_T0 = float(os.environ.get("RUN_T0", 0))
prefix = os.environ["SERVE_PREFIX"]
concs = os.environ["CONCS"].split()
rounds = int(os.environ["ROUNDS"])

def fresh(p):
    return os.path.exists(p) and os.path.getmtime(p) >= RUN_T0

def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

for C in concs:
    print(f"\nconcurrency {C}  (median of {rounds} rounds; spread = min..max out tok/s)")
    print(f"  {'backend':11s} {'out tok/s':>10s} {'total tok/s':>11s} {'vs native':>10s} "
          f"{'TTFT p50':>9s} {'ITL p50':>8s} {'ITL p99':>8s}  spread")
    base = None
    for short in ("native", "fi_dg", "fi_cutedsl"):
        ds = [json.load(open(p))
              for r in range(1, rounds + 1)
              if fresh(p := f"results/serving_{prefix}_c{C}_{short}_r{r}.json")]
        if not ds:
            print(f"  {short:11s} MISSING")
            continue
        outs = [d["output_throughput"] for d in ds]
        v = med(outs)
        d = next(x for x in ds if x["output_throughput"] == v)  # median round
        if base is None:
            base = v
        def g(k, s=1.0):
            x = d.get(k)
            return f"{x * s:8.1f}" if x is not None else f"{'-':>8s}"
        print(f"  {short:11s} {v:10.0f} {d['total_token_throughput']:11.0f} "
              f"{v / base:9.3f}x {g('median_ttft_ms', 1e-3):>9s} "
              f"{g('median_itl_ms')} {g('p99_itl_ms')}  "
              f"{min(outs):.0f}..{max(outs):.0f}")
print("\n  output tok/s is the headline (decode-dominated workload; ratio is")
print("  fi vs native on it, median round vs median round). TTFT seconds,")
print("  ITL milliseconds; latency columns come from the median round.")
PY
