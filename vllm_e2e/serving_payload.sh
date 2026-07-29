#!/bin/bash
# Serving-mode e2e sweep payload — runs INSIDE the container, $W as cwd.
#
# The offline sweep (bench_offline.py) times llm.generate() in-process. This
# is the same comparison run the way a deployment sees it: one process serves
# an OpenAI endpoint (`vllm serve --moe-backend <be>`), a second drives it
# (`vllm bench serve`, random dataset, fixed lengths). Backend selection is a
# server CLI flag here, not the MOE_BACKEND env the offline harness uses —
# that flag is the demonstration.
#
# The cells are the offline sweep's four workloads, verbatim — same lengths,
# same request counts, same per-cell engine settings — so each serving table
# row corresponds 1:1 to an offline table row:
#
#   pre8k   1024 in /    1 out, 256 requests, all in flight   [headline]
#   dec1k    128 in /  256 out, 1024 requests, all in flight  [headline]
#   lc100k 100000 in / 1024 out, 32 requests @ conc 32        [interactivity]
#   ctx32k  32768 in /   32 out, 32 requests @ conc 32        [interactivity]
#
# One server boot per (cell, backend): the per-cell engine settings (capture
# ladder, max-model-len, max-num-seqs, gpu-mem-util) are part of the cell
# definition, exactly as in §3c, so they cannot be shared across cells.
#
# Each cell runs ROUNDS+1 client rounds against the same server; round 0 is
# a warmup and is excluded from the summary, which reports the median of the
# timed rounds with min..max spread. Same design as bench_offline, for the
# same two reasons: the first requests after boot pay one-time JIT/tuning
# costs (measured: fi_cutedsl conc-1024 read 26101 tok/s cold vs 28122 warm),
# and the native baseline drifts round-over-round (measured: 26786 vs 28997
# tok/s on two nodes under an identical warm protocol — larger than the
# fi-vs-native effect itself).
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
#   CELLS             subset of "pre8k dec1k lc100k ctx32k"
#   ROUNDS            timed client rounds/cell (default 3; lc100k runs 2,
#                     like the offline sweep — override with ROUNDS_LC100K)
#   PORT              server port                (default 30000)
#   HEALTH_TIMEOUT_S  server-boot budget         (default 2400)
#
# Output: results/serving_${SERVE_PREFIX}_${cell}_${backend}_r${round}.json
# (r0 = warmup, excluded from the summary). Server logs in
# logs/serving_${SERVE_PREFIX}_${cell}_${backend}_<jobid>.log.
set -uo pipefail

: "${SERVE_MX:?see header}" "${SERVE_NVFP4:?see header}" "${SERVE_PREFIX:?see header}"
SERVE_KNOB_CACHE=${SERVE_KNOB_CACHE:-}
CELLS=${CELLS:-pre8k dec1k lc100k ctx32k}
ROUNDS=${ROUNDS:-3}
ROUNDS_LC100K=${ROUNDS_LC100K:-2}
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

# cell_params <cell> -> sets ISL OSL NPROMPTS CONC CELL_ROUNDS SERVER_ARGS.
# The values mirror §3c / job_vllm_pr_runbook_sweep_*.sh cell for cell; keep
# them in sync or the serving row stops corresponding to the offline row.
cell_params() {
    case $1 in
        pre8k)  ISL=1024;  OSL=1;    NPROMPTS=256;  CONC=256;  CELL_ROUNDS=$ROUNDS
                SERVER_ARGS='--max-model-len 4096 --max-num-batched-tokens 8192
                    --compilation-config {"max_cudagraph_capture_size":8192,"cudagraph_capture_sizes":[256,2048,4096,8192]}' ;;
        dec1k)  ISL=128;   OSL=256;  NPROMPTS=1024; CONC=1024; CELL_ROUNDS=$ROUNDS
                SERVER_ARGS='--max-model-len 4096 --max-num-seqs 1024 --max-num-batched-tokens 4096
                    --compilation-config {"max_cudagraph_capture_size":4096,"cudagraph_capture_sizes":[256,1024,2048,4096]}' ;;
        lc100k) ISL=100000; OSL=1024; NPROMPTS=32;  CONC=32;   CELL_ROUNDS=$ROUNDS_LC100K
                SERVER_ARGS='--max-model-len 102400 --max-num-seqs 32 --max-num-batched-tokens 8192
                    --gpu-memory-utilization 0.93
                    --compilation-config {"max_cudagraph_capture_size":8192,"cudagraph_capture_sizes":[32,256,2048,8192]}' ;;
        ctx32k) ISL=32768; OSL=32;   NPROMPTS=32;   CONC=32;   CELL_ROUNDS=$ROUNDS
                SERVER_ARGS='--max-model-len 33792 --max-num-seqs 32 --max-num-batched-tokens 8192
                    --gpu-memory-utilization 0.93
                    --compilation-config {"max_cudagraph_capture_size":8192,"cudagraph_capture_sizes":[32,256,2048,8192]}' ;;
        *) echo "unknown cell: $1 (pre8k|dec1k|lc100k|ctx32k)" >&2; return 2 ;;
    esac
}

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

# serve_cell <cell> <short> <backend> <model>
serve_cell() {
    local cell=$1 short=$2 be=$3 model=$4
    cell_params "$cell" || return 2
    local slog=logs/serving_${SERVE_PREFIX}_${cell}_${short}_${JOBTAG}.log
    local -a senv=(FI_MOE_EP_SKIP_VERSION_CHECK=1)
    [[ $short == fi_cutedsl && -n $SERVE_KNOB_CACHE ]] && \
        senv+=("FLASHINFER_MOE_EP_KNOB_CACHE=$SERVE_KNOB_CACHE")

    echo; echo "=== $SERVE_PREFIX / $cell / $short: vllm serve --moe-backend $be (model=$(basename "$model")) ==="
    echo "    server log: $slog"
    # SERVER_ARGS is deliberately unquoted: it is a flat flag list (the
    # compilation-config JSON contains no spaces).
    env "${senv[@]}" vllm serve "$model" \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --moe-backend "$be" \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --no-enable-prefix-caching \
        $SERVER_ARGS \
        --host 127.0.0.1 --port "$PORT" \
        >"$slog" 2>&1 &
    local spid=$!

    local up=0 i
    for i in $(seq 1 $((HEALTH_TIMEOUT_S / 5))); do
        if ! kill -0 "$spid" 2>/dev/null; then
            echo "[serving] $cell/$short: server died during startup; tail of $slog:"
            tail -25 "$slog"
            wait_gpu_idle
            return 1
        fi
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        sleep 5
    done
    if [[ $up -ne 1 ]]; then
        echo "[serving] $cell/$short: no /health after ${HEALTH_TIMEOUT_S}s; tail of $slog:"
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
    echo "[serving] $cell/$short: server healthy, banner check passed ($banners/8 fi ranks)"

    # Round 0 is a full warmup round, discarded — bench_offline's round 0.
    local r rc=0
    for r in $(seq 0 "$CELL_ROUNDS"); do
        local out=serving_${SERVE_PREFIX}_${cell}_${short}_r${r}.json
        echo; echo "--- $SERVE_PREFIX / $cell / $short round $r/$CELL_ROUNDS$( [[ $r -eq 0 ]] && echo ' (warmup)' ) ---"
        vllm bench serve \
            --backend vllm --host 127.0.0.1 --port "$PORT" \
            --model "$model" --tokenizer-mode deepseek_v4 \
            --dataset-name random \
            --random-input-len "$ISL" --random-output-len "$OSL" \
            --random-range-ratio 0 --seed 0 \
            --num-prompts "$NPROMPTS" --max-concurrency "$CONC" \
            --ignore-eos --disable-tqdm \
            --save-result --result-dir results --result-filename "$out" \
            || { echo "[serving] $cell/$short round $r FAILED"; rc=1; }
    done

    stop_server "$spid"
    return $rc
}

for cell in $CELLS; do
    # All three backends of a cell back to back, same order as the offline
    # sweep — the ratio is the claim, so the baseline runs closest in time.
    serve_cell "$cell" native     deep_gemm_mega_moe               "$SERVE_MX"    || echo "[serving] $cell/native FAILED (continuing)"
    serve_cell "$cell" fi_dg      flashinfer_moe_ep_mega_deep_gemm "$SERVE_MX"    || echo "[serving] $cell/fi_dg FAILED (continuing)"
    serve_cell "$cell" fi_cutedsl flashinfer_moe_ep_mega_cutedsl   "$SERVE_NVFP4" || echo "[serving] $cell/fi_cutedsl FAILED (continuing)"
done

echo; echo '########## SUMMARY'
SERVE_PREFIX=$SERVE_PREFIX CELLS=$CELLS ROUNDS=$ROUNDS ROUNDS_LC100K=$ROUNDS_LC100K python - <<'PY'
import json, os

RUN_T0 = float(os.environ.get("RUN_T0", 0))
prefix = os.environ["SERVE_PREFIX"]
cells = os.environ["CELLS"].split()
rounds = int(os.environ["ROUNDS"])
rounds_lc = int(os.environ["ROUNDS_LC100K"])

def fresh(p):
    return os.path.exists(p) and os.path.getmtime(p) >= RUN_T0

def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

NOTE = {
    "pre8k": "1024 in / 1 out, 256 requests",
    "dec1k": "128 in / 256 out, 1024 requests",
    "lc100k": "100K in / 1K out @ conc 32",
    "ctx32k": "32K in / 32 out @ conc 32",
}
for cell in cells:
    n = rounds_lc if cell == "lc100k" else rounds
    print(f"\n{cell}  ({NOTE.get(cell, '')}; median of {n} rounds, r0 warmup discarded)")
    print(f"  {'backend':11s} {'tok/s':>9s} {'vs native':>10s} "
          f"{'TTFT p50':>9s} {'ITL p50':>8s} {'ITL p99':>8s}  spread")
    base = None
    for short in ("native", "fi_dg", "fi_cutedsl"):
        ds = [json.load(open(p))
              for r in range(1, n + 1)
              if fresh(p := f"results/serving_{prefix}_{cell}_{short}_r{r}.json")]
        if not ds:
            print(f"  {short:11s} MISSING")
            continue
        # Headline on TOTAL tok/s, matching the offline tables (§1/§2).
        tots = [d["total_token_throughput"] for d in ds]
        v = med(tots)
        d = next(x for x in ds if x["total_token_throughput"] == v)  # median round
        if base is None:
            base = v
        def g(k, s=1.0):
            x = d.get(k)
            return f"{x * s:8.1f}" if x is not None else f"{'-':>8s}"
        print(f"  {short:11s} {v:9.0f} {v / base:9.3f}x {g('median_ttft_ms', 1e-3):>9s} "
              f"{g('median_itl_ms')} {g('p99_itl_ms')}  "
              f"{min(tots):.0f}..{max(tots):.0f}")
print("\n  tok/s is TOTAL token throughput (input+output, client-measured),")
print("  the same headline as the offline tables; ratios compare median")
print("  rounds. TTFT seconds, ITL milliseconds, from the median round.")
PY
