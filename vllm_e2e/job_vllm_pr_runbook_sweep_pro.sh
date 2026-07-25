#!/bin/bash
#SBATCH --job-name=deepseekv4pro.vllm_pr_sweep_pro
#SBATCH --account=coreai_libraries_cudnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --output=vllm_pr_sweep_pro_%j.log
#
# DeepSeek-V4-Pro at EP8/TP8 on 1x8 GB200, against the reshaped API (two
# backends, module at vllm/utils/flashinfer_moe_ep.py). Same four cells as the
# Flash sweeps, so Pro and Flash rows are directly comparable.
#
#   tier 1        config checks (no model)
#   prefill-8k    prefill:1024:1  x256, capture 8192   [headline]
#   decode-1k     decode:128:256  x1024, capture 4096  [headline]
#   longctx-100k  100000:1024     x32 @conc 32         [interactivity]
#   ctx32k        32768:32        x32 @conc 32         [interactivity, new]
#
# The 0.6.15 venv is below the new flashinfer floor, so the version gate is
# explicitly skipped -- that is the documented pre-release escape hatch.
#
# Provenance: derived 2026-07-25 from job_vllm_pr_runbook_sweep_ep8.sh (the
# script behind job 2337204) by swapping the checkpoints to V4-Pro and the knob
# cache to knob_cache_pro_ep8.json. The cells are otherwise byte-identical to
# that job, which is what makes the Pro and Flash tables comparable -- keep them
# that way.
#
# Usage:
#   cd <repo>/vllm_e2e && sbatch job_vllm_pr_runbook_sweep_pro.sh
#
#   The %j log lands in the submit directory, so cd here first.
#
# Overrides (all optional; sbatch exports the submitting env by default):
#   ROOT=/my/scratch          checkout root; default below
#   IMG=/path/to.sqsh         container image
#   ROUNDS=3                  timed rounds per cell (plus round 0 warmup)
#   MODEL=... MODEL_NVFP4=... checkpoints; unset falls back to the mirror
#                             paths compiled into bench_offline.py
#   EXTRA_MOUNTS=a:a,b:b      appended to --container-mounts, e.g. when the
#                             checkpoints live outside $ROOT
#   sbatch -A <acct> -p <part> ...   CLI flags override the #SBATCH lines above
set -uo pipefail

# NB: do not "improve" this into a BASH_SOURCE-derived path -- sbatch copies the
# script to a spool dir, so the script's own location is not the checkout.
ROOT=${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}
W=$ROOT/moe_ep_benchmark/vllm_e2e
IMG=${IMG:-$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh}
ROUNDS=${ROUNDS:-3}
MOUNTS="$ROOT:$ROOT,/lustre/share:/lustre/share:ro"
[[ -n "${EXTRA_MOUNTS:-}" ]] && MOUNTS="$MOUNTS,$EXTRA_MOUNTS"

# Forward the checkpoint overrides only when set, so an unset MODEL stays unset
# inside the container rather than becoming "" (which bench_offline.py would
# treat as an explicit empty path instead of falling back to its default).
FWD=""
[[ -n "${MODEL:-}" ]]       && FWD+="export MODEL='$MODEL'; "
[[ -n "${MODEL_NVFP4:-}" ]] && FWD+="export MODEL_NVFP4='$MODEL_NVFP4'; "

# The Pro cells resolve their checkpoint from MODEL_MX_PRO / MODEL_NVFP4_PRO
# rather than from bench_offline's Flash-specific defaults, so these two must
# reach the container. They are required, not optional: fail here instead of
# tripping the container's `set -u` an hour into the allocation.
for v in MODEL_MX_PRO MODEL_NVFP4_PRO; do
    [[ -n "${!v:-}" ]] || { echo "$v is unset -- submit via scratch_runbook_submit_sweep_pro.sh"; exit 2; }
    [[ -d "${!v}" ]]   || { echo "$v=${!v} is not a directory"; exit 2; }
    FWD+="export $v='${!v}'; "
done

srun --ntasks=1 \
  --container-image="$IMG" \
  --container-name=fivllm_sweep \
  --container-mounts="$MOUNTS" \
  --container-workdir="$W" \
  bash -lc "
set -uo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=$ROOT/.cache/huggingface
export PIP_CACHE_DIR=$ROOT/.cache/pip
export FLASHINFER_WORKSPACE_BASE=$ROOT/.cache/flashinfer-root-ws
export FI_MOE_EP_SKIP_VERSION_CHECK=1
export TP=8   # V4-Pro at EP8/TP8 (1x8 GPU)
$FWD

echo '=== node ==='; hostname; nvidia-smi -L | head -4
echo '=== harness ==='; git -C $ROOT/moe_ep_benchmark log --oneline -1
echo '=== flashinfer ==='; git -C $ROOT/flashinfer-2/flashinfer-moe_ep log --oneline -1
echo '=== checkpoints ==='; echo \"MODEL=\${MODEL:-<bench_offline default>}\"; echo \"MODEL_NVFP4=\${MODEL_NVFP4:-<bench_offline default>}\"
source venv0251/bin/activate || exit 1
bash patch_0251/apply.sh || exit 1

echo; echo '########## TIER 1'
python test_backend_registration.py; t1=\$?
[[ \$t1 -ne 0 ]] && { echo \"tier1 FAILED (\$t1)\"; exit \$t1; }

DG=flashinfer_moe_ep_mega_deep_gemm
CUTEDSL=flashinfer_moe_ep_mega_cutedsl

# cell <tag> <extra-env-as-string> <bench args...>
cell() {
    local name=\$1; shift
    local envs=\$1; shift
    for be in deep_gemm_mega_moe \$DG \$CUTEDSL; do
        local short=native
        [[ \$be == \$DG ]] && short=fi_dg
        [[ \$be == \$CUTEDSL ]] && short=fi_cutedsl
        # V4-Pro: pass --model explicitly per backend (resolve_model's default is
        # Flash-specific). native/fi_dg -> Pro mx; fi_cutedsl -> pinned Pro NVFP4.
        local model=\$MODEL_MX_PRO
        local cache=''
        if [[ \$short == fi_cutedsl ]]; then
            model=\$MODEL_NVFP4_PRO
            cache=FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/knob_cache_pro_ep8.json
        fi
        echo; echo \"--- \$name / \$short (model=\$(basename \$model)) ---\"
        env \$envs MOE_BACKEND=\$be \$cache \
            python bench_offline.py --model \"\$model\" --tag sw_pro_\${name}_\${short} \"\$@\" \
            --out results/sweep_pro_\${name}_\${short}.json 2>&1 \
            | grep -E '^\\[bench_offline\\]|Error|Traceback' | tail -8
    done
}

# Stale-result guard: the summary below only accepts JSONs written after
# this point. Without it, a run whose cells all fail still prints a full
# plausible table from the result files committed in the repo (observed:
# job 2337618, 12/12 cells failed on the DSL guard, rc=0, summary looked
# perfect). Committed results must never masquerade as a fresh run.
export RUN_T0=\$(date +%s)

echo; echo '########## PREFILL-8K (headline)'
cell pre8k 'ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 CAPTURE_SIZES=256,2048,4096,8192' \
    --workload prefill:1024:1 --num-prompts 256 --rounds $ROUNDS

echo; echo '########## DECODE-1K (headline)'
# CAPTURE_SIZES pinned 2026-07-25 (measured in jobs 2337473 / 2337487). Without
# it this was the only cell setting MAX_CAPTURE while inheriting vLLM's dense
# default capture ladder, and the cudagraph memory profiler then reserved
# ~48 GiB/GPU for the flashinfer backends against a real capture cost of
# ~6 GiB -- the same ~6 GiB it estimates correctly for native. The phantom
# reservation came out of the KV cache: on V4-Pro EP8, fi_dg held only 189 of
# the 1024 requested sequences (0.44x native, with *better* ITL because the
# batches were tiny) and fi_cutedsl could not allocate a KV cache at all.
# Pinning restores fi_dg to 1.02x and fi_cutedsl to 1.19x.
# NB: this changes the cell. Native loses ~3% to padding, so dec1k numbers
# recorded before 2026-07-25 are not comparable with ones recorded after.
cell dec1k 'ENFORCE_EAGER=0 MAX_CAPTURE=4096 MAX_NUM_SEQS=1024 CAPTURE_SIZES=256,1024,2048,4096' \
    --workload decode:128:256 --num-prompts 1024 --rounds $ROUNDS

echo; echo '########## LONGCTX 100K/1K @ conc 32'
cell lc100k 'ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 CAPTURE_SIZES=32,256,2048,8192 MAX_NUM_SEQS=32 MAX_MODEL_LEN=102400 GPU_MEM_UTIL=0.93 REQUIRE_LATENCY=1' \
    --workload longctx:100000:1024 --num-prompts 32 --rounds 2

echo; echo '########## CTX 32K/32 @ conc 32'
cell ctx32k 'ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 CAPTURE_SIZES=32,256,2048,8192 MAX_NUM_SEQS=32 MAX_MODEL_LEN=33792 GPU_MEM_UTIL=0.93 REQUIRE_LATENCY=1' \
    --workload longctx:32768:32 --num-prompts 32 --rounds $ROUNDS

echo; echo '########## SUMMARY'
python - <<'PY'
import json, os

RUN_T0 = float(os.environ.get('RUN_T0', 0))

def fresh(path):
    return os.path.exists(path) and os.path.getmtime(path) >= RUN_T0

CELLS = [
    ('prefill-8k',      'pre8k',  'prefill 1024x1, 256 prompts, capture 8192'),
    ('decode-1k',       'dec1k',  'decode 128->256, 1024 seqs, capture 4096'),
    ('100K ISL / 1K',   'lc100k', '32 concurrent'),
    ('32K ISL / 32',    'ctx32k', '32 concurrent'),
]
ROWS = [('native', 'native'), ('fi_dg', 'fi_dg'), ('fi_cutedsl', 'fi_cutedsl')]

for title, stem, note in CELLS:
    print(f\"\\n{title}  ({note})\")
    print(f\"  {'backend':11s} {'tok/s':>9s} {'vs native':>10s} \"
          f\"{'TTFT p50':>9s} {'ITL p50':>9s} {'ITL p99':>9s}\")
    base = None
    for label, short in ROWS:
        p = f'results/sweep_pro_{stem}_{short}.json'
        if not fresh(p):
            print(f'  {label:11s} MISSING'); continue
        d = json.load(open(p))
        v = d['median_total_tok_per_s']
        if base is None:
            base = v
        m = d.get('median_latency') or {}
        def g(k, s=1.0):
            x = m.get(k)
            return f'{x * s:9.1f}' if x is not None else f'{\"-\":>9s}'
        print(f\"  {label:11s} {v:9.0f} {v / base:9.3f}x {g('ttft_s_p50')} \"
              f\"{g('itl_s_p50', 1e3)} {g('itl_s_p99', 1e3)}\")
print('\\n  TTFT seconds, ITL milliseconds. Latency only collected on the')
print('  interactivity cells (REQUIRE_LATENCY=1).')
PY
"
rc=$?
echo "=== job exit rc=$rc ==="
exit $rc
