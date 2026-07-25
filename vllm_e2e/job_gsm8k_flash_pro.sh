#!/bin/bash
#SBATCH --job-name=deepseekv4.gsm8k_flash_pro
#SBATCH --account=coreai_libraries_cudnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --output=gsm8k_flash_pro_%j.log
#
# GSM8K accuracy gate across BOTH checkpoints of BOTH models:
#   Flash EP4/TP4 : native, fi_dg (mx)  +  fi_cutedsl (NVFP4 cast)
#   Pro   EP8/TP8 : native, fi_dg (mx)  +  fi_cutedsl (NVFP4 cast)
#
# WHY THIS IS NOT A RERUN OF THE 2026-07-25 03:42 GSM8K (job 2337127 step 5):
# that run exported MODEL=<mx path> for the whole container. resolve_model()
# gives MODEL priority over the per-backend NVFP4 default:
#
#     if explicit: return explicit
#     if os.environ.get("MODEL"): return os.environ["MODEL"]      <-- hit
#     if MOE_BACKEND == ...cutedsl: return MODEL_NVFP4 or default
#
# so its "fi_nvfp4" row scored the MX weights through the cutedsl kernel's
# dequant path, not the NVFP4 checkpoint. results/gsm8k_fi_nvfp4.json records
# model=...hf-6e76323_orig, which is the mx original. The NVFP4 checkpoints
# that every fi_cutedsl throughput number is measured on have therefore never
# been accuracy-gated -- which is exactly what this eval exists to do.
#
# So: MODEL / MODEL_NVFP4 are explicitly UNSET inside the container and every
# cell passes --model, which is the one input resolve_model ranks above the
# environment. Each result JSON records the path it actually loaded, and the
# summary prints it and hard-flags any fi_cutedsl row that did not run an
# nvfp4 checkpoint.
#
# Usage:
#   cd <repo>/vllm_e2e && sbatch job_gsm8k_flash_pro.sh
#   (submit via scratch_runbook_submit_gsm8k.sh, which sets the four paths)

ROOT=${ROOT:-/lustre/fsw/coreai_libraries_cudnn/mhoqueanik}
W=$ROOT/moe_ep_benchmark/vllm_e2e
IMG=${IMG:-$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh}
NQ=${NQ:-200}
MIN_ACC=${MIN_ACC:-0.93}
MOUNTS="$ROOT:$ROOT,/lustre/share:/lustre/share:ro"
[[ -n "${EXTRA_MOUNTS:-}" ]] && MOUNTS="$MOUNTS,$EXTRA_MOUNTS"

FWD=""
for v in MODEL_MX_FLASH MODEL_NVFP4_FLASH MODEL_MX_PRO MODEL_NVFP4_PRO; do
    [[ -n "${!v:-}" ]] || { echo "$v is unset -- submit via scratch_runbook_submit_gsm8k.sh"; exit 2; }
    [[ -d "${!v}" ]]   || { echo "$v=${!v} is not a directory"; exit 2; }
    FWD+="export $v='${!v}'; "
done

srun --ntasks=1 \
  --container-image="$IMG" \
  --container-name=fivllm_gsm8k \
  --container-mounts="$MOUNTS" \
  --container-workdir="$W" \
  bash -lc "
set -uo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=$ROOT/.cache/huggingface
export PIP_CACHE_DIR=$ROOT/.cache/pip
export FLASHINFER_WORKSPACE_BASE=$ROOT/.cache/flashinfer-root-ws
export FI_MOE_EP_SKIP_VERSION_CHECK=1
$FWD
# Non-negotiable: either of these would silently override --model's intent for
# anyone who later edits a cell to drop --model.
unset MODEL MODEL_NVFP4

echo '=== node ==='; hostname; nvidia-smi -L
echo '=== harness ==='; git -C $ROOT/moe_ep_benchmark log --oneline -1
echo '=== flashinfer ==='; git -C $ROOT/flashinfer-2/flashinfer-moe_ep log --oneline -1
source venv0251/bin/activate || exit 1
bash patch_0251/apply.sh || exit 1

DG=flashinfer_moe_ep_mega_deep_gemm
CUTEDSL=flashinfer_moe_ep_mega_cutedsl

# gcell <label> <backend-short> <tp> <model-path> <out-stem> [knob-cache]
# The knob cache is per (model, EP size) -- Flash EP4 and Pro EP8 tuned
# different geometries -- so it is a parameter, not a constant.
gcell() {
    local label=\$1 short=\$2 tp=\$3 model=\$4 stem=\$5 knob=\${6:-}
    local be=deep_gemm_mega_moe cache=''
    [[ \$short == fi_dg ]] && be=\$DG
    if [[ \$short == fi_cutedsl ]]; then
        be=\$CUTEDSL
        [[ -n \$knob ]] && cache=FLASHINFER_MOE_EP_KNOB_CACHE=$W/results/\$knob
    fi
    local log=results/gsm8k_\${label}.log
    echo; echo \"===== \$label : backend=\$short tp=\$tp model=\$(basename \$model) =====\"
    env MOE_BACKEND=\$be \$cache TP=\$tp \
        python eval_gsm8k.py --tag \$label --tp \$tp --model \"\$model\" \
        --num-questions $NQ --min-acc $MIN_ACC \
        --out results/\${stem}.json > \$log 2>&1
    local rc=\$?
    echo \"rc=\$rc  (full log: \$log)\"
    grep -E '^\\[eval_gsm8k\\]' \$log | tail -4 | sed 's/^/    /'
    if [[ \$rc -ne 0 ]]; then
        echo '  -- failure tail --'
        grep -viE 'cudaDeviceReset|libcudart_stub|ProcessGroupNCCL' \$log | tail -10 | sed 's/^/    /'
    fi
}

# The Flash cells run TP4 even on an 8-GPU node: this is an accuracy gate, not
# a throughput measurement, and TP4 is what produced the recorded 0.960/0.960/
# 0.970. That is why an EP4 knob cache (knob_cache_dsv4_8k.json) ships on an
# otherwise 8-GPU-only branch.
echo; echo '########## DSV4-FLASH  (EP4/TP4)'
gcell flash_native     native     4 \"\$MODEL_MX_FLASH\"    gsm8k2_flash_native
gcell flash_fi_dg      fi_dg      4 \"\$MODEL_MX_FLASH\"    gsm8k2_flash_fi_dg
gcell flash_fi_cutedsl fi_cutedsl 4 \"\$MODEL_NVFP4_FLASH\" gsm8k2_flash_fi_cutedsl knob_cache_dsv4_8k.json

echo; echo '########## DSV4-PRO  (EP8/TP8)'
gcell pro_native       native     8 \"\$MODEL_MX_PRO\"      gsm8k2_pro_native
gcell pro_fi_dg        fi_dg      8 \"\$MODEL_MX_PRO\"      gsm8k2_pro_fi_dg
gcell pro_fi_cutedsl   fi_cutedsl 8 \"\$MODEL_NVFP4_PRO\"   gsm8k2_pro_fi_cutedsl knob_cache_pro_ep8.json

echo; echo '########## SUMMARY'
python - <<'PY'
import json, os

rows = [
    ('Flash', 'native',     'gsm8k2_flash_native'),
    ('Flash', 'fi_dg',      'gsm8k2_flash_fi_dg'),
    ('Flash', 'fi_cutedsl', 'gsm8k2_flash_fi_cutedsl'),
    ('Pro',   'native',     'gsm8k2_pro_native'),
    ('Pro',   'fi_dg',      'gsm8k2_pro_fi_dg'),
    ('Pro',   'fi_cutedsl', 'gsm8k2_pro_fi_cutedsl'),
]
print('%-7s %-12s %9s %9s  %s' % ('model', 'backend', 'accuracy', 'correct', 'checkpoint actually loaded'))
acc = {}
problems = []
for mdl, be, stem in rows:
    p = 'results/%s.json' % stem
    if not os.path.exists(p):
        print('%-7s %-12s %9s' % (mdl, be, 'MISSING'))
        problems.append('%s/%s did not produce a result' % (mdl, be))
        continue
    d = json.load(open(p))
    a = d['accuracy']
    ckpt = os.path.basename(d['model'].rstrip('/'))
    acc[(mdl, be)] = a
    print('%-7s %-12s %9.4f %9s  %s' % (mdl, be, a, '%d/%d' % (d['correct'], d['num_questions']), ckpt))
    # the whole point of the gate: cutedsl must have run an nvfp4 checkpoint
    if be == 'fi_cutedsl' and 'nvfp4' not in ckpt.lower():
        problems.append('%s/fi_cutedsl loaded %s -- NOT an nvfp4 checkpoint, gate did not test the cast' % (mdl, ckpt))

print()
print('cross-checkpoint spread (fi_cutedsl on the NVFP4 cast vs native on mx):')
for mdl in ('Flash', 'Pro'):
    if (mdl, 'native') in acc and (mdl, 'fi_cutedsl') in acc:
        n, c = acc[(mdl, 'native')], acc[(mdl, 'fi_cutedsl')]
        print('  %-6s native %.4f  vs  fi_cutedsl %.4f   delta %+.4f' % (mdl, n, c, c - n))
        if abs(c - n) > 0.03:
            problems.append('%s: fi_cutedsl differs from native by %+.4f (>0.03)' % (mdl, c - n))

print()
if problems:
    print('PROBLEMS:')
    for p in problems:
        print('  !! ' + p)
else:
    print('all rows present, every fi_cutedsl row ran an nvfp4 checkpoint,')
    print('and both backends agree within 0.03 on both models.')
PY
echo; echo \"=== job exit rc=\$? ===\"
"
