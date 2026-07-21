#!/bin/bash
# fi_dg nondeterminism: batch-shape schedule diff across identical reruns.
# Runs smoke_infer twice per backend (fi_dg, native) with FI_MOE_EP_SHAPE_LOG
# capturing the per-MoE-call scheduled token counts, then diffs the shape
# sequences AND the generations. TP4 eager — the original repro conditions.
# Usage: orchestrate_shape_determinism.sh <jobid>
# Markers: logs/shape_det.{log,done}
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$W/results/shape_det_$STAMP
mkdir -p "$OUT"

run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

until [ "$(squeue -j "$JOBID" -h -o %t 2>/dev/null)" = "R" ]; do sleep 30; done
echo "node up: $(squeue -j "$JOBID" -h -o %N)"

rc=0
for backend in fi_dg native; do
  case $backend in
    fi_dg)  ENVS="FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega" ;;
    native) ENVS="FI_MOE_EP=0" ;;
  esac
  for r in a b; do
    echo "=== smoke $backend run $r ==="
    run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
      env $ENVS FI_MOE_EP_SHAPE_LOG=$OUT/shapes_${backend}_${r} \
      python smoke_infer.py --tag ${backend}_${r} \
        --out $OUT/smoke_${backend}_${r}.json" \
      > "$W/logs/shape_det_${STAMP}_${backend}_${r}.log" 2>&1 || rc=1
    tail -2 "$W/logs/shape_det_${STAMP}_${backend}_${r}.log" | head -1
  done
done

echo "=== schedule + output diffs ==="
python3 - "$OUT" <<'PY'
import glob, json, sys, os
out = sys.argv[1]
for backend in ("fi_dg", "native"):
    for rank_file_a in sorted(glob.glob(f"{out}/shapes_{backend}_a.rank*")):
        rank = rank_file_a.rsplit(".", 1)[1]
        rank_file_b = f"{out}/shapes_{backend}_b.{rank}"
        if not os.path.exists(rank_file_b):
            print(f"{backend} {rank}: run-b shape log MISSING")
            continue
        a = open(rank_file_a).read().split()
        b = open(rank_file_b).read().split()
        if a == b:
            print(f"{backend} {rank}: schedules IDENTICAL ({len(a)} MoE calls)")
        else:
            n = min(len(a), len(b))
            first = next((i for i in range(n) if a[i] != b[i]), n)
            print(f"{backend} {rank}: schedules DIFFER (len {len(a)} vs {len(b)}, "
                  f"first diff at call {first}: {a[first] if first < len(a) else '-'} "
                  f"vs {b[first] if first < len(b) else '-'})")
    try:
        ja = json.load(open(f"{out}/smoke_{backend}_a.json"))
        jb = json.load(open(f"{out}/smoke_{backend}_b.json"))
        ta = [o["token_ids"] for o in ja["records"]]
        tb = [o["token_ids"] for o in jb["records"]]
        exact = sum(x == y for x, y in zip(ta, tb))
        print(f"{backend}: generations exact {exact}/{len(ta)}")
    except Exception as exc:
        print(f"{backend}: output compare failed: {exc}")
PY

echo "=== shape_det exit: $rc ==="
echo "$rc" > "$W/logs/shape_det.done"
exit $rc
