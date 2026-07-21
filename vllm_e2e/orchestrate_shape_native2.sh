#!/bin/bash
# Round 2: native x2 with LINE-BUFFERED shape logs (round 1 logs were lost to
# a 64KB write buffer at worker exit). Native diverged 3/8 today — this diffs
# its per-MoE-call batch-shape schedule across the two runs.
set -uo pipefail
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
JOBID=$1
OUT=$W/results/shape_det2_$(date +%H%M%S)
mkdir -p "$OUT"
run() { JOBID=$JOBID bash "$W/in_container.sh" "$1"; }

rc=0
for r in a b; do
  echo "=== smoke native run $r (line-buffered logs) ==="
  run "source venv0251/bin/activate && bash patch_0251/apply.sh >/dev/null && \
    env FI_MOE_EP=0 FI_MOE_EP_SHAPE_LOG=$OUT/shapes_native_${r} \
    python smoke_infer.py --tag native2_${r} --out $OUT/smoke_native_${r}.json" \
    > "$W/logs/shape_det2_native_${r}.log" 2>&1 || rc=1
done

echo "=== round-2 diffs ==="
python3 - "$OUT" <<'PY'
import glob, json, sys, os
out = sys.argv[1]
for fa in sorted(glob.glob(f"{out}/shapes_native_a.rank*")):
    rank = fa.rsplit(".", 1)[1]
    fb = f"{out}/shapes_native_b.{rank}"
    a = open(fa).read().split()
    b = open(fb).read().split() if os.path.exists(fb) else []
    if not a and not b:
        print(f"native {rank}: logs EMPTY again")
    elif a == b:
        print(f"native {rank}: schedules IDENTICAL ({len(a)} MoE calls)")
    else:
        n = min(len(a), len(b))
        first = next((i for i in range(n) if a[i] != b[i]), n)
        print(f"native {rank}: schedules DIFFER (len {len(a)} vs {len(b)}, first diff "
              f"call {first}: {a[first] if first < len(a) else '-'} vs "
              f"{b[first] if first < len(b) else '-'})")
ja = json.load(open(f"{out}/smoke_native_a.json"))
jb = json.load(open(f"{out}/smoke_native_b.json"))
ta = [o["token_ids"] for o in ja["records"]]
tb = [o["token_ids"] for o in jb["records"]]
print(f"native round2: generations exact {sum(x==y for x,y in zip(ta,tb))}/{len(ta)}")
PY
echo "$rc" > "$W/logs/shape_det2.done"
exit $rc
