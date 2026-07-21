"""Per-layer work/wait/skew decomposition of mega-kernel launches.

The mega kernel is a collective: each rank's launch spins in device barriers
until peers arrive, so a rank's measured duration = true work + absorbed
peer-wait. For the k-th mega launch matched across all ranks (same step and
layer — launches are strictly ordered per rank):

  work_k  = min over ranks of duration          (last-arriving rank waits least)
  wait_kr = duration_r - work_k                 (per-rank absorbed peer wait)
  skew_k  = max over ranks of start - min start (arrival spread)

Aggregated per layer (launch index mod 43) over the steady-state tail this
splits "fi kernels are slower in-engine" into work (kernel-side) vs wait
(host-arrival-side), per layer and per backend.
"""

from __future__ import annotations

import sqlite3
import statistics
import sys

N_LAYERS = 43


def load_mega(path: str):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """SELECT k.start s, k.end e, k.deviceId dev, si.value name
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds si ON k.shortName = si.id"""
    ).fetchall()
    per_dev: dict[int, list] = {}
    for r in rows:
        n = r["name"].lower()
        if "megamoe" in n or "mega_moe" in n:
            per_dev.setdefault(r["dev"], []).append((r["s"], r["e"]))
    for d in per_dev:
        per_dev[d].sort()
    return per_dev


def analyze(path: str):
    per_dev = load_mega(path)
    devs = sorted(per_dev)
    if len(devs) < 2:
        print(f"{path}: <2 devices with mega launches"); return None
    n = min(len(per_dev[d]) for d in devs)
    lo = int(n * 0.4)  # steady-state tail
    work_by_layer: list[list[float]] = [[] for _ in range(N_LAYERS)]
    wait_by_layer: list[list[float]] = [[] for _ in range(N_LAYERS)]
    skew_by_layer: list[list[float]] = [[] for _ in range(N_LAYERS)]
    for k in range(lo, n):
        group = [per_dev[d][k] for d in devs]
        durs = [(e - s) / 1e3 for s, e in group]
        starts = [s / 1e3 for s, _ in group]
        work = min(durs)
        layer = k % N_LAYERS
        work_by_layer[layer].append(work)
        skew_by_layer[layer].append(max(starts) - min(starts))
        for du in durs:
            wait_by_layer[layer].append(du - work)
    med = lambda xs: statistics.median(xs) if xs else 0.0
    layers = [
        dict(
            layer=i,
            work=med(work_by_layer[i]),
            wait=med(wait_by_layer[i]),
            skew=med(skew_by_layer[i]),
        )
        for i in range(N_LAYERS)
        if work_by_layer[i]
    ]
    return layers


def summarize(tag: str, layers) -> None:
    w = [x["work"] for x in layers]
    wt = [x["wait"] for x in layers]
    sk = [x["skew"] for x in layers]
    print(f"\n== {tag} ==")
    print(
        f"  per-launch medians: WORK {statistics.median(w):7.1f}us "
        f"(min {min(w):.0f} max {max(w):.0f}) | absorbed WAIT "
        f"{statistics.median(wt):6.1f}us | arrival SKEW {statistics.median(sk):6.1f}us"
    )
    worst = sorted(layers, key=lambda x: -(x["work"] + x["wait"]))[:5]
    print("  worst layers (work+wait):")
    for x in worst:
        print(
            f"    layer~{x['layer']:>2}: work {x['work']:7.1f}  wait {x['wait']:6.1f}"
            f"  skew {x['skew']:6.1f}"
        )


if __name__ == "__main__":
    results = {}
    for path in sys.argv[1:]:
        layers = analyze(path)
        if layers:
            tag = path.rsplit("/", 1)[-1].replace(".sqlite", "")
            results[tag] = layers
            summarize(tag, layers)
    if len(results) >= 2:
        print("\n== cross-backend (median across layers) ==")
        for tag, layers in results.items():
            tot_work = sum(x["work"] for x in layers)
            tot_wait = sum(x["wait"] for x in layers)
            print(
                f"  {tag:>28}: sum-work {tot_work / 1e3:6.2f}ms/step "
                f"sum-wait {tot_wait / 1e3:6.2f}ms/step "
                f"({100 * tot_wait / max(tot_work + tot_wait, 1e-9):4.1f}% wait)"
            )
