"""Steady-state window analysis of the gap-attribution nsys sqlite exports.

For each trace: find the MoE mega-kernel instances, take the window covering
the last 40% of them (timed-rounds steady state, past load/warmup/capture),
and within it compute per-device wall vs GPU-busy (union of kernel
intervals), the idle gap, and the top kernels. Splits the fi-vs-native
residual into "GPU does more work" vs "GPU sits idle waiting for the host".
"""

from __future__ import annotations

import glob
import sqlite3
import sys

MEGA_PAT = ("%mega_moe%", "%megamoe%", "%MegaMoE%")


def analyze(path: str) -> None:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    # kernel rows: CUPTI_ACTIVITY_KIND_KERNEL(start, end, deviceId, shortName->StringIds)
    kq = """SELECT k.start AS s, k.end AS e, k.deviceId AS dev, si.value AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds si ON k.shortName = si.id"""
    rows = db.execute(kq).fetchall()
    if not rows:
        print(f"{path}: no kernel rows")
        return
    mega = [r for r in rows if "mega" in r["name"].lower()]
    if not mega:
        print(f"{path}: no mega kernels found")
        return
    mega.sort(key=lambda r: r["s"])
    w0 = mega[int(len(mega) * 0.6)]["s"]
    w1 = mega[-1]["e"]
    wall = (w1 - w0) / 1e9
    win = [r for r in rows if r["s"] >= w0 and r["e"] <= w1]

    devs = sorted({r["dev"] for r in win})
    busy_by_dev = {}
    for d in devs:
        iv = sorted((r["s"], r["e"]) for r in win if r["dev"] == d)
        busy = 0
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s > ce:
                busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        busy_by_dev[d] = busy / 1e9
    avg_busy = sum(busy_by_dev.values()) / len(devs)

    agg: dict[str, list[float]] = {}
    for r in win:
        a = agg.setdefault(r["name"], [0.0, 0])
        a[0] += (r["e"] - r["s"]) / 1e9
        a[1] += 1
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:6]
    n_mega_win = sum(1 for r in win if "mega" in r["name"].lower())

    tag = path.rsplit("/", 1)[-1].replace(".sqlite", "")
    print(f"\n== {tag} ==")
    print(
        f"  window {wall:6.2f}s | GPU busy avg {avg_busy:6.2f}s "
        f"({100 * avg_busy / wall:5.1f}%) | idle {wall - avg_busy:6.2f}s "
        f"| per-dev busy {[f'{b:.2f}' for b in busy_by_dev.values()]}"
    )
    print(f"  mega launches in window: {n_mega_win}")
    for name, (t, n) in top:
        print(f"    {t:6.2f}s n={n:>7} avg={1e6 * t / n:8.1f}us  {name[:58]}")


if __name__ == "__main__":
    pats = sys.argv[1:] or ["results/nsys_gap_*.sqlite"]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            analyze(f)
