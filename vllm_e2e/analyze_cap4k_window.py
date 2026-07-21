"""Time-windowed analysis for graphs-mode traces: the timed rounds sit at the
trace end; capture warmup (hundreds of sizes) pollutes launch-count windows.
Window = last `round_seconds` (from the bench JSON) before the last kernel."""
import json, sqlite3, statistics, sys

def analyze(db_path, json_path, tag):
    rounds = json.load(open(json_path))["rounds"]
    round_s = statistics.median(r["elapsed_s"] for r in rounds if not r["warmup"])
    db = sqlite3.connect(db_path); db.row_factory = sqlite3.Row
    rows = db.execute("""SELECT k.start s, k.end e, k.deviceId dev, si.value name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds si ON k.shortName=si.id""").fetchall()
    t_end = max(r["e"] for r in rows)
    w0 = t_end - round_s * 1e9
    win = [r for r in rows if r["s"] >= w0]
    wall = (t_end - w0) / 1e9
    devs = sorted({r["dev"] for r in win})
    busy = {}
    for d in devs:
        iv = sorted((r["s"], r["e"]) for r in win if r["dev"] == d)
        tot, cs, ce = 0, *iv[0]
        for s, e in iv[1:]:
            if s > ce: tot += ce - cs; cs, ce = s, e
            else: ce = max(ce, e)
        busy[d] = (tot + ce - cs) / 1e9
    avg_busy = sum(busy.values()) / len(devs)
    agg = {}
    for r in win:
        a = agg.setdefault(r["name"], [0.0, 0]); a[0] += (r["e"] - r["s"]) / 1e9; a[1] += 1
    print(f"\n== {tag} (window={wall:.2f}s = one timed round) ==")
    print(f"  GPU busy avg {avg_busy:.2f}s ({100*avg_busy/wall:.1f}%) | idle {wall-avg_busy:.2f}s | per-dev {[f'{b:.2f}' for b in busy.values()]}")
    for name, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:7]:
        print(f"    {t:5.2f}s n={n:>6} avg={1e6*t/n:8.1f}us  {name[:58]}")

analyze(sys.argv[1], sys.argv[2], sys.argv[3])
