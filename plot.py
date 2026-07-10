#!/usr/bin/env python3
"""Plot MoE-EP sweep results: one line per backend, tokens/rank vs e2e p50 latency.

Picks the latest sweep CSV for each section (vllm_split, vllm_mega, fi_mega) from
the results directory, then draws an "-o-" line plot with one series per backend
(a section may contribute several backends, e.g. fi_mega has three).

    x axis : tokens_per_rank (log2)
    y axis : e2e_us_p50 (microseconds)

Usage:
    python moe_ep_benchmark/plot.py                 # latest sweep per section
    python moe_ep_benchmark/plot.py -o my_plot.png  # custom output path
    python moe_ep_benchmark/plot.py --csv a.csv b.csv   # explicit files
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless; write PNG without a display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SECTIONS = ("vllm_split", "vllm_mega", "fi_mega")


def latest_sweep_per_section(results_dir: str) -> list[str]:
    """Return the newest ``sweep_<stamp>_<section>.csv`` for each known section."""
    picked: list[str] = []
    for sec in SECTIONS:
        pat = re.compile(rf"^sweep_\d{{8}}_\d{{6}}_{re.escape(sec)}\.csv$")
        cands = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if pat.match(f)
        ]
        if cands:
            picked.append(max(cands, key=os.path.getmtime))
    return picked


def load_series(csv_paths: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Map ``<path>:<compute_kernel>`` -> sorted [(tokens_per_rank, e2e_us_p50)]."""
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    x = int(row["tokens_per_rank"])
                    y = float(row["e2e_us_p50"])
                except (KeyError, ValueError):
                    continue
                label = f"{row['path']}:{row['compute_kernel']}"
                series[label].append((x, y))
    # De-dup on x (keep the last-written point) and sort by tokens/rank.
    out: dict[str, list[tuple[int, float]]] = {}
    for label, pts in series.items():
        by_x = dict(pts)
        out[label] = sorted(by_x.items())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        nargs="+",
        default=None,
        help="explicit sweep CSVs (default: latest sweep per section in results/)",
    )
    p.add_argument(
        "-o",
        "--out",
        default=os.path.join(RESULTS_DIR, "sweep_e2e_p50.png"),
        help="output PNG path",
    )
    a = p.parse_args()

    csv_paths = a.csv or latest_sweep_per_section(RESULTS_DIR)
    if not csv_paths:
        raise SystemExit(f"no sweep CSVs found in {RESULTS_DIR}")

    print("Using CSVs:")
    for c in csv_paths:
        print(f"  {c}")

    series = load_series(csv_paths)
    if not series:
        raise SystemExit("no plottable rows (missing tokens_per_rank/e2e_us_p50)")

    fig, ax = plt.subplots(figsize=(9, 6))
    for label in sorted(series):
        pts = series[label]
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens / rank")
    ax.set_ylabel("e2e latency p50 (us)")
    ax.set_title("MoE-EP sweep: e2e p50 latency vs tokens/rank (per backend)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="path:backend", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"\nSaved plot -> {a.out}")
    print(f"Backends plotted: {len(series)}")


if __name__ == "__main__":
    main()
