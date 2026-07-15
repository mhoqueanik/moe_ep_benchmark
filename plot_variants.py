"""Grouped bar plot: MegaMoE combine-leg variants vs deep_gemm per tok/rank.

Reads BENCH_CSV-format sweep CSVs (the ``bench_moe_ep_mega.py`` rows), groups
by ``tokens_per_rank``, and draws one subplot per token count with bars
[deep_gemm, nvfp4 bf16, +ikr, +combine_nvfp4, +combine_mxfp8].  Per-group
y-scale; value labels on every bar; the group's best speedup vs deep_gemm is
annotated in italics.

Usage:
    python plot_variants.py --csv results/sweep_20260715_*_fi_mega.csv \
        --tokens 1024 2048 4096 8192 -o results/megamoe_variants_latency.png
"""

from __future__ import annotations

import argparse
import csv
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [
    ("deep_gemm_mega", "deep_gemm", "#9e9e9e"),
    ("nvfp4_cutedsl", "nvfp4 bf16", "#2f7fe0"),
    ("nvfp4_cutedsl+ikr", "+ikr", "#2bb681"),
    ("nvfp4_cutedsl+combine_nvfp4", "+combine_nvfp4", "#f2a71b"),
    ("nvfp4_cutedsl+combine_mxfp8", "+combine_mxfp8", "#1a7d1a"),
]


def load(csv_globs):
    """{(compute_kernel, tokens_per_rank): p50_us} — last write wins."""
    data = {}
    for pattern in csv_globs:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for row in csv.DictReader(f):
                    try:
                        tok = int(row["tokens_per_rank"])
                        p50 = float(row["e2e_us_p50"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    data[(row["compute_kernel"], tok)] = p50
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", nargs="+", required=True, help="CSV globs")
    ap.add_argument("--tokens", nargs="+", type=int, required=True)
    ap.add_argument("-o", "--out", default="megamoe_variants_latency.png")
    ap.add_argument(
        "--timing-label",
        default="e2e_pipelined p50",
        help="Timing-mode text for the title",
    )
    ap.add_argument(
        "--subtitle",
        default=(
            "4x GB200 (EP=4), 7168 hidden / 2048 inter / 256 experts / top-8, "
            "corrected K-major weight layout (2026-07-15) · per-group "
            "y-scale · italic = best speedup vs deep_gemm in the group"
        ),
    )
    args = ap.parse_args()

    data = load(args.csv)
    groups = args.tokens

    fig, axes = plt.subplots(
        1, len(groups), figsize=(5.8 * len(groups), 8.6), dpi=100
    )
    if len(groups) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fbfbf8")

    for ax, tok in zip(axes, groups):
        ax.set_facecolor("#fbfbf8")
        vals = []
        for key, _label, _color in SERIES:
            v = data.get((key, tok))
            if v is None:
                raise SystemExit(f"missing CSV row for {key!r} @ {tok} tok/rank")
            vals.append(v)

        dg = vals[0]
        best_idx = min(range(1, len(vals)), key=lambda i: vals[i])
        xs = range(len(vals))
        for i, (v, (_k, _l, color)) in enumerate(zip(vals, SERIES)):
            ax.bar(i, v, width=0.62, color=color, zorder=3)
            label = f"{v:,.0f}"
            ymax = max(vals)
            ax.text(
                i, v + ymax * 0.02, label, ha="center", va="bottom", fontsize=13
            )
            if i == best_idx:
                ax.text(
                    i,
                    v + ymax * 0.085,
                    f"{dg / v:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    style="italic",
                )

        ax.set_title(f"{tok} tok/rank", fontsize=16, pad=14)
        ax.set_xticks([])
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="y", color="#d8d8d2", linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=12, length=0)

    axes[0].set_ylabel("latency p50 (µs)", fontsize=14)

    fig.suptitle(
        "CuTeDSL MegaMoE combine-leg variants vs deep_gemm — "
        + args.timing_label,
        x=0.035,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(0.035, 0.93, args.subtitle, ha="left", fontsize=12.5, color="#444")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for _k, _l, color in SERIES
    ]
    fig.legend(
        handles,
        [label for _k, label, _c in SERIES],
        loc="upper right",
        ncol=len(SERIES),
        frameon=False,
        fontsize=12.5,
        columnspacing=1.1,
        handlelength=1.2,
        bbox_to_anchor=(0.998, 1.005),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"Saved plot -> {args.out}")


if __name__ == "__main__":
    main()
