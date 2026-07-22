"""Turn model_shapes sweep CSV(s) into per-shape markdown tables (RESULTS.md).

Usage:
    python make_tables.py results/model_shapes_*.csv [-o RESULTS.md]

Rows are grouped by geometry (hidden, inter, num_experts, top_k) and mapped
back to model names via shapes.tsv. Each shape gets a table:

    tokens/rank | fi_dg | fi_fp4 (vs dg) | fi_ikr | fi_combine_fp8 | fi_combine_fp4

Cells are e2e p50 microseconds (MEGA_TIMING region); fp4-family cells carry a
speedup-vs-dg ratio. Later CSVs win on duplicate cells (re-runs supersede).
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# compute_kernel column -> variant name
KERNEL_TO_VARIANT = {
    "deep_gemm_mega": "fi_dg",
    "nvfp4_cutedsl": "fi_fp4",
    "nvfp4_cutedsl+ikr": "fi_ikr",
    "nvfp4_cutedsl+combine_mxfp8": "fi_combine_fp8",
    "nvfp4_cutedsl+combine_nvfp4": "fi_combine_fp4",
}
VARIANTS = ["fi_dg", "fi_fp4", "fi_ikr", "fi_combine_fp8", "fi_combine_fp4"]


def load_shape_names():
    names = {}
    with open(HERE / "shapes.tsv") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name, hidden, inter, experts, topk = parts[:5]
            names[(int(hidden), int(inter), int(experts), int(topk))] = name
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("-o", "--out", default=str(HERE / "RESULTS.md"))
    args = ap.parse_args()

    shape_names = load_shape_names()
    # {geometry: {tokens: {variant: row}}}
    cells = {}
    for path in args.csvs:
        with open(path) as f:
            for row in csv.DictReader(f):
                variant = KERNEL_TO_VARIANT.get(row["compute_kernel"])
                if variant is None:
                    continue
                geom = (
                    int(row["hidden"]),
                    int(row["inter"]),
                    int(row["num_experts"]),
                    int(row["top_k"]),
                )
                tokens = int(row["tokens_per_rank"])
                cells.setdefault(geom, {}).setdefault(tokens, {})[variant] = row

    lines = [
        "# Model-shape microbenchmark results",
        "",
        "e2e_pipelined p50 latency in microseconds per variant; fp4-family "
        "cells show speedup vs `fi_dg` at the same point. `tok/s` and "
        "accuracy-loss columns are in the raw CSVs.",
        "",
    ]
    for geom in sorted(cells, key=lambda g: (shape_names.get(g, ""), g)):
        hidden, inter, experts, topk = geom
        name = shape_names.get(geom, "unknown_shape")
        lines.append(
            f"## {name} — hidden {hidden}, inter {inter}, "
            f"{experts} experts, top-{topk}"
        )
        lines.append("")
        lines.append(
            "| tok/rank | fi_dg | fi_fp4 | fi_ikr | fi_combine_fp8 | fi_combine_fp4 |"
        )
        lines.append("|---|---|---|---|---|---|")
        for tokens in sorted(cells[geom]):
            byvar = cells[geom][tokens]
            dg = byvar.get("fi_dg")
            dg_us = float(dg["e2e_us_p50"]) if dg else None
            out = [str(tokens)]
            for v in VARIANTS:
                row = byvar.get(v)
                if row is None:
                    out.append("—")
                    continue
                us = float(row["e2e_us_p50"])
                if v == "fi_dg" or not dg_us:
                    out.append(f"{us:.1f}")
                else:
                    out.append(f"{us:.1f} ({dg_us / us:.2f}x)")
            lines.append("| " + " | ".join(out) + " |")
        lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
