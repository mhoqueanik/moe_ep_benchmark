#!/usr/bin/env python3
"""Turn an iket_breakdown job log (+ its CSVs) into a markdown breakdown.

Usage:
    python make_report.py --log slurm_XXXX.log --out RESULTS.md

Parses:
  * "== cell: backend=... timing=... pt=... tokens/rank=N" headers
  * BENCH_CSV rows (latency cells)
  * PT_INFO / PT_CSV rows (clock64 phase breakdown, one block per rank)
and emits, per tokens/rank:
  1. dg vs cutedsl latency (e2e + kernel timing modes) with the ratio
  2. the cutedsl in-kernel phase table (mean over ranks, plus worst rank),
     grouped by warp role, with %-of-kernel and approximate us.
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics

CELL_RE = re.compile(
    r"== cell: backend=(\S+) timing=(\S+) pt=(\d) tokens/rank=(\d+)"
)

ROLE_ORDER = [
    ("dispatch warps (w8-11)", ["dispatch_prep", "dispatch_barrier",
                                "dispatch_pull", "dispatch_total"]),
    ("sched warp (w7)", ["sched_pre_init_wait", "sched_loop_total",
                         "sched_publish"]),
    ("TMA-A warp (w5)", ["tma_a_consume_work", "tma_a_loop_total"]),
    ("TMA-B warp (w6)", ["tma_b_consume_work", "tma_b_fc1_wait",
                         "tma_b_fc2_wait", "tma_b_loop_total"]),
    ("MMA warp (w4)", ["mma_consume_work", "mma_fc1", "mma_fc2",
                       "mma_loop_total"]),
    ("epilogue warps (w0-3)", ["epi_consume_work", "epi_fc1_wait", "epi_fc1",
                               "epi_fc2_wait", "epi_fc2", "epi_drain_barrier",
                               "epi_flag", "epi_loop_total"]),
    ("kernel tail", ["tail_rendezvous", "tail_nvlink_drain",
                     "tail_shared_reset", "tail_nvlink_publish",
                     "tail_local_reset"]),
    ("whole kernel", ["kernel_total"]),
]

NOTES = {
    "sched_pre_init_wait": "cross-rank dispatch arrival, seen by compute",
    "sched_publish": "backpressure: consumers not draining tiles",
    "tma_b_fc1_wait": "dispatch->fc1 token arrival spin",
    "tma_b_fc2_wait": "fc1->fc2 handoff spin",
    "epi_fc1": "whole fc1 epi call (work = epi_fc1 - epi_fc1_wait)",
    "epi_fc2": "whole fc2 call incl. combine STG (work = epi_fc2 - epi_fc2_wait)",
    "*_consume_work": "idle waiting for scheduled work",
}


def parse_log(path):
    cells = []  # (backend, timing, pt, tokens)
    bench_rows = []  # (cellkey, dict from BENCH_CSV)
    pt_rows = collections.defaultdict(list)  # cellkey -> [(rank, slot, ...)]
    pt_info = collections.defaultdict(dict)  # cellkey -> {rank: (nctas, ktmean, us)}
    cur = None
    for line in open(path, errors="replace"):
        m = CELL_RE.search(line)
        if m:
            cur = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
            cells.append(cur)
            continue
        if line.startswith("BENCH_CSV,") and cur:
            f = line.strip().split(",")[1:]
            # header: path,algo,comm_backend,compute_kernel,quant_timed,
            # weight_dtype,input_dtype,act_compute_dtype,tokens_per_rank,gpus,
            # num_experts,top_k,hidden,inter,e2e_us_p50,e2e_us_min,e2e_us_max,
            # tok_s,acc_loss_pct
            bench_rows.append((cur, {
                "compute_kernel": f[3], "tokens": int(f[8]),
                "p50": float(f[14]), "min": float(f[15]), "max": float(f[16]),
            }))
        elif line.startswith("PT_INFO,") and cur:
            d = dict(kv.split("=", 1) for kv in line.strip().split(",")[1:]
                     if "=" in kv)
            if "rank" in d and "active_ctas" in d:
                pt_info[cur][int(d["rank"])] = (
                    int(d["active_ctas"]),
                    float(d.get("kernel_total_mean_cycles", "nan")),
                    float(d.get("measured_p50_us", "nan")),
                )
        elif line.startswith("PT_CSV,") and cur:
            f = line.strip().split(",")
            pt_rows[cur].append((int(f[1]), f[2], float(f[3]), float(f[4]),
                                 float(f[5]), float(f[6]), float(f[7])))
    return cells, bench_rows, pt_rows, pt_info


def fmt_us(v):
    return f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobid", default="?")
    args = ap.parse_args()

    cells, bench_rows, pt_rows, pt_info = parse_log(args.log)
    tokens_list = sorted({c[3] for c in cells})

    def lat(backend, timing, pt, tokens):
        for cell, row in bench_rows:
            if cell == (backend, timing, pt, tokens):
                return row
        return None

    out = []
    out.append("# Mega-path low-tokens/rank breakdown (nvfp4_cutedsl vs "
               "deep_gemm_mega)\n")
    out.append(f"SLURM job {args.jobid}; 1x8 B200, EP8 (DP8/TP1), 256 experts "
               "top-8, hidden 7168, inter 2048; DSL 4.5.2. clock64 phase "
               "timing (`+pt` rows) perturbs latency and is never "
               "perf-quotable; latency comparisons use the uninstrumented "
               "cells.\n")

    # 1. latency comparison
    out.append("## 1. Latency: dg vs cutedsl\n")
    out.append("| tok/rank | dg e2e us | cutedsl e2e us | e2e ratio | "
               "dg kernel us | cutedsl kernel us | kernel ratio |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|")
    for t in tokens_list:
        cols = []
        de = lat("deep_gemm_mega", "e2e", 0, t)
        ce = lat("nvfp4_cutedsl", "e2e", 0, t)
        dk = lat("deep_gemm_mega", "kernel", 0, t)
        ck = lat("nvfp4_cutedsl", "kernel", 0, t)
        for a, b in ((de, ce), (dk, ck)):
            if a and b:
                cols += [fmt_us(a["p50"]), fmt_us(b["p50"]),
                         f"{b['p50'] / a['p50']:.2f}x"]
            else:
                cols += ["-", "-", "-"]
        out.append(f"| {t} | " + " | ".join(cols) + " |")
    out.append("")

    # 2. phase breakdown per tokens
    out.append("## 2. cutedsl in-kernel phase breakdown (clock64, "
               "MEGA_TIMING=kernel)\n")
    out.append("Cycles are per-CTA means over active CTAs; `mean` averages "
               "the 8 ranks, `worst rank` is the rank with the largest "
               "value (skew signal). `% kern` is vs that rank-mean "
               "kernel_total. Approx us scales cycles by the measured p50 "
               "of the instrumented run.\n")
    for t in tokens_list:
        key = ("nvfp4_cutedsl", "kernel", 1, t)
        rows = pt_rows.get(key)
        if not rows:
            continue
        by_slot = collections.defaultdict(dict)
        for rank, slot, mean_c, max_c, sum_c, pct, us in rows:
            by_slot[slot][rank] = (mean_c, pct, us)
        out.append(f"### tokens/rank = {t}\n")
        info = pt_info.get(key, {})
        if info:
            us_meas = statistics.median(v[2] for v in info.values())
            out.append(f"measured p50 (instrumented): {us_meas:.1f} us; "
                       f"active CTAs/rank: "
                       f"{sorted({v[0] for v in info.values()})}\n")
        out.append("| phase | mean cyc | mean % kern | mean ~us | "
                   "worst-rank ~us | note |")
        out.append("|---|---:|---:|---:|---:|---|")
        for role, slots in ROLE_ORDER:
            out.append(f"| **{role}** | | | | | |")
            for s in slots:
                if s not in by_slot:
                    continue
                per_rank = by_slot[s]
                mean_c = statistics.mean(v[0] for v in per_rank.values())
                mean_pct = statistics.mean(v[1] for v in per_rank.values())
                mean_us = statistics.mean(v[2] for v in per_rank.values())
                worst_us = max(v[2] for v in per_rank.values())
                note = NOTES.get(s, NOTES.get("*_consume_work", "")
                                 if s.endswith("consume_work") else "")
                out.append(f"| {s} | {mean_c:.0f} | {mean_pct:.1f} | "
                           f"{mean_us:.1f} | {worst_us:.1f} | {note} |")
        out.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
