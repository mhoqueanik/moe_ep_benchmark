"""GPU-time attribution chart: 100%-stacked horizontal bars, one per backend.

Data: per-step GPU kernel self-time under the PRODUCTION configs (CUDA
graphs; prefill additionally max_cudagraph_capture_size=4096), from the
run-30 time-windowed nsys analysis (analyze_cap4k_window.py) plus the run-27
decode-graphs windows; see RUNS.md runs 27-30. fi prefill reduce/staging and
both "other" bands use the shared-model-code measurement (identical kernels
both backends; nsys under-expands DSL kernels inside captured graphs).
Regenerate: python make_time_attribution_chart.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
# Reference categorical palette, slots 1-5 in fixed documented order
# (adjacent-pair validated for stacked segments, light mode).
COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]

SEGMENTS = [
    "MoE mega kernel",
    "MoE topk-reduce + staging",
    "allreduce / EP comm",
    "attention",
    "other (norm, rope/kv, dense, misc)",
]

# ms of GPU kernel self-time per step, per GPU (identical workload per panel)
PANELS = [
    (
        "prefill — 8192-token chunks, sparse-capture graphs   (run 34 e2e: native 45.7k / fi_dg 47.5k (1.04x) / fi_nvfp4 53.9k (1.18x) tok/s)",
        {
            "native": [74.84, 5.98, 16.88, 23.04, 58.31],
            "fi_dg": [74.50, 0.65, 16.93, 22.65, 58.12],
            "fi_nvfp4": [51.21, 3.45, 16.52, 22.99, 58.16],
        },
    ),
    (
        "decode — 1024-seq steps, full-capture graphs   (run 37/38 e2e: native 21.0k / fi_dg 21.5k (1.02x) / fi_nvfp4 22.6k (1.07x) out tok/s)",
        {
            "native": [13.26, 0.79, 3.43, 5.49, 12.39],
            "fi_dg": [13.21, 0.13, 3.36, 5.42, 12.41],
            "fi_nvfp4": [11.08, 1.57, 3.57, 5.60, 12.61],
        },
    ),
]

# text color per segment fill (dark hues get white labels)
LABEL_INK = ["#ffffff", "#ffffff", INK, INK, INK]

fig, axes = plt.subplots(2, 1, figsize=(13, 9.2), dpi=200)
fig.patch.set_facecolor(SURFACE)

for ax, (panel_title, data) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    backends = list(data.keys())
    totals = {b: sum(v) for b, v in data.items()}
    ypos = {b: i for i, b in enumerate(reversed(backends))}
    BAR_H = 0.58

    for b in backends:
        y = ypos[b]
        left = 0.0
        tot = totals[b]
        for si, ms in enumerate(data[b]):
            frac = ms / tot * 100
            ax.barh(
                y, frac, left=left, height=BAR_H,
                color=COLORS[si], edgecolor=SURFACE, linewidth=2, zorder=3,
            )
            if frac >= 4.0:
                ax.text(
                    left + frac / 2, y, f"{frac:.0f}%\n({ms:.1f}ms)",
                    ha="center", va="center",
                    fontsize=8.5 if frac >= 5.5 else 7.2,
                    color=LABEL_INK[si], zorder=4, linespacing=1.1,
                )
            else:
                # Too thin for an in-segment label (e.g. the topk-reduce +
                # staging band): annotate just above the bar in the segment
                # color so small bands stay readable.
                ax.text(
                    left + frac / 2, y + BAR_H / 2 + 0.05,
                    f"{frac:.1f}% ({ms:.1f}ms)",
                    ha="center", va="bottom", fontsize=6.8,
                    color=COLORS[si], zorder=4,
                )
            left += frac
        ax.text(
            101.2, y, f"{tot:.1f}ms\nGPU/step", ha="left", va="center",
            fontsize=9, color=INK2, linespacing=1.2,
        )

    ax.set_yticks([ypos[b] for b in backends])
    ax.set_yticklabels(backends, fontsize=11, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_title(panel_title, fontsize=10, color=INK, loc="left", pad=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
axes[1].legend(
    handles, SEGMENTS, loc="upper center", bbox_to_anchor=(0.5, -0.22),
    ncol=3, frameon=False, fontsize=9, labelcolor=INK,
)

fig.suptitle(
    "Where the GPU time goes — DeepSeek-V4-Flash, vLLM 0.25.1, 4x GB200 (TP4+EP4, CUDA graphs) — bands: node-mode nsys 2026-07-19/20 (decode = bs-1024 graph-replay composition)\n"
    "per-step kernel self-time per GPU, normalized per backend; fi_dg isolates the integration layer (same kernel as native), fi_nvfp4 adds the kernel swap",
    fontsize=11, color=INK, x=0.02, ha="left",
)

plt.tight_layout(rect=(0, 0.02, 1, 0.94))
out = __file__.rsplit("/", 1)[0] + "/results/gpu_time_attribution.png"
plt.savefig(out, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
