"""GPU-time attribution chart: 100%-stacked horizontal bars, one per backend.

Data: nsys cuda_gpu_kern_sum buckets (see RUNS.md / FINDINGS.md), prefill
workload, 2026-07-15. Regenerate: python make_time_attribution_chart.py
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
    "TP allreduce (skew slack)",
    "MoE staging + small torch kernels",
    "attention",
    "other (norm, rope/kv, dense, misc)",
]

# seconds of GPU kernel self-time per bucket (identical prefill workload)
DATA = {
    "native":   [11.32, 12.37, 1.66, 2.33, 6.67],
    "fi_dg":    [12.01, 6.25, 3.38, 2.23, 6.42],
    "fi_nvfp4": [14.10, 14.74, 11.70, 2.33, 8.60],
}

# text color per segment fill (dark hues get white labels)
LABEL_INK = ["#ffffff", "#ffffff", INK, INK, INK]

backends = list(DATA.keys())
totals = {b: sum(v) for b, v in DATA.items()}

fig, ax = plt.subplots(figsize=(13, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ypos = {b: i for i, b in enumerate(reversed(backends))}
BAR_H = 0.58

for b in backends:
    y = ypos[b]
    left = 0.0
    tot = totals[b]
    for si, sec in enumerate(DATA[b]):
        frac = sec / tot * 100
        ax.barh(
            y, frac, left=left, height=BAR_H,
            color=COLORS[si], edgecolor=SURFACE, linewidth=2, zorder=3,
        )
        if frac >= 4.0:
            ax.text(
                left + frac / 2, y, f"{frac:.0f}%\n({sec:.1f}s)",
                ha="center", va="center",
                fontsize=8.5 if frac >= 5.5 else 7.2,
                color=LABEL_INK[si], zorder=4, linespacing=1.1,
            )
        left += frac
    ax.text(
        101.2, y, f"{tot:.1f}s\nGPU total", ha="left", va="center",
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

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
ax.legend(
    handles, SEGMENTS, loc="upper center", bbox_to_anchor=(0.5, -0.14),
    ncol=3, frameon=False, fontsize=9, labelcolor=INK,
)

ax.set_title(
    "Where the GPU time goes — DeepSeek-V4-Flash prefill, vLLM 0.25.1, 4x GB200 (TP4+EP4, eager)\n"
    "identical workload per bar; nsys kernel self-time, normalized per backend",
    fontsize=11, color=INK, loc="left", pad=12,
)

plt.tight_layout()
out = __file__.rsplit("/", 1)[0] + "/results/gpu_time_attribution.png"
plt.savefig(out, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
