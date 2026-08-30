#!/usr/bin/env python3
"""assets/gptq_h2h.png — 4-bit reconstruction head-to-head on Qwen2.5-7B (W4g128):
Pollard's full-Hessian error-feedback GPTQ vs round-to-nearest, PPL vs the f16 model.
Same dataviz system as plot_kl_win.py.

    python experiments/plot_gptq_h2h.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "assets", "gptq_h2h.png")
SURFACE = "#ffffff"; INK = "#0b0b0b"; INK_2 = "#52514e"; MUTED = "#8a8f98"
POLLARD = "#2a78d6"; WIN = "#0ca30c"; GRID = "#ececea"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# Qwen2.5-7B-Instruct, W4g128, PPL on the eval, f16 reference = 6.2881 (gptq7b.log)
F16 = 6.2881
BARS = [  # (label, ppl, color)
    ("f16\n(reference)", 6.2881, MUTED),
    ("Pollard GPTQ\nerror-feedback", 6.5555, POLLARD),
    ("round-to-nearest\n(RTN)", 6.7829, "#c9ccd1"),
]

fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
fig.subplots_adjust(left=0.11, right=0.96, top=0.80, bottom=0.13)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="y", color=GRID, lw=0.9, zorder=0)
ax.tick_params(length=0)

xs = range(len(BARS))
ax.bar(xs, [b[1] for b in BARS], width=0.62, color=[b[2] for b in BARS],
       zorder=3, edgecolor=SURFACE, linewidth=1.5)
ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in BARS], fontsize=10)
ax.set_ylim(6.0, 6.95)
ax.set_ylabel("perplexity vs f16   ·   lower is better", fontsize=10.5)
for i, b in enumerate(BARS):
    ax.annotate(f"{b[1]:.2f}", (i, b[1]), textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=11, fontweight="bold",
                color=POLLARD if b[2] == POLLARD else INK_2)
    if i:
        ax.annotate(f"+{b[1]-F16:.2f}", (i, b[1]), textcoords="offset points",
                    xytext=(0, 20), ha="center", fontsize=9.5, color=INK_2)
# the win: Pollard vs RTN
gain = (BARS[2][1] - BARS[1][1]) / (BARS[2][1] - F16) * 100
ax.annotate(f"−{gain:.0f}% of RTN's error recovered", (1, BARS[1][1]),
            textcoords="offset points", xytext=(0, 42), ha="center",
            fontsize=10, fontweight="bold", color=WIN)

fig.text(0.11, 0.945, "Pollard's error-feedback GPTQ beats round-to-nearest at 4-bit",
         fontsize=14.5, fontweight="bold", color=INK, ha="left")
fig.text(0.11, 0.895, "Qwen2.5-7B-Instruct · W4 group-128 · the reconstruction lever an "
         "imatrix alone can't do", fontsize=10.5, color=INK_2, ha="left")
fig.text(0.11, 0.028, "PPL vs the f16 model on held-out eval. Pollard `pollard-gptq` "
         "sequential/act-order, block-offload on a 16 GB GPU. Single machine; replication invited.",
         fontsize=8.5, color=MUTED, ha="left")
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote", os.path.abspath(OUT))
