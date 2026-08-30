#!/usr/bin/env python3
"""Regenerate assets/gold_card.png — the multi-model board: Pollard's mixed build
vs the uniform 1-bit / 2-bit trellis quants it sits between, on real 7B and 14B
models. Quality-vs-size: the uniform IQ1_KT->IQ2_KT slide (recessive gray) and the
Pollard mix (hero blue) landing BELOW the naive interpolation at its size. Same
dataviz system as plot_kl_win.py (validated palette, thin marks, honest per-point
annotation).

    python experiments/plot_gold_card.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "assets", "gold_card.png")

SURFACE = "#ffffff"; INK = "#0b0b0b"; INK_2 = "#52514e"; MUTED = "#8a8f98"
POLLARD = "#2a78d6"; WIN = "#0ca30c"; GRID = "#ececea"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# measured on WikiText-2 (raw), ctx 2048, 145 chunks, ik_llama.cpp trellis quants.
# (size_gb, ppl) for the three bars; mix is the Pollard build.
PANELS = [
    ("7B", "Qwen2.5-7B-Instruct",
     {"iq1": (1.80, 11.86), "mix": (1.90, 10.23), "iq2": (2.21, 8.19)}),
    ("14B", "Qwen2.5-14B-Instruct",
     {"iq1": (3.37, 9.65), "mix": (3.65, 8.27), "iq2": (4.30, 6.92)}),
]


def interp(iq1, iq2, size):
    (s1, p1), (s2, p2) = iq1, iq2
    return p1 + (size - s1) / (s2 - s1) * (p2 - p1)


fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), dpi=150)
fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.13, wspace=0.20)

for ax, (tag, model, d) in zip(axes, PANELS):
    iq1, mix, iq2 = d["iq1"], d["mix"], d["iq2"]
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color=GRID, lw=0.9, zorder=0)
    ax.tick_params(length=0)

    # uniform slide iq1 -> iq2 (recessive gray), the naive-mix baseline
    ux = [iq1[0], iq2[0]]; uy = [iq1[1], iq2[1]]
    ax.plot(ux, uy, "-", color=MUTED, lw=2, zorder=2)
    ax.scatter(ux, uy, s=42, color=MUTED, zorder=3, edgecolor=SURFACE, linewidth=1.4)
    ax.annotate("uniform IQ1_KT", iq1, textcoords="offset points", xytext=(6, 8),
                fontsize=9.5, color=INK_2)
    ax.annotate("uniform IQ2_KT", iq2, textcoords="offset points", xytext=(-6, 8),
                fontsize=9.5, color=INK_2, ha="right")

    # pollard mix (hero) — below the line
    ax.scatter([mix[0]], [mix[1]], s=150, color=POLLARD, marker="D", zorder=6,
               edgecolor=SURFACE, linewidth=1.8)
    # drop line to the interpolation at the mix's size (shows the win)
    lin = interp(iq1, iq2, mix[0])
    ax.plot([mix[0], mix[0]], [mix[1], lin], ":", color=POLLARD, lw=1.4, zorder=4)
    gain = (lin - mix[1]) / lin * 100
    vs1 = (iq1[1] - mix[1]) / iq1[1] * 100
    ax.annotate(f"PollardMix\n{mix[1]:.2f} PPL @ {mix[0]:.2f} GB",
                mix, textcoords="offset points", xytext=(12, -6), fontsize=10.5,
                fontweight="bold", color=POLLARD)
    ax.annotate(f"+{gain:.0f}% under the slide", (mix[0], (mix[1] + lin) / 2),
                textcoords="offset points", xytext=(10, 0), fontsize=9.5,
                fontweight="bold", color=WIN, va="center")

    ax.set_xlabel("file size  (GB)", fontsize=10.5)
    if ax is axes[0]:
        ax.set_ylabel("perplexity vs f16   ·   lower is better", fontsize=10.5)
    ax.annotate(f"{tag}  ·  {model}", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 26), textcoords="offset points", fontsize=12.5,
                fontweight="bold", color=INK, ha="left")
    ax.annotate(f"−{vs1:.0f}% PPL vs uniform 1-bit, same size class",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", fontsize=10.5, color=POLLARD,
                ha="left", fontweight="bold")
    ax.margins(x=0.16, y=0.18)

fig.text(0.07, 0.955, "PollardMix beats uniform 1-bit — on real models",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.07, 0.915, "crush the expert/FFN body to 1-bit, protect attention + "
         "residual writers — same 1-bit size class, measurably better", fontsize=11,
         color=INK_2, ha="left")
lx = 0.985
fig.text(lx, 0.955, "◆ PollardMix", color=POLLARD, fontsize=11.5, fontweight="bold", ha="right")
fig.text(lx, 0.918, "● uniform trellis (IQ1_KT / IQ2_KT)", color=MUTED, fontsize=11, ha="right")
fig.text(0.07, 0.028, "PPL on WikiText-2 raw, ctx 2048, 145 chunks, ik_llama.cpp "
         "trellis quants. Also wins on KL-to-f16, top-1 agreement and chat. "
         "Single machine; replication invited.", fontsize=8.5, color=MUTED, ha="left")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote", os.path.abspath(OUT))
