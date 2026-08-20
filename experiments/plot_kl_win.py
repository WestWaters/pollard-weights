#!/usr/bin/env python3
"""Regenerate assets/kl_win.png from the firm KL sweeps in experiments/data/.

Two panels — dense and MoE — each a quality-vs-size Pareto view: the uniform
imatrix-IQ curve (recessive gray) and the pollard measured-sensitivity builds
(hero blue). Honest by construction: the win is annotated per point straight from
the data (linear/naive-mix interpolation of the uniform curve at each pollard
size), losses shown in red. Design follows the dataviz system (validated palette,
recessive grid, thin marks, selective direct labels).

    python experiments/plot_kl_win.py
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "assets", "kl_win.png")

# --- dataviz palette (validated reference instance) ---
SURFACE = "#ffffff"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#8a8f98"     # recessive baseline / grid
POLLARD  = "#2a78d6"     # categorical slot 1 (the hero series)
WIN      = "#0ca30c"     # status: good
LOSS     = "#d03b3b"     # status: critical
GRID     = "#ececea"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def load(path):
    poll, uni = [], []
    for r in csv.DictReader(open(path)):
        s, k = float(r["size_gb"]), float(r["mean_kl"])
        (poll if r["model"].startswith("PFIT") else uni).append((s, k))
    return sorted(poll), sorted(uni)


def uni_at(uni, size):
    # LINEAR interpolation between adjacent measured uniform quants — the honest
    # naive-mix baseline (dithering to the next quant lands you on this line).
    xs = [s for s, _ in uni]; ys = [k for _, k in uni]
    if size <= xs[0]:
        i = 0
    elif size >= xs[-1]:
        i = len(xs) - 2
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= size)
    f = (size - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + f * (ys[i + 1] - ys[i])


PANELS = [
    ("kl_win_dense.csv", "Dense", "Qwen2.5-1.5B"),
    ("kl_win_moe.csv", "MoE", "granite-3B-a800m · 40 experts"),
]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), dpi=150)
fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.13, wspace=0.20)

for ax, (fname, kind, model) in zip(axes, PANELS):
    poll, uni = load(os.path.join(HERE, "data", fname))
    ax.set_yscale("log")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color=GRID, lw=0.9, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax.tick_params(length=0)

    # baseline: recessive gray
    ux = [s for s, _ in uni]; uy = [k for _, k in uni]
    ax.plot(ux, uy, "-", color=MUTED, lw=2, zorder=2)
    ax.scatter(ux, uy, s=26, color=MUTED, zorder=3, edgecolor=SURFACE, linewidth=1.2)

    # hero: pollard measured-sensitivity
    px = [s for s, _ in poll]; py = [k for _, k in poll]
    ax.plot(px, py, "-", color=POLLARD, lw=2.4, zorder=4)
    ax.scatter(px, py, s=88, color=POLLARD, marker="D", zorder=5,
               edgecolor=SURFACE, linewidth=1.6)

    wins = 0
    for s, k in poll:
        u = uni_at(uni, s); imp = (u - k) / u * 100
        if imp >= 2:
            txt, col, wins = f"+{imp:.0f}%", WIN, wins + 1
        elif imp <= -2:
            txt, col = f"{imp:.0f}%", LOSS
        else:
            txt, col = "±0%", MUTED
        ax.annotate(txt, (s, k), textcoords="offset points", xytext=(9, -4),
                    fontsize=10.5, fontweight="bold", color=col, zorder=6)

    ax.set_xlabel("file size  (GB)", fontsize=10.5)
    if ax is axes[0]:
        ax.set_ylabel("mean KL vs f16   ·   lower is better", fontsize=10.5)
    # panel title: kind (bold) + model (muted), + the headline win
    imps = [(uni_at(uni, s) - k) / uni_at(uni, s) * 100 for s, k in poll]
    good = [i for i in imps if i >= 2]
    head = f"{wins}/{len(poll)} wins · +{min(good):.0f}–{max(good):.0f}% lower KL" if good else ""
    ax.annotate(f"{kind}  ·  {model}", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 26), textcoords="offset points", fontsize=12.5,
                fontweight="bold", color=INK, ha="left")
    ax.annotate(head, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9),
                textcoords="offset points", fontsize=10.5, color=POLLARD,
                ha="left", fontweight="bold")

# figure title + legend + source
fig.text(0.07, 0.955, "Measured-sensitivity allocation beats uniform imatrix-IQ",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.07, 0.915, "same size, same imatrix — pollard keeps the bits where a "
         "measured KL sweep says they matter", fontsize=11, color=INK_2, ha="left")
# manual legend (top-right), identity by mark not color-alone
lx = 0.985
fig.text(lx, 0.955, "◆ pollard", color=POLLARD, fontsize=11.5, fontweight="bold", ha="right")
fig.text(lx, 0.918, "● uniform imatrix-IQ", color=MUTED, fontsize=11, ha="right")
fig.text(0.07, 0.028, "KL vs f16 on held-out WikiText (48K tokens). Uniform-at-size "
         "by linear interpolation between adjacent quants. Single machine; "
         "replication invited.", fontsize=8.5, color=MUTED, ha="left")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote", os.path.abspath(OUT))
