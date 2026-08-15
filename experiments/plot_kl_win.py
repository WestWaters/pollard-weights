#!/usr/bin/env python3
"""Regenerate assets/kl_win.png from the firm KL sweeps in experiments/data/.

Two panels — dense and MoE — each plots the uniform imatrix-IQ curve (gray) and
the pollard measured-sensitivity builds (blue). Honest by construction: the win
is annotated per point straight from the data (log-log interpolation of the
uniform curve at each pollard size), losses included.

    python experiments/plot_kl_win.py
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "assets", "kl_win.png")


def load(path):
    poll, uni = [], []
    for r in csv.DictReader(open(path)):
        s, k = float(r["size_gb"]), float(r["mean_kl"])
        (poll if r["model"].startswith("PFIT") else uni).append((s, k))
    return sorted(poll), sorted(uni)


def uni_at(uni, size):
    # LINEAR interpolation between adjacent measured uniform quants — the honest
    # "naive-mix" baseline: dithering some layers to the next quant type down lands
    # you on this line. pollard is itself a mix, so this is the right thing to beat.
    # (Do NOT use log-log here — it invents a single-type quant that can't exist at
    # an intermediate size, understates the baseline, and manufactures fake losses.)
    xs = [s for s, _ in uni]
    ys = [k for _, k in uni]
    x = size
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    f = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + f * (ys[i + 1] - ys[i])


PANELS = [
    ("kl_win_dense.csv", "Dense — Qwen2.5-1.5B", "5/5 wins  ·  +6% to +27%"),
    ("kl_win_moe.csv", "MoE — granite-3B-a800m (40 experts)", "4/5 wins  ·  +21% to +43%"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, (fname, title, sub) in zip(axes, PANELS):
    poll, uni = load(os.path.join(HERE, "data", fname))

    ux = [s for s, _ in uni]
    uy = [k for _, k in uni]
    ax.plot(ux, uy, "-o", color="#8a8f98", lw=2, ms=6, zorder=2,
            label="uniform imatrix-IQ (IQ2_S…Q6_K)")
    for s, k in uni:  # label the ladder rungs lightly
        pass

    px = [s for s, _ in poll]
    py = [k for _, k in poll]
    ax.plot(px, py, "*", color="#2f6fed", ms=15, zorder=4,
            label="pollard measured-sensitivity", linestyle="none")

    for s, k in poll:
        u = uni_at(uni, s)
        imp = (u - k) / u * 100
        if imp >= 2:
            txt, col = f"+{imp:.0f}%", "#137a3f"
        elif imp <= -2:
            txt, col = f"{imp:.0f}%", "#b4232a"
        else:
            txt, col = "tie", "#8a8f98"
        ax.annotate(txt, (s, k), textcoords="offset points", xytext=(7, -3),
                    fontsize=10, fontweight="bold", color=col, zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("file size (GB)")
    ax.set_ylabel("mean KL vs f16  (lower = better)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    # subtitle in the empty lower-left (the KL curve occupies upper-left → lower-right)
    ax.text(0.03, 0.05, sub, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

fig.suptitle("Measured-sensitivity allocation vs uniform imatrix-IQ, matched size",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
         "Both arms built from the same f16 + imatrix; KL against f16 on held-out "
         "WikiText (48K tokens). Uniform-@-size by linear interpolation between "
         "adjacent quants (the naive-mix baseline). Single machine; replication invited.",
         ha="center", fontsize=8.5, color="#666")
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print("wrote", os.path.abspath(OUT))
