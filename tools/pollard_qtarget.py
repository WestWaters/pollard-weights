#!/usr/bin/env python3
"""Quality-targeted GGUF rungs: the SMALLEST measured-allocation build whose MEASURED fidelity meets a quality class.

A size-targeted rung (the ladder's Q6_K / IQ4_XS / IQ3_S / IQ2, each a fraction of the f16) is whatever quality that
size happens to give. On a model that quantizes hard below 5 bits (Nemotron-3.5-Bolt-E4B / Gemma 4 E4B: the IQ4_XS
rung measured KLD 0.198, top-1 87.6% against the IQ4 class gate of 0.08 / 90) that rung fails its gate and nothing
small ships. A quality rung turns the question round: for a class (Q90 = KLD <= 0.08 and top-1 >= 90%, Q85 = KLD <=
0.2 and top-1 >= 85%), bisect the budget of the SAME measured allocation (pollard-fit with the sensitivity profile and
imatrix -- bytes go where the probe says the body is sensitive), build each candidate, measure its real KLD / top-1
against the f16's KL base, and keep the smallest that passes. Nothing is guessed: every kept rung was measured.

    pollard --gguf model-f16.gguf --ladder --rung Q90 --imatrix m.imatrix --sensitivity m.sensitivity.json \
            --heldout calib.heldout.txt --run                       # -> <model>-Pollard-Q90-<size>GB.gguf

The rung's name says what it is: its quality class and its size. pollard release builds them with `lanes.gguf.quality:
[Q90, Q85]` (and `lanes.gguf.ladder:` to drop size-targeted labels the quality rungs replace).
"""
from __future__ import annotations

import os
import re

#: class -> (KLD max, top-1 min %). The thresholds are the gate's own for the bit-width classes these replace
#: (gates.yaml: IQ4 class 0.08 / 90, IQ3 class 0.2 / 85), so a quality rung passes the same bar a stock rung would.
QUALITY_CLASSES = {"Q90": (0.08, 90.0), "Q85": (0.2, 85.0)}
#: stop bisecting when the passing and failing budgets project within this many GB of each other
TOL_GB = 0.2
MAX_STEPS = 6


def parse_class(label):
    """'Q90' / 'q85' -> the canonical class name, else None."""
    k = str(label or "").strip().upper()
    return k if k in QUALITY_CLASSES else None


def passes(m, kld_max, top1_min):
    """Does a measurement {"kld", "top1"} meet the class?"""
    return (m.get("kld") is not None and float(m["kld"]) <= kld_max
            and m.get("top1") is not None and float(m["top1"]) >= top1_min)


def rung_tag(cls, path_or_gb):
    """'Q90-4.6GB': the class and the real file size -- what the rung is, not a stock type it is not."""
    gb = path_or_gb if isinstance(path_or_gb, (int, float)) else os.path.getsize(path_or_gb) / 1e9
    return f"{cls}-{gb:.1f}GB"


def search(points, cls, build, measure, lo=None, hi=None, tol_gb=TOL_GB, max_steps=MAX_STEPS, log=print,
           discard=None, top_passes=False):
    """The smallest budget whose build passes the class.

    points: [(ram, preset, projected_gb)] from pollard_auto.ladder_points (sorted by ram); build(ram) -> path of the
    built GGUF; measure(path) -> {"kld", "top1", "ppl"}; discard(path) deletes a candidate that is not kept.
    lo / hi: projected-GB bounds of the search (default: the whole grid). Returns
    {"path", "ram", "gb", "measured", "tried": [(ram, gb, measured, ok)]} for the winner, or {"path": None, ...}
    when even the top of the range fails (the class is not reachable below `hi`). top_passes: the top of the range is
    already known to pass (a stricter class's rung sits there), so it is not rebuilt -- the search only looks for
    something smaller, and returns no path when nothing smaller passes."""
    kld_max, top1_min = QUALITY_CLASSES[cls]
    grid = sorted({(round(r, 2), float(g)) for r, _, g in points}, key=lambda x: x[0])
    if lo is not None:
        grid = [p for p in grid if p[1] >= lo]
    if hi is not None:
        grid = [p for p in grid if p[1] <= hi]
    if not grid:
        raise SystemExit(f"quality rung {cls}: no budget in the plan grid between {lo} and {hi} GB")
    tried, best = [], None

    def attempt(i):
        ram, gb = grid[i]
        path = build(ram)
        m = measure(path)
        ok = passes(m, kld_max, top1_min)
        tried.append((ram, gb, m, ok))
        log(f"   {cls} candidate --ram {ram:<5} ~{gb:.2f} GB: KLD {float(m.get('kld') or 0):.4f}  "
            f"top-1 {float(m.get('top1') or 0):.2f}%  -> {'PASS' if ok else 'fail'} "
            f"(want KLD <= {kld_max}, top-1 >= {top1_min})")
        return path, m, ok

    hi_i, lo_i = len(grid) - 1, -1                  # lo_i: the biggest budget known to FAIL (-1: none yet)
    if top_passes:
        best = None
    else:
        path, m, ok = attempt(hi_i)                 # the top must pass, or nothing in range reaches the class
        if not ok:
            if discard:
                discard(path)
            return {"path": None, "tried": tried, "class": cls}
        best = {"path": path, "ram": grid[hi_i][0], "gb": grid[hi_i][1], "measured": m}
    while len(tried) < max_steps and hi_i - lo_i > 1 and grid[hi_i][1] - grid[lo_i + 1][1] > tol_gb:
        mid = (lo_i + 1 + hi_i) // 2
        if mid == hi_i:
            mid = hi_i - 1
        if discard and best and best.get("path"):
            discard(best["path"])            # ONE candidate on disk at a time: the winner is rebuilt at the end
            best["path"] = None
        path, m, ok = attempt(mid)
        if ok:
            best, hi_i = {"path": path, "ram": grid[mid][0], "gb": grid[mid][1], "measured": m}, mid
        else:
            if discard:
                discard(path)
            lo_i = mid
    if best and not best.get("path"):
        best["path"] = build(best["ram"])     # deterministic: the same budget gives the same file it measured
    return {**(best or {"path": None}), "tried": tried, "class": cls}


def manifest_tag_ok(tag):
    return bool(re.fullmatch(r"Q\d\d-\d+(\.\d+)?GB", str(tag)))
