#!/usr/bin/env python3
"""pollard-errsrc — attribute MEASURED quant error to a tensor AND to the trigger that caused it,
so the budget goes where it actually buys something.

pollard-sensitivity already answers *which* tensor costs how much: it crushes one group at a time and
measures the real KL hit. What it cannot say is *why* that group cost it -- and the why decides the
cheapest fix:

  CALIBRATION   the imatrix barely saw this tensor (missing, or a call count far below the model's
                median). The error is a coverage hole, not a hardness fact. Spending bits here pays
                to preserve noise. Fix the calib and re-measure.
  OUTLIER       importance is concentrated in a few input channels, so one shared scale has to span
                them and the rest loses range. A transform (smoothing / a protected scale) buys more
                per byte than bits do.
  REPRESENTATION  broad, well-covered importance and still expensive. This one genuinely wants bits.

That split is the point: error from the first two categories is **recoverable without spending bits**,
which frees budget to steer at the tensors in the third. The summary reports that recoverable share as
the steering budget.

  pollard-errsrc --sensitivity model.sensitivity.json --imatrix model.imatrix
  pollard-errsrc --sensitivity s.json --imatrix m.imatrix --json > errsrc.json

⚠️ The KL costs are measured. The TRIGGERS are inferred from imatrix statistics and are a hypothesis
about cause -- test a call before trusting it (fix the calib, or apply the transform, then re-run
pollard-sensitivity and check the cost actually fell). Read-only; nothing is rebuilt here.
"""
import argparse
import json
import os
import struct
import sys


def read_imatrix(path):
    """name -> {ncall, values[]}. Format: int32 n, then per entry: int32 len, bytes name,
    int32 ncall, int32 nval, float[nval]. Same layout pollard-automap parses."""
    out = {}
    with open(path, "rb") as f:
        d = f.read()
    try:
        n = struct.unpack_from("<i", d, 0)[0]
        off = 4
        for _ in range(n):
            ln = struct.unpack_from("<i", d, off)[0]
            off += 4
            name = d[off:off + ln].decode("utf-8", "replace")
            off += ln
            ncall = struct.unpack_from("<i", d, off)[0]
            nval = struct.unpack_from("<i", d, off + 4)[0]
            off += 8
            vals = struct.unpack_from("<%df" % nval, d, off)
            off += 4 * nval
            out[name] = {"ncall": ncall, "values": vals}
    except Exception as e:
        print(f"   (imatrix parse stopped early: {e})", file=sys.stderr)
    return out


def _spread(vals):
    """p99.9 / median over the per-channel importance. High = a few channels dominate the scale."""
    v = sorted(x for x in vals if x > 0)
    if len(v) < 8:
        return None
    med = v[len(v) // 2]
    hi = v[min(len(v) - 1, int(len(v) * 0.999))]
    return (hi / med) if med > 0 else None


# Coarse and stated out loud so they can be argued with and tuned per family.
THIN_CALL_FRAC = 0.25    # ncall below this fraction of the model median = a coverage hole
SPREAD_HI = 8.0          # p99.9/median importance above this = concentrated in few channels


def classify(entry, median_ncall):
    if entry is None:
        return "CALIBRATION", "no imatrix entry at all -- this tensor was never covered"
    if median_ncall and entry["ncall"] < THIN_CALL_FRAC * median_ncall:
        return "CALIBRATION", (f"imatrix saw it {entry['ncall']} times vs a model median of "
                               f"{median_ncall} -- thin coverage, so the signal is noise")
    sp = _spread(entry["values"])
    if sp is not None and sp >= SPREAD_HI:
        return "OUTLIER", (f"importance concentrated in a few channels (p99.9/median {sp:.1f}x) -- "
                           "one shared scale cannot span them")
    return "REPRESENTATION", "well covered, importance broad -- this one genuinely wants bits"


def tensor_name(group, layer):
    """sensitivity groups -> the imatrix tensor name for that layer."""
    g = group.lower()
    mapping = {"attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v",
               "attn_output": "attn_output", "attn_out": "attn_output",
               "ffn_gate": "ffn_gate", "ffn_up": "ffn_up", "ffn_down": "ffn_down"}
    stem = mapping.get(g, g)
    return f"blk.{layer}.{stem}.weight"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--sensitivity", required=True, help="pollard-sensitivity profile (measured KL)")
    ap.add_argument("--imatrix", required=True, help="the imatrix those builds used")
    ap.add_argument("--top", type=int, default=20, help="how many costliest entries to show")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    prof = json.load(open(a.sensitivity))
    im = read_imatrix(a.imatrix)
    if not im:
        raise SystemExit("could not read any imatrix entries")
    calls = sorted(e["ncall"] for e in im.values() if e["ncall"] > 0)
    median_ncall = calls[len(calls) // 2] if calls else 0

    rows = []
    for group, per_layer in prof.items():
        if not isinstance(per_layer, dict):
            continue                      # noise/ref/probe/... metadata keys
        for layer, kl in per_layer.items():
            if kl is None:
                continue
            name = tensor_name(group, layer)
            entry = im.get(name)
            trig, why = classify(entry, median_ncall)
            rows.append({"tensor": name, "group": group, "layer": int(layer),
                         "kl_cost": float(kl), "trigger": trig, "why": why})

    if not rows:
        raise SystemExit("no measured entries in the sensitivity profile")
    rows.sort(key=lambda r: -r["kl_cost"])
    total = sum(r["kl_cost"] for r in rows)
    by_trig = {}
    for r in rows:
        by_trig[r["trigger"]] = by_trig.get(r["trigger"], 0.0) + r["kl_cost"]

    if a.json:
        print(json.dumps({"rows": rows, "total_kl": total, "by_trigger": by_trig,
                          "median_ncall": median_ncall}, indent=2))
        return

    print(f"{'tensor':38s} {'KL cost':>9s}  trigger")
    for r in rows[:a.top]:
        print(f"{r['tensor'][:38]:38s} {r['kl_cost']:9.4f}  {r['trigger']}")
        print(f"{'':38s} {'':9s}  {r['why']}")

    print(f"\nmeasured KL attributed: {total:.4f} across {len(rows)} tensors "
          f"(imatrix median call count {median_ncall})")
    for t in ("CALIBRATION", "OUTLIER", "REPRESENTATION"):
        v = by_trig.get(t, 0.0)
        print(f"  {t:15s} {v:8.4f}  ({v/total*100 if total else 0:4.1f}%)")

    recoverable = by_trig.get("CALIBRATION", 0.0) + by_trig.get("OUTLIER", 0.0)
    print(f"\nSTEERING BUDGET: {recoverable/total*100 if total else 0:.1f}% of the measured error is "
          f"attributed to coverage or scale, not to hardness.")
    print("  That share should be attacked with a better calib and a transform FIRST -- bits spent")
    print("  there buy less than bits moved to the REPRESENTATION tensors.")
    print("\nTriggers are inferred from imatrix statistics: treat them as a hypothesis, fix one, and")
    print("re-run pollard-sensitivity to confirm the cost actually fell.")


if __name__ == "__main__":
    main()
