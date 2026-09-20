#!/usr/bin/env python3
"""pollard-fragile -- which tensors will quantization hurt most, from the weights alone.

Deciding what to protect currently takes one of two routes. The hand-coded recipe (protect attn-q,
attn-output, ffn_down, the edge blocks) is fast but is a prior, not a measurement of THIS model. A
measured profile is real but costs a forward pass, an eval corpus and -- on a big model -- hardware
that can hold it.

This is a third signal and it costs neither. Quantization error is driven by the shape of a weight
distribution, not by what the model does with it: a tensor whose values are tightly clustered loses
little to a shared scale, while one with heavy tails spends most of its range representing a few
outliers and leaves the bulk poorly resolved. That shows up in the weights and nowhere else, so it
needs no calibration, no GPU and no eval text -- just the file.

    pollard-fragile --gguf model-f16.gguf
    pollard-fragile --gguf model-f16.gguf --protect iq2_kt --top 8    # emit a --custom-q line

Three statistics, because each catches a different failure:

  excess kurtosis   heavy tails. Gaussian is 0; large positive means rare extreme values that a
                    block scale must stretch to cover, wasting resolution on everything else.
  crest factor      max|w| / median|w|. This is what an absmax quantizer actually divides by, so it
                    maps directly onto how much of the range the typical weight gets to use.
  outlier fraction  share of |w| beyond 6 sigma. A handful of outliers is a mixed-precision case;
                    a broad tail is not fixable by protecting a few values.

What this is NOT: a substitute for a measured profile. It says which tensors are hard to REPRESENT,
not which ones the model's output depends on -- a tensor can be ugly and unimportant, or clean and
load-bearing. Read it as independent corroboration: where it agrees with the recipe, that is two
signals; where it disagrees, that is a tensor worth measuring properly.

Memory is O(one tensor), so model size does not matter. Imports nothing from the rest of Pollard.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def _stats(w):
    """(excess kurtosis, crest factor, outlier fraction) for one tensor, as float64."""
    import numpy as np
    x = np.asarray(w, dtype=np.float64).ravel()
    if x.size < 64:
        return None
    mu = x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        return None
    z = (x - mu) / sd
    kurt = float((z ** 4).mean() - 3.0)
    a = np.abs(x)
    med = float(np.median(a))
    crest = float(a.max() / med) if med > 0 else float("inf")
    outl = float((np.abs(z) > 6.0).mean())
    return kurt, crest, outl


def _dequant(t):
    """Tensor values as float32, whatever the GGUF stored them as."""
    import numpy as np
    if t.tensor_type.name in ("F32", "F16", "BF16"):
        return t.data.astype(np.float32)
    from gguf.quants import dequantize
    return dequantize(t.data, t.tensor_type).astype(np.float32)


def scan(gguf, min_dim=2):
    """Per-tensor stats for every quantizable matmul in the file."""
    from gguf import GGUFReader
    rows = []
    for t in GGUFReader(gguf).tensors:
        if len(t.shape) < min_dim:
            continue                                        # norms and biases are not quantized
        try:
            s = _stats(_dequant(t))
        except Exception:
            continue
        if s is None:
            continue
        kurt, crest, outl = s
        rows.append({"name": t.name, "kind": re.sub(r"^blk\.\d+\.", "", t.name),
                     "layer": int(m.group(1)) if (m := re.match(r"blk\.(\d+)\.", t.name)) else -1,
                     "type": t.tensor_type.name, "params": int(t.shape[0]) * int(t.shape[1]),
                     "kurtosis": kurt, "crest": crest, "outlier_frac": outl})
    return rows


def by_kind(rows):
    """Collapse per-layer tensors into their kind, since that is what a --custom-q rule names."""
    agg = {}
    for r in rows:
        a = agg.setdefault(r["kind"], {"n": 0, "kurt": 0.0, "crest": 0.0, "outl": 0.0,
                                       "params": 0, "worst": (-1e9, -1)})
        a["n"] += 1
        a["kurt"] += r["kurtosis"]
        a["crest"] += r["crest"]
        a["outl"] += r["outlier_frac"]
        a["params"] += r["params"]
        if r["kurtosis"] > a["worst"][0]:
            a["worst"] = (r["kurtosis"], r["layer"])
    out = []
    for k, a in agg.items():
        out.append({"kind": k, "n": a["n"], "params": a["params"],
                    "kurtosis": a["kurt"] / a["n"], "crest": a["crest"] / a["n"],
                    "outlier_frac": a["outl"] / a["n"],
                    "worst_layer": a["worst"][1], "worst_kurtosis": a["worst"][0]})
    return sorted(out, key=lambda r: -r["kurtosis"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16 preferred; a quantized "
                                                  "file is dequantized per tensor and still ranks)")
    ap.add_argument("--top", type=int, default=6, help="how many kinds to name in the suggestion")
    ap.add_argument("--protect", help="atom to suggest for the most fragile kinds (e.g. iq2_kt)")
    ap.add_argument("--out", help="write the full per-tensor scan as JSON")
    a = ap.parse_args()
    if not os.path.exists(a.gguf):
        sys.exit(f"not found: {a.gguf}")

    print(f"== pollard-fragile :: {os.path.basename(a.gguf)}", flush=True)
    rows = scan(a.gguf)
    if not rows:
        sys.exit("no 2-D tensors read -- is this a GGUF with weights in it?")
    kinds = by_kind(rows)

    print(f"   {len(rows)} matmuls, {len(kinds)} kinds. Ranked by heaviest tails.\n")
    print(f"  {'kind':28} {'n':>4} {'params':>10} {'kurtosis':>9} {'crest':>8} {'>6sig':>8}")
    for k in kinds:
        print(f"  {k['kind']:28} {k['n']:>4} {k['params']/1e6:>9.0f}M "
              f"{k['kurtosis']:>9.1f} {k['crest']:>8.1f} {k['outlier_frac']*100:>7.3f}%")

    # The suggestion is deliberately a STARTING POINT, not a recipe: this ranks representability,
    # and a build also has to fit a budget. Naming the kinds is the useful part.
    head = [k for k in kinds[:a.top] if k["kurtosis"] > 1.0]
    if head:
        print(f"\n  heaviest-tailed kinds (protect these first, then measure):")
        for k in head:
            print(f"    {k['kind']:26} kurtosis {k['kurtosis']:7.1f}  "
                  f"worst at layer {k['worst_layer']} ({k['worst_kurtosis']:.1f})")
        if a.protect:
            rule = ",".join(f"{k['kind']}={a.protect}" for k in head)
            print(f"\n  --custom-q \"{rule}\"")
    else:
        print("\n  no kind stands out on tail weight -- this model quantizes evenly, and the "
              "hand-coded\n  recipe is as good a prior as anything here would give you.")

    if a.out:
        json.dump({"gguf": a.gguf, "kinds": kinds, "tensors": rows}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
