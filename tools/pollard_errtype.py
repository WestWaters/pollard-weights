#!/usr/bin/env python3
"""pollard-errtype — classify WHY a tensor will quantize badly, so the right lever is chosen
without building every candidate first.

pollard-precondition answers "which preconditioner wins" the honest way: build each candidate and
score KL. That is correct and it is expensive -- a full build per candidate per model. But the
measured behaviour it records is not arbitrary, and the pattern points at a cause:

  * diagonal smoothing helps SCALAR quants and HURTS IQ codebook quants
  * rotation helps IQ, and its win GROWS as bits drop (a wash at 3-bit, -9.7% at IQ2_XXS)

Those are different errors, not one error with two cures. Smoothing fixes per-channel SCALE spread:
a handful of fat channels force a shared scale that wastes range on everyone else. Rotation fixes
CONCENTRATION: energy packed into few directions, so a codebook spends its entries on empty space.
A tensor whose weights are already flat and well-spread has neither problem -- nothing but more bits
will help it.

So: measure the two error sources per tensor, cheaply, from the weights alone.

  outlier_ratio   p99.9(|w|) / p50(|w|) over per-output-channel norms -- scale spread
  incoherence     max|W| * sqrt(m*n) / ||W||_F   -- the standard mu-incoherence; how far from flat
  spectral_flat   geometric/arithmetic mean of the singular value spectrum -- concentration
                  (0 = all energy in one direction, 1 = perfectly flat)

and read off the type:

  OUTLIER      high scale spread            -> smoothing (scalar targets); rotation also helps
  CONCENTRATED low spectral flatness        -> rotation, and more so the lower the bits
  FLAT         neither                      -> no preconditioner earns its cost; spend bits instead

  pollard-errtype --model Qwen/Qwen2.5-0.5B
  pollard-errtype --model ./local-dir --layers 0,15,23 --json

⚠️ This PREDICTS which lever should win from weight statistics. It does not replace measuring.
Treat it as the cheap first pass that tells pollard-precondition which candidates are worth
building, and check the prediction against the measured KL before trusting it on a new family.

Read-only. Loads weights layer by layer; never writes a model.
"""
import argparse
import json
import os
import re
import sys

TARGET = re.compile(r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")


def _stats(w):
    """Error-source statistics for one 2-D weight. `w` is a torch tensor on CPU."""
    import torch
    w = w.detach().to(torch.float32)
    if w.ndim != 2:
        return None
    m, n = w.shape

    # scale spread across output channels: what a shared per-tensor scale has to cover
    chan = w.abs().amax(dim=1)                       # per-output-channel peak
    p50 = chan.median().item()
    p999 = torch.quantile(chan, 0.999).item()
    outlier_ratio = (p999 / p50) if p50 > 0 else float("inf")

    # mu-incoherence: 1.0 means perfectly flat, large means one entry dominates
    fro = w.norm().item()
    incoherence = (w.abs().max().item() * (m * n) ** 0.5 / fro) if fro > 0 else float("inf")

    # spectral flatness on a subsample (full SVD on a big tensor is not worth the wall-clock)
    k = min(m, n, 512)
    sub = w[:k, :k] if (m > k or n > k) else w
    try:
        sv = torch.linalg.svdvals(sub.to(torch.float32))
        sv = sv[sv > 0]
        spectral_flat = float(torch.exp(torch.log(sv).mean()) / sv.mean()) if len(sv) else 0.0
    except Exception:
        spectral_flat = float("nan")

    return {"outlier_ratio": round(outlier_ratio, 3),
            "incoherence": round(incoherence, 3),
            "spectral_flat": round(spectral_flat, 4),
            "shape": [m, n]}


# Thresholds are deliberately coarse and are the part most in need of calibration against measured
# KL. They are stated here rather than buried so they can be argued with and tuned per family.
OUTLIER_HI = 4.0        # p99.9/p50 channel peak above this = a few channels dominate the scale
FLAT_LO = 0.35          # spectral flatness below this = energy concentrated in few directions


def classify(s):
    if s is None:
        return "SKIP", "not a 2-D weight"
    outlier = s["outlier_ratio"] >= OUTLIER_HI
    concentrated = s["spectral_flat"] <= FLAT_LO
    if outlier and concentrated:
        return "OUTLIER+CONCENTRATED", ("smoothing for scalar targets, rotation for codebook targets; "
                                        "both error sources are present")
    if outlier:
        return "OUTLIER", ("smoothing (scalar targets only -- it hurts IQ codebooks); "
                           "rotation is the codebook-side answer")
    if concentrated:
        return "CONCENTRATED", "rotation, and the win should grow as the bit target drops"
    return "FLAT", "no preconditioner earns its cost here -- spend bits, not transforms"


def iter_weights(model, layers=None, limit=0):
    """Yield (name, tensor) for the projection weights, without loading the whole model."""
    from safetensors import safe_open
    files, base = [], model
    if os.path.isdir(model):
        idx = os.path.join(model, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted({v for v in json.load(open(idx))["weight_map"].values()})
        elif os.path.exists(os.path.join(model, "model.safetensors")):
            files = ["model.safetensors"]
    else:
        from huggingface_hub import hf_hub_download
        try:
            wm = json.load(open(hf_hub_download(model, "model.safetensors.index.json")))["weight_map"]
            files = sorted(set(wm.values()))
        except Exception:
            files = ["model.safetensors"]
        base = None

    n = 0
    for fn in files:
        path = os.path.join(base, fn) if base else __import__("huggingface_hub").hf_hub_download(model, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if not TARGET.search(name) or not name.endswith(".weight"):
                    continue
                if layers is not None:
                    m = re.search(r"layers\.(\d+)\.", name)
                    if not m or int(m.group(1)) not in layers:
                        continue
                yield name, f.get_tensor(name)
                n += 1
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local dir (fp16/bf16 source)")
    ap.add_argument("--layers", help="comma-separated layer indices (default: a sample)")
    ap.add_argument("--limit", type=int, default=0, help="cap tensors examined (0 = all selected)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    layers = None
    if a.layers:
        layers = {int(x) for x in a.layers.split(",") if x.strip() != ""}

    rows = []
    for name, w in iter_weights(a.model, layers, a.limit):
        s = _stats(w)
        kind, advice = classify(s)
        rows.append({"tensor": name, "type": kind, "advice": advice, **(s or {})})

    if not rows:
        raise SystemExit("no projection weights found")

    if a.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"{'tensor':52s} {'type':22s} {'outlier':>8s} {'flatness':>9s} {'incoh':>7s}")
    for r in rows:
        print(f"{r['tensor'][:52]:52s} {r['type']:22s} "
              f"{r.get('outlier_ratio', float('nan')):8.2f} "
              f"{r.get('spectral_flat', float('nan')):9.4f} {r.get('incoherence', float('nan')):7.1f}")

    counts = {}
    for r in rows:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    total = len(rows)
    print("\nerror-type mix: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # The useful result is the SPREAD, not a single verdict. A model whose tensors disagree cannot be
    # served well by one model-wide preconditioner, which is what the lever tools apply today.
    top, top_n = max(counts.items(), key=lambda kv: kv[1])
    share = top_n / total
    if share >= 0.8:
        print(f"uniform: {share:.0%} of tensors are {top} -- one model-wide lever is a reasonable fit")
    else:
        print(f"MIXED: the most common type is only {share:.0%} of tensors ({top}).")
        print("       A single model-wide preconditioner cannot suit all of them. The per-tensor")
        print("       `advice` column is the actionable output; --json carries it per tensor.")
    print("\nPREDICTION ONLY -- these thresholds are uncalibrated. Confirm against measured KL with")
    print("pollard-precondition before trusting the call on an unfamiliar family.")


if __name__ == "__main__":
    main()
