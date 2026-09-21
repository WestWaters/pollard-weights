#!/usr/bin/env python3
"""pollard-failmode -- is this build's damage repairable, or did a component fail outright?

Low-bit damage comes in two shapes, and they want opposite responses.

  SIGNAL DEGRADATION   The computation still runs. Precision erodes and error accumulates
                       gradually with depth. Preconditioning recovers most of it, which is what
                       `pollard-doctor --repair` does.

  COMPUTATION COLLAPSE A component stopped working. The signal is destroyed early and everything
                       after it is processing noise. Preconditioning does NOT recover this --
                       repairing it wastes a full reconvert and can ship a build that looks
                       repaired because perplexity on short contexts partially recovers.

Telling them apart is cheap: run the reference and the build over the same tokens and record how
far apart the hidden states are at every layer. Degradation is a curve that climbs. Collapse is a
cliff, and it happens early.

    pollard-failmode --ref f16-model --model quantized --calib calib.txt

This does not refuse anything. A collapse verdict names the layer that broke and the levers that
actually address it, because "repair will not help" is only useful next to "this will".
"""
from __future__ import annotations

import argparse
import json
import sys

# A layer is "broken" when the quantized hidden state has lost most of its agreement with the
# reference. Cosine, not magnitude: a scaled-but-aligned activation still carries its information,
# and RMS drift alone would flag healthy layers on models with big activation outliers.
BROKEN = 0.70          # cosine below this = the layer's output no longer resembles the reference
EARLY = 0.35           # a break inside the first third of the stack is structural, not cumulative
CLIFF = 0.25           # a single-layer cosine drop this large is a component failing, not erosion


def classify(cosines, depth_frac=None):
    """Which failure mode, from per-layer cosine similarity between reference and quantized.

    `cosines` is one value per layer, in order. Returns the verdict and the reasoning, because a
    one-word answer that cannot be checked is not worth acting on.
    """
    n = len(cosines)
    if n == 0:
        # still carries advice: every verdict has a next step, including this one
        return {"mode": "unknown", "reason": "no layers measured", "repairable": None,
                "layers": 0, "min_cosine": None, "first_broken_layer": None,
                "worst_drop": None, "worst_drop_layer": None, "advice": _advice("unknown")}

    depth = depth_frac or [i / max(n - 1, 1) for i in range(n)]
    drops = [0.0] + [cosines[i - 1] - cosines[i] for i in range(1, n)]
    worst_drop = max(drops)
    worst_at = drops.index(worst_drop)

    broken = [i for i, c in enumerate(cosines) if c < BROKEN]
    first_broken = broken[0] if broken else None

    # a cliff: one layer loses a large fraction of agreement on its own
    cliff = worst_drop >= CLIFF
    # early: it happens while there is still most of the stack left to run on the damage
    early = first_broken is not None and depth[first_broken] <= EARLY

    if cliff and depth[worst_at] <= EARLY:
        mode, repairable = "computation-collapse", False
        reason = (f"cosine falls {worst_drop:.2f} at layer {worst_at} "
                  f"({depth[worst_at]:.0%} depth) -- a component failed, and every layer after it "
                  f"is processing noise")
    elif early:
        mode, repairable = "computation-collapse", False
        reason = (f"agreement is already below {BROKEN} by layer {first_broken} "
                  f"({depth[first_broken]:.0%} depth) -- the signal is destroyed before the model "
                  f"has done most of its work")
    elif first_broken is not None:
        mode, repairable = "signal-degradation", True
        reason = (f"agreement holds until layer {first_broken} ({depth[first_broken]:.0%} depth) "
                  f"then erodes -- error accumulating with depth, not a component failing")
    else:
        mode, repairable = "healthy", True
        reason = f"every layer stays above {BROKEN} agreement (min {min(cosines):.3f})"

    return {
        "mode": mode, "repairable": repairable, "reason": reason,
        "layers": n, "min_cosine": round(min(cosines), 4),
        "first_broken_layer": first_broken,
        "worst_drop": round(worst_drop, 4), "worst_drop_layer": worst_at,
        "advice": _advice(mode),
    }


def _advice(mode):
    """What to actually do. A verdict without a next step is half an answer."""
    return {
        "healthy": ["Nothing to repair. Gate it (pollard-verify) and ship."],
        "signal-degradation": [
            "pollard-doctor --repair -- smoothing migrates the difficulty off the weights",
            "pollard-precondition -- measure WHICH preconditioner wins for this model first",
            "pollard-rotate -- incoherence preconditioning, the bigger lever at low bit-width",
        ],
        "computation-collapse": [
            "Repair will NOT recover this; do not spend a reconvert on it.",
            "pollard-errsrc -- find which tensor broke, and what triggered it",
            "Raise the bit-width on the broken layer (pollard-automap --protect), or pin it:",
            "  a collapsed layer is usually one tensor that needed imatrix coverage and lacked it",
            "pollard-fragile -- check whether that layer's tensors were flagged before the build",
        ],
        "unknown": ["Measure more layers, or widen the calibration set."],
    }[mode]


# --- capture -----------------------------------------------------------------------------------
def load_backbone(model_id, device="cpu"):
    """This tool's OWN loader -- model tooling does not depend on a shared or brain-side one."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    return m.to(device).eval()


def text_layers(model):
    """This tool's OWN layer walk. A VL model keeps its text stack under `model.language_model`."""
    for path in ("model.language_model", "language_model.model", "model"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "layers"):
                return obj.layers
        except AttributeError:
            continue
    raise SystemExit("could not find the transformer layers on this model")


def hidden_states(model, ids, device="cpu"):
    """Per-layer output for one batch of tokens."""
    import torch
    got = []
    hooks = [l.register_forward_hook(
        lambda _m, _i, o: got.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()))
        for l in text_layers(model)]
    with torch.no_grad():
        model(ids.unsqueeze(0).to(device))
    for h in hooks:
        h.remove()
    return got


def per_layer_cosine(ref_states, q_states):
    import torch
    out = []
    for a, b in zip(ref_states, q_states):
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        out.append(float(torch.nn.functional.cosine_similarity(a2, b2, dim=-1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--ref", help="reference (f16/bf16) model dir")
    ap.add_argument("--model", help="the quantized build to classify")
    ap.add_argument("--calib", help="text file to run through both")
    ap.add_argument("--cosines", help="comma-separated per-layer cosines, to classify without loading")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="write the verdict here as JSON")
    a = ap.parse_args()

    if a.cosines:
        cos = [float(x) for x in a.cosines.split(",")]
    elif a.ref and a.model and a.calib:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.ref)
        ids = tok(open(a.calib, encoding="utf-8", errors="ignore").read(),
                  return_tensors="pt").input_ids[0][:a.seqlen]
        ref = hidden_states(load_backbone(a.ref, a.device), ids, a.device)
        qnt = hidden_states(load_backbone(a.model, a.device), ids, a.device)
        cos = per_layer_cosine(ref, qnt)
    else:
        ap.error("need --ref, --model and --calib (or --cosines for a dry classification)")

    v = classify(cos)
    print(f"== pollard-failmode :: {v['layers']} layers")
    for i, c in enumerate(cos):
        bar = "#" * int(max(0.0, c) * 40)
        flag = "  <-- breaks here" if i == v["first_broken_layer"] else ""
        print(f"  layer {i:3d}  {c:6.3f}  {bar}{flag}")
    print(f"\n{v['mode'].upper()}  --  {v['reason']}")
    print(f"repairable: {v['repairable']}")
    for line in v["advice"]:
        print(f"  {line}")
    if a.out:
        json.dump({"verdict": v, "cosines": cos}, open(a.out, "w"), indent=2)
    raise SystemExit(0 if v["mode"] in ("healthy", "signal-degradation") else 1)


if __name__ == "__main__":
    main()
