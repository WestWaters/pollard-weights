#!/usr/bin/env python3
"""pollard-routecheck -- did quantization change WHICH experts fire?

Every other quantization metric assumes a fixed computation graph. A MoE does not have one: the
router picks a small subset of experts per token, and if quantization shifts that choice the model
is running different weights than the one you measured. Perplexity can absorb a surprising amount
of that and still look fine.

The trap is that matching router LOGITS is not enough. Top-k only cares about ORDER, so a tiny
shift near the boundary between the k-th selected expert and the (k+1)-th unselected one swaps an
expert while the logits barely move. What has to be preserved is the MARGIN at that boundary.

What this measures, per layer:

    top1_swap      how often the single most-weighted expert changed
    topk_swap      how often the selected SET changed at all
    margin_p05     the 5th-percentile boundary margin at reference -- how much room there was
    margin_drop    how much of that margin quantization ate
    at_risk        tokens sitting close enough to the boundary to flip on noise

    pollard-routecheck --ref f16-model --model quantized-model --calib calib.txt
    pollard-routecheck --logits ref.npz,quant.npz          # compare captures you already have

A swap rate near zero means the quantized model routes like the reference. A high one means the
number on your card was measured on a different set of experts than the model will actually use.

MoE-only by nature: a dense model has no router, so it exits 0 saying there is nothing to compare.
That is not a limitation on dense models -- every other Pollard gate applies to them unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


# --- the metrics. Pure numpy so they are testable without loading a model. --------------------
def route_stats(ref_logits, q_logits, top_k, at_risk_eps=0.05):
    """Compare two [tokens, experts] router-logit matrices from the SAME layer and tokens.

    Returns the routing-consistency numbers. Logits, not probabilities: softmax is monotonic, so
    top-k is identical either way, and logits keep the margins on a scale that does not collapse
    when one expert dominates.
    """
    import numpy as np

    ref = np.asarray(ref_logits, dtype=np.float64)
    qnt = np.asarray(q_logits, dtype=np.float64)
    if ref.shape != qnt.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs quantized {qnt.shape}")
    n_tok, n_exp = ref.shape
    k = min(top_k, n_exp)

    ref_order = np.argsort(-ref, axis=1)
    q_order = np.argsort(-qnt, axis=1)
    ref_top, q_top = ref_order[:, :k], q_order[:, :k]

    top1_swap = float((ref_top[:, 0] != q_top[:, 0]).mean())
    # the SET, not the order: a reordering inside the selection is not a swap
    swapped = np.array([len(set(a.tolist()) ^ set(b.tolist())) > 0 for a, b in zip(ref_top, q_top)])
    topk_swap = float(swapped.mean())
    # how many selected experts changed, averaged -- a set that loses 3 of 8 is worse than 1 of 8
    churn = float(np.mean([len(set(a.tolist()) - set(b.tolist())) for a, b in zip(ref_top, q_top)]))

    def boundary_margin(logits, order):
        """Gap between the last SELECTED expert and the first UNSELECTED one."""
        if n_exp <= k:
            return np.zeros(n_tok)
        rows = np.arange(n_tok)
        return logits[rows, order[:, k - 1]] - logits[rows, order[:, k]]

    m_ref = boundary_margin(ref, ref_order)
    # the same experts, scored by the quantized router -- how much room is left at that boundary
    rows = np.arange(n_tok)
    m_q = (qnt[rows, ref_order[:, k - 1]] - qnt[rows, ref_order[:, k]]) if n_exp > k \
        else np.zeros(n_tok)

    scale = float(np.median(np.abs(ref))) or 1.0
    return {
        "tokens": int(n_tok), "experts": int(n_exp), "top_k": int(k),
        "top1_swap": round(top1_swap, 5),
        "topk_swap": round(topk_swap, 5),
        "experts_changed_per_token": round(churn, 4),
        "margin_p05": round(float(np.percentile(m_ref, 5)), 5),
        "margin_median": round(float(np.median(m_ref)), 5),
        "margin_drop": round(float(np.median(m_ref - m_q)), 5),
        # tokens whose reference margin is small relative to the logit scale: these are the ones
        # that flip on noise, and the count is a property of the MODEL, not of the quantization
        "at_risk": round(float((m_ref < at_risk_eps * scale).mean()), 5),
    }


def verdict(per_layer, swap_budget=0.02):
    """One pass/fail a build can be gated on, plus the layers that caused it."""
    if not per_layer:
        return {"pass": False, "reason": "no MoE layers measured", "layers": 0}
    worst = max(per_layer, key=lambda r: r["topk_swap"])
    mean_swap = sum(r["topk_swap"] for r in per_layer) / len(per_layer)
    bad = [r for r in per_layer if r["topk_swap"] > swap_budget]
    return {
        "pass": not bad,
        "layers": len(per_layer),
        "mean_topk_swap": round(mean_swap, 5),
        "worst_layer": worst.get("layer"),
        "worst_topk_swap": worst["topk_swap"],
        "over_budget": [r.get("layer") for r in bad],
        "budget": swap_budget,
        "reason": ("routing preserved" if not bad else
                   f"{len(bad)}/{len(per_layer)} layers route differently "
                   f"(worst {worst.get('layer')}: {worst['topk_swap']:.1%} of tokens)"),
    }


# --- capture ----------------------------------------------------------------------------------
ROUTER_HINTS = ("block_sparse_moe.gate", "mlp.gate", "ffn_gate_inp", "router", ".gate_proj_router")


def load_backbone(model_id, device="cpu"):
    """This tool's OWN loader -- model tooling does not depend on a shared or brain-side one."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                             trust_remote_code=False)
    return m.to(device).eval()


def routers(model):
    """Every router module, by name. A router is a Linear whose output width is the expert count."""
    import torch.nn as nn
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(h in name for h in ROUTER_HINTS):
            out[name] = mod
    return out


def capture(model, calib_ids, device="cpu"):
    """Router logits for every MoE layer over the calibration tokens."""
    import torch
    got: dict[str, list] = {}
    hooks = []

    def make(name):
        def hook(_m, _i, o):
            got.setdefault(name, []).append(o.detach().float().reshape(-1, o.shape[-1]).cpu())
        return hook

    rs = routers(model)
    if not rs:
        return {}
    for name, mod in rs.items():
        hooks.append(mod.register_forward_hook(make(name)))
    with torch.no_grad():
        for ids in calib_ids:
            model(ids.unsqueeze(0).to(device))
    for h in hooks:
        h.remove()
    return {n: torch.cat(v).numpy() for n, v in got.items()}


def _calib(model_id, path, n, seqlen):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    text = open(path, encoding="utf-8", errors="ignore").read()
    import torch
    enc = tok(text, return_tensors="pt").input_ids[0]
    return [enc[i:i + seqlen] for i in range(0, max(1, enc.numel() - seqlen), seqlen)][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--ref", help="reference (f16/bf16) model dir")
    ap.add_argument("--model", help="quantized model dir to check against it")
    ap.add_argument("--logits", help="ref.npz,quant.npz -- compare captures instead of loading models")
    ap.add_argument("--calib", help="calibration text file")
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=0, help="experts per token (default: read the config)")
    ap.add_argument("--swap-budget", type=float, default=0.02,
                    help="fail above this fraction of tokens routing differently (default 2%%)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="write the full per-layer report here as JSON")
    a = ap.parse_args()

    if a.logits:
        import numpy as np
        rp, qp = a.logits.split(",")
        ref, qnt = np.load(rp), np.load(qp)
        names = [n for n in ref.files if n in qnt.files]
    elif a.ref and a.model and a.calib:
        ref_model = load_backbone(a.ref, a.device)
        if not routers(ref_model):
            # Not a refusal and not a failure. A dense model has no router, so there is no
            # routing to preserve -- every other Pollard gate applies to it exactly as usual.
            print("== pollard-routecheck :: dense model, no routers to compare.")
            print("   Nothing to check here. Routing consistency is a MoE-only question;")
            print("   run pollard-bench / pollard-eval / pollard-kl for this build as normal.")
            raise SystemExit(0)
        top_k = a.top_k or int(getattr(ref_model.config, "num_experts_per_tok", 0) or 2)
        ids = _calib(a.ref, a.calib, a.nsamples, a.seqlen)
        ref = capture(ref_model, ids, a.device)
        del ref_model
        qnt = capture(load_backbone(a.model, a.device), ids, a.device)
        names = [n for n in ref if n in qnt]
    else:
        ap.error("need --ref, --model and --calib (or --logits ref.npz,quant.npz)")

    top_k = a.top_k or 2
    rows = []
    for n in sorted(names):
        st = route_stats(ref[n], qnt[n], top_k, )
        st["layer"] = n
        rows.append(st)

    v = verdict(rows, a.swap_budget)
    print(f"== pollard-routecheck :: {len(rows)} MoE layers, top-{top_k}")
    print(f"{'layer':<44} {'top1':>7} {'topk':>7} {'churn':>7} {'margin':>8} {'drop':>8} {'risk':>7}")
    for r in rows:
        print(f"{r['layer'][:44]:<44} {r['top1_swap']:7.2%} {r['topk_swap']:7.2%} "
              f"{r['experts_changed_per_token']:7.3f} {r['margin_median']:8.4f} "
              f"{r['margin_drop']:8.4f} {r['at_risk']:7.2%}")
    print(f"\n{'PASS' if v['pass'] else 'FAIL'} -- {v['reason']}")
    if a.out:
        json.dump({"verdict": v, "layers": rows}, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    raise SystemExit(0 if v["pass"] else 1)


if __name__ == "__main__":
    main()
