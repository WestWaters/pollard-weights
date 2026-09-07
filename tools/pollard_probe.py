#!/usr/bin/env python3
"""pollard-probe — the CHEAP, any-box sensitivity profile. Same output as
pollard-sensitivity, a fraction of the cost, so measured allocation runs on a
laptop instead of needing a big-GPU GGUF sweep.

pollard-sensitivity is the ground truth: it CRUSHES each tensor group to a real
GGUF quant and runs a full perplexity KL pass — 2·layers subprocess builds + KL
evals. Accurate, but heavy and GPU-hungry. This does the same measurement the
cheap way EXL3 does it: perturb one group at a time IN-PROCESS (torch), read the
logit-KL directly, restore. No GGUF, no imatrix, no separate eval binary — one
forward pass per group. The RANKING is what the allocator needs, and injected
quant-error ranks the groups the same way for a tiny cost.

Emits the identical `sensitivity.json` schema pollard-fit consumes:
  {"ffn": {"<layer>": kl_cost}, "attn": {...}, "noise": {type: uniform_kl}, ...}

    pollard-probe --model <hf-dir-or-id> --eval held-out.txt --out model.sensitivity.json
    pollard-fit --gguf model-f16.gguf --ram 16 --sensitivity model.sensitivity.json

Note: this is the torch/RTN proxy for the GGUF crush — the per-group ranking
matches; absolute KL is a proxy, not the ik_llama trellis error. For the final
published card, confirm the winner with a pollard-sensitivity run on the box.
"""
import argparse, json, sys
import torch, torch.nn.functional as F

# LADDER types -> (bpw, RTN bits) so the cheap noise curve keys match pollard-fit.
LADDER_BITS = [("q6_K", 6), ("q5_K", 5), ("iq4_xs", 4), ("iq3_s", 3),
               ("iq2_s", 2), ("iq2_xxs", 2)]
GROUP_ATTR = {"ffn": ("mlp", ("gate_proj", "up_proj", "down_proj")),
              "attn": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))}


def _chunks(tok, text, seqlen, n):
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (len(ids) - seqlen) // max(n, 1)) if len(ids) > seqlen else seqlen
    out = [ids[i:i + seqlen] for i in range(0, max(1, len(ids) - seqlen + 1), step)][:n]
    return [c for c in out if len(c) >= 8] or [ids[:seqlen]]


@torch.no_grad()
def _rtn(W, bits, gs=64):
    """Per-row absmax symmetric RTN to `bits`, group size gs — the actual quant
    error we perturb with (deterministic, cheap). Returns the quantized weight."""
    if bits >= 16:
        return W
    q = 2 ** (bits - 1) - 1 or 1
    out, D = W.float(), W.shape[1]
    r = out.reshape(out.shape[0], -1, min(gs, D)) if D % min(gs, D) == 0 else out.unsqueeze(1)
    scale = r.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
    r = (r / scale).round().clamp(-q - 1, q) * scale
    return r.reshape_as(W).to(W.dtype)


@torch.no_grad()
def _logits(model, chunks, dev):
    return [model(c.unsqueeze(0).to(dev)).logits[0, :-1].float().log_softmax(-1) for c in chunks]


@torch.no_grad()
def _kl_vs(model, chunks, ref_logp, dev):
    """Mean KL(clean || perturbed) over the eval chunks."""
    tot = ntok = 0.0
    for c, lp0 in zip(chunks, ref_logp):
        lp1 = model(c.unsqueeze(0).to(dev)).logits[0, :-1].float().log_softmax(-1)
        p0 = lp0.exp()
        tot += (p0 * (lp0 - lp1)).sum(-1).sum().item()
        ntok += lp0.size(0)
    return tot / max(ntok, 1)


def _linears(model, layer, group):
    parent, names = GROUP_ATTR[group]
    mod = getattr(model.model.layers[layer], parent)
    return [getattr(mod, n) for n in names if hasattr(mod, n)]


@torch.no_grad()
def _stream_sensitivity(model, chunks, dev, groups, layers, probe_bits, ladder_bits):
    """ONE forward pass over the calib set, sensitivity for EVERY group at once — for models where
    the perturb+KL loop (layers×groups full passes) is infeasible (744B over 1.5TB).

    For a linear y=Wx, the expected output error from quantizing W->Ŵ is
        E‖(W-Ŵ)x‖² = Σ_ij (W-Ŵ)_ij² · E[x_j²]
    i.e. the squared quant error weighted by the Hessian DIAGONAL h_j = E[x_j²]. We accumulate h_j
    per target linear with hooks in a single pass (no per-group forward), then score each group as
    Σ ΔW² · h. Same ranking the perturb+KL probe gives, at a fraction of the cost."""
    h, cnt, index = {}, {}, {}
    hooks = []

    def mk(lid):
        def hook(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            s = (x * x).sum(0)
            h[lid] = s if lid not in h else h[lid] + s
            cnt[lid] = cnt.get(lid, 0) + x.shape[0]
        return hook

    for i in range(layers):
        for g in groups:
            for lin in _linears(model, i, g):
                lid = id(lin); index[lid] = (g, i)
                hooks.append(lin.register_forward_hook(mk(lid)))
    for c in chunks:
        model(c.unsqueeze(0).to(dev))
    for hk in hooks:
        hk.remove()

    def cost_at(bits):
        cost = {g: {} for g in groups}
        for i in range(layers):
            for g in groups:
                tot = 0.0
                for lin in _linears(model, i, g):
                    lid = id(lin)
                    if lid not in h or cnt.get(lid, 0) == 0:      # module never fired (unused/pruned expert)
                        continue
                    hj = (h[lid] / cnt[lid]).to(lin.weight.device)   # E[x_j²]
                    dW = lin.weight.data.float() - _rtn(lin.weight.data, bits).float()
                    tot += float((dW * dW * hj.unsqueeze(0)).sum().item())
                cost[g][str(i)] = tot
        return cost

    profile = cost_at(probe_bits)
    noise = {t: sum(sum(cost_at(bits)[g].values()) for g in groups) for t, bits in ladder_bits}
    return profile, noise


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True, help="held-out text (disjoint from any calib)")
    ap.add_argument("--out", help="profile path (default <model>.sensitivity.json)")
    ap.add_argument("--groups", default="ffn,attn")
    ap.add_argument("--probe-bits", type=int, default=2, help="RTN bits to crush a group to (default 2)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stream", action="store_true",
                    help="ONE-pass Hessian-diagonal estimator instead of per-group perturb+KL — for "
                         "models too big to run layers×groups forward passes (744B-scale)")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    print(f"== pollard-probe :: {a.model}  probe={a.probe_bits}bit  dev={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    layers = len(model.model.layers)
    ch = _chunks(tok, open(a.eval, encoding="utf-8").read(), a.seqlen, a.chunks)

    if a.stream:
        print(f"  {layers} layers, {len(ch)} calib chunks — one-pass Hessian-diagonal estimator", flush=True)
        profile, noise = _stream_sensitivity(model, ch, dev, groups, layers, a.probe_bits, LADDER_BITS)
        method = "pollard-probe stream (Hessian-diagonal proxy)"
        for g in groups:
            vals = list(profile[g].values())
            print(f"  {g}: streamed {len(vals)} layers", flush=True)
    else:
        method = "pollard-probe (torch RTN proxy)"
        ref = _logits(model, ch, dev)
        print(f"  {layers} layers, {len(ch)} eval chunks, clean logits cached", flush=True)

        # per-model noise curve: RTN-crush EVERY group to each rung, measure KL. Cheap
        # (len(LADDER) forwards) and it's what lets the allocator see catastrophic rungs.
        noise = {}
        saved_all = {}
        for t, bits in LADDER_BITS:
            for i in range(layers):
                for g in groups:
                    for lin in _linears(model, i, g):
                        saved_all.setdefault(id(lin), lin.weight.data.clone())
                        lin.weight.data = _rtn(saved_all[id(lin)], bits)
            noise[t] = _kl_vs(model, ch, ref, dev)
            for i in range(layers):                                  # restore
                for g in groups:
                    for lin in _linears(model, i, g):
                        lin.weight.data = saved_all[id(lin)].clone()
            print(f"  noise {t:8} ({bits}b): KL={noise[t]:.4f}", flush=True)

        # per-(group,layer) sensitivity: crush ONE group at ONE layer, measure KL hit.
        profile = {g: {} for g in groups}
        for i in range(layers):
            row = []
            for g in groups:
                lins = _linears(model, i, g)
                saved = [l.weight.data.clone() for l in lins]
                for l in lins:
                    l.weight.data = _rtn(l.weight.data, a.probe_bits)
                profile[g][str(i)] = _kl_vs(model, ch, ref, dev)
                for l, w in zip(lins, saved):
                    l.weight.data = w
                row.append(f"{g}={profile[g][str(i)]:.4f}")
            print(f"  layer {i:>3}/{layers}  " + "  ".join(row), flush=True)

    if a.out:
        out = a.out
    else:
        try:
            import pollard_workspace as ws, os
            out = os.path.join(ws.calibration_dir(a.model, create=True),
                               ws.model_basename(a.model) + ".sensitivity.json")
        except Exception:
            out = a.model.rstrip("/").split("/")[-1] + ".sensitivity.json"
    payload = {**profile, "noise": noise, "probe": f"rtn{a.probe_bits}",
               "layers": layers, "source": a.model, "method": method}
    json.dump(payload, open(out, "w"), indent=2)
    for g in groups:
        vals = list(profile[g].values())
        lo, hi = min(vals), max(vals)
        print(f"  {g}: spread {hi/max(lo,1e-9):.1f}x  (min {lo:.4f}  max {hi:.4f})", flush=True)
    print(f"\ndone: {out}\n  feed it:  pollard-fit --gguf <f16>.gguf --ram <GB> --sensitivity {out}", flush=True)


if __name__ == "__main__":
    main()
