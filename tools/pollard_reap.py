#!/usr/bin/env python3
"""pollard-reap -- REAP expert saliency for a MoE: which routed experts carry the output, measured on the
calibration corpus, so pollard-prune drops the ones that don't.

REAP (Router-weighted Expert Activation Pruning, Cerebras 2025) scores expert j by the mean, over the tokens the
router sends to j, of the router weight times the norm of j's output:

    S_j = mean_{x routed to j}  g_j(x) * || f_j(x) ||_2

An expert that fires often but contributes little (small g, small output) scores low; a rarely-picked expert that
dominates when it fires scores high. Frequency alone (the imatrix token counts) cannot see that difference.

    pollard-reap --model ./OLMoE-1B-7B --calib calib.txt --out olmoe.reap.json
    pollard-prune --gguf olmoe-f16.gguf --keep 0.5 --score reap --scores olmoe.reap.json --out olmoe-p50-f16.gguf

How it measures: one forward pass over the calibration rows with a pre-hook on every fused experts module
(transformers >= 5: `Experts.forward(hidden_states, top_k_index, top_k_weights)` with `gate_up_proj` [E, 2I, H] /
`down_proj` [E, H, I], the layout 17+ MoE families share -- Granite-MoE, OLMoE, Qwen2/3-MoE, Mixtral, DeepSeek...).
The hook recomputes each routed expert's output from the module's own weights and activation and accumulates
g * ||f||, the token count and the summed ||f|| (EAN) per expert. The forward itself is untouched. Expert order is
the checkpoint's, which is also llama.cpp's ffn_*_exps order, so the scores apply to the GGUF directly.

Needs torch + transformers (pip install "pollard-weights[exlr8]"). CPU is fine for a small MoE; --device mps/cuda.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

FORMAT = "pollard-reap"


def _layer_index(name):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name + ".")
    return int(m.group(1)) if m else None


def is_experts_module(mod):
    """a fused experts module: 3-D down_proj plus a 3-D gate_up_proj (gated, SwiGLU) or up_proj (gateless, e.g.
    Nemotron-H's relu2 experts)."""
    dn = getattr(mod, "down_proj", None)
    up = getattr(mod, "gate_up_proj", None)
    if up is None:
        up = getattr(mod, "up_proj", None)
    return getattr(dn, "ndim", 0) == 3 and getattr(up, "ndim", 0) == 3


def find_expert_modules(model, layer=None):
    """[(layer, module)] for every fused experts module. With `layer`, `model` is that one decoder block (the
    streamed path) and every experts module inside it is that layer's."""
    out = []
    for name, mod in model.named_modules():
        if not is_experts_module(mod):
            continue
        L = layer if layer is not None else _layer_index(name)
        if L is not None:
            out.append((L, mod))
    return out


def _n_experts(mod):
    return int(mod.down_proj.shape[0])


def expert_outputs(mod, x, j):
    """f_j(x) for the rows x routed to expert j, from the module's own weights and activation."""
    import torch
    D = mod.down_proj[j]
    H = x.shape[-1]
    act = getattr(mod, "act_fn", None) or torch.nn.functional.silu
    gu = getattr(mod, "gate_up_proj", None)
    if gu is not None:
        W = gu[j]
        y = x @ W.T if W.shape[-1] == H else x @ W                    # [E, 2I, H] or transposed [E, H, 2I]
        gate, up = y.chunk(2, dim=-1)
        h = act(gate) * up
    else:
        W = mod.up_proj[j]
        h = act(x @ W.T if W.shape[-1] == H else x @ W)
    return h @ D.T if D.shape[-1] == h.shape[-1] else h @ D


class ReapAccumulator:
    """per (layer, expert): sum of g * ||f||, token count, sum of ||f||."""

    def __init__(self, n_experts_by_layer):
        self.reap, self.count, self.ean = {}, {}, {}
        for L, E in n_experts_by_layer.items():
            self.ensure(L, E)

    def ensure(self, L, E):
        import torch
        for d in (self.reap, self.count, self.ean):
            d.setdefault(L, torch.zeros(E, dtype=torch.float64))

    def add(self, L, mod, hidden, top_k_index, top_k_weights):
        import torch
        with torch.no_grad():
            x = hidden.reshape(-1, hidden.shape[-1])
            idx = top_k_index.reshape(x.shape[0], -1)
            w = top_k_weights.reshape(x.shape[0], -1).to(torch.float64)
            E = self.reap[L].numel()
            for j in torch.unique(idx).tolist():
                if j < 0 or j >= E:
                    continue
                tok, pos = torch.where(idx == j)
                f = expert_outputs(mod, x[tok], j).to(torch.float64)
                n = f.norm(dim=-1)
                self.reap[L][j] += (w[tok, pos] * n).sum().cpu()
                self.count[L][j] += float(tok.numel())
                self.ean[L][j] += n.sum().cpu()

    def result(self):
        out = {}
        for L in sorted(self.reap):
            c = self.count[L]
            out[str(L)] = {"reap": (self.reap[L] / c.clamp_min(1.0)).tolist(), "count": c.tolist(),
                           "ean": self.ean[L].tolist()}
        return out


def measure(model, rows, device="cpu", log=print):
    """run rows (list of 1-D LongTensors) through the model with the REAP hooks; returns {layer: scores}."""
    import torch
    mods = find_expert_modules(model)
    if not mods:
        raise SystemExit("no fused experts module found (gate_up_proj/down_proj 3-D) -- not a transformers>=5 MoE, "
                         "or an architecture whose experts are stored differently")
    acc = ReapAccumulator({L: _n_experts(m) for L, m in mods})
    hooks = _hook(mods, acc)
    try:
        with torch.no_grad():
            for i, ids in enumerate(rows):
                model(ids.unsqueeze(0).to(device))
                if log and (i + 1) % 16 == 0:
                    log(f"  {i + 1}/{len(rows)} rows")
    finally:
        for h in hooks:
            h.remove()
    return acc.result()


def _hook(mods, acc):
    hooks = []
    for L, m in mods:
        def pre(mod, args, kwargs, _L=L):
            hs = args[0] if args else kwargs.get("hidden_states")
            ti = args[1] if len(args) > 1 else kwargs.get("top_k_index")
            tw = args[2] if len(args) > 2 else kwargs.get("top_k_weights")
            if hs is not None and ti is not None and tw is not None:
                acc.add(_L, mod, hs, ti, tw)
        hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))
    return hooks


def measure_streamed(model_dir, rows, device="cpu", trust_remote_code="auto", max_ram_gb=None, log=print):
    """REAP through pollard-stream: the calibration rows go through ONE decoder block at a time (weights read from
    disk, block freed after), so a MoE bigger than RAM -- Bolt's 150 GB bf16 on a 128 GB Spark -- is scored too.
    rows: lists of token ids."""
    import pollard_stream as S
    device = S.stream_device(device)
    bb = S.Backbone(model_dir, trust_remote_code, log=log)
    plan = S.fit_memory(bb, max(1, len(rows)), max(len(r) for r in rows), S.ram_budget(max_ram_gb), log=log)
    rows = rows[:plan["nsamples"]]
    bb.dtype = S.plan_dtype(plan)
    eng = S.Engine(bb, device=device, log=log)
    eng.record(rows)
    acc, n = None, {}
    for L in range(bb.n_layers):
        eng.load_layer(L)
        mods = find_expert_modules(bb.layers[L], layer=L)
        hooks = []
        if mods:
            n.update({L: _n_experts(m) for _, m in mods})
            acc = acc or ReapAccumulator({})
            for _, m in mods:
                acc.ensure(L, _n_experts(m))
            hooks = _hook(mods, acc)
        try:
            eng.hidden = eng.run_layer(L)
        finally:
            for h in hooks:
                h.remove()
        bb.free_layer(L)
        if log:
            log(f"  block {L + 1}/{bb.n_layers}" + (f"  experts {n[L]}" if L in n else ""))
    if acc is None:
        raise SystemExit("no fused experts module in any block -- not a MoE this engine can score")
    return acc.result()


def calib_rows(tok, path, n_rows=64, seqlen=512):
    import torch
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    rows = [ids[i:i + seqlen] for i in range(0, max(1, ids.numel() - seqlen), seqlen)][:n_rows]
    return [r for r in rows if r.numel() > 16] or [ids[:seqlen]]


def load_scores(path, key="reap"):
    """{layer: [score per expert]} from a pollard-reap JSON (key reap | count | ean)."""
    d = json.load(open(path))
    if d.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return {int(L): v[key] for L, v in d["layers"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF checkpoint dir of the MoE")
    ap.add_argument("--calib", required=True, help="calibration text (pollard-calib)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="one decoder block at a time through pollard-stream (a MoE bigger than RAM)")
    ap.add_argument("--max-ram-gb", type=float, help="--stream: cap the RAM the engine plans for")
    a = ap.parse_args(argv)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit('pollard-reap needs torch + transformers: pip install "pollard-weights[exlr8]"')
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    rows = calib_rows(tok, a.calib, a.rows, a.seqlen)
    print(f"== pollard-reap :: {a.model}  {len(rows)} rows x <= {a.seqlen} tokens  device {a.device}"
          + ("  (streamed)" if a.stream else ""))
    if a.stream:
        layers = measure_streamed(a.model, [r.tolist() for r in rows], a.device,
                                  "on" if a.trust_remote_code else "auto", a.max_ram_gb)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                     trust_remote_code=a.trust_remote_code).to(a.device).eval()
        layers = measure(model, rows, a.device)
    json.dump({"format": FORMAT, "model": os.path.abspath(a.model), "calib": os.path.abspath(a.calib),
               "rows": len(rows), "seqlen": a.seqlen, "layers": layers}, open(a.out, "w"))
    for L, v in list(layers.items())[:3]:
        r = sorted(v["reap"], reverse=True)
        print(f"  layer {L}: reap max {r[0]:.4f}  median {r[len(r) // 2]:.4f}  min {r[-1]:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
