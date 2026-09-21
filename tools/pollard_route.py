#!/usr/bin/env python3
"""pollard-route -- record which MoE experts YOUR workload actually uses.

`pollard-experts` reads a routing capture and reports what ran hot. This produces that capture, for
any MoE on the Hub, from your own prompts -- so the whole expert-residency path works from a clean
install instead of needing a trace nobody outside the project could generate.

  pollard-route --model Qwen/Qwen3-30B-A3B --prompts my_workload.txt --out routing.jsonl
  pollard-route --model <hf-id> --prompts p.txt --gen 128 --out decode.jsonl   # DECODE routing
  pollard-experts --jsonl decode.jsonl --keep-frac 0.90 --out keeplist.json

Capture DECODE, not just prefill, if you are deciding what to keep resident. Prefill touches nearly
the whole pool whatever you feed it -- one domain lit up 97.6% of experts in our measurements -- so a
prefill-only capture will tell you there is no structure to exploit. Generation concentrates roughly
2x, and generation is where an agent spends its time. `--gen N` runs N tokens of greedy decode per
prompt and records the routing for those steps.

What this is for, said plainly: deciding what to keep NEAR THE COMPUTE, not what to delete. Routers
are load-balanced during training, so almost every expert fires for almost any workload -- "my model
does not need the coding experts" is not a thing the measurement supports. What it does support is
residency: keep the hot (layer, expert) pairs on the fast device, stream the cold ones. That is a
real win on a single GPU, and it does not change a single weight.

Rows are one JSON object per token per layer, which is what pollard-experts expects:

  {"prompt": 0, "pos": 12, "phase": "decode", "layer": 3, "experts": [17, 52]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import torch
except ImportError as _e:
    raise SystemExit("pollard-route needs torch and transformers: "
                     "pip install 'pollard-weights[convert]'") from None


def find_routers(model, n_experts=None):
    """Every MoE router in the model, as (layer, name, module, n_experts).

    Matched by SHAPE, not by class or by a list of names. A router is whatever module holds a
    (num_experts x hidden) weight and sits at a `gate`/`router` attribute -- that is true of a plain
    nn.Linear on some families and of a custom class on others (transformers 5.x wraps Mixtral's in
    MixtralTopKRouter, which is NOT an nn.Linear, so an isinstance check finds nothing at all). A
    name list goes stale with the next architecture; the weight shape does not.
    """
    found = []
    for name, mod in model.named_modules():
        tail = name.rsplit(".", 1)[-1]
        if tail not in ("gate", "router", "gate_proj_router", "router_gate"):
            continue
        if "experts" in name:                    # an expert's own projection, not the router
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            continue
        if n_experts and w.shape[0] != n_experts:
            continue
        layer = next((int(x) for x in name.split(".") if x.isdigit()), -1)
        found.append((layer, name, mod, w.shape[0]))
    return found


def capture(model, tok, prompts, gen, device, out_path, top_k=None):
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    k = top_k or getattr(text_cfg, "num_experts_per_tok", None) or \
        getattr(text_cfg, "num_selected_experts", None) or 2
    n_exp_cfg = getattr(text_cfg, 'num_local_experts', None) or \
        getattr(text_cfg, 'num_experts', None)
    routers = find_routers(model, n_exp_cfg)
    if not routers:
        raise SystemExit("no MoE routers found -- is this a dense model? "
                         "pollard-experts only applies to mixture-of-experts.")
    print(f"  {len(routers)} routers, {routers[0][3]} experts each, top-{k} per token", flush=True)

    state = {"prompt": 0, "phase": "prefill", "base": 0}
    rows = []

    n_exp = routers[0][3]

    def selected(out):
        """The chosen expert ids for each token, however this router reports them.

        Some routers hand back the indices directly (transformers 5.x returns
        (scores, top_scores, indices)); others return only logits. Prefer real indices when they are
        there -- recomputing topk from logits can disagree with what the model actually ran when the
        router applies its own normalisation or jitter first.
        """
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:                                   # an integer tensor IS the selection
            if torch.is_tensor(o) and not o.is_floating_point() and o.dim() >= 2:
                return o.reshape(-1, o.shape[-1])
        for o in outs:                                   # otherwise score and take the top k
            if torch.is_tensor(o) and o.is_floating_point() and o.shape[-1] == n_exp:
                lg = o.reshape(-1, n_exp).float()
                return lg.topk(min(k, n_exp), dim=-1).indices
        return None

    def make_hook(layer):
        def hook(_mod, _inp, out):
            idx = selected(out)
            if idx is None:
                return out
            for t in range(idx.shape[0]):
                rows.append({"prompt": state["prompt"], "pos": state["base"] + t,
                             "phase": state["phase"], "layer": layer,
                             "experts": idx[t].tolist()})
            return out
        return hook

    handles = [mod.register_forward_hook(make_hook(layer)) for layer, _n, mod, _e in routers]
    try:
        with torch.no_grad():
            for pi, text in enumerate(prompts):
                state["prompt"], state["phase"], state["base"] = pi, "prefill", 0
                ids = tok(text, return_tensors="pt").input_ids.to(device)
                o = model(ids, use_cache=True)
                past, nprefill = o.past_key_values, ids.shape[1]
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                for step in range(gen):
                    state["phase"], state["base"] = "decode", nprefill + step
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                print(f"    prompt {pi+1}/{len(prompts)}  {nprefill} prefill + {gen} decode "
                      f"-> {len(rows):,} rows", flush=True)
    finally:
        for h in handles:
            h.remove()

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dec = sum(1 for r in rows if r["phase"] == "decode")
    print(f"\n  wrote {out_path}: {len(rows):,} rows ({dec:,} decode, {len(rows)-dec:,} prefill)")
    print(f"  pollard-experts --jsonl {out_path} --keep-frac 0.90 --out keeplist.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF MoE model id or directory")
    ap.add_argument("--prompts", required=True, help="your workload, one prompt per line")
    ap.add_argument("--out", default="routing.jsonl", help="capture file for pollard-experts")
    ap.add_argument("--gen", type=int, default=64,
                    help="decode tokens per prompt (default 64; 0 = prefill only, which will "
                         "understate concentration)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N prompts")
    ap.add_argument("--top-k", type=int, default=0, help="override experts-per-token")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not os.path.isfile(a.prompts):
        raise SystemExit(f"no such prompts file: {a.prompts}")
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    if a.limit:
        prompts = prompts[:a.limit]
    if not prompts:
        raise SystemExit("the prompts file is empty")

    dt = {"auto": "auto", "float16": torch.float16,
          "bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    capture(model, tok, prompts, a.gen, a.device, a.out, a.top_k or None)


if __name__ == "__main__":
    main()
