#!/usr/bin/env python3
"""pollard-gptq — full-Hessian error-feedback quantization (GPTQ-class), our own impl.

The lever llama.cpp's imatrix CANNOT replicate. imatrix stores only the DIAGONAL of
the activation second moment (per-channel importance) and bends the rounding toward
hot channels. GPTQ uses the FULL Hessian H = X Xᵀ: it quantizes one column at a time
and pushes each column's rounding error into the not-yet-quantized columns via H⁻¹ —
cross-channel error compensation the diagonal can't express. That off-diagonal term is
the entire reason GPTQ beats round-to-nearest (and imatrix) at low bits.

This is a PyTorch/HF-side quantizer (needs real activations via forward hooks), which
is exactly the INT3/INT4-GPU world where AWQ/GPTQ live. It measures WikiText-2 PPL for
fp16 vs RTN vs our GPTQ at a target bit-width so we can see the error-feedback win, and
is the base to stack rotation + Pollard allocation on top of.

Usage:
  pollard-gptq --model hf_qwen05 --bits 4 --groupsize 128 --nsamples 128
"""
import argparse, sys, time
import torch, torch.nn as nn


def _hms(s):
    s = int(s); m, s = divmod(s, 60); return f"{m}m{s:02d}s" if m else f"{s}s"


def get_wikitext(tokenizer, split, seqlen, n=None):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    chunks = [enc[i:i + seqlen] for i in range(0, enc.numel() - seqlen, seqlen)]
    if n:
        chunks = chunks[:n]
    return torch.stack(chunks)


def quantize_group(w, scale, zero, maxq):
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def group_params(W, maxq):
    """asymmetric min/max scale+zero over a [rows, group] block."""
    wmax = W.amax(1, keepdim=True); wmin = W.amin(1, keepdim=True)
    scale = (wmax - wmin).clamp(min=1e-8) / maxq
    zero = torch.round(-wmin / scale)
    return scale, zero


def gptq_quantize(W, H, bits, groupsize, percdamp=0.01, act_order=False):
    """GPTQ on one linear weight W [rows, cols] with Hessian H [cols, cols].
    act_order: quantize columns in decreasing-Hessian-diagonal order (most
    important first) — recovers markedly more of RTN's loss. Returns dequantized
    weights (same shape)."""
    W = W.clone().float()
    rows, cols = W.shape
    maxq = 2 ** bits - 1
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0; W[:, dead] = 0.0
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]; H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    damp = percdamp * torch.mean(torch.diag(H)).clamp(min=1e-8)
    H[range(cols), range(cols)] += damp
    # H^-1, upper-Cholesky (GPTQ's stable column ordering)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    for i in range(cols):
        if groupsize and i % groupsize == 0:
            g = W[:, i:i + groupsize]
            scale, zero = group_params(g, maxq)
        d = Hinv[i, i]
        w = W[:, i]
        q = quantize_group(w.unsqueeze(1), scale, zero, maxq).squeeze(1)
        Q[:, i] = q
        err = (w - q) / d
        W[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
    if act_order:
        Q = Q[:, invperm]
    return Q.to(torch.float16)


def rtn_quantize(W, bits, groupsize):
    """Round-to-nearest baseline, same INT grid/groups, no error feedback."""
    W = W.clone().float(); rows, cols = W.shape; maxq = 2 ** bits - 1
    Q = torch.zeros_like(W)
    for i in range(0, cols, groupsize or cols):
        g = W[:, i:i + (groupsize or cols)]
        scale, zero = group_params(g, maxq)
        Q[:, i:i + g.shape[1]] = quantize_group(g, scale, zero, maxq)
    return Q.to(torch.float16)


@torch.no_grad()
def eval_ppl(model, testchunks, dev):
    model.eval(); nll = 0.0; ntok = 0
    for c in testchunks:
        ids = c.unsqueeze(0).to(dev)
        out = model(ids, labels=ids)
        nll += out.loss.float().item() * (ids.numel() - 1)
        ntok += ids.numel() - 1
    return torch.exp(torch.tensor(nll / ntok)).item()


def linear_layers(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def sequential_gptq(model, calib, dev, bits, groupsize, act_order):
    """The PROPER GPTQ: process transformer blocks in order, feeding each block's
    QUANTIZED outputs into the next block's Hessian — so every layer compensates for
    the error earlier layers actually introduced. Recovers far more of RTN's loss
    than the one-shot fp16-Hessian version."""
    layers = model.model.layers
    # --- capture the input to block 0 (+ the kwargs each block needs) for every sample.
    # A forward-PRE-hook avoids replacing the layer (so model-level attribute access like
    # `.attention_type` still works) and stops the pass right before block 0 runs.
    inps, cache = [], {}
    def catch(mod, args, kwargs):
        inps.append((args[0] if args else kwargs["hidden_states"]).detach())
        cache.update(kwargs)
        raise RuntimeError("caught")
    h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    for c in calib:
        try: model(c.unsqueeze(0).to(dev))
        except RuntimeError: pass
    h.remove()
    kw = {k: v for k, v in cache.items() if k != "hidden_states"}   # small, stays on dev
    inps = [x.cpu() for x in inps]                                    # park activations on CPU

    def empty():
        if dev == "mps": torch.mps.empty_cache()
        elif dev == "cuda": torch.cuda.empty_cache()

    for i, layer in enumerate(layers):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        H = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
        cnt = {n: 0 for n in lins}
        hooks = []
        def mk(n):
            def hook(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[n].add_(2.0 * x.t() @ x); cnt[n] += x.shape[0]
            return hook
        for n, m in lins.items():
            hooks.append(m.register_forward_hook(mk(n)))
        with torch.no_grad():
            for inp in inps:
                layer(inp.to(dev), **kw)
        for h in hooks: h.remove()
        for n, m in lins.items():
            m.weight.data = gptq_quantize(m.weight.data, H[n] / max(cnt[n], 1),
                                          bits, groupsize, act_order=act_order).to(dev)
            del H[n]
        empty()
        # re-run the now-QUANTIZED block to produce inputs for the next block (back to CPU)
        with torch.no_grad():
            inps = [layer(inp.to(dev), **kw)[0].cpu() for inp in inps]
        empty()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--eval-chunks", type=int, default=40)
    ap.add_argument("--method", default="all",
                    choices=["rtn", "gptq", "gptq-ao", "gptq-seq", "gptq-seq-ao", "both", "all"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"== pollard-gptq :: {a.model}  W{a.bits}g{a.groupsize}  dev={dev}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(dev)

    print("loading wikitext-2 ...")
    calib = get_wikitext(tok, "train", a.seqlen, a.nsamples)
    test = get_wikitext(tok, "test", a.seqlen, a.eval_chunks)

    ppl_fp16 = eval_ppl(model, test, dev)
    print(f"fp16 PPL: {ppl_fp16:.4f}")

    import copy
    fp16_state = copy.deepcopy(model.state_dict())

    def collect_hessians(lins):
        H = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
        cnt = {n: 0 for n in lins}
        hooks = []
        def mk(n):
            def hook(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[n].add_(2.0 * x.t() @ x); cnt[n] += x.shape[0]
            return hook
        for n, m in lins.items():
            hooks.append(m.register_forward_hook(mk(n)))
        with torch.no_grad():
            for c in calib:
                model(c.unsqueeze(0).to(dev))
        for h in hooks: h.remove()
        return {n: H[n] / max(cnt[n], 1) for n in lins}

    def run(method):
        model.load_state_dict(fp16_state)
        lins = linear_layers(model.model.layers)              # only the transformer-block linears
        t0 = time.time()
        if method in ("gptq-seq", "gptq-seq-ao"):
            sequential_gptq(model, calib, dev, a.bits, a.groupsize,
                            act_order=method.endswith("-ao"))
        elif method in ("gptq", "gptq-ao"):
            model.load_state_dict(fp16_state)                 # Hessians from the fp16 activations
            Hs = collect_hessians(lins)
            ao = (method == "gptq-ao")
            for n, m in lins.items():
                m.weight.data = gptq_quantize(m.weight.data, Hs[n], a.bits, a.groupsize,
                                              act_order=ao).to(dev)
        else:
            for n, m in lins.items():
                m.weight.data = rtn_quantize(m.weight.data, a.bits, a.groupsize).to(dev)
        ppl = eval_ppl(model, test, dev)
        gap = ppl - ppl_fp16
        print(f"{method.upper():8s} W{a.bits}g{a.groupsize} PPL: {ppl:.4f}   "
              f"(+{gap:.4f} vs fp16)   [{_hms(time.time()-t0)}]", flush=True)
        return ppl

    methods = {"rtn": ["rtn"], "gptq": ["gptq"], "gptq-ao": ["gptq-ao"],
               "gptq-seq": ["gptq-seq"], "gptq-seq-ao": ["gptq-seq-ao"],
               "both": ["rtn", "gptq"], "all": ["rtn", "gptq", "gptq-seq"]}[a.method]
    res = {}
    for meth in methods:
        res[meth] = run(meth)
    rtn_gap = res.get("rtn", ppl_fp16) - ppl_fp16
    print("\n== recovery of RTN's quantization loss (higher = better) ==", flush=True)
    for m in ("gptq", "gptq-ao", "gptq-seq", "gptq-seq-ao"):
        if m in res and rtn_gap > 0:
            rec = 100 * (res["rtn"] - res[m]) / rtn_gap
            print(f"  {m:8s}: PPL {res[m]:.4f}  (+{res[m]-ppl_fp16:.4f})  recovers {rec:.1f}% of RTN's loss")


if __name__ == "__main__":
    main()
