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


def _chunks(tokenizer, text, seqlen, n=None):
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = [enc[i:i + seqlen] for i in range(0, enc.numel() - seqlen, seqlen)]
    if n:
        chunks = chunks[:n]
    return torch.stack(chunks)


def get_wikitext(tokenizer, split, seqlen, n=None, path=None):
    """Tokenize into seqlen chunks. From a local text FILE if `path` is given
    (robust across datasets-library versions); else the wikitext-2 HF dataset."""
    if path:
        with open(path, encoding="utf-8") as f:
            return _chunks(tokenizer, f.read(), seqlen, n)
    from datasets import load_dataset
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split=split)
            return _chunks(tokenizer, "\n\n".join(ds["text"]), seqlen, n)
        except Exception:
            continue
    raise RuntimeError("could not load wikitext — pass --calib-file/--eval-file instead")


def quantize_group(w, scale, zero, maxq):
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def group_params(W, maxq):
    """asymmetric min/max scale+zero over a [rows, group] block."""
    wmax = W.amax(1, keepdim=True); wmin = W.amin(1, keepdim=True)
    scale = (wmax - wmin).clamp(min=1e-8) / maxq
    zero = torch.round(-wmin / scale)
    return scale, zero


def _col_quant(w, scale, zero, maxq, qmode):
    """Quantize one column w [rows] given the group's scale/zero. Returns dequant [rows].
    The GPTQ error-feedback loop is quantizer-agnostic — this is the only per-alphabet
    part, so ternary/binary get the SAME cross-channel compensation as INT."""
    if qmode == "int":
        return quantize_group(w.unsqueeze(1), scale, zero, maxq).squeeze(1)
    s = scale[:, 0]                                          # per-row group scale [rows]
    if qmode == "ternary":
        return torch.clamp(torch.round(w / s), -1, 1) * s    # {-1,0,+1}
    if qmode == "binary":
        return torch.sign(w) * s                             # {-1,+1}
    raise ValueError(qmode)


def gptq_quantize(W, H, bits, groupsize, percdamp=0.01, act_order=False, qmode="int"):
    """GPTQ on one linear weight W [rows, cols] with Hessian H [cols, cols].
    act_order: quantize columns in decreasing-Hessian-diagonal order (most
    important first) — recovers markedly more of RTN's loss.
    qmode: 'int' (asymmetric scale+zero, `bits` levels) | 'ternary' {-1,0,1} |
    'binary' {-1,+1} — the last two use a symmetric per-group abs-mean scale and get
    the SAME error feedback (that's the lever RTN ternary was missing). Returns
    dequantized weights (same shape)."""
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
            if qmode == "int":
                scale, zero = group_params(g, maxq)
            else:                                            # symmetric abs-mean scale
                scale = g.abs().mean(1, keepdim=True).clamp(min=1e-8); zero = None
        d = Hinv[i, i]
        w = W[:, i]
        q = _col_quant(w, scale, zero, maxq, qmode)
        Q[:, i] = q
        err = (w - q) / d
        W[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
    if act_order:
        Q = Q[:, invperm]
    return Q.to(torch.float16)


def imatrix_quantize(W, wdiag, bits, groupsize):
    """llama.cpp-imatrix-equivalent: per group, pick the scale that minimizes the
    IMPORTANCE-WEIGHTED squared error (wdiag = per-input-channel activation
    importance = the diagonal of the Hessian). This is exactly what imatrix does —
    weighted scale selection, NO error feedback — so it isolates 'GPTQ's off-diagonal
    compensation vs imatrix's diagonal weighting' in ONE harness."""
    W = W.clone().float(); rows, cols = W.shape; maxq = 2 ** bits - 1
    w = wdiag.clamp(min=1e-8).float()
    Q = torch.zeros_like(W)
    for c0 in range(0, cols, groupsize or cols):
        g = W[:, c0:c0 + (groupsize or cols)]
        wg = w[c0:c0 + g.shape[1]][None, :]
        gmax = g.amax(1, keepdim=True); gmin = g.amin(1, keepdim=True)
        rng = (gmax - gmin).clamp(min=1e-8)
        best_err = bq = None
        # search the group scale (llama.cpp make_qx_quants sweeps ~19 candidates);
        # proper asymmetric range-based scale + clamped zero-point
        for is_ in range(-9, 10):
            scale = rng / maxq * (1.0 + is_ * 0.02)
            zero = torch.clamp(torch.round(-gmin / scale), 0, maxq)
            q = torch.clamp(torch.round(g / scale) + zero, 0, maxq)
            deq = scale * (q - zero)
            err = (wg * (deq - g) ** 2).sum(1, keepdim=True)
            if best_err is None:
                best_err, bq = err, deq
            else:
                take = err < best_err                         # [rows,1] broadcasts over the group
                best_err = torch.where(take, err, best_err)
                bq = torch.where(take, deq, bq)
        Q[:, c0:c0 + g.shape[1]] = bq
    return Q.to(torch.float16)


def make_recipe(kind, ablate="none"):
    """Per-tensor rate map for the protected-mix build (Grok's Attempt C in torch):
    crush the fat MLP body, PROTECT attention + down_proj + first/last-2 blocks + head.
    `ablate` drops ONE protect class to the body atom (protect-set ablation): one of
    {none, firstlast, attn (qkv), attnout, down} — measures which protection earns its bits.
    Returns recipe(layer_idx, tensor_name, nlayers) -> (bits, qmode)."""
    body = (1, "binary") if kind == "aggr" else (2, "ternary")
    P = (2, "int")                                    # the protect atom
    def recipe(i, name, nlayers):
        first_last = bool(nlayers) and (i < 2 or i >= nlayers - 2)   # first-2 + last-2
        if first_last and ablate != "firstlast":
            return P
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            return body if ablate == "attn" else P
        if "o_proj" in name:
            return body if ablate == "attnout" else P
        if "down_proj" in name:
            return body if ablate == "down" else P
        return body                                   # gate/up = the fat MLP body
    return recipe


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
def eval_ppl(model, testchunks, dev=None):
    model.eval(); nll = 0.0; ntok = 0
    dev = next(model.parameters()).device                 # run on wherever the model lives
    for c in testchunks:
        ids = c.unsqueeze(0).to(dev)
        out = model(ids, labels=ids)
        nll += out.loss.float().item() * (ids.numel() - 1)
        ntok += ids.numel() - 1
    return torch.exp(torch.tensor(nll / ntok)).item()


def linear_layers(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def sequential_gptq(model, calib, dev, bits, groupsize, act_order, offload=False, qmode="int",
                    recipe=None, nlayers=None):
    """The PROPER GPTQ: process transformer blocks in order, feeding each block's
    QUANTIZED outputs into the next block's Hessian — so every layer compensates for
    the error earlier layers actually introduced. Recovers far more of RTN's loss
    than the one-shot fp16-Hessian version.

    offload=True keeps the whole model on CPU and moves ONE block to `dev` at a time —
    this is what lets a 7B (15 GB) quantize on a 16 GB GPU: peak VRAM is one block +
    its Hessians, never the whole model."""
    layers = model.model.layers
    # --- capture the input to block 0 (+ the kwargs each block needs) for every sample.
    # A forward-PRE-hook avoids replacing the layer (so model-level attribute access like
    # `.attention_type` still works) and stops the pass right before block 0 runs.
    inps, cache = [], {}
    def catch(mod, args, kwargs):
        inps.append((args[0] if args else kwargs["hidden_states"]).detach())
        cache.update(kwargs)
        raise RuntimeError("caught")
    cap_dev = next(model.model.embed_tokens.parameters()).device     # where the model currently is
    h = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    for c in calib:
        try: model(c.unsqueeze(0).to(cap_dev))
        except RuntimeError: pass
    h.remove()
    # small kwargs (position_embeddings, attention_mask) stay on dev; DROP cache-related
    # kwargs — a shared DynamicCache would accumulate KV across replays and blow up shapes.
    drop = {"hidden_states", "past_key_values", "past_key_value", "use_cache"}
    kw = {k: v for k, v in cache.items() if k not in drop}
    kw["use_cache"] = False
    inps = [x.cpu() for x in inps]                                    # park activations on CPU

    def to_dev(v):
        if torch.is_tensor(v): return v.to(dev)
        if isinstance(v, tuple): return tuple(to_dev(x) for x in v)
        return v
    kw = {k: to_dev(v) for k, v in kw.items()}                       # pos-emb/mask onto dev once

    def fwd(layer, inp):
        out = layer(inp, **kw)
        return out[0] if isinstance(out, tuple) else out

    def empty():
        if dev == "mps": torch.mps.empty_cache()
        elif dev == "cuda": torch.cuda.empty_cache()

    for i, layer in enumerate(layers):
        if offload: layer.to(dev)                                     # one block on the GPU
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
                fwd(layer, inp.to(dev))
        for h in hooks: h.remove()
        for n, m in lins.items():
            b, qm = recipe(i, n, nlayers) if recipe else (bits, qmode)
            if qm == "skip":                                          # leave this tensor fp16
                del H[n]; continue
            m.weight.data = gptq_quantize(m.weight.data, H[n] / max(cnt[n], 1),
                                          b, groupsize, act_order=act_order, qmode=qm).to(dev)
            del H[n]
        empty()
        # re-run the now-QUANTIZED block to produce inputs for the next block (back to CPU)
        with torch.no_grad():
            inps = [fwd(layer, inp.to(dev)).cpu() for inp in inps]
        if offload: layer.to("cpu")                                   # evict the block
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
    ap.add_argument("--calib-file", help="local text file for calibration (skips HF datasets)")
    ap.add_argument("--eval-file", help="local text file for eval PPL (skips HF datasets)")
    ap.add_argument("--method", default="all",
                    choices=["rtn", "imatrix", "gptq", "gptq-ao", "gptq-seq", "gptq-seq-ao", "both", "all"])
    ap.add_argument("--qmode", default="int", choices=["int", "ternary", "binary"],
                    help="alphabet: int (asymmetric `bits`-bit) | ternary {-1,0,1} ~1.58b | binary {-1,+1} ~1.0b")
    ap.add_argument("--recipe", default="none", choices=["none", "handmix", "aggr"],
                    help="protected-mix build (gptq-seq only): crush MLP body, protect attn/down/first-last/head")
    ap.add_argument("--ablate", default="none", choices=["none", "firstlast", "attn", "attnout", "down"],
                    help="protect-set ablation: drop ONE protect class to the body atom")
    ap.add_argument("--head-bits", type=int, default=0, help="quantize lm_head to N bits (0=leave fp16). Head/embed sweep.")
    ap.add_argument("--embed-bits", type=int, default=0, help="quantize token embeddings to N bits (0=leave fp16).")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--offload", action="store_true",
                    help="keep the model on CPU and move ONE block to the GPU at a time — "
                    "required to quantize a model bigger than VRAM (e.g. a 7B on 16 GB)")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    mdev = "cpu" if a.offload else dev                    # where the model itself lives
    print(f"== pollard-gptq :: {a.model}  W{a.bits}g{a.groupsize}  dev={dev}"
          f"{'  (block-offload)' if a.offload else ''}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(mdev)

    print("loading wikitext-2 ...")
    calib = get_wikitext(tok, "train", a.seqlen, a.nsamples, path=a.calib_file)
    test = get_wikitext(tok, "test", a.seqlen, a.eval_chunks, path=a.eval_file)

    ppl_fp16 = eval_ppl(model, test)
    print(f"fp16 PPL: {ppl_fp16:.4f}", flush=True)

    methods = {"rtn": ["rtn"], "imatrix": ["imatrix"], "gptq": ["gptq"], "gptq-ao": ["gptq-ao"],
               "gptq-seq": ["gptq-seq"], "gptq-seq-ao": ["gptq-seq-ao"],
               "both": ["rtn", "gptq"],
               "all": ["rtn", "imatrix", "gptq", "gptq-seq"]}[a.method]
    import copy
    # only snapshot fp16 weights when we need to run MORE than one method (avoids
    # holding a second full copy of a 7B in RAM)
    fp16_state = copy.deepcopy(model.state_dict()) if len(methods) > 1 else None

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
        if fp16_state is not None:
            model.load_state_dict(fp16_state)
        lins = linear_layers(model.model.layers)              # only the transformer-block linears
        t0 = time.time()
        if method in ("gptq-seq", "gptq-seq-ao"):
            rec = make_recipe("aggr" if a.recipe == "aggr" else "handmix", a.ablate) if a.recipe != "none" else None
            sequential_gptq(model, calib, dev, a.bits, a.groupsize,
                            act_order=method.endswith("-ao"), offload=a.offload, qmode=a.qmode,
                            recipe=rec, nlayers=len(model.model.layers))
        elif method in ("gptq", "gptq-ao"):
            Hs = collect_hessians(lins)                       # Hessians from the fp16 activations
            ao = (method == "gptq-ao")
            for n, m in lins.items():
                m.weight.data = gptq_quantize(m.weight.data, Hs[n], a.bits, a.groupsize,
                                              act_order=ao, qmode=a.qmode).to(m.weight.device)
        elif method == "imatrix":
            Hs = collect_hessians(lins)                       # diagonal = per-channel importance
            for n, m in lins.items():
                m.weight.data = imatrix_quantize(m.weight.data, torch.diag(Hs[n]),
                                                 a.bits, a.groupsize).to(m.weight.device)
        else:                                                 # rtn (alphabet-aware)
            for n, m in lins.items():
                if a.qmode == "ternary":
                    from pollard_lowbit import q_ternary
                    m.weight.data = q_ternary(m.weight.data, a.groupsize).to(m.weight.device)
                elif a.qmode == "binary":
                    from pollard_lowbit import q_binary
                    m.weight.data = q_binary(m.weight.data, a.groupsize).to(m.weight.device)
                else:
                    m.weight.data = rtn_quantize(m.weight.data, a.bits, a.groupsize).to(m.weight.device)
        # head/embed sweep: RTN-quantize lm_head / token_embd (the "fat head" cliff).
        # On tied models (0.5B/1.5B) lm_head shares embed's tensor, so embed-bits drives both.
        emb = model.model.embed_tokens
        if a.embed_bits:
            emb.weight.data = rtn_quantize(emb.weight.data, a.embed_bits, a.groupsize).to(emb.weight.device)
        lm = getattr(model, "lm_head", None)
        if a.head_bits and lm is not None and lm.weight.data_ptr() != emb.weight.data_ptr():
            lm.weight.data = rtn_quantize(lm.weight.data, a.head_bits, a.groupsize).to(lm.weight.device)
        ppl = eval_ppl(model, test)
        gap = ppl - ppl_fp16
        SYM = {"int": float(a.bits), "ternary": 1.585, "binary": 1.0}
        if a.recipe != "none" and method in ("gptq-seq", "gptq-seq-ao"):
            rec = make_recipe("aggr" if a.recipe == "aggr" else "handmix", a.ablate)
            nl = len(model.model.layers)
            qbits = 0; qw = 0                                  # quantized layer weights
            for n, m in model.model.layers.named_modules():
                if isinstance(m, nn.Linear):
                    # recover (layer_idx, subname) from the full module path
                    idx = int(n.split(".")[0]); sub = ".".join(n.split(".")[1:])
                    b, qm = rec(idx, sub, nl)
                    per = 16.0 if qm == "skip" else SYM.get(qm, float(b)) + 16.0 / a.groupsize
                    qbits += per * m.weight.numel(); qw += m.weight.numel()
            bpw = qbits / qw                                   # body-only (matches ternary baseline)
            tag = f"MIX-{a.recipe}/ablate-{a.ablate}@~{bpw:.2f}bpw body (embed/head fp16)"
        else:
            bpw = SYM[a.qmode] + 16.0 / a.groupsize
            tag = f"{a.qmode}@~{bpw:.2f}bpw"
        if a.embed_bits or a.head_bits:
            tag += f" [embed={a.embed_bits or 'fp16'} head={a.head_bits or 'fp16'}]"
        print(f"{method.upper():8s} {tag} PPL: {ppl:.4f}   "
              f"(+{gap:.4f} vs fp16)   [{_hms(time.time()-t0)}]", flush=True)
        return ppl

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
