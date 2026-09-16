#!/usr/bin/env python3
"""pollard-abliterate -- OPTIONAL refusal-direction ablation, applied on the FP16
weights BEFORE quantization so it composes with a Pollard build for free.

This is the published "abliteration" technique (Arditi et al. 2024, "Refusal in
LLMs is mediated by a single direction"; FailSpy's implementation): find the one
residual-stream direction that mediates refusal (diff-of-means of activations on a
"refuses" vs "complies" prompt set), then ORTHOGONALIZE every residual-writing
weight (attn o_proj, mlp down_proj, and the embedding) against it, so the model
can no longer write that direction into the stream.

It edits the FP16 model only; quantization runs afterward on the modified weights
(exactly how every abliterated GGUF on HF is made). So it slots into the pipeline
as one OPT-IN pass, OFF by default and clearly labelled:

    FP16 --[pollard-abliterate --harmful A.txt --harmless B.txt]--> FP16' --> pollard build

Honest scope: this is a behaviour-changing transform the USER opts into on THEIR
model; it can cost some coherence, and stacking it on an extreme low-bit crush can
compound that -- so measure the PPL/KL delta vs the un-ablated build (pollard-kl)
before trusting it, same as everything else. The contrast prompt SETS are supplied
by the user (one prompt per line); this tool ships only a tiny benign smoke-test
default so `--selftest` runs -- it is NOT a real refusal set.

Usage:
  pollard-abliterate --model <hf-dir-or-id> --harmful refuse.txt --harmless comply.txt \\
      --out ./model-abliterated
  pollard-abliterate --model Qwen/Qwen2.5-0.5B-Instruct --selftest   # mechanism canary
"""
import argparse, os, sys
import torch


# tiny BENIGN placeholder sets -- only so --selftest exercises the mechanism.
# NOT a refusal set; supply real contrast prompts via --harmful/--harmless.
_SMOKE_A = ["Describe a stormy sea at night.", "Explain how a bicycle stays upright.",
            "Summarize the plot of a heist movie.", "Write a limerick about the moon."]
_SMOKE_B = ["Describe a calm meadow at noon.", "Explain how a kite flies.",
            "Summarize the plot of a comedy.", "Write a limerick about the sun."]


def _load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


@torch.no_grad()
def _mean_resid(model, tok, prompts, dev, batch=8):
    """Mean last-token residual-stream vector at EVERY layer for a prompt set.
    Returns [n_layers+1, d_model] (index 0 = embeddings, i = block i output)."""
    acc = None
    n = 0
    chat = getattr(tok, "chat_template", None)
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        if chat:
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in chunk]
        else:
            texts = chunk
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)            # [L+1, B, T, D]
        # last NON-pad token per sequence
        last = enc["attention_mask"].sum(1) - 1           # [B]
        idx = last.view(1, -1, 1, 1).expand(hs.size(0), -1, 1, hs.size(-1))
        vec = hs.gather(2, idx).squeeze(2).float()        # [L+1, B, D]
        s = vec.sum(1)                                    # [L+1, D]
        acc = s if acc is None else acc + s
        n += len(chunk)
    return acc / max(n, 1)


def _pick_layer(diff, layer):
    """diff: [L+1, D]. Choose the direction layer. 'auto' = largest-norm diff
    among the middle-to-late blocks (embeddings excluded), where refusal is
    typically most linearly separable."""
    norms = diff.norm(dim=-1)                              # [L+1]
    if layer != "auto":
        return int(layer)
    L = diff.size(0) - 1
    lo = max(1, int(L * 0.35))                             # skip early blocks
    hi = max(lo + 1, int(L * 0.85))                        # and the last blocks (norm blows up)
    j = lo + int(torch.argmax(norms[lo:hi]).item())
    return j


@torch.no_grad()
def abliterate(model, r_hat, dev, strength=-1.0):
    """Scale the r_hat component of every residual-WRITING weight by (1 + strength).

    The published technique removes a direction: W -= r r^T W. That is this function at
    strength=-1.0, and it stays the default. But removal is one point on a dial, and the same
    diff-of-means direction that mediates refusal also mediates any behaviour you can write two
    contrasting prompt sets for -- terse against discursive, tool-calling against prose. So the
    coefficient is exposed:

        -1.0   remove the direction entirely (abliteration, the default)
        -0.5   halve it -- a softer touch when full removal costs coherence
         0.0   no-op
        +0.5   amplify by 1.5x: steer the model TOWARD the behaviour set A shows

    Amplifying is not free and is not symmetric with removal. Pushing a direction hard enough will
    make a model do that one thing at the cost of everything else, and the failure looks like
    fluent nonsense rather than an error. Start near +0.25, measure with pollard-kl against the
    unedited build, and stop when the KL delta stops buying you behaviour.

    o_proj/down_proj write columns into the stream (out-dim = D): W += a * r r^T W.
    embed_tokens rows ARE stream vectors (dim 1 = D):             W += a * (W r) r^T.
    """
    r = r_hat.to(dev).float()
    a = float(strength)
    edited = 0
    layers = model.model.layers
    for blk in layers:
        for lin in (blk.self_attn.o_proj, blk.mlp.down_proj):
            W = lin.weight.data.float()                   # [D, in]
            lin.weight.data = (W + a * torch.outer(r, r @ W)).to(lin.weight.dtype)
            edited += 1
    emb = model.model.embed_tokens.weight.data.float()    # [vocab, D]
    model.model.embed_tokens.weight.data = (emb + a * torch.outer(emb @ r, r)).to(model.model.embed_tokens.weight.dtype)
    edited += 1
    return edited


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16)")
    ap.add_argument("--harmful", "--set-a", dest="harmful",
                    help="prompt set A, one per line: the behaviour the direction points TOWARD "
                         "(refusals, for ablation; the target mode, for steering)")
    ap.add_argument("--harmless", "--set-b", dest="harmless",
                    help="matched contrast set B, one per line")
    ap.add_argument("--strength", type=float, default=-1.0,
                    help="how much of the direction to keep: -1.0 removes it (abliteration, the "
                         "default), -0.5 halves it, +0.5 amplifies it by 1.5x to steer TOWARD set "
                         "A. Measure with pollard-kl before trusting any positive value.")
    ap.add_argument("--out", help="output dir for the abliterated FP16 model")
    ap.add_argument("--layer", default="auto", help="direction layer index, or 'auto'")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code (custom archs); 'auto' = only if config has auto_map")
    ap.add_argument("--selftest", action="store_true",
                    help="mechanism canary on the benign smoke sets -- writes nothing")
    a = ap.parse_args()

    if not a.selftest and not (a.harmful and a.harmless):
        sys.exit("ERROR: real use needs --harmful and --harmless (or use --selftest).")
    if not a.selftest and not a.out:                        # default output into the workspace
        try:
            import pollard_workspace as ws
            a.out = os.path.join(ws.model_dir(a.model, create=True), ws.model_basename(a.model) + "-abliterated")
            print(f"   (no --out) -> workspace: {a.out}")
        except Exception:
            sys.exit("ERROR: pass --out for the abliterated model.")

    from transformers import AutoTokenizer
    from pollard_load import load_backbone
    import pollard_workspace as ws
    trc = ws.resolve_trust_remote_code(a.model, a.trust_remote_code)
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"== pollard-abliterate :: {a.model}  dev={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_backbone(a.model, torch.float16, dev, trust_remote_code=trc)

    A = _load_lines(a.harmful) if a.harmful else _SMOKE_A
    B = _load_lines(a.harmless) if a.harmless else _SMOKE_B
    print(f"  contrast sets: {len(A)} vs {len(B)}"
          f"{'  (BENIGN smoke set -- mechanism only)' if a.selftest else ''}", flush=True)

    ma = _mean_resid(model, tok, A, dev)
    mb = _mean_resid(model, tok, B, dev)
    diff = (ma - mb)                                       # [L+1, D]
    j = _pick_layer(diff, a.layer)
    r_hat = diff[j] / (diff[j].norm() + 1e-8)
    print(f"  refusal direction from layer {j}/{diff.size(0)-1}  "
          f"(||diff||={diff[j].norm():.3f})", flush=True)

    # measure how much of the direction lives in the writers before/after (sanity).
    # hidden_states[j] is the OUTPUT of block j-1, so that block's o_proj produced it.
    sj = min(max(j - 1, 0), len(model.model.layers) - 1)
    o0 = model.model.layers[sj].self_attn.o_proj.weight.data.float()
    before = (r_hat.to(dev).float() @ o0).norm().item()
    edited = abliterate(model, r_hat, dev, a.strength)
    o1 = model.model.layers[sj].self_attn.o_proj.weight.data.float()
    after = (r_hat.to(dev).float() @ o1).norm().item()
    want = "collapse to ~0" if a.strength <= -0.999 else f"scale by {1 + a.strength:.2f}x"
    verb = "orthogonalized" if a.strength < 0 else "amplified"
    print(f"  {verb} {edited} residual-writers at strength {a.strength:+.2f}; "
          f"proj(o_proj@blk{sj}) {before:.3f} -> {after:.3f} (should {want})", flush=True)
    if a.strength > 0:
        print("  NOTE: steering TOWARD a direction can degrade everything else and the failure "
              "reads as fluent nonsense.\n        Measure against the unedited build with "
              "pollard-kl before you ship this.", flush=True)

    if a.selftest:
        # coherence canary: the model must still produce fluent text after surgery
        enc = tok([tok.apply_chat_template([{"role": "user", "content": "In one sentence, why is the sky blue?"}],
                                           tokenize=False, add_generation_prompt=True)],
                  return_tensors="pt").to(dev)
        gen = model.generate(**enc, max_new_tokens=40, do_sample=False)
        txt = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  post-surgery sample: {txt!r}", flush=True)
        ok = after < before * 0.05 and len(txt.strip()) > 0
        print(f"SELFTEST {'PASS' if ok else 'CHECK'} -- direction collapsed & model still generates.", flush=True)
        return

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"wrote abliterated FP16 -> {a.out}\n"
          f"  next: convert to GGUF and run a Pollard build; compare PPL/KL vs the "
          f"un-ablated build (pollard-kl) to see the quality cost you're opting into.", flush=True)


if __name__ == "__main__":
    main()
