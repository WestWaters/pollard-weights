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
before trusting it, same as everything else.

The embedding edit is where "abliterated model repeats itself" comes from (the failure
huihui reported, 2026-09): token_embd rows are the model's vocabulary, and bending every
row against one direction shifts control tokens too, which is exactly the CONTROL TOKENS
/ REPEATED verdict the coherence gate gives a crushed embedding. So after surgery this
tool runs that gate on a few benign prompts (same bar as pollard-bench: half the tail
n-grams repeating = loop). `--embed auto` (default) edits the embedding, and if the gate
trips, restores it and gates again -- the residual writers alone usually carry the
ablation. `--embed off` (`--skip-embed`) never touches it; `--embed on` keeps the classic
recipe regardless. The verdict is written next to the model as abliterate_gate.json. The contrast prompt SETS are supplied
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


def load_backbone(model_id, dtype=None, device="cpu", eval_mode=True, **kw):
    """This tool's OWN loader -- model tooling does not depend on a shared/brain-side one.

    AutoModelForCausalLM refuses a vision-language config outright ("Unrecognized configuration
    class Qwen2VLConfig for this kind of AutoModel"), so fall back to the vision-language auto
    classes: a VL model's text stack quantizes like any other. Both failures are reported if
    neither works, not just the last one.

    Pass `device_map=` to shard across GPU+CPU+disk; accelerate owns placement after that, so
    `device` is ignored in that case.
    """
    import torch as _t, transformers as _tf
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = _t.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    except ValueError as text_only_err:
        model, errs = None, [f"AutoModelForCausalLM: {text_only_err}"]
        for _n in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            _c = getattr(_tf, _n, None)
            if _c is None:
                continue
            try:
                model = _c.from_pretrained(model_id, dtype=dtype, **kw)
                break
            except Exception as e:
                errs.append(f"{_n}: {e}")
        if model is None:
            raise SystemExit(f"could not load {model_id!r} as a causal LM or a vision-language "
                             "model:\n  " + "\n  ".join(str(e)[:160] for e in errs)) from None
    if kw.get("device_map") is None:            # accelerate already placed a dispatched model
        model = model.to(device)
    return model.eval() if eval_mode else model


def text_layers(model):
    """This tool's OWN layer walk. A VL model keeps its text stack under `model.language_model`."""
    for path in ("model.language_model", "language_model.model", "model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        layers = getattr(node, "layers", None) if node is not None else None
        if layers is not None:
            return layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    raise SystemExit(f"could not find the decoder layers on {type(model).__name__}; "
                     "this tool's text_layers() needs a path for this architecture")


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
def abliterate(model, r_hat, dev, strength=-1.0, skip_embed=False):
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
    layers = text_layers(model)
    for blk in layers:
        for lin in (blk.self_attn.o_proj, blk.mlp.down_proj):
            W = lin.weight.data.float()                   # [D, in]
            lin.weight.data = (W + a * torch.outer(r, r @ W)).to(lin.weight.dtype)
            edited += 1
    if skip_embed:
        return edited
    emb = model.model.embed_tokens.weight.data.float()    # [vocab, D]
    model.model.embed_tokens.weight.data = (emb + a * torch.outer(emb @ r, r)).to(model.model.embed_tokens.weight.dtype)
    edited += 1
    return edited


# the same control-token shape pollard-bench's gate looks for: <|im_start|>, <|channel|>, <pad>, <unk>
_CTRL = r"<\|[^|>]{1,32}\|?>|<[a-z_]{2,16}>"
# (prompt, words a real answer contains) -- fluent word salad neither loops nor leaks control tokens,
# so "does it know the answer" is the third check, as in pollard-bench
_GATE_PROMPTS = [("In one sentence, why is the sky blue?", ["scatter", "blue", "light", "wavelength"]),
                 ("Name three fruits and one thing they have in common.", ["apple", "banana", "orange", "fruit", "sweet", "seed"]),
                 ("Write two short sentences about a bicycle.", ["bicycle", "bike", "wheel", "pedal", "ride"])]
_RANK = {"PASS": 2, "WEAK": 1, "FAIL": 0}


def judge_sample(text: str, expect=()) -> dict:
    """Score one post-surgery generation the way pollard-bench's coherence gate scores a chat turn.

    CONTROL TOKENS first (the embedding was bent -- the fix is to restore it, not to soften the
    strength), then a tail loop (half the tail n-grams repeating, or a handful of words over and
    over), then empty, then -- if `expect` is given -- whether any expected word is in the answer
    (salad reads as WEAK). Pure function: unit-tested on strings, no model needed.
    Returns {"verdict": PASS|WEAK|FAIL, "ok", "reason", "repeat"}."""
    import re
    from pollard_bench import tail_repeat
    body = (text or "").strip()
    if not body:
        return {"verdict": "FAIL", "ok": False, "reason": "NO OUTPUT", "repeat": 0.0}
    rep = tail_repeat(body)
    words = body.split()
    loop = rep >= 0.5 or (len(set(words)) <= 2 and len(words) > 8)
    if re.search(_CTRL, body):
        return {"verdict": "FAIL", "ok": False, "repeat": rep,
                "reason": "CONTROL TOKENS in the output (embedding bent -- restore it)"}
    if loop:
        return {"verdict": "FAIL", "ok": False, "reason": "REPEATED output (loop)", "repeat": rep}
    if expect and not any(e.lower() in body.lower() for e in expect):
        return {"verdict": "WEAK", "ok": False, "reason": "no known answer in the output", "repeat": rep}
    return {"verdict": "PASS", "ok": True, "reason": "coherent", "repeat": rep}


@torch.no_grad()
def coherence_gate(model, tok, dev, prompts=None, max_new_tokens=60) -> dict:
    """Greedy-decode a few benign prompts through the model's own chat template and judge each.
    Special tokens are kept in the decode so a leaked <|im_start|> is seen, not hidden."""
    rows = []
    for p, expect in (prompts or _GATE_PROMPTS):
        try:
            txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True)
        except Exception:                                  # base model: no template
            txt = p + "\n"
        enc = tok([txt], return_tensors="pt").to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        out = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        # the model's own end-of-turn is not a leak: cut there, drop padding; anything else in
        # <|...|> form left in the body is a bent embedding talking
        for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", tok.eos_token):
            if t and t in out:
                out = out.split(t)[0]
        if tok.pad_token:
            out = out.replace(tok.pad_token, "")
        j = judge_sample(out, expect)
        j.update({"prompt": p, "sample": out.strip()[:200]})
        rows.append(j)
    verdict = min((r["verdict"] for r in rows), key=_RANK.get)
    return {"verdict": verdict, "rows": rows}


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
    ap.add_argument("--embed", default="auto", choices=["auto", "on", "off"],
                    help="edit the embedding too: 'auto' (default) edits it and backs the edit out if the "
                         "coherence gate trips; 'off' never touches it; 'on' = classic recipe, no back-out")
    ap.add_argument("--skip-embed", dest="embed", action="store_const", const="off",
                    help="alias for --embed off")
    ap.add_argument("--no-gate", action="store_true", help="skip the post-surgery coherence gate")
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
    sj = min(max(j - 1, 0), len(text_layers(model)) - 1)
    o0 = text_layers(model)[sj].self_attn.o_proj.weight.data.float()
    before = (r_hat.to(dev).float() @ o0).norm().item()
    emb_saved = model.model.embed_tokens.weight.data.clone() if a.embed == "auto" else None
    edited = abliterate(model, r_hat, dev, a.strength, skip_embed=(a.embed == "off"))
    o1 = text_layers(model)[sj].self_attn.o_proj.weight.data.float()
    after = (r_hat.to(dev).float() @ o1).norm().item()
    want = "collapse to ~0" if a.strength <= -0.999 else f"scale by {1 + a.strength:.2f}x"
    verb = "orthogonalized" if a.strength < 0 else "amplified"
    print(f"  {verb} {edited} residual-writers at strength {a.strength:+.2f} (embedding: {a.embed}); "
          f"proj(o_proj@blk{sj}) {before:.3f} -> {after:.3f} (should {want})", flush=True)

    gate = None
    if not a.no_gate:
        gate = coherence_gate(model, tok, dev)
        gate["embed"] = "edited" if a.embed != "off" else "untouched"
        print(f"  gate: {gate['verdict']}  " + "; ".join(r["reason"] for r in gate["rows"]), flush=True)
        if gate["verdict"] != "PASS" and emb_saved is not None:
            # the embedding edit is the usual culprit (huihui's repetition): put it back, judge again
            model.model.embed_tokens.weight.data = emb_saved
            gate2 = coherence_gate(model, tok, dev)
            gate2["embed"] = "restored (gate tripped with it edited)"; gate2["first_attempt"] = gate
            gate = gate2
            print(f"  gate after restoring the embedding: {gate['verdict']}  "
                  + "; ".join(r["reason"] for r in gate["rows"]), flush=True)
        for r in gate["rows"]:
            print(f"    {r['verdict']:4s} {r['prompt'][:40]:40s} {r['sample'][:90]!r}", flush=True)
    del emb_saved
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
        ok = after < before * 0.05 and len(txt.strip()) > 0 and (gate is None or gate["verdict"] == "PASS")
        print(f"SELFTEST {'PASS' if ok else 'CHECK'} -- direction collapsed & model still generates"
              f"{'' if gate is None else ' & gate ' + gate['verdict']}.", flush=True)
        return

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    if gate is not None:
        import json
        gate.update({"model": a.model, "strength": a.strength, "layer": int(j)})
        with open(os.path.join(a.out, "abliterate_gate.json"), "w") as f:
            json.dump(gate, f, indent=1)
        if gate["verdict"] != "PASS":
            print(f"  GATE FAIL -- written anyway so you can inspect it, but do not build from this "
                  f"without fixing it (try --strength -0.5).", flush=True)
    print(f"wrote abliterated FP16 -> {a.out}\n"
          f"  next: convert to GGUF and run a Pollard build; compare PPL/KL vs the "
          f"un-ablated build (pollard-kl) to see the quality cost you're opting into.", flush=True)


if __name__ == "__main__":
    main()
