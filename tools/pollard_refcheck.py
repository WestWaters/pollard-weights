#!/usr/bin/env python3
"""pollard-refcheck - prove the reference forward is sane BEFORE measuring anything on it.

Every Pollard lane that measures rather than guesses -- Hessians, per-tensor sensitivity, routing
statistics, any KL or NLL gate -- reads its numbers out of a full-precision forward pass. If that
forward is subtly wrong, nothing downstream is salvageable: the allocator gets a confident ranking of
the wrong tensors and the build looks fine all the way to the card.

The cheapest check that catches it is the model's own loss on in-domain text. A correct bf16 forward on
chat/prose/code rows lands near 1.5-2.5 nats; a forward with scrambled positions lands near 5 and gets
WORSE deeper into the sequence, because every layer is attending at the wrong relative offsets.

Two defects found this way, both invisible to a shape or key audit:

  * `hy_v4` (Hy4-preview) on transformers main: rotate-half (NeoX) rotary applied to a checkpoint that
    stores `q_pe`/`k_pe` interleaved. bf16 NLL 5.02 (ppl 151) instead of 1.855 (ppl 6.39), 19% top-1,
    and routing statistics wrong enough to matter -- top-32 mass median 62% vs 38% after the fix, per
    layer routing-mass correlation 0.56. Reported by bot-lab-21 (issue #65).
  * K2-Horizon: tensor-for-tensor identical to llama, so NORM rotary looked right by symmetry, but the
    reference fork specifies NEOX. Same class of failure, caught late.

  pollard-refcheck --list                                  # known model-code defects and their fixes
  pollard-refcheck --model <hf-dir-or-id> --calib rows.txt  # measure the reference NLL, verdict
  pollard-refcheck --model <hf> --calib rows.txt --fix      # apply known fixes, report before/after

Exit code is 1 when the forward looks broken, so it belongs in front of a capture in any script."""
import argparse
import importlib
import json
import math
import os
import sys

# A correct forward on in-domain rows sits well under this; scrambled positions sit far above it.
# Deliberately loose -- this separates "broken" from "fine", it is not a quality metric.
NLL_SUSPECT = 3.0
# Positional damage grows along the sequence, so the tail/head ratio is a second, independent signal.
# A CORRECT forward gets better with more context, so this ratio sits at or below 1.0 in the healthy
# case. The bar is set just above that rather than at some comfortable distance: Hy4's broken forward
# went 5.1 -> 5.8 across the sequence, a ratio of only 1.137, and a threshold picked for looking safe
# (1.15) would have missed the exact defect this check exists to catch.
TAIL_RATIO_SUSPECT = 1.10


# ---- rotary conventions ----------------------------------------------------------------------
# Written against a duck-typed array so the convention itself is testable with numpy alone, with no
# torch and no checkpoint. The bug is a one-line difference between these two and it is worth having
# a test that states which is which.
def rope_rotate_half(x, cos, sin, xp):
    """NeoX / rotate-half: pairs element i with i + d/2."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = xp.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


def rope_interleaved(x, cos, sin, xp):
    """Megatron / PTM interleaved: pairs element 2i with 2i+1.

    `cos`/`sin` arrive in the half-width layout transformers builds (d/2 distinct angles, doubled by
    concatenation). Interleaved pairing needs the same angles doubled by REPEAT instead, so each
    adjacent pair shares one angle.
    """
    d = cos.shape[-1] // 2
    cos_i = xp.repeat(cos[..., :d], 2, axis=-1)
    sin_i = xp.repeat(sin[..., :d], 2, axis=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = xp.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos_i + rot * sin_i


# ---- known model-code defects ---------------------------------------------------------------
def _patch_hy_v4():
    """Replace hy_v4's module-level rotary with the interleaved pairing its checkpoint uses.

    Patching the module-level function (rather than the attention class) is deliberate and load
    bearing: attention AND the DSA indexer both call it, and fixing only attention leaves the indexer
    selecting on wrong positions -- which is exactly the path routing statistics come from.
    """
    import torch
    m = importlib.import_module("transformers.models.hy_v4.modeling_hy_v4")

    def apply_rotary_interleaved(q, k, cos, sin, unsqueeze_dim=1, **kw):
        d = cos.shape[-1] // 2
        cos_i = cos[..., :d].repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
        sin_i = sin[..., :d].repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)

        def rot(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        return (q * cos_i) + (rot(q) * sin_i), (k * cos_i) + (rot(k) * sin_i)

    before = getattr(m, "apply_rotary_pos_emb", None)
    m.apply_rotary_pos_emb = apply_rotary_interleaved
    return before is not apply_rotary_interleaved


KNOWN_DEFECTS = {
    "hy_v4": {
        "what": "transformers main applies rotate-half (NeoX) rotary, but the checkpoint stores "
                "q_pe/k_pe interleaved (Megatron/PTM). vLLM's hy_v4 builds its rotary with "
                "is_neox_style=False for both attention and the DSA indexer.",
        "symptom": "bf16 NLL ~5.0 (ppl ~151) on in-domain rows instead of ~1.86 (ppl ~6.4), rising "
                   "along the sequence; routing statistics wrong (top-32 mass 62% -> 38%).",
        "breaks": "Hessians, per-tensor sensitivity, routing capture, any NLL/KL gate. Norm-seam "
                  "absmax survives roughly intact (per-layer ratio 0.84-1.19), which is why a "
                  "smoothing pass can look fine while everything measured is not.",
        "expect_nll": 1.86,
        "patch": _patch_hy_v4,
        "credit": "bot-lab-21, issue #65 (2026-09-10, transformers 5.17.0.dev0)",
    },
}


def model_type_of(path_or_id):
    """The `model_type` in a model's config, local dir or Hub id."""
    try:
        if os.path.isdir(path_or_id):
            return json.load(open(os.path.join(path_or_id, "config.json"))).get("model_type", "")
        import urllib.request
        url = f"https://huggingface.co/{path_or_id}/raw/main/config.json"
        req = urllib.request.Request(url, headers={"User-Agent": "pollard-refcheck"})
        with urllib.request.urlopen(req, timeout=40) as fh:
            return json.load(fh).get("model_type", "")
    except Exception:
        return ""


def apply_known_fixes(model_type):
    """Apply every known model-code fix for this architecture. Returns what it touched."""
    d = KNOWN_DEFECTS.get(model_type)
    if not d:
        return []
    try:
        changed = d["patch"]()
    except Exception as e:                                                # noqa: BLE001
        print(f"WARNING: could not apply the {model_type} fix ({e})", file=sys.stderr)
        return []
    return [model_type] if changed else []


# ---- the measurement ------------------------------------------------------------------------
def reference_nll(model_id, rows, max_rows=16, seq=1024, dtype="bfloat16", device=None):
    """Mean NLL over `rows` of text, plus a head/tail split.

    The split is the point: positional damage is not uniform, it compounds with distance, so a forward
    with scrambled pairing scores worse at the end of a sequence than the start. A single mean can be
    explained away as "hard rows"; a rising profile cannot.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype),
        device_map=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    tot, ntok, head, head_n, tail, tail_n = 0.0, 0, 0.0, 0, 0.0, 0
    with torch.no_grad():
        for text in rows[:max_rows]:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=seq).input_ids
            if ids.shape[-1] < 16:
                continue
            ids = ids.to(model.device)
            logits = model(ids).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
            tot += nll.sum().item(); ntok += nll.numel()
            cut = max(1, nll.numel() // 4)
            head += nll[:cut].sum().item(); head_n += cut
            tail += nll[-cut:].sum().item(); tail_n += cut
    if not ntok:
        raise RuntimeError("no usable calibration rows")
    return {"nll": tot / ntok, "ppl": math.exp(tot / ntok),
            "head_nll": head / head_n, "tail_nll": tail / tail_n,
            "tail_ratio": (tail / tail_n) / (head / head_n), "tokens": ntok}


def verdict(r, expect=None):
    """(ok, reasons) from a measurement."""
    bad = []
    if expect is not None and r["nll"] > expect * 1.5:
        bad.append(f"NLL {r['nll']:.3f} is far above the expected {expect:.2f} for this architecture")
    if r["nll"] > NLL_SUSPECT:
        bad.append(f"NLL {r['nll']:.3f} exceeds {NLL_SUSPECT} on in-domain rows")
    if r["tail_ratio"] > TAIL_RATIO_SUSPECT:
        bad.append(f"loss RISES along the sequence (tail/head {r['tail_ratio']:.2f}) - the signature "
                   f"of wrong relative positions")
    return (not bad), bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", help="HF model dir or id to check")
    ap.add_argument("--calib", help="text file of calibration rows, one per line")
    ap.add_argument("--rows", type=int, default=16, help="rows to score (default 16)")
    ap.add_argument("--seq", type=int, default=1024, help="tokens per row (default 1024)")
    ap.add_argument("--expect-nll", type=float, default=None,
                    help="expected NLL for this architecture (else the registry's value, if known)")
    ap.add_argument("--fix", action="store_true",
                    help="apply known model-code fixes for this architecture and report before/after")
    ap.add_argument("--list", action="store_true", help="list the known defects and exit")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.list:
        for mt, d in KNOWN_DEFECTS.items():
            print(f"\n{mt}")
            print(f"  defect : {d['what']}")
            print(f"  shows  : {d['symptom']}")
            print(f"  breaks : {d['breaks']}")
            print(f"  expect : bf16 NLL about {d['expect_nll']} on in-domain rows")
            print(f"  credit : {d['credit']}")
        return 0

    if not a.model or not a.calib:
        ap.error("--model and --calib are both required (or use --list)")
    if not os.path.exists(a.calib):
        ap.error(f"no such calibration file: {a.calib}")
    rows = [l.strip() for l in open(a.calib, encoding="utf-8", errors="replace") if l.strip()]
    if not rows:
        ap.error("the calibration file has no non-empty lines")

    mt = model_type_of(a.model)
    known = KNOWN_DEFECTS.get(mt)
    expect = a.expect_nll if a.expect_nll is not None else (known or {}).get("expect_nll")
    print(f"model_type: {mt or 'unknown'}"
          + (f"   KNOWN DEFECT on record (see --list)" if known else ""))

    out = {"model": a.model, "model_type": mt}
    before = reference_nll(a.model, rows, a.rows, a.seq)
    ok, why = verdict(before, expect)
    out["before"] = before
    print(f"\nreference forward: NLL {before['nll']:.4f}  ppl {before['ppl']:.2f}  "
          f"head {before['head_nll']:.3f} -> tail {before['tail_nll']:.3f} "
          f"(x{before['tail_ratio']:.2f}) over {before['tokens']} tokens")
    for r in why:
        print(f"   SUSPECT: {r}")

    if a.fix:
        touched = apply_known_fixes(mt)
        if not touched:
            print("\nno known fix for this architecture; nothing applied")
        else:
            after = reference_nll(a.model, rows, a.rows, a.seq)
            out["after"] = after
            ok, why = verdict(after, expect)
            print(f"\nafter the {', '.join(touched)} fix: NLL {after['nll']:.4f}  "
                  f"ppl {after['ppl']:.2f}  head {after['head_nll']:.3f} -> "
                  f"tail {after['tail_nll']:.3f} (x{after['tail_ratio']:.2f})")
            print(f"   NLL moved {before['nll']:.4f} -> {after['nll']:.4f}")
            for r in why:
                print(f"   STILL SUSPECT: {r}")

    out["ok"] = ok
    if a.json:
        print(json.dumps(out, indent=1))
    print("\nVERDICT: the reference forward looks sound - measurements taken on it are trustworthy."
          if ok else
          "\nVERDICT: this forward is NOT safe to measure on. Every Hessian, sensitivity ranking, "
          "routing statistic and NLL/KL gate taken from it would be invalid.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
