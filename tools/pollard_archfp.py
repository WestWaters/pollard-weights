#!/usr/bin/env python3
"""pollard-archfp — what IS this architecture, structurally, and which known one is it a twin of?

Pollard keeps meeting models whose `model_type` / `general.architecture` nobody has seen, and the
question that actually decides the work is never the name: it is whether the *layout* is something a
runtime already implements. A new name with a standard layout is a boilerplate port (an enum, a name,
a tensor map, reuse an existing builder). A new name with a genuinely new block is real work. Guessing
wrong in either direction costs hours -- or ships a mis-built model.

So this fingerprints the layout from the tensor names themselves and reports the nearest known family,
the exact differences, and any blocker that makes "it's a twin" a lie.

Source-agnostic by design: it normalises GGUF names (`blk.0.attn_q.weight`), HF/safetensors names
(`model.layers.0.self_attn.q_proj.weight`) and the export lanes' variants into ONE canonical role set,
so the same answer comes back whether you point it at a GGUF, an HF repo, or a converted checkpoint.

  pollard-archfp --gguf model-f16.gguf
  pollard-archfp --model Qwen/Qwen3-8B
  pollard-archfp --model ./local-hf-dir --json

Read-only. No weights are loaded: GGUF reads the tensor table, HF reads the safetensors index.
"""
import argparse
import json
import os
import re
import sys

# ---- canonical roles ---------------------------------------------------------------------------
# One vocabulary both naming conventions collapse into. Anything unmatched is kept verbatim so a
# genuinely novel tensor shows up as a difference instead of being silently dropped.
_ROLE_PATTERNS = [
    # --- MLA (DeepSeek-V2/V3 latent attention, and the GLM/Hy DSA descendants) -------------------
    # Tried first: `q_a_proj` must not be captured by the plain `q_proj` matcher below.
    (r"(?:self_attn|attn)[._]q_a_proj\b|attn_q_a\b",              "attn_q_a"),
    (r"(?:self_attn|attn)[._]q_b_proj\b|attn_q_b\b",              "attn_q_b"),
    (r"(?:self_attn|attn)[._]q_a_layernorm\b|attn_q_a_norm\b",    "attn_q_a_norm"),
    (r"(?:self_attn|attn)[._]kv_a_proj_with_mqa\b|attn_kv_a_mqa\b", "attn_kv_a"),
    (r"(?:self_attn|attn)[._]kv_b_proj\b|attn_kv_b\b",            "attn_kv_b"),
    (r"(?:self_attn|attn)[._]kv_a_layernorm\b|attn_kv_a_norm\b",  "attn_kv_a_norm"),
    (r"(?:self_attn|attn)[._]indexer[._]|attn_indexer",            "attn_indexer"),
    (r"(?:self_attn|attn)[._]linear_gate\b",                       "attn_gate"),
    # --- standard attention ----------------------------------------------------------------------
    (r"(?:self_attn|attn)[._](?:q_proj|q)\b",                     "attn_q"),
    (r"(?:self_attn|attn)[._](?:k_proj|k)\b",                     "attn_k"),
    (r"(?:self_attn|attn)[._](?:v_proj|v)\b",                     "attn_v"),
    (r"(?:self_attn|attn)[._](?:o_proj|output|out_proj|dense)\b",  "attn_out"),
    (r"(?:self_attn|attn)[._]q_norm\b|attn_q_norm\b",              "attn_q_norm"),
    (r"(?:self_attn|attn)[._]k_norm\b|attn_k_norm\b",              "attn_k_norm"),
    (r"(?:self_attn|attn)[._][a-z_]*g_proj\b|attn_gate\b",         "attn_gate"),
    (r"input_layernorm|attn_norm\b",                               "attn_norm"),
    (r"post_attention_layernorm|ffn_norm\b",                       "ffn_norm"),
    (r"(?:mlp|feed_forward|block_sparse_moe)(?:\.experts|\.shared_experts)?(?:\.\d+)?[._]gate_proj\b|ffn_gate\b",            "ffn_gate"),
    (r"(?:mlp|feed_forward|block_sparse_moe)(?:\.experts|\.shared_experts)?(?:\.\d+)?[._]up_proj\b|ffn_up\b",                "ffn_up"),
    (r"(?:mlp|feed_forward|block_sparse_moe)(?:\.experts|\.shared_experts)?(?:\.\d+)?[._]down_proj\b|ffn_down\b",            "ffn_down"),
    (r"gate_up_proj|ffn_gate_up(?:_exps|_shexp)?\b",               "ffn_gate_up"),
    (r"ffn_gate(?:_exps|_shexp)\b",                                "ffn_gate"),
    (r"ffn_up(?:_exps|_shexp)\b",                                  "ffn_up"),
    (r"ffn_down(?:_exps|_shexp)\b",                                "ffn_down"),
    (r"(?:mlp\.)?(?:gate|router)\b(?!_proj)|ffn_gate_inp\b",       "router"),
]
_EXPERT_RE = re.compile(r"experts?[._]|_exps\b|\.experts\.")
_BLOCK_RE = re.compile(r"(?:^|\.)(?:blk|layers|h|model\.layers)\.(\d+)\.")


def _strip_block(name):
    """`blk.12.attn_q.weight` / `model.layers.12.self_attn.q_proj.weight` -> the per-block remainder."""
    m = _BLOCK_RE.search(name)
    return name[m.end():] if m else None


def _role(tail):
    for pat, role in _ROLE_PATTERNS:
        if re.search(pat, tail):
            return role
    return None


def fingerprint(names):
    """Canonical layout fingerprint from a list of tensor names, from ANY source."""
    roles, unknown, has_bias, n_blocks, expert_roles = set(), set(), set(), 0, set()
    for n in names:
        m = _BLOCK_RE.search(n)
        if m:
            n_blocks = max(n_blocks, int(m.group(1)) + 1)
        tail = _strip_block(n)
        if tail is None:
            continue
        is_expert = bool(_EXPERT_RE.search(tail))
        r = _role(tail)
        if r is None:
            clean = re.sub(r"\.(weight|bias)$", "", tail)
            # An expert tensor is a BLOCKER even when its exact name is unfamiliar, so record it as
            # an expert rather than filing it under "unrecognised" and losing the MoE signal.
            (expert_roles if is_expert else unknown).add(clean)
            continue
        (expert_roles if is_expert else roles).add(r)
        if tail.endswith(".bias"):
            has_bias.add(r)
    return {"roles": roles, "expert_roles": expert_roles, "unknown": unknown,
            "biases": has_bias, "n_blocks": n_blocks}


# ---- known families ------------------------------------------------------------------------------
# Deliberately small: these are the layouts a llama.cpp-style runtime already has a builder for, which
# is the only question this tool answers. `biases` is the set of roles that carry a bias tensor.
DENSE_CORE = {"attn_q", "attn_k", "attn_v", "attn_out", "attn_norm",
              "ffn_gate", "ffn_up", "ffn_down", "ffn_norm"}
MOE_CORE = {"attn_q", "attn_k", "attn_v", "attn_out", "attn_norm", "ffn_norm", "router"}
# MLA: latent q/kv projections replace q/k/v entirely (deepseek_v2/v3, glm4_moe, glm_moe_dsa, hy_v4)
MLA_ATTN = {"attn_q_a", "attn_q_b", "attn_q_a_norm", "attn_kv_a", "attn_kv_b",
            "attn_kv_a_norm", "attn_out", "attn_norm", "ffn_norm"}
FAMILIES = {
    "llama":     {"roles": DENSE_CORE, "biases": set(), "moe": False},
    "qwen2":     {"roles": DENSE_CORE, "biases": {"attn_q", "attn_k", "attn_v"}, "moe": False},
    "qwen3":     {"roles": DENSE_CORE | {"attn_q_norm", "attn_k_norm"}, "biases": set(), "moe": False},
    "qwen2moe":  {"roles": MOE_CORE, "biases": {"attn_q", "attn_k", "attn_v"},
                  "moe": True, "experts": {"ffn_gate", "ffn_up", "ffn_down"}},
    "qwen3moe":  {"roles": MOE_CORE | {"attn_q_norm", "attn_k_norm"}, "biases": set(),
                  "moe": True, "experts": {"ffn_gate", "ffn_up", "ffn_down"}},
    "llama-moe": {"roles": MOE_CORE, "biases": set(),
                  "moe": True, "experts": {"ffn_gate", "ffn_up", "ffn_down"}},
    # MLA, dense FFN (rare but real)
    "mla-dense": {"roles": MLA_ATTN | {"ffn_gate", "ffn_up", "ffn_down"}, "biases": set(),
                  "moe": False},
    # MLA + MoE: DeepSeek-V2/V3 and friends
    "deepseek-mla-moe": {"roles": MLA_ATTN | {"router"}, "biases": set(),
                         "moe": True, "experts": {"ffn_gate", "ffn_up", "ffn_down"}},
    # + the DeepSeek-V3.2 lightning indexer (GLM-5.3 glm_moe_dsa, Tencent hy_v4)
    "dsa-mla-moe": {"roles": MLA_ATTN | {"router", "attn_indexer"}, "biases": set(),
                    "moe": True, "experts": {"ffn_gate", "ffn_up", "ffn_down"}},
}


def twin(fp, hparams=None):
    """Nearest known family + the exact deltas + anything that makes 'twin' unsafe to act on."""
    hparams = hparams or {}
    best, best_score, best_diff = None, None, None
    is_moe = bool(fp["expert_roles"])
    for fam, spec in FAMILIES.items():
        if spec.get("moe", False) != is_moe:
            continue                       # never score a dense model against an MoE family, or vice versa
        missing = spec["roles"] - fp["roles"]
        extra = fp["roles"] - spec["roles"]
        bias_delta = spec["biases"] ^ (fp["biases"] & (spec["biases"] | fp["biases"]))
        exp_delta = (spec.get("experts", set()) ^ fp["expert_roles"]) if spec.get("moe") else set()
        score = (len(missing) * 2 + len(extra) * 2 + len(bias_delta)
                 + len(exp_delta) * 2 + len(fp["unknown"]))
        if best_score is None or score < best_score:
            best, best_score = fam, score
            best_diff = {"missing": sorted(missing), "extra": sorted(extra),
                         "bias_delta": sorted(bias_delta), "expert_delta": sorted(exp_delta),
                         "unknown": sorted(fp["unknown"])}

    # A gap is NOT "unsupported forever" -- it is the specific work that would support it. Refusing a
    # silent mis-build is the point; refusing to say what it would take is just unhelpful.
    gaps = []
    if best is None:
        gaps.append({"what": "no known family of this kind (dense/MoE) to compare against",
                     "to_support": "add a family entry describing this block's roles, then reuse or "
                                   "write the matching builder"})
    seen = set()
    for key, label, how in (
        ("value_expert_count", "value experts (MoVA)",
         "attention-side expert routing: per-layer attn_v_gate + attn_v_exps, selected by "
         "n_value_expert/n_value_expert_used. No current family models it -- extend the dense "
         "builder with a gated value path, then add it here as its own family"),
        ("expert_count", "experts",
         "route the FFN through the expert tensors and the router (an MoE family already covers "
         "the common shapes)"),
    ):
        for k, v in hparams.items():
            if k in seen or not k.endswith(key) or not isinstance(v, (int, float)) or not v:
                continue
            seen.add(k)                    # a key matches the most specific rule only
            gaps.append({"what": f"{label}: {k}={v} -- a plain twin would silently mis-build this",
                         "to_support": how})
    if fp["unknown"]:
        names = ", ".join(sorted(fp["unknown"])[:6])
        gaps.append({"what": "per-block tensors no family accounts for: " + names,
                     "to_support": "decide each tensor's role, add it to _ROLE_PATTERNS, and extend "
                                   "the nearest family (or add one) so the layout scores exactly"})
    blockers = [g["what"] for g in gaps]        # back-compat for existing callers

    exact = best_score == 0 and not blockers
    return {"twin": best, "exact": exact, "score": best_score, "diff": best_diff,
            "gaps": gaps, "blockers": blockers, "n_blocks": fp["n_blocks"]}


# ---- sources ---------------------------------------------------------------------------------------
def names_from_gguf(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_calc import read_gguf_meta
    meta = read_gguf_meta(path)
    return meta.get("_tensor_names") or [], meta


def names_from_hf(model):
    """safetensors index only — no weights downloaded."""
    idx = None
    if os.path.isdir(model):
        for cand in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            p = os.path.join(model, cand)
            if os.path.exists(p):
                idx = json.load(open(p))
                break
        cfg_p = os.path.join(model, "config.json")
        cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
        if idx is None:                      # single-shard: read the safetensors header only
            p = os.path.join(model, "model.safetensors")
            if os.path.exists(p):
                import struct
                with open(p, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(n))
                return [k for k in hdr if k != "__metadata__"], cfg
    else:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(model, "config.json")))
        try:
            idx = json.load(open(hf_hub_download(model, "model.safetensors.index.json")))
        except Exception:
            idx = None
        if idx is None:
            raise SystemExit("no safetensors index — pass a local dir, or a GGUF with --gguf")
        return list(idx.get("weight_map", {})), cfg
    return list((idx or {}).get("weight_map", {})), cfg


def report(res, arch_name=None):
    out = []
    if arch_name:
        out.append(f"arch: {arch_name}")
    out.append(f"layers: {res['n_blocks']}")
    if res["twin"] is None:
        out.append("layout: no known family of this kind to compare against")
        for g in res.get("gaps", []):
            out.append(f"   GAP: {g['what']}")
            out.append(f"        to support: {g['to_support']}")
        return "\n".join(out)
    if res["exact"]:
        out.append(f"layout: EXACT twin of `{res['twin']}` -- a runtime port is boilerplate "
                   f"(arch enum + name + tensor map, reuse the existing builder)")
    else:
        out.append(f"layout: nearest `{res['twin']}` (distance {res['score']}) -- NOT a drop-in")
        d = res["diff"]
        for k, label in (("missing", "missing vs that family"), ("extra", "extra vs that family"),
                         ("bias_delta", "bias mismatch"), ("expert_delta", "expert tensor mismatch"),
                         ("unknown", "unrecognised")):
            if d.get(k):
                out.append(f"   {label}: {', '.join(d[k])}")
    for g in res.get("gaps", []):
        out.append(f"   GAP: {g['what']}")
        out.append(f"        to support: {g['to_support']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", help="a GGUF to fingerprint")
    ap.add_argument("--model", help="an HF repo id or local model dir to fingerprint")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    if not (a.gguf or a.model):
        ap.error("pass --gguf or --model")

    if a.gguf:
        names, meta = names_from_gguf(a.gguf)
        arch = meta.get("general.architecture")
        hp = {k: v for k, v in meta.items() if not k.startswith("_")}
    else:
        names, cfg = names_from_hf(a.model)
        arch = cfg.get("model_type") or (cfg.get("architectures") or [None])[0]
        hp = {k: v for k, v in cfg.items() if isinstance(v, (int, float))}

    if not names:
        raise SystemExit("no tensor names found — cannot fingerprint")
    res = twin(fingerprint(names), hp)
    print(json.dumps({**res, "arch": arch}, indent=2, default=list) if a.json else report(res, arch))


if __name__ == "__main__":
    main()
