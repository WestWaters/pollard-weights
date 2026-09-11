#!/usr/bin/env python3
"""pollard-onboard — audit a NEW model architecture and emit a ready-to-submit contribution.

When a model's `config.json` has a `model_type` Pollard hasn't seen (an `auto_map` = custom code),
this does the audit-first step automatically: pulls the REAL tensor names, checks them against
Pollard's arch-agnostic matchers, flags what needs attention (custom gates, embedding names, MoE/MLA
signals, remote code), and writes a findings report PLUS a filled-in contribution stub you can PR back
so the next person's model of that family one-shots. That's how Pollard scales — every onboarding feeds
the tool.

  pollard-onboard --model XHToken/Spark-X2.5-4B                 # audit + print findings
  pollard-onboard --model <repo-or-dir> --contribute            # also write a PR-ready contribution file
  pollard-onboard --model <repo-or-dir> --contribute --out onboarding/spark2_5.md

Read-only and network-light (config.json + the safetensors index only). No weights downloaded, no code
executed. See notes/custom-arch-onboarding.md for the full playbook."""
import argparse
import json
import os
import re
import sys

# Pollard's arch-agnostic matchers (kept in sync with pollard_export).
ATTN_PROJ = r"self_attn\.[a-z_]*proj[a-z0-9_]*"
FFN_PROJ = r"(?:mlp|block_sparse_moe)(?:\.experts\.(?:\d+|N))?\.(?:gate|up|down)_proj"   # N = collapsed layer/expert index (tensor_patterns)
# DeepSeek-V3.2-family additions (GLM-5.3 `glm_moe_dsa`, Tencent Hy4 `hy_v4`), measured on 744B/750B checkpoints 2026-09:
FUSED_EXPERTS = r"\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)$"   # stacked [E, ...] tensors, no per-expert index
SHARED_EXPERTS = r"\.shared_experts\.(?:gate|up|down)_proj"
ROUTER = r"\.mlp\.gate$"                                   # router weight (`mlp.gate.weight`) — protect, do not quantize
DSA_INDEXER = r"self_attn\.indexer\.(?:wq_b|wk|weights_proj|k_norm)"   # lightning-indexer projections (small; keep high)
ATTN_GATE = r"self_attn\.(?:linear_gate|gate_proj)$"      # attention output gate (gated MLA) — treat like a router
MTP_PREFIX = r"^model\.mtp_layers\.N\."                   # MTP side model stored outside model.layers
MTP_GLUE = r"\.(?:eh_proj|enorm|hnorm|shared_head\.norm|final_layernorm)$"   # MTP projection + norms: keep bf16 (vLLM builds eh_proj unquantized)
KNOWN_TYPES = {"llama", "qwen2", "qwen3", "mistral", "mixtral", "gemma", "gemma2", "gemma3",
               "phi3", "deepseek_v2", "deepseek_v3", "glm4", "glm4_moe", "cohere", "starcoder2"}


def _fetch(model_id, fname):
    if os.path.isdir(model_id):
        p = os.path.join(model_id, fname)
        return open(p) if os.path.exists(p) else None
    try:
        from huggingface_hub import hf_hub_download
        return open(hf_hub_download(model_id, fname))
    except Exception:
        return None


def tensor_patterns(model_id):
    """Unique per-layer tensor patterns (layer index collapsed to N) from the safetensors index."""
    f = _fetch(model_id, "model.safetensors.index.json")
    if not f:
        return []
    try:
        wm = json.load(f).get("weight_map", {})
    except Exception:
        return []
    return sorted(set(re.sub(r"\.\d+\.", ".N.", k) for k in wm))


def audit(model_id):
    cfg = {}
    f = _fetch(model_id, "config.json")
    if f:
        try:
            cfg = json.load(f)
        except Exception:
            pass
    mtype = cfg.get("model_type", "?")
    custom = bool(cfg.get("auto_map"))
    known = mtype in KNOWN_TYPES
    pats = tensor_patterns(model_id)
    weight_pats = [p for p in pats if p.endswith(".weight")]
    findings = {"model": model_id, "model_type": mtype, "arch": (cfg.get("architectures") or ["?"])[0],
                "custom_code": custom, "known_type": known, "n_tensor_patterns": len(pats)}

    covered, unmatched, flags = [], [], []
    fused = bool([p for p in pats if re.search(FUSED_EXPERTS, p)])   # fused expert tensors carry no ".weight" suffix
    for p in weight_pats:
        body = p[:-len(".weight")]
        if re.search(ATTN_PROJ, body) or re.search(FFN_PROJ, body) or re.search(SHARED_EXPERTS, body) or \
           re.search(DSA_INDEXER, body) or re.search(ATTN_GATE, body) or re.search(ROUTER, body) or re.search(MTP_GLUE, body) or \
           re.search(r"(input_layernorm|post_attention_layernorm|\bnorm\b|layernorm)", body):
            covered.append(p)
        elif "embed" in body or body.endswith("lm_head"):
            covered.append(p)                                   # vocab carriers (handled generically)
        else:
            unmatched.append(p)
    if fused:
        flags.append("FUSED EXPERTS: routed experts stored as stacked tensors (`experts.gate_up_proj` [E, 2I, H], "
                     "`experts.down_proj` [E, H, I]) — per-expert lanes must slice dim 0; FFN_PROJ never matches them")
    if any(re.search(DSA_INDEXER, p) for p in weight_pats):
        flags.append("DSA sparse-attention indexer (wq_b/wk/weights_proj/k_norm) — small, keep at high precision; "
                     "its per-layer key cache adds to KV bytes (pollard-calc accounts for it via index_head_dim)")
    if any(re.search(ATTN_GATE, p) for p in weight_pats):
        flags.append("gated attention output (linear_gate/gate_proj under self_attn) — protect like a router")
    if any(re.search(MTP_PREFIX, p) for p in pats):
        flags.append("MTP side model under `model.mtp_layers.N.*` (not `model.layers.<last+1>`) — separate prefix for export/skip rules")
    if any(".hc_" in p or "hc_head" in p for p in pats):
        flags.append("hyperconnections (hc_*): the residual is hc_mult streams — activation memory and band hand-off scale by hc_mult; "
                     "hc tensors are tiny, keep fp32/bf16")
    # heuristic flags for the onboarder
    if any("self_attn" in p and re.search(r"\bg_proj\b|gate", p) for p in weight_pats):
        flags.append("attention output GATE present (protect it high — like a router)")
    if any("q_k_v_proj" in p or "qkv" in p for p in weight_pats):
        flags.append("FUSED QKV (one tensor — allocate as a single unit)")
    if any("embedding" in p and "embed_tokens" not in p for p in weight_pats):
        flags.append("non-standard embedding name (Pollard now matches any *embed*)")
    if cfg.get("kv_lora_rank") is not None:
        flags.append("MLA attention (q_a/q_b/kv_a/kv_b) — covered by ATTN_PROJ")
    if any(cfg.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        flags.append("MoE — use the MoE dynamic map / automap MoE path")
    if custom:
        flags.append("custom modeling code — run lanes with --trust-remote-code (auto-detected)")
    # A shape-and-name audit cannot see a wrong RoPE pairing: the model loads, generates, and measures
    # the wrong thing. Hy4-preview scored 5.02 nats instead of 1.855 that way, with routing mass off by
    # 24 points, and K2-Horizon nearly shipped NORM where the reference says NEOX. So every onboarding
    # is told to gate the reference forward before capturing anything from it.
    mt = (cfg.get("model_type") or "").lower()
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pollard_refcheck import KNOWN_DEFECTS
    except Exception:                                                      # noqa: BLE001
        KNOWN_DEFECTS = {}
    if mt in KNOWN_DEFECTS:
        d = KNOWN_DEFECTS[mt]
        flags.append(f"KNOWN MODEL-CODE DEFECT for `{mt}` — {d['symptom']} Run "
                     f"`pollard-refcheck --model <this> --calib rows.txt --fix` BEFORE any capture; "
                     f"measurements taken without it are invalid ({d['credit']})")
    else:
        flags.append("BEFORE measuring anything: `pollard-refcheck --model <this> --calib rows.txt`. "
                     "A wrong RoPE pairing (rotate-half vs interleaved) or a wrong rope_type passes "
                     "every shape and name check, loads, generates — and silently invalidates every "
                     "Hessian, sensitivity ranking and routing statistic. ~1.5-2.5 nats on in-domain "
                     "rows is healthy; ~5, or loss RISING along the sequence, means positions are "
                     "scrambled")
    findings.update(covered=len(covered), unmatched=unmatched, flags=flags, patterns=pats)
    return findings


def verdict(f):
    if f["known_type"] and not f["unmatched"]:
        return "READY", "known arch, all tensors covered — one-shots as-is."
    if not f["unmatched"]:
        return "LIKELY-READY", ("custom code but every tensor maps to Pollard's generic matchers — try the "
                                "one-shot with --trust-remote-code; verify with pollard-verify.")
    return "NEEDS-ONBOARDING", f"{len(f['unmatched'])} tensor pattern(s) unmatched — see below."


def contribution_md(f, v):
    lines = [f"# Onboarding: `{f['model_type']}` ({f['arch']})", "",
             f"- **Example model:** {f['model']}",
             f"- **Custom code (auto_map):** {'yes — needs --trust-remote-code' if f['custom_code'] else 'no'}",
             f"- **Verdict:** {v[0]} — {v[1]}", "",
             "## Tensor patterns", "```", *f["patterns"], "```", ""]
    if f["flags"]:
        lines += ["## Flags / features", *[f"- {x}" for x in f["flags"]], ""]
    if f["unmatched"]:
        lines += ["## Unmatched tensors (need a mapping)", *[f"- `{x}`" for x in f["unmatched"]],
                  "", "Suggested: extend the arch-agnostic matcher or add the role mapping in "
                  "`pollard_export.py` / `pollard_mlx.py`, then re-run `pollard-onboard` until clean.", ""]
    lines += ["## Per-lane status (fill in after a build)",
              "| Lane | Works? | Notes |", "|---|---|---|",
              "| GGUF (llama.cpp) |  | needs converter support for this arch (check upstream PRs) |",
              "| GPTQ / MX |  | trust_remote_code auto; verify custom load |",
              "| MLX |  | needs the arch in mlx_lm |",
              "| EXL3 |  | needs exllamav3 support |", "",
              "_Generated by `pollard-onboard`. Verify every build with `pollard-verify` before reporting works=yes._"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id or local model dir")
    ap.add_argument("--contribute", action="store_true", help="also write a PR-ready contribution file")
    ap.add_argument("--out", help="contribution file path (default onboarding/<model_type>.md)")
    a = ap.parse_args()

    f = audit(a.model)
    v = verdict(f)
    print(f"== pollard-onboard :: {a.model}")
    print(f"   model_type={f['model_type']}  arch={f['arch']}  custom_code={f['custom_code']}  "
          f"known={f['known_type']}  tensors={f['n_tensor_patterns']}")
    print(f"   coverage: {f['covered']} matched, {len(f['unmatched'])} unmatched")
    for fl in f["flags"]:
        print(f"   • {fl}")
    if f["unmatched"]:
        print("   UNMATCHED:")
        for p in f["unmatched"]:
            print(f"     - {p}")
    print(f"\n   VERDICT: {v[0]} — {v[1]}")

    if a.contribute:
        out = a.out or os.path.join("onboarding", f"{f['model_type']}.md")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w", encoding="utf-8").write(contribution_md(f, v))
        print(f"\n   contribution written -> {out}")
        print("   Contribute it back so the next person's model one-shots:")
        print(f"     git checkout -b onboard-{f['model_type']} && git add {out} && \\")
        print(f"       git commit -m 'onboard: {f['model_type']} arch findings' && \\")
        print("       gh pr create --repo WestWaters/pollard-weights --fill")


if __name__ == "__main__":
    main()
