# Onboarding a custom architecture (the Spark2_5 case)

Most models are a known arch (llama/qwen/mistral/mixtral/deepseek/glm) and one-shot with no work. A
**custom architecture** — its `config.json` has `model_type` set to something new and an `auto_map`
pointing at its own `modeling_*.py` — needs a short, GENERIC onboarding. This is the playbook, written
against the first one we did: **Spark-X2.5-4B** (`Spark2_5ForCausalLM`, XHToken).

## The one command: `pollard-onboard`
This whole audit is automated — `pollard-onboard --model <repo-or-dir>` pulls the config + tensor names,
checks coverage against Pollard's matchers, flags custom features, and prints a verdict
(READY / LIKELY-READY / NEEDS-ONBOARDING). Add `--contribute` and it writes a PR-ready
`onboarding/<model_type>.md` + the exact git/gh steps to submit it, so each onboarding grows the tool.
The rest of this doc is what that tool checks (and what to do when it says NEEDS-ONBOARDING).

## Rule: audit the real tensor names first
Don't guess from the arch name. Pull the actual names and decide from them:

    curl -sL https://huggingface.co/<repo>/resolve/main/model.safetensors.index.json \
      | python -c "import json,sys,re;print(sorted(set(re.sub(r'\.\d+\.','.N.',k) for k in json.load(sys.stdin)['weight_map'])))"

Spark2_5's (290 tensors) came back as:

    model.embedding.weight                         # tied embeddings (no lm_head), NOT embed_tokens
    model.layers.N.input_layernorm.weight
    model.layers.N.post_attention_layernorm.weight
    model.layers.N.self_attn.q_k_v_proj.weight     # FUSED QKV (one tensor)
    model.layers.N.self_attn.g_proj.weight         # head-wise attention output GATE (sigmoid) — new
    model.layers.N.self_attn.out_proj.weight
    model.layers.N.mlp.{gate,up,down}_proj.weight  # standard SwiGLU names (activation is GELU)
    model.norm.weight

Other custom features (from the modeling code / Grok's read): hybrid attention (3 sliding + 1 full,
window 512, repeating every 4 layers), layer-dependent RoPE (sliding: partial_rotary 1.0/θ 10000;
full: 0.25/θ 5e6), dense (no experts), tied embeddings.

## What Pollard already handles generically (no per-arch code)
The arch-agnostic work pays off here — these needed **no** Spark-specific hacks:
- **Fused QKV** (`q_k_v_proj`): the `ATTN_PROJ` matcher (`self_attn\.[a-z_]*proj[a-z0-9_]*`) catches it, and
  a fused tensor is one tensor so there's no "don't split the group" problem.
- **`out_proj`, standard MLP projs**: matched by `ATTN_PROJ` / `FFN_PROJ`.

## What onboarding added (all GENERIC — helps any future custom arch)
- **`--trust-remote-code {auto,on,off}`** on `pollard-export` / `-mlx` / `-mx` and the `pollard` one-shot.
  `auto` reads `config.json` as plain JSON (no code executed) and enables remote code ONLY when an
  `auto_map` is present. Threaded to `GPTQModel.load`, `llm-compressor oneshot` (`trust_remote_code_model`,
  tolerated if absent), and `mlx_lm.convert` (tolerated if absent). Shared helper: `pollard_workspace.resolve_trust_remote_code`.
- **Generic embedding name**: MLX now protects any `*embed*` tensor (Spark's `model.embedding`, not just
  `embed_tokens`) at high bits.
- **Attention output gate protected**: `self_attn.*g_proj` is selection-critical (like a router) and is
  pinned high in the GPTQ dynamic map and MLX plan, at every layer. No-op for arches without it.

## Per-lane support status for Spark2_5
| Lane | Status |
|---|---|
| **GGUF** | ✅ upstream: llama.cpp **PR #27868** (merged ~2026-09-06) adds `spark2_5` (fused QKV, sigmoid gates, hybrid layers, dual RoPE). Build needs a llama.cpp at/after that commit. Then Pollard's imatrix + automap run as usual — verify automap's role map covers the GGUF tensor names it emits. |
| **GPTQ / MX** | ✅ code-ready (trust_remote_code + gate/embed handling). Needs a run on the CUDA box to confirm the custom modeling loads + quantizes. |
| **MLX** | ⚠️ code-ready, but MLX conversion needs the arch supported inside `mlx_lm` — a truly custom block may not convert until mlx_lm (or a community port) adds it. Try it; if `convert` errors on the arch, that's an mlx_lm gap, not ours. |
| **EXL3** | ❌ exllamav3 doesn't know `spark2_5` yet — skip until upstream adds it (don't force a recipe). |

## Verify, always
Custom code + custom attention means the reconstruction check matters more, not less: gate every build
with `pollard-verify` (decode-vs-source + end-to-end forward) and, on the served side, `pollard-serve-eval`.
Never trust a proxy metric.
