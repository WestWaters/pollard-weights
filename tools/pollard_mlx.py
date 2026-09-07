#!/usr/bin/env python3
"""pollard-mlx — emit an MLX (Apple Silicon) mixed-bit model carrying Pollard's measured allocation.

The Apple deployment lane: llama.cpp/GGUF is the CPU/portable flagship, GPTQ (pollard-export) is the
vLLM/SGLang server lane, and THIS is the on-device Apple lane (M-series, ANE/Metal via MLX). Same
Pollard brain — detect arch, allocate bits by measured sensitivity under a size budget — a different
EMITTER. MLX group-quant supports mixed bits via a per-module `quant_predicate`, so the profile's hot
modules stay high (8-bit) and the tolerant body is crushed (4-bit); router + shared experts pinned high.

  pollard-mlx --model Qwen/Qwen2.5-1.5B-Instruct --out ./Qwen2.5-1.5B-Pollard-MLX
  pollard-mlx --model <hf> --sensitivity q15b.sensitivity.json --hot-frac 0.35 --plan-only
  # then run on a Mac:  mlx_lm.generate --model ./...-Pollard-MLX --prompt "hi"

MLX group-quant bits are {2,3,4,6,8}; we mix 4/8 (like the GPTQ lane) by default. --plan-only needs no
deps; the actual convert needs `pip install mlx_lm` on Apple Silicon (this box has `mlx`)."""
import argparse, json, re, sys

HIGH, LOW = 8, 4


def detect_moe(model_id, layers_hint=0):
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        n = getattr(cfg, "num_hidden_layers", 0) or layers_hint
        moe = any(getattr(cfg, k, 0) for k in
                  ("num_experts", "num_local_experts", "n_routed_experts"))
        return bool(moe), int(n)
    except Exception:
        return False, layers_hint


def allocate(sens, n_layers, hot_frac):
    """Pollard profile -> {layer: {"attn":bits, "ffn":bits}}. Top `hot_frac` of layers stay HIGH."""
    ffn = {int(k): float(v) for k, v in (sens.get("ffn") or {}).items()}
    attn = {int(k): float(v) for k, v in (sens.get("attn") or {}).items()}
    if not ffn and not attn:
        return {i: {"attn": LOW, "ffn": LOW} for i in range(n_layers)}

    def hot(d):
        k = max(1, int(round(hot_frac * len(d))))
        return set(sorted(d, key=lambda i: d[i], reverse=True)[:k])
    ha, hf = hot(attn or ffn), hot(ffn or attn)
    return {i: {"attn": HIGH if i in ha else LOW, "ffn": HIGH if i in hf else LOW}
            for i in range(n_layers)}


def layer_of(path):
    m = re.search(r"\.layers\.(\d+)\.", path)
    return int(m.group(1)) if m else None


def bits_for(path, alloc, is_moe, gsize):
    """Per-module bit-width for MLX (returns a {'bits','group_size'} dict, or False to skip).
    Mirrors the Pollard policy: attn follows the attn profile; the FFN/expert body follows ffn;
    MoE router (`.gate`, not `gate_proj`) and shared experts are pinned HIGH; embeddings kept HIGH."""
    if "embed" in path or path.endswith("lm_head"):    # embed_tokens (llama) OR embedding (Spark2_5, tied)
        return {"bits": HIGH, "group_size": gsize}      # vocab carriers: keep high
    if is_moe and (re.search(r"\.(mlp|block_sparse_moe)\.gate$", path)
                   or "shared_expert" in path):
        return {"bits": HIGH, "group_size": gsize}      # router + shared: selection integrity
    if "self_attn" in path and path.endswith("g_proj"):  # head-wise attn output gate (Spark2_5): selection-critical
        return {"bits": HIGH, "group_size": gsize}
    li = layer_of(path)
    if li is None or li not in alloc:
        return {"bits": LOW, "group_size": gsize}
    if "self_attn" in path:
        return {"bits": alloc[li]["attn"], "group_size": gsize}
    if "mlp" in path or "block_sparse_moe" in path or "experts" in path:
        return {"bits": alloc[li]["ffn"], "group_size": gsize}
    return {"bits": LOW, "group_size": gsize}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (else uniform 4-bit)")
    ap.add_argument("--out", help="output MLX model dir (required unless --plan-only)")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--hot-frac", type=float, default=0.35)
    ap.add_argument("--group-size", type=int, default=64, help="MLX quant group size (default 64)")
    ap.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"],
                    help="run a model's own modeling code (custom archs); 'auto' = only if config has auto_map. "
                         "Note: MLX needs the arch supported in mlx_lm to convert a truly custom model.")
    ap.add_argument("--moe", dest="moe", action="store_true", default=None,
                    help="force MoE policy (default: auto-detect)")
    ap.add_argument("--dense", dest="moe", action="store_false", help="force dense policy")
    ap.add_argument("--plan-only", action="store_true", help="print the per-module bit plan, build nothing")
    a = ap.parse_args()

    sens = json.load(open(a.sensitivity)) if a.sensitivity else {}
    n_layers = a.layers or int(sens.get("layers") or 0)
    auto_moe, det = detect_moe(a.model, n_layers)
    n_layers = n_layers or det
    if not n_layers:
        sys.exit("ERROR: could not read layer count — pass --layers.")
    is_moe = auto_moe if a.moe is None else a.moe
    alloc = allocate(sens, n_layers, a.hot_frac)
    avg = sum(g["attn"] + g["ffn"] for g in alloc.values()) / (2 * len(alloc))
    kind = "MoE" if is_moe else "dense"
    print(f"== pollard-mlx :: {a.model}  [{kind}]  {n_layers} layers · mix {LOW}/{HIGH}-bit "
          f"avg {avg:.2f} · group_size {a.group_size}")
    if is_moe:
        print("   MoE: router + shared experts pinned 8-bit; cold experts 4-bit")

    if a.plan_only:
        # representative module paths for the first 2 layers + the vocab carriers
        print("   per-module plan (sample):")
        samples = ["model.embed_tokens"]
        for i in range(min(2, n_layers)):
            samples += [f"model.layers.{i}.self_attn.q_proj",
                        f"model.layers.{i}.mlp.experts.0.gate_proj" if is_moe
                        else f"model.layers.{i}.mlp.gate_proj",
                        f"model.layers.{i}.mlp.down_proj"]
            if is_moe:
                samples.append(f"model.layers.{i}.mlp.gate")
        for p in samples:
            print(f"     {p:52s} -> {bits_for(p, alloc, is_moe, a.group_size)['bits']}-bit")
        return

    if not a.out:                                          # no --out -> organized workspace path
        try:
            import pollard_workspace as ws
            a.out = ws.resolve_out(a.model, "mlx", tag=f"{LOW}.{HIGH}")
            print(f"   (no --out) -> workspace: {a.out}")
        except Exception:
            sys.exit("ERROR: --out required unless --plan-only.")
    try:
        from mlx_lm import convert
    except Exception:
        sys.exit("ERROR: mlx_lm not installed. On Apple Silicon: pip install mlx_lm ; then rerun.")

    def predicate(path, module, config=None):              # mlx_lm calls with (path, module)
        return bits_for(path, alloc, is_moe, a.group_size)

    import pollard_workspace as ws
    trc = ws.resolve_trust_remote_code(a.model, a.trust_remote_code)
    ckw = dict(mlx_path=a.out, quantize=True, q_bits=LOW, q_group_size=a.group_size, quant_predicate=predicate)
    try:
        convert(a.model, trust_remote_code=trc, **ckw)     # newer mlx_lm accepts it
    except TypeError:
        if trc:
            print("   (this mlx_lm lacks trust_remote_code; a truly custom arch may not convert)")
        convert(a.model, **ckw)
    try:
        import pollard_workspace as ws
        ws.record_build(a.model, "mlx", a.out, tag=f"{LOW}.{HIGH}")
    except Exception:
        pass
    print(f"wrote MLX model -> {a.out}\n"
          f"  run:  mlx_lm.generate --model {a.out} --prompt \"hello\"\n"
          f"  (mixed {LOW}/{HIGH}-bit, Pollard-allocated; runs on Apple Silicon via MLX/Metal.)")


if __name__ == "__main__":
    main()
