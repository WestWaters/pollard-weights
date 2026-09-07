#!/usr/bin/env python3
"""pollard-mlx — emit an MLX (Apple Silicon) model with Pollard's allocation. One of the output lanes:
GGUF / GPTQ / MLX / EXL3 / MX — same measured allocation, MLX's mixed 4/8-bit format.

MLX runs on Apple Metal via `mlx_lm`. Pollard's edge here is the same "spend bits where it matters":
a per-layer quant predicate keeps salient/protected layers at 8-bit and the cold body at 4-bit, driven
by the measured sensitivity profile — instead of a flat 4-bit convert.

  pollard-mlx --model Qwen/Qwen3-8B --out ./Qwen3-8B-Pollard-MLX --sensitivity sens.json   # emit
  pollard-mlx --model <hf> --plan-only                                                       # recipe only

Real emit needs `mlx_lm` on Apple Silicon (`pip install mlx-lm`); `--plan-only` needs nothing and prints
the recipe + avg bits. Verify a build by loading it: `mlx_lm.generate --model <out> --prompt ...`.
"""
import argparse, json, os, sys
import pollard_workspace as ws


def allocate(sens, n_layers, hot_frac):
    """Hottest `hot_frac` of layers -> 8-bit, the rest -> 4-bit (same hot/cold split as the other lanes)."""
    if sens:
        order = sorted(sens.items(), key=lambda kv: -float(kv[1]))
        return set(k for k, _ in order[:max(1, int(round(hot_frac * len(order))))])
    n_hot = max(1, int(round(hot_frac * n_layers)))
    return set(str(i) for i in list(range(n_hot // 2)) + list(range(n_layers - (n_hot - n_hot // 2), n_layers)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (fp16/bf16 source)")
    ap.add_argument("--out", help="output MLX dir (required unless --plan-only)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (else protect the ends)")
    ap.add_argument("--low-bits", type=int, default=4, help="body bits (default 4)")
    ap.add_argument("--high-bits", type=int, default=8, help="protected/salient-layer bits (default 8)")
    ap.add_argument("--group-size", type=int, default=64, help="MLX quant group size")
    ap.add_argument("--hot-frac", type=float, default=0.25, help="fraction of layers kept at high-bits")
    ap.add_argument("--layers", type=int, default=0, help="n decoder layers (else read from config)")
    ap.add_argument("--plan-only", action="store_true", help="print the recipe + avg bits, build nothing")
    a = ap.parse_args()

    n_layers = a.layers
    if not n_layers:
        try:
            from transformers import AutoConfig
            n_layers = getattr(AutoConfig.from_pretrained(a.model), "num_hidden_layers", 0)
        except Exception:
            n_layers = 0
    sens = {}
    if a.sensitivity and os.path.exists(a.sensitivity):
        s = json.load(open(a.sensitivity))
        sens = s.get("layers", s) if isinstance(s, dict) else {}

    hot = allocate(sens, n_layers or 32, a.hot_frac)
    frac_hi = len(hot) / (n_layers or 32)
    avg = round(a.low_bits * (1 - frac_hi) + a.high_bits * frac_hi, 2)
    print(f"== pollard-mlx :: {a.model}  ~{avg} bpw avg  ({len(hot)} layers @ {a.high_bits}-bit, "
          f"rest @ {a.low_bits}-bit, group {a.group_size})")
    print("   Pollard intent -> MLX mixed precision: protect measured-hot layers at high-bit, crush the body.")
    if a.plan_only:
        return
    if not a.out:
        a.out = ws.resolve_out(a.model, "mlx", tag=f"{a.low_bits}.{a.high_bits}-{avg}bpw")
        print(f"   (no --out) -> workspace: {a.out}")
    try:
        from mlx_lm import convert
    except Exception:
        sys.exit("ERROR: mlx_lm not installed here. On Apple Silicon: pip install mlx-lm ; then rerun. "
                 "(--plan-only works anywhere and prints the recipe.)")

    hi = set(hot)
    def predicate(path, module, _cfg):
        # per-layer bits: salient/protected layers -> high, else low
        import re
        m = re.search(r"layers\.(\d+)\.", path)
        bits = a.high_bits if (m and m.group(1) in hi) else a.low_bits
        return {"group_size": a.group_size, "bits": bits}

    print(f"   emitting via mlx_lm.convert -> {a.out}")
    convert(a.model, mlx_path=a.out, quantize=True, q_group_size=a.group_size,
            q_bits=a.low_bits, quant_predicate=predicate)
    ws.record_build(a.model, "mlx", a.out, tag=f"{a.low_bits}.{a.high_bits}", bpw=avg)
    print(f"wrote MLX model -> {a.out}\n  load:  mlx_lm.generate --model {a.out} --prompt 'The capital of France is'")


if __name__ == "__main__":
    main()
