#!/usr/bin/env python3
"""pollard-export — take a Pollard sensitivity profile and emit a vLLM/SGLang-loadable
GPTQ checkpoint with Pollard's measured allocation carried over as a gptqmodel `dynamic`
4/8-bit mix. This is the GPU-runtime lane: llama.cpp/ik_llama.cpp gets the 1-bit trellis
flagship; vLLM/SGLang get a sensitivity-allocated 4/8 GPTQ that runs Marlin-accelerated.

Why 4/8 and not 1-2: vLLM/SGLang's fast Marlin kernel supports ONLY 4-bit and 8-bit, so
the measured mix lives there — sensitive modules kept at 8-bit, the tolerant body crushed
to 4-bit, allocated by Pollard's profile. gptqmodel's per-module `dynamic` is "fully
integrated into vLLM"; SGLang loads GPTQ but its mixed-bit is fragile (layer fusion), so
`--uniform` emits a plain W4 for SGLang.

Hard runtime constraints baked in (verified Aug 2026):
  * NEVER split bits inside a fused group: q/k/v_proj share one bit-width; gate/up_proj
    share one. (o_proj, down_proj are free.) Marlin fuses these.
  * group_size 128, desc_act False, sym True  -> the well-trodden Marlin fast path.
  * bits in {4, 8} only.

    pollard-export --model Qwen/Qwen2.5-7B-Instruct --sensitivity qwen7b.sensitivity.json \\
        --calib calib.txt --out ./Qwen2.5-7B-Instruct-Pollard-GPTQ
    # then: vllm serve ./...-Pollard-GPTQ --quantization gptq
"""
import argparse, json, re, sys

HIGH, LOW = 8, 4                       # Marlin supports only these
# module groups within a decoder layer. q/k/v are ONE fused unit; gate/up are ONE fused unit.
FUSED = {"attn_qkv": ["q_proj", "k_proj", "v_proj"], "gate_up": ["gate_proj", "up_proj"]}
FREE = {"attn_o": ["o_proj"], "ffn_down": ["down_proj"]}


def allocate(sens, n_layers, hot_frac):
    """Pollard profile -> {layer: {group: bits}}. The most-sensitive `hot_frac` of layers
    keep their group at 8-bit; the rest go 4-bit. Fused groups get ONE bit-width (constraint).
    'attn' sensitivity drives q/k/v+o; 'ffn' sensitivity drives gate/up+down."""
    ffn = {int(k): float(v) for k, v in (sens.get("ffn") or {}).items()}
    attn = {int(k): float(v) for k, v in (sens.get("attn") or {}).items()}
    if not ffn and not attn:                       # no profile -> uniform W4 (still valid)
        return {i: {"attn": LOW, "ffn": LOW} for i in range(n_layers)}

    def hot_layers(d):
        k = max(1, int(round(hot_frac * len(d))))
        return set(sorted(d, key=lambda i: d[i], reverse=True)[:k])
    hot_attn, hot_ffn = hot_layers(attn or ffn), hot_layers(ffn or attn)
    return {i: {"attn": HIGH if i in hot_attn else LOW,
                "ffn": HIGH if i in hot_ffn else LOW} for i in range(n_layers)}


def dynamic_config(alloc):
    """gptqmodel `dynamic` regex map. Base is LOW (4-bit); we add 8-bit overrides for the
    hot groups. q/k/v+o share the attn bit; gate/up+down share the ffn bit (fused-safe)."""
    dyn = {}
    for i, g in alloc.items():
        if g["attn"] == HIGH:
            dyn[rf".*\.layers\.{i}\.self_attn\.(q|k|v|o)_proj"] = {"bits": HIGH}
        if g["ffn"] == HIGH:
            dyn[rf".*\.layers\.{i}\.mlp\.(gate|up|down)_proj"] = {"bits": HIGH}
    return dyn


def avg_bits(alloc):
    b = [g["attn"] for g in alloc.values()] + [g["ffn"] for g in alloc.values()]
    return sum(b) / len(b) if b else LOW


def kv_note(model_id):
    """Victor's point: vLLM pre-allocates KV hard (gpu_memory_utilization ~0.9, paged
    attention) — far more headroom than llama.cpp. Flag it so the target VRAM is realistic."""
    return ("vLLM reserves KV cache up front (gpu_memory_utilization ~0.9). Budget "
            "weights + a large KV pool + activations, not just the weight bytes — a model "
            "that fits in GGUF on a card can OOM in vLLM. Lower --gpu-memory-utilization or "
            "--max-model-len if it won't fit; run one load test before publishing the fit.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="HF model dir or id (FP16/BF16)")
    ap.add_argument("--sensitivity", help="Pollard sensitivity.json (pollard-probe/-sensitivity)")
    ap.add_argument("--calib", required=True, help="calibration text (one sample per line or a corpus)")
    ap.add_argument("--out", required=True, help="output dir for the GPTQ checkpoint")
    ap.add_argument("--layers", type=int, default=0, help="n decoder layers (else read from config)")
    ap.add_argument("--hot-frac", type=float, default=0.35, help="fraction of layers kept at 8-bit")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--uniform", action="store_true", help="plain uniform W4 (for SGLang mixed-bit fragility)")
    ap.add_argument("--plan-only", action="store_true", help="print the allocation + dynamic map, build nothing")
    a = ap.parse_args()

    sens = json.load(open(a.sensitivity)) if a.sensitivity else {}
    n_layers = a.layers or int(sens.get("layers") or 0)
    if not n_layers:
        try:
            from transformers import AutoConfig
            n_layers = AutoConfig.from_pretrained(a.model).num_hidden_layers
        except Exception:
            sys.exit("ERROR: could not read layer count — pass --layers.")

    alloc = allocate(sens, n_layers, a.hot_frac)
    dyn = {} if a.uniform else dynamic_config(alloc)
    ab = LOW if a.uniform else avg_bits(alloc)
    print(f"== pollard-export :: {a.model}")
    print(f"   {n_layers} layers · {'UNIFORM W4 (SGLang-safe)' if a.uniform else f'4/8 dynamic mix, avg {ab:.2f} bits'}"
          f" · group_size {a.group_size} · desc_act False · sym True")
    print(f"   8-bit modules: {len(dyn)} groups (sensitivity-ranked hot set)")
    print(f"   NOTE (KV/memory): {kv_note(a.model)}")
    if a.plan_only:
        print("   dynamic map:")
        for k, v in list(dyn.items())[:8]:
            print(f"     {k}  -> {v}")
        if len(dyn) > 8:
            print(f"     … +{len(dyn)-8} more")
        return

    # ---- build with gptqmodel (runs on a CUDA box; produces a vLLM/SGLang GPTQ checkpoint)
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except Exception:
        sys.exit("ERROR: gptqmodel not installed here. Run this on the CUDA box: "
                 "pip install gptqmodel ; the checkpoint it writes loads in vLLM/SGLang.")
    calib = [ln.strip() for ln in open(a.calib, encoding="utf-8") if ln.strip()]
    qcfg = QuantizeConfig(bits=LOW, group_size=a.group_size, desc_act=False, sym=True,
                          dynamic=(dyn or None))
    model = GPTQModel.load(a.model, qcfg)
    model.quantize(calib)
    model.save(a.out)
    print(f"wrote GPTQ checkpoint -> {a.out}\n"
          f"  vLLM:   vllm serve {a.out} --quantization gptq\n"
          f"  SGLang: python -m sglang.launch_server --model-path {a.out} --quantization gptq\n"
          f"  (mixed 4/8 loads dynamic-accelerated on vLLM; use --uniform for SGLang.)")


if __name__ == "__main__":
    main()
