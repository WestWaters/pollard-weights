#!/usr/bin/env python3
"""pollard-fit — build Pollard Weights: a memory-fit quantized model for YOUR machine.

Takes any GGUF and a RAM budget, computes a role- and depth-aware bit allocation
(attention and routers protected, expert FFNs carry the compression, hot layers
keep more bits when a routing profile is supplied), then drives llama.cpp's
`llama-quantize` per-tensor overrides to emit the build. The output is a normal
GGUF: it runs in stock llama.cpp, Ollama, LM Studio — anywhere.

Usage:
  pollard-fit --gguf model-f16.gguf --ram 16 --out model-pollard.gguf
  pollard-fit --gguf model-f16.gguf --ram 16 --profile routing_stats.json --imatrix imatrix.dat
  pollard-fit --gguf model.gguf --ram 16 --plan-only     # print the allocation, build nothing

Best results start from an f16/bf16 source GGUF (requantizing an already-4-bit
file compounds loss). A routing profile from experiments/e2 makes the depth
allocation measured instead of uniform.
"""
import argparse
import json
import shutil
import subprocess
import sys

from pollard_calc import (read_gguf_meta, gguf_to_config, analyse,
                          detect_available_ram_gb)

# quant types llama-quantize accepts for --tensor-type overrides, with effective
# bits/weight (format overhead included) used for budget math.
# base ggml types accepted by --tensor-type (mix presets like Q4_K_M are NOT
# valid there — they are whole-model presets only)
# Aggressive tier uses IQ lattice types, not Q_K: measured KL-divergence on
# granite experts, IQ2_S beat Q2_K by 23% mean / 27% median AND was smaller
# (experiments/e11 + KL runs). IQ types need an --imatrix to shine.
QTYPES = [("q8_0", 8.5), ("q6_K", 6.6), ("q5_K", 5.5), ("q4_K", 4.6),
          ("iq3_s", 3.4), ("iq2_s", 2.5), ("iq2_xxs", 2.1)]
BPW = dict(QTYPES)

# roles, most-protected first — simple name fragments in llama-quantize's own
# example style (attn_q=q8_0). Embeddings/output use their dedicated flags.
PROTECTED = [
    ("attention", "attn_"),
    ("norms", "_norm"),
    ("router", "ffn_gate_inp"),
    ("shared_expert", "_shexp"),
]
EXPERT_FRAGS = ["blk.{layer}.ffn_up_exps", "blk.{layer}.ffn_down_exps",
                "blk.{layer}.ffn_gate_exps"]
# patterns are REGEXES in llama-quantize: escape the dot or "ffn_up." swallows
# "ffn_up_exps" and promotes every expert tensor (cost us a 2.8GB "q3" build)
DENSE_FFN = [("dense_ffn_up", r"ffn_up\.weight"), ("dense_ffn_down", r"ffn_down\.weight"),
             ("dense_ffn_gate", r"ffn_gate\.weight")]


def plan_allocation(arch, ram_gb, profile, reserve_gb, cold_override=None):
    """Return (protected_type, per-layer expert types, projected GB)."""
    budget = ram_gb * 0.85 - reserve_gb
    if budget <= 0:
        sys.exit(f"ERROR: RAM budget {ram_gb}GB minus {reserve_gb}GB activation "
                 f"reserve leaves nothing for weights.")
    layers = arch["layers"]
    total = arch["total"]
    if arch["kind"] != "moe":
        # dense: single global type that fits
        for name, bpw in QTYPES:
            if total * bpw / 8 / 1e9 <= budget:
                return name, {}, total * bpw / 8 / 1e9
        sys.exit("ERROR: model does not fit this budget at any supported type. "
                 "pollard-calc will tell you the tier it needs.")

    expert_p = arch["expert_params"] * arch["n_experts"]     # per layer
    other = total - expert_p * layers                         # attn/embed/etc
    # layer heat: measured from profile if given, else uniform
    heat = [1.0] * layers
    if profile:
        freq = profile.get("reuse_by_layer",
                           profile.get("layer_activation", profile.get("layers")))
        if isinstance(freq, list) and len(freq) == layers:
            heat = [float(x) for x in freq]
    order = sorted(range(layers), key=lambda i: heat[i])      # coldest first
    if cold_override:
        rest = [i for i in order if i not in cold_override]
        order = cold_override + rest

    # protected tensors at Q6_K; experts start at Q4_K_M and step the coldest
    # layers down until the projection fits the budget.
    prot_name = "q6_K"
    expert_types = {i: "q4_K" for i in range(layers)}

    def projected():
        gb = other * BPW[prot_name] / 8 / 1e9
        for i in range(layers):
            gb += expert_p * BPW[expert_types[i]] / 8 / 1e9
        return gb

    ladder = ["q4_K", "iq3_s", "iq2_s", "iq2_xxs"]
    while projected() > budget:
        stepped = False
        for step_from, step_to in zip(ladder, ladder[1:]):
            for i in order:                                   # coldest first
                if expert_types[i] == step_from:
                    expert_types[i] = step_to
                    stepped = True
                    break
            if stepped:
                break
        if not stepped:
            if prot_name == "q6_K":                           # last resort
                prot_name = "q4_K"
                continue
            sys.exit(f"ERROR: cannot fit {total/1e9:.1f}B params into "
                     f"{budget:.1f}GB even at IQ2_XXS experts. pollard-calc will "
                     f"tell you the streaming tier instead.")
    return prot_name, expert_types, projected()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16 preferred)")
    ap.add_argument("--ram", required=True,
                    help="target machine RAM in GB, or 'auto' to measure what is "
                         "actually available right now")
    ap.add_argument("--out", help="output path (default: <src>-pollard.gguf)")
    ap.add_argument("--profile", help="routing profile json from experiments/e2 "
                                      "(per-layer activation heat)")
    ap.add_argument("--cold-layers", help="comma-separated layer indices to step down "
                                          "FIRST (e.g. from imatrix sensitivity ranking); "
                                          "overrides the profile ordering")
    ap.add_argument("--imatrix", help="importance matrix from llama-imatrix")
    ap.add_argument("--reserve", type=float, default=3.0,
                    help="GB reserved for activations/KV (default 3)")
    ap.add_argument("--llama-quantize", default="llama-quantize",
                    help="path to llama.cpp's llama-quantize binary")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the allocation and the command, build nothing")
    a = ap.parse_args()

    if str(a.ram).lower() == "auto":
        avail = detect_available_ram_gb()
        if avail is None:
            sys.exit("ERROR: could not measure available RAM — pass --ram <GB>.")
        a.ram = avail
        print(f"[--ram auto] measured available memory: {avail:.1f} GB")
    else:
        a.ram = float(a.ram)
    meta = read_gguf_meta(a.gguf)
    cfg = gguf_to_config(meta, a.gguf)
    arch = analyse(cfg)
    profile = json.load(open(a.profile)) if a.profile else None

    cold = [int(x) for x in a.cold_layers.split(",")] if a.cold_layers else None
    prot, expert_types, gb = plan_allocation(arch, a.ram, profile, a.reserve, cold)
    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + "-pollard.gguf"

    print(f"== pollard-fit :: {a.gguf}")
    print(f"machine budget      : {a.ram:.0f} GB RAM ({a.reserve:.0f} GB reserved) "
          f"-> {a.ram*0.85 - a.reserve:.1f} GB for weights")
    print(f"projected build     : {gb:.1f} GB  ({arch['kind']})")
    print(f"protected tensors   : {prot}  (embeddings, attention, norms, routers, shared experts)")
    if arch["kind"] == "moe":
        from collections import Counter
        c = Counter(expert_types.values())
        mix = ", ".join(f"{n} layers @ {t}" for t, n in sorted(c.items()))
        src = "measured routing profile" if profile else "uniform (no profile — pass one for measured allocation)"
        print(f"expert FFN mix      : {mix}")
        print(f"depth allocation    : {src}")

    # already-quantized source? requantizing needs the explicit flag (and a
    # warning — an f16/bf16 source always gives better quality).
    src_bpw = None
    if cfg.get("_gguf_file_bytes") and arch["total"]:
        src_bpw = cfg["_gguf_file_bytes"] * 8.0 / arch["total"]
    requant = src_bpw is not None and src_bpw < 10.0

    overrides = [f"{pat}={prot}" for _, pat in PROTECTED + DENSE_FFN]
    if arch["kind"] == "moe":
        for i, t in expert_types.items():
            overrides += [frag.format(layer=i) + f"={t}" for frag in EXPERT_FRAGS]
    tt_file = out + ".tensor-types.txt"

    cmd = [a.llama_quantize]
    if requant:
        cmd += ["--allow-requantize"]
    if a.imatrix:
        cmd += ["--imatrix", a.imatrix]
    cmd += ["--token-embedding-type", prot, "--output-tensor-type", prot,
            "--tensor-type-file", tt_file,
            a.gguf, out, "Q4_K_M"]   # base preset for anything unmatched

    if requant:
        print(f"NOTE: source is already quantized (~{src_bpw:.1f} bpw) — "
              f"requantizing with --allow-requantize. An f16/bf16 source "
              f"gives better quality.")
    print()
    if a.plan_only:
        print(f"plan only — {len(overrides)} tensor overrides; command that would run:")
        print("  " + " ".join(cmd))
        return
    if shutil.which(cmd[0]) is None:
        sys.exit(f"ERROR: '{cmd[0]}' not found — build llama.cpp (see install.sh) "
                 f"or pass --llama-quantize /path/to/llama-quantize")
    with open(tt_file, "w") as f:
        f.write("\n".join(overrides) + "\n")
    print("building…")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"\ndone: {out}")
    print("run it with stock llama.cpp / Ollama / LM Studio — it is a normal GGUF.")


if __name__ == "__main__":
    main()
