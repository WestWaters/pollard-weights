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
import heapq
import json
import math
import re
import subprocess
import sys

from pollard_calc import (read_gguf_meta, gguf_to_config, analyse,
                          detect_available_ram_gb, read_gguf_tensor_names,
                          find_llama_bin, imatrix_covered_tensors)

# quant types llama-quantize accepts for --tensor-type overrides, with effective
# bits/weight (format overhead included) used for budget math.
# base ggml types accepted by --tensor-type (mix presets like Q4_K_M are NOT
# valid there — they are whole-model presets only)
# Aggressive tier uses IQ lattice types, not Q_K: measured KL-divergence on
# granite experts, IQ2_S beat Q2_K by 23% mean / 27% median AND was smaller
# (experiments/e11 + KL runs). IQ types need an --imatrix to shine.
# IQ4 (iq4_xs) sits where q4_K used to: same ~4-bit size, LOWER KL per byte, so
# the bulk never wastes bits on a K-quant when an IQ type is strictly better.
# measured: uniform IQ4_XS (0.041 KL) beat a q4_K-bulk build (0.053) at equal size.
QTYPES = [("q8_0", 8.5), ("q6_K", 6.6), ("q5_K", 5.5), ("iq4_xs", 4.25),
          ("iq3_s", 3.4), ("iq2_s", 2.5), ("iq2_xxs", 2.1),
          ("iq1_m", 1.75), ("iq1_s", 1.56)]      # 1-bit floor (opt-in, --allow-1bit)
BPW = dict(QTYPES)
# the whole-model PRESET that carries "everything unmatched" — DERIVED from the
# chosen bulk type, never hardcoded. (--tensor-type wants base types; the
# positional base arg wants a preset name.) IQ presets need an --imatrix.
PRESET = {"q8_0": "Q8_0", "q6_K": "Q6_K", "q5_K": "Q5_K_M", "iq4_xs": "IQ4_XS",
          "iq3_s": "IQ3_S", "iq2_s": "IQ2_S", "iq2_xxs": "IQ2_XXS",
          "iq1_m": "IQ1_M", "iq1_s": "IQ1_S"}
LADDER = ["q6_K", "q5_K", "iq4_xs", "iq3_s", "iq2_s", "iq2_xxs"]  # high -> low
# 1-bit rungs are OFF by default (heavy quality loss); --allow-1bit extends the
# floor here for the giant-MoE case, where redundancy absorbs it (753B GLM at ~q1
# stays coherent — a community datapoint). Never used unless the budget forces it.
LADDER_1BIT = LADDER + ["iq1_m", "iq1_s"]
# NOISE[type] = KL cost of that type per unit importance = the KL of a UNIFORM
# build at that type (uniform KL = total_importance x noise). Measured on real
# models (Qwen2.5-1.5B here; ratios match granite in notes/e11-e12). These are
# what let the allocator SEE that crushing to iq2_xxs (~10x iq3_s) is catastrophic
# and avoid victim layers. Data, not logic — remeasure per family to refine.
# iq1_* are extrapolated from the curve's slope (no measured point yet); a
# pollard-sensitivity run measures them per model and overrides these.
NOISE = {"q8_0": 0.00048, "q6_K": 0.00479, "q5_K": 0.01033, "iq4_xs": 0.04090,
         "iq3_s": 0.12282, "iq2_s": 0.63394, "iq2_xxs": 1.22235,
         "iq1_m": 2.17, "iq1_s": 2.96}
# embeddings/output/norms kept HIGH by default — measured across sizes, keeping
# them here beat letting the allocator crush them (they carry the whole vocab).
# Stepped down only if the budget truly can't hold them. Data, not a magic rule.
EMB_FLOOR = "q6_K"

# llama-quantize REFUSES these types (and their whole-model presets) unless the
# imatrix covers the tensor — Frank's DeepSeek IQ2_S build bailed at tensor 3 on
# output_hc_fn. Base Q2_K and IQ3_S are exempt, which is the safe fallback.
IMATRIX_REQUIRED_PRESETS = {"IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M", "Q2_K_S"}
# with no imatrix, swap the imatrix-only IQ2 types for Q2_K (exempt, ~same bpw) so
# the build succeeds instead of crashing. iq4_xs / iq3_s are already exempt.
NOIMATRIX_TYPE_SUB = {"iq2_xxs": "q2_K", "iq2_xs": "q2_K", "iq2_s": "q2_K"}
NOIMATRIX_PRESET_SUB = {"IQ2_XXS": "Q2_K", "IQ2_XS": "Q2_K", "IQ2_S": "Q2_K"}

EXPERT_FRAGS = ["blk.{layer}.ffn_up_exps", "blk.{layer}.ffn_down_exps",
                "blk.{layer}.ffn_gate_exps"]
# patterns are REGEXES in llama-quantize: escape the dot or "ffn_up." swallows
# "ffn_up_exps" and promotes every expert tensor (cost us a 2.8GB "q3" build).
# per-LAYER dense FFN patterns, so each layer's bulk can take its own type
# (this is how per-layer sensitivity allocation reaches a dense model)
DENSE_FFN_FRAGS = [r"blk\.{layer}\.ffn_up\.weight", r"blk\.{layer}\.ffn_down\.weight",
                   r"blk\.{layer}\.ffn_gate\.weight"]


def _alloc_klaware(items, budget, noise=NOISE, ladder=LADDER):
    """items: [(params, importance)] — an FFN layer, an expert layer, or the
    attn/embed 'other' group. Assign each a ladder type to MINIMIZE total KL
    (~ sum importance_frac * noise[type]) subject to total size <= budget GB.
    `noise` is THIS model's measured KL-per-type curve (default only as fallback).
    Greedy on marginal KL-per-byte-saved (a heap of moves): start everything at
    the top of the ladder, then repeatedly take the single step that costs the
    LEAST KL per byte saved, until it fits. A catastrophic type carries a
    catastrophic marginal cost, so no victim layers; high-importance items stay
    high on their own (protection is emergent, not a rule). Returns
    ([type per item], gb) or None if it cannot fit even at the floor."""
    tot_imp = sum(imp for _, imp in items) or 1.0
    lvl = [0] * len(items)                              # LADDER index per item, 0 = top

    def size_gb():
        return sum(p * BPW[ladder[lvl[j]]] for j, (p, _) in enumerate(items)) / 8 / 1e9

    def marginal(j):
        p, imp = items[j]
        i = lvl[j]
        if i + 1 >= len(ladder):
            return None                                 # already at the floor
        a, b = ladder[i], ladder[i + 1]
        dbytes = p * (BPW[a] - BPW[b]) / 8
        dkl = (imp / tot_imp) * (noise[b] - noise[a])
        return ((dkl / dbytes) if dbytes > 0 else float("inf"), j)

    heap = [m for j in range(len(items)) if (m := marginal(j)) is not None]
    heapq.heapify(heap)
    while size_gb() > budget and heap:
        _, j = heapq.heappop(heap)                      # cheapest KL-per-byte step
        lvl[j] += 1
        nxt = marginal(j)
        if nxt is not None:
            heapq.heappush(heap, nxt)
    if size_gb() > budget:
        return None
    return [ladder[lvl[j]] for j in range(len(items))], size_gb()


def _fill_noise(measured):
    """Complete the noise curve for every LADDER type from THIS model's measured
    points, log-interpolating / extrapolating in bpw for any that failed to
    measure (a uniform build can hiccup) — keeps it model-specific. The baked
    default is used only if fewer than two points were measured."""
    pts = sorted((BPW[t], math.log(v)) for t, v in measured.items() if v)
    def est(t):
        if measured.get(t):
            return measured[t]
        if len(pts) < 2:
            return NOISE[t]
        x = BPW[t]
        if x <= pts[0][0]:
            (x0, y0), (x1, y1) = pts[0], pts[1]           # extrapolate low
        elif x >= pts[-1][0]:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]         # extrapolate high
        else:
            for i in range(len(pts) - 1):
                if pts[i][0] <= x <= pts[i + 1][0]:
                    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                    break
        return math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return {t: est(t) for t in LADDER}


def plan_allocation(arch, ram_gb, reserve_gb, sensitivity=None, allow_1bit=False):
    """Return (overrides, emb_type, projected_GB, base_preset, (summary, src)).
    KL-aware per-GROUP allocation for dense AND moe: every per-layer FFN/expert
    group AND every per-layer attention group is allocated separately, weighted
    by MEASURED sensitivity (a pollard-sensitivity profile) — else uniform. There
    is no imatrix-magnitude proxy: magnitude misranks (see e13), so without a
    measured profile we allocate uniformly rather than worse-than-uniform.
    Embeddings/output/norms are one 'other' group, kept high AND counted for size
    (so the projection matches the real build). `overrides` is [(regex, type)]."""
    budget = ram_gb * 0.85 - reserve_gb
    if budget <= 0:
        sys.exit(f"ERROR: RAM budget {ram_gb}GB minus {reserve_gb}GB activation "
                 f"reserve leaves nothing for weights.")
    layers = arch["layers"]
    total = arch["total"]
    bulk_pl = (arch["expert_params"] * arch["n_experts"] if arch["kind"] == "moe"
               else (arch.get("dense_ffn_params") or 0))
    attn_pl = arch.get("attn_params") or 0
    other = max(0, total - bulk_pl * layers - attn_pl * layers)   # emb + output + norms

    if bulk_pl <= 0:                                    # dims unknown -> safe uniform
        for name, bpw in QTYPES:
            if total * bpw / 8 / 1e9 <= budget:
                return [], name, total * bpw / 8 / 1e9, PRESET[name], (f"uniform {name}", "no dims")
        sys.exit("ERROR: does not fit at any supported type; see pollard-calc.")

    # per-group sensitivity: MEASURED profile, else UNIFORM. We deliberately do NOT
    # rank layers by imatrix MAGNITUDE — e13 proved magnitude != KL sensitivity (it
    # says "protect attention" when attention is only 0.48x as sensitive as FFN), so
    # a magnitude-ranked build can land WORSE than uniform. The imatrix still feeds
    # llama-quantize for IQ-type quality; it just never (mis)decides the allocation.
    if sensitivity:
        ffn_imp = {int(k): float(v) for k, v in sensitivity.get("ffn", {}).items()}
        attn_imp = {int(k): float(v) for k, v in sensitivity.get("attn", {}).items()}
        src = "measured KL sensitivity (pollard-sensitivity profile)"
    else:
        ffn_imp = {i: 1.0 for i in range(layers)}
        attn_imp = {i: 1.0 for i in range(layers)}
        src = ("uniform (no --sensitivity profile — run pollard-sensitivity for the "
               "per-layer win; any --imatrix is used for IQ quality only)")

    # noise curve: THIS model's measured KL-per-type if the profile carries it
    # (it shifts model to model), else the calibrated default. Nothing baked in.
    prof_noise = (sensitivity or {}).get("noise") or {}
    noise = _fill_noise(prof_noise) if prof_noise else dict(NOISE)
    if prof_noise:
        missing = [t for t in LADDER if not prof_noise.get(t)]
        src += " + measured noise curve" + (f" ({len(missing)} interpolated)" if missing else "")
    ladder = LADDER_1BIT if allow_1bit else LADDER      # opt-in 1-bit floor
    for t in ladder:                                    # every rung needs a noise value
        noise.setdefault(t, NOISE[t])

    # items: per-layer bulk + per-layer attention (embeddings handled separately)
    bulk_frag = EXPERT_FRAGS if arch["kind"] == "moe" else DENSE_FFN_FRAGS
    items, meta = [], []                                # meta: (kind, [patterns])
    for i in range(layers):
        items.append((bulk_pl, max(ffn_imp.get(i, 1.0), 1e-9)))
        meta.append(("bulk", [f.format(layer=i) for f in bulk_frag]))
        if attn_pl > 0:
            items.append((attn_pl, max(attn_imp.get(i, 1.0), 1e-9)))
            meta.append(("attn", [rf"blk\.{i}\.attn_.*"]))

    # Embeddings/output kept at EMB_FLOOR (measured: keeping them high wins; they
    # carry the whole vocab). Reserve their size, allocate FFN+attn over the rest;
    # step emb down a rung only if the budget genuinely can't hold it.
    emb_type = EMB_FLOOR
    while True:
        emb_gb = other * BPW[emb_type] / 8 / 1e9
        res = _alloc_klaware(items, budget - emb_gb, noise, ladder)
        if res is not None:
            break
        ei = LADDER.index(emb_type)
        if ei + 1 < len(LADDER):
            emb_type = LADDER[ei + 1]                    # embeddings never go 1-bit
        else:
            hint = ("" if allow_1bit else
                    " Or --allow-1bit to extend the floor to iq1 (heavy loss; for giant MoE).")
            sys.exit(f"ERROR: cannot fit {total/1e9:.1f}B into {budget:.1f}GB even at "
                     f"the floor. pollard-calc will tell you the streaming tier.{hint}")
    types, alloc_gb = res
    gb = alloc_gb + emb_gb

    from collections import Counter
    overrides, bulk_types = [], []
    for (kind, pats), t in zip(meta, types):
        overrides += [(p, t) for p in pats]
        if kind == "bulk":
            bulk_types.append(t)
    base_t = Counter(bulk_types).most_common(1)[0][0] if bulk_types else LADDER[0]
    c = Counter(bulk_types)
    summary = ", ".join(f"{n}L@{t}" for t, n in sorted(c.items(), key=lambda x: -BPW[x[0]]))
    return overrides, emb_type, gb, PRESET[base_t], (summary, src)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16 preferred)")
    ap.add_argument("--ram", required=True,
                    help="target machine RAM in GB, or 'auto' to measure what is "
                         "actually available right now")
    ap.add_argument("--out", help="output path (default: <src>-pollard.gguf)")
    ap.add_argument("--imatrix", help="importance matrix from llama-imatrix — required "
                                      "for IQ-type quality; does NOT decide the allocation")
    ap.add_argument("--sensitivity", help="measured sensitivity profile from "
                                          "pollard-sensitivity (the calibration that "
                                          "beats uniform quants — real KL cost per tensor)")
    ap.add_argument("--reserve", type=float, default=3.0,
                    help="GB reserved for activations/KV (default 3)")
    ap.add_argument("--llama-quantize", default="llama-quantize",
                    help="path to llama.cpp's llama-quantize binary")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the allocation and the command, build nothing")
    ap.add_argument("--allow-grow", action="store_true",
                    help="permit a build LARGER than an already-quantized source "
                         "(normally refused — requantizing up only loses)")
    ap.add_argument("--allow-1bit", action="store_true",
                    help="extend the floor to 1-bit (iq1_m/iq1_s) for models that won't "
                         "fit at iq2_xxs — heavy quality loss, but giant MoEs absorb it. "
                         "Only used where the budget forces it.")
    a = ap.parse_args()

    if str(a.ram).lower() == "auto":
        avail = detect_available_ram_gb()
        if avail is None:
            sys.exit("ERROR: could not measure available RAM — pass --ram <GB>.")
        a.ram = avail
        print(f"[--ram auto] measured available memory: {avail:.1f} GB")
    else:
        a.ram = float(a.ram)
    if a.allow_1bit and not a.imatrix:
        sys.exit("ERROR: --allow-1bit needs --imatrix — the 1-bit (iq1) types require a "
                 "calibration matrix to build at all (and substituting them to a non-imatrix "
                 "type would defeat the point by growing the file). Run llama-imatrix first.")
    meta = read_gguf_meta(a.gguf)
    cfg = gguf_to_config(meta, a.gguf)
    arch = analyse(cfg)
    sensitivity = json.load(open(a.sensitivity)) if a.sensitivity else None
    overrides, emb_type, gb, base_preset, (summary, src) = plan_allocation(
        arch, a.ram, a.reserve, sensitivity, a.allow_1bit)
    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + "-pollard.gguf"

    # ---- source facts + safety guards (hardened after Frank's DeepSeek/Qwen30B logs) ----
    src_bytes = cfg.get("_gguf_file_bytes") or 0
    src_gb = src_bytes / 1e9
    src_bpw = (src_bytes * 8.0 / arch["total"]) if (src_bytes and arch["total"]) else None
    requant = src_bpw is not None and src_bpw < 10.0     # source already quantized

    # GUARD 1 — refuse a build LARGER than an already-quantized source. Requantizing
    # UP only adds size and loses quality (Frank's Qwen30B: Q4_K_M 18.6 -> Q6_K 23.4 GB,
    # 23% slower — no Pollard content, just llama-quantize promoting every tensor).
    if requant and gb > src_gb * 1.02 and not a.allow_grow:
        sys.exit(
            f"ERROR: this build (~{gb:.1f} GB) would be LARGER than the source "
            f"(~{src_gb:.1f} GB, ~{src_bpw:.1f} bpw).\n"
            f"Requantizing an already-quantized file UP only adds size and loses "
            f"quality — there is no Pollard benefit.\n"
            f"Fix: start from an f16/bf16 source, or lower --ram. Override with "
            f"--allow-grow if you really mean it.")

    # GUARD 2 — no imatrix: swap the imatrix-only IQ2 types for Q2_K (exempt, ~same
    # bpw) so the build succeeds instead of crashing on uncovered tensors.
    subbed = False
    if not a.imatrix:
        subbed = base_preset in NOIMATRIX_PRESET_SUB or \
            any(t in NOIMATRIX_TYPE_SUB for _, t in overrides) or emb_type in NOIMATRIX_TYPE_SUB
        overrides = [(p, NOIMATRIX_TYPE_SUB.get(t, t)) for p, t in overrides]
        base_preset = NOIMATRIX_PRESET_SUB.get(base_preset, base_preset)
        emb_type = NOIMATRIX_TYPE_SUB.get(emb_type, emb_type)
        for a_t, b_t in NOIMATRIX_TYPE_SUB.items():      # keep the printed summary honest
            summary = summary.replace(a_t, b_t)

    # GUARD 3 — an imatrix-required type (IQ2/IQ1) on a tensor the imatrix DOESN'T
    # cover hard-crashes llama-quantize (DeepSeek's compressors; and MTP/`nextn`
    # layer tensors, which look like standard attention — `blk.64.attn_k.weight` —
    # but are never calibrated). Read the imatrix's REAL coverage and pin any tensor
    # that would take an imatrix-required type but isn't covered. Fall back to the
    # "matches no override" heuristic if the imatrix can't be parsed.
    base_pins = 0
    covered = imatrix_covered_tensors(a.imatrix) if a.imatrix else None
    _REQ = {"iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m", "q2_k_s"}
    handled = ("token_embd.weight", "output.weight")
    if covered is not None:
        ovc = [(re.compile(p), str(t).lower()) for p, t in overrides]
        for nm in read_gguf_tensor_names(a.gguf):
            if nm in handled or nm in covered:
                continue
            planned = base_preset.lower()
            for pat, ty in ovc:
                if pat.search(nm):
                    planned = ty
            if planned in _REQ:
                overrides.append((re.escape(nm), EMB_FLOOR))
                base_pins += 1
    elif base_preset in IMATRIX_REQUIRED_PRESETS:            # fallback: no imatrix parse
        for nm in read_gguf_tensor_names(a.gguf):
            if nm not in handled and not any(re.search(p, nm) for p, _ in overrides):
                overrides.append((re.escape(nm), EMB_FLOOR))
                base_pins += 1

    print(f"== pollard-fit :: {a.gguf}")
    print(f"machine budget      : {a.ram:.0f} GB RAM ({a.reserve:.0f} GB reserved) "
          f"-> {a.ram*0.85 - a.reserve:.1f} GB for weights")
    print(f"projected build     : {gb:.1f} GB  ({arch['kind']})")
    print(f"embeddings/output   : {emb_type}")
    label = "expert+attn mix" if arch["kind"] == "moe" else "FFN+attn mix   "
    print(f"{label}     : {summary}  (base {base_preset})")
    print(f"sensitivity source  : {src}")

    # WARN — no calibration signal at all means a UNIFORM build with no per-layer
    # benefit. Say it loudly; this is the difference between Pollard and llama-quantize.
    if not a.sensitivity:
        extra = " The imatrix here only sets IQ-type quality, not the allocation." if a.imatrix else ""
        print("WARNING: no --sensitivity profile — this is a UNIFORM allocation, no "
              f"per-layer benefit.{extra}\n         Run `pollard-sensitivity` to actually "
              "beat uniform quants (imatrix magnitude is NOT used — it misranks).")
    if subbed:
        print("NOTE: no --imatrix — imatrix-only IQ2 types swapped to Q2_K so the build "
              "won't crash. For the real win, add --imatrix + --sensitivity.")
    if base_pins:
        print(f"NOTE: pinned {base_pins} unmatched tensor(s) to {EMB_FLOOR} so the "
              f"aggressive base preset can't crash on imatrix-uncovered tensors.")
    if requant:
        print(f"NOTE: source is already quantized (~{src_bpw:.1f} bpw) — requantizing "
              f"with --allow-requantize. An f16/bf16 source gives better quality.")
    if arch.get("multimodal"):
        print(f"NOTE: {arch['multimodal']} model — this builds the TEXT model only. To KEEP "
              f"vision, download the mmproj (vision projector) GGUF and ship it alongside; "
              f"run with `llama-* --mmproj mmproj-….gguf`. Do NOT quantize the mmproj.")
    n_1bit = sum(1 for _, t in overrides if str(t).startswith("iq1")) \
        + (1 if base_preset in ("IQ1_M", "IQ1_S") else 0)
    if n_1bit:
        print(f"WARNING: {n_1bit} group(s) hit the 1-bit floor (iq1) — the budget forced "
              f"it. Expect real quality loss; sanity-check the output. Viable mainly on "
              f"giant MoE (redundancy absorbs it), rough on small/dense models.")

    tt_file = out + ".tensor-types.txt"
    ov_lines = [f"{pat}={t}" for pat, t in overrides]
    cmd = [a.llama_quantize]
    if requant:
        cmd += ["--allow-requantize"]
    if a.imatrix:
        cmd += ["--imatrix", a.imatrix]
    cmd += ["--token-embedding-type", emb_type, "--output-tensor-type", emb_type,
            "--tensor-type-file", tt_file,
            a.gguf, out, base_preset]   # base preset DERIVED from the plan
    print()
    if a.plan_only:
        print(f"plan only — {len(ov_lines)} tensor overrides; command that would run:")
        print("  " + " ".join(cmd))
        return
    resolved = find_llama_bin(cmd[0])
    if resolved is None:
        sys.exit(f"ERROR: {cmd[0]} not found. install.sh builds it into "
                 f"runtime/llama.cpp/build/bin — re-run install.sh, or pass "
                 f"--llama-quantize /path/to/llama-quantize.")
    cmd[0] = resolved
    with open(tt_file, "w") as f:
        f.write("\n".join(ov_lines) + "\n")
    print("building…")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"\ndone: {out}")
    print("run it with stock llama.cpp / Ollama / LM Studio — it is a normal GGUF.")


if __name__ == "__main__":
    main()
