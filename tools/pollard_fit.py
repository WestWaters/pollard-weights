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
import shutil
import struct
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
# IQ4 (iq4_xs) sits where q4_K used to: same ~4-bit size, LOWER KL per byte, so
# the bulk never wastes bits on a K-quant when an IQ type is strictly better.
# measured: uniform IQ4_XS (0.041 KL) beat a q4_K-bulk build (0.053) at equal size.
QTYPES = [("q8_0", 8.5), ("q6_K", 6.6), ("q5_K", 5.5), ("iq4_xs", 4.25),
          ("iq3_s", 3.4), ("iq2_s", 2.5), ("iq2_xxs", 2.1)]
BPW = dict(QTYPES)
# the whole-model PRESET that carries "everything unmatched" — DERIVED from the
# chosen bulk type, never hardcoded. (--tensor-type wants base types; the
# positional base arg wants a preset name.) IQ presets need an --imatrix.
PRESET = {"q8_0": "Q8_0", "q6_K": "Q6_K", "q5_K": "Q5_K_M", "iq4_xs": "IQ4_XS",
          "iq3_s": "IQ3_S", "iq2_s": "IQ2_S", "iq2_xxs": "IQ2_XXS"}
LADDER = ["q6_K", "q5_K", "iq4_xs", "iq3_s", "iq2_s", "iq2_xxs"]  # high -> low
# NOISE[type] = KL cost of that type per unit importance = the KL of a UNIFORM
# build at that type (uniform KL = total_importance x noise). Measured on real
# models (Qwen2.5-1.5B here; ratios match granite in notes/e11-e12). These are
# what let the allocator SEE that crushing to iq2_xxs (~10x iq3_s) is catastrophic
# and avoid victim layers. Data, not logic — remeasure per family to refine.
NOISE = {"q8_0": 0.00048, "q6_K": 0.00479, "q5_K": 0.01033, "iq4_xs": 0.04090,
         "iq3_s": 0.12282, "iq2_s": 0.63394, "iq2_xxs": 1.22235}
# embeddings/output/norms kept HIGH by default — measured across sizes, keeping
# them here beat letting the allocator crush them (they carry the whole vocab).
# Stepped down only if the budget truly can't hold them. Data, not a magic rule.
EMB_FLOOR = "q6_K"

EXPERT_FRAGS = ["blk.{layer}.ffn_up_exps", "blk.{layer}.ffn_down_exps",
                "blk.{layer}.ffn_gate_exps"]
# patterns are REGEXES in llama-quantize: escape the dot or "ffn_up." swallows
# "ffn_up_exps" and promotes every expert tensor (cost us a 2.8GB "q3" build).
# per-LAYER dense FFN patterns, so each layer's bulk can take its own type
# (this is how per-layer sensitivity allocation reaches a dense model)
DENSE_FFN_FRAGS = [r"blk\.{layer}\.ffn_up\.weight", r"blk\.{layer}\.ffn_down\.weight",
                   r"blk\.{layer}\.ffn_gate\.weight"]


def parse_imatrix_sensitivity(path, layers):
    """Per-layer sensitivity from a GGUF imatrix: aggregate the FFN importance
    (in_sum2 = sum of activation^2) per block. Returns {layer: score} or None.
    THIS is the measured signal that decides which layers keep bits — the thing
    Unsloth's 'dynamic' quants use, extracted from the imatrix we already build."""
    try:
        d = open(path, "rb").read()
    except Exception:
        return None
    if d[:4] != b"GGUF":
        return None                                    # old flat imatrix: no per-layer read
    off = 4
    struct.unpack_from("<I", d, off); off += 4         # version
    nt, = struct.unpack_from("<Q", d, off); off += 8
    nkv, = struct.unpack_from("<Q", d, off); off += 8
    _G = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def _rs():
        nonlocal off
        n, = struct.unpack_from("<Q", d, off); off += 8
        s = d[off:off + n].decode("utf-8", "replace"); off += n
        return s

    def _skip(t):
        nonlocal off
        if t == 8:
            n, = struct.unpack_from("<Q", d, off); off += 8 + n
        elif t == 9:
            et, = struct.unpack_from("<I", d, off); off += 4
            n, = struct.unpack_from("<Q", d, off); off += 8
            for _ in range(n):
                _skip(et)
        else:
            off += _G[t]

    try:
        for _ in range(nkv):
            _rs(); t, = struct.unpack_from("<I", d, off); off += 4; _skip(t)
        infos = []
        for _ in range(nt):
            name = _rs(); nd, = struct.unpack_from("<I", d, off); off += 4
            dims = struct.unpack_from(f"<{nd}Q", d, off); off += 8 * nd
            typ, = struct.unpack_from("<I", d, off); off += 4
            toff, = struct.unpack_from("<Q", d, off); off += 8
            infos.append((name, dims, typ, toff))
        base = (off + 31) // 32 * 32                    # data section is 32-aligned
        ffn, attn = {}, 0.0                              # per-layer FFN + total attention
        for name, dims, typ, toff in infos:
            if "in_sum2" not in name or typ != 0:       # importance sums, F32 only
                continue
            n = 1
            for x in dims:
                n *= x
            s = sum(struct.unpack_from(f"<{n}f", d, base + toff))
            m = re.search(r"blk\.(\d+)\.", name)
            if "ffn" in name and m:
                ffn[int(m.group(1))] = ffn.get(int(m.group(1)), 0.0) + s
            elif "attn" in name:
                attn += s
        if not ffn:
            return None
        return {"ffn": {i: ffn.get(i, 0.0) for i in range(layers)}, "other": attn}
    except Exception:
        return None


def _alloc_klaware(items, budget, noise=NOISE):
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
        return sum(p * BPW[LADDER[lvl[j]]] for j, (p, _) in enumerate(items)) / 8 / 1e9

    def marginal(j):
        p, imp = items[j]
        i = lvl[j]
        if i + 1 >= len(LADDER):
            return None                                 # already at the floor
        a, b = LADDER[i], LADDER[i + 1]
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
    return [LADDER[lvl[j]] for j in range(len(items))], size_gb()


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


def plan_allocation(arch, ram_gb, imatrix_path, reserve_gb, sensitivity=None):
    """Return (overrides, emb_type, projected_GB, base_preset, (summary, src)).
    KL-aware per-GROUP allocation for dense AND moe: every per-layer FFN/expert
    group AND every per-layer attention group is allocated separately, weighted
    by sensitivity — MEASURED from a pollard-sensitivity profile if given, else
    the imatrix magnitude (weaker), else uniform. Embeddings/output/norms are one
    'other' group, kept high AND counted for size (so the projection matches the
    real build). `overrides` is [(regex, type)] for --tensor-type-file."""
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

    # per-group sensitivity: MEASURED profile > imatrix magnitude > uniform
    if sensitivity:
        ffn_imp = {int(k): float(v) for k, v in sensitivity.get("ffn", {}).items()}
        attn_imp = {int(k): float(v) for k, v in sensitivity.get("attn", {}).items()}
        src = "measured KL sensitivity (pollard-sensitivity profile)"
    else:
        sens = parse_imatrix_sensitivity(imatrix_path, layers) if imatrix_path else None
        if sens:
            ffn_imp = sens["ffn"]
            attn_imp = {i: sens["other"] / max(1, layers) for i in range(layers)}
            src = "imatrix magnitude proxy (weaker — run pollard-sensitivity for the real signal)"
        else:
            ffn_imp = {i: 1.0 for i in range(layers)}
            attn_imp = {i: 1.0 for i in range(layers)}
            src = "uniform (no imatrix/profile)"

    # noise curve: THIS model's measured KL-per-type if the profile carries it
    # (it shifts model to model), else the calibrated default. Nothing baked in.
    prof_noise = (sensitivity or {}).get("noise") or {}
    noise = _fill_noise(prof_noise) if prof_noise else dict(NOISE)
    if prof_noise:
        missing = [t for t in LADDER if not prof_noise.get(t)]
        src += " + measured noise curve" + (f" ({len(missing)} interpolated)" if missing else "")

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
        res = _alloc_klaware(items, budget - emb_gb, noise)
        if res is not None:
            break
        ei = LADDER.index(emb_type)
        if ei + 1 < len(LADDER):
            emb_type = LADDER[ei + 1]
        else:
            sys.exit(f"ERROR: cannot fit {total/1e9:.1f}B into {budget:.1f}GB even at "
                     f"the floor. pollard-calc will tell you the streaming tier.")
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
    ap.add_argument("--profile", help="routing profile json from experiments/e2 "
                                      "(per-layer activation heat)")
    ap.add_argument("--cold-layers", help="comma-separated layer indices to step down "
                                          "FIRST (e.g. from imatrix sensitivity ranking); "
                                          "overrides the profile ordering")
    ap.add_argument("--imatrix", help="importance matrix from llama-imatrix")
    ap.add_argument("--sensitivity", help="measured sensitivity profile from "
                                          "pollard-sensitivity (the calibration that "
                                          "beats uniform quants — real KL cost per tensor)")
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
    sensitivity = json.load(open(a.sensitivity)) if a.sensitivity else None
    overrides, emb_type, gb, base_preset, (summary, src) = plan_allocation(
        arch, a.ram, a.imatrix, a.reserve, sensitivity)
    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + "-pollard.gguf"

    print(f"== pollard-fit :: {a.gguf}")
    print(f"machine budget      : {a.ram:.0f} GB RAM ({a.reserve:.0f} GB reserved) "
          f"-> {a.ram*0.85 - a.reserve:.1f} GB for weights")
    print(f"projected build     : {gb:.1f} GB  ({arch['kind']})")
    print(f"embeddings/output   : {emb_type}")
    label = "expert+attn mix" if arch["kind"] == "moe" else "FFN+attn mix   "
    print(f"{label}     : {summary}  (base {base_preset})")
    print(f"sensitivity source  : {src}")

    # IQ-lattice types (and measured sensitivity) need an imatrix — say so.
    if any("iq" in str(t) for t in [emb_type, base_preset.lower()] + [t for _, t in overrides]) \
            and not a.imatrix:
        print("WARNING: this build uses IQ-lattice types, which NEED an --imatrix to "
              "hold quality (llama-imatrix over a calibration corpus). Without one, "
              "expect degraded output — pass --imatrix.")

    # already-quantized source? requantizing needs the explicit flag.
    src_bpw = None
    if cfg.get("_gguf_file_bytes") and arch["total"]:
        src_bpw = cfg["_gguf_file_bytes"] * 8.0 / arch["total"]
    requant = src_bpw is not None and src_bpw < 10.0

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

    if requant:
        print(f"NOTE: source is already quantized (~{src_bpw:.1f} bpw) — "
              f"requantizing with --allow-requantize. An f16/bf16 source "
              f"gives better quality.")
    print()
    if a.plan_only:
        print(f"plan only — {len(ov_lines)} tensor overrides; command that would run:")
        print("  " + " ".join(cmd))
        return
    if shutil.which(cmd[0]) is None:
        sys.exit(f"ERROR: '{cmd[0]}' not found — build llama.cpp (see install.sh) "
                 f"or pass --llama-quantize /path/to/llama-quantize")
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
