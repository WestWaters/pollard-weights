#!/usr/bin/env python3
"""pollard-fit -- build Pollard Weights: a memory-fit quantized model for YOUR machine.

Takes any GGUF and a RAM budget, computes a role- and depth-aware bit allocation
(attention and routers protected, expert FFNs carry the compression, hot layers
keep more bits when a routing profile is supplied), then drives llama.cpp's
`llama-quantize` per-tensor overrides to emit the build. The output is a normal
GGUF: it runs in stock llama.cpp, Ollama, LM Studio -- anywhere.

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
import os
import re
import subprocess
import sys

from pollard_calc import (read_gguf_meta, gguf_to_config, analyse,
                          detect_available_ram_gb, read_gguf_tensor_names,
                          find_llama_bin, imatrix_covered_tensors, parse_mem_gb)

# quant types llama-quantize accepts for --tensor-type overrides, with effective
# bits/weight (format overhead included) used for budget math.
# base ggml types accepted by --tensor-type (mix presets like Q4_K_M are NOT
# valid there -- they are whole-model presets only)
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
# q2_K is deliberately NOT in QTYPES -- it is not a bulk-allocation candidate (iq2_s/iq2_xxs beat
# it per byte when an imatrix is present). It is here only so its SIZE is known, because it is the
# imatrix-free substitute NOIMATRIX_TYPE_SUB falls back to: embed/output are not covered by an
# imatrix, so their ladder descends through this instead of into iq2_xxs.
BPW["q2_K"] = 2.63
# the whole-model PRESET that carries "everything unmatched" -- DERIVED from the
# chosen bulk type, never hardcoded. (--tensor-type wants base types; the
# positional base arg wants a preset name.) IQ presets need an --imatrix.
PRESET = {"q8_0": "Q8_0", "q6_K": "Q6_K", "q5_K": "Q5_K_M", "iq4_xs": "IQ4_XS",
          "iq3_s": "IQ3_S", "iq2_s": "IQ2_S", "iq2_xxs": "IQ2_XXS",
          "iq1_m": "IQ1_M", "iq1_s": "IQ1_S"}
LADDER = ["q6_K", "q5_K", "iq4_xs", "iq3_s", "iq2_s", "iq2_xxs"]  # high -> low
# 1-bit rungs are OFF by default (heavy quality loss); --allow-1bit extends the
# floor here for the giant-MoE case, where redundancy absorbs it (753B GLM at ~q1
# stays coherent -- a community datapoint). Never used unless the budget forces it.
LADDER_1BIT = LADDER + ["iq1_m", "iq1_s"]
# NOISE[type] = KL cost of that type per unit importance = the KL of a UNIFORM
# build at that type (uniform KL = total_importance x noise). Measured on real
# models (Qwen2.5-1.5B here; ratios match granite in notes/e11-e12). These are
# what let the allocator SEE that crushing to iq2_xxs (~10x iq3_s) is catastrophic
# and avoid victim layers. Data, not logic -- remeasure per family to refine.
# iq1_* are extrapolated from the curve's slope (no measured point yet); a
# pollard-sensitivity run measures them per model and overrides these.
NOISE = {"q8_0": 0.00048, "q6_K": 0.00479, "q5_K": 0.01033, "iq4_xs": 0.04090,
         "iq3_s": 0.12282, "iq2_s": 0.63394, "iq2_xxs": 1.22235,
         "iq1_m": 2.17, "iq1_s": 2.96}
# embeddings/output/norms kept HIGH by default -- measured across sizes, keeping
# them here beat letting the allocator crush them (they carry the whole vocab).
# Stepped down only if the budget truly can't hold them. Data, not a magic rule.
EMB_FLOOR = "q6_K"

# llama-quantize REFUSES these types (and their whole-model presets) unless the
# imatrix covers the tensor -- Frank's DeepSeek IQ2_S build bailed at tensor 3 on
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
# finer-grained (alloc-v2): split the FFN into the crushable gate+up half and the
# protect-worthy down half, so a granular sensitivity profile can put them on DIFFERENT
# ladder rungs (measured down-protection, not a rule). SwiGLU => gate=up=down ~ 1/3 each.
EXPERT_FRAGS_GATEUP = ["blk.{layer}.ffn_up_exps", "blk.{layer}.ffn_gate_exps"]
EXPERT_FRAGS_DOWN = ["blk.{layer}.ffn_down_exps"]
DENSE_FFN_GATEUP = [r"blk\.{layer}\.ffn_up\.weight", r"blk\.{layer}\.ffn_gate\.weight"]
DENSE_FFN_DOWN = [r"blk\.{layer}\.ffn_down\.weight"]


#: the types a measured lookup table (Gemma 4's per-layer token embedding) walks, high -> low. It starts where the old
#: pin was (q8_0), so measuring it can only ever make it smaller; no imatrix-only type (llama-imatrix never sees a
#: get_rows table), q2_K standing in for the IQ2 rung. Matches pollard_probe.TABLE_LADDER_BITS.
TABLE_LADDER = ["q8_0", "q6_K", "q5_K", "iq4_xs", "iq3_s", "q2_K"]


def _alloc_klaware(items, budget, noise=NOISE, ladder=LADDER):
    """items: [(params, importance)] -- an FFN layer, an expert layer, or the
    attn/embed 'other' group. Assign each a ladder type to MINIMIZE total KL
    (~ sum importance_frac * noise[type]) subject to total size <= budget GB.
    `noise` is THIS model's measured KL-per-type curve (default only as fallback).
    Greedy on marginal KL-per-byte-saved (a heap of moves): start everything at
    the top of the ladder, then repeatedly take the single step that costs the
    LEAST KL per byte saved, until it fits. A catastrophic type carries a
    catastrophic marginal cost, so no victim layers; high-importance items stay
    high on their own (protection is emergent, not a rule). Returns
    ([type per item], gb) or None if it cannot fit even at the floor."""
    # An item is (params, importance), or (params, importance, costs, own_ladder) for one whose cost at each type was
    # MEASURED directly (a lookup table: pollard-probe's `tables`), in the same units as importance x noise.
    tot_imp = sum(it[1] for it in items if len(it) < 3) or 1.0
    lads = [it[3] if len(it) > 3 else ladder for it in items]
    lvl = [0] * len(items)                              # ladder index per item, 0 = top

    def size_gb():
        return sum(it[0] * BPW[lads[j][lvl[j]]] for j, it in enumerate(items)) / 8 / 1e9

    def marginal(j):
        p, imp = items[j][0], items[j][1]
        lad, i = lads[j], lvl[j]
        if i + 1 >= len(lad):
            return None                                 # already at the floor
        a, b = lad[i], lad[i + 1]
        dbytes = p * (BPW[a] - BPW[b]) / 8
        if len(items[j]) > 2:
            dkl = items[j][2][b] - items[j][2][a]
        else:
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
    return [lads[j][lvl[j]] for j in range(len(items))], size_gb()


def _fill_noise(measured):
    """Complete the noise curve for every LADDER type from THIS model's measured
    points, log-interpolating / extrapolating in bpw for any that failed to
    measure (a uniform build can hiccup) -- keeps it model-specific. The baked
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


# K-quants work in blocks of 256 (QK_K). A tensor whose row length is not a multiple of 256
# cannot be stored as one, and llama.cpp normally falls back for such tensors on its own -- but an
# EXPLICIT --token-embedding-type bypasses that fallback, and ggml then aborts mid-write:
#
#   ggml.c: GGML_ASSERT(start % type_traits[type].blck_size == 0) failed
#
# The process dies partway through and leaves a TRUNCATED .gguf behind, which is far worse than an
# error: a 0.5B model produced a 5.9 MB file that no reader will open, and the build reported
# nothing wrong. Qwen2.5-0.5B is exactly this case -- hidden 896, and 896 / 256 = 3.5.
QK_K = 256
_KQUANT_PREFIXES = ("q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "iq2_", "iq3_", "iq4_", "iq1_")


def _is_kquant(t: str) -> bool:
    return any(t.lower().startswith(p.lower()) for p in _KQUANT_PREFIXES)


def _env_threads():
    """POLLARD_THREADS, so a shared box can be configured once instead of per-command."""
    try:
        v = int(os.environ.get("POLLARD_THREADS", "") or 0)
        return v if v > 0 else None
    except ValueError:
        return None


def block_safe_type(t: str, row_len: int, fallback: str = "q8_0") -> str:
    """The requested type if this row length can hold it, else one that can.

    q8_0 has a block of 32, so it divides anything 32-aligned and costs a little size on a tensor
    that is usually a single embedding matrix. Silence here is the failure mode worth avoiding, so
    callers print when this substitutes.
    """
    if not row_len or not _is_kquant(t) or row_len % QK_K == 0:
        return t
    return fallback


def plan_allocation(arch, ram_gb, reserve_gb, sensitivity=None, allow_1bit=False, tiers=None,
                    emb_imatrix_ok=True, disk=None, want_pinned=False):
    """Return (overrides, emb_type, projected_GB, base_preset, (summary, src)), plus -- with want_pinned -- the
    patterns the caller pinned explicitly (--disk tables, --tier classes), which no later guard may override.
    KL-aware per-GROUP allocation for dense AND moe: every per-layer FFN/expert
    group AND every per-layer attention group is allocated separately, weighted
    by MEASURED sensitivity (a pollard-sensitivity profile) -- else uniform. There
    is no imatrix-magnitude proxy: magnitude misranks (see e13), so without a
    measured profile we allocate uniformly rather than worse-than-uniform.
    Embeddings/output/norms are one 'other' group, kept high AND counted for size
    (so the projection matches the real build). `overrides` is [(regex, type)].

    `tiers` PINS a whole component class to one type -- {"attn": "q8_0"} keeps every attention
    tensor at 8 bits regardless of what measurement says, reserves its bytes, and lets the rest of
    the model compete for what is left. That is the axis a per-tensor allocator cannot express, and
    it is how the vendor recipes are written (NVIDIA's Qwen3.8-27B-NVFP4: MLP at NVFP4, every
    attention path at FP8, vision untouched). Pinning is a CONSTRAINT, not a measurement -- it says
    "I want this guarantee, spend the remainder wisely" -- so the steering it frees is the point.

    `disk` (from pollard_ngram.disk_plan) names lookup tables that llama.cpp reads row by row from the file
    (--lazy-mode): an n-gram / per-layer embedding table. Their params leave the 'other' group, they are pinned
    to disk['type'], and their bytes are NOT charged to RAM -- so a 51B table no longer crushes the transformer.
    disk['type'] `auto` / `measured`: when the profile MEASURED the table (pollard-probe --from-imatrix --text, its
    `tables` entry), the table is allocated like every other tensor instead -- its own cost per type, its bytes in
    the budget (they are in the file), walking TABLE_LADDER down from q8_0 only where that is the cheapest KL per
    byte. It is still read lazily from the SSD at run time. `auto` without a measurement keeps the q8_0 pin."""
    # Conv/attention hybrids take one of two measured plans (LFM3-Audio / -Omni / -1.7B, 2026-10-29..31):
    #  3-bit rungs -- the exact plan: mixers counted at the base type, attention floored there, the rest re-allocated
    #                 (KLD 0.226 -> 0.073, speech back to f16 level);
    #  4-bit and up -- the bulk planned as before (mixers outside the knapsack, which leaves the FFNs their q5_K/q6_K)
    #                 and attention raised a step on top: the exact plan's leaner FFNs dropped Spanish sentences on 2 of
    #                 6 seeds; this one dropped none, matched the old rung in every language, KLD 0.034 -> 0.013, +2.7%.
    if arch.get("hybrid") and arch.get("attn_layers") and not arch.get("_hybrid_pass"):
        args = (ram_gb, reserve_gb, sensitivity, allow_1bit, tiers, emb_imatrix_ok, disk, want_pinned)
        exact = plan_allocation(dict(arch, _hybrid_pass="exact"), *args)
        bulk = [t for pat, t in exact[0] if ".ffn_" in pat]
        base = max(set(bulk), key=bulk.count) if bulk else None
        if base and BPW[base] >= BPW["iq4_xs"]:
            return plan_allocation(dict(arch, _hybrid_pass="bulk", mixers={}, attn_layers=[],
                                        _real_attn=list(arch["attn_layers"])), *args)
        return exact

    budget = ram_gb * 0.85 - reserve_gb
    if budget <= 0:
        sys.exit(f"ERROR: RAM budget {ram_gb}GB minus {reserve_gb}GB activation "
                 f"reserve leaves nothing for weights.")
    layers = arch["layers"]
    total = arch["total"]
    bulk_pl = (arch["expert_params"] * arch["n_experts"] if arch["kind"] == "moe"
               else (arch.get("dense_ffn_params") or 0))
    attn_pl = arch.get("attn_params") or 0
    # a conv/attention hybrid has attention on only some blocks: count (and allocate) those, not every block
    attn_layers = set(arch.get("attn_layers") or range(layers))
    # a hybrid's own token mixer (LFM shortconv in/out, an SSM's projections): a real share of the weights (12.6% of
    # LFM3) that llama-quantize writes at the rung's type -- left in `other` it was budgeted at the embedding type and
    # the rung came out 6% under its budget. Each block's mixer is its own item, weighted like that block's FFN.
    mixers = {int(i): v for i, v in (arch.get("mixers") or {}).items()}
    mixer_pl = {i: sum(p for _, p in v) for i, v in mixers.items()}
    other = max(0, total - bulk_pl * layers - attn_pl * len(attn_layers) - sum(mixer_pl.values()))
    disk_params = (disk or {}).get("params") or 0
    if disk_params:                                    # the table lives on the SSD, not in the RAM budget
        other = max(0, other - disk_params)
        total = total - disk_params

    if bulk_pl <= 0:                                    # dims unknown -> safe uniform
        dpins = [(pat, disk["type"]) for pat in disk["patterns"]] if disk_params else []
        for name, bpw in QTYPES:
            if total * bpw / 8 / 1e9 <= budget:
                res = (dpins, name, total * bpw / 8 / 1e9, PRESET[name], (f"uniform {name}", "no dims"))
                return res + ({p for p, _ in dpins},) if want_pinned else res
        sys.exit("ERROR: does not fit at any supported type; see pollard-calc.")

    # per-group sensitivity: MEASURED profile, else UNIFORM. We deliberately do NOT
    # rank layers by imatrix MAGNITUDE -- e13 proved magnitude != KL sensitivity (it
    # says "protect attention" when attention is only 0.48x as sensitive as FFN), so
    # a magnitude-ranked build can land WORSE than uniform. The imatrix still feeds
    # llama-quantize for IQ-type quality; it just never (mis)decides the allocation.
    # alloc-v2: a GRANULAR profile carries per-layer ffn_down (and optionally ffn_gateup)
    # separately, so the crushable gate+up and the protect-worthy down land on different rungs.
    granular = bool(sensitivity and sensitivity.get("ffn_down"))
    if sensitivity:
        ffn_imp = {int(k): float(v) for k, v in sensitivity.get("ffn", {}).items()}
        attn_imp = {int(k): float(v) for k, v in sensitivity.get("attn", {}).items()}
        down_imp = {int(k): float(v) for k, v in (sensitivity.get("ffn_down") or {}).items()}
        gu_imp = {int(k): float(v) for k, v in
                  (sensitivity.get("ffn_gateup") or sensitivity.get("ffn") or {}).items()}
        src = ("measured KL sensitivity (pollard-sensitivity profile"
               + (", granular gate+up / down)" if granular else ")"))
    else:
        ffn_imp = {i: 1.0 for i in range(layers)}
        attn_imp = {i: 1.0 for i in range(layers)}
        down_imp = gu_imp = {}
        src = ("uniform (no --sensitivity profile -- run pollard-sensitivity for the "
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

    # items: per-layer bulk + per-layer attention (embeddings handled separately). In granular
    # mode the FFN bulk splits into gate+up (~2/3 of params) and down (~1/3) as SEPARATE items.
    moe = arch["kind"] == "moe"
    bulk_frag = EXPERT_FRAGS if moe else DENSE_FFN_FRAGS
    gu_frag = EXPERT_FRAGS_GATEUP if moe else DENSE_FFN_GATEUP
    down_frag = EXPERT_FRAGS_DOWN if moe else DENSE_FFN_DOWN
    items, meta = [], []                                # meta: (kind, [patterns])
    for i in range(layers):
        if granular:
            gu_p = max(bulk_pl * 2 // 3, 1)
            items.append((gu_p, max(gu_imp.get(i, 1.0), 1e-9)))
            meta.append(("bulk", [f.format(layer=i) for f in gu_frag]))
            items.append((max(bulk_pl - gu_p, 1), max(down_imp.get(i, 1.0), 1e-9)))
            meta.append(("bulk", [f.format(layer=i) for f in down_frag]))
        else:
            items.append((bulk_pl, max(ffn_imp.get(i, 1.0), 1e-9)))
            meta.append(("bulk", [f.format(layer=i) for f in bulk_frag]))
        if attn_pl > 0 and i in attn_layers:
            items.append((attn_pl, max(attn_imp.get(i, 1.0), 1e-9)))
            meta.append(("attn", [rf"blk\.{i}\.attn_.*"]))
        if mixer_pl.get(i):
            items.append((mixer_pl[i], max((gu_imp if granular else ffn_imp).get(i, 1.0), 1e-9)))
            meta.append(("mixer", [re.escape(n) for n, _ in mixers[i]]))

    # Embeddings/output kept at EMB_FLOOR (measured: keeping them high wins; they
    # carry the whole vocab). Reserve their size, allocate FFN+attn over the rest;
    # step emb down a rung only if the budget genuinely can't hold it.
    # A pinned class is removed from the contest and its bytes reserved up front, so the remaining
    # budget is spent only where allocation still has a choice.
    tiers = tiers or {}
    pinned, pinned_gb = [], 0.0
    table_meta, table_pats = [], set()
    if disk_params:
        dtype = str(disk.get("type"))
        prof_tables = (sensitivity or {}).get("tables") or {}
        names = disk.get("names") or []
        measured = (dtype in ("auto", "measured") and bool(prof_noise) and names
                    and all(n in prof_tables and all(t in (prof_tables[n].get("cost") or {}) for t in TABLE_LADDER)
                            for n in names))
        if dtype == "measured" and not measured:
            sys.exit("ERROR: --disk-type measured, but the sensitivity profile has no measured cost for "
                     f"{', '.join(names) or 'the table'} -- run pollard-probe --from-imatrix with --text and "
                     "--tokenizer (or use --disk-type auto / q8_0).")
        if measured:
            for n, pat in zip(names, disk["patterns"]):
                tc = prof_tables[n]
                items.append((int(tc.get("params") or disk_params), 1.0, dict(tc["cost"]), TABLE_LADDER))
                table_meta.append((n, pat))
                table_pats.add(pat)
            src += " + measured lookup table" + ("s" if len(names) > 1 else "")
        else:
            ptype = "q8_0" if dtype in ("auto", "measured") else dtype
            pinned += [(pat, ptype) for pat in disk["patterns"]]       # pinned, but 0 GB of RAM
    tables_items = items[len(meta):]                    # measured lookup tables ride at the end, never tier-pinned
    items = items[:len(meta)]
    if tiers:
        keep_items, keep_meta = [], []
        for (kind, pats), (params, imp) in zip(meta, items):
            t = tiers.get(kind)
            if t:
                pinned += [(p, t) for p in pats]
                pinned_gb += params * BPW[t] / 8 / 1e9
            else:
                keep_items.append((params, imp))
                keep_meta.append((kind, pats))
        items, meta = keep_items, keep_meta
    n_meta = len(items)
    items = items + tables_items

    # The embed/output ladder must not descend into an imatrix-REQUIRED type when the imatrix
    # does not cover those tensors. llama-imatrix does not collect token_embd/output, so on a
    # tight budget this walked output.weight down to iq2_xxs and llama-quantize died on
    # GGML_ASSERT(imatrix != NULL) -- 866 tensors in, after the whole plan had printed. It bites
    # hardest exactly where it is least expected: a big vocab makes embed+output a large share of
    # the budget (Qwen3.8-27B: 248320 vocab = 2.54B params, ~2.1GB at q6_K of a 5.9GB budget), so
    # the descent is forced. Substitute the non-imatrix equivalent instead of walking into a crash.
    emb_ladder = LADDER if emb_imatrix_ok else [NOIMATRIX_TYPE_SUB.get(t, t) for t in LADDER]
    dedup = []
    for t in emb_ladder:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    emb_ladder = dedup
    emb_type = tiers.get("emb", emb_ladder[0])
    ei = emb_ladder.index(emb_type) if emb_type in emb_ladder else 0
    while True:
        emb_gb = other * BPW[emb_type] / 8 / 1e9
        res = _alloc_klaware(items, budget - emb_gb - pinned_gb, noise, ladder)
        if res is not None:
            break
        if tiers.get("emb"):
            sys.exit(f"ERROR: --tier emb={emb_type} does not fit in {budget:.1f}GB. Drop the tier "
                     f"or raise --ram.")
        if ei + 1 < len(emb_ladder):
            ei += 1
            emb_type = emb_ladder[ei]                    # embeddings never go 1-bit
        else:
            hint = ("" if allow_1bit else
                    " Or --allow-1bit to extend the floor to iq1 (heavy loss; for giant MoE).")
            sys.exit(f"ERROR: cannot fit {total/1e9:.1f}B into {budget:.1f}GB even at "
                     f"the floor. pollard-calc will tell you the streaming tier.{hint}")
    types, alloc_gb = res
    from collections import Counter
    # Hybrid mixers are COUNTED, not contested: bidding them against the FFNs (by the FFN's sensitivity) bought the
    # early shortconv projections q5_K out of the FFNs' bits and cost LFM3-Audio/-Omni their Japanese long-form speech
    # at IQ4_XS (2026-10-30). They are pinned at the base type the bulk settles on -- what llama-quantize would give
    # them anyway -- and everything else is allocated again into the budget that leaves.
    mix_j = [j for j, (kind, _) in enumerate(meta) if kind == "mixer"]
    if mix_j:
        tot = sum(it[1] for it in items if len(it) < 3) or 1.0
        free = [it if len(it) > 2 else (it[0], it[1], {t: it[1] / tot * noise[t] for t in ladder}, list(ladder))
                for it in items]
        tried = {}
        for _ in range(4):                    # the base can move once the mixers' bytes are settled: to a fixpoint
            bulk_now = [t for (kind, _), t in zip(meta, types[:len(meta)]) if kind == "bulk"]
            mix_t = Counter(bulk_now).most_common(1)[0][0] if bulk_now else types[mix_j[0]]
            if mix_t in tried:                # settled -- or flipping between two bases at a starved budget, where
                low_t = min(tried, key=lambda t: BPW[t])          # the cheaper pin is the one that stays in budget
                items, (types, alloc_gb) = tried[low_t]
                break
            pin = [(it[0], it[1], it[2], [mix_t]) if j in mix_j else it for j, it in enumerate(free)]
            res_m = _alloc_klaware(pin, budget - emb_gb - pinned_gb, noise, ladder)
            if res_m is None:
                break
            tried[mix_t] = (pin, res_m)
            types, alloc_gb = res_m
            items = pin
        if items is not free and any(kind == "mixer" for kind, _ in meta):
            src += f" + hybrid mixers pinned at {items[mix_j[0]][3][0]}"
    # Hybrid attention floor. On a conv/attention hybrid the few attention blocks carry every long-range
    # dependency, and a sensitivity read from text (the imatrix route above all) rates them cheap: LFM3-Audio-2B's
    # IQ3_S put all eleven at iq2_xxs and moved the bits into those blocks' FFNs. Measured 2026-10-29/30 on
    # LFM3-Audio / -Omni / -1.7B (pollard-audioeval, 18 languages, several seeds):
    #   3-bit rungs: attention at the rung's own base type, the rest re-allocated around it -- KLD 0.226 -> 0.073,
    #                speech back to f16 level (holding the extra FFN bits instead measured worse);
    #   4-bit and up: attention ONE step above the base, and the floor ADDS its bytes instead of taking them from
    #                the FFNs -- paying with the early FFNs made Omni cut sentences short; added, KLD 0.038 -> 0.016
    #                with no sentence dropped over six seeds (+2.7% size).
    real_attn = set(arch.get("_real_attn") or arch.get("attn_layers") or [])
    if arch.get("hybrid") and real_attn and len(real_attn) < layers:
        from collections import Counter
        attn_of = lambda pats: int(pats[0].split("\\.")[1]) if pats and pats[0].startswith("blk") else -1
        bulk_now = [t for (kind, _), t in zip(meta, types[:len(meta)]) if kind == "bulk"]
        base_now = Counter(bulk_now).most_common(1)[0][0] if bulk_now else None
        if base_now and BPW[base_now] >= BPW["iq4_xs"]:
            floor_t = ladder[max(0, ladder.index(base_now) - 1)]
            low = [j for j, ((kind, pats), t) in enumerate(zip(meta, types[:len(meta)]))
                   if kind == "attn" and attn_of(pats) in real_attn and BPW[t] < BPW[floor_t]]
            if low:
                types = list(types)
                for j in low:
                    if not arch.get("_size_probe"):   # the ladder sizes a rung by its plan, then the floor rides on top
                        alloc_gb += items[j][0] * (BPW[floor_t] - BPW[types[j]]) / 8 / 1e9
                    types[j] = floor_t
                src += (f" + hybrid attention floor ({len(low)} attention block(s) raised to {floor_t}, "
                        "bytes added, nothing else lowered)")
        elif base_now:
            floor_t = base_now
            low = [j for j, ((kind, _), t) in enumerate(zip(meta, types[:len(meta)]))
                   if kind == "attn" and BPW[t] < BPW[floor_t]]
            if low:
                floor_lad = [t for t in ladder if BPW[t] >= BPW[floor_t]]
                tot = sum(it[1] for it in items if len(it) < 3) or 1.0
                # every item as (params, importance, cost per type, own ladder): the same units _alloc_klaware
                # uses (importance share x noise), so only the floored items' ladders change, never anyone's weight
                fl_items = [(it[0], it[1], it[2], floor_lad) if len(it) > 2 and j in low else it if len(it) > 2 else
                            (it[0], it[1], {t: it[1] / tot * noise[t] for t in ladder},
                             floor_lad if j in low else list(ladder))
                            for j, it in enumerate(items)]
                res2 = _alloc_klaware(fl_items, budget - emb_gb - pinned_gb, noise, ladder)
                if res2 is not None:
                    types, alloc_gb = res2
                    src += f" + hybrid attention floor ({len(low)} attention block(s) held at >= {floor_t})"
    gb = alloc_gb + emb_gb + pinned_gb
    table_types = types[n_meta:]
    types = types[:n_meta]

    from collections import Counter
    overrides, bulk_types = list(pinned), []
    for (n, pat), t in zip(table_meta, table_types):
        overrides.append((pat, t))
    for (kind, pats), t in zip(meta, types):
        overrides += [(p, t) for p in pats]
        if kind == "bulk":
            bulk_types.append(t)
    base_t = Counter(bulk_types).most_common(1)[0][0] if bulk_types else LADDER[0]
    c = Counter(bulk_types)
    summary = ", ".join(f"{n}L@{t}" for t, n in sorted(c.items(), key=lambda x: -BPW[x[0]]))
    if tiers:
        summary += "  [pinned: " + ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())) + "]"
    if table_meta:
        summary += "  [measured table(s), read lazily from the SSD: " + ", ".join(
            f"{n.split('.')[0]} @ {t}" for (n, _), t in zip(table_meta, table_types)) + "]"
    elif disk_params:
        ptype = "q8_0" if str(disk["type"]) in ("auto", "measured") else disk["type"]
        summary += f"  [disk: {len(disk['patterns'])} table(s) @ {ptype}, {disk['gb']:.1f} GB on SSD]"
    if not bulk_types:
        base_t = tiers.get("bulk") or LADDER[0]
    res = (overrides, emb_type, gb, PRESET[base_t], (summary, src))
    return res + ({p for p, _ in pinned} | table_pats,) if want_pinned else res


def stale_moe_profile(sensitivity, arch):
    """Why an imatrix-derived profile must be re-made before it allocates this MoE, or None.

    `pollard-probe --from-imatrix` profiles made before 2026-10-22 summed every expert's CONDITIONAL cost (E[x^2]
    over only the tokens routed to it) as if every token reached every expert: n_expert / n_used too heavy against
    attention, which sees every token. On clef-omni (128 experts, top-8) the IQ4_XS rung put 23 layers of attention
    at iq2_xxs while the experts kept iq4_xs-q6_K. A current profile carries `"experts": "routing-frequency"`."""
    if not sensitivity or (arch or {}).get("kind") != "moe":
        return None
    if sensitivity.get("method") != "imatrix-hessian" or sensitivity.get("experts"):
        return None
    return ("this sensitivity profile is from `pollard-probe --from-imatrix` before MoE experts were weighted by "
            "routing frequency: it over-weights every expert by n_expert / n_used against attention, so attention "
            "lands far too low. Re-run `pollard-probe --from-imatrix <imatrix> --gguf <f16>` (minutes, no forward "
            "pass) and rebuild from the new profile.")


def pin_uncovered(overrides, names, covered, base_preset, explicit=()):
    """(overrides, n_pinned): every tensor the imatrix does not cover, held at EMB_FLOOR wherever the plan
    puts it below that, the pins placed FIRST.

    llama-quantize takes the first --tensor-type pattern that matches (ik_llama's custom-q too), and the
    patterns are unanchored searches: `blk\\.0\\.ffn_down\\.weight` also matches a decision head's
    `dec.blk.0.ffn_down.weight`. Appended, a pin lost to that pattern and the build crashed anyway
    (Cloudflare/clef-flash, 2026-10-09: "Missing importance matrix for tensor dec.blk.0.ffn_down.weight").
    The planned type is read the same first-match way.

    Held whenever planned BELOW the floor, not only when the type would crash: a low-bit type with no
    imatrix behind it builds silently and badly. That was the clef IQ4_XS rung -- the decision head, the
    part whose output IS the product, at iq4 uncalibrated. MTP / nextn layers and DeepSeek's compressors
    are the other uncovered tensors this has always been for."""
    req = {"iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m", "q2_k_s"}
    bpw = {"q4_k": 4.5, "q3_k": 3.4375, "q2_k": 2.625, "q4_0": 4.5, "q5_0": 5.5}   # K/legacy types not in BPW
    bpw.update({k.lower(): v for k, v in BPW.items()})
    floor = bpw.get(EMB_FLOOR.lower(), 6.56)
    # A preset NAME is not a tensor type: "Q5_K_M" / "Q4_K_M" missed the bpw table, read as the floor, and pinned
    # nothing -- uncovered tensors then built at 4.5-5.5 bpw uncalibrated (found in review by Joey, 2026-10-10).
    preset_base = {v.lower(): k.lower() for k, v in PRESET.items()}
    preset_base.update({"q4_k_m": "q4_k", "q4_k_s": "q4_k", "q5_k_s": "q5_k", "q3_k_m": "q3_k",
                        "q3_k_s": "q3_k", "q3_k_l": "q3_k", "q2_k": "q2_k"})
    explicit = set(explicit or ())
    ovc = [(re.compile(p), str(t).lower(), p in explicit) for p, t in overrides]
    pins = []
    for nm in names:
        if nm in ("token_embd.weight", "output.weight") or nm in covered:
            continue
        planned, chosen = preset_base.get(base_preset.lower(), base_preset.lower()), False
        for pat, ty, exp in ovc:
            if pat.search(nm):
                planned, chosen = ty, exp
                break
        # An explicit --disk / --tier choice is the user's, not the allocator's: a q5_K n-gram table stays q5_K
        # (pins-first had silently promoted it to q6_K -- Joey's F1).
        if chosen:
            continue
        if planned in req or bpw.get(planned, floor) < floor:
            pins.append((re.escape(nm), EMB_FLOOR))
    return pins + list(overrides), len(pins)


def _resolve_quantize(a, cfg):
    arch_name = cfg.get("_gguf_arch")
    resolved = find_llama_bin(a.llama_quantize, arch=arch_name)
    if resolved is None:
        sys.exit(f"ERROR: no llama-quantize that knows the '{arch_name}' architecture was found -- pass "
                 f"--llama-quantize /path/to/llama-quantize (a build new enough for this model).")
    return resolved


def _fit_passthrough(a, pt, src_tensors, arch, cfg, sensitivity, covered, tiers, out):
    """pollard-fit on an already-quantized source: the passthrough policy (see pollard_passthrough)."""
    classes = tuple(c.strip() for c in a.passthrough_classes.split(",") if c.strip())
    bad = [c for c in classes if c not in pt.CLASSES]
    if bad:
        sys.exit(f"--passthrough-classes: unknown class {bad} (choose from {', '.join(pt.CLASSES)})")
    budget = (a.ram * 0.85 - a.reserve) * 1e9
    if budget <= 0:
        sys.exit(f"ERROR: RAM budget {a.ram}GB minus {a.reserve}GB activation reserve leaves nothing for weights.")
    try:
        plan = pt.plan_passthrough(src_tensors, budget, sensitivity, classes, imatrix_ok=bool(a.imatrix),
                                   covered=covered, pins=tiers)
    except pt.PassthroughError as e:
        sys.exit(f"ERROR: {e}")
    print(f"== pollard-fit :: {a.gguf}  [PASSTHROUGH -- source is already quantized]")
    from collections import Counter
    src_types = Counter(t["type"] for t in src_tensors if t["ndim"] >= 2)
    print(f"source              : {plan['source_bytes']/1e9:.2f} GB  ("
          + ", ".join(f"{n}x {t}" for t, n in src_types.most_common(6)) + ")")
    print(f"machine budget      : {a.ram:.0f} GB RAM ({a.reserve:.0f} GB reserved) -> {budget/1e9:.1f} GB for weights")
    print(f"passthrough build   : {pt.describe(plan)}")
    print(f"changeable classes  : {', '.join(classes)}  (everything else is copied byte-for-byte)")
    print("sensitivity source  : " + ("measured profile" + (" (passthrough, vs this file)" if (sensitivity or {}).get("passthrough") else
                                       " (f16-based -- run pollard-sensitivity --passthrough for one measured on this file)")
                                       if sensitivity else "uniform (no --sensitivity profile)"))
    if not plan["fits"]:
        floor = pt.floor_bytes(src_tensors, classes, bool(a.imatrix), covered)
        sys.exit(f"ERROR: cannot reach {budget/1e9:.2f} GB by downgrading {', '.join(classes)} alone (floor "
                 f"{floor/1e9:.2f} GB{'' if a.imatrix else ' without an imatrix -- the 1-bit and IQ2 rungs need one'}). "
                 f"Raise --ram or widen --passthrough-classes.")
    if plan["changed"] and not a.imatrix:
        print("NOTE: no --imatrix -- the IQ2 / 1-bit rungs are off; downgrades use q2_K and above only.")
    if plan["changed"]:
        print("NOTE: downgraded tensors are requantized from the SOURCE's dequantized values, not the original "
              "bf16 (two roundings).\n      For a QAT source that is the right input; for a plain PTQ file a "
              "rung from bf16 usually wins.")
    lines, flags, stats = pt.quantize_plan(src_tensors, plan["assign"])
    tt_file = out + ".tensor-types.txt"
    cmd = pt.build_command(a.llama_quantize, a.gguf, out, tt_file, flags,
                           a.imatrix if plan["changed"] else None, a.threads)
    print(f"tensor map          : {stats['copy']} copied verbatim ({stats['copy_bytes']/1e9:.2f} GB), "
          f"{stats['requant']} requantized")
    print()
    if a.plan_only:
        print(f"plan only -- {len(lines)} exact-name tensor overrides; command that would run:")
        print("  " + " ".join(cmd))
        return
    cmd[0] = _resolve_quantize(a, cfg)
    with open(tt_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("building...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    if a.verify:
        rep = pt.verify_build(src_tensors, out, plan["assign"])
        if not rep["ok"]:
            sys.exit("ERROR: passthrough verification FAILED -- " + "; ".join((rep["mismatch"] + rep["missing"])[:6]))
        print(f"verified: {rep['identical']} tensor(s) byte-identical to the source, {rep['requantized']} requantized as planned")
    try:
        import pollard_workspace as ws
        bpw = round(os.path.getsize(out) * 8.0 / arch["total"], 2) if arch.get("total") else None
        m = re.search(r"-(PT\d+)\.gguf$", os.path.basename(out))      # a passthrough-ladder rung: its own tag
        ws.record_build(a.gguf, "gguf", out, tag=m.group(1) if m else "passthrough", bpw=bpw)
    except Exception as e:
        print(f"   (warning: could not record build to the workspace manifest: {e})")
    print(f"\ndone: {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16 preferred)")
    ap.add_argument("--ram", required=True,
                    help="target machine RAM in GB, 'auto' to measure what is actually available "
                         "right now, or a GPU/rack preset from pollard-calc's card table "
                         "(e.g. 'rubin', 'b300x8', 'nvl72')")
    ap.add_argument("--out", help="output path (default: <src>-pollard.gguf)")
    ap.add_argument("--tag", help="the build's tag in the workspace MANIFEST (default: its base preset) -- a "
                                  "quality-targeted rung is tagged by what it is, e.g. Q90-4.6GB")
    ap.add_argument("--no-record", dest="record", action="store_false",
                    help="do not record this build in the MANIFEST (a candidate a search may discard)")
    ap.add_argument("--imatrix", help="importance matrix from llama-imatrix -- required "
                                      "for IQ-type quality; does NOT decide the allocation")
    ap.add_argument("--sensitivity", help="measured sensitivity profile from "
                                          "pollard-sensitivity (the calibration that "
                                          "beats uniform quants -- real KL cost per tensor)")
    ap.add_argument("--reserve", type=float, default=3.0,
                    help="GB reserved for activations/KV (default 3)")
    ap.add_argument("--llama-quantize", default="llama-quantize",
                    help="path to llama.cpp's llama-quantize binary")
    ap.add_argument("--threads", type=int, default=_env_threads(),
                    help="number of threads for the heavy step. Default: the tool's own choice, which is usually every core. Set it lower to leave the machine usable -- a quantize that takes the whole box is a quantize you cannot run while anything else matters. POLLARD_THREADS sets it for every tool.")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the allocation and the command, build nothing")
    ap.add_argument("--allow-grow", action="store_true",
                    help="permit a build LARGER than an already-quantized source "
                         "(normally refused -- requantizing up only loses)")
    ap.add_argument("--tier", action="append", default=[], metavar="CLASS=TYPE",
                    help="pin a whole component class to one type instead of letting measurement "
                         "decide it: attn / bulk / emb. e.g. --tier attn=q8_0 keeps every attention "
                         "tensor at 8 bits, reserves its bytes, and spends the rest of the budget "
                         "where allocation still has a choice. This is how the vendor recipes are "
                         "written (NVIDIA NVFP4: MLP 4-bit, all attention FP8, vision untouched); "
                         "it is a guarantee you are asking for, not a measurement. Repeatable.")
    ap.add_argument("--disk", default="auto", metavar="auto|off|REGEX[,REGEX]",
                    help="lookup tables (n-gram / per-layer embeddings) that llama.cpp reads from the SSD on demand: "
                         "pinned to --disk-type and NOT charged to --ram (default auto: detect, if this runtime "
                         "lazy-reads the architecture). See pollard-ngram.")
    ap.add_argument("--disk-type", default="auto", choices=["auto", "measured", "q8_0", "q6_K", "q5_K", "f16", "bf16"],
                    help="precision for --disk tables. auto (default): MEASURED like every other tensor when the "
                         "--sensitivity profile scored the table (pollard-probe --from-imatrix --text), its bytes "
                         "counted in the budget and its type the allocator's call from q8_0 down; q8_0 otherwise. "
                         "measured: require the measurement. A type pins it there, outside the RAM budget")
    ap.add_argument("--allow-1bit", action="store_true",
                    help="extend the floor to 1-bit (iq1_m/iq1_s) for models that won't "
                         "fit at iq2_xxs -- heavy quality loss, but giant MoEs absorb it. "
                         "Only used where the budget forces it.")
    ap.add_argument("--passthrough", default="auto", choices=["auto", "on", "off"],
                    help="ALREADY-QUANTIZED source (e.g. a vendor QAT GGUF): copy every tensor whose type does not "
                         "change byte-for-byte, requantize only the tensors the measured allocation moves DOWN, and "
                         "never upgrade one. 'auto' (default) turns it on when most of the source's weight bytes are "
                         "already block-quantized. See tools/pollard_passthrough.py for the requant-source caveat.")
    ap.add_argument("--passthrough-classes", default="bulk", metavar="CLASS[,CLASS]",
                    help="which tensor classes a passthrough build may downgrade: bulk (routed experts + dense FFN, "
                         "default), attn, shexp (shared experts), emb, other. Everything else is copied as-is.")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="passthrough: skip the post-build byte check that copied tensors are identical to the source")
    a = ap.parse_args()

    tiers = {}
    for spec in a.tier:
        if "=" not in spec:
            sys.exit(f"--tier wants CLASS=TYPE, got '{spec}' (classes: attn, bulk, emb)")
        k, v = spec.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k not in ("attn", "bulk", "emb"):
            sys.exit(f"--tier class must be attn, bulk or emb; got '{k}'")
        if v not in BPW and not (a.passthrough != "off" and v.lower() in ("q8_0", "q4_k", "q2_k", "nvfp4", "mxfp4")):
            sys.exit(f"--tier type '{v}' is not a known quant type (see the ladder)")
        tiers[k] = v

    if str(a.ram).lower() == "auto":
        avail = detect_available_ram_gb()
        if avail is None:
            sys.exit("ERROR: could not measure available RAM -- pass --ram <GB>.")
        a.ram = avail
        print(f"[--ram auto] measured available memory: {avail:.1f} GB")
    else:
        # A preset is the same table pollard-calc --gpu reads, so `--ram rubin` budgets the 288 GB a
        # Rubin GPU has instead of dying on float('rubin').
        ram = parse_mem_gb(a.ram)
        if ram is None:
            ap.error(f"--ram '{a.ram}': not a number of GB nor a known GPU/rack preset "
                     "(e.g. 16, auto, rubin, b300x8, nvl72)")
        a.ram = ram
    if a.allow_1bit and not a.imatrix:
        sys.exit("ERROR: --allow-1bit needs --imatrix -- the 1-bit (iq1) types require a "
                 "calibration matrix to build at all (and substituting them to a non-imatrix "
                 "type would defeat the point by growing the file). Run llama-imatrix first.")
    meta = read_gguf_meta(a.gguf)
    cfg = gguf_to_config(meta, a.gguf)
    arch = analyse(cfg)
    sensitivity = json.load(open(a.sensitivity)) if a.sensitivity else None
    # Read imatrix coverage BEFORE allocating: whether the imatrix covers token_embd/output
    # decides which types the embed/output ladder is allowed to descend through. Unknown
    # coverage (unparseable imatrix) counts as NOT covered -- the safe direction, since the
    # cost is a slightly larger embedding and the alternative is a hard crash mid-build.
    covered = imatrix_covered_tensors(a.imatrix) if a.imatrix else None
    emb_imatrix_ok = bool(covered) and {"token_embd.weight", "output.weight"} <= covered
    # PASSTHROUGH -- an already-quantized source keeps what it has: copy unchanged tensors verbatim,
    # requantize only measured downgrades, refuse upgrades (pollard_passthrough). Its own planner works
    # from the real per-tensor types, so the f16-oriented planner and guards below never run for it.
    if a.passthrough != "off":
        import pollard_passthrough as pt
        try:
            src_tensors = pt.read_tensors(a.gguf)
        except Exception as e:                          # unreadable by gguf-py: only fatal when asked for
            if a.passthrough == "on":
                sys.exit(f"ERROR: --passthrough on, but the source's tensor table could not be read: {e}")
            src_tensors = []
        if src_tensors and (a.passthrough == "on" or pt.is_quantized_source(src_tensors)):
            out = a.out
            if not out:
                try:
                    import pollard_workspace as ws
                    out = ws.resolve_out(a.gguf, "gguf", tag="passthrough", ext=".gguf")
                    print(f"   (no --out) -> workspace: {out}")
                except Exception:
                    out = a.gguf.rsplit(".gguf", 1)[0] + "-pollard-pt.gguf"
            return _fit_passthrough(a, pt, src_tensors, arch, cfg, sensitivity, covered, tiers, out)
    for k, v in tiers.items():
        if v not in BPW:
            sys.exit(f"--tier type '{v}' is not a known quant type (see the ladder)")
    from pollard_ngram import disk_plan
    disk = disk_plan(meta, a.disk, a.disk_type)
    overrides, emb_type, gb, base_preset, (summary, src), explicit = plan_allocation(
        arch, a.ram, a.reserve, sensitivity, a.allow_1bit, tiers, emb_imatrix_ok,
        disk if disk and disk["params"] else None, want_pinned=True)
    if a.out:
        out = a.out
    else:
        try:                                                 # no --out -> organized workspace path
            import pollard_workspace as ws
            out = ws.resolve_out(a.gguf, "gguf", tag=base_preset, ext=".gguf")
            print(f"   (no --out) -> workspace: {out}")
        except Exception:
            out = a.gguf.rsplit(".gguf", 1)[0] + "-pollard.gguf"

    # ---- source facts + safety guards (hardened after Frank's DeepSeek/Qwen30B logs) ----
    src_bytes = cfg.get("_gguf_file_bytes") or 0
    src_gb = src_bytes / 1e9
    src_bpw = (src_bytes * 8.0 / arch["total"]) if (src_bytes and arch["total"]) else None
    requant = src_bpw is not None and src_bpw < 10.0     # source already quantized

    # GUARD 1 -- refuse a build LARGER than an already-quantized source. Requantizing
    # UP only adds size and loses quality (Frank's Qwen30B: Q4_K_M 18.6 -> Q6_K 23.4 GB,
    # 23% slower -- no Pollard content, just llama-quantize promoting every tensor).
    disk_gb = disk["gb"] if disk and disk["params"] else 0.0
    if requant and gb + disk_gb > src_gb * 1.02 and not a.allow_grow:
        sys.exit(
            f"ERROR: this build (~{gb:.1f} GB) would be LARGER than the source "
            f"(~{src_gb:.1f} GB, ~{src_bpw:.1f} bpw).\n"
            f"Requantizing an already-quantized file UP only adds size and loses "
            f"quality -- there is no Pollard benefit.\n"
            f"Fix: start from an f16/bf16 source, or lower --ram. Override with "
            f"--allow-grow if you really mean it.")

    # GUARD 2 -- no imatrix: swap the imatrix-only IQ2 types for Q2_K (exempt, ~same
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

    # GUARD 3 -- an imatrix-required type (IQ2/IQ1) on a tensor the imatrix DOESN'T
    # cover hard-crashes llama-quantize (DeepSeek's compressors; and MTP/`nextn`
    # layer tensors, which look like standard attention -- `blk.64.attn_k.weight` --
    # but are never calibrated). Read the imatrix's REAL coverage and pin any tensor
    # that would take an imatrix-required type but isn't covered. Fall back to the
    # "matches no override" heuristic if the imatrix can't be parsed.
    # Pins go FIRST and cover low-bit, not just crashing, types -- see pin_uncovered.
    base_pins = 0
    handled = ("token_embd.weight", "output.weight")
    if covered is not None:
        overrides, base_pins = pin_uncovered(overrides, read_gguf_tensor_names(a.gguf), covered, base_preset,
                                             explicit)
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
    if disk and disk["params"] and "measured table" in summary:
        print(f"on the SSD (lazy)   : {disk['params']/1e9:.2f}B params, MEASURED -- type and bytes in the plan above "
              f"(counted in the budget): {', '.join(disk['names'][:3])}{' ...' if len(disk['names']) > 3 else ''}")
        print(f"   run it with:  pollard-ngram run <build>.gguf --ram {a.ram:.0f}   (mmap + --lazy-mode on; never --no-mmap)")
    elif disk and disk["params"]:
        dt = "q8_0" if str(disk["type"]) in ("auto", "measured") else disk["type"]
        print(f"on the SSD (lazy)   : {disk['params']/1e9:.2f}B params @ {dt} = {disk['gb']:.1f} GB, not in the RAM "
              f"budget: {', '.join(disk['names'][:3])}{' ...' if len(disk['names']) > 3 else ''}")
        print(f"   run it with:  pollard-ngram run <build>.gguf --ram {a.ram:.0f}   (mmap + --lazy-mode on; never --no-mmap)")
    elif disk and disk.get("note"):
        print(f"NOTE: {disk['note']}")

    stale = stale_moe_profile(sensitivity, arch)
    if stale:
        print(f"WARNING: {stale}")

    # NOTE -- scope the evidence honestly. The reallocation can cost more than it buys, and on
    # Qwen2.5-0.5B at matched size it did (imatrix-only IQ3_S +7.48% over fp16 vs imatrix +
    # measured allocation +14.17%). That is one 0.5B; it is not a verdict on every dense model,
    # so this informs rather than blocks.
    if a.sensitivity and arch["kind"] != "moe":
        print("NOTE: measured profile on a DENSE model. Reallocation can cost more than it buys -- "
              "on\n      Qwen2.5-0.5B at matched size imatrix-only was +7.48% over fp16 and "
              "imatrix+profile\n      +14.17%. That is a single small model; larger dense models "
              "are untested. Bench this\n      against a plain imatrix build at the same budget "
              "before shipping.")

    # WARN -- no calibration signal at all means a UNIFORM build with no per-layer
    # benefit. Say it loudly; this is the difference between Pollard and llama-quantize.
    if not a.sensitivity:
        extra = " The imatrix here only sets IQ-type quality, not the allocation." if a.imatrix else ""
        print("WARNING: no --sensitivity profile -- this is a UNIFORM allocation, no "
              f"per-layer benefit.{extra}\n         Run `pollard-sensitivity` to actually "
              "beat uniform quants (imatrix magnitude is NOT used -- it misranks).")
    if subbed:
        print("NOTE: no --imatrix -- imatrix-only IQ2 types swapped to Q2_K so the build "
              "won't crash. For the real win, add --imatrix + --sensitivity.")
    if base_pins:
        print(f"NOTE: pinned {base_pins} unmatched tensor(s) to {EMB_FLOOR} so the "
              f"aggressive base preset can't crash on imatrix-uncovered tensors.")
    if requant:
        print(f"NOTE: source is already quantized (~{src_bpw:.1f} bpw) -- requantizing "
              f"with --allow-requantize. An f16/bf16 source gives better quality.")
    if arch.get("multimodal"):
        print(f"NOTE: {arch['multimodal']} model -- this builds the TEXT model only. To KEEP "
              f"vision, download the mmproj (vision projector) GGUF and ship it alongside; "
              f"run with `llama-* --mmproj mmproj-....gguf`. Do NOT quantize the mmproj.")
    n_1bit = sum(1 for _, t in overrides if str(t).startswith("iq1")) \
        + (1 if base_preset in ("IQ1_M", "IQ1_S") else 0)
    if n_1bit:
        print(f"WARNING: {n_1bit} group(s) hit the 1-bit floor (iq1) -- the budget forced "
              f"it. Expect real quality loss; sanity-check the output. Viable mainly on "
              f"giant MoE (redundancy absorbs it), rough on small/dense models.")

    tt_file = out + ".tensor-types.txt"
    ov_lines = [f"{pat}={t}" for pat, t in overrides]
    cmd = [a.llama_quantize]
    if requant:
        cmd += ["--allow-requantize"]
    if a.imatrix:
        cmd += ["--imatrix", a.imatrix]
    # a forced K-quant on a row length that cannot hold one aborts ggml mid-write (see block_safe_type)
    hidden = int(arch.get("hidden") or 0)
    safe_emb = block_safe_type(emb_type, hidden)
    if safe_emb != emb_type:
        print(f"   embeddings: {emb_type} -> {safe_emb}  (row length {hidden} is not a multiple of "
              f"{QK_K}; a forced K-quant aborts the build and leaves a truncated file)")
        emb_type = safe_emb
    cmd += ["--token-embedding-type", emb_type, "--output-tensor-type", emb_type,
            "--tensor-type-file", tt_file,
            a.gguf, out, base_preset]   # base preset DERIVED from the plan
    if a.threads:
        cmd += [str(a.threads)]          # llama-quantize takes nthreads as a trailing positional
    print()
    if a.plan_only:
        print(f"plan only -- {len(ov_lines)} tensor overrides; command that would run:")
        print("  " + " ".join(cmd))
        return
    arch_name = cfg.get("_gguf_arch")
    resolved = find_llama_bin(cmd[0], arch=arch_name)
    if resolved is None and find_llama_bin(cmd[0]) is not None and arch_name and arch_name != "unknown":
        # Present but too old for this architecture: update the managed engine from upstream and retry,
        # instead of stopping a build the day a new model family lands (POLLARD_AUTO_UPDATE=0 disables).
        from pollard_runtime_update import auto_update_for
        if auto_update_for(arch_name):
            resolved = find_llama_bin(cmd[0], arch=arch_name)
    if resolved is None:
        # distinguish "no binary at all" from "binary present but too old for THIS arch"
        present = find_llama_bin(cmd[0])
        if present is not None and arch_name and arch_name != "unknown":
            # Do NOT tell anyone to `git pull` a runtime tree unguarded. A tree can be carrying
            # local support a published model depends on, only as uncommitted edits -- that is how
            # `spark2_5` was lost, and how a model we shipped ended up with no runtime that could
            # open it. Capture first, then update, then put the patches back.
            sys.exit(
                f"ERROR: the llama.cpp build found ({present}) does not support the "
                f"'{arch_name}' architecture -- it predates support for this model.\n"
                f"Check what upstream has and whether this tree is carrying local work:\n"
                f"    pollard-runtime --arch {arch_name}\n"
                f"    pollard-runtime --dirty\n"
                f"Then update it safely (capture anything uncommitted BEFORE pulling):\n"
                f"    pollard-runtime --scan runtime/llama.cpp --capture pre-update\n"
                f"    cd runtime/llama.cpp && git pull && cmake --build build -j\n"
                f"    pollard-runtime --scan runtime/llama.cpp --apply pre-update\n"
                f"    pollard-runtime --verify\n"
                f"or point at a capable binary with --llama-quantize /path/to/llama-quantize.")
        sys.exit(f"ERROR: {cmd[0]} not found. install.sh builds it into "
                 f"runtime/llama.cpp/build/bin -- re-run install.sh, or pass "
                 f"--llama-quantize /path/to/llama-quantize.")
    cmd[0] = resolved
    with open(tt_file, "w") as f:
        f.write("\n".join(ov_lines) + "\n")
    print("building...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    try:
        import pollard_workspace as ws
        bpw = round(os.path.getsize(out) * 8.0 / arch["total"], 2) if arch.get("total") else None
        if a.record:
            ws.record_build(a.gguf, "gguf", out, tag=a.tag or base_preset, bpw=bpw)
    except Exception as e:
        print(f"   (warning: could not record build to the workspace manifest: {e})")
    print(f"\ndone: {out}")
    print("run it with stock llama.cpp / Ollama / LM Studio -- it is a normal GGUF.")

    # This build is STEP 1 of the gold recipe, not the whole thing. Say so, every time, unless the
    # caller is pollard-auto (which runs the remaining steps itself). A ladder rung shipped as if it
    # were the finished method is the single easiest mistake to make with this tool.
    if not os.environ.get("POLLARD_AUTO"):
        print("\n" + "=" * 72)
        print("NOTE: this is the fit-your-RAM BASELINE -- step 1 of the gold Pollard recipe.")
        print("The full method is:  Calib 3.0 imatrix -> measured automap FLAGSHIP mix -> coherence gate")
        print("Still to run for a gold build:")
        print("  2. pollard-automap ...     the mixed-precision flagship (the winner; needs ik_llama.cpp")
        print("                             for trellis IQ*_KT atoms)")
        print("  3. pollard-bench --gguf <build> --coherence     is it actually usable?")
        print("Or do all of it in one shot:  pollard --gguf <f16> --run")
        print("=" * 72)


if __name__ == "__main__":
    main()
