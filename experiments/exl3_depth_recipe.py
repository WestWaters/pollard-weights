#!/usr/bin/env python3
"""exl3_depth_recipe.py — self-QC for an exllamav3 (EXL3) cook, and the fix: read the converter's per-layer error lines,
build the depth profile, flag base-bit layers whose error is anomalous for their bit class, and emit (a) a per-tensor recipe
that moves routed-expert bits from the allocator's easiest above-base layers into the flagged layers (budget-neutral by default)
and (b) the bands to re-cook. Companion to notes/glm-5.3-744b-routing-and-smoothing-data.md §4.

Why: exllamav3's budgeted allocator decides bits per tensor from config + flags, before any measurement. On GLM-5.3 it gave the
first and last MoE layers 4 bits and everything else 3, while the measured per-layer error peaked mid-stack (layers 32-49,
28 dB sqnr vs 35-38 dB elsewhere at the same bits). The converter prints that error per module as it goes — so the cook can
check itself and re-run only the affected bands with a corrected recipe (each band restarts from the full-precision residual
stream, see the note), instead of a full re-cook.

    # dump the allocator's strategy for your flags (config-only, CPU; exllamav3 importable)
    python3 exl3_depth_recipe.py dump --model <hf-dir> --bits 3.2 --hq --out strategy.json
    # QC a finished (or running) cook from the converter logs; bands = the layer ranges each node converted
    python3 exl3_depth_recipe.py qc --logs work*/run.log --strategy strategy.json \
        --bands 0-7,8-15,16-23,24-31,32-39,40-47,48-55,56-63,64-70,71-77 --out qc.json --recipe recipe.yaml
    # SMART second pass: measured allocator over (layer, matrix) expert bits under the pass-1 byte budget
    python3 exl3_depth_recipe.py plan --logs work*/run.log --strategy strategy.json --bands ... --out recipe_plan.yaml
    # or hand-pick the move
    python3 exl3_depth_recipe.py recipe --strategy strategy.json --lift 32-44 --lift-bits 4 --drop auto --drop-bits 3 --out recipe.yaml
    # then: python convert.py ... --recipe recipe.yaml   (re-run only the bands the tool lists)

Flag rule (qc): within the BASE bit class (routed experts at the allocator's minimum bits), a layer is flagged when its sqnr is
more than --sqnr-gap dB (default 4) below the class median, or its rfn is more than --rfn-mult (default 2.5) times the class
median. Above-base layers are never lifted (they already have the extra bit); the ones with the highest sqnr are the drop
candidates. Lifted layers are widened to whole bands (so a re-run keeps the within-band sequential calibration) when --bands is
given. Budget-neutral = drop as many above-base layers as are lifted; --allow-bpw-increase lifts only (fewer bands to re-run,
more bytes). --lift-matrices down_proj lifts only the expert matrix that carries ~2x the error of gate/up at 3 bits
(one third of the bytes of a full layer lift for roughly half of the error reduction).
"""
import argparse, glob, json, math, re, statistics, sys

LINE = re.compile(r"Quantized: model\.layers\.(\d+)\s+bpw:\s*([\d.]+)\s+rfn:\s*([\d.]+)\s+cos:\s*([\d.]+)\s+sqnr:\s*([\d.]+)\s+\[([\d.]+) s\]")

def _layer(key):
    m = re.match(r"model\.layers\.(\d+)\.", key); return int(m[1]) if m else None

def _rng(s):
    if not s or s == "auto": return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-"); out.update(range(int(a), int(b or a) + 1))
    return sorted(out)

def _bands(s):
    if not s: return []
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-"); out.append((int(a), int(b or a)))
    return out

def parse_logs(paths):
    rows = {}
    for p in paths:
        for ln in open(p, errors="ignore"):
            m = LINE.search(ln)
            if m: rows[int(m[1])] = dict(layer=int(m[1]), bpw=float(m[2]), rfn=float(m[3]), cos=float(m[4]), sqnr=float(m[5]), secs=float(m[6]))
    return [rows[k] for k in sorted(rows)]

def expert_bits_by_layer(T):
    out = {}
    for k, b in T.items():
        if ".mlp.experts." in k: out.setdefault(_layer(k), set()).add(b)
    return {L: min(v) for L, v in out.items()}

def cmd_dump(a):
    from exllamav3 import Config, Model
    from exllamav3.conversion.allocation import create_q_strategy
    cfg = Config.from_directory(a.model); model = Model.from_config(cfg)
    mtp = Model.from_config(cfg, component="mtp") if "mtp" in cfg.model_classes else None
    strat, bpw = create_q_strategy(model, mtp, cfg, a.bits, a.head_bits, a.mtp_bits, a.hq)
    json.dump({"bits": a.bits, "hq": a.hq, "head_bits": a.head_bits, "mtp_bits": a.mtp_bits, "final_bpw": bpw, "tensors": strat}, open(a.out, "w"))
    print(f"allocator strategy: {len(strat)} budgeted tensors, {bpw:.3f} bpw -> {a.out}")

def make_recipe(T, lift, lift_bits, drop, drop_bits, out, note_extra="", lift_mats=("gate_proj", "up_proj", "down_proj")):
    import yaml
    R = dict(T); changed = 0
    for k in R:
        if ".mlp.experts." not in k: continue
        L = _layer(k)
        if L in lift and k.rsplit(".", 1)[-1] in lift_mats and R[k] != lift_bits: R[k] = lift_bits; changed += 1
        elif L in drop and R[k] != drop_bits: R[k] = drop_bits; changed += 1
    def ebits(s):
        e = [b for k, b in s.items() if ".mlp.experts." in k]; return sum(e) / len(e)
    yaml.safe_dump({"head_bits": 6, "note": f"routed experts ({'/'.join(lift_mats)}) lifted to {lift_bits} bits in layers {sorted(lift)} and dropped to {drop_bits} in layers {sorted(drop)}. {note_extra}".strip(), "tensors": R}, open(out, "w"), sort_keys=False)
    return changed, ebits(T), ebits(R)

def rerun_bands(bands, layers):
    return sorted({(a, b) for (a, b) in bands if any(a <= L <= b for L in layers)})

def cmd_recipe(a):
    d = json.load(open(a.strategy)); T = d["tensors"]; eb = expert_bits_by_layer(T); base = min(eb.values())
    above = sorted(L for L, b in eb.items() if b > base)
    lift = set(_rng(a.lift) or []); drop = set(above if a.drop == "auto" else (_rng(a.drop) or []))
    changed, e0, e1 = make_recipe(T, lift, a.lift_bits, drop, a.drop_bits, a.out, lift_mats=tuple(a.lift_matrices.split(",")))
    print(f"allocator: routed experts base {base} bits, above-base layers {above}")
    print(f"recipe: {changed} tensors changed; routed-expert mean bits {e0:.3f} -> {e1:.3f}; -> {a.out}")
    if a.bands: print("re-run bands:", rerun_bands(_bands(a.bands), list(lift | drop)))

MLINE = re.compile(r"Quantized: model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\s+bpw:\s*([\d.]+)\s+proxy_err:\s*([\d.]+)")

def parse_matrix_logs(paths):
    """mean per-expert proxy_err per (layer, matrix, bits) from the converter's per-tensor lines"""
    acc = {}
    for p in paths:
        for ln in open(p, errors="ignore"):
            m = MLINE.search(ln)
            if m: acc.setdefault((int(m[1]), m[3], int(round(float(m[4])))), []).append(float(m[5]))
    return {k: sum(v) / len(v) for k, v in acc.items()}

def cmd_plan(a):
    """measured allocator: fit err(bits) per expert matrix, amplification per layer, then greedy bit moves under a byte budget"""
    paths = sorted(sum((glob.glob(p) for p in a.logs), []))
    rows = parse_logs(paths); mats = parse_matrix_logs(paths)
    if not rows or not mats: sys.exit("need per-layer and per-matrix 'Quantized:' lines in the logs")
    d = json.load(open(a.strategy)); T = d["tensors"]; eb = expert_bits_by_layer(T)
    MATS = ("gate_proj", "up_proj", "down_proj"); by_layer = {r["layer"]: r for r in rows}
    layers = sorted(L for L in eb if L in by_layer and all((L, m, eb[L]) in mats for m in MATS))
    # 1) per-matrix error-vs-bits ratio (geometric per bit), from layers measured at two different bit-widths
    ratio = {}
    for m in MATS:
        lo = [v for (L, mm, b), v in mats.items() if mm == m and b == 3]; hi = [v for (L, mm, b), v in mats.items() if mm == m and b == 4]
        ratio[m] = (statistics.median(lo) / statistics.median(hi)) if lo and hi else a.default_ratio
    def err_m(L, m, b):   # predicted per-matrix proxy error at b bits, anchored on the layer's own measurement
        b0 = eb[L]; e0 = mats[(L, m, b0)]; return e0 * ratio[m] ** (b0 - b)
    # 2) per-layer amplification: measured output error / summed weight proxies at the pass-1 bits
    amp = {}
    for L in layers:
        out_err = 10 ** (-by_layer[L]["sqnr"] / 20); amp[L] = out_err / sum(err_m(L, m, eb[L]) for m in MATS)
    def layer_noise(L, bits):   # predicted output noise POWER of layer L with per-matrix bits
        return (amp[L] * sum(err_m(L, m, bits[m]) for m in MATS)) ** 2
    # numel per expert matrix (all equal for gate/up/down: hidden x moe_intermediate) — bytes scale with bits only
    cur = {L: {m: eb[L] for m in MATS} for L in layers}
    E = a.experts
    total0 = sum(cur[L][m] for L in layers for m in MATS)      # in "bits x (numel*E)" units
    budget = total0 + (a.extra_bits_per_layer_units or 0)
    noise = {L: layer_noise(L, cur[L]) for L in layers}
    def gain_up(L, m):
        if cur[L][m] >= a.max_bits: return None
        nb = dict(cur[L]); nb[m] += 1; return noise[L] - layer_noise(L, nb)          # noise removed per +1 bit-unit
    def loss_down(L, m):
        if cur[L][m] <= a.min_bits: return None
        nb = dict(cur[L]); nb[m] -= 1; return layer_noise(L, nb) - noise[L]          # noise added per -1 bit-unit
    moves = 0
    while True:
        ups = [(g, L, m) for L in layers for m in MATS for g in [gain_up(L, m)] if g is not None]
        if not ups: break
        g, L, m = max(ups)
        used = sum(cur[L2][m2] for L2 in layers for m2 in MATS)
        if used + 1 <= budget:            # budget allows growth: take the best lift
            cur[L][m] += 1; noise[L] = layer_noise(L, cur[L]); moves += 1; continue
        downs = [(l, L2, m2) for L2 in layers for m2 in MATS for l in [loss_down(L2, m2)] if l is not None and not (L2 == L and m2 == m)]
        if not downs: break
        l, L2, m2 = min(downs)
        if g <= l * (1 + a.min_improvement): break     # no swap improves total noise by the required margin
        cur[L][m] += 1; cur[L2][m2] -= 1; noise[L] = layer_noise(L, cur[L]); noise[L2] = layer_noise(L2, cur[L2]); moves += 1
    # 3) emit
    R = dict(T); changed_layers = set()
    for k in R:
        if ".mlp.experts." not in k: continue
        L = _layer(k); m = k.rsplit(".", 1)[-1]
        if L in cur and R[k] != cur[L][m]: R[k] = cur[L][m]; changed_layers.add(L)
    tot1 = sum(cur[L][m] for L in layers for m in MATS)
    print(f"plan: {len(layers)} MoE layers measured; per-bit error ratios gate {ratio['gate_proj']:.1f}x up {ratio['up_proj']:.1f}x down {ratio['down_proj']:.1f}x; {moves} bit moves")
    print(f"expert bit-units {total0} -> {tot1} ({'+' if tot1>=total0 else ''}{(tot1-total0)/max(total0,1)*100:.1f} %), total predicted noise power {sum(layer_noise(L, {m: eb[L] for m in MATS}) for L in layers):.3e} -> {sum(noise.values()):.3e}")
    print("layer : bits(gate/up/down) pass1 -> plan | measured sqnr -> predicted")
    for L in layers:
        b1 = eb[L]; nb = cur[L]; s1 = by_layer[L]["sqnr"]; s2 = -20 * math.log10(noise[L] ** 0.5)
        if nb != {m: b1 for m in MATS}: print(f"  L{L:02d}: {b1}/{b1}/{b1} -> {nb['gate_proj']}/{nb['up_proj']}/{nb['down_proj']} | {s1:.1f} -> {s2:.1f} dB")
    bands = _bands(a.bands)
    if bands: print("re-run bands:", rerun_bands(bands, sorted(changed_layers)), f"({len(rerun_bands(bands, sorted(changed_layers)))} of {len(bands)})")
    if a.out:
        import yaml
        yaml.safe_dump({"head_bits": 6, "note": f"measured allocation from pass-1 telemetry (exl3_depth_recipe.py plan): per-(layer,matrix) expert bits chosen to minimise predicted output noise under the pass-1 byte budget; {moves} moves, {len(changed_layers)} layers changed", "tensors": R}, open(a.out, "w"), sort_keys=False)
        json.dump({"ratio": ratio, "amp": amp, "bits": {str(L): cur[L] for L in layers}, "changed_layers": sorted(changed_layers), "pred_sqnr": {str(L): -20 * math.log10(noise[L] ** 0.5) for L in layers}}, open(a.out + ".plan.json", "w"), indent=1)
        print("->", a.out)

def cmd_qc(a):
    rows = parse_logs(sorted(sum((glob.glob(p) for p in a.logs), [])))
    if not rows: sys.exit("no 'Quantized: model.layers.N' lines found")
    by_layer = {r["layer"]: r for r in rows}
    d = json.load(open(a.strategy)) if a.strategy else None; T = d["tensors"] if d else {}
    eb = expert_bits_by_layer(T) if T else {}; base = min(eb.values()) if eb else None
    above = sorted(L for L, b in eb.items() if b > base) if eb else []
    moe_layers = set(eb) if eb else {r["layer"] for r in rows}
    base_rows = [r for r in rows if r["layer"] in moe_layers and (not eb or eb[r["layer"]] == base)]
    report, flagged = [], []
    if len(base_rows) >= 3:
        med_s = statistics.median(r["sqnr"] for r in base_rows); med_r = statistics.median(r["rfn"] for r in base_rows)
        for r in base_rows:
            bad = (med_s - r["sqnr"] > a.sqnr_gap) or (r["rfn"] > a.rfn_mult * med_r)
            report.append(dict(r, cls="base", cls_median_sqnr=med_s, cls_median_rfn=med_r, flagged=bad))
            if bad: flagged.append(r["layer"])
    for r in rows:
        if r["layer"] not in {x["layer"] for x in report}: report.append(dict(r, cls="above-base" if r["layer"] in above else "other", flagged=False))
    flagged = sorted(flagged); bands = _bands(a.bands)
    lift = set(flagged)
    if bands:
        for L in flagged:
            for (x, y) in bands:
                if x <= L <= y: lift.update(l for l in range(x, y + 1) if l in moe_layers and (not eb or eb[l] == base))
    drop = []
    if eb and not a.allow_bpw_increase and lift:
        cand = sorted((by_layer[L] for L in above if L in by_layer), key=lambda r: -r["sqnr"])    # easiest above-base first
        drop = [r["layer"] for r in cand[:len(lift)]]
        if len(drop) < len(lift): print(f"warning: only {len(drop)} above-base layers seen in logs to drop; recipe will raise bpw", file=sys.stderr)
    out = dict(layers_seen=len(rows), base_bits=base, allocator_above_base_layers=above, flagged=flagged, lift=sorted(lift), drop=sorted(drop),
               rule=dict(sqnr_gap_db=a.sqnr_gap, rfn_mult=a.rfn_mult, bands=a.bands), per_layer=sorted(report, key=lambda r: r["layer"]))
    if bands:
        out["rerun_bands_budget_neutral"] = rerun_bands(bands, sorted(lift) + drop); out["rerun_bands_lift_only"] = rerun_bands(bands, sorted(lift))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
    print(f"QC: {len(rows)} layers seen ({len(base_rows)} at base {base} bits); flagged {flagged}")
    for r in out["per_layer"]:
        if r["flagged"]: print(f"  L{r['layer']:02d} bpw {r['bpw']:.2f} sqnr {r['sqnr']:.1f} (base-class median {r['cls_median_sqnr']:.1f}) rfn {r['rfn']:.4f} (median {r['cls_median_rfn']:.4f})")
    if not flagged: print("no layer flagged — nothing to fix"); return
    if bands: print(f"re-run bands: budget-neutral {out['rerun_bands_budget_neutral']} | lift-only {out['rerun_bands_lift_only']}")
    if a.recipe and eb:
        changed, e0, e1 = make_recipe(T, lift, base + 1, set(drop), base, a.recipe, "generated by exl3_depth_recipe.py qc", lift_mats=tuple(a.lift_matrices.split(",")))
        print(f"recipe -> {a.recipe}: lift {sorted(lift)} -> {base+1} bits, drop {sorted(drop)} -> {base} bits; {changed} tensors; routed-expert mean bits {e0:.3f} -> {e1:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("dump"); p.add_argument("--model", required=True); p.add_argument("--bits", type=float, required=True); p.add_argument("--hq", action="store_true")
    p.add_argument("--head-bits", type=int, default=6); p.add_argument("--mtp-bits", type=int, default=8); p.add_argument("--out", default="strategy.json")
    p = sp.add_parser("recipe"); p.add_argument("--strategy", required=True); p.add_argument("--lift", required=True); p.add_argument("--lift-bits", type=int, default=4)
    p.add_argument("--drop", default="auto"); p.add_argument("--drop-bits", type=int, default=3); p.add_argument("--bands", default=""); p.add_argument("--out", default="recipe.yaml")
    p.add_argument("--lift-matrices", default="gate_proj,up_proj,down_proj", help="which expert matrices to lift (down_proj carries ~2x the error of gate/up at 3 bits)")
    p = sp.add_parser("qc"); p.add_argument("--logs", nargs="+", required=True); p.add_argument("--strategy"); p.add_argument("--bands", default=""); p.add_argument("--out"); p.add_argument("--recipe")
    p.add_argument("--sqnr-gap", type=float, default=4.0); p.add_argument("--rfn-mult", type=float, default=2.5); p.add_argument("--allow-bpw-increase", action="store_true")
    p.add_argument("--lift-matrices", default="gate_proj,up_proj,down_proj")
    p = sp.add_parser("plan", help="measured allocator: move expert bits between (layer, matrix) groups using pass-1 telemetry")
    p.add_argument("--logs", nargs="+", required=True); p.add_argument("--strategy", required=True); p.add_argument("--bands", default=""); p.add_argument("--out")
    p.add_argument("--min-bits", type=int, default=2); p.add_argument("--max-bits", type=int, default=5); p.add_argument("--experts", type=int, default=256)
    p.add_argument("--extra-bits-per-layer-units", type=int, default=0, help="budget slack in (layer,matrix) bit units; 0 = budget-neutral")
    p.add_argument("--min-improvement", type=float, default=0.02, help="a swap must cut total noise by this fraction of the loss side to be taken")
    p.add_argument("--default-ratio", type=float, default=4.0, help="per-bit error ratio if a matrix was measured at one bit-width only")
    a = ap.parse_args(); {"dump": cmd_dump, "recipe": cmd_recipe, "qc": cmd_qc, "plan": cmd_plan}[a.cmd](a)
