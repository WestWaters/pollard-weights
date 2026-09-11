#!/usr/bin/env python3
"""dsv41_exl3_alloc.py — budgeted K allocation per (layer, gu|down) for the DSV41 route-B EXL3 experts, from measured ledgers.

cost(L, m, K) = gain_L^2 * sum_e tokens_e * proxy_err_e(m, K)      proxy_err = tr(E^T H E) / tr(W^T H W) = relative output-energy error
                                                                      (exllamav3 quantize.py), token-weighted per expert, gu = w1 + w3.
gain_L = |hc_ffn_post_mean| / rms(stream_rms_after) from the capture stats (how much a unit of ffn-output error moves the residual stream;
7x at L3-6, <1 from L28, 0.04 at L39). Greedy: start all K=3, upgrade the (L, m) with the best (cost3 - cost4) per added bit until the
target bpw is met. Missing K=4 ledgers are ESTIMATED from the measured K4/K3 proxy ratio of tensors that have both (fallback 0.28) and
reported as "needs cook" — that list is the gu K=4 top-up pass. Output: recipe json (layers.L.gu/down = K, sources, bpw, needs_cook).
usage: dsv41_exl3_alloc.py --k3 ledgers/k3 [--k4w2 ledgers/k4w2] [--k4gu ledgers/k4gu] --stats ledgers/stats_r78 ledgers/stats_r312
       --target-bpw 3.5 --out recipe_3p5.json
"""
import argparse, glob, json, math, os, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--k3", required=True); ap.add_argument("--k4w2", default=""); ap.add_argument("--k4gu", default="")
ap.add_argument("--stats", nargs="+", required=True); ap.add_argument("--target-bpw", type=float, default=3.5)
ap.add_argument("--layers", type=int, default=40); ap.add_argument("--out", default=""); ap.add_argument("--no-gain", action="store_true")
ap.add_argument("--margin", type=int, default=4, help="extra next-best candidates per matrix type to cook beyond the picked set")
ap.add_argument("--k4-ratio", type=float, default=None, help="override the K4/K3 proxy_err ratio used for missing K=4 ledgers")
a = ap.parse_args()
NUMEL = 5120 * 2304; MATS = {"gu": ("w1", "w3"), "down": ("w2",)}

def load_dir(d):
    out = {}
    for f in glob.glob(os.path.join(d, "exl3_L*.json")):
        j = json.load(open(f)); out[int(j["layer"])] = j["experts"]
    return out

def cost(experts, mats):
    """token-weighted sum of proxy_err over experts and the given matrices; None if any matrix is missing."""
    tot = 0.0; n = 0
    for e, rec in experts.items():
        tok = max(int(rec.get("tokens", 0)), 1)
        for m in mats:
            if m not in rec or rec[m].get("proxy_err", -1) < 0: continue
            tot += tok * rec[m]["proxy_err"]; n += 1
    return (tot, n) if n else (None, 0)

def gain(L):
    g = []
    for d in a.stats:
        p = os.path.join(d, f"extra_L{L:02d}.json")
        if not os.path.exists(p): continue
        j = json.load(open(p)); post = j["hc_ffn_post_mean"]; s = j["stream_rms_after"]
        g.append(math.sqrt(sum(x * x for x in post)) / math.sqrt(sum(x * x for x in s) / len(s)))
    return st.mean(g) if g else 1.0

k3 = load_dir(a.k3); k4 = {"down": load_dir(a.k4w2) if a.k4w2 else {}, "gu": load_dir(a.k4gu) if a.k4gu else {}}
layers = sorted(L for L in k3 if L < a.layers)
missing = [L for L in range(a.layers) if L not in k3]
# measured K4/K3 ratio where both exist
ratios = []
for m, mats in MATS.items():
    for L in layers:
        if L in k4[m]:
            c3, _ = cost(k3[L], mats); c4, _ = cost(k4[m][L], mats)
            if c3 and c4: ratios.append(c4 / c3)
k4_ratio = a.k4_ratio if a.k4_ratio is not None else (st.median(ratios) if ratios else 0.28)
rows = {}
for L in layers:
    gL = 1.0 if a.no_gain else gain(L)
    for m, mats in MATS.items():
        c3, n3 = cost(k3[L], mats)
        if c3 is None: continue
        if L in k4[m]:
            c4, _ = cost(k4[m][L], mats); measured = True
        else:
            c4 = c3 * k4_ratio; measured = False
        rows[(L, m)] = {"gain": gL, "c3": c3 * gL * gL, "c4": c4 * gL * gL, "measured_k4": measured, "n": n3}
# greedy upgrade
K = {(L, m): 3 for (L, m) in rows}
def bpw():
    bits = sum(K[(L, m)] * len(MATS[m]) for (L, m) in K); mats_n = sum(len(MATS[m]) for (L, m) in K)
    return bits / mats_n + 0.01                                   # +0.01: suh/svh/scale overhead measured (3.010 at K=3)
cands = sorted(rows, key=lambda k: -(rows[k]["c3"] - rows[k]["c4"]) / len(MATS[k[1]]))
picked = []
for key in cands:
    if bpw() >= a.target_bpw: break
    K[key] = 4; picked.append(key)
tot3 = sum(r["c3"] for r in rows.values()); tot = sum(rows[k]["c4"] if K[k] == 4 else rows[k]["c3"] for k in rows)
print(f"layers with K=3 ledgers: {len(layers)}/{a.layers} (missing {missing if missing else 'none'}); K4/K3 proxy ratio: "
      f"{'measured ' + format(k4_ratio, '.3f') + ' from ' + str(len(ratios)) + ' tensors' if ratios else 'assumed ' + format(k4_ratio, '.2f')}")
print(f"target bpw {a.target_bpw}: reached {bpw():.3f}; weighted error energy {tot / tot3:.3f} of all-K3; upgrades {len(picked)} of {len(rows)} (layer, mat) slots")
print("\n L  gain   gu:K  cost3      down:K  cost3      | picked")
for L in layers:
    g = rows.get((L, "gu")); d = rows.get((L, "down"))
    if not g or not d: continue
    print(f"{L:2d} {g['gain']:5.2f}   {K[(L,'gu')]}  {g['c3']:.3e}    {K[(L,'down')]}  {d['c3']:.3e}   | "
          + " ".join(m for m in ("gu", "down") if K[(L, m)] == 4))
needs = sorted((L, m) for (L, m) in picked if not rows[(L, m)]["measured_k4"])
# cook lists = picked-but-unmeasured + the next `margin` best unpicked candidates per matrix type (so the final measured allocation has room)
cook = {m: [L for (L, mm) in needs if mm == m] for m in MATS}
for m in MATS:
    extra = [L for (L, mm) in cands if mm == m and (L, mm) not in picked and not rows[(L, mm)]["measured_k4"]][:a.margin]
    cook[m] = sorted(set(cook[m]) | set(extra))
print(f"cook lists (picked + margin {a.margin}): gu {cook['gu']}  down {cook['down']}")
print(f"\nneeds K=4 cook (not yet measured): gu {[L for L, m in needs if m == 'gu']} down {[L for L, m in needs if m == 'down']}")
per_rank_gib = sum(K[(L, m)] * len(MATS[m]) * NUMEL * 384 for (L, m) in K) / 8 / 4 / 2**30 * (a.layers / max(len(layers), 1))
print(f"experts per rank at this recipe ≈ {per_rank_gib:.1f} GiB (shipped MXFP4 4.25 bpw ≈ {4.25 * 3 * NUMEL * 384 * a.layers / 8 / 4 / 2**30:.1f} GiB)")
if a.out:
    json.dump({"target_bpw": a.target_bpw, "bpw": bpw(), "k4_ratio": k4_ratio, "layers": {str(L): {m: K[(L, m)] for m in MATS if (L, m) in K} for L in layers},
               "needs_cook": [{"layer": L, "mat": m} for L, m in needs], "cook_gu": cook["gu"], "cook_down": cook["down"], "rows": {f"{L}.{m}": r for (L, m), r in rows.items()}}, open(a.out, "w"), indent=1)
    print("wrote", a.out)
