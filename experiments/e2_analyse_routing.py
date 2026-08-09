"""E2 (analysis half) — does routing actually REUSE, and how big must the hot cache be?

README §6 lists four open questions. This script answers the first three from one capture, and it is
built to be able to KILL the idea, the way E1 killed the dense version:

  1. Routing-pattern reuse — the decisive number. The design assumes real traffic revisits a small
     set of expert combinations out of C(256,8) ~= 4e14 per layer. If the new-pattern rate never
     falls, the online index never converges and it is over.
  2. Hot-expert cache hit rate — README §4 marks the ~126 tok/s figure as resting on a *Zipf
     assumption, not a measurement*. Expert frequencies give us the real curve, so we replace the
     assumption with a number and report the hit rate an actual RAM budget buys.
  3. Depth dependence — E1 found sparsity concentrated in the last third on a 0.6B dense model and
     flagged it as possibly an artifact. Per-layer reuse tests whether depth structure survives at
     MoE scale.
  4. Distribution shift — code / prose / math are captured separately, so "converged" can be
     checked as per-domain rather than assumed global.

The headline sensitivity: per-token flash traffic is (1 - hit_rate) x active_bytes. Because
active_bytes is ~2 GB, the arithmetic is brutally non-linear near the top — 90% is not "most of the
way there". §5 of the output prints exactly what hit rate the target speed demands.

Run:  python e2_analyse_routing.py --jsonl routing.jsonl
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# From README §3, measured on the real Laguna-XS.2 config — not guesses.
EXPERT_MB = 6.29           # one expert's weights
FLASH_GBPS = 3.0           # 4 MB-block regime, README §1 (2.4-4.9 GB/s measured)
RAM_REALISTIC_TOKS = 70.0  # README §4: realistic M4 RAM-bandwidth ceiling for this active size


def load(path):
    """rows -> (tag, prompt, pos, layer, frozenset(experts))"""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append((d["tag"], d["prompt"], d["pos"], d["layer"], frozenset(d["experts"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(Path(__file__).parent / "routing.jsonl"))
    ap.add_argument("--ram-budget-gb", type=float, default=8.0,
                    help="RAM available for the hot-expert cache on a 16GB machine")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load(a.jsonl)
    if not rows:
        raise SystemExit(f"no routing records in {a.jsonl} — did the capture run?")

    layers = sorted({r[3] for r in rows})
    tags = sorted({r[0] for r in rows})
    n_used = len(rows[0][4])
    tokens = len({(r[1], r[2]) for r in rows})
    print(f"E2: {len(rows):,} records | {tokens:,} tokens | {len(layers)} layers | "
          f"top-{n_used} | domains: {', '.join(tags)}")

    # ---- 1. reuse, per layer -------------------------------------------------
    # Occurrences vs distinct patterns. reuse = fraction of hits an already-populated index serves.
    per_layer = defaultdict(Counter)
    for tag, _p, _pos, layer, experts in rows:
        per_layer[layer][experts] += 1

    print(f"\n{'layer':>6}{'occurrences':>13}{'distinct':>10}{'reuse%':>9}{'top-pattern%':>14}")
    reuse_by_layer = {}
    for layer in layers:
        c = per_layer[layer]
        total = sum(c.values())
        distinct = len(c)
        reuse = 1.0 - distinct / total
        top = c.most_common(1)[0][1] / total
        reuse_by_layer[layer] = reuse
        if layer < 3 or layer >= max(layers) - 2 or layer % 8 == 0:
            print(f"{layer:>6}{total:>13,}{distinct:>10,}{reuse*100:>8.1f}{top*100:>13.1f}")

    mean_reuse = sum(reuse_by_layer.values()) / len(reuse_by_layer)
    third = max(1, len(layers) // 3)
    band = {
        "first": sum(reuse_by_layer[l] for l in layers[:third]) / third,
        "middle": sum(reuse_by_layer[l] for l in layers[third:2 * third]) / third,
        "last": sum(reuse_by_layer[l] for l in layers[2 * third:]) / max(1, len(layers) - 2 * third),
    }
    print(f"\n  MEAN reuse {mean_reuse*100:.1f}%   "
          f"by depth: first {band['first']*100:.1f}%  middle {band['middle']*100:.1f}%  "
          f"last {band['last']*100:.1f}%   (open Q3)")

    # ---- 2. convergence -----------------------------------------------------
    # Does the new-pattern rate DECAY? A flat rate means an index that grows forever.
    ordered = sorted(rows, key=lambda r: (r[1], r[2], r[3]))
    seen, curve, bucket, new_in_bucket = set(), [], 0, 0
    BUCKET = max(1, len(ordered) // 10)
    for i, (_tag, _p, _pos, layer, experts) in enumerate(ordered):
        key = (layer, experts)
        if key not in seen:
            seen.add(key)
            new_in_bucket += 1
        bucket += 1
        if bucket == BUCKET:
            curve.append(new_in_bucket / bucket)
            bucket, new_in_bucket = 0, 0

    print("\n  new-pattern rate over time (decile):")
    print("   " + "  ".join(f"{r*100:.0f}%" for r in curve))
    if len(curve) >= 2:
        decay = curve[-1] / curve[0] if curve[0] > 0 else 0.0
        print(f"   first decile {curve[0]*100:.1f}%  ->  last {curve[-1]*100:.1f}%  "
              f"(ratio {decay:.2f}; <1 means converging)")

    # ---- 3. expert frequency -> real cache curve ----------------------------
    # Replaces README §4's Zipf ASSUMPTION. Cache holds the top-C experts per layer; hit rate is the
    # share of that layer's expert-slot demand those C serve.
    per_layer_expert = defaultdict(Counter)
    for _tag, _p, _pos, layer, experts in rows:
        for e in experts:
            per_layer_expert[layer][e] += 1

    def hit_rate_at(c_per_layer):
        hit = tot = 0
        for layer in layers:
            counts = per_layer_expert[layer].most_common()
            tot += sum(n for _e, n in counts)
            hit += sum(n for _e, n in counts[:c_per_layer])
        return hit / tot if tot else 0.0

    budget_c = int((a.ram_budget_gb * 1024) / (EXPERT_MB * len(layers)))
    print(f"\n  hot-expert cache (per layer, {EXPERT_MB} MB/expert, {len(layers)} layers):")
    print(f"{'experts/layer':>15}{'cache GB':>11}{'hit%':>8}{'implied tok/s':>15}")
    active_gb = n_used * len(layers) * EXPERT_MB / 1024.0
    rows_out = []
    for c in sorted({1, 2, 4, 8, 16, 32, 64, 128, max(1, budget_c)}):
        if c > 256:
            continue
        h = hit_rate_at(c)
        gb = c * len(layers) * EXPERT_MB / 1024.0
        miss_gb = (1.0 - h) * active_gb
        toks = FLASH_GBPS / miss_gb if miss_gb > 1e-9 else float("inf")
        toks = min(toks, RAM_REALISTIC_TOKS)  # RAM bandwidth binds above this (README §4)
        mark = "  <- your budget" if c == budget_c else ""
        print(f"{c:>15}{gb:>11.1f}{h*100:>8.1f}{toks:>15.1f}{mark}")
        rows_out.append({"experts_per_layer": c, "cache_gb": gb, "hit_rate": h, "tok_s": toks})

    # ---- 4. distribution shift ---------------------------------------------
    print("\n  cross-domain pattern overlap (open Q4):")
    by_tag = defaultdict(set)
    for tag, _p, _pos, layer, experts in rows:
        by_tag[tag].add((layer, experts))
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1:]:
            inter = len(by_tag[t1] & by_tag[t2])
            union = len(by_tag[t1] | by_tag[t2])
            print(f"    {t1:>6} vs {t2:<6} jaccard {inter/union*100:>5.1f}%  "
                  f"({inter:,} shared of {union:,})")

    # ---- 5. what the target actually demands -------------------------------
    print(f"\n  sensitivity — active {active_gb:.2f} GB/token at {FLASH_GBPS} GB/s flash:")
    for target in (10, 30, 70, 126):
        need_miss = FLASH_GBPS / target
        need_h = 1.0 - need_miss / active_gb
        verdict = "impossible" if need_h >= 1.0 else f"needs {need_h*100:.2f}% hit"
        print(f"    {target:>4} tok/s -> {verdict}")

    # ---- verdict ------------------------------------------------------------
    print("\n  VERDICT:", end=" ")
    converging = len(curve) >= 2 and curve[-1] < curve[0] * 0.5
    if mean_reuse < 0.10:
        print("NO REUSE — patterns are nearly all unique. The index cannot converge; "
              "H1 dies here the way it died on dense in E1.")
    elif mean_reuse < 0.50:
        print(f"WEAK reuse ({mean_reuse*100:.1f}%) — an index helps some, but cold-start dominates. "
              "Marginal; the hot-expert cache is doing the real work, not the index.")
    else:
        print(f"REUSE IS REAL ({mean_reuse*100:.1f}%)"
              + (" and converging" if converging else " but NOT yet converging on this sample")
              + " — proceed to E3 (measured tok/s end to end).")
    print("  NOTE: prefill-only capture on a few prompts. Reuse measured within similar text is an\n"
          "        UPPER BOUND on reuse across a real workload. Treat as a ceiling, not a forecast.")

    out = Path(a.out or Path(__file__).parent.parent / "data" / "e2_routing_reuse.json")
    out.write_text(json.dumps({
        "records": len(rows), "tokens": tokens, "layers": len(layers), "top_k": n_used,
        "mean_reuse": mean_reuse, "reuse_by_depth": band,
        "new_pattern_curve": curve,
        "reuse_by_layer": {str(k): v for k, v in reuse_by_layer.items()},
        "cache_curve": rows_out,
        "ram_budget_gb": a.ram_budget_gb, "active_gb_per_token": active_gb,
    }, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
