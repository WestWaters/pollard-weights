#!/usr/bin/env python3
"""pollard-experts — surface the measured expert usage from a routing capture.

"Which experts does my workload actually use, and is there dead weight to skip?"
This reads a routing trace (an experiments/e2 capture: one JSON row per
token-layer, {"layer": L, "experts": [...]}) and reports, per layer, which
experts run hot, how concentrated the traffic is, and how much of the pool ever
gets touched.

What it answers — honestly:
  - the HOT list — the top experts per layer by activation frequency. These are
    the residency candidates: keep them near the compute, stream the rest.
  - COVERAGE — how much of the pool the workload touches at all. On a
    load-balanced router this is near-total in prefill (notes/e5: 97.6% of
    experts touched by ONE domain). So you cannot prune experts by topic; "hot"
    is a residency property measured live, not a fixed per-domain skip list.
    Decode concentrates ~2x (notes/e10); capture with --gen to see it.
  - a keep-list json (--out) — the (layer, expert) pairs that carry a target
    share of traffic, for a residency planner to consume.

This is a measurement report, not a runtime. It tells you what your workload
does; it does NOT delete experts (they nearly all fire). Pair it with
pollard-fit (bits) and pollard-run (placement).

Usage:
  pollard-experts --jsonl routing.jsonl
  pollard-experts --jsonl routing.jsonl --top 12 --layers 0,1,15,30
  pollard-experts --jsonl routing.jsonl --keep-frac 0.90 --out keeplist.json
"""
import argparse
import json
from collections import Counter, defaultdict


def load(path):
    """rows -> (prompt, pos, layer, [experts]). Tolerant of blank and torn lines
    (a capture killed mid-write leaves a truncated final row — skip, don't crash)."""
    rows, skipped = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rows.append((d.get("prompt", 0), d.get("pos", 0), int(d["layer"]),
                             [int(e) for e in d["experts"]]))
            except (ValueError, KeyError, TypeError):
                skipped += 1
    if skipped:
        print(f"[note] skipped {skipped} malformed line(s) (torn capture) — using the rest")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--jsonl", required=True, help="routing capture from experiments/e2")
    ap.add_argument("--top", type=int, default=8, help="hot experts to list per layer (default 8)")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices to detail (default: a sample)")
    ap.add_argument("--keep-frac", type=float, default=None,
                    help="report the (layer,expert) set carrying this share of traffic per layer")
    ap.add_argument("--out", default=None, help="write the keep-list json here")
    a = ap.parse_args()

    rows = load(a.jsonl)
    if not rows:
        raise SystemExit(f"no routing records in {a.jsonl} — did the capture run?")

    per_layer = defaultdict(Counter)                 # layer -> Counter(expert -> hits)
    for _p, _pos, layer, experts in rows:
        for e in experts:
            per_layer[layer][e] += 1
    layers = sorted(per_layer)
    top_k = len(rows[0][3])
    tokens = len({(p, pos) for p, pos, _l, _e in rows})
    n_experts = max(max(c) for c in per_layer.values()) + 1     # inferred from ids seen

    print(f"== pollard-experts :: {a.jsonl}")
    print(f"records {len(rows):,} | {tokens:,} tokens | {len(layers)} layers | "
          f"top-{top_k} | pool ~{n_experts} experts/layer (inferred from ids)")

    # ---- coverage: is there dead weight to skip? ----------------------------
    touched = [len(per_layer[l]) for l in layers]
    cov = sum(touched) / (len(layers) * n_experts)
    print(f"\ncoverage      : {cov*100:.1f}% of the expert pool is touched "
          f"({min(touched)}-{max(touched)} of ~{n_experts} per layer)")

    # ---- concentration: how hot is the hot set? -----------------------------
    q = max(1, n_experts // 4)
    shares = []
    for l in layers:
        c = per_layer[l]
        tot = sum(c.values())
        shares.append(sum(n for _e, n in c.most_common(q)) / tot if tot else 0.0)
    tq = sum(shares) / len(shares)
    print(f"concentration : the hottest 25% of experts carry {tq*100:.1f}% of traffic "
          f"(flat = 25%; higher = a real hot set)")
    if cov > 0.90 and tq < 0.35:
        print("  -> near-uniform: no per-topic dead weight here (prefill-style). "
              "Capture decode\n     traffic (--gen) — it concentrates ~2x (notes/e10).")
    elif tq >= 0.45:
        print("  -> real hot set: keep the top experts resident and stream the rest "
              "(feed this to pollard-run).")

    # ---- the hot list -------------------------------------------------------
    if a.layers:
        show = sorted({int(x) for x in a.layers.split(",")} & set(layers))
    else:
        mid = len(layers) // 2
        show = sorted(set(layers[:2] + layers[mid:mid + 1] + layers[-2:]))
    print(f"\nhot experts (top {a.top} per layer, expert:hits):")
    for l in show:
        c = per_layer[l]
        tot = sum(c.values())
        top = c.most_common(a.top)
        share = sum(n for _e, n in top) / tot if tot else 0.0
        listed = "  ".join(f"{e}:{n}" for e, n in top)
        print(f"  L{l:>3} ({share*100:4.1f}% of layer): {listed}")

    # ---- keep-list for a residency planner ----------------------------------
    if a.keep_frac is not None:
        keep, kept, total = {}, 0, 0
        for l in layers:
            c = per_layer[l]
            tot = sum(c.values())
            total += tot
            acc, ids = 0, []
            for e, n in c.most_common():
                if tot and acc / tot >= a.keep_frac:
                    break
                ids.append(e)
                acc += n
            keep[str(l)] = ids
            kept += acc
        n_pairs = sum(len(v) for v in keep.values())
        pool = len(layers) * n_experts
        print(f"\nkeep-list     : {n_pairs} (layer,expert) pairs — {n_pairs/pool*100:.0f}% "
              f"of the pool — carry {kept/total*100:.1f}% of traffic at frac {a.keep_frac}")
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"keep_frac": a.keep_frac, "n_experts": n_experts,
                           "top_k": top_k, "experts_by_layer": keep}, f, indent=2)
            print(f"  -> {a.out}")

    print("\nnote: 'touched' is not 'prunable' — a load-balanced router fires nearly every\n"
          "      expert (notes/e5). This lists what runs HOT so it can stay resident; it does\n"
          "      not delete experts. Bits: pollard-fit. Placement: pollard-run.")


if __name__ == "__main__":
    main()
