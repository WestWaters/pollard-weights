"""E5 — simulate the ACTUAL Pollard-weights design: a bounded hot-expert cache, streamed on demand.

This is the experiment that should have been run first. E2/E3 asked whether routing could be
*predicted*; the design never needed prediction. It needs the working set to be small:

    only ~3.1% of the weights are active per token, so hold a few GB hot and stream the misses.

So: replay the real routing trace token by token, keep a fixed-size cache of experts, and count how
many of each token's 8-experts-per-layer are already resident. That is exactly the machine the design
described — no index, no lookahead, just a hot set and flash behind it.

Reported per cache size:
  * hit rate (share of expert loads served from RAM)
  * bytes actually pulled from flash per token
  * implied tok/s at the measured flash rate, capped by RAM bandwidth

The number that matters is whether a 3–6 GB cache gets the flash traffic small enough to be fast.

Run:  python e5_cache_sim.py --jsonl routing_code.jsonl
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

EXPERT_MB = 1.945       # MEASURED from the Q4_K_M tensor table (down Q6_K + gate/up Q4_K)
NON_EXPERT_GIB = 0.5    # attention/embeddings/norms — always resident, excluded from the cache
FLASH_GBPS = 3.0        # README §1, 4 MB-block regime (2.4-4.9 GB/s measured)
RAM_TOKS_CAP = 202.0    # RAM-bandwidth ceiling for this active size (120 GB/s / 0.593 GiB)


def load(path):
    """Ordered [(prompt, pos, layer, [experts])] — replayed in token order."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append((d["prompt"], d["pos"], d["layer"], d["experts"]))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))     # token order, then depth
    return rows


def simulate(rows, cache_mb, warm=False):
    """LRU over (layer, expert). Returns (hit_rate, hits, misses, tokens).

    Keyed by (layer, expert) because expert 7 of layer 3 and expert 7 of layer 9 are different
    weights — a common off-by-one that would silently inflate the hit rate.
    """
    capacity = max(1, int(cache_mb / EXPERT_MB))
    cache = OrderedDict()
    hits = misses = 0
    if warm:                                   # pre-load the most-used experts (steady state)
        freq = {}
        for _p, _pos, layer, experts in rows:
            for e in experts:
                freq[(layer, e)] = freq.get((layer, e), 0) + 1
        for k, _n in sorted(freq.items(), key=lambda kv: -kv[1])[:capacity]:
            cache[k] = None
    for _p, _pos, layer, experts in rows:
        for e in experts:
            k = (layer, e)
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                cache[k] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    tokens = len({(r[0], r[1]) for r in rows})
    total = hits + misses
    return (hits / total if total else 0.0), hits, misses, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(Path(__file__).parent / "routing_code.jsonl"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load(a.jsonl)
    if not rows:
        raise SystemExit(f"no routing records in {a.jsonl}")
    tokens = len({(r[0], r[1]) for r in rows})
    layers = len({r[2] for r in rows})
    K = len(rows[0][3])
    pool = len({(r[2], e) for r in rows for e in r[3]})
    active_gib = EXPERT_MB * K * layers / 1024

    print(f"E5: {len(rows):,} records | {tokens:,} tokens | {layers} layers | top-{K}")
    print(f"    distinct (layer,expert) pairs touched: {pool:,} of {layers*256:,} possible "
          f"({pool/(layers*256)*100:.1f}%)")
    print(f"    active per token: {active_gib:.3f} GiB   (all experts: "
          f"{EXPERT_MB*256*layers/1024:.2f} GiB)")

    print(f"\n{'cache':>8}{'experts':>9}{'cold hit%':>11}{'warm hit%':>11}"
          f"{'flash GiB/tok':>15}{'tok/s':>9}")
    out_rows = []
    for gb in (1, 2, 3, 4, 6, 8, 10, 12):
        mb = gb * 1024
        cold, _h, _m, _t = simulate(rows, mb, warm=False)
        warm, _h, _m, _t = simulate(rows, mb, warm=True)
        miss_gib = (1.0 - warm) * active_gib
        toks = min(FLASH_GBPS / miss_gib, RAM_TOKS_CAP) if miss_gib > 1e-9 else RAM_TOKS_CAP
        cap = int(mb / EXPERT_MB)
        print(f"{gb:>6} GB{cap:>9}{cold*100:>10.1f}%{warm*100:>10.1f}%"
              f"{miss_gib:>15.4f}{toks:>9.1f}")
        out_rows.append({"cache_gb": gb, "experts": cap, "hit_cold": cold, "hit_warm": warm,
                         "flash_gib_per_token": miss_gib, "tok_s": toks})

    # What the design's own targets demand.
    print(f"\n  to hit a target, the cache must serve:")
    for target in (30, 70, 170):
        need_miss = FLASH_GBPS / target
        need_hit = 1.0 - need_miss / active_gib
        print(f"    {target:>4} tok/s -> {need_hit*100:6.2f}% hit  "
              f"(<= {need_miss*1024:.0f} MiB/token from flash)")

    best = max(r["hit_warm"] for r in out_rows)
    six = next((r for r in out_rows if r["cache_gb"] == 6), None)
    print("\n  VERDICT:", end=" ")
    if six and six["tok_s"] >= 70:
        print(f"THE DESIGN WORKS — a 6 GB hot cache serves {six['hit_warm']*100:.1f}% and implies "
              f"{six['tok_s']:.0f} tok/s while resident weights stay ~{6+NON_EXPERT_GIB:.1f} GB.")
    elif best < 0.5:
        print(f"working set is TOO WIDE — even the largest cache tried holds only {best*100:.1f}%. "
              "Expert demand in this workload is spread, so streaming on demand stays flash-bound.")
    else:
        print(f"PARTIAL — best hit rate {best*100:.1f}%. Faster than pure streaming, short of the "
              "target; the gap is how much of the pool this workload actually touches.")

    outp = Path(a.out or Path(__file__).parent.parent / "data" / "e5_cache_sim.json")
    outp.write_text(json.dumps({"tokens": tokens, "layers": layers, "top_k": K,
                                "pairs_touched": pool, "pairs_possible": layers * 256,
                                "active_gib_per_token": active_gib, "curve": out_rows}, indent=2))
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()
