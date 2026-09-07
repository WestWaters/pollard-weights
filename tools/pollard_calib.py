#!/usr/bin/env python3
"""pollard-calib — build a MULTI-DOMAIN calibration corpus (Calib 3.0).

The calibration corpus is what lets imatrix / pollard-sensitivity route to the tensors that
matter — and the biggest measured quality lever the good quantizers (Unsloth Dynamic 3.0) pay
for is NOT exotic types, it's a bigger, domain-MIXED calib (chat + code + math + multilingual +
prose) with enough coverage. Undercovered experts hard-fail low-bit ("Missing importance matrix
… bailing out") and bloat the build; a prose-only calib misranks. This tool assembles a balanced,
high-coverage corpus so every downstream imatrix/sensitivity pass sees all the domains.

It emits a raw text corpus (newline-separated samples) that works for BOTH `llama-imatrix -f`
(reads raw text) and `pollard-export --calib` / gptqmodel (one sample per line). Prefers real HF
datasets when `datasets` is importable and reachable; ALWAYS falls back to bundled seeds so it
produces a valid (smaller) corpus offline. Optionally writes a held-out split so you can check the
imatrix isn't overfit (KL on unseen text should track KL on the calib text).

  pollard-calib --out calib.txt                          # balanced default (~all domains)
  pollard-calib --out calib.txt --per-domain 400 --held-out calib.heldout.txt
  pollard-calib --out calib.txt --domains code,math,chat # only these (e.g. a coding model)
  pollard-calib --out calib.txt --min-chars 200 --seed 0
The imatrix/sensitivity COMPUTE (GPU) is a separate step — this only builds the corpus.
"""
import argparse, random, sys, textwrap

DOMAINS = ["prose", "code", "math", "chat", "multilingual"]

# HF datasets to try per domain (id, split, config, text-field or (a,b) pair to join).
# Kept small/streamable; any that fails to load just falls back to the bundled seed.
# Proven-working source FIRST per domain, then fallbacks, then bundled seed. Updated for `datasets` 5.x
# (bare "wikitext", bigcode/the-stack-smol, and script-based flores200 all broke — do not restore them).
_HF = {
    "prose":        [("Salesforce/wikitext", "train", "wikitext-103-raw-v1", "text"),  # large real prose
                     ("Salesforce/wikitext", "test", "wikitext-2-raw-v1", "text")],
    "code":         [("codeparrot/codeparrot-clean-valid", "train", None, "content"),
                     ("bigcode/the-stack-smol-xs", "train", "python", "content")],
    "math":         [("openai/gsm8k", "train", "main", ("question", "answer"))],
    "chat":         [("tatsu-lab/alpaca", "train", None, ("instruction", "output"))],
    "multilingual": [("papluca/language-identification", "train", None, "text"),
                     ("Salesforce/wikitext", "validation", "wikitext-2-raw-v1", "text")],
}

# Bundled seeds — always available so the tool never emits an empty domain. Short but real,
# spanning structure the imatrix needs to see (prose cadence, code tokens, digits/operators,
# instruction/answer turns, non-English scripts). Repeated+shuffled up to --per-domain.
_SEED = {
    "prose": [
        "The river carved the valley over ten thousand years, leaving terraces of pale stone.",
        "In economics, a market clears when the quantity supplied equals the quantity demanded.",
        "She opened the letter slowly, unsure whether the news would change everything or nothing.",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    ],
    "code": [
        "def binary_search(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] == x: return mid\n        if a[mid] < x: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
        "import numpy as np\narr = np.arange(12).reshape(3, 4)\nprint(arr.sum(axis=0), arr.mean())",
        "SELECT user_id, COUNT(*) AS n FROM events WHERE ts > NOW() - INTERVAL '7 days' GROUP BY user_id ORDER BY n DESC LIMIT 10;",
        "const memo = new Map();\nfunction fib(n){ if(n<2) return n; if(memo.has(n)) return memo.get(n); const v=fib(n-1)+fib(n-2); memo.set(n,v); return v; }",
    ],
    "math": [
        "Question: If 3x + 7 = 22, what is x?\nAnswer: 3x = 15, so x = 5.",
        "Question: A train travels 240 km in 3 hours. What is its average speed?\nAnswer: 240 / 3 = 80 km/h.",
        "Question: What is the derivative of f(x) = x^3 - 2x?\nAnswer: f'(x) = 3x^2 - 2.",
        "Question: Compute 17 * 24.\nAnswer: 17 * 24 = 408.",
    ],
    "chat": [
        "<|user|> Explain recursion in one sentence.\n<|assistant|> Recursion is when a function solves a problem by calling itself on a smaller version of that problem until it reaches a base case.",
        "<|user|> Give three tips for better sleep.\n<|assistant|> Keep a consistent schedule, avoid screens before bed, and keep the room cool and dark.",
        "<|user|> Summarize the water cycle.\n<|assistant|> Water evaporates, condenses into clouds, falls as precipitation, and flows back to the sea, repeating continuously.",
        "<|user|> What is the capital of France?\n<|assistant|> The capital of France is Paris.",
    ],
    "multilingual": [
        "La lluvia caía suavemente sobre los tejados de la ciudad dormida.",
        "Die Wissenschaft schreitet voran, indem sie Vermutungen streng überprüft.",
        "科学は仮説を厳密に検証することで前進する。",
        "La connaissance partagée est la seule qui grandit lorsqu'on la donne.",
        "المعرفة التي تُشارَك هي الوحيدة التي تكبر حين تُعطى.",
    ],
}


def _from_hf(domain, need, min_chars):
    """Try each HF source for a domain; return a list of text samples or [] if unavailable."""
    try:
        from datasets import load_dataset
    except Exception:
        return []
    out = []
    for src in _HF.get(domain, []):
        name, split, cfg, field = src
        try:
            ds = load_dataset(name, cfg, split=split, streaming=True) if cfg \
                else load_dataset(name, split=split, streaming=True)
            for ex in ds:
                if isinstance(field, tuple):
                    txt = "\n".join(str(ex.get(f, "")).strip() for f in field)
                else:
                    txt = str(ex.get(field, "")).strip()
                if len(txt) >= min_chars:
                    out.append(txt)
                if len(out) >= need:
                    break
            if out:
                return out
        except Exception as e:
            print(f"  [{domain}] HF source {name} unavailable ({repr(e)[:60]}) — trying next/seed",
                  file=sys.stderr)
    return out


def build_domain(domain, need, min_chars, rng):
    """`need` samples for one domain: real HF data first, topped up (shuffled/repeated) from seeds."""
    got = _from_hf(domain, need, min_chars)
    src = "hf" if got else "seed"
    seeds = _SEED.get(domain, [])
    while len(got) < need and seeds:
        batch = seeds[:]
        rng.shuffle(batch)
        got.extend(batch)
    return got[:need], src


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out", help="output corpus (raw text; default: workspace calibration/)")
    ap.add_argument("--domains", default=",".join(DOMAINS),
                    help=f"comma list from {DOMAINS} (default: all)")
    ap.add_argument("--per-domain", type=int, default=300, help="target samples per domain")
    ap.add_argument("--min-chars", type=int, default=120, help="drop samples shorter than this")
    ap.add_argument("--held-out", help="also write a disjoint held-out corpus (overfit check)")
    ap.add_argument("--held-frac", type=float, default=0.1, help="fraction of each domain held out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    doms = [d.strip() for d in a.domains.split(",") if d.strip()]
    bad = [d for d in doms if d not in DOMAINS]
    if bad:
        sys.exit(f"ERROR: unknown domain(s) {bad}; choose from {DOMAINS}")

    train, held, stats = [], [], []
    for d in doms:
        samples, src = build_domain(d, a.per_domain, a.min_chars, rng)
        rng.shuffle(samples)
        nh = int(len(samples) * a.held_frac) if a.held_out else 0
        held += [(d, s) for s in samples[:nh]]
        train += [(d, s) for s in samples[nh:]]
        stats.append((d, src, len(samples) - nh, nh))

    rng.shuffle(train)
    rng.shuffle(held)
    chars = sum(len(s) for _, s in train)
    if not a.out:
        try:
            import pollard_workspace as ws, os
            a.out = os.path.join(ws.calibration_dir(create=True), "calib-3.0.txt")
        except Exception:
            a.out = "calib-3.0.txt"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(s for _, s in train))
    print(f"== pollard-calib :: Calib 3.0 multi-domain corpus")
    for d, src, ntr, nh in stats:
        print(f"   {d:13s} {ntr:5d} train + {nh:4d} held  [{src}]")
    print(f"   TOTAL {len(train)} samples · ~{chars/1e6:.2f}M chars · ~{chars//4/1e3:.0f}K tokens (est)")
    print(f"   wrote {a.out}")
    if a.held_out:
        with open(a.held_out, "w", encoding="utf-8") as f:
            f.write("\n".join(s for _, s in held))
        print(f"   wrote {a.held_out} ({len(held)} held-out samples — overfit check)")
    if any(src == "seed" for _, src, _, _ in stats):
        print("   NOTE: some domains fell back to bundled seeds (no `datasets` / offline). Install "
              "`datasets` on the box for full-scale real-data coverage.", file=sys.stderr)


if __name__ == "__main__":
    main()
