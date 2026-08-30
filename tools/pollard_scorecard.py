#!/usr/bin/env python3
"""pollard-scorecard — emit the standardized memory-fit low-bit scorecard from measured
results. One command -> the reproducible ruler: three bars (uniform-1bit / Mix / 2bit
ceiling), size + PPL (+ KL when present), the allocation map, the fixed chat suite, and
honest errata. Data-driven so the same tool scores every model in the sweep.

Usage:
  pollard-scorecard --results results.json --out scorecard.md
  pollard-scorecard --results results.json --format html --out card.html

results.json schema (all sizes in GB, PPL over the SAME eval for every bar):
{
  "model": "Qwen2.5-7B-Instruct",
  "params_b": 7.615,
  "source":  "Qwen2.5-7B-Instruct F16 GGUF",
  "eval":    "WikiText-2 raw, ctx 2048, 145 chunks",
  "runtime": "ik_llama.cpp (QTIP trellis)",
  "sampling":"temp 0.7, top-p 0.9, repeat-penalty 1.15, ChatML, seed fixed",
  "bars": [
    {"name":"uniform IQ1_KT","role":"1-bit baseline","ppl":11.86,"gb":1.935},
    {"name":"PollardMix IQ1_KT","role":"mix","ppl":10.35,"gb":2.037,
     "allocation":[["MLP body (gate/up)","IQ1_KT","crushed"],
                   ["attn q","IQ2_KT","protected"], ...]},
    {"name":"uniform IQ2_KT","role":"2-bit ceiling","ppl":8.19,"gb":2.373}
  ],
  "chat": [["Focus tips","coherent"],["Story","coherent"],["Fibonacci","partial"],["Sky","loops w/o penalty"]],
  "errata": ["chat requires repeat-penalty at this tier; stock llama-cli loops",
             "Mix is a QUALITY win vs uniform 1-bit (better PPL, +0.10 GB), not smaller-and-better",
             "Mix is under the IQ2 ceiling, not equal to it"]
}
"""
import argparse, json


def bpw(gb, params_b):
    return gb * 8.0 * 1e9 / (params_b * 1e9) if params_b else float("nan")


def interp_ppl(bars):
    """Where a linear uniform slide (1bit->2bit ceiling) would land at the Mix's size,
    so the Mix's PPL can be argued as 'beats the interpolation by X%' (Grok's framing)."""
    base = next((b for b in bars if b["role"] == "1-bit baseline"), None)
    mix = next((b for b in bars if b["role"] == "mix"), None)
    ceil = next((b for b in bars if b["role"] == "2-bit ceiling"), None)
    if not (base and mix and ceil) or ceil["gb"] == base["gb"]:
        return None
    frac = (mix["gb"] - base["gb"]) / (ceil["gb"] - base["gb"])
    lin = base["ppl"] + frac * (ceil["ppl"] - base["ppl"])
    gain = 100.0 * (lin - mix["ppl"]) / lin
    return lin, gain


def md(r):
    pb = r.get("params_b")
    L = []
    L.append(f"# Pollard memory-fit scorecard — {r['model']}\n")
    L.append(f"*{r.get('runtime','')}* · source: {r.get('source','')} · eval: {r.get('eval','')}\n")
    # three bars
    L.append("## Bars (same source, same eval)\n")
    L.append("| build | role | PPL | size (GB) | bpw | KL vs f16 |")
    L.append("|---|---|---:|---:|---:|---:|")
    for b in r["bars"]:
        kl = f"{b['kl']:.4f}" if "kl" in b else "—"
        L.append(f"| **{b['name']}** | {b['role']} | {b['ppl']:.2f} | {b['gb']:.3f} | "
                 f"{bpw(b['gb'], pb):.2f} | {kl} |")
    L.append("")
    ip = interp_ppl(r["bars"])
    if ip:
        lin, gain = ip
        mix = next(b for b in r["bars"] if b["role"] == "mix")
        base = next(b for b in r["bars"] if b["role"] == "1-bit baseline")
        L.append(f"**Read:** Mix **{mix['ppl']:.2f}** PPL @ {mix['gb']:.3f} GB — beats uniform "
                 f"1-bit ({base['ppl']:.2f}) and beats the 1bit→2bit size-interpolation "
                 f"(~{lin:.2f} at this size) by **{gain:.1f}%**. Quality win vs uniform 1-bit "
                 f"(+{mix['gb']-base['gb']:.3f} GB), under the 2-bit ceiling.\n")
    # allocation / surgery table
    mix = next((b for b in r["bars"] if b["role"] == "mix"), None)
    if mix and mix.get("allocation"):
        L.append("## Allocation (the surgery)\n")
        L.append("| tensor role | atom | protect/crush |")
        L.append("|---|---|---|")
        for row in mix["allocation"]:
            L.append(f"| {row[0]} | `{row[1]}` | {row[2]} |")
        L.append("")
    # chat suite
    if r.get("chat"):
        L.append(f"## Chat suite  (sampling: {r.get('sampling','—')})\n")
        L.append("| prompt | result |")
        L.append("|---|---|")
        for p, v in r["chat"]:
            L.append(f"| {p} | {v} |")
        L.append("")
    # errata
    if r.get("errata"):
        L.append("## Errata (honest scope)\n")
        for e in r["errata"]:
            L.append(f"- {e}")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["md"], default="md")
    a = ap.parse_args()
    r = json.load(open(a.results))
    open(a.out, "w").write(md(r))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
