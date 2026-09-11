#!/usr/bin/env python3
"""dsv41_exl3_mix.py — assemble the per-layer exl3_L##.safetensors (+ ledger json) the splice wants from the pass outputs per recipe:
gate/up (w1, w3) from the K chosen for `gu`, down (w2) from the K chosen for `down`. Sources: --k3 DIR (full K=3 pass), --k4gu DIR
(gate/up-only K=4 pass), --k4w2 DIR (down-only K=4 pass). Pure safetensors/json (no torch): tensors are copied as raw bytes.
usage: dsv41_exl3_mix.py --recipe recipe_3p5.json --k3 DIR --k4gu DIR --k4w2 DIR --out DIR [--layers 0-39]"""
import argparse, json, os, struct, sys
ap = argparse.ArgumentParser(); ap.add_argument("--recipe", required=True); ap.add_argument("--k3", required=True); ap.add_argument("--k4gu", default="")
ap.add_argument("--k4w2", default=""); ap.add_argument("--out", required=True); ap.add_argument("--layers", default="0-39"); a = ap.parse_args()
def rng(s): out=set(); [out.update(range(int(p.split("-")[0]), int(p.split("-")[-1]) + 1)) for p in s.split(",")]; return sorted(out)
def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    return hdr, base
def write_st(path, parts, metadata):
    """parts: list of (name, info, src_path, src_base) — copies each tensor's bytes from its source file."""
    hdr = {"__metadata__": metadata}; off = 0; layout = []
    for name, info, sp, sb in parts:
        s, e = info["data_offsets"]; hdr[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [off, off + (e - s)]}; layout.append((sp, sb + s, e - s)); off += e - s
    h = json.dumps(hdr, separators=(",", ":")).encode(); pad = (8 - len(h) % 8) % 8; h += b" " * pad
    tmp = path + ".tmp"
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(h))); w.write(h)
        for sp, start, n in layout:
            with open(sp, "rb") as r:
                r.seek(start); left = n
                while left: b = r.read(min(left, 64 << 20)); w.write(b); left -= len(b)
    os.replace(tmp, path)
recipe = json.load(open(a.recipe))["layers"]; os.makedirs(a.out, exist_ok=True)
for L in rng(a.layers):
    LL = f"{L:02d}"; r = recipe.get(str(L), {"gu": 3, "down": 3}); src = {"gu": a.k3 if r["gu"] == 3 else a.k4gu, "down": a.k3 if r["down"] == 3 else a.k4w2}
    for m, d in src.items():
        if not d or not os.path.exists(os.path.join(d, f"exl3_L{LL}.safetensors")): sys.exit(f"L{LL}: no source for {m} at K={r[m]} ({d!r})")
    parts = []; led = {"layer": L, "experts": {}, "recipe": r, "sources": {m: src[m] for m in src}}
    for m, mats in (("gu", ("w1", "w3")), ("down", ("w2",))):
        p = os.path.join(src[m], f"exl3_L{LL}.safetensors"); hdr, base = read_st(p)
        for name, info in hdr.items():
            if name == "__metadata__": continue
            if name.split(".")[-2] in mats: parts.append((name, info, p, base))
        lj = json.load(open(os.path.join(src[m], f"exl3_L{LL}.json")))
        for e, rec in lj["experts"].items():
            led["experts"].setdefault(e, {"tokens": rec.get("tokens")}); [led["experts"][e].__setitem__(mm, rec[mm]) for mm in mats if mm in rec]
    parts.sort(key=lambda t: t[0]); out = os.path.join(a.out, f"exl3_L{LL}.safetensors")
    if r["gu"] == r["down"] and src["gu"] == src["down"]:
        os.path.exists(out) or os.link(os.path.join(src["gu"], f"exl3_L{LL}.safetensors"), out)   # same source file: hardlink
    else: write_st(out, parts, {"layer": str(L), "format": "pt", "mix": json.dumps(r)})
    json.dump(led, open(os.path.join(a.out, f"exl3_L{LL}.json"), "w"))
    print(f"L{LL}: gu K={r['gu']} down K={r['down']} -> {out} ({os.path.getsize(out) / 1e9:.2f} GB, {len(parts)} tensors)")
print("MIX-DONE")
