#!/usr/bin/env python3
"""Turn a learned (MAttr) tensor ordering into a llama-quantize --tensor-type-file at a REFERENCE build's
byte budget, so the two allocations differ only in WHERE the bits went.

The reference is a Pollard rung built from the probe's group ranking (its .tensor-types.txt). Its
per-atom byte totals are measured from the GGUF's own tensor shapes; the learned build gets the same
bytes in each atom, but assigns the high atom to the most-protected tensors and the low atom to the
most-crushable ones in MAttr's order. Same file size, same atoms, different placement -- the fair test.

    python experiments/mattr_to_tensor_types.py --gguf f16.gguf --ref-types rung.tensor-types.txt \
        --mattr experiments/mattr_qwen2.5-0.5b.json --out mattr.tensor-types.txt
"""
import argparse, json, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime", "llama.cpp", "gguf-py"))
import gguf

BPW = {"iq2_xxs": 2.0625, "iq2_xs": 2.3125, "iq2_s": 2.5, "iq3_xxs": 3.0625, "iq3_s": 3.4375, "iq3_m": 3.66,
       "iq4_xs": 4.25, "iq4_nl": 4.5, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625, "q8_0": 8.5}


def tensor_elems(path):
    r = gguf.GGUFReader(path)
    return {t.name: int(t.n_elements) for t in r.tensors}


def tensor_ne0(path):
    """First dimension per tensor. The IQ atoms (iq2_xxs / iq3_s / iq4_xs) need ne0 % 256 == 0;
    llama-quantize silently upgrades anything else to iq4_nl, in EVERY build. Those tensors are not
    allocatable and must not count against either side's budget."""
    r = gguf.GGUFReader(path)
    return {t.name: int(t.shape[0]) for t in r.tensors}


def ref_alloc(types_path, elems):
    """{tensor: atom} for every blk tensor the reference file assigns (patterns are regexes)."""
    rules = []
    for line in open(types_path):
        line = line.strip()
        if not line or "=" not in line:
            continue
        pat, typ = line.rsplit("=", 1)
        rules.append((re.compile(pat), typ.lower()))
    out = {}
    for name in elems:
        if not name.startswith("blk."):
            continue
        for pat, typ in rules:                       # first match wins, as llama-quantize applies them
            if pat.search(name):
                out[name] = typ
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="the f16 source (for tensor shapes)")
    ap.add_argument("--ref-types", required=True, help="the reference rung's .tensor-types.txt")
    ap.add_argument("--mattr", required=True, help="pollard_mattr result json (tensors[] with gguf + score_mean)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    elems = tensor_elems(a.gguf)
    ne0 = tensor_ne0(a.gguf)
    ref = ref_alloc(a.ref_types, elems)
    base = "iq3_s"
    fixed = {n for n in ref if ne0[n] % 256 != 0}          # iq4_nl whatever anyone asks for
    print(f"{len(fixed)} of {len(ref)} blk tensors are not 256-divisible (ne0={sorted({ne0[n] for n in fixed})}) -> iq4_nl in both builds")
    bytes_by_atom = {}
    for name, typ in ref.items():
        if name in fixed:
            continue
        bytes_by_atom[typ] = bytes_by_atom.get(typ, 0) + elems[name] * BPW[typ] / 8
    covered = set(ref) - fixed
    print("reference bytes per atom (blk tensors):", {k: f"{v/1e6:.1f} MB" for k, v in bytes_by_atom.items()})
    atoms = sorted(bytes_by_atom, key=lambda t: BPW[t])
    high = [t for t in atoms if BPW[t] > BPW[base]]
    low = [t for t in atoms if BPW[t] < BPW[base]]
    order = sorted(json.load(open(a.mattr))["tensors"], key=lambda r: r["score_mean"])    # most protected first
    names = [r["gguf"] for r in order if r["gguf"] in covered]
    missing = [r["gguf"] for r in order if r["gguf"] not in elems]
    if missing:
        print("WARNING: mattr names not in the gguf:", missing[:5])
    assign = {n: base for n in names}
    # protect from the top of the order until the high atoms' bytes are spent
    i = 0
    for t in sorted(high, key=lambda t: -BPW[t]):
        budget, spent = bytes_by_atom[t], 0.0
        while i < len(names) and spent + elems[names[i]] * BPW[t] / 8 <= budget * 1.02:
            assign[names[i]] = t; spent += elems[names[i]] * BPW[t] / 8; i += 1
    # crush from the bottom of the order until the low atoms' bytes are spent
    j = len(names) - 1
    for t in sorted(low, key=lambda t: BPW[t]):
        budget, spent = bytes_by_atom[t], 0.0
        while j > i and spent + elems[names[j]] * BPW[t] / 8 <= budget * 1.02:
            assign[names[j]] = t; spent += elems[names[j]] * BPW[t] / 8; j -= 1
    new_bytes = {}
    for n, t in assign.items():
        new_bytes[t] = new_bytes.get(t, 0) + elems[n] * BPW[t] / 8
    print("learned bytes per atom (blk tensors):  ", {k: f"{v/1e6:.1f} MB" for k, v in new_bytes.items()})
    tot_ref, tot_new = sum(bytes_by_atom.values()), sum(new_bytes.values())
    print(f"blk bytes: reference {tot_ref/1e6:.1f} MB  learned {tot_new/1e6:.1f} MB  ({(tot_new/tot_ref-1)*100:+.2f}%)")
    with open(a.out, "w") as f:
        for n in names:
            f.write(re.escape(n) + "=" + assign[n] + "\n")
        for n in sorted(fixed):
            f.write(re.escape(n) + "=iq4_nl\n")
    from collections import Counter
    print("learned atom counts:", dict(Counter(assign.values())), "->", a.out)
    print("protected (high atom):", [n for n in names if assign[n] in high][:10])


if __name__ == "__main__":
    main()
