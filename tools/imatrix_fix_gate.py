#!/usr/bin/env python3
"""Cover SwiGLU gate tensors in an ik_llama imatrix by copying their up counterpart.

In SwiGLU, gate_proj and up_proj take the SAME input (the layer hidden state), so
their input-activation importance -- exactly what the imatrix records -- is identical.
ik_llama's imatrix routinely SKIPS the gate side of fused MoE ffn (observed on
Qwen3-30B-A3B AND DeepSeek-V2-Lite AND Hy4): ffn_gate_exps, ffn_gate_shexp, and the
dense ffn_gate all go uncovered, which forces the trellis build to pin every gate
tensor to a high bit-width (q6_K) -- bloating the biggest param group and blowing the
size band. Copying each ffn_up* entry into its ffn_gate* name is mathematically exact
and lets the mix crush gate cleanly.

One rule: for every entry whose name contains 'ffn_up', synthesize the 'ffn_gate'
counterpart (ffn_up_exps->ffn_gate_exps, ffn_up_shexp->ffn_gate_shexp,
ffn_up.weight->ffn_gate.weight) if it is not already present.

Binary format: int32 n_entries, then per entry int32 name_len, name,
int32 ncall, int32 nval, float[nval].

Usage: imatrix_fix_gate.py <in.imatrix> [out.imatrix=in]
"""
import struct, sys

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path
d = open(path, "rb").read()
n = struct.unpack_from("<i", d, 0)[0]
pos = 4
entries = []
for _ in range(n):
    ln = struct.unpack_from("<i", d, pos)[0]; pos += 4
    name = d[pos:pos + ln]; pos += ln
    ncall = struct.unpack_from("<i", d, pos)[0]; pos += 4
    nval = struct.unpack_from("<i", d, pos)[0]; pos += 4
    floats = d[pos:pos + 4 * nval]; pos += 4 * nval
    entries.append((name, ncall, nval, floats))

have = {e[0] for e in entries}
added = {}
for name, ncall, nval, floats in list(entries):
    if b"ffn_up" in name:
        gname = name.replace(b"ffn_up", b"ffn_gate")
        if gname not in have:
            entries.append((gname, ncall, nval, floats))
            have.add(gname)
            kind = gname.split(b".")[-2] if b"." in gname else gname
            added[kind] = added.get(kind, 0) + 1

buf = struct.pack("<i", len(entries))
for name, ncall, nval, floats in entries:
    buf += struct.pack("<i", len(name)) + name + struct.pack("<i", ncall) + struct.pack("<i", nval) + floats
open(out, "wb").write(buf)
summary = ", ".join(f"{k.decode('ascii','replace')}:+{v}" for k, v in sorted(added.items()))
print(f"entries {n} -> {len(entries)} (copied up->gate: {summary or 'none'}); wrote {out}")
