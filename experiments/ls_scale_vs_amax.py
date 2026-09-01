#!/usr/bin/env python3
"""Weighted least-squares scale vs absmax — measure the 'Sherry'-style encoder lever.

For a low-bit group, the scale d sets reconstruction error. absmax pins d to the single
largest |weight| (outlier-dominated); the importance-weighted LS optimum for fixed codes u
is closed-form:  d = Σ(w·u·x) / Σ(w·u²)  (iterate: reassign u=clamp(round(x/d)), resolve d).

Tencent's Hy4 MIX-STQ1_0 card claims this beats absmax by -89.7% weighted-SSD. We do NOT
trust that number — this measures it. Pure-Python synthetic runs anywhere (no deps); the
real-weight mode (numpy + ik_llama gguf-py) reads an actual f16 tensor + its imatrix.

MEASURED — LS vs AMAX (real 30B tensor `blk.0.ffn_gate_inp.weight` + its imatrix, gs 256):
    ternary -> LS cuts weighted-SSD 32.1% ; 2-bit 19.4% ; 3-bit 13.0%   (LS always beats amax)
MEASURED — LS vs the ASYMMETRIC SWEEP (torch A/B, outlier block): the sweep already beats amax,
and vs it LS wins ONLY at ternary:
    ternary -> LS +32.6%  ;  2-bit -> LS -9.7% ; 3-bit -> LS -20.9%
So the honest result: symmetric weighted-LS is the right encoder for the TERNARY / 1-bit crush
tier (the STQ1_0/Sherry tier, where Pollard crushes experts) — there it beats both amax and the
sweep by ~a third. At 2-3 bit the asymmetric sweep wins (symmetric LS wastes a level). NOT the
-89.7% the Sherry card claims. Adopted as `imatrix_quantize(..., scale="ls")` (use it at ternary);
the GGUF lane would need the same solve in ik_llama's C make_qx_quants kernel.

    python experiments/ls_scale_vs_amax.py                       # synthetic (stdlib only)
    python experiments/ls_scale_vs_amax.py --gguf f16.gguf --imatrix ik.imatrix   # real (numpy+gguf-py)
"""
import argparse, random, statistics, struct, sys


def _cr(v, Q):
    q = round(v)
    return -Q if q < -Q else (Q if q > Q else q)


def amax_scale(x, Q):
    return max(abs(v) for v in x) / Q


def ls_scale(x, w, Q, iters=5):
    d = amax_scale(x, Q) or 1e-9
    for _ in range(iters):
        num = den = 0.0
        for xi, wi in zip(x, w):
            u = _cr(xi / d, Q); num += wi * u * xi; den += wi * u * u
        if den > 0:
            d = num / den
    return d


def wssd(x, w, d, Q):
    return sum(wi * (xi - d * _cr(xi / d, Q)) ** 2 for xi, wi in zip(x, w))


def synthetic():
    random.seed(0)
    print("synthetic (LLM-like: small weights + fat-tail outliers, imatrix-like importance)")
    for name, Q in [("ternary", 1), ("2-bit", 2), ("3-bit", 4)]:
        rs = []
        for _ in range(400):
            x = [random.gauss(0, 0.02) for _ in range(256)]
            for i in random.sample(range(256), random.randint(1, 4)):
                x[i] = random.gauss(0, 0.3)
            w = [abs(random.gauss(0, 1)) ** 2 + 0.05 for _ in range(256)]
            ea, el = wssd(x, w, amax_scale(x, Q), Q), wssd(x, w, ls_scale(x, w, Q), Q)
            rs.append(el / ea if ea > 0 else 1.0)
        print(f"  {name:>7}: LS cuts weighted-SSD {100*(1-statistics.mean(rs)):.1f}% "
              f"(median {100*(1-statistics.median(rs)):.1f}%)")


def real(gguf, imatrix, gguf_py):
    import numpy as np
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    d = open(imatrix, "rb").read(); n = struct.unpack_from("<i", d, 0)[0]; off = 4; cov = {}
    for _ in range(n):
        ln = struct.unpack_from("<i", d, off)[0]; off += 4
        nm = d[off:off+ln].decode("utf-8", "replace"); off += ln
        nc = struct.unpack_from("<i", d, off)[0]; off += 4
        nv = struct.unpack_from("<i", d, off)[0]; off += 4
        cov[nm] = np.frombuffer(d[off:off+4*nv], dtype=np.float32) / max(nc, 1); off += 4*nv
    r = GGUFReader(gguf)
    for t in r.tensors:
        if not t.name.endswith(".weight") or len(t.shape) != 2:
            continue
        key = t.name if t.name in cov else (t.name[:-7] if t.name[:-7] in cov else None)
        if key is None:
            continue
        W = np.array(t.data, dtype=np.float32)
        if W.shape[1] != cov[key].shape[0]:
            continue
        imp = cov[key]
        print(f"real tensor {t.name}  W{list(W.shape)}")
        for name, Q in [("ternary", 1), ("2-bit", 2), ("3-bit", 4)]:
            ea = el = 0.0
            for c0 in range(0, W.shape[1] - W.shape[1] % 256, 256):
                x = W[:, c0:c0+256]; ww = imp[c0:c0+256][None, :]
                da = (np.abs(x).max(1, keepdims=True) / Q).clip(1e-9)
                dl = da.copy()
                for _ in range(5):
                    u = np.clip(np.round(x / dl), -Q, Q)
                    num = (ww*u*x).sum(1, keepdims=True); den = (ww*u*u).sum(1, keepdims=True)
                    dl = np.where(den > 0, num/np.maximum(den, 1e-12), dl)
                ea += (ww*(x-da*np.clip(np.round(x/da), -Q, Q))**2).sum()
                el += (ww*(x-dl*np.clip(np.round(x/dl), -Q, Q))**2).sum()
            print(f"  {name:>7}: LS cuts weighted-SSD {100*(1-el/ea):.1f}%")
        return
    print("no covered 2D tensor matched the imatrix")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf"); ap.add_argument("--imatrix")
    ap.add_argument("--gguf-py", default=r"C:\pollard\ik_llama.cpp\gguf-py")
    a = ap.parse_args()
    if a.gguf and a.imatrix:
        real(a.gguf, a.imatrix, a.gguf_py)
    else:
        synthetic()
