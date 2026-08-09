#!/usr/bin/env python3
"""E11 — reproduce (or refute) the sign-asymmetry finding on our own quants.

The claim (from the Moet writeup): naive low-bit quantization destroys MoE
experts not because the error is large, but because the codebook is
sign-ASYMMETRIC — the optimal-L2 reconstruction drops one sign's tail, so each
expert acquires a small directional bias, and the bias compounds over dozens of
layers. If true, the fix is a sign-symmetric codebook at the SAME error.

This does not need their kernels. It needs only: dequantize our own Q2_K expert
tensors, compare against the f16 originals, and measure whether the residual
(quant - original) has a systematic sign bias per expert — and whether that
bias is worse in the deep layers our KL tail-spike implicated.

Usage:
  e11_sign_asymmetry.py --f16 granite-f16.gguf --q2k granite-q2k.gguf
"""
import argparse
import sys

import numpy as np
import gguf
from gguf.quants import dequantize


def load_expert_tensors(path):
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        if "ffn_" in t.name and "_exps" in t.name:
            out[t.name] = t
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--f16", required=True)
    ap.add_argument("--q2k", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N expert tensors (0=all)")
    a = ap.parse_args()

    f16 = load_expert_tensors(a.f16)
    q2k = load_expert_tensors(a.q2k)
    names = sorted(set(f16) & set(q2k),
                   key=lambda n: int(n.split(".")[1]))     # by layer
    if a.limit:
        names = names[:a.limit]
    if not names:
        sys.exit("no shared expert tensors found")

    print(f"comparing {len(names)} expert tensors\n")
    print(f"{'tensor':32s} {'rel_rms_err':>11s} {'mean_resid':>11s} "
          f"{'%neg_resid':>10s}  sign_bias")
    rows = []
    for n in names:
        orig = dequantize(f16[n].data, f16[n].tensor_type).astype(np.float64)
        quant = dequantize(q2k[n].data, q2k[n].tensor_type).astype(np.float64)
        if orig.shape != quant.shape:
            continue
        resid = quant - orig                                # reconstruction error
        rel_rms = np.sqrt((resid**2).mean()) / (np.sqrt((orig**2).mean()) + 1e-12)
        mean_resid = resid.mean()                           # 0 if symmetric
        pct_neg = (resid < 0).mean() * 100
        # sign bias: how far the residual's mean is from zero, in units of its
        # own spread. >0 systematically = asymmetric codebook (the Moet claim).
        sign_bias = mean_resid / (resid.std() + 1e-12)
        layer = int(n.split(".")[1])
        rows.append((layer, rel_rms, mean_resid, pct_neg, sign_bias))
        print(f"{n:32s} {rel_rms:11.4f} {mean_resid:11.3e} "
              f"{pct_neg:9.1f}% {sign_bias:+.4f}")

    arr = np.array([(r[0], r[1], abs(r[4])) for r in rows])
    print("\n--- verdict ---")
    print(f"mean |sign_bias| across experts : {arr[:,2].mean():.4f}")
    print(f"mean rel_rms error              : {arr[:,1].mean():.4f}")
    # depth correlation: does sign bias worsen with layer depth?
    if len(arr) > 3:
        c = np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]
        print(f"corr(depth, |sign_bias|)        : {c:+.3f}")
    frac_neg_dom = np.mean([r[3] > 50 for r in rows]) * 100
    print(f"experts with >50% negative residual: {frac_neg_dom:.0f}%  "
          f"(≈50% = symmetric; far from 50% = the asymmetry Moet describes)")
    print("\nif |sign_bias| is systematically non-zero and one sign dominates,")
    print("the finding reproduces -> a sign-symmetric codebook at equal rel_rms")
    print("is the fix, and it can be validated the same way before any kernel.")


if __name__ == "__main__":
    main()
