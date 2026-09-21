#!/usr/bin/env python3
"""pollard-trellis -- Pollard's own trellis-coded quantizer. No fork required.

WHY THIS EXISTS

Trellis quantization is the strongest weight-only method below 3 bits, and today the only way to
run it is ik_llama's fork. That costs Pollard twice: a trellis build loads in one runtime instead
of all of them, and the method is unavailable on the GPTQ / MLX / MX / EXL3 lanes entirely.

PROVENANCE -- read this before editing

This is written from the published DESCRIPTION of trellis coded quantization with a computed
codebook (QTIP, arXiv:2406.11235). It is NOT a translation of any existing implementation:

  * Cornell-RelaxML/qtip is GPL-3. Copying from it would force GPL-3 on everything here, and
    on anything that ever vendored it.
  * ik_llama.cpp is MIT, so its code COULD be attributed and reused -- but the attempt to carry
    those types into mainline llama.cpp (PR #19726) was closed over exactly that provenance
    question, and the repository was locked. A port derived from that source walks into the same
    wall.

Algorithms are not copyrightable; implementations are. So this is built from the paper's
description of the method, which leaves it under Pollard's own licence, usable on every lane, and
submittable upstream without the argument that closed the last attempt.

HOW IT WORKS

  1. A trellis state is an L-bit sliding window. Emitting K bits shifts them in:
         next = ((state << K) | symbol) & (2^L - 1)
     so each state has 2^K successors and the whole structure is implied by a shift. Nothing is
     stored, which is the property that makes decoding fast.

  2. Each state maps to a reconstruction value through a COMPUTED codebook -- a multiply-add and
     a bit extraction that turns the state into an approximately Gaussian sample. No lookup table,
     so the codebook costs no memory and scales to any state width.

  3. Viterbi finds the minimum-distortion path through the trellis for a row of weights. Because
     consecutive weights share overlapping states, the code spends fractional bits per weight on
     shape rather than integer bits per weight on magnitude, which is where the gain over a
     scalar quantizer comes from.

  4. Distortion is Hessian-weighted when a diagonal is supplied, so this minimises the same
     objective pollard-gptq does rather than raw MSE.

Incoherence processing (the Hadamard rotation trellis codes assume) is already a Pollard tool --
pollard-rotate -- so it is not duplicated here. Rotate first, then quantize.

    python -m pollard_trellis --self-test
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# Computed-codebook constants. The multiplier is a large odd 32-bit constant so the low bits of
# the product are well mixed; the addend breaks the fixed point at state 0. Any such pair works --
# what matters is that the map is deterministic, cheap, and produces a spread that is close to
# Gaussian after the transform below.
_MUL = np.uint32(0x9E3779B1)        # golden-ratio derived, the usual choice for a cheap mixer
_ADD = np.uint32(0x85EBCA6B)


def codebook(states, scale=1.0):
    """State -> reconstruction value, computed rather than stored.

    One multiply-add mixes the state, then two 16-bit halves are summed. Summing two independent
    uniforms gives a triangular distribution; that is already much closer to Gaussian than one
    uniform, and it costs a shift and an add. This is the cheap end of the paper's computed
    codebook family, chosen because it has to run per-weight on a CPU decode path.
    """
    s = np.asarray(states, dtype=np.uint64).astype(np.uint32)
    x = (s * _MUL + _ADD).astype(np.uint32)
    hi = (x >> np.uint32(16)).astype(np.float64)
    lo = (x & np.uint32(0xFFFF)).astype(np.float64)
    # two uniforms on [0,1) summed -> triangular on [0,2), centred and scaled to unit variance
    u = (hi + lo) / 65536.0 - 1.0
    return (u * np.sqrt(6.0) * scale).astype(np.float64)


def quantize_row(w, L=12, K=2, hdiag=None, scale=None):
    """Viterbi over the bitshift trellis for one row of weights.

    Returns (reconstruction, symbols, final_state). `symbols` is K bits per weight, so the rate is
    K bits per weight plus L bits once for the initial state.

    Hessian weighting: the paper minimises proxy loss, and with a diagonal Hessian that is
    sum_i h_i (w_i - q_i)^2. Passing hdiag makes a channel the model actually uses cost more to
    get wrong than one it does not.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    n = w.size
    if n == 0:
        return w.copy(), np.zeros(0, dtype=np.int64), 0
    n_states = 1 << L
    n_sym = 1 << K
    mask = np.uint64(n_states - 1)

    if scale is None:
        scale = float(np.sqrt(np.mean(w * w))) or 1.0
    h = np.ones(n) if hdiag is None else np.asarray(hdiag, dtype=np.float64).ravel()
    if h.size != n:
        raise ValueError(f"hdiag has {h.size} entries for {n} weights")

    all_states = np.arange(n_states, dtype=np.uint64)
    # successor table: from every state, the K-bit shift gives 2^K next states
    succ = np.empty((n_states, n_sym), dtype=np.int64)
    for j in range(n_sym):
        succ[:, j] = (((all_states << np.uint64(K)) | np.uint64(j)) & mask).astype(np.int64)
    # every state's reconstruction value, computed once
    values = codebook(all_states, scale)

    INF = np.float64(1e30)
    cost = np.zeros(n_states, dtype=np.float64)      # uniform start: any initial state allowed
    back = np.empty((n, n_states), dtype=np.uint16 if L <= 16 else np.uint32)

    for t in range(n):
        d = h[t] * (values - w[t]) ** 2              # cost of LANDING in each state at step t
        nxt = np.full(n_states, INF, dtype=np.float64)
        bk = np.zeros(n_states, dtype=back.dtype)
        for j in range(n_sym):
            tgt = succ[:, j]
            cand = cost + d[tgt]
            # keep the cheapest predecessor for each target state
            np.minimum.at(nxt, tgt, cand)
            better = cand <= nxt[tgt]
            bk[tgt[better]] = np.arange(n_states, dtype=back.dtype)[better]
        cost, back[t] = nxt, bk

    state = int(np.argmin(cost))
    symbols = np.zeros(n, dtype=np.int64)
    path = np.zeros(n, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        path[t] = state
        prev = int(back[t][state])
        symbols[t] = state & (n_sym - 1)
        state = prev
    return values[path], symbols, int(path[0])


def bits_per_weight(n, L=12, K=2):
    """K bits per weight plus the initial state, amortised over the row."""
    return K + L / max(n, 1)


def quantize(W, L=12, K=2, H=None, rows=None):
    """Quantize a weight matrix row by row. H is an optional Hessian (its diagonal is used)."""
    W = np.asarray(W, dtype=np.float64)
    hdiag = None
    if H is not None:
        H = np.asarray(H, dtype=np.float64)
        hdiag = np.diag(H) if H.ndim == 2 else H.ravel()
    out = np.empty_like(W)
    syms = []
    for i in range(W.shape[0] if rows is None else min(rows, W.shape[0])):
        q, s, _ = quantize_row(W[i], L=L, K=K, hdiag=hdiag)
        out[i] = q
        syms.append(s)
    return out, syms


def _self_test():
    rng = np.random.default_rng(0)
    print("pollard-trellis self-test")
    print(f"{'rows x cols':>14} {'K':>3} {'L':>4} {'bpw':>6} {'rel.err':>9} {'vs RTN':>9}")
    ok = True
    for rows, cols in ((4, 256), (8, 512)):
        W = rng.standard_normal((rows, cols)) * 0.02
        for K in (2, 3):
            Q, _ = quantize(W, L=10, K=K)
            err = np.linalg.norm(W - Q) / np.linalg.norm(W)
            # round-to-nearest at the same rate, as the thing to beat
            lo, hi = W.min(), W.max()
            levels = 1 << K
            step = (hi - lo) / (levels - 1)
            R = np.round((W - lo) / step) * step + lo
            rtn = np.linalg.norm(W - R) / np.linalg.norm(W)
            bpw = bits_per_weight(cols, 10, K)
            flag = "" if err < rtn else "   <-- NOT beating RTN"
            ok = ok and err < rtn
            print(f"{rows:6d} x {cols:<5d} {K:3d} {10:4d} {bpw:6.2f} {err:9.4f} {rtn:9.4f}{flag}")
    print("\nPASS" if ok else "\nFAIL -- the trellis should beat round-to-nearest at equal rate")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="check the quantizer beats round-to-nearest at the same bit rate")
    ap.add_argument("--bits", type=int, default=2, help="K, bits per weight")
    ap.add_argument("--state-bits", type=int, default=12, help="L, trellis state width")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(_self_test())
    ap.print_help()


if __name__ == "__main__":
    main()
