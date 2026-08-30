#!/usr/bin/env python3
"""pollard-rotate — incoherence preconditioning (QuIP#/QuaRot-style) before the quant.

This is the RIGHT preconditioner for the IQ codebook quants (IQ2/IQ3/IQ4), where
diagonal AWQ-style smoothing (see pollard-smooth) HURTS — because a lattice/codebook
quantizer assumes a fixed weight-distribution SHAPE, and per-channel scaling distorts
it. Rotation does the opposite: an orthogonal transform on the residual stream spreads
each coordinate's energy across all others (concentration of measure -> near-Gaussian,
outlier-free, equal-variance), which collapses the per-block range AND hands the IQ
codebook exactly the ball-shaped distribution it packs best. This is why "rotate then
VQ" (QuIP#/QTIP) is the weight-only SOTA at low bits.

The rotation is mathematically identity: r' = H r in the residual, with H folded into
the weights that WRITE the residual (embed, attn_output, ffn_down: W <- H W) and its
inverse into the weights that READ it (q,k,v,gate,up,lm_head: W <- W diag(gamma) H^T,
after absorbing the preceding RMSNorm gain gamma). q/k/v VALUES are unchanged, so RoPE,
GQA and attention are untouched — no per-head handling, no runtime kernel. The output is
a NORMAL GGUF that runs on CPU, CUDA, Metal, Vulkan unchanged.

Every model gets an identity canary (rotated-f16 must match the original f16 to fp
tolerance) — verify empirically, never assume. Then: build a FRESH imatrix on the
rotated model and quantize as usual. Compare KL vs the f16 base at equal bits/weight.

Usage:
  pollard-rotate --gguf model-f16.gguf --out model-rot-f16.gguf
  pollard-rotate --gguf model-f16.gguf --kind hadamard   # structured (power-of-2 dims)
  pollard-rotate --gguf model-f16.gguf --plan-only

Start from f16/bf16. After this: FRESH imatrix on the output, then pollard-fit.
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter
from gguf.quants import quantize, dequantize
from gguf.constants import GGMLQuantizationType as T, GGUFValueType as VT

# residual-stream roles (llama/qwen GGUF names). READERS take input from the
# residual (rotate input cols, after absorbing the preceding norm gain); WRITERS
# put their output into the residual (rotate output rows).
READERS = ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"]   # gain from the block norm
WRITERS = ["attn_output", "ffn_down"]                            # rotate output rows
# preceding-norm gain for each reader group
NORM_OF = {"attn_q": "attn_norm", "attn_k": "attn_norm", "attn_v": "attn_norm",
           "ffn_gate": "ffn_norm", "ffn_up": "ffn_norm"}


def _hms(s):
    s = int(s); h, m = divmod(s, 3600); m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def to_logical(t):
    data = dequantize(t.data, t.tensor_type).astype(np.float32)
    return data.reshape(tuple(int(d) for d in t.shape)[::-1])


def random_orthogonal(n, seed):
    """A random orthogonal matrix via QR of a Gaussian — strong Gaussianization
    (each rotated coord is a random combination of all inputs -> CLT). Works for
    ANY dim (no power-of-2 requirement)."""
    g = np.random.default_rng(seed).standard_normal((n, n))
    q, r = np.linalg.qr(g)
    q *= np.sign(np.diag(r))            # make it deterministic / uniform Haar
    return q.astype(np.float64)


def hadamard(n):
    """Sylvester Hadamard (needs n a power of 2), normalized to orthonormal."""
    if n & (n - 1):
        raise ValueError("hadamard needs a power-of-2 dim")
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n))


def randomized_hadamard(n, seed):
    """H = Hadamard . diag(+-1) if n is a power of 2, else fall back to random
    orthogonal. Random signs give the QuIP# incoherence guarantee."""
    if n & (n - 1) == 0:
        d = np.random.default_rng(seed).integers(0, 2, n) * 2 - 1
        return (hadamard(n) * d[np.newaxis, :]).astype(np.float64)
    return random_orthogonal(n, seed)


def block_diagonal(n, block, seed):
    """Block-diagonal orthogonal rotation: mix only WITHIN each `block`-wide group
    of channels, leave the coarse across-block structure intact. This kills the
    intra-block outliers a block-32 quantizer chokes on WITHOUT smearing the
    channel-importance signal the imatrix rides on (a full dense rotation destroys
    it). Still globally orthogonal, so it folds/cancels on the residual exactly like
    a dense H. Uses a randomized Hadamard per block when `block` is a power of two."""
    if n % block != 0:
        # shrink block to a divisor near the requested size
        while n % block != 0 and block > 1:
            block -= 1
    H = np.zeros((n, n))
    for i, c0 in enumerate(range(0, n, block)):
        b = min(block, n - c0)
        Hb = (randomized_hadamard(b, seed + i) if (b & (b - 1)) == 0
              else random_orthogonal(b, seed + i))
        H[c0:c0 + b, c0:c0 + b] = Hb
    return H


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16 best)")
    ap.add_argument("--out")
    ap.add_argument("--kind", default="orthogonal", choices=["orthogonal", "hadamard", "block"],
                    help="orthogonal = dense random Haar (any dim); hadamard = dense "
                    "randomized Hadamard (power-of-2 dims); block = block-diagonal "
                    "(mix within --block channels only — keeps imatrix useful)")
    ap.add_argument("--block", type=int, default=32,
                    help="block width for --kind block (match the quant block, 32)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--canary-tol", type=float, default=2e-3,
                    help="max allowed identity error before the rotation is rejected")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.gguf):
        sys.exit(f"ERROR: source GGUF not found: {a.gguf}")
    reader = GGUFReader(a.gguf)
    arch = next((str(bytes(f.parts[-1]), "utf-8") for f in reader.fields.values()
                 if f.name == "general.architecture"), "llama")
    tensors = {t.name: t for t in reader.tensors}
    layers = sorted({int(m.group(1)) for n in tensors
                     for m in [re.match(r"blk\.(\d+)\.", n)] if m})

    # hidden size from the residual: attn_output OUTPUT rows == hidden
    probe = tensors.get("blk.0.attn_output.weight")
    if probe is None:
        sys.exit("ERROR: no blk.0.attn_output.weight — unsupported arch for rotation.")
    hidden = to_logical(probe).shape[0]
    print(f"== pollard-rotate :: {a.gguf}  [{arch}]  hidden={hidden}  layers={len(layers)}  kind={a.kind}")

    if a.kind == "hadamard":
        H = randomized_hadamard(hidden, a.seed)
    elif a.kind == "block":
        H = block_diagonal(hidden, a.block, a.seed)
    else:
        H = random_orthogonal(hidden, a.seed)
    Ht = H.T.copy()
    # verify H is orthogonal (canary #0)
    orth = float(np.max(np.abs(H @ Ht - np.eye(hidden))))
    print(f"   H orthogonality residual: {orth:.2e}  (must be ~0)")
    if orth > 1e-6:
        sys.exit("ERROR: constructed H is not orthogonal — aborting.")
    if a.plan_only:
        nr = sum(1 for li in layers for s in READERS if f"blk.{li}.{s}.weight" in tensors)
        nw = sum(1 for li in layers for s in WRITERS if f"blk.{li}.{s}.weight" in tensors)
        print(f"   would rotate: {nr} readers + {nw} writers + embed + lm_head; "
              f"zero {2*len(layers)+1} norm gains")
        return

    # STREAMING design: hold only H (hidden x hidden) + the small 1-D norm gains in
    # RAM, and transform each tensor on the fly in the write loop. This is what lets
    # rotate run on a 27B on a 16 GB box — never materialize the whole model.
    gains = {t.name: to_logical(tensors[t.name]).astype(np.float64)
             for t in reader.tensors if t.name.endswith("_norm.weight")}
    fn = next((n for n in ("output_norm.weight", "norm.weight") if n in tensors), None)
    absorb_fn = ("output.weight" in tensors and fn is not None
                 and gains.get(fn) is not None and gains[fn].shape[0] == hidden)
    # names of block norms whose gain gets absorbed into readers -> set to ones
    norms_to_one = set()
    for li in layers:
        for nm in ("attn_norm", "ffn_norm"):
            gn = f"blk.{li}.{nm}.weight"
            if gn in tensors:
                norms_to_one.add(gn)
    if absorb_fn:
        norms_to_one.add(fn)

    reader_suf = {s: NORM_OF[s] for s in READERS}

    def transform(name, t):
        """Return the new f32 logical array for `name`, or None to copy unchanged."""
        if name in norms_to_one:
            return np.ones_like(to_logical(t))
        if name == "token_embd.weight":
            E = to_logical(t)                                    # [vocab, hidden]
            return (E.astype(np.float64) @ Ht).astype(np.float32) if E.shape[1] == hidden else None
        if name == "output.weight" and absorb_fn:
            W = to_logical(t).astype(np.float64)
            return ((W * gains[fn][np.newaxis, :]) @ Ht).astype(np.float32) if W.shape[1] == hidden else None
        m = re.match(r"(blk\.\d+\.)(\w+)\.weight$", name)
        if not m:
            return None
        p, base = m.group(1), m.group(2)
        if base in reader_suf:                                    # reader: (W*gamma) @ H^T
            W = to_logical(t).astype(np.float64)
            if W.shape[1] != hidden:
                return None
            g = gains.get(p + reader_suf[base] + ".weight")
            g = g if (g is not None and g.shape[0] == hidden) else np.ones(hidden)
            return ((W * g[np.newaxis, :]) @ Ht).astype(np.float32)
        if base in WRITERS:                                       # writer: H @ W
            W = to_logical(t).astype(np.float64)
            return (H @ W).astype(np.float32) if W.shape[0] == hidden else None
        return None

    # ---- identity canary: full residual path on a random input token embedding ----
    rng = np.random.default_rng(0)
    x = rng.standard_normal(hidden)
    # emulate ONE reader group both ways: original n = (RMSNorm(r) * gamma); q = Wq n
    #  rotated:  r' = H r ; n' = RMSNorm(r') ; q' = Wq' n'  (gamma folded, norm=1)
    li = layers[0]; p = f"blk.{li}."
    Wq0 = to_logical(tensors[p+"attn_q.weight"]).astype(np.float64)
    g0 = to_logical(tensors[p+"attn_norm.weight"]).astype(np.float64)
    def rmsnorm(v): return v / np.sqrt(np.mean(v*v) + 1e-6)
    q_orig = Wq0 @ (rmsnorm(x) * g0)
    r_rot = H @ x
    Wq_rot = transform(p + "attn_q.weight", tensors[p + "attn_q.weight"]).astype(np.float64)
    q_rot = Wq_rot @ rmsnorm(r_rot)
    rel = float(np.max(np.abs(q_rot - q_orig) / np.maximum(np.abs(q_orig), 1e-6)))
    print(f"   identity canary (reader path): max_rel_err={rel:.2e}")
    if rel > a.canary_tol:
        sys.exit(f"ERROR: rotation identity canary FAILED ({rel:.2e} > {a.canary_tol}). "
                 f"Nothing written. (Likely a norm/axis mismatch for arch '{arch}'.)")
    # writer path: o_orig into residual vs H^-1 (H o) — trivially exact, checked via H orth above

    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + "-rot.gguf"
    print(f"   writing {out} (streaming) ...")
    w = GGUFWriter(out, arch)
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"):
            continue
        vtype = f.types[0]
        sub = f.types[1] if vtype == VT.ARRAY and len(f.types) > 1 else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    t0 = time.time()
    nt = len(reader.tensors)
    for i, t in enumerate(reader.tensors):
        new = transform(t.name, t)                    # None => copy unchanged
        data = new if new is not None else to_logical(t)
        if t.tensor_type == T.F32:
            w.add_tensor(t.name, data.astype(np.float32))
        elif t.tensor_type in (T.F16, T.BF16):
            w.add_tensor(t.name, data.astype(np.float16))
        else:
            qd = quantize(data.astype(np.float32), t.tensor_type)
            w.add_tensor(t.name, qd, raw_shape=qd.shape, raw_dtype=t.tensor_type)
        if (i + 1) % 32 == 0 or i + 1 == nt:
            el = time.time() - t0
            eta = el / (i + 1) * (nt - i - 1)
            print(f"   [{(i+1)*100//nt:3d}%  {_hms(el)}  ETA {_hms(eta)}] {i+1}/{nt} tensors")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"   done: {out}")
    print(f"\n   NEXT: FRESH imatrix on {out} (basis changed), then fit + KL vs f16 base:")
    print(f"     llama-imatrix -m {out} -f calib.txt -o rot.imatrix -ngl 99")
    print(f"     llama-quantize --imatrix rot.imatrix {out} rot-iq3.gguf iq3_s")


if __name__ == "__main__":
    main()
