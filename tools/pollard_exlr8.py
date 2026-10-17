#!/usr/bin/env python3
"""pollard-exlr8 -- the EXLR8 lane: encode every tensor ONCE at several trellis widths, then each rung is a
manifest that picks a width per tensor under a byte budget, chosen from Pollard's MEASURED sensitivity.

    pollard-exlr8 --model ./Qwen2.5-0.5B-Instruct --calib calib.txt --sensitivity s.json --bpw 3.25 --out ./R
    pollard-exlr8 --model <hf dir> --calib calib.txt --sensitivity s.json --ladder 2.5,3,3.25,4,5,6
    pollard-exlr8 --model <hf dir> --encode-only --store ./store            # pay the encode, choose rungs later
    pollard-exlr8 --decode ./M-Pollard-EXLR8-3.25bpw --out ./M-decoded     # bf16 HF checkpoint (reference loader)
    pollard-exlr8 --inspect ./M-Pollard-EXLR8-3.25bpw                       # widths, bytes, predicted KL
    pollard-exlr8 --model <hf dir> --ladder 2.5,3,4 --plan-only             # print the plan, build nothing

WHAT EXLR8 IS (and where the format comes from)

EXLR8 (kindlingai/exlr8) stores a model's matrices as exllamav3-style trellis codes whose codebook values are rounded
to FP8 E4M3, at SEVERAL widths K, and ships manifests that pick a width per unit -- one download, many sizes. GR8
(kindlingai/gr8, MIT) is the CUDA engine that serves it. Licences, re-checked 2026-10-15:
  * kindlingai/gr8: MIT (repo-wide LICENSE).
  * kindlingai/exlr8: NO repository licence. Its loader/ directory alone carries an MIT LICENSE (Turboderp's
    exllamav3 notice) -- loader/exlr8.py is the reference decoder; FORMAT.md / exlr8writer/ are unlicensed.
This module is written from the published format description (FORMAT.md's tables: record layout, tile order, bit
packing, the mcg codebook + E4M3 rounding, the 128-block Hadamard). No exlr8writer code is used. The decode
primitives it restates (codebook, bit packing, tile order, Hadamard) are exllamav3's (MIT, Turboderp), as the format
itself says; the encoder (Viterbi + LDLQ) is Pollard's own, written from the trellis-coding literature.

ON DISK (EXLR8 v2 dense records -- what this lane writes)

  dense/exlr8-dense-LLL.safetensors   layer L: per matrix M, `M.record` I64[8] (format, out, in, K, side, 128, 0,
                                      solve) + `M.checksum` U64 + `M.trellis` U8[entries, bytes] + `M.suh`/`M.svh`
                                      F16; vectors (norms, biases) as stored
  dense/exlr8-vocab.safetensors       embed_tokens / lm_head (FP8 per row) + the final norm
  dense/exlr8-extra-model.safetensors anything outside the decoder layers, full HF names
  manifests/exlr8-manifest-NAME.json  this rung's width per tensor + how it was chosen
  pollard-exlr8.json                  the rung marker (lane, bpw, widths, predicted KL)
  config.json / tokenizer files       copied from the source
All safetensors: data at 65,536, tensor offsets multiples of 65,536 (`__pad.N` fills), metadata format=exlr8-v2.
Decode of a trellis record: W(in, out) = diag(suh) Hb T Hb diag(svh), T from 16x16 tiles of K-bit tail-biting
trellis states through the mcg codebook rounded to E4M3; HF weight = W^T.

THE POLLARD WAY (why this lane is not just a converter)

  1. ONE encode per tensor per width (default K = 2,3,4,5,6): sign-flip + 128-block Hadamard (incoherence), per
     output-channel scale, then LDLQ (Hessian error feedback, 16-row strips) with a tail-biting Viterbi per 16x16
     tile against the E4M3-rounded codebook. Hessians come from the shared calibration corpus (pollard-calib,
     tool-aware: agent traces rendered through the model's own template) -- ws.calib_samples().
  2. For every tensor and width the store records the MEASURED output error e(t, K) = tr(dW H dW^T)/tr(W H W^T),
     and the same quantity for the probe's RTN quantizer (rtn4), so Pollard's measured layer sensitivity S
     (pollard-probe: KL when a layer group is RTN-quantized) converts into a per-tensor KL estimate
     KL(t, K) = S_t * e(t, K) / e_rtn(t).
  3. A rung = an EXACT multiple-choice knapsack (DP over the byte budget) minimising the summed KL estimate subject
     to the trellis bytes <= bpw * numel / 8. Every rung of the ladder comes out of the same encode in seconds.
  4. Every rung is gated by pollard-verify (KLD + top-1 vs the bf16 original, through the reference decode) and
     tool-gated by pollard-toolcall -- the estimate chooses, the measurement judges.

RUNNING A RUNG -- read this before promising a runtime

GR8 today serves ONE model: GLM-5.3 (GlmMoeDsaForCausalLM) on 3/4/6 DGX Spark (GB10, sm_121a) boxes, with an
OpenAI-compatible API on the head node (`./gr8 glm53 up ...`; docs/MODEL-SPECIFIC.md lists what is hard-wired).
It does not load other architectures, and it does not run on a desktop RTX card. So a Pollard EXLR8 rung of any
other model runs through the REFERENCE decode here: `--decode` writes a bf16 HF checkpoint (bit-exact to the stored
codes) that transformers / vLLM load as-is; pollard-verify and pollard-toolcall measure exactly those weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import time
import zlib

import numpy as np

# ------------------------------------------------------------------------------------------------ constants

MCG = 0xCBAC1FED
TRELLIS, INT4, MXFP8, FP8_BLOCK, FP8_ROW, BF16, FP32 = 1, 2, 3, 4, 5, 6, 7     # record field 0
SIDE_REPLICATED, SIDE_OUT, SIDE_IN = 0, 1, 2                                    # record field 4
DATA = {TRELLIS: ("trellis", "suh", "svh"), INT4: ("q4", "s4"), MXFP8: ("fp8", "e8m0"),
        FP8_BLOCK: ("fp8", "scale"), FP8_ROW: ("fp8", "scale"), BF16: ("bf16",), FP32: ("fp32",)}
ALIGN = 65536                       # EXLR8: data start and every tensor offset
DEFAULT_WIDTHS = (2, 3, 4, 5, 6)    # GR8's kernels decode K2-K4 today; K5/K6 decode with the reference loader
DEFAULT_LADDER = (2.5, 3.0, 3.25, 4.0, 5.0, 6.0)
STORE_FORMAT = "pollard-exlr8-store"
MANIFEST_FORMAT = "pollard-exlr8-manifest"
MARKER = "pollard-exlr8.json"
#: matrices whose input side is the sharded side (row-parallel in TP), EXLR8's convention for o_proj / down
_IN_SIDE = re.compile(r"(o_proj|down_proj|out_proj|dense_4h_to_h|fc2|w2|wo|c_proj)$")
_COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja",
               "chat_template.json", "added_tokens.json", "preprocessor_config.json", "LICENSE")

# ------------------------------------------------------------------------------------------------ E4M3 + codebook


def e4m3_round(x):
    """float -> nearest FP8 E4M3 (fn: no inf, max 448) value, round-half-even, saturating. numpy, exact."""
    x = np.clip(np.asarray(x, dtype=np.float64), -448.0, 448.0)
    a = np.abs(x)
    _, e = np.frexp(a)                              # a = m * 2^e, m in [0.5, 1)
    e = np.maximum(e - 1, -6)                       # floor(log2 a), subnormals share the -6 binade
    step = np.ldexp(1.0, (e - 3).astype(np.int64))  # 3 mantissa bits
    q = np.rint(a / step) * step
    return (np.sign(x) * np.minimum(q, 448.0)).astype(np.float32)


_E4M3_POS = None


def _e4m3_table():
    """values of codes 0x00..0x7E (non-negative, increasing)."""
    global _E4M3_POS
    if _E4M3_POS is None:
        c = np.arange(127)
        e, m = c >> 3, c & 7
        _E4M3_POS = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7)).astype(np.float64)
    return _E4M3_POS


def e4m3_encode(x):
    """float32 -> uint8 E4M3 codes (rounded first)."""
    q = e4m3_round(x).astype(np.float64)
    code = np.searchsorted(_e4m3_table(), np.abs(q)).astype(np.uint8)
    return np.where(np.signbit(q), code | 0x80, code).astype(np.uint8)


def e4m3_decode(b):
    b = np.asarray(b, dtype=np.uint8)
    v = _e4m3_table()[np.minimum(b & 0x7F, 126)].astype(np.float32)
    return np.where(b & 0x80, -v, v)


def codebook(states):
    """16-bit trellis states -> fp32 weights: the mcg codebook (two fp16 halves summed in fp16), then E4M3
    rounding. The rounding is part of the format: the encoder searches the ROUNDED values."""
    s = np.asarray(states, dtype=np.uint64) & np.uint64(0xFFFF)
    x = (s * np.uint64(MCG)) & np.uint64(0xFFFFFFFF)
    x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    lo = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (x >> np.uint64(16)).astype(np.uint16).view(np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        v = (lo + hi).astype(np.float32)            # fp16 + fp16 in fp16, as the reference does
    return e4m3_round(v)


_CB = None


def codebook_table():
    global _CB
    if _CB is None:
        _CB = codebook(np.arange(65536))
    return _CB


def codebook_rms():
    cb = codebook_table().astype(np.float64)
    return float(np.sqrt(np.mean(cb * cb)))


def tile_perm():
    """perm[j] = row*16 + col of the tile element at stream position j (exllamav3's tensor-core order)."""
    perm = []
    for t in range(32):
        r0, c0 = 2 * (t % 4), t // 4
        for r, c in ((r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                     (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)):
            perm.append(16 * r + c)
    return np.array(perm, dtype=np.int64)


_PERM = tile_perm()

# ------------------------------------------------------------------------------------------------ bit packing


def unpack_values(raw, K):
    """Packed tiles uint8 [n, 32K] -> the K-bit values [n, 256]. int16 words little-endian, the two words of each
    pair swapped, bits MSB first."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1, 32 * K)
    n = raw.shape[0]
    words = raw.view("<u2").reshape(n, 8 * K, 2)[:, :, ::-1].reshape(n, 16 * K)
    bits = np.unpackbits(words.astype(">u2").view(np.uint8).reshape(n, 32 * K), axis=1)
    w = (1 << np.arange(K - 1, -1, -1)).astype(np.int64)
    return bits.reshape(n, 256, K).astype(np.int64) @ w


def pack_values(vals, K):
    """The inverse of unpack_values: K-bit values [n, 256] -> uint8 [n, 32K]."""
    vals = np.asarray(vals, dtype=np.int64).reshape(-1, 256)
    n = vals.shape[0]
    if vals.size and (vals.min() < 0 or vals.max() >= (1 << K)):
        raise ValueError(f"a value exceeds {K} bits")
    bits = ((vals[:, :, None] >> np.arange(K - 1, -1, -1)) & 1).astype(np.uint8).reshape(n, 256 * K)
    words = np.packbits(bits, axis=1).view(">u2").reshape(n, 8 * K, 2)[:, :, ::-1].reshape(n, 16 * K)
    return np.ascontiguousarray(words.astype("<u2")).view(np.uint8).reshape(n, 32 * K)


def states_from_values(vals, K):
    """Tail-biting: state i = the 16 bits ending with value i (wrapping)."""
    vals = np.asarray(vals, dtype=np.int64)
    s = np.zeros_like(vals)
    for j in range(-(-16 // K)):
        s |= np.roll(vals, j, axis=1) << (j * K)
    return s & 0xFFFF


def decode_tiles(raw, K):
    """Packed tiles uint8 [n, 32K] -> fp32 [n, 16, 16], (input, output) within each tile."""
    st = states_from_values(unpack_values(raw, K), K)
    n = st.shape[0]
    out = np.empty((n, 256), dtype=np.float32)
    out[:, _PERM] = codebook_table()[st]
    return out.reshape(n, 16, 16)


def tiles_to_stream(tiles):
    """[n, 16, 16] tile matrices -> [n, 256] in stream order (the inverse of decode_tiles' placement)."""
    t = np.asarray(tiles).reshape(-1, 256)
    return t[:, _PERM]


# ------------------------------------------------------------------------------------------------ tile layout
#  grid[k16, c16] = one 16x16 tile of T (in, out). Output-side entries (side 1) hold 128 output columns, tiles in
#  (k16, c) order; input-side entries (side 2) hold 128 input rows, tiles in (b, r, c) order.

def grid_to_entries(grid, side):
    """grid [in/16, out/16, X] (packed bytes or values per tile) -> [entries, tiles_per_entry, X] in stored order."""
    k16, c16 = grid.shape[:2]
    X = grid.shape[2:]
    if side == SIDE_IN:
        g = grid.reshape(k16 // 8, 8, c16 // 8, 8, *X)            # (g, r, b, c)
        g = np.moveaxis(g, 2, 1)                                   # (g, b, r, c)
        return g.reshape(k16 // 8, (c16 // 8) * 64, *X)
    g = grid.reshape(k16, c16 // 8, 8, *X)                         # (k16, g, c)
    g = np.moveaxis(g, 1, 0)                                       # (g, k16, c)
    return g.reshape(c16 // 8, k16 * 8, *X)


def entries_to_grid(ent, side, rows, cols):
    """The inverse of grid_to_entries."""
    k16, c16 = rows // 16, cols // 16
    X = ent.shape[2:]
    if side == SIDE_IN:
        g = ent.reshape(k16 // 8, c16 // 8, 8, 8, *X)               # (g, b, r, c)
        g = np.moveaxis(g, 1, 2)                                   # (g, r, b, c)
        return g.reshape(k16, c16, *X)
    g = ent.reshape(c16 // 8, k16, 8, *X)                          # (g, k16, c)
    g = np.moveaxis(g, 0, 1)                                       # (k16, g, c)
    return g.reshape(k16, c16, *X)


def grid_to_matrix(grid):
    """[in/16, out/16, 16, 16] -> (in, out)."""
    k16, c16 = grid.shape[:2]
    return np.ascontiguousarray(grid.transpose(0, 2, 1, 3)).reshape(k16 * 16, c16 * 16)


def matrix_to_grid(T):
    rows, cols = T.shape
    return np.ascontiguousarray(T.reshape(rows // 16, 16, cols // 16, 16).transpose(0, 2, 1, 3))


# ------------------------------------------------------------------------------------------------ rotation


def hadamard(n=128):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / math.sqrt(n)).astype(np.float32)


def reconstruct(T, suh, svh):
    """W (in, out) = diag(suh) Hb T Hb diag(svh), Hb the block-diagonal 128-block Hadamard."""
    k, n = T.shape
    had = hadamard(128)
    x = np.matmul(had, T.reshape(-1, 128, n)).reshape(k, n) * np.asarray(suh, np.float32)[:, None]
    return np.matmul(x.reshape(k, -1, 128), had).reshape(k, n) * np.asarray(svh, np.float32)[None, :]


# ------------------------------------------------------------------------------------------------ dtype helpers


def f32_to_bf16_bits(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_bits_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# ------------------------------------------------------------------------------------------------ dense decode


def decode_record(rec, t):
    """One stored matrix as the HF weight (out, in) fp32. rec: the 8-field record; t: {suffix: numpy array}."""
    fmt, out, inp, K, side = (int(x) for x in np.asarray(rec)[:5])
    if fmt == TRELLIS:
        ent = np.asarray(t["trellis"], dtype=np.uint8)
        n_ent = ent.shape[0]
        tiles = decode_tiles(ent.reshape(-1, 32 * K), K).reshape(n_ent, -1, 16, 16)
        T = grid_to_matrix(entries_to_grid(tiles, side, inp, out))
        W = reconstruct(T, np.asarray(t["suh"], np.float32), np.asarray(t["svh"], np.float32))
        return np.ascontiguousarray(W.T)
    if fmt == INT4:
        q4, s4 = np.asarray(t["q4"]), np.asarray(t["s4"], np.float32)
        if side == SIDE_IN:
            q4 = q4.transpose(1, 0, 2).reshape(out, -1)
            s4 = s4.transpose(1, 0, 2).reshape(out, -1)
        n = np.stack([(q4 & 15).astype(np.int32) - 8, (q4 >> 4).astype(np.int32) - 8], -1).reshape(out, inp)
        return n.astype(np.float32) * np.repeat(s4, 32, axis=1)
    if fmt == MXFP8:
        w = e4m3_decode(t["fp8"])
        e = np.asarray(t["e8m0"]).astype(np.int32) - 127
        return w * np.repeat(np.ldexp(np.float32(1), e).astype(np.float32), 32, axis=1)
    if fmt == FP8_BLOCK:
        w = e4m3_decode(t["fp8"])
        sc = np.asarray(t["scale"], np.float32)
        return w * np.repeat(np.repeat(sc, 128, 0)[:out], 128, 1)[:, :inp]
    if fmt == FP8_ROW:
        return e4m3_decode(t["fp8"]) * np.asarray(t["scale"], np.float32)[:, None]
    if fmt == BF16:
        return bf16_bits_to_f32(t["bf16"]).reshape(out, inp)
    if fmt == FP32:
        return np.asarray(t["fp32"], np.float32).reshape(out, inp)
    raise ValueError(f"unknown record format {fmt}")


# ------------------------------------------------------------------------------------------------ xxhash64


_P1, _P2, _P3, _P4, _P5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x85EBCA77C2B2AE63,
                           0x27D4EB2F165667C5)
_M64 = (1 << 64) - 1


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & _M64


def _round(acc, lane):
    return (_rotl((acc + lane * _P2) & _M64, 31) * _P1) & _M64


def _xxh64_py(data, seed=0):
    """xxHash64 (the published algorithm), pure Python -- the fallback when the xxhash package is absent."""
    n, i = len(data), 0
    if n >= 32:
        v = [(seed + _P1 + _P2) & _M64, (seed + _P2) & _M64, seed & _M64, (seed - _P1) & _M64]
        lanes = np.frombuffer(data, dtype="<u8", count=(n // 32) * 4).tolist()
        for j in range(0, len(lanes), 4):
            v = [_round(v[0], lanes[j]), _round(v[1], lanes[j + 1]), _round(v[2], lanes[j + 2]),
                 _round(v[3], lanes[j + 3])]
        h = (_rotl(v[0], 1) + _rotl(v[1], 7) + _rotl(v[2], 12) + _rotl(v[3], 18)) & _M64
        for x in v:
            h = ((h ^ _round(0, x)) * _P1 + _P4) & _M64
        i = (n // 32) * 32
    else:
        h = (seed + _P5) & _M64
    h = (h + n) & _M64
    while i + 8 <= n:
        k = struct.unpack_from("<Q", data, i)[0]
        h = ((_rotl(h ^ _round(0, k), 27) * _P1) + _P4) & _M64
        i += 8
    if i + 4 <= n:
        k = struct.unpack_from("<I", data, i)[0]
        h = ((_rotl(h ^ ((k * _P1) & _M64), 23) * _P2) + _P3) & _M64
        i += 4
    while i < n:
        h = (_rotl(h ^ ((data[i] * _P5) & _M64), 11) * _P1) & _M64
        i += 1
    h ^= h >> 33
    h = (h * _P2) & _M64
    h ^= h >> 29
    h = (h * _P3) & _M64
    return h ^ (h >> 32)


def xxh64(*parts):
    """xxhash64, seed 0, over the bytes of the given arrays in order."""
    blob = b"".join(np.ascontiguousarray(p).tobytes() for p in parts)
    try:
        import xxhash
        return xxhash.xxh64(blob, seed=0).intdigest()
    except ImportError:
        return _xxh64_py(blob)


def record_checksums(rec, t):
    """`M.checksum`: one xxhash64 per 128-channel block of the sharded side (per 128 output rows if replicated)."""
    fmt, out, inp, K, side = (int(x) for x in np.asarray(rec)[:5])
    sums = []
    for b in range(-(-(inp if side == SIDE_IN else out) // 128)):
        r = slice(128 * b, 128 * b + 128)
        if fmt == TRELLIS:
            own, other = ("suh", "svh") if side == SIDE_IN else ("svh", "suh")
            parts = [t["trellis"][b], t[own][r], t[other]]
        elif fmt == INT4 and side == SIDE_IN:
            parts = [t["q4"][b], t["s4"][b]]
        elif fmt == FP8_BLOCK:
            parts = [t["fp8"][r], t["scale"][b]]
        else:
            parts = [t[k][r] for k in DATA[fmt]]
        sums.append(xxh64(*parts))
    return np.array(sums, dtype=np.uint64)


# ------------------------------------------------------------------------------------------------ safetensors I/O

_ST_DTYPE = {np.dtype(np.uint8): "U8", np.dtype(np.int8): "I8", np.dtype(np.int16): "I16",
             np.dtype(np.int32): "I32", np.dtype(np.int64): "I64", np.dtype(np.uint64): "U64",
             np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}
_NP_DTYPE = {"U8": np.uint8, "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64, "U64": np.uint64,
             "F16": np.float16, "F32": np.float32, "BF16": np.uint16, "U16": np.uint16, "BOOL": np.uint8,
             "F64": np.float64}


def write_safetensors(path, tensors, metadata=None, align=ALIGN):
    """EXLR8-layout safetensors: header padded with spaces so data starts at `align`, every tensor at a multiple
    of `align`, gaps filled by `__pad.N` U8 tensors. tensors: {name: ndarray | (ndarray, 'BF16')}."""
    header, layout, off, npad = {}, [], 0, 0
    for name, v in tensors.items():
        arr, dt = (v if isinstance(v, tuple) else (v, None))
        arr = np.ascontiguousarray(arr)
        dt = dt or _ST_DTYPE[arr.dtype]
        if off % align:
            gap = align - off % align
            header[f"__pad.{npad}"] = {"dtype": "U8", "shape": [gap], "data_offsets": [off, off + gap]}
            layout.append(np.zeros(gap, np.uint8))
            off += gap
            npad += 1
        n = arr.nbytes
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + n]}
        layout.append(arr)
        off += n
    if metadata:
        header["__metadata__"] = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    room = max(align, -(-(len(raw) + 8) // align) * align) - 8
    if len(raw) > room:
        raise ValueError("header too large")
    raw += b" " * (room - len(raw))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for arr in layout:
            f.write(arr.tobytes())
    os.replace(tmp, path)
    return path


class SafeFile:
    """Reads named tensors from a safetensors file (numpy; BF16 comes back as uint16 bits)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.meta = self.header.pop("__metadata__", None) or {}
        self.start = 8 + n

    def names(self):
        return [k for k in self.header if not k.startswith("__pad.")]

    def dtype(self, name):
        return self.header[name]["dtype"]

    def get(self, name):
        info = self.header[name]
        a, b = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.start + a)
            data = f.read(b - a)
        return np.frombuffer(data, dtype=_NP_DTYPE[info["dtype"]]).reshape(info["shape"]).copy()

    def raw(self, name):
        """(array, dtype string) -- for copying a tensor into another file unchanged."""
        return self.get(name), self.header[name]["dtype"]


def matrix_names(f):
    return [n[:-len(".record")] for n in f.names() if n.endswith(".record")]


def load_matrices(f, prefix):
    """Every tensor of a dense / vocab / extra file as {HF name: fp32 ndarray (matrices) | ndarray (vectors)}."""
    out, owned = {}, set()
    for m in matrix_names(f):
        rec = f.get(m + ".record")
        t = {s: f.get(f"{m}.{s}") for s in DATA[int(rec[0])]}
        out[prefix(m) + ".weight"] = decode_record(rec, t)
        owned.update(f"{m}.{s}" for s in ("record", "checksum") + DATA[int(rec[0])])
    for n in f.names():
        if n not in owned:
            a, dt = f.raw(n)
            out[prefix(n)] = bf16_bits_to_f32(a) if dt == "BF16" else a
    return out


# ------------------------------------------------------------------------------------------------ allocation


def _gcd_list(xs):
    g = 0
    for x in xs:
        g = math.gcd(g, int(x))
    return g or 1


def knapsack(items, budget, max_cells=300_000_000):
    """EXACT multiple-choice knapsack: pick one option per item minimising the summed cost with summed size <=
    budget. items: [[(size, cost, label), ...], ...] (integer sizes). DP over the budget in units of the gcd of
    every size; when items x units would exceed max_cells the sizes are rounded UP to a coarser unit (never over
    budget) and `granularity` says so. Returns {choice: [label per item], size, cost, granularity, exact}."""
    if not items:
        return {"choice": [], "size": 0, "cost": 0.0, "granularity": 1, "exact": True}
    sizes = [s for it in items for s, _, _ in it]
    unit = _gcd_list(sizes)
    exact = True
    while len(items) * (budget // unit + 1) > max_cells:
        unit *= 2
        exact = False
    B = budget // unit
    INF = float("inf")
    dp = np.zeros(B + 1)                    # min cost with total size <= b (items so far)
    picks = np.zeros((len(items), B + 1), dtype=np.uint8)
    for i, opts in enumerate(items):
        new = np.full(B + 1, INF)
        arg = np.zeros(B + 1, dtype=np.uint8)
        for j, (s, c, _) in enumerate(opts):
            u = -(-int(s) // unit)
            if u > B:
                continue
            cand = np.full(B + 1, INF)
            cand[u:] = dp[:B + 1 - u] + float(c)
            better = cand < new
            new[better] = cand[better]
            arg[better] = j
        dp = new
        picks[i] = arg
        if not np.isfinite(dp[-1]):
            raise ValueError("budget below the smallest width of every tensor"
                             + ("" if exact else f" (at the coarsened {unit}-byte granularity)"))
    b = B
    choice = [None] * len(items)
    size = 0
    for i in range(len(items) - 1, -1, -1):
        j = int(picks[i, b])
        s, c, lab = items[i][j]
        choice[i] = lab
        size += int(s)
        b -= -(-int(s) // unit)
    return {"choice": choice, "size": size, "cost": float(dp[-1]), "granularity": unit, "exact": exact}


def _group_of(key):
    import pollard_exl3 as X
    return X._group_of(key)


def _layer_of(key):
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def tensor_sensitivity(tensors, sens):
    """{tensor: S_t}: Pollard's measured group KL (pollard-probe: attn / ffn per layer) split across the group's
    tensors by size -- the same split pollard-exl3's recipe bridge uses. No profile: S_t = numel share (the
    allocation then minimises the summed relative output error)."""
    numel = {k: v["numel"] for k, v in tensors.items()}
    if not sens:
        tot = float(sum(numel.values())) or 1.0
        return {k: n / tot for k, n in numel.items()}
    grp = {g: {int(k): float(v) for k, v in ((sens or {}).get(g) or {}).items()} for g in ("attn", "ffn")}
    allv = sorted(v for d in grp.values() for v in d.values())
    fallback = allv[len(allv) // 2] if allv else 1.0
    gsize = {}
    for k in tensors:
        key = (_group_of(k), _layer_of(k))
        gsize[key] = gsize.get(key, 0) + numel[k]
    out = {}
    for k in tensors:
        g, li = _group_of(k), _layer_of(k)
        other = "ffn" if g == "attn" else "attn"
        sg = grp[g].get(li, grp[other].get(li, fallback)) if li is not None else fallback
        out[k] = max(sg, 0.0) * numel[k] / max(gsize[(g, li)], 1)
    return out


def plan_rung(store_meta, bpw, sens=None, widths=None, min_width=None, max_width=None):
    """The rung's width per tensor: exact knapsack on KL(t, K) = S_t * e(t, K) / e_rtn(t) under
    sum(trellis bytes) <= bpw * numel / 8. Returns {widths, bpw, bytes, predicted_kld, granularity, exact}."""
    tensors = store_meta["tensors"]
    avail = [int(k) for k in (widths or store_meta["widths"])]
    if min_width:
        avail = [k for k in avail if k >= int(min_width)]
    if max_width:
        avail = [k for k in avail if k <= int(max_width)]
    if not avail:
        raise ValueError("no width left to choose from")
    S = tensor_sensitivity(tensors, sens)
    names = sorted(tensors)
    total = sum(tensors[n]["numel"] for n in names)
    if not (min(avail) <= bpw + 1e-9 and bpw <= max(avail) + 1e-9):
        raise ValueError(f"{bpw} bpw is outside the stored widths {min(avail)}..{max(avail)}")
    items = []
    for n in names:
        t = tensors[n]
        ref = float(t.get("err_rtn") or 0.0) or 1.0
        opts = []
        for K in avail:
            e = t["err"].get(str(K))
            if e is None:
                continue
            opts.append((int(t["bytes"][str(K)]), S[n] * float(e) / ref, K))
        if not opts:
            raise ValueError(f"{n}: no stored width among {avail}")
        items.append(opts)
    budget = int(math.floor(bpw * total / 8.0 + 1e-6))
    res = knapsack(items, budget)
    chosen = dict(zip(names, res["choice"]))
    return {"widths": chosen, "bpw": round(8.0 * res["size"] / total, 4), "bytes": res["size"],
            "target_bpw": float(bpw), "predicted_kld": round(res["cost"], 6),
            "granularity": res["granularity"], "exact": res["exact"], "numel": total}


# ------------------------------------------------------------------------------------------------ encoder (torch)


def _signs(name, n, salt):
    rng = np.random.default_rng(zlib.crc32(f"{salt}:{name}".encode()))
    return np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)


def viterbi(x, K, cb, prefix=None, end_mask=True):
    """Min-squared-error path through the K-bit, 16-bit-state trellis for each row of x [n, 256] (stream order).
    prefix [n]: force the high 16-K bits of the first state (tail-biting pass 2); the last state's low 16-K bits
    must then equal it. Returns (states [n, 256] int64, cost [n])."""
    import torch
    n = x.shape[0]
    S, P, KN = 65536, 1 << (16 - K), 1 << K
    dev = x.device
    ar = torch.arange(S, device=dev)
    cost = (cb[None, :] - x[:, :1]) ** 2
    if prefix is not None:
        cost = torch.where((ar >> K)[None, :] == prefix[:, None], cost, torch.full_like(cost, float("inf")))
    back = torch.empty((255, n, P), dtype=torch.uint8, device=dev)
    for i in range(1, 256):
        m, arg = cost.view(n, KN, P).min(dim=1)
        back[i - 1] = arg.to(torch.uint8)
        cost = m.repeat_interleave(KN, dim=1) + (cb[None, :] - x[:, i:i + 1]) ** 2
    if prefix is not None and end_mask:
        cost = torch.where((ar & (P - 1))[None, :] == prefix[:, None], cost, torch.full_like(cost, float("inf")))
    best, s = cost.min(dim=1)
    states = torch.empty((n, 256), dtype=torch.int64, device=dev)
    states[:, 255] = s
    for i in range(255, 0, -1):
        low = s >> K
        top = back[i - 1].gather(1, low[:, None]).squeeze(1).to(torch.int64)
        s = top * P + low
        states[:, i - 1] = s
    return states, best


def viterbi_tailbiting(x, K, cb, mem_bytes=1 << 30):
    """Two passes: an open one finds the best end state, whose overlap bits then pin both ends of the second, so
    state 0 = the 16 bits ending at value 0 with the stream wrapping -- what the decoder reconstructs."""
    import torch
    P = 1 << (16 - K)
    per = 255 * P + 65536 * 12                   # back pointers + the cost rows, bytes per tile
    step = max(1, int(mem_bytes // per))
    outs = []
    for a in range(0, x.shape[0], step):
        xb = x[a:a + step]
        st, _ = viterbi(xb, K, cb)
        prefix = st[:, 255] & (P - 1)
        st, _ = viterbi(xb, K, cb, prefix=prefix)
        outs.append(st)
    return torch.cat(outs) if len(outs) > 1 else outs[0]


def _rtn_rows(Wg, bits, gs=64):
    """pollard-probe's RTN (per-row absmax symmetric, group 64): the quantizer its sensitivity was measured with."""
    import torch
    q = 2 ** (bits - 1) - 1 or 1
    D = Wg.shape[1]
    r = Wg.reshape(Wg.shape[0], -1, min(gs, D)) if D % min(gs, D) == 0 else Wg.unsqueeze(1)
    scale = r.abs().amax(-1, keepdim=True).clamp_min(1e-8) / q
    return ((r / scale).round().clamp(-q - 1, q) * scale).reshape_as(Wg)


def rel_err(Wg, Wq, H):
    """tr(dW H dW^T) / tr(W H W^T) -- the share of the layer's output energy the quantization error carries."""
    import torch
    d = (Wq - Wg).float()
    if H is None:
        return float((d * d).sum() / (Wg.float() ** 2).sum().clamp_min(1e-30))
    return float(((d @ H) * d).sum() / ((Wg.float() @ H) * Wg.float()).sum().clamp_min(1e-30))


def encode_matrix(Wg, K, H=None, name="", damp=0.01, mem_bytes=1 << 30, salt="pollard-exlr8"):
    """Encode one matrix (HF weight (out, in), torch, on the encode device) at width K.

    T = Hb diag(1/suh) W diag(1/svh) Hb with W = Wg^T, suh = random signs (input side), svh = signs * per-output
    channel RMS / codebook RMS (both rounded to fp16 first: the decoder uses the fp16 values). With a Hessian
    H (in x in, from calibration) the 16-row strips of T are quantized in order with LDLQ error feedback:
    E = dStrip U_bb^-1, T_rest -= U_b,rest^T E^T, U = upper Cholesky of the damped inverse Hessian in T's space.
    Returns {trellis grid bytes [in/16, out/16, 32K] uint8, suh, svh (fp16 numpy), W_hat (torch, HF orientation),
    solve}."""
    import torch
    dev = Wg.device
    out_f, in_f = Wg.shape
    W = Wg.float().T.contiguous()                                   # (in, out)
    cb = torch.tensor(codebook_table(), device=dev)
    had = torch.tensor(hadamard(128), device=dev)
    suh = torch.tensor(_signs(name + ".in", in_f, salt), device=dev)
    g = W.norm(dim=0) / math.sqrt(in_f)
    g = torch.where(g > 0, g, torch.full_like(g, float(g.max().clamp_min(1e-8)) * 1e-3))
    svh = torch.tensor(_signs(name + ".out", out_f, salt), device=dev) * g / codebook_rms()
    suh = suh.half().float()
    svh = svh.half().float()
    svh = torch.where(svh == 0, torch.full_like(svh, 6.1e-5), svh)

    def rot(M):                                                     # Hb M Hb for an (in, out) matrix
        k, n = M.shape
        x = (had @ M.reshape(-1, 128, n)).reshape(k, n)
        return (x.reshape(k, -1, 128) @ had).reshape(k, n)
    T = rot(W / suh[:, None] / svh[None, :])
    perm = torch.tensor(_PERM, device=dev)
    k16, c16 = in_f // 16, out_f // 16
    grid_states = torch.empty((k16, c16, 256), dtype=torch.int64, device=dev)

    def quant_strips(Tblock, r0):
        """Quantize rows r0.. of the strip block (rows multiple of 16) without feedback; returns Q (same shape)."""
        rows = Tblock.shape[0]
        tiles = Tblock.reshape(rows // 16, 16, c16, 16).permute(0, 2, 1, 3).reshape(-1, 256)
        x = tiles[:, perm]
        st = viterbi_tailbiting(x, K, cb, mem_bytes)
        grid_states[r0 // 16:r0 // 16 + rows // 16] = st.reshape(rows // 16, c16, 256)
        v = torch.empty_like(x)
        v[:, perm] = cb[st]
        return v.reshape(rows // 16, c16, 16, 16).permute(0, 2, 1, 3).reshape(rows, out_f)

    if H is None:
        Q = quant_strips(T, 0)
        solve = 0
    else:
        Hs = H.float().to(dev)
        Ht = Hs * suh[:, None] * suh[None, :]
        Ht = rot(Ht)                                                # Hb diag(suh) H diag(suh) Hb
        Ht = (Ht + Ht.T) / 2
        dmean = Ht.diagonal().mean().clamp_min(1e-8)
        Ht = Ht + torch.eye(in_f, device=dev) * damp * dmean
        L = torch.linalg.cholesky(Ht)
        U = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
        Tw = T.clone()
        Q = torch.empty_like(T)
        for b in range(k16):
            r = slice(16 * b, 16 * b + 16)
            q = quant_strips(Tw[r], 16 * b)
            Q[r] = q
            if 16 * b + 16 < in_f:
                d = (Tw[r] - q).T                                   # (out, 16)
                E = torch.linalg.solve_triangular(U[r, r], d, upper=True, left=False)
                Tw[16 * b + 16:] -= (E @ U[r, 16 * b + 16:]).T
        solve = 1
    W_hat = (rot(Q) * suh[:, None] * svh[None, :]).T.contiguous()   # HF orientation (out, in)
    vals = (grid_states & ((1 << K) - 1)).cpu().numpy()
    packed = pack_values(vals.reshape(-1, 256), K).reshape(k16, c16, 32 * K)
    return {"grid": packed, "suh": suh.cpu().numpy().astype(np.float16), "svh": svh.cpu().numpy().astype(np.float16),
            "W_hat": W_hat, "solve": solve}


def trellis_record(enc, out_f, in_f, K, side):
    """The stored tensors of one trellis matrix: record, trellis [entries, bytes], suh, svh, checksum."""
    ent = grid_to_entries(enc["grid"], side)
    ent = np.ascontiguousarray(ent.reshape(ent.shape[0], -1))
    rec = np.array([TRELLIS, out_f, in_f, K, side, 128, 0, enc["solve"]], dtype=np.int64)
    t = {"trellis": ent, "suh": enc["suh"], "svh": enc["svh"]}
    return {"record": rec, "checksum": record_checksums(rec, t), **t}


def fp8_row_record(Wg):
    """FP8 E4M3 per-row record (EXLR8 format 5) -- what the GLM release uses for embed_tokens / lm_head."""
    W = np.asarray(Wg, np.float32)
    scale = (np.abs(W).max(axis=1) / 448.0).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1.0))
    fp8 = e4m3_encode(W / scale[:, None])
    rec = np.array([FP8_ROW, W.shape[0], W.shape[1], 0, SIDE_REPLICATED, 0, 0, 0], dtype=np.int64)
    t = {"fp8": fp8, "scale": scale}
    return {"record": rec, "checksum": record_checksums(rec, t), **t}


def bf16_record(Wg):
    W = np.asarray(Wg, np.float32)
    rec = np.array([BF16, W.shape[0], W.shape[1], 0, SIDE_REPLICATED, 0, 0, 0], dtype=np.int64)
    t = {"bf16": f32_to_bf16_bits(W)}
    return {"record": rec, "checksum": record_checksums(rec, t), **t}


def _record_tensors(stored, recd):
    out = {}
    for k, v in recd.items():
        if k == "bf16":
            out[f"{stored}.bf16"] = (v, "BF16")
        else:
            out[f"{stored}.{k}"] = v
    return out


def is_trellis_shape(shape):
    return len(shape) == 2 and shape[0] % 128 == 0 and shape[1] % 128 == 0


def side_of(stored_name):
    return SIDE_IN if _IN_SIDE.search(stored_name) else SIDE_OUT


# ------------------------------------------------------------------------------------------------ store


def _store_meta_path(store):
    return os.path.join(store, "store.json")


def read_store(store):
    p = _store_meta_path(store)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _write_store(store, meta):
    tmp = _store_meta_path(store) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, _store_meta_path(store))


def store_complete(store, widths=None):
    meta = read_store(store)
    if not meta or not meta.get("complete"):
        return False
    return all(int(k) in meta["widths"] for k in (widths or meta["widths"]))


def _calib_rows(tok, calib, max_rows, seqlen, max_tokens):
    """Token rows from the shared corpus: pollard-calib's whole samples (agent traces kept intact) when it wrote
    them, else its lines (ws.calib_samples)."""
    import pollard_workspace as ws
    samples = ws.calib_samples(calib) if calib and os.path.exists(calib) else []
    rows, used = [], 0
    for s in samples:
        ids = tok(s, add_special_tokens=False)["input_ids"][:seqlen]
        if len(ids) < 8:
            continue
        rows.append(ids)
        used += len(ids)
        if len(rows) >= max_rows or used >= max_tokens:
            break
    return rows


class _Stop(Exception):
    pass


def _layer_hessians(model, layer_mod, linears, rows, dev):
    """X^T X of every linear's input in one decoder layer, from the calibration rows. Linears that see the same
    input tensor (q/k/v, gate/up) share one Hessian. The forward stops after this layer."""
    import torch
    H, key_of, first = {}, {}, {}
    hooks = []

    def mk(name):
        def hook(mod, inp, _out=None):
            x = inp[0].detach()
            ptr = (x.data_ptr(), tuple(x.shape))
            if name not in key_of:
                key_of[name] = first.setdefault(ptr, name)
            k = key_of[name]
            if k != name:
                return
            x2 = x.reshape(-1, x.shape[-1]).float()
            h = x2.T @ x2
            H[k] = H[k] + h if k in H else h
        return hook
    for name, lin in linears.items():
        hooks.append(lin.register_forward_pre_hook(mk(name)))

    def stop(*_a):
        raise _Stop()
    hooks.append(layer_mod.register_forward_hook(stop))
    n_tok = 0
    try:
        with torch.no_grad():
            for r in rows:
                try:
                    model(torch.tensor([r], device=dev))
                except _Stop:
                    pass
                n_tok += len(r)
    finally:
        for h in hooks:
            h.remove()
    return {name: (H[key_of[name]] / max(n_tok, 1)) if name in key_of else None for name in linears}, n_tok


def _decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers",
                 "language_model.model.layers", "backbone.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        return path + ".", obj
    raise SystemExit("could not find the decoder layers (model.layers / transformer.h ...)")


def _trc_mode(trust_remote_code):
    return {True: "on", False: "auto", None: "auto"}.get(trust_remote_code, trust_remote_code) \
        if not isinstance(trust_remote_code, str) else trust_remote_code


def _float_record(arr, src_dt):
    a = np.asarray(arr, np.float32)
    if src_dt in ("BF16",) or src_dt not in ("F16", "F32", "F64"):
        return (f32_to_bf16_bits(a), "BF16")
    return a.astype(np.float16) if src_dt == "F16" else a


def _block_matrix(rel, shape):
    """A block tensor the trellis encodes: 2-D, tile-shaped, not a norm and not a MoE router."""
    if not rel.endswith(".weight") or len(shape) != 2 or not is_trellis_shape(tuple(shape)):
        return False
    mod = rel[:-len(".weight")]
    if "norm" in mod.rsplit(".", 1)[-1] or re.search(r"(^|\.)(gate|router)$", mod):
        return False
    return True


def encode_model(model_dir, store, widths=DEFAULT_WIDTHS, calib=None, device="auto", probe_bits=4,
                 cal_rows=128, cal_seqlen=2048, cal_tokens=131072, embed="fp8", mem_gb=1.0, align=ALIGN,
                 trust_remote_code=False, log=print, max_ram_gb=None):
    """Encode every decoder-block matrix of `model_dir` once at each width into `store`. Resumable per layer.

    STREAMED (pollard_stream): the model is never loaded whole -- the text backbone is resolved from config.json
    (native class, a VLM / omni's language model, a *_puzzle per-layer config, or the checkpoint's own code with
    consent), the calibration rows go through one block at a time read from disk (dequantized on read when the
    source is already quantized), each block's Hessians are taken from its real inputs and every matrix of the block
    is encoded at every width. MoE experts are encoded one dense record per expert matrix (checkpoint names), so a
    MoE rung decodes to the source's own checkpoint layout. Towers / talkers / projectors go to the extra file at
    source precision."""
    import torch
    import pollard_stream as S
    from transformers import AutoTokenizer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    widths = sorted({int(k) for k in widths})
    if any(k < 1 or k > 8 for k in widths):
        raise SystemExit("widths must be 1..8 bits")
    os.makedirs(store, exist_ok=True)
    meta = read_store(store) or {}
    if meta and (sorted(meta.get("widths", [])) != widths or meta.get("source") != os.path.abspath(model_dir)):
        raise SystemExit(f"{store} holds a different encode ({meta.get('source')}, widths {meta.get('widths')}); "
                         "pass another --store")
    bb = S.Backbone(model_dir, _trc_mode(trust_remote_code), log=log)
    src = bb.src
    log(f"   backbone: {bb.how}  ({bb.n_layers} blocks, streamed on {device})")
    tok = AutoTokenizer.from_pretrained(src.base_dir if src.overlay else model_dir, trust_remote_code=bb.remote)
    rows = _calib_rows(tok, calib, cal_rows, cal_seqlen, cal_tokens) if calib else []
    plan = S.fit_memory(bb, max(1, len(rows)), cal_seqlen, S.ram_budget(max_ram_gb), log=log)
    rows = rows[:plan["nsamples"]] if rows else rows
    log(f"   calibration: {len(rows)} rows / {sum(len(r) for r in rows)} tokens"
        + (f" from {os.path.basename(calib)}" if calib else " (none: plain Viterbi, no Hessian feedback)")
        + f"; memory plan peak ~{plan['peak_gb']} GB")
    prefix = bb.layer_ckpt_prefix
    if not meta:
        cfg = json.load(open(os.path.join(src.base_dir if src.overlay else model_dir, "config.json"),
                             encoding="utf-8"))
        meta = {"format": STORE_FORMAT, "version": 1, "source": os.path.abspath(model_dir), "widths": widths,
                "tensor_prefix": prefix, "num_layers": bb.n_layers, "probe": f"rtn{probe_bits}",
                "calib": os.path.abspath(calib) if calib else None, "cal_rows": len(rows),
                "cal_tokens": sum(len(r) for r in rows), "embed": embed, "align": align,
                "architecture": (cfg.get("architectures") or [None])[0], "tensors": {}, "layers_done": [],
                "complete": False, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "backbone": bb.describe(), "engine": "pollard-stream"}
        _write_store(store, meta)
    for K in widths:
        os.makedirs(os.path.join(store, f"k{K}"), exist_ok=True)
    os.makedirs(os.path.join(store, "base", "dense"), exist_ok=True)
    md = {"format": "exlr8-v2", "scheme": "exlr8-e4m3-v2", "codebook": "mcg 0xCBAC1FED, E4M3 rounding",
          "encoder": "pollard-exlr8", "model": meta["architecture"] or "", "tensor_prefix": prefix}
    mem = int(mem_gb * (1 << 30))
    t0 = time.time()
    bb.dtype = S.plan_dtype(plan)
    eng = S.Engine(bb, device=device, log=log)
    if rows:
        eng.record(rows)
    for L in range(bb.n_layers):
        if L in meta["layers_done"] and not rows:
            continue
        if L in meta["layers_done"]:
            # resuming: the block's outputs still have to be propagated for the blocks after it
            eng.load_layer(L)
            eng.hidden = eng.run_layer(L)
            bb.free_layer(L)
            continue
        lp = f"{prefix}{L}."
        if rows:
            eng.load_layer(L)
            with S.InputStats(eng, L, hessian=True, amax=False) as st:
                outs = eng.run_layer(L)
        else:
            st, outs = None, None
        base, per_k = {}, {K: {} for K in widths}
        n_mat = 0
        for k in bb.layer_keys[L]:
            rel = k[len(lp):] if k.startswith(lp) else k
            shape = src.shape(k)
            stored = rel[:-len(".weight")] if rel.endswith(".weight") else rel
            if not _block_matrix(rel, shape):
                arr = src.read(k)
                if rel.endswith(".weight") and arr.ndim == 2:
                    base.update(_record_tensors(stored, bf16_record(arr)))
                elif arr.dtype.kind == "f":
                    base[rel] = _float_record(arr, src.dtype(k))
                else:
                    base[rel] = arr
                continue
            Wg = src.read_torch(k, torch.float32).to(device)
            H, ntok = (st.hessian_for(k) if st is not None else (None, 0))
            H = H.to(device) if H is not None else None
            ref = _rtn_rows(Wg, probe_bits)
            e_rtn = rel_err(Wg, ref, H)
            side = side_of(stored)
            entry = {"layer": L, "stored": stored, "shape": list(Wg.shape), "numel": int(Wg.numel()), "side": side,
                     "err_rtn": e_rtn, "err": {}, "bytes": {}, "hessian_tokens": ntok}
            if src.encoding(k) is not None:
                entry["source_encoding"] = src.encoding(k)["algo"]
            for K in widths:
                enc = encode_matrix(Wg, K, H, lp + stored, mem_bytes=mem)
                rec = trellis_record(enc, Wg.shape[0], Wg.shape[1], K, side)
                per_k[K].update(_record_tensors(stored, rec))
                entry["err"][str(K)] = rel_err(Wg, enc["W_hat"].to(Wg.device), H)
                entry["bytes"][str(K)] = int(rec["trellis"].nbytes)
                del enc
            meta["tensors"][lp + stored + ".weight"] = entry
            n_mat += 1
        for K in widths:
            write_safetensors(os.path.join(store, f"k{K}", f"exlr8-dense-{L:03d}.safetensors"), per_k[K],
                              {**md, "layer": str(L), "k": str(K)}, align)
        write_safetensors(os.path.join(store, "base", "dense", f"exlr8-dense-{L:03d}.safetensors"), base,
                          {**md, "layer": str(L)}, align)
        if rows:
            bb.free_layer(L)
            eng.hidden = outs                      # every width is encoded from the ORIGINAL block's inputs
        meta["layers_done"].append(L)
        meta["peak_rss_gb"] = round(max(eng.peak_rss, S.peak_rss_bytes()) / 1e9, 2)
        _write_store(store, meta)
        log(f"   layer {L + 1}/{bb.n_layers}: {n_mat} matrices x {len(widths)} widths "
            f"({(time.time() - t0) / 60:.1f} min so far, peak RSS {meta['peak_rss_gb']} GB)")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    # vocab + everything outside the decoder blocks
    vocab, extra, names = {}, {}, {}
    cfg_t = bb.text_config()
    tied = bool(getattr(cfg_t, "tie_word_embeddings", False) or getattr(bb.root_config, "tie_word_embeddings", False))
    for n in bb.outer_keys + bb.other_keys:
        arr = src.read(n)
        if n.endswith(("embed_tokens.weight", "embeddings.weight", "wte.weight")) and n in bb.outer_keys:
            stored = "embed_tokens"
        elif n.endswith("lm_head.weight") and arr.ndim == 2 and n.count("lm_head") == 1 \
                and not any(t in n for t in ("talker", "code2wav", "audio", "visual", "vision")):
            if tied:
                continue
            stored = "lm_head"
        elif n in bb.outer_keys and arr.ndim == 1 and re.search(r"(^|\.)(norm|norm_f|ln_f|final_layernorm)\.weight$", n):
            stored = "norm.weight"
        else:
            extra[n] = _float_record(arr, src.dtype(n)) if arr.dtype.kind == "f" else arr
            continue
        names[stored] = n[:-len(".weight")] if stored != "norm.weight" else n
        if stored in ("embed_tokens", "lm_head"):
            vocab.update(_record_tensors(stored, fp8_row_record(arr) if embed == "fp8" else bf16_record(arr)))
        else:
            vocab[stored] = _float_record(arr, src.dtype(n))
    write_safetensors(os.path.join(store, "base", "dense", "exlr8-vocab.safetensors"), vocab,
                      {**md, "hf_names": names, "tie_word_embeddings": str(tied).lower()}, align)
    if extra:
        write_safetensors(os.path.join(store, "base", "dense", "exlr8-extra-model.safetensors"), extra, md, align)
    S.copy_side_files_of(src, os.path.join(store, "base"))
    cfgp = os.path.join(store, "base", "config.json")
    if os.path.exists(cfgp):                               # decoded rungs are plain float checkpoints
        c = json.load(open(cfgp, encoding="utf-8"))
        if c.pop("quantization_config", None) is not None:
            json.dump(c, open(cfgp, "w", encoding="utf-8"), indent=2)
    hq = os.path.join(store, "base", "hf_quant_config.json")
    if os.path.exists(hq):
        os.remove(hq)
    meta["complete"] = True
    meta["encode_minutes"] = round((time.time() - t0) / 60, 2)
    _write_store(store, meta)
    return meta


# ------------------------------------------------------------------------------------------------ rungs


def bpw_tag(bpw):
    """3.25 -> '3.25bpw', 3 -> '3.0bpw', 2.5 -> '2.5bpw' (the rung's folder name and gate tag)."""
    s = f"{float(bpw):.2f}".rstrip("0")
    return (s + "0" if s.endswith(".") else s) + "bpw"


def build_rung(store, plan, out, name=None, sens_path=None, align=None):
    """Assemble one rung folder from the store: each layer's base tensors + every matrix's records at the width
    the plan chose (copied, not re-encoded), the vocab / extra files, config + tokenizer, the manifest, the
    marker."""
    meta = read_store(store)
    if not meta or not meta.get("complete"):
        raise SystemExit(f"{store}: the encode is not complete -- run the encode first")
    align = align or meta.get("align", ALIGN)
    name = name or bpw_tag(plan["target_bpw"])
    tmp = out.rstrip("/\\") + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "dense"))
    os.makedirs(os.path.join(tmp, "manifests"))
    prefix = meta["tensor_prefix"]
    by_layer = {}
    for hf, K in plan["widths"].items():
        t = meta["tensors"][hf]
        by_layer.setdefault(t["layer"], {})[t["stored"]] = int(K)
    for L in range(meta["num_layers"]):
        base = SafeFile(os.path.join(store, "base", "dense", f"exlr8-dense-{L:03d}.safetensors"))
        tensors = {n: base.raw(n) for n in base.names()}
        files = {}
        for stored, K in sorted((by_layer.get(L) or {}).items()):
            f = files.get(K) or files.setdefault(K, SafeFile(os.path.join(store, f"k{K}",
                                                                          f"exlr8-dense-{L:03d}.safetensors")))
            for s in ("record", "checksum") + DATA[TRELLIS]:
                tensors[f"{stored}.{s}"] = f.raw(f"{stored}.{s}")
        mdat = dict(base.meta)
        mdat["layer"] = str(L)
        write_safetensors(os.path.join(tmp, "dense", f"exlr8-dense-{L:03d}.safetensors"), tensors, mdat, align)
    for fn in ("exlr8-vocab.safetensors", "exlr8-extra-model.safetensors"):
        src = os.path.join(store, "base", "dense", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp, "dense", fn))
    for fn in os.listdir(os.path.join(store, "base")):
        p = os.path.join(store, "base", fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(tmp, fn))
    hist = {}
    for K in plan["widths"].values():
        hist[str(K)] = hist.get(str(K), 0) + 1
    manifest = {"format": MANIFEST_FORMAT, "name": name, "version": 1, "source": meta["source"],
                "rule": "pollard: exact multiple-choice knapsack on S_t * e(t,K) / e_rtn(t) under the byte budget "
                        "(S = measured layer sensitivity, e = measured output error at width K)",
                "sensitivity": os.path.abspath(sens_path) if sens_path else None,
                "target_bpw": plan["target_bpw"], "bpw": plan["bpw"], "trellis_bytes": plan["bytes"],
                "numel": plan["numel"], "predicted_kld": plan["predicted_kld"], "granularity": plan["granularity"],
                "exact": plan["exact"], "width_histogram": hist, "tensor_prefix": prefix,
                "widths": dict(sorted(plan["widths"].items())), "store_widths": meta["widths"]}
    with open(os.path.join(tmp, "manifests", f"exlr8-manifest-{name}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    marker = {"lane": "exlr8", "format": "exlr8-v2", "rung": name, "bpw": plan["bpw"],
              "target_bpw": plan["target_bpw"], "predicted_kld": plan["predicted_kld"], "width_histogram": hist,
              "manifest": f"manifests/exlr8-manifest-{name}.json", "source": meta["source"],
              "architecture": meta.get("architecture"), "embed": meta.get("embed"),
              "runtime": "reference decode (pollard-exlr8 --decode) -> transformers / vLLM; GR8 serves GLM-5.3 only"}
    with open(os.path.join(tmp, MARKER), "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=1)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(tmp, out)
    return manifest


def is_rung(path):
    return os.path.isfile(os.path.join(str(path), MARKER))


def iter_rung_tensors(rung):
    """(HF name, fp32 ndarray) for every tensor of a rung -- the reference decode, one file at a time."""
    dense = os.path.join(rung, "dense")
    files = sorted(f for f in os.listdir(dense) if f.endswith(".safetensors"))
    for fn in files:
        f = SafeFile(os.path.join(dense, fn))
        if fn.startswith("exlr8-dense-"):
            prefix = f.meta.get("tensor_prefix", "model.layers.") + str(int(f.meta["layer"])) + "."
            yield from load_matrices(f, lambda n, p=prefix: p + n).items()
        elif fn == "exlr8-vocab.safetensors":
            names = json.loads(f.meta.get("hf_names") or "{}")
            for n, v in load_matrices(f, lambda n: n).items():
                stem = n[:-len(".weight")] if n.endswith(".weight") and n[:-len(".weight")] in names else None
                if stem:
                    yield names[stem] + ".weight", v
                else:
                    yield names.get(n, n), v
        else:
            yield from load_matrices(f, lambda n: n).items()


def decoded_bytes(rung, bytes_per=2):
    """Bytes the rung occupies once decoded to bf16 (records: out x in; vectors: as stored), headers only."""
    total = 0
    dense = os.path.join(rung, "dense")
    for fn in os.listdir(dense):
        if not fn.endswith(".safetensors"):
            continue
        f = SafeFile(os.path.join(dense, fn))
        mats = set(matrix_names(f))
        for m in mats:
            rec = f.get(m + ".record")
            total += int(rec[1]) * int(rec[2]) * bytes_per
        owned = {f"{m}.{s}" for m in mats for s in ("record", "checksum", *DATA[TRELLIS], "fp8", "scale", "bf16",
                                                    "fp32", "q4", "s4", "e8m0")}
        for n in f.names():
            if n not in owned:
                total += int(np.prod(f.header[n]["shape"] or [1])) * bytes_per
    return total


def load_model(rung, device="cpu", dtype=None, trust_remote_code=False):
    """A transformers model holding the rung's decoded weights (bit-exact to the stored codes) -- built the way
    pollard_stream builds any backbone (native class, VLM / omni language model, *_puzzle per-layer configs, or the
    checkpoint's own code with consent), loaded through transformers' own key conversion (MoE experts fused)."""
    import torch
    import pollard_stream as S
    dtype = dtype or torch.bfloat16
    dec = dict(iter_rung_tensors(rung))
    bb = S.Backbone(rung, _trc_mode(trust_remote_code), source=S.MemSource(rung, dec), dtype=dtype)
    keys = [k for k in dec if k in bb.kmap]
    bb.load(keys, device, dtype)
    if getattr(bb.root_config, "tie_word_embeddings", False) and hasattr(bb.root, "tie_weights"):
        bb.root.tie_weights()
    left = [n for n, p in bb.root.named_parameters() if p.is_meta]
    text_left = [n for n in left if n.startswith(bb.text_root_path) or "lm_head" in n]
    if text_left:
        raise SystemExit(f"{rung}: decoded weights miss {text_left[:5]}")
    return bb.root.eval()


def materialize(rung, out, dtype="bf16"):
    """Write the decoded rung as a plain HF checkpoint (bf16 safetensors + config/tokenizer) -- what transformers
    and vLLM load. Bit-exact to the stored codes up to the bf16 cast. Written shard by shard (bounded memory)."""
    import pollard_stream as S
    os.makedirs(out, exist_ok=True)
    w = S.ShardWriter(out)
    for n, v in iter_rung_tensors(rung):
        if dtype == "bf16" and v.dtype in (np.float32, np.float16, np.float64):
            w.add(n, "BF16", f32_to_bf16_bits(v.astype(np.float32)))
        else:
            dt = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16", np.dtype(np.uint16): "BF16",
                  np.dtype(np.int64): "I64", np.dtype(np.int32): "I32", np.dtype(np.uint8): "U8"}.get(v.dtype, "F32")
            w.add(n, dt, v)
    w.close()
    for fn in os.listdir(rung):
        if os.path.isfile(os.path.join(rung, fn)) and fn != MARKER and not fn.endswith(".safetensors"):
            shutil.copy2(os.path.join(rung, fn), os.path.join(out, fn))
    with open(os.path.join(out, "pollard-exlr8-decoded.json"), "w", encoding="utf-8") as f:
        json.dump({"from": os.path.abspath(rung), "dtype": dtype}, f)
    return out


def inspect(path):
    """A rung (marker + manifest) or a store (store.json) as a dict for printing."""
    if is_rung(path):
        mk = json.load(open(os.path.join(path, MARKER), encoding="utf-8"))
        man = json.load(open(os.path.join(path, mk["manifest"]), encoding="utf-8"))
        return {"kind": "rung", **mk, "trellis_bytes": man["trellis_bytes"], "exact": man["exact"]}
    meta = read_store(path)
    if meta:
        return {"kind": "store", "source": meta["source"], "widths": meta["widths"], "complete": meta["complete"],
                "tensors": len(meta["tensors"]), "layers_done": len(meta["layers_done"]),
                "num_layers": meta["num_layers"]}
    raise SystemExit(f"{path}: not an EXLR8 rung or store")


# ------------------------------------------------------------------------------------------------ CLI


def _record(model, out, bpw_target, plan):
    try:
        import pollard_workspace as ws
        ws.record_build(model, "exlr8", out, tag=bpw_tag(bpw_target), bpw=plan["bpw"],
                        extra=None)
    except Exception as e:                                              # noqa: BLE001
        print(f"   (warning: could not record the build in the workspace MANIFEST: {e})")


def _default_store(model):
    try:
        import pollard_workspace as ws
        return os.path.join(ws.model_dir(model, create=True), "exlr8-store")
    except Exception:                                                   # noqa: BLE001
        return os.path.basename(str(model).rstrip("/\\")) + "-exlr8-store"


def _rung_out(model, bpw, explicit=None):
    try:
        import pollard_workspace as ws
        return ws.resolve_out(model, "exlr8", bpw_tag(bpw), explicit)
    except Exception:                                                   # noqa: BLE001
        return explicit or f"{os.path.basename(str(model).rstrip('/'))}-Pollard-EXLR8-{bpw_tag(bpw)}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard-exlr8", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", help="the HF source dir (bf16/fp16; the smoothed copy in a release)")
    ap.add_argument("--out", help="the rung folder (one --bpw), or the parent of every rung (--ladder)")
    ap.add_argument("--store", help="where the one-time encode lives (default: <workspace>/models/<M>/exlr8-store)")
    ap.add_argument("--bpw", type=float, help="one rung at this average trellis bits/weight")
    ap.add_argument("--ladder", help="comma list of bpw: every rung from the one encode (e.g. 2.5,3,3.25,4,5,6)")
    ap.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)), help="trellis widths K to encode")
    ap.add_argument("--min-width", type=int, help="no tensor below this width in any rung")
    ap.add_argument("--calib", help="calibration corpus (pollard-calib output; its .samples.jsonl twin is used)")
    ap.add_argument("--sensitivity", help="pollard-probe sensitivity.json (attn/ffn per layer) -- the measured "
                    "allocation; without it the knapsack minimises summed relative output error")
    ap.add_argument("--device", default="auto", help="encode device: auto (cuda if present) | cuda | cpu | mps")
    ap.add_argument("--cal-rows", type=int, default=128)
    ap.add_argument("--cal-seqlen", type=int, default=2048)
    ap.add_argument("--cal-tokens", type=int, default=131072)
    ap.add_argument("--embed", choices=("fp8", "bf16"), default="fp8", help="embed_tokens / lm_head record format")
    ap.add_argument("--mem-gb", type=float, default=1.0, help="Viterbi back-pointer memory per batch")
    ap.add_argument("--align", type=int, default=None, help=f"tensor alignment in the safetensors files (default "
                    f"{ALIGN}, EXLR8's own; a smaller one saves ~64 KB per tensor on small models)")
    ap.add_argument("--encode-only", action="store_true", help="fill the store, build no rung")
    ap.add_argument("--plan-only", action="store_true", help="print what would run (and, with a complete store, "
                    "every rung's widths / size / predicted KL), build nothing")
    ap.add_argument("--decode", metavar="RUNG", help="write RUNG decoded as a bf16 HF checkpoint to --out")
    ap.add_argument("--inspect", metavar="PATH", help="print a rung's or a store's summary")
    ap.add_argument("--trust-remote-code", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="run the checkpoint's own code when no native transformers class covers it (never silent: "
                         "'auto' asks on a terminal, refuses elsewhere naming the files)")
    ap.add_argument("--max-ram-gb", type=float, help="cap the encode's RAM (default 85%% of what is free)")
    a = ap.parse_args(argv)

    if a.inspect:
        print(json.dumps(inspect(a.inspect), indent=1))
        return 0
    if a.decode:
        if not a.out:
            ap.error("--decode needs --out")
        materialize(a.decode, a.out)
        print(f"wrote {a.out} (bf16, decoded from {a.decode}) -- transformers / vLLM load it as-is")
        return 0
    if not a.model:
        ap.error("--model is required")
    widths = [int(x) for x in str(a.widths).split(",") if x.strip()]
    ladder = [float(x) for x in a.ladder.split(",")] if a.ladder else ([a.bpw] if a.bpw else [])
    if not ladder and not a.encode_only:
        ap.error("pass --bpw, --ladder or --encode-only")
    store = a.store or _default_store(a.model)
    sens = json.load(open(a.sensitivity, encoding="utf-8")) if a.sensitivity and os.path.exists(a.sensitivity) \
        else None
    if a.sensitivity and sens is None:
        raise SystemExit(f"--sensitivity {a.sensitivity} not found")
    print(f"== pollard-exlr8 :: {a.model}")
    print(f"   store {store}  widths K={widths}  ladder {[bpw_tag(b) for b in ladder] or '(encode only)'}")
    print(f"   allocation: {'measured sensitivity ' + os.path.basename(a.sensitivity) if sens else 'no profile -- summed relative output error'}")
    done = store_complete(store, widths)
    if a.plan_only:
        print(f"   encode: {'complete (reused)' if done else 'NEEDED -- every matrix x ' + str(len(widths)) + ' widths'}")
        if done:
            meta = read_store(store)
            for b in ladder:
                p = plan_rung(meta, b, sens, widths, a.min_width)
                print(f"   {bpw_tag(b):>8}: {p['bpw']:.3f} bpw  {p['bytes'] / 1e6:.1f} MB trellis  "
                      f"predicted KL {p['predicted_kld']:.4g}")
        return 0
    if not done:
        if a.calib is None:
            print("   !! no --calib: encoding WITHOUT Hessian feedback (plain Viterbi) -- pass the pollard-calib corpus")
        encode_model(a.model, store, widths, a.calib, a.device, cal_rows=a.cal_rows, cal_seqlen=a.cal_seqlen,
                     cal_tokens=a.cal_tokens, embed=a.embed, mem_gb=a.mem_gb, align=a.align or ALIGN,
                     trust_remote_code=a.trust_remote_code, max_ram_gb=a.max_ram_gb)
    else:
        print("   encode: complete -- reusing it (one encode, many rungs)")
    if a.encode_only:
        return 0
    meta = read_store(store)
    for b in ladder:
        p = plan_rung(meta, b, sens, widths, a.min_width)
        if a.ladder:
            out = os.path.join(a.out, f"{os.path.basename(str(a.model).rstrip('/'))}-Pollard-EXLR8-{bpw_tag(b)}") \
                if a.out else _rung_out(a.model, b)
        else:
            out = _rung_out(a.model, b, a.out)
        man = build_rung(store, p, out, sens_path=a.sensitivity, align=a.align)
        _record(a.model, out, b, p)
        print(f"   {bpw_tag(b):>8} -> {out}  ({p['bpw']:.3f} bpw, widths {man['width_histogram']}, "
              f"predicted KL {p['predicted_kld']:.4g})")
    print("   run a rung: pollard-exlr8 --decode <rung> --out <dir>  (bf16 HF checkpoint for transformers / vLLM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
