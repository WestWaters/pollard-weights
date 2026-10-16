#!/usr/bin/env python3
"""pollard-mx MXFP6 rungs and measured MX mixes -- the model-free encoder behind `pollard-mx --format mxfp6-e3m2 |
mxfp6-e2m3` and `pollard-mx --mix <bpw>`.

FORMATS (OCP Microscaling Formats spec v1.0): a block of 32 weights along K shares one E8M0 scale byte
(2^(e-127)); each weight is an element code. MXFP6 E3M2 (bias 3, max 28) has range, E2M3 (bias 1, max 7.5) has
precision; neither has Inf/NaN codes. 32 x 6 bits + 8 = 6.25 bits per weight. Blackwell (sm_100/sm_120) has native
FP6 MMA. Every encode here is a CPU reference: numpy only, round-to-nearest-even, saturating.

    element        bits/weight   on disk
    MXFP4  (E2M1)  4.25          2 codes per byte, element 2i in the low nibble          + U8 E8M0 [N, K/32]
    NVFP4  (E2M1)  4.5           as MXFP4, E4M3 scale per 16 + F32 tensor scale (compressed-tensors only)
    MXFP6  (E3M2 / E2M3) 6.25    4 codes per 3 bytes, c0 | c1<<6 | c2<<12 | c3<<18 LE   + U8 E8M0 [N, K/32]
    MXFP8  (E4M3)  8.25          one F8_E4M3 byte per weight                              + U8 E8M0 [N, K/32]

The 4-in-3 pack is the one compressed-tensors (`mxfp6-pack-quantized`, vllm-project/compressed-tensors#941), AMD
Quark's export and Graft's native MXFP6 types (ids 405 / 406, `{u8 e8m0; u8 qs[24]}`) all use, so one encode is a
byte copy into each of them.

LAYOUTS (where the bytes go; what loads them, as of 2026-10-15):
  quark               quant_method "quark", per-module `layer_quant_config` (dtype fp6_e3m2 / fp6_e2m3 / fp4 /
                      fp8_e4m3, per_group 32, scale_format e8m0), tensors `<m>.weight` + `<m>.weight_scale`. vLLM main
                      loads it today (Quark OCP-MX scheme: MXFP6 E3M2 and E2M3, MXFP4, MXFP8, mixed per layer; the
                      MXFP6 linears take the Humming kernel on CUDA where it applies, emulation otherwise). The only
                      mainstream path for E3M2. Graft's importer does not read Quark configs (yet).
  compressed-tensors  `mxfp6-pack-quantized` (E2M3 only: compressed-tensors' num_bits=6 float IS E2M3), tensors
                      `<m>.weight_packed` + `<m>.weight_scale`. Graft repacks a uniform one losslessly into
                      MXFP6_E2M3 (canary max diff 0); compressed-tensors main has it (#941, merged 2026-10-16); vLLM
                      loads it once vllm-project/vllm#61666 lands. A mix here is `mixed-precision` (NVFP4 / MXFP4 /
                      MXFP6-E2M3 / MXFP8 config groups), which Graft's importer does not read.
  auto (default)      uniform E2M3 -> compressed-tensors (Graft + the compressed-tensors ecosystem); anything with
                      E3M2, and every mix -> quark (what vLLM main serves now).

MIXES (--mix <target bpw>): the MX lane's measured allocation, per fused group (q/k/v together, gate/up together,
all experts of a layer together -- vLLM wants one scheme per fused layer). Each group's relative reconstruction
error is measured at every atom (imatrix-weighted when given), scaled by the layer's measured sensitivity
(pollard-probe), and a greedy convex-hull knapsack upgrades the groups with the best error drop per extra bit until
the average hits the target: sensitive layers end at MXFP8 / MXFP6, robust ones at FP4. Routers, lm_head,
embeddings, norms and vision / projector towers are never quantized (they stay at the checkpoint's precision).
The FP6 atom is E2M3, E3M2, or `auto` = whichever element reconstructs that group better.

Weight-only (activations stay 16-bit): no calibration set, CPU only, minutes for a small model. Every rung carries
`pollard_mx.json` (format, layout, per-module atom, bpw, which runtimes load it) for the card and the verify step.
"""
from __future__ import annotations

import heapq, json, os, re, shutil, struct, sys

import numpy as np

BLOCK = 32
#: element -> (exponent bits, mantissa bits, bias)  -- OCP MX v1.0 section 5.3
ELEMS = {"e2m1": (2, 1, 1), "e3m2": (3, 2, 3), "e2m3": (2, 3, 1), "e4m3": (4, 3, 7)}
#: unbiased exponent of the largest normal (shared exponent = floor(log2 amax) - emax)
EMAX = {"e2m1": 2, "e3m2": 4, "e2m3": 2, "e4m3": 8}

#: atom -> bits per weight (scales included), element, packing
ATOMS = {
    "mxfp4": {"bpw": 4.25, "elem": "e2m1"},
    "nvfp4": {"bpw": 4.5, "elem": "e2m1"},
    "mxfp6_e3m2": {"bpw": 6.25, "elem": "e3m2"},
    "mxfp6_e2m3": {"bpw": 6.25, "elem": "e2m3"},
    "mxfp8": {"bpw": 8.25, "elem": "e4m3"},
}
FORMATS = {"mxfp6-e3m2": "mxfp6_e3m2", "mxfp6-e2m3": "mxfp6_e2m3"}
LAYOUTS = ("auto", "quark", "compressed-tensors")
SCALE_RULES = ("ocp", "search")
QUARK_DTYPE = {"mxfp4": "fp4", "mxfp6_e3m2": "fp6_e3m2", "mxfp6_e2m3": "fp6_e2m3", "mxfp8": "fp8_e4m3"}
SIDECAR = "pollard_mx.json"


# ----------------------------------------------------------------------------------------------- elements

def elem_table(elem):
    """Every code's value (float64, code order; sign is the top bit), decoded field by field from the spec.
    E4M3: S.1111.111 is NaN. FP6 / FP4 have no special codes."""
    eb, mb, bias = ELEMS[elem]
    n = 1 + eb + mb
    out = np.empty(1 << n, np.float64)
    for c in range(1 << n):
        s = -1.0 if (c >> (n - 1)) & 1 else 1.0
        e = (c >> mb) & ((1 << eb) - 1)
        m = c & ((1 << mb) - 1)
        if elem == "e4m3" and e == (1 << eb) - 1 and m == (1 << mb) - 1:
            out[c] = np.nan
            continue
        out[c] = s * (m / (1 << mb) * 2.0 ** (1 - bias) if e == 0 else (1 + m / (1 << mb)) * 2.0 ** (e - bias))
    return out


def elem_bits(elem):
    eb, mb, _ = ELEMS[elem]
    return 1 + eb + mb


_POS = {}


def _positive(elem):
    """(finite positive magnitudes in code order, midpoints) -- magnitudes rise with the code."""
    if elem not in _POS:
        t = elem_table(elem)
        half = len(t) // 2
        pos = t[:half]
        pos = pos[np.isfinite(pos)]
        _POS[elem] = (pos, (pos[1:] + pos[:-1]) / 2)
    return _POS[elem]


def quantize_elem(y, elem):
    """Nearest code of `y` (already divided by the block scale): round to nearest, ties to the even code (even
    mantissa), saturating at the largest finite magnitude. NaN -> 0. Returns uint8 codes."""
    pos, mids = _positive(elem)
    y = np.asarray(y, np.float64)
    a = np.nan_to_num(np.abs(y), nan=0.0, posinf=np.inf)
    idx = np.searchsorted(mids, a, side="left")
    tie = (idx < len(mids)) & (a == mids[np.minimum(idx, len(mids) - 1)]) & (idx % 2 == 1)
    idx = np.where(tie, idx + 1, idx)
    n = elem_bits(elem)
    sign = ((y < 0) & (idx > 0)).astype(np.uint8) << (n - 1)
    return idx.astype(np.uint8) | sign


def decode_elem(codes, elem):
    return elem_table(elem)[np.asarray(codes, np.int64)]


def e8m0_value(e):
    """E8M0 byte -> 2^(e-127) as float64 (0xFF is NaN)."""
    e = np.asarray(e, np.int64)
    return np.where(e == 255, np.nan, np.ldexp(1.0, e - 127))


# ----------------------------------------------------------------------------------------------- MX blocks

def _floor_log2(a):
    m, x = np.frexp(a)                                   # a = m * 2^x, m in [0.5, 1)
    return x.astype(np.int64) - 1


def mx_encode(w, elem, rule="search", imp=None):
    """[N, K] float -> (codes uint8 [N, K], E8M0 scales uint8 [N, K/32]).

    rule "ocp": the spec's shared exponent, floor(log2(block amax)) - emax(elem).
    rule "search": per block, that exponent or one either side, whichever has the lower (imatrix-weighted) squared
    error -- the output is still standard MX, just a better-chosen scale byte."""
    w = np.asarray(w, np.float64)
    N, K = w.shape
    if K % BLOCK:
        raise ValueError(f"K={K} is not a multiple of the MX block ({BLOCK})")
    if rule not in SCALE_RULES:
        raise ValueError(f"scale rule is one of {SCALE_RULES}")
    blk = w.reshape(N, K // BLOCK, BLOCK)
    amax = np.abs(blk).max(-1)
    e = np.where(amax > 0, _floor_log2(np.where(amax > 0, amax, 1.0)) - EMAX[elem], -127)
    e = np.clip(e, -127, 127)

    def at(ex):
        ex = np.clip(ex, -127, 127)
        codes = quantize_elem(blk / np.ldexp(1.0, ex)[..., None], elem)
        return codes, ex
    codes, ex = at(e)
    if rule == "search":
        I = None if imp is None else np.asarray(imp, np.float64).reshape(1, K // BLOCK, BLOCK)

        def err(c, x):
            d = (blk - decode_elem(c, elem) * np.ldexp(1.0, x)[..., None]) ** 2
            return (d if I is None else d * I).sum(-1)
        best = err(codes, ex)
        for delta in (-1, 1):
            c2, x2 = at(e + delta)
            e2 = err(c2, x2)
            take = e2 < best
            codes = np.where(take[..., None], c2, codes)
            ex = np.where(take, x2, ex)
            best = np.where(take, e2, best)
    return codes.reshape(N, K), (ex + 127).astype(np.uint8)


def mx_decode(codes, scales, elem):
    c = np.asarray(codes)
    N, K = c.shape
    v = decode_elem(c, elem).reshape(N, K // BLOCK, BLOCK) * e8m0_value(scales)[..., None]
    return v.reshape(N, K)


def pack6(codes):
    """[N, K] 6-bit codes -> [N, 3K/4] bytes: word = c0 | c1<<6 | c2<<12 | c3<<18, three bytes little-endian."""
    c = np.asarray(codes, np.uint32)
    N, K = c.shape
    if K % 4:
        raise ValueError("K must be a multiple of 4 to pack FP6")
    g = c.reshape(N, K // 4, 4)
    word = g[..., 0] | (g[..., 1] << 6) | (g[..., 2] << 12) | (g[..., 3] << 18)
    out = np.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], -1)
    return out.astype(np.uint8).reshape(N, K // 4 * 3)


def unpack6(packed):
    p = np.asarray(packed, np.uint32)
    N, B = p.shape
    t = p.reshape(N, B // 3, 3)
    word = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
    return np.stack([(word >> s) & 0x3F for s in (0, 6, 12, 18)], -1).astype(np.uint8).reshape(N, B // 3 * 4)


def pack4(codes):
    c = np.asarray(codes, np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack4(packed):
    p = np.asarray(packed, np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), np.uint8)
    out[..., 0::2], out[..., 1::2] = p & 0xF, p >> 4
    return out


def graft_blocks(row_codes_bytes, scales, elem):
    """The bytes of Graft's native MX type for these rows: per block of 32, the E8M0 byte then the block's code
    bytes as the checkpoint stores them (graft-quant mx.rs `to_native`: a byte copy, no arithmetic)."""
    b = np.asarray(row_codes_bytes, np.uint8)
    s = np.asarray(scales, np.uint8)
    N, G = s.shape
    qb = BLOCK * elem_bits(elem) // 8
    return np.concatenate([s[..., None], b.reshape(N, G, qb)], -1).reshape(N, G * (qb + 1))


# ----------------------------------------------------------------------------------------------- atoms

def encode_atom(w, atom, imp=None, rule="search", global_scale=None):
    """Encode one [N, K] weight as `atom`. Returns (tensors {suffix: (st_dtype, array)}, decoded float32)."""
    w = np.asarray(w, np.float32)
    if atom == "nvfp4":
        import pollard_mx as X
        packed, sc, g, _ = X.nvfp4_encode(w, "6", imp, global_scale=global_scale)
        dec = X.nvfp4_decode(packed, sc, g)
        e4 = _f32_to_e4m3_bytes(sc)
        return ({"weight_packed": ("U8", packed), "weight_scale": ("F8_E4M3", e4),
                 "weight_global_scale": ("F32", np.array([g], np.float32))}, dec.astype(np.float32))
    elem = ATOMS[atom]["elem"]
    codes, scales = mx_encode(w, elem, rule, imp)
    dec = mx_decode(codes, scales, elem).astype(np.float32)
    if atom == "mxfp8":
        data = ("F8_E4M3", codes.astype(np.uint8))
    elif atom == "mxfp4":
        data = ("U8", pack4(codes))
    else:
        data = ("U8", pack6(codes))
    return {"weight": data, "weight_scale": ("U8", scales)}, dec


def decode_atom(tensors, atom):
    """Inverse of encode_atom on the stored tensors ({suffix: array}); float32 [N, K]."""
    if atom == "nvfp4":
        import pollard_mx as X
        sc = decode_elem(np.asarray(tensors["weight_scale"], np.uint8), "e4m3")
        return X.nvfp4_decode(tensors["weight_packed"], sc, float(np.asarray(tensors["weight_global_scale"]).reshape(-1)[0])
                              ).astype(np.float32)
    elem = ATOMS[atom]["elem"]
    data = np.asarray(tensors.get("weight", tensors.get("weight_packed")), np.uint8)
    codes = data if atom == "mxfp8" else unpack4(data) if atom == "mxfp4" else unpack6(data)
    return mx_decode(codes, tensors["weight_scale"], elem).astype(np.float32)


def _f32_to_e4m3_bytes(x):
    """Float32 values that are exact E4M3 values -> their E4M3 code bytes."""
    vals = elem_table("e4m3")
    x = np.asarray(x, np.float64)
    codes = quantize_elem(x, "e4m3")
    if not np.array_equal(vals[codes], np.where(x == 0, 0.0, x)):
        raise ValueError("NVFP4 block scales are not exact E4M3 values")
    return codes


# ----------------------------------------------------------------------------------------------- safetensors

ST_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U8": np.uint8, "I8": np.int8, "F8_E4M3": np.uint8,
         "F8_E5M2": np.uint8, "I32": np.int32, "I64": np.int64, "F64": np.float64, "BOOL": np.bool_, "U16": np.uint16,
         "I16": np.int16, "U32": np.uint32, "U64": np.uint64}


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def st_read(path, name, hdr=None):
    """(st dtype, raw numpy array in the storage dtype -- BF16 as uint16)."""
    hdr, base = hdr or st_header(path)
    info = hdr[name]
    a, b = info["data_offsets"]
    arr = np.fromfile(path, dtype=ST_NP[info["dtype"]], count=(b - a) // np.dtype(ST_NP[info["dtype"]]).itemsize,
                      offset=base + a)
    return info["dtype"], arr.reshape(info["shape"])


def to_f32(dtype, arr):
    if dtype == "BF16":
        return (arr.astype(np.uint32) << 16).view(np.float32)
    if dtype in ("F32", "F16", "F64"):
        return arr.astype(np.float32)
    raise ValueError(f"not a float tensor: {dtype}")


def st_write(path, tensors, metadata=None):
    """tensors: {name: (st dtype, array)} in order. Pure numpy, the safetensors on-disk format."""
    hdr, off, blobs = {}, 0, []
    for name, (dt, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        raw = a.tobytes()
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for raw in blobs:
            f.write(raw)
    os.replace(tmp, path)


def source_shards(model_dir):
    names = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not names:
        raise SystemExit(f"no *.safetensors in {model_dir}")
    return [os.path.join(model_dir, n) for n in names]


# ----------------------------------------------------------------------------------------------- what to quantize

def _globs():
    import pollard_mx as X
    pats = X.ROUTER_GLOBS + X.VISION_GLOBS + X.PROJECTOR_GLOBS
    return [re.compile(p[3:]) for p in pats]


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def quantizable(name, shape, quantize_vision=False):
    """(True, module) for a decoder Linear weight this lane quantizes; (False, why) otherwise."""
    if not name.endswith(".weight") or len(shape) != 2:
        return False, "not a 2-D weight"
    mod = name[: -len(".weight")]
    last = mod.rsplit(".", 1)[-1]
    if "norm" in last or "embed" in mod.split(".")[-1] or mod.endswith("lm_head") or "embed_tokens" in mod:
        return False, "norm / embedding / lm_head"
    if not _LAYER.search(name):
        return False, "outside the decoder layers"
    import pollard_mx as X
    for rx, pat in zip(_globs(), X.ROUTER_GLOBS + X.VISION_GLOBS + X.PROJECTOR_GLOBS):
        if quantize_vision and pat in X.VISION_GLOBS:
            continue
        if rx.fullmatch(mod):
            return False, "router / vision / projector"
    if shape[1] % BLOCK:
        return False, f"K={shape[1]} not a multiple of {BLOCK}"
    return True, mod


def fused_key(mod):
    """vLLM fuses q/k/v and gate/up into one layer, and all experts of a layer into one MoE layer: each must take
    ONE scheme, so they are allocated together."""
    if ".experts." in mod:
        return mod.split(".experts.", 1)[0] + ".experts"
    head, last = mod.rsplit(".", 1)
    if last in ("q_proj", "k_proj", "v_proj"):
        return head + ".qkv_proj"
    if last in ("gate_proj", "up_proj"):
        return head + ".gate_up_proj"
    return mod


def layer_of(mod):
    m = _LAYER.search(mod + ".")
    return int(m.group(1)) if m else -1


def scan(model_dir, quantize_vision=False):
    """[(shard, tensor name, module, shape, dtype)] for every quantizable linear, plus the skipped 2-D weights."""
    lin, skipped = [], []
    for sh in source_shards(model_dir):
        hdr, _ = st_header(sh)
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            ok, mod = quantizable(name, info["shape"], quantize_vision)
            if ok:
                lin.append((sh, name, mod, tuple(info["shape"]), info["dtype"]))
            elif len(info["shape"]) == 2 and name.endswith(".weight") and _LAYER.search(name):
                skipped.append((name[: -len(".weight")], mod))      # (module, why)
    return lin, skipped


# ----------------------------------------------------------------------------------------------- allocation

def _imp_for(importance, mod, K):
    if not importance:
        return None
    v = importance.get(mod)
    if v is None:
        return None
    v = np.asarray(v, np.float64).reshape(-1)
    return v if v.size == K else None


def measure_errors(model_dir, lin, atoms, importance=None, rule="search", log=print):
    """{fused group: {"n": params, "layer": i, "members": [...], "w2": sum ||W||^2_imp, "err": {atom: sum err}}}."""
    groups = {}
    hdrs = {}
    for k, (sh, name, mod, shape, dt) in enumerate(lin):
        hdrs.setdefault(sh, st_header(sh))
        w = to_f32(*st_read(sh, name, hdrs[sh]))
        imp = _imp_for(importance, mod, shape[1])
        I = 1.0 if imp is None else imp[None, :]
        g = groups.setdefault(fused_key(mod), {"n": 0, "layer": layer_of(mod), "members": [], "w2": 0.0,
                                               "err": {a: 0.0 for a in atoms}})
        g["n"] += int(np.prod(shape))
        g["members"].append(mod)
        g["w2"] += float((w.astype(np.float64) ** 2 * I).sum())
        for a in atoms:
            _, dec = encode_atom(w, a, imp, rule)
            g["err"][a] += float(((w.astype(np.float64) - dec) ** 2 * I).sum())
        if log and (k + 1) % 50 == 0:
            log(f"   measured {k + 1}/{len(lin)} linears")
    return groups


def _hull(points):
    """Lower convex hull of [(bits, cost, atom)] sorted by bits, keeping only cost-decreasing points."""
    pts = sorted(points)
    keep = []
    for p in pts:
        if keep and p[1] >= keep[-1][1]:
            continue
        while len(keep) >= 2:
            (b0, c0, _), (b1, c1, _) = keep[-2], keep[-1]
            if (c0 - c1) * (p[0] - b1) <= (c1 - p[1]) * (b1 - b0):   # middle point is not on the hull
                keep.pop()
            else:
                break
        keep.append(p)
    return keep


def allocate_mix(groups, target_bpw, atoms, sens=None, focus=(), fp6="auto"):
    """Pick an atom per fused group so the parameter-weighted average bits/weight is <= target_bpw.

    cost(group, atom) = s_layer * (n_group / n_layer) * rel_err(atom) / mean FP4 rel_err -- the layer's measured
    damage (pollard-probe sensitivity; 1 without a profile), apportioned by parameter share, scaled by how much
    noise that atom leaves in this group relative to the model's typical FP4 noise. Greedy on the lower convex
    hull: the best error drop per extra bit first. `focus` layers are forced to the top atom first. `fp6`
    "auto" keeps only the better FP6 element per group."""
    sens = {str(k): float(v) for k, v in (sens or {}).items()}
    fp4 = [a for a in atoms if ATOMS[a]["bpw"] < 5]
    if len(fp4) != 1:
        raise ValueError("a mix needs exactly one FP4 atom (mxfp4 or nvfp4)")
    fp4 = fp4[0]
    n_layer = {}
    for g in groups.values():
        n_layer[g["layer"]] = n_layer.get(g["layer"], 0) + g["n"]
    rel = {k: {a: g["err"][a] / max(g["w2"], 1e-30) for a in atoms} for k, g in groups.items()}
    ref = float(np.mean([r[fp4] for r in rel.values()])) or 1.0
    smean = float(np.mean(list(sens.values()))) if sens else 1.0
    total = sum(g["n"] for g in groups.values())
    if target_bpw < ATOMS[fp4]["bpw"]:
        raise SystemExit(f"--mix {target_bpw}: below the FP4 floor ({ATOMS[fp4]['bpw']} bpw with scales)")
    budget = target_bpw * total
    choice, hulls, heap = {}, {}, []
    for k, g in groups.items():
        s = sens.get(str(g["layer"]), smean) / (smean or 1.0)
        share = g["n"] / max(n_layer[g["layer"]], 1)
        cand = list(atoms)
        if fp6 == "auto":
            six = [a for a in atoms if a.startswith("mxfp6")]
            if len(six) > 1:
                worse = max(six, key=lambda a: rel[k][a])
                cand = [a for a in atoms if a != worse]
        pts = [(ATOMS[a]["bpw"], s * share * rel[k][a] / ref, a) for a in cand]
        hulls[k] = _hull(pts)
        choice[k] = 0
        budget -= hulls[k][0][0] * g["n"]
    for k in groups:
        if groups[k]["layer"] in set(focus):
            while choice[k] + 1 < len(hulls[k]):
                b0, b1 = hulls[k][choice[k]][0], hulls[k][choice[k] + 1][0]
                budget -= (b1 - b0) * groups[k]["n"]
                choice[k] += 1
    for k, g in groups.items():
        i = choice[k]
        if i + 1 < len(hulls[k]):
            (b0, c0, _), (b1, c1, _) = hulls[k][i], hulls[k][i + 1]
            heapq.heappush(heap, (-(c0 - c1) / ((b1 - b0) * g["n"]), k, i))
    while heap:
        _, k, i = heapq.heappop(heap)
        if choice[k] != i:
            continue
        g = groups[k]
        (b0, _, _), (b1, _, _) = hulls[k][i], hulls[k][i + 1]
        extra = (b1 - b0) * g["n"]
        if extra > budget + 1e-6:
            continue
        budget -= extra
        choice[k] = i + 1
        if i + 2 < len(hulls[k]):
            (b0, c0, _), (b1, c1, _) = hulls[k][i + 1], hulls[k][i + 2]
            heapq.heappush(heap, (-(c0 - c1) / ((b1 - b0) * g["n"]), k, i + 1))
    return {k: hulls[k][choice[k]][2] for k in groups}


def avg_bpw(groups, assign):
    tot = sum(g["n"] for g in groups.values())
    return sum(ATOMS[assign[k]]["bpw"] * g["n"] for k, g in groups.items()) / max(tot, 1)


# ----------------------------------------------------------------------------------------------- configs

def quark_spec(atom):
    return {"weight": {"dtype": QUARK_DTYPE[atom], "qscheme": "per_group", "group_size": BLOCK, "ch_axis": -1,
                       "scale_format": "e8m0", "is_dynamic": False, "symmetric": True, "round_method": "half_even",
                       "mx_element_dtype": None},
            "input_tensors": None, "output_tensors": None, "bias": None}


def _glob_for(mod):
    """A Quark layer_quant_config key for one module that survives a VLM name prefix (vLLM matches with fnmatch):
    `*.layers.3.mlp.up_proj` -- the dot before `layers` keeps layer 3 from matching layer 13."""
    i = mod.find(".layers.")
    return "*" + mod[i:] if i >= 0 else mod


def _ignore_re(mod):
    return "re:.*" + re.escape(mod[mod.find(".layers."):] if ".layers." in mod else mod) + "$"


def quark_config(module_atoms, ignore_mods):
    from collections import Counter
    common = Counter(module_atoms.values()).most_common(1)[0][0]
    import pollard_mx as X
    layer = {_glob_for(m): quark_spec(a) for m, a in sorted(module_atoms.items()) if a != common}
    return {"quant_method": "quark", "export": {"kv_cache_group": [], "pack_method": "reorder"},
            "exclude": ["lm_head", "re:.*embed_tokens$"] + X.ROUTER_GLOBS + X.VISION_GLOBS + X.PROJECTOR_GLOBS
            + [_ignore_re(m) for m in sorted(ignore_mods)],
            "global_quant_config": quark_spec(common), "layer_type_quant_config": {},
            "layer_quant_config": layer, "producer": {"name": "pollard-mx"}}


CT_WEIGHTS = {
    "mxfp4": ({"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32, "scale_dtype": "torch.uint8",
               "zp_dtype": "torch.uint8"}, "mxfp4-pack-quantized"),
    "nvfp4": ({"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16,
               "scale_dtype": "torch.float8_e4m3fn", "zp_dtype": "torch.float8_e4m3fn"}, "nvfp4-pack-quantized"),
    "mxfp6_e2m3": ({"num_bits": 6, "type": "float", "strategy": "group", "group_size": 32, "scale_dtype": "torch.uint8",
                    "zp_dtype": "torch.uint8"}, "mxfp6-pack-quantized"),
    "mxfp8": ({"num_bits": 8, "type": "float", "strategy": "group", "group_size": 32, "scale_dtype": "torch.uint8",
               "zp_dtype": "torch.uint8"}, "mxfp8-quantized"),
}


def ct_config(module_atoms, ignore_mods):
    """compressed-tensors: one config group per atom; `mixed-precision` when there is more than one."""
    import pollard_mx as X
    if "mxfp6_e3m2" in module_atoms.values():
        raise ValueError("compressed-tensors has no MXFP6 E3M2 (its num_bits=6 float is E2M3): use --layout quark")
    used = sorted(set(module_atoms.values()), key=lambda a: ATOMS[a]["bpw"])
    groups = {}
    for i, a in enumerate(used):
        w, fmt = CT_WEIGHTS[a]
        mods = sorted(m for m, x in module_atoms.items() if x == a)
        groups[f"group_{i}"] = {"targets": ["Linear"] if len(used) == 1 else mods,
                                "weights": {**w, "symmetric": True, "dynamic": False, "actorder": None,
                                            "block_structure": None, "observer": None, "observer_kwargs": {}},
                                "input_activations": None, "output_activations": None, "format": fmt}
    return {"quant_method": "compressed-tensors", "quantization_status": "compressed",
            "format": CT_WEIGHTS[used[0]][1] if len(used) == 1 else "mixed-precision",
            "config_groups": groups, "kv_cache_scheme": None, "global_compression_ratio": None,
            "ignore": ["lm_head", "re:.*embed_tokens$"] + X.ROUTER_GLOBS + X.VISION_GLOBS + X.PROJECTOR_GLOBS
            + sorted(ignore_mods), "producer": "pollard-mx"}


def pick_layout(layout, atoms_used, mix):
    if layout != "auto":
        return layout
    if not mix and set(atoms_used) == {"mxfp6_e2m3"}:
        return "compressed-tensors"
    return "quark"


# ----------------------------------------------------------------------------------------------- runtimes

def runtimes(layout, atoms_used):
    """Which runtimes load this rung, as of 2026-10-15 -- [(runtime, status, how)]. status: yes | pending | no."""
    used = set(atoms_used)
    out = []
    if layout == "quark":
        out.append(("vLLM", "yes", "vLLM main, Quark OCP-MX scheme (`vllm serve <dir>`; MXFP6 via the Humming kernel "
                                   "on CUDA where it applies, else emulation, which needs `pip install amd-quark`)"))
        out.append(("Graft", "no", "Graft's importer reads compressed-tensors MX configs, not Quark ones"))
    else:
        out.append(("vLLM", "pending", "compressed-tensors MXFP6 loads once vllm-project/vllm#61666 merges"
                    if "mxfp6_e2m3" in used else "compressed-tensors MX/NVFP4 schemes"))
        if used == {"mxfp6_e2m3"}:
            out.append(("Graft", "yes", "`graft repack <dir>` -> MXFP6_E2M3 (type 406), byte copy, canary max diff 0; "
                                        "then `graft serve -m <dir>.MXFP6_E2M3.graft.gguf`"))
        else:
            out.append(("Graft", "no", "Graft's importer reads one uniform MX format per checkpoint, not "
                                       "`mixed-precision`"))
        out.append(("compressed-tensors", "yes" if used <= {"mxfp6_e2m3", "mxfp8", "mxfp4", "nvfp4"} else "no",
                    "main (mxfp6-pack-quantized: vllm-project/compressed-tensors#941)"))
    return out


def graft_ok(meta):
    return any(r == "Graft" and s == "yes" for r, s, _ in (tuple(x) for x in meta.get("runtimes", [])))


# ----------------------------------------------------------------------------------------------- build

def build(model_dir, out, fmt=None, mix=None, layout="auto", fp6="auto", fp4=None, rule="search", importance=None,
          sens=None, focus=(), quantize_vision=False, log=print, plan_only=False):
    """Encode `model_dir` (an HF bf16/fp16 checkpoint) into an MX rung at `out`. Returns the sidecar dict."""
    if (fmt is None) == (mix is None):
        raise ValueError("exactly one of fmt / mix")
    lin, skipped = scan(model_dir, quantize_vision)
    if not lin:
        raise SystemExit(f"no quantizable linears in {model_dir}")
    if fmt:
        atom = FORMATS[fmt]
        lay = pick_layout(layout, [atom], False)
        if lay == "compressed-tensors" and atom == "mxfp6_e3m2":
            raise SystemExit("compressed-tensors has no MXFP6 E3M2 (num_bits=6 float is E2M3): use --layout quark")
        groups = {}
        for sh, name, mod, shape, dt in lin:
            g = groups.setdefault(fused_key(mod), {"n": 0, "layer": layer_of(mod), "members": []})
            g["n"] += int(np.prod(shape))
            g["members"].append(mod)
        assign = {k: atom for k in groups}
    else:
        lay0 = layout if layout != "auto" else "quark"
        f4 = fp4 or ("nvfp4" if lay0 == "compressed-tensors" else "mxfp4")
        if lay0 == "quark" and f4 == "nvfp4":
            raise SystemExit("NVFP4 is a compressed-tensors scheme here: use --layout compressed-tensors or --fp4 mxfp4")
        six = {"auto": ["mxfp6_e2m3", "mxfp6_e3m2"], "e2m3": ["mxfp6_e2m3"], "e3m2": ["mxfp6_e3m2"]}[fp6]
        if lay0 == "compressed-tensors":
            six = ["mxfp6_e2m3"]
        atoms = [f4] + six + ["mxfp8"]
        log(f"   measuring {len(lin)} linears at {', '.join(atoms)} ({'imatrix-weighted' if importance else 'MSE'})")
        groups = measure_errors(model_dir, lin, atoms, importance, rule, log)
        assign = allocate_mix(groups, float(mix), atoms, sens, focus, fp6)
        lay = pick_layout(layout, set(assign.values()), True)
    module_atoms = {m: assign[k] for k, g in groups.items() for m in g["members"]}
    used = sorted(set(module_atoms.values()), key=lambda a: ATOMS[a]["bpw"])
    q_params = sum(g["n"] for g in groups.values())
    bpw = avg_bpw(groups, assign)
    hist = {a: sum(g["n"] for k, g in groups.items() if assign[k] == a) for a in used}
    meta = {"format": fmt or "mix", "target_bpw": None if mix is None else float(mix), "layout": lay,
            "bpw": round(bpw, 4), "quantized_params": q_params,
            "atoms": {a: round(n / q_params, 4) for a, n in hist.items()},
            "modules": module_atoms, "scale_rule": rule, "weight_only": True,
            "skipped": dict(sorted(skipped)), "runtimes": [list(r) for r in runtimes(lay, used)],
            "spec": "OCP Microscaling Formats v1.0 (32-element blocks, E8M0 scale); FP6 4-in-3 LSB-first pack",
            "producer": "pollard-mx"}
    log(f"   {meta['format']} -> {lay}: {bpw:.3f} bpw over {q_params / 1e6:.1f}M quantized params  "
        + "  ".join(f"{a} {100 * v:.1f}%" for a, v in meta["atoms"].items()))
    if plan_only:
        return meta
    os.makedirs(out, exist_ok=True)
    nv_global = {}
    if "nvfp4" in used:                                  # one tensor scale per fused NVFP4 layer
        for k, g in groups.items():
            if assign[k] == "nvfp4":
                amax = 0.0
                for sh, name, mod, shape, dt in lin:
                    if mod in g["members"]:
                        amax = max(amax, float(np.abs(to_f32(*st_read(sh, name))).max()))
                nv_global[k] = 448.0 * 6.0 / (amax or 1.0)
    by_name = {name: (mod, shape) for sh, name, mod, shape, dt in lin}
    weight_map = {}
    for sh in source_shards(model_dir):
        hdr, _ = st_header(sh)
        meta_in = hdr.get("__metadata__") or {}
        tensors = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            dt, arr = st_read(sh, name, (hdr, _))
            if name not in by_name:
                tensors[name] = (dt, arr)
                continue
            mod, shape = by_name[name]
            atom = module_atoms[mod]
            w = to_f32(dt, arr)
            imp = _imp_for(importance, mod, shape[1])
            parts, _dec = encode_atom(w, atom, imp, rule, nv_global.get(fused_key(mod)))
            for suf, (sdt, a) in parts.items():
                if lay == "compressed-tensors" and suf == "weight" and atom != "mxfp8":
                    suf = "weight_packed"
                tensors[f"{mod}.{suf}"] = (sdt, a)
        dst = os.path.join(out, os.path.basename(sh))
        st_write(dst, tensors, {**meta_in, "format": "pt"})
        weight_map.update({n: os.path.basename(sh) for n in tensors})
    if os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        json.dump({"metadata": {}, "weight_map": weight_map},
                  open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    for f in os.listdir(model_dir):
        p = os.path.join(model_dir, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f != "model.safetensors.index.json" \
                and f != SIDECAR and not f.endswith(".tmp"):
            shutil.copy2(p, os.path.join(out, f))
    cfgp = os.path.join(out, "config.json")
    cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    ign = {m for m, _ in skipped}
    cfg["quantization_config"] = quark_config(module_atoms, ign) if lay == "quark" else ct_config(module_atoms, ign)
    json.dump(cfg, open(cfgp, "w"), indent=2)
    json.dump(meta, open(os.path.join(out, SIDECAR), "w"), indent=1)
    return meta


# ----------------------------------------------------------------------------------------------- read back

def read_meta(path):
    if not path or not os.path.isdir(str(path)):
        return None
    try:
        return json.load(open(os.path.join(path, SIDECAR)))
    except (OSError, ValueError):
        return None


def dequantized_weights(rung_dir):
    """{module.weight: float32 [N, K]} for every quantized linear of a Pollard MX rung, decoded from its bytes."""
    meta = read_meta(rung_dir)
    if not meta:
        raise SystemExit(f"{rung_dir} has no {SIDECAR}: not a pollard-mx MXFP6 / mix rung")
    mods = meta["modules"]
    found = {}
    for sh in source_shards(rung_dir):
        hdr = st_header(sh)
        for name in hdr[0]:
            if name == "__metadata__":
                continue
            mod, _, suf = name.rpartition(".")
            if mod in mods:
                dt, arr = st_read(sh, name, hdr)
                found.setdefault(mod, {})[suf] = arr
    out = {}
    for mod, atom in mods.items():
        if mod not in found:
            raise SystemExit(f"{rung_dir}: {mod} is in {SIDECAR} but not in the shards")
        out[mod + ".weight"] = decode_atom(found[mod], atom)
    return out


def load_hf_model(rung_dir, device="cpu", trust_remote_code=False):
    """A transformers model of this rung WITHOUT any quantization loader: the architecture from its config, every
    unquantized tensor as stored, every quantized linear decoded exactly from the rung's bytes (weight-only, so this is
    the math a runtime does with them). How the tool gate and pollard-verify run a rung where no MXFP6 kernel is."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(rung_dir, trust_remote_code=trust_remote_code)
    if hasattr(cfg, "quantization_config"):
        delattr(cfg, "quantization_config")
    dtype = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None) or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, trust_remote_code=trust_remote_code)
    except TypeError:                                    # transformers 4.x spelling
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, trust_remote_code=trust_remote_code)
    sd = model.state_dict()
    mods = set((read_meta(rung_dir) or {}).get("modules", {}))
    seen = set()

    def put(name, t):
        key = name if name in sd else next((k for k in sd if k.endswith("." + name.split(".", 1)[-1])), None)
        if key is None:
            return
        sd[key].copy_(t.to(sd[key].dtype))
        seen.add(key)
    for sh in source_shards(rung_dir):
        hdr = st_header(sh)
        for name in hdr[0]:
            mod, _, suf = name.rpartition(".")
            if name == "__metadata__" or (mod in mods and suf.startswith("weight")):
                continue
            dt, arr = st_read(sh, name, hdr)
            if dt == "BF16":
                t = torch.from_numpy(arr.astype(np.int16, copy=False).copy()).view(torch.bfloat16)
            elif dt in ("F32", "F16"):
                t = torch.from_numpy(arr.copy())
            else:
                continue
            put(name, t)
    for name, w in dequantized_weights(rung_dir).items():
        put(name, torch.from_numpy(w))
    missing = [k for k in sd if k not in seen and not k.endswith("lm_head.weight") and "rotary" not in k]
    if missing:
        raise SystemExit(f"{rung_dir}: no tensor for {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
    elif not any(k.endswith("lm_head.weight") for k in seen) and any(k.endswith("lm_head.weight") for k in sd):
        raise SystemExit(f"{rung_dir}: no lm_head tensor and the config does not tie it")
    return model.to(device).eval()


def vllm_load_problem(model_dir):
    """Why the vLLM in THIS environment cannot load this rung (None when it should, or it is not ours)."""
    meta = read_meta(model_dir)
    if not meta:
        return None
    if meta.get("layout") == "compressed-tensors" and any(a.startswith("mxfp6") for a in meta.get("atoms", {})):
        try:
            from compressed_tensors.config import CompressionFormat
            if not any(str(getattr(f, "value", f)) == "mxfp6-pack-quantized" for f in CompressionFormat):
                raise ImportError
        except Exception:                                # noqa: BLE001
            return ("this compressed-tensors has no mxfp6-pack-quantized (needs compressed-tensors#941 + "
                    "vllm#61666); serve the quark-layout rung instead")
    return None


def serve_lines(meta, path):
    out = []
    for rt, status, how in (tuple(x) for x in meta.get("runtimes", [])):
        if rt == "vLLM" and status == "yes":
            out.append(f"vllm serve {path}")
        elif rt == "Graft" and status == "yes":
            out.append(f"graft repack {path} && graft serve -m <the .graft.gguf it writes>")
    return out


if __name__ == "__main__":
    sys.exit("run this through pollard-mx --format mxfp6-e3m2|mxfp6-e2m3 or pollard-mx --mix <bpw>")
