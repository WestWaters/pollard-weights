#!/usr/bin/env python3
"""pollard-fq -- bit-exact fake quantization for pollard-qat.

Torch ports of ggml's ENCODERS and decoders, so a QAT run trains against exactly the weights the GGUF
will hold: `quant_dequant(W, "iq2_xxs", imatrix)` returns, bit for bit, what
`dequantize(ggml_quantize_chunk(IQ2_XXS, W, imatrix))` returns from the runtime's own libggml. The
STQ1_0 lesson (experiments/qat_stq1): a fake-quant that only matches the format's statistics trains the
model for a quantizer it will never meet.

    import pollard_fq as fq
    Wq = fq.fake_quant(W, "q2_K", imatrix)          # STE: forward = quantized, grad -> W
    lin = fq.FakeQuantLinear(linear, "iq2_xs", imatrix)

Types and what "bit-exact" was measured against (tests/test_fq_parity.py, the HARD GATE):
  q2_K, q3_K       the imatrix encoders (quantize_row_q{2,3}_K_impl) -- an imatrix is required, as it is
                   for every Pollard build; the imatrix-free *_ref encoders are not ported
  tq2_0            quantize_row_tq2_0_ref (ignores the imatrix, as ggml does)
  iq2_xxs, iq2_xs  quantize_row_iq2_{xxs,xs}_impl (imatrix required, as in ggml)
  iq2_s            quantize_row_iq2_s_impl, with or without an imatrix
  stq1_0           Pollard's patched quantize_row_stq1_0_imatrix (runtime-patches/llama-stq-stq1_0-quants.patch)
`SUPPORTED` lists the types whose parity gate passed; anything else raises rather than approximates.

Exactness notes -- the C is float32 and sequential, and its rounding depends on how the compiler
lowered it. With clang's default -ffp-contract=on an `a*b + c` inside one expression MAY be fused, and the
vectoriser decides per loop: an element-wise contraction stays a fused fmla, but a reduction it vectorises
(`acc += w*x` over 16 or 32 lanes) becomes rounded products added in order. The port reproduces each site
as the runtime build compiled it -- `_fma` (one rounding, via the exact float64 product) where the build
fuses, `_mad` where it does not, every reduction a sequential loop (vectorised across blocks, never a
tree sum), `nearest_int` as round-half-even. The per-site pattern was read off byte-identical probe
builds of ggml-quants.c (Apple clang 21, -O3, arm64 -- the runtime's own flags) and is pinned by the
parity gate. A runtime built WITHOUT contraction (MSVC's default, gcc -ffp-contract=off) fuses nothing:
set POLLARD_FQ_CONTRACT=0 and re-run tests/test_fq_parity.py against that build. A different compiler
that fuses differently is caught by the same gate -- never assumed.

Grids: IQ2 dequant grids are read from the pinned runtime's ggml/src/ggml-common.h and the encoder
grids (kgrid_2bit_*) from ggml-quants.c -- POLLARD_GGML_SRC overrides the directory. They are
cross-checked against each other on load.
"""
import functools
import os
import re

QK_K = 256
CHUNK_BLOCKS = 2048          # super-blocks encoded per pass (~0.5M weights): peak RSS well under 2 GB
CONTRACT = os.environ.get("POLLARD_FQ_CONTRACT", "1") != "0"
# filled in from the parity gate (tests/test_fq_parity.py). A type is listed only once its
# fq == dequant(quantize) check passed bit-exactly against the runtime's libggml.
SUPPORTED = ("q2_k", "q3_k", "tq2_0", "iq2_xxs", "iq2_xs", "iq2_s", "stq1_0")
NEEDS_IMATRIX = ("q2_k", "q3_k", "iq2_xxs", "iq2_xs")


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit("pollard-fq needs torch:  pip install 'pollard-weights[forge]'")


# ------------------------------------------------------------------------------------------------
# float32 primitives that behave like the C
# ------------------------------------------------------------------------------------------------
def _c(v):
    """A C float literal / int-to-float conversion, as a float32 scalar tensor."""
    torch = _torch()
    return torch.tensor(float(v), dtype=torch.float32)


def _fma(a, b, c):
    """fmaf(a, b, c) when contraction is on (float64 holds the exact float32 product; one rounding
    to float32 at the end), else a*b + c with the product rounded first."""
    torch = _torch()
    a, b, c = (x if isinstance(x, torch.Tensor) else _c(x) for x in (a, b, c))
    if CONTRACT:
        return (a.double() * b.double() + c.double()).float()
    return a.float() * b.float() + c.float()


def _rne(x):
    """nearest_int(): (x + 1.5*2^23) as float32 -> round half to even, as an int64 tensor."""
    return _torch().round(x).to(_torch().int64)


def _sqrtf(x):
    """sqrtf(x), correctly rounded on every platform. torch.sqrt on float32 is not: x86 wheels route it
    through MKL's vector sqrt (accurate to < 1 ulp, not correctly rounded), so linux/amd64 returned values
    1 ulp off the C. Take the root in float64, round once to float32, then settle the last bit exactly: a
    float32 midpoint has 25 significant bits, so its square is exact in float64 and comparing it against x
    picks the correctly rounded neighbour whatever the library's sqrt accuracy (no ties are possible)."""
    torch = _torch()
    xd = x.double()
    r = torch.sqrt(xd).float()
    up = torch.nextafter(r, torch.full_like(r, float("inf")))
    dn = torch.nextafter(r, torch.zeros_like(r))
    rd = r.double()
    hi = (rd + up.double()) * 0.5
    lo = (rd + dn.double()) * 0.5
    r = torch.where(hi * hi < xd, up, r)
    r = torch.where((r > 0) & (lo * lo > xd), dn, r)
    return r


def _fp16(x):
    return x.to(_torch().float16).to(_torch().float32)


def _maxc(a, b):
    """MAX(a, b) = a > b ? a : b."""
    return _torch().where(a > b, a, b)


# ------------------------------------------------------------------------------------------------
# shared encoder helpers (ggml-quants.c)
# ------------------------------------------------------------------------------------------------
def _mad(a, b, c):
    """a*b + c where clang vectorised the loop: the products are taken 4-wide (fmul.4s), the sum is
    accumulated in order -- two roundings, no fusion, even with contraction on."""
    return a * b + c


def _sumsq(x):
    """for (i < QK_K) sumx2 += x[i]*x[i];  -- a vectorised loop: products rounded, adds in order."""
    torch = _torch()
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32)
    sq = x * x
    for i in range(x.shape[-1]):
        acc = acc + sq[..., i]
    return acc


# make_qx_quants is inlined (twice, n = 16) into quantize_q3_K: the first sums loop is vectorised (rounded
# products, in-order adds), the +-9 search loops stay scalar and fused. Measured on a byte-identical probe.
_QXF = {None: dict(lx0=False, l20=False, lx=True, l2=True, isc=True)}


def _make_qx_quants(x, nmax, qw):
    """make_qx_quants(n, nmax, x, L, rmse_type=1, qw) over rows [G, n]. Returns (scale [G], L [G, n] in
    0..2*nmax-1)."""
    torch = _torch()
    G, n = x.shape
    amax = torch.zeros(G, dtype=torch.float32)
    mx = torch.zeros(G, dtype=torch.float32)
    for i in range(n):
        ax = x[:, i].abs()
        up = ax > amax
        amax = torch.where(up, ax, amax)
        mx = torch.where(up, x[:, i], mx)
    zero = amax < _c(1e-15)
    safe = torch.where(zero, torch.ones_like(mx), mx)
    iscale = _c(-nmax) / safe

    fz = _QXF.get(nmax, _QXF[None])

    def sums(iscale, fa, fb):
        sumlx = torch.zeros(G, dtype=torch.float32)
        suml2 = torch.zeros(G, dtype=torch.float32)
        Ls = []
        for i in range(n):
            l = _rne(iscale * x[:, i]).clamp(-nmax, nmax - 1)
            lf = l.to(torch.float32)
            w = qw[:, i]
            sumlx = (_fma if fa else _mad)(w * x[:, i], lf, sumlx)
            suml2 = (_fma if fb else _mad)(w * lf, lf, suml2)
            Ls.append(l + nmax)
        return sumlx, suml2, torch.stack(Ls, 1)

    sumlx, suml2, L = sums(iscale, fz['lx0'], fz['l20'])
    scale = torch.where(suml2 != 0, sumlx / torch.where(suml2 != 0, suml2, _c(1)), _c(0))
    best = scale * sumlx
    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        isc = -(_fma if fz['isc'] else _mad)(_c(0.1), _c(is_), _c(nmax)) / safe
        slx, sl2, Lc = sums(isc, fz['lx'], fz['l2'])
        up = (sl2 > 0) & (slx * slx > best * sl2)
        L = torch.where(up[:, None], Lc, L)
        new_scale = slx / torch.where(sl2 > 0, sl2, _c(1))
        scale = torch.where(up, new_scale, scale)
        best = torch.where(up, new_scale * slx, best)
    scale = torch.where(zero, _c(0), scale)
    L = torch.where(zero[:, None], torch.zeros_like(L), L)
    return scale, L


# make_qkx3_quants is inlined into quantize_row_q2_K_impl with n = 16 and fully unrolled; clang keeps
# every contraction fused EXCEPT the three is-loop sums (sum_l / sum_l2 / sum_xl), which it computes as
# rounded products added in order. Found against a byte-identical probe build of the C (the gate test).
_QF = {'sum_x': True, 'diff': True, 'best': True, 'sums': False}


def _make_qkx3_quants(x, w, nmax, rmin, rdelta, nstep, use_mad):
    """make_qkx3_quants over rows [G, n] (weights given). Returns (scale, the_min, L)."""
    torch = _torch()
    G, n = x.shape
    mn = x[:, 0].clone()
    mx = x[:, 0].clone()
    sum_w = w[:, 0].clone()
    sum_x = sum_w * x[:, 0]
    for i in range(1, n):
        mn = torch.where(x[:, i] < mn, x[:, i], mn)
        mx = torch.where(x[:, i] > mx, x[:, i], mx)
        sum_w = sum_w + w[:, i]
        sum_x = (_fma if _QF['sum_x'] else _mad)(w[:, i], x[:, i], sum_x)
    mn = torch.where(mn > 0, _c(0), mn)
    degenerate = mx <= mn
    rng = torch.where(degenerate, _c(1), mx - mn)
    iscale = _c(nmax) / rng
    scale = _c(1) / iscale
    best = torch.zeros(G, dtype=torch.float32)
    Ls = []
    for i in range(n):
        l = _rne(iscale * (x[:, i] - mn)).clamp(0, nmax)
        Ls.append(l)
        diff = (_fma if _QF['diff'] else _mad)(scale, l.to(torch.float32), mn) - x[:, i]
        diff = diff.abs() if use_mad else diff * diff
        best = (_fma if _QF['best'] else _mad)(w[:, i], diff, best)
    L = torch.stack(Ls, 1)
    for is_ in range(nstep + 1):
        # `min` moves when a step wins, and both the range and the offset below follow it (as in the C)
        rng = torch.where(degenerate, _c(1), mx - mn)
        isc = (_fma(_c(rdelta), _c(is_), _c(rmin)) + _c(nmax)) / rng
        sum_l = torch.zeros(G, dtype=torch.float32)
        sum_l2 = torch.zeros(G, dtype=torch.float32)
        sum_xl = torch.zeros(G, dtype=torch.float32)
        La = []
        for i in range(n):
            l = _rne(isc * (x[:, i] - mn)).clamp(0, nmax)
            La.append(l)
            lf = l.to(torch.float32)
            wl = w[:, i] * lf
            f = _fma if _QF['sums'] else _mad
            sum_l = f(w[:, i], lf, sum_l)
            sum_l2 = f(wl, lf, sum_l2)
            sum_xl = f(wl, x[:, i], sum_xl)
        Laux = torch.stack(La, 1)
        D = _fma(sum_w, sum_l2, -(sum_l * sum_l))
        ok = D > 0
        Ds = torch.where(ok, D, _c(1))
        this_scale = _fma(sum_w, sum_xl, -(sum_x * sum_l)) / Ds
        this_min = _fma(sum_l2, sum_x, -(sum_l * sum_xl)) / Ds
        pos = this_min > 0
        this_min = torch.where(pos, _c(0), this_min)
        this_scale = torch.where(pos, sum_xl / torch.where(sum_l2 != 0, sum_l2, _c(1)), this_scale)
        mad = torch.zeros(G, dtype=torch.float32)
        for i in range(n):
            diff = (_fma if _QF['diff'] else _mad)(this_scale, Laux[:, i].to(torch.float32), this_min) - x[:, i]
            diff = diff.abs() if use_mad else diff * diff
            mad = (_fma if _QF['best'] else _mad)(w[:, i], diff, mad)
        up = ok & (mad < best) & ~degenerate
        L = torch.where(up[:, None], Laux, L)
        best = torch.where(up, mad, best)
        scale = torch.where(up, this_scale, scale)
        mn = torch.where(up, this_min, mn)
    scale = torch.where(degenerate, _c(0), scale)
    L = torch.where(degenerate[:, None], torch.zeros_like(L), L)
    return scale, -mn, L


def _make_qp_quants(x, nmax, qw):
    """make_qp_quants over rows [G, n]; L is uint8 in the C (negative values wrap), so it is here too.
    Returns (scale [G], L [G, n]).

    ggml calls this one out of line (generic n), and clang vectorises its three error/sum loops for
    n % 16 == 0: the per-element `x - scale*L` stays a fused fmla, but the accumulations become a 4-wide
    product followed by in-order scalar adds -- unfused. The itry refinement loop is scalar and fused."""
    torch = _torch()
    G, n = x.shape
    if n % 16:
        raise ValueError("make_qp_quants port covers n % 16 == 0 (the vectorised path ggml takes for 16/32)")
    mx = torch.zeros(G, dtype=torch.float32)
    for i in range(n):
        mx = _maxc(mx, x[:, i])
    zero = mx < _c(1e-15)
    safe = torch.where(zero, _c(1), mx)
    iscale = _c(nmax) / safe
    L0 = [(_rne(iscale * x[:, i])) & 0xFF for i in range(n)]
    scale = _c(1) / iscale
    best = torch.zeros(G, dtype=torch.float32)
    for i in range(n):
        diff = _fma(-scale, L0[i].to(torch.float32), x[:, i])
        best = _mad(qw[:, i] * diff, diff, best)
    for is_ in range(-4, 5):
        if is_ == 0:
            continue
        isi = _fma(_c(0.1), _c(is_), _c(nmax)) / safe
        sci = _c(1) / isi
        mse = torch.zeros(G, dtype=torch.float32)
        for i in range(n):
            l = _rne(isi * x[:, i]).clamp(max=nmax)
            diff = _fma(-sci, l.to(torch.float32), x[:, i])
            mse = _mad(qw[:, i] * diff, diff, mse)
        up = mse < best
        best = torch.where(up, mse, best)
        iscale = torch.where(up, isi, iscale)
    sumlx = torch.zeros(G, dtype=torch.float32)
    suml2 = torch.zeros(G, dtype=torch.float32)
    L = []
    for i in range(n):
        l = _rne(iscale * x[:, i]).clamp(max=nmax)
        lf = l.to(torch.float32)
        w = qw[:, i]
        sumlx = _mad(w * x[:, i], lf, sumlx)
        suml2 = _mad(w * lf, lf, suml2)
        L.append(l & 0xFF)
    active = torch.ones(G, dtype=torch.bool)
    for _ in range(5):
        changed = torch.zeros(G, dtype=torch.bool)
        for i in range(n):
            w = qw[:, i]
            Li = L[i].to(torch.float32)
            wx = w * x[:, i]
            slx = _fma(-wx, Li, sumlx)
            sl2 = _fma(-(w * Li), Li, suml2)
            c1 = (slx > 0) & (sl2 > 0)
            new_l = _rne((x[:, i] * sl2) / torch.where(c1, slx, _c(1))).clamp(max=nmax)
            c2 = new_l != L[i]
            nl = new_l.to(torch.float32)
            slx2 = _fma(wx, nl, slx)
            sl22 = _fma(w * nl, nl, sl2)
            c3 = (slx2 * slx2) * suml2 > (sumlx * sumlx) * sl22
            up = active & c1 & c2 & c3
            L[i] = torch.where(up, new_l & 0xFF, L[i])
            sumlx = torch.where(up, slx2, sumlx)
            suml2 = torch.where(up, sl22, suml2)
            changed = changed | up
        active = active & changed
        if not bool(active.any()):
            break
    out = torch.where(suml2 > 0, sumlx / torch.where(suml2 > 0, suml2, _c(1)), _c(0))
    out = torch.where(zero, _c(0), out)
    L = torch.stack(L, 1)
    L = torch.where(zero[:, None], torch.zeros_like(L), L)
    return out, L


def _blocks(W, imatrix):
    """[rows, n] -> x [B, 256] (row-major blocks) and the matching importance [B, 256] (or None)."""
    torch = _torch()
    rows, n = W.shape
    if n % QK_K:
        raise ValueError(f"row length {n} is not a multiple of {QK_K}")
    x = W.reshape(rows * (n // QK_K), QK_K)
    qw = None
    if imatrix is not None:
        im = torch.as_tensor(imatrix, dtype=torch.float32).reshape(n // QK_K, QK_K)
        qw = im.repeat(rows, 1)
    return x, qw


# ------------------------------------------------------------------------------------------------
# Q2_K (quantize_row_q2_K_impl + dequantize_row_q2_K)
# ------------------------------------------------------------------------------------------------
def _q2_k(x, qw):
    torch = _torch()
    B = x.shape[0]
    sigma2 = _sumsq(x) / _c(QK_K)
    xs = x.reshape(B, 16, 16)
    weight = qw.reshape(B, 16, 16) * _sqrtf(_fma(xs, xs, sigma2[:, None, None]))
    sw = torch.zeros(B, 16, dtype=torch.float32)
    for l in range(16):
        sw = sw + weight[:, :, l]
    scales, mins, L = _make_qkx3_quants(xs.reshape(-1, 16), weight.reshape(-1, 16), 3, -0.9, 0.05, 36, False)
    scales, mins, L = scales.reshape(B, 16), mins.reshape(B, 16), L.reshape(B, 16, 16)
    dm, Ls = _make_qp_quants(scales, 15, sw)
    mm, Lm = _make_qp_quants(mins, 15, sw)
    d16, m16 = _fp16(dm), _fp16(mm)
    sc = (Ls | (Lm << 4)) & 0xFF
    lo, hi = (sc & 0xF).to(torch.float32), (sc >> 4).to(torch.float32)
    d = d16[:, None] * lo
    m = m16[:, None] * hi
    ok = d != 0
    lq = _rne((xs + m[..., None]) / torch.where(ok, d, _c(1))[..., None]).clamp(0, 3)
    L = torch.where(ok[..., None], lq, L)
    # dequantize_row_q2_K: y = dl * q - ml  (contracted)
    dl, ml = d16[:, None] * lo, m16[:, None] * hi
    y = _fma(dl[..., None], L.to(torch.float32), -ml[..., None])
    return y.reshape(B, QK_K)


# ------------------------------------------------------------------------------------------------
# Q3_K (quantize_row_q3_K_impl + dequantize_row_q3_K)
# ------------------------------------------------------------------------------------------------
def _q3_k(x, qw):
    torch = _torch()
    B = x.shape[0]
    sigma2 = (_c(2) * _sumsq(x)) / _c(QK_K)
    xs = x.reshape(B, 16, 16)
    weight = qw.reshape(B, 16, 16) * _sqrtf(_fma(xs, xs, sigma2[:, None, None]))
    sw = torch.zeros(B, 16, dtype=torch.float32)
    for l in range(16):
        sw = sw + weight[:, :, l]
    scales, L = _make_qx_quants(xs.reshape(-1, 16), 4, weight.reshape(-1, 16))
    scales, L = scales.reshape(B, 16), (L - 4).reshape(B, 16, 16)
    d_block, Ls = _make_qx_quants(scales, 32, sw)
    d16 = _fp16(d_block)
    sc = (Ls - 32).to(torch.float32)
    d = d16[:, None] * sc
    ok = d != 0
    lq = _rne(xs / torch.where(ok, d, _c(1))[..., None]).clamp(-4, 3)
    L = torch.where(ok[..., None], lq, L)
    y = (d16[:, None] * sc)[..., None] * L.to(torch.float32)
    return y.reshape(B, QK_K)


# ------------------------------------------------------------------------------------------------
# TQ2_0 (quantize_row_tq2_0_ref + dequantize_row_tq2_0)
# ------------------------------------------------------------------------------------------------
def _tq2_0(x, qw):
    torch = _torch()
    amax = torch.zeros(x.shape[0], dtype=torch.float32)
    for j in range(QK_K):
        amax = _maxc(amax, x[:, j].abs())
    d = amax
    idv = torch.where(d != 0, _c(1) / torch.where(d != 0, d, _c(1)), _c(0))
    p = (x * idv[:, None]).double()
    q = torch.sign(p) * torch.floor(p.abs() + 0.5)                     # lroundf: half away from zero
    code = (q.to(torch.int64) + 1) & 3
    return (code - 1).to(torch.float32) * _fp16(d)[:, None]


# ------------------------------------------------------------------------------------------------
# STQ1_0 (Pollard's quantize_row_stq1_0_imatrix + dequantize_row_stq1_0)
# ------------------------------------------------------------------------------------------------
def _stq1_0(x, qw):
    torch = _torch()
    B = x.shape[0]
    w = qw if qw is not None else torch.ones_like(x)
    zpos, num, den = [], torch.zeros(B, dtype=torch.float64), torch.zeros(B, dtype=torch.float64)
    for g in range(QK_K // 4):
        base = (g // 16) * 64 + (g % 16)
        zm = torch.full((B,), float(3.4028234663852886e38), dtype=torch.float32)
        zp = torch.zeros(B, dtype=torch.int64)
        for p in range(4):
            idx = base + p * 16
            m = w[:, idx] * x[:, idx].abs()
            up = m < zm
            zm = torch.where(up, m, zm)
            zp = torch.where(up, torch.full_like(zp, p), zp)
        zpos.append(zp)
        for p in range(4):
            idx = base + p * 16
            keep = zp != p
            wd = w[:, idx].double()
            num = torch.where(keep, num + wd * x[:, idx].abs().double(), num)
            den = torch.where(keep, den + wd, den)
    d = torch.where(den > 0, (num / torch.where(den > 0, den, torch.ones_like(den))).float(), _c(0))
    bad = ~(d > 0)
    if bool(bad.any()):
        amax = torch.zeros(B, dtype=torch.float32)
        for j in range(QK_K):
            a = x[:, j].abs()
            amax = torch.where(a > amax, a, amax)
        d = torch.where(bad, amax, d)
    d16 = _fp16(d)
    y = torch.empty_like(x)
    for g in range(QK_K // 4):
        base = (g // 16) * 64 + (g % 16)
        for p in range(4):
            idx = base + p * 16
            q = torch.where(zpos[g] == p, _c(0), torch.where(x[:, idx] < 0, _c(-1), _c(1)))
            y[:, idx] = q * d16
    return y


# ------------------------------------------------------------------------------------------------
# IQ2 family: grids, kmap and neighbour lists (iq2xs_init_impl), encoders, decoders
# ------------------------------------------------------------------------------------------------
def ggml_src_dir():
    env = os.environ.get("POLLARD_GGML_SRC")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "runtime", "llama.cpp", "ggml", "src")


def _table(text, name):
    """The integer initialiser of a ggml table: GGML_TABLE_BEGIN(type, name, n) ... GGML_TABLE_END() in
    ggml-common.h, or `static const type name[n] = { ... };` in ggml-quants.c."""
    m = re.search(r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + re.escape(name) + r"\s*,[^)]*\)(.*?)GGML_TABLE_END", text, re.S) \
        or re.search(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise ValueError(f"table {name} not found")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body)]


@functools.lru_cache(maxsize=None)
def _iq2_tables(kind):
    """(qgrid [N, 8] int (2l+1 encoder grid), dgrid [N, 8] float32 (decoder bytes), kmap [43692],
    nbr [43692, maxN] padded with -1) for kind in iq2_xxs / iq2_xs / iq2_s."""
    import numpy as np
    torch = _torch()
    src = ggml_src_dir()
    try:
        common = open(os.path.join(src, "ggml-common.h"), encoding="utf-8").read()
        quants = open(os.path.join(src, "ggml-quants.c"), encoding="utf-8").read()
    except OSError as e:
        raise RuntimeError(f"pollard-fq needs the runtime's ggml sources for the {kind} grids ({e}); set "
                           "POLLARD_GGML_SRC to <llama.cpp>/ggml/src") from e
    gname, kname, nwant = {"iq2_xxs": ("iq2xxs_grid", "kgrid_2bit_256", 2),
                           "iq2_xs": ("iq2xs_grid", "kgrid_2bit_512", 2),
                           "iq2_s": ("iq2s_grid", "kgrid_2bit_1024", 1)}[kind]
    dg = np.array(_table(common, gname), dtype=np.uint64)
    kg = np.array(_table(quants, kname), dtype=np.int64)
    if len(dg) != len(kg):
        raise RuntimeError(f"{gname} ({len(dg)}) and {kname} ({len(kg)}) disagree in size")
    dbytes = dg.view(np.uint8).reshape(-1, 8).astype(np.int64)
    lvl = np.stack([(kg >> (2 * i)) & 3 for i in range(8)], 1)
    qgrid = 2 * lvl + 1
    if any(len(set(dbytes[lvl == l].tolist())) != 1 for l in range(4) if (lvl == l).any()):
        raise RuntimeError(f"{gname} does not match {kname} level-for-level -- mismatched ggml sources")
    kmap_size = 43692
    kmap = np.full(kmap_size, -1, dtype=np.int64)
    idx = (lvl << (2 * np.arange(8))).sum(1)
    kmap[idx] = np.arange(len(kg))
    off = np.nonzero(kmap < 0)[0]
    pos = np.stack([2 * ((off >> (2 * k)) & 3) + 1 for k in range(8)], 1)
    lists, maxn = [], 0
    for c0 in range(0, len(off), 4096):
        p = pos[c0:c0 + 4096]
        d2 = ((qgrid[None, :, :] - p[:, None, :]) ** 2).sum(-1)
        order = np.argsort(d2, axis=1, kind="stable")              # qsort by (d2, j)
        sd = np.take_along_axis(d2, order, 1)
        for r in range(len(p)):
            row, dd = order[r], sd[r]
            n, nhave, cur = 0, 1, dd[0]
            for j in range(len(row)):
                if dd[j] > cur:
                    if nhave == nwant:
                        break
                    cur = dd[j]
                    nhave += 1
                n += 1
            lists.append(row[:n])
            maxn = max(maxn, n)
    nbr = np.full((kmap_size, maxn), -1, dtype=np.int64)
    for u, row in zip(off, lists):
        nbr[u, :len(row)] = row
    return (torch.as_tensor(qgrid), torch.as_tensor(dbytes.astype(np.float32)), torch.as_tensor(kmap),
            torch.as_tensor(nbr))


def _best_neighbour(kind, u, xval, waux, scale, chunk=4096):
    """iq2_find_best_neighbour for rows: u [R], xval/waux [R, 8], scale [R] -> grid index [R]. Chunked: the
    candidate tensor is [rows, up to 292 neighbours, 8]."""
    torch = _torch()
    if u.shape[0] > chunk:
        return torch.cat([_best_neighbour(kind, u[i:i + chunk], xval[i:i + chunk], waux[i:i + chunk],
                                          scale[i:i + chunk], chunk) for i in range(0, u.shape[0], chunk)])
    qgrid, _, _, nbr = _iq2_tables(kind)
    cand = nbr[u]                                                   # [R, maxN]
    valid = cand >= 0
    pg = qgrid[cand.clamp(min=0)].to(torch.float32)                 # [R, maxN, 8]
    d2 = torch.zeros(cand.shape, dtype=torch.float32)
    for i in range(8):
        diff = _fma(scale[:, None], pg[..., i], -xval[:, None, i])
        d2 = _fma(waux[:, None, i] * diff, diff, d2)
    d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
    best = d2.min(1, keepdim=True).values
    ar = torch.arange(cand.shape[1]).expand_as(cand)
    first = torch.where(d2 == best, ar, torch.full_like(ar, cand.shape[1])).min(1).values
    return cand.gather(1, first[:, None]).squeeze(1)


def _quantize_groups(kind, xval, waux, scale_or_id, by_id):
    """One 8-wide pass: L from nearest_int(0.5*(id*xval-1)) clamped to 0..2, snapped to the grid (off-grid
    -> best neighbour at `scale`). xval/waux [G, 8]. Returns (L [G, 8], grid index [G], on_grid [G])."""
    torch = _torch()
    qgrid, _, kmap, _ = _iq2_tables(kind)
    idv, scale = scale_or_id if by_id else (None, None)
    if not by_id:
        scale = scale_or_id
        idv = _c(1) / scale
    l = torch.stack([_rne(_c(0.5) * _fma(idv, xval[:, i], _c(-1))).clamp(0, 2) for i in range(8)], 1)
    u = (l << (2 * torch.arange(8))).sum(1)
    gi = kmap[u]
    on = gi >= 0
    if bool((~on).any()):
        r = torch.nonzero(~on).squeeze(1)
        sc = scale if scale.dim() else scale.expand(len(on))
        gi = gi.clone()
        gi[r] = _best_neighbour(kind, u[r], xval[r], waux[r], sc[r])
    L = torch.where(on[:, None], l, (qgrid[gi.clamp(min=0)] - 1) // 2)
    return L, gi, on


def _signs_parity(xb, weight):
    """IQ2_XXS/XS sign handling for [G, 8]: xval = |x|, an odd number of negatives flips the least
    important one (min weight*x*x) so the parity bit can be implicit. Returns (xval, sign (+1/-1) [G, 8])."""
    torch = _torch()
    neg = ~(xb >= 0)
    xval = torch.where(neg, -xb, xb)
    odd = (neg.sum(1) % 2) == 1
    mn = (weight[:, 0] * xb[:, 0]) * xb[:, 0]
    imin = torch.zeros(xb.shape[0], dtype=torch.int64)
    for i in range(1, 8):
        ax = (weight[:, i] * xb[:, i]) * xb[:, i]
        up = ax < mn
        mn = torch.where(up, ax, mn)
        imin = torch.where(up, torch.full_like(imin, i), imin)
    flip = odd[:, None] & (torch.arange(8)[None, :] == imin[:, None])
    xval = torch.where(flip, -xval, xval)
    s = neg ^ flip
    return xval, s


def _sum_qx(weight, xval, L, fused=True):
    torch = _torch()
    f = _fma if fused else _mad
    sumqx = torch.zeros(xval.shape[0], dtype=torch.float32)
    sumq2 = torch.zeros(xval.shape[0], dtype=torch.float32)
    for i in range(xval.shape[1]):
        w = weight[:, i]
        q = (2 * L[:, i] + 1).to(torch.float32)
        sumqx = f(w * xval[:, i], q, sumqx)
        sumq2 = f(w * q, q, sumq2)
    return sumqx, sumq2


def _iq2_scale_encode(scales, d_mul=None):
    """Super-block scale: d = max_scale/31 -> fp16 (x0.9875 for IQ2_S), 4-bit sub-scales from 1/d.
    Returns (d16 [B], ls [B, n])."""
    torch = _torch()
    mx = torch.zeros(scales.shape[0], dtype=torch.float32)
    for i in range(scales.shape[1]):
        mx = _maxc(mx, scales[:, i])
    d = mx / _c(31)
    d16 = _fp16(d * _c(d_mul)) if d_mul is not None else _fp16(d)
    idv = _c(1) / torch.where(d != 0, d, _c(1))
    ls = _rne(_c(0.5) * _fma(idv[:, None], scales, _c(-1))).clamp(0, 15)
    d16 = torch.where(mx != 0, d16, _c(0))
    return d16, ls


def _iq2_decode(kind, d16, ls, gidx, sign, gsize):
    """y = (d * (0.5 + ls)) * 0.25 * grid_byte * (+-1) for groups of `gsize` sharing one sub-scale."""
    torch = _torch()
    _, dgrid, _, _ = _iq2_tables(kind)
    db = (d16[:, None] * (_c(0.5) + ls.to(torch.float32))) * _c(0.25)     # [B, nsub]
    B, nsub = ls.shape
    g = dgrid[gidx].reshape(B, nsub, gsize)                                  # [B, nsub, gsize]
    s = torch.where(sign.reshape(B, nsub, gsize), _c(-1), _c(1))
    return ((db[..., None] * g) * s).reshape(B, QK_K)


def _iq2_16(kind, x, qw):
    """IQ2_XS / IQ2_S: 16-weight sub-blocks, a 19-step scale search, snap-to-grid fix-up."""
    torch = _torch()
    B = x.shape[0]
    s_kind = kind == "iq2_s"
    sumx2 = _sumsq(x)
    sigma2 = (_c(2) * sumx2) / _c(QK_K) if s_kind else sumx2 / _c(QK_K)
    xb = x.reshape(B * 16, 16)
    sg = sigma2.repeat_interleave(16)
    if qw is not None:
        weight = qw.reshape(B * 16, 16) * _sqrtf(_fma(xb, xb, sg[:, None]))
    else:
        if not s_kind:
            raise ValueError("iq2_xs needs an imatrix (ggml asserts quant_weights)")
        weight = _fma(_c(0.25), sg[:, None], xb * xb)
    waux = _sqrtf(weight)
    G = xb.shape[0]
    if s_kind:
        neg = ~(xb >= 0)
        xval = torch.where(neg, -xb, xb)
        sign = neg
    else:
        xv, sg_ = [], []
        for k in range(2):
            a, b = _signs_parity(xb[:, 8 * k:8 * k + 8], weight[:, 8 * k:8 * k + 8])
            xv.append(a); sg_.append(b)
        xval, sign = torch.cat(xv, 1), torch.cat(sg_, 1)
    mx = xval[:, 0].clone()
    for i in range(1, 16):
        mx = _maxc(mx, xval[:, i])
    skip = mx < _c(1e-8 if s_kind else 1e-15)
    safe = torch.where(skip, _c(1), mx)
    best = torch.zeros(G, dtype=torch.float32)
    scale = safe / _c(5)
    L = torch.zeros(G, 16, dtype=torch.int64)
    on_grid = torch.ones(G, 2, dtype=torch.bool)
    for is_ in range(-9, 10):
        idv = _fma(_c(is_), _c(0.1), _c(5)) / safe
        this_scale = _c(1) / idv
        La, ona = [], []
        for k in range(2):
            l, _, on = _quantize_groups(kind, xval[:, 8 * k:8 * k + 8], waux[:, 8 * k:8 * k + 8],
                                        (idv, this_scale), True)
            La.append(l); ona.append(on)
        Laux = torch.cat(La, 1)
        sumqx, sumq2 = _sum_qx(weight, xval, Laux)
        up = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        nsc = sumqx / torch.where(sumq2 > 0, sumq2, _c(1))
        scale = torch.where(up, nsc, scale)
        best = torch.where(up, nsc * sumqx, best)
        L = torch.where(up[:, None], Laux, L)
        on_grid = torch.where(up[:, None], torch.stack(ona, 1), on_grid)
    fix = (~on_grid.all(1)) & (scale > 0)
    if bool(fix.any()):
        r = torch.nonzero(fix).squeeze(1)
        Lr = L[r].clone()
        for k in range(2):
            need = ~on_grid[r, k]
            if not bool(need.any()):
                continue
            rr = torch.nonzero(need).squeeze(1)
            sub = r[rr]
            l, _, _ = _quantize_groups(kind, xval[sub, 8 * k:8 * k + 8], waux[sub, 8 * k:8 * k + 8], scale[sub], False)
            Lr[rr, 8 * k:8 * k + 8] = l
        L[r] = Lr
        sumqx, sumq2 = _sum_qx(weight[r], xval[r], Lr)
        scale[r] = torch.where(sumq2 > 0, sumqx / torch.where(sumq2 > 0, sumq2, _c(1)), scale[r])
    negs = scale < 0
    scale = torch.where(negs, -scale, scale)
    sign = torch.where(negs[:, None], ~sign, sign)
    qgrid, _, kmap, _ = _iq2_tables(kind)
    u = torch.stack([(L[:, 8 * k:8 * k + 8] << (2 * torch.arange(8))).sum(1) for k in range(2)], 1)
    gidx = kmap[u]
    if bool(((gidx < 0) & ~skip[:, None]).any()):
        raise RuntimeError(f"{kind}: a point fell off the grid (ggml aborts here too)")
    scale = torch.where(skip, _c(0), scale)
    gidx = torch.where(skip[:, None], torch.zeros_like(gidx), gidx)
    sign = torch.where(skip[:, None], torch.zeros_like(sign), sign)
    dead = (scale.reshape(B, 16) == 0).all(1).repeat_interleave(16)        # !max_scale: block zeroed
    sign = torch.where(dead[:, None], torch.zeros_like(sign), sign)
    d16, ls = _iq2_scale_encode(scale.reshape(B, 16), 0.9875 if s_kind else None)
    return _iq2_decode(kind, d16, ls, gidx.reshape(B, 32), sign.reshape(B, 16, 16), 16)


def _iq2_xxs(x, qw):
    """IQ2_XXS: 32-weight sub-blocks, make_qp_quants seed, 13-step search, always re-snapped. Its 32-wide
    sumqx / sumq2 loops are vectorised by clang (rounded products, in-order adds); IQ2_XS/S's 16-wide
    ones stay fused."""
    torch = _torch()
    kind = "iq2_xxs"
    if qw is None:
        raise ValueError("iq2_xxs needs an imatrix (ggml asserts quant_weights)")
    B = x.shape[0]
    sigma2 = _sumsq(x) / _c(QK_K)
    xb = x.reshape(B * 8, 32)
    sg = sigma2.repeat_interleave(8)
    weight = qw.reshape(B * 8, 32) * _sqrtf(_fma(xb, xb, sg[:, None]))
    waux = _sqrtf(weight)
    G = xb.shape[0]
    xv, sv = [], []
    for k in range(4):
        a, b = _signs_parity(xb[:, 8 * k:8 * k + 8], weight[:, 8 * k:8 * k + 8])
        xv.append(a); sv.append(b)
    xval, sign = torch.cat(xv, 1), torch.cat(sv, 1)
    mx = xval[:, 0].clone()
    for i in range(1, 32):
        mx = _maxc(mx, xval[:, i])
    skip = mx < _c(1e-15)
    scale, Lqp = _make_qp_quants(xval, 4, weight)
    eff_max = scale * _c(3)
    skip = skip | (eff_max <= 0)
    safe_eff = torch.where(skip, _c(1), eff_max)
    L = Lqp
    best = torch.zeros(G, dtype=torch.float32)
    for is_ in range(-6, 7):
        idv = _fma(_c(is_), _c(0.1), _c(5)) / safe_eff
        this_scale = _c(1) / idv
        La = []
        for k in range(4):
            l, _, _ = _quantize_groups(kind, xval[:, 8 * k:8 * k + 8], waux[:, 8 * k:8 * k + 8], (idv, this_scale), True)
            La.append(l)
        Laux = torch.cat(La, 1)
        sumqx, sumq2 = _sum_qx(weight, xval, Laux, fused=False)
        up = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        nsc = sumqx / torch.where(sumq2 > 0, sumq2, _c(1))
        scale = torch.where(up, nsc, scale)
        best = torch.where(up, nsc * sumqx, best)
        L = torch.where(up[:, None], Laux, L)
    pos = (scale > 0) & ~skip
    if bool(pos.any()):
        r = torch.nonzero(pos).squeeze(1)
        Lr = []
        for k in range(4):
            l, _, _ = _quantize_groups(kind, xval[r, 8 * k:8 * k + 8], waux[r, 8 * k:8 * k + 8], scale[r], False)
            Lr.append(l)
        Lr = torch.cat(Lr, 1)
        L[r] = Lr
        sumqx, sumq2 = _sum_qx(weight[r], xval[r], Lr, fused=False)
        scale[r] = torch.where(sumq2 > 0, sumqx / torch.where(sumq2 > 0, sumq2, _c(1)), scale[r])
    negs = scale < 0
    scale = torch.where(negs, -scale, scale)
    # the 8th sign is implicit parity; flipping 7 stored bits flips the parity bit too
    sign = torch.where(negs[:, None], ~sign, sign)
    _, _, kmap, _ = _iq2_tables(kind)
    u = torch.stack([(L[:, 8 * k:8 * k + 8] << (2 * torch.arange(8))).sum(1) for k in range(4)], 1)
    gidx = kmap[u.clamp(0, 43691)]
    if bool(((gidx < 0) & ~skip[:, None]).any()):
        raise RuntimeError("iq2_xxs: a point fell off the grid (ggml aborts here too)")
    scale = torch.where(skip, _c(0), scale)
    gidx = torch.where(skip[:, None], torch.zeros_like(gidx), gidx)
    sign = torch.where(skip[:, None], torch.zeros_like(sign), sign)
    dead = (scale.reshape(B, 8) == 0).all(1).repeat_interleave(8)          # !max_scale: qs zeroed
    sign = torch.where(dead[:, None], torch.zeros_like(sign), sign)
    gidx = torch.where(dead[:, None], torch.zeros_like(gidx), gidx)
    d16, ls = _iq2_scale_encode(scale.reshape(B, 8))
    return _iq2_decode(kind, d16, ls, gidx.reshape(B, 32), sign.reshape(B, 8, 32), 32)


_ENC = {"q2_k": _q2_k, "q3_k": _q3_k, "tq2_0": _tq2_0, "stq1_0": _stq1_0, "iq2_xxs": _iq2_xxs,
        "iq2_xs": lambda x, qw: _iq2_16("iq2_xs", x, qw), "iq2_s": lambda x, qw: _iq2_16("iq2_s", x, qw)}


def quant_dequant(W, qtype, imatrix=None):
    """dequantize(quantize(W, imatrix)) exactly as the runtime's ggml computes it. W: [rows, n] (any float
    dtype, n a multiple of 256); imatrix: [n] importances (llama-imatrix's per-input-channel values) or
    None. Returns float32 [rows, n]. No autograd."""
    torch = _torch()
    t = str(qtype).lower()
    if t not in _ENC:
        raise ValueError(f"pollard-fq has no port of {qtype} (supported: {', '.join(SUPPORTED)})")
    if t not in SUPPORTED:
        raise ValueError(f"{qtype}: the port did not pass the bit-exact parity gate; refusing to approximate")
    if t in NEEDS_IMATRIX and imatrix is None:
        raise ValueError(f"{qtype}: pass the imatrix -- this port follows ggml's imatrix encoder, which is "
                         "what every Pollard build uses (the imatrix-free *_ref encoder is not ported)")
    with torch.no_grad():
        W2 = W.detach().to("cpu", torch.float32)
        if W2.dim() != 2:
            W2 = W2.reshape(-1, W2.shape[-1])
        x, qw = _blocks(W2.contiguous(), imatrix)
        step = CHUNK_BLOCKS                      # bounded memory: the encoders hold [blocks, 256] temporaries
        ys = [_ENC[t](x[i:i + step].contiguous(), None if qw is None else qw[i:i + step].contiguous())
              for i in range(0, x.shape[0], step)]
        return torch.cat(ys).reshape(W2.shape)


def fake_quant(W, qtype, imatrix=None, ste=True):
    """Forward: the exact deployed weights. Backward (STE): the gradient flows to W unchanged."""
    Wq = quant_dequant(W, qtype, imatrix).to(W.device, W.dtype).reshape(W.shape)
    return W + (Wq - W).detach() if ste else Wq


def _linear_cls():
    torch = _torch()

    class FakeQuantLinear(torch.nn.Module):
        """A Linear whose forward uses fake_quant(weight). `refresh_every` > 1 caches the quantized
        weight for that many forwards (the encoder is the expensive part; the STE gradient still
        reaches the live weight)."""

        def __init__(self, linear, qtype, imatrix=None, refresh_every=1):
            super().__init__()
            self.weight = torch.nn.Parameter(linear.weight.detach().clone())
            self.bias = linear.bias
            self.qtype, self.imatrix, self.refresh_every = qtype, imatrix, max(1, int(refresh_every))
            self.on, self._n, self._cache = True, 0, None

        def forward(self, x):
            if not self.on:
                return torch.nn.functional.linear(x, self.weight, self.bias)
            if self._cache is None or self._n % self.refresh_every == 0:
                self._cache = quant_dequant(self.weight, self.qtype, self.imatrix).to(self.weight.device,
                                                                                      self.weight.dtype)
            self._n += 1
            w = self.weight + (self._cache - self.weight).detach()
            return torch.nn.functional.linear(x, w, self.bias)

    return FakeQuantLinear


def __getattr__(name):
    if name == "FakeQuantLinear":
        return _linear_cls()
    raise AttributeError(name)
