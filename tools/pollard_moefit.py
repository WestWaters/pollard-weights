#!/usr/bin/env python3
"""pollard-moefit -- per-expert-tensor allocation for a MoE at ~1.6-2.5 bpw, with 1-bit experts on the table.

The question this answers: on a big sparse MoE, which routed-expert matrices can go to 1 bit, which need 2-4,
and what has to stay high -- chosen from MEASUREMENT, not a role rule. SAGE-EXL3's 1.62 bpw DeepSeek-V4.1-Flash
(58% of expert matrices at K1, head 16-bit) showed a K1-heavy mix is usable when the allocation is per tensor and
measured; this is Pollard's version of that allocator, for every lane that can store it.

    pollard-moefit routing --imatrix m.imatrix.gguf                          # per-layer expert traffic, from the imatrix
    pollard-moefit errmap  --gguf m-f16.gguf --imatrix m.imatrix.gguf --out m.errmap.json
    pollard-moefit plan    --errmap m.errmap.json --sensitivity m.sensitivity.json --gb 0.55 --out plan.json
    pollard-moefit build   --gguf m-f16.gguf --imatrix m.imatrix.gguf --plan plan.json --out m-Pollard-MoE.gguf
    pollard-moefit exl3-recipe --model ./M --routing routing.json --sensitivity s.json --bpw 1.8 --out recipe.yaml
    pollard-moefit lowbit  --gguf m-f16.gguf --imatrix m.imatrix.gguf --eval sens.txt --out m.lowbit.json
    pollard-moefit auto    --gguf m-f16.gguf --imatrix m.imatrix.gguf --errmap m.errmap.json --sensitivity s.json \
                           --lowbit m.lowbit.json --gb 0.36 --screen-eval sens.txt --eval judge.txt --out m-MOE.gguf

THE AUTO-AWARE PRUNE (auto) -- the default for a MoE release rung

  No fixed prune fraction: every p in {0, 10, 25, 40, 50 %} (REAP scores pick which experts go) is screened at the
  SAME byte budget -- the pruned model's own KL measured, the bit mix for what is left planned on the error map
  restricted to the kept experts -- and the best 2-3 are built and MEASURED (KLD / top-1 / PPL on a held-out judge
  eval, the tool gate). The lowest measured KLD whose tool calls hold wins; the report (and the card) carries every
  candidate and why the winner won.

THE 1-BIT COST (lowbit)

  e(t, q) is a weight-space error; the KL a 1-bit tensor costs is not linear in it. lowbit MEASURES each expert
  class (layer x gate/up/down) at iq1_s, iq1_m and the iq2_xxs anchor -- quantize that class alone, KL on held-out
  text with the real routing -- and plan prices those types by the measurement (other types scale from the
  nearest measured anchor). A 1-bit tensor is bought only where its measured cost per byte beats the alternatives.

THE MEASUREMENT (errmap)

  For every candidate GGUF type q, the model is quantized ONCE with every block tensor at q (one encode per type,
  the "one encode, many rungs" pattern). Each tensor is then dequantized (gguf-py) and compared with the source:

      e_x(t, q) = sum_c a_xc * sum_r (W - W_q)^2_xrc  /  sum_c a_xc * sum_r W^2_xrc

  per routed EXPERT x (a = that expert's mean squared input activation from the GGUF imatrix, `in_sum2 / counts`),
  i.e. the share of the expert's output energy the quantization error carries -- tr(dW H dW^T) / tr(W H W^T) with
  the diagonal H the imatrix holds. The expert's token COUNT (how often the router picked it on the calibration
  corpus) is recorded alongside. A tensor's error is the count-weighted aggregate over its experts, so a cold
  expert's error weighs what its traffic weighs.

THE ESTIMATE

  KL(t, q) = S_t * e(t, q) / e(t, probe)
  S_t = Pollard's measured group sensitivity (pollard-sensitivity: the KL when a layer's ffn / attn group is crushed
  to the probe type, default iq2_xxs) split across the group's tensors by size; per-expert units (EXL3 / EXLR8, which
  store one record per expert matrix) take S_t * (count_x * energy_x) / sum(count * energy). Without a profile
  S_t = the tensor's share of the weights (the allocation then minimises summed relative output error).

THE CHOICE

  An EXACT multiple-choice knapsack (pollard-exlr8's DP over the byte budget) picks one type per tensor minimising
  the summed KL estimate subject to the file's bytes <= budget. Roles bound the choice: attention / shared expert /
  dense FFN have a floor (default iq4_xs), the router (ffn_gate_inp) is never quantized, the head and embeddings are
  fixed (default q8_0 / q6_K). Routed experts range from iq1_s to q6_K; --no-1bit stops them at iq2_xxs.

  GGUF stores one type per TENSOR, and llama.cpp merges a layer's experts into one ffn_*_exps tensor -- so the GGUF
  lane allocates per routed-expert TENSOR (layer x gate/up/down), with each choice driven by the per-expert errors
  and counts underneath. Per-EXPERT widths need a format that stores experts separately: EXL3 (exllamav3 recipe,
  K = 1..8) and EXLR8 (one trellis record per expert matrix) -- see `exl3-recipe` and pollard-exlr8 --routing.

  The estimate chooses; the measurement judges: every rung goes through llama-perplexity --kl-divergence against the
  bf16/f16 base and pollard-toolcall, like every other Pollard rung.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

FORMAT_ERRMAP = "pollard-moefit-errmap"
FORMAT_PLAN = "pollard-moefit-plan"

#: candidate GGUF types (stock llama.cpp -- every one loads in upstream llama.cpp, LM Studio, Ollama), low -> high
TYPES_1BIT = ("iq1_s", "iq1_m")
TYPES_LOW = ("iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs")
TYPES_HIGH = ("q5_K", "q6_K", "q8_0")
DEFAULT_TYPES = TYPES_1BIT + TYPES_LOW + TYPES_HIGH
#: bytes per (block size, block bytes) -- the exact ggml layouts (gguf-py GGML_QUANT_SIZES agrees; kept here so the
#: planner needs no gguf import)
QUANT_SIZES = {"f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q8_0": (32, 34), "q6_K": (256, 210),
               "q5_K": (256, 176), "q4_K": (256, 144), "q3_K": (256, 110), "q2_K": (256, 84),
               "iq4_xs": (256, 136), "iq4_nl": (32, 18), "iq3_s": (256, 110), "iq3_xxs": (256, 98),
               "iq2_s": (256, 82), "iq2_xs": (256, 74), "iq2_xxs": (256, 66), "iq1_m": (256, 56), "iq1_s": (256, 50),
               # vendor formats a QAT source may carry (kept byte-for-byte in passthrough, never a target here)
               "nvfp4": (64, 36), "mxfp4": (32, 17), "tq1_0": (256, 54), "tq2_0": (256, 66)}
#: types llama-quantize refuses without imatrix coverage of the tensor
IMATRIX_REQUIRED = {"iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"}

ROLE_EXPERT, ROLE_SHARED, ROLE_ATTN, ROLE_DENSE, ROLE_ROUTER = "expert", "shared", "attn", "dense_ffn", "router"
ROLE_EMBD, ROLE_HEAD, ROLE_OTHER = "embd", "head", "other"
#: the default floor per role: what may NOT go below it (the knapsack chooses above it)
DEFAULT_FLOORS = {ROLE_EXPERT: "iq1_s", ROLE_SHARED: "iq4_xs", ROLE_ATTN: "iq4_xs", ROLE_DENSE: "iq4_xs"}
DEFAULT_CEIL = "q8_0"
DEFAULT_HEAD, DEFAULT_EMBD = "q8_0", "q6_K"

_BLK = re.compile(r"^blk\.(\d+)\.")
_EXPERT = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.weight$")
_SHARED = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_(shexp|chexp)\.weight$")
_ROUTER = re.compile(r"^blk\.\d+\.(ffn_gate_inp|ffn_gate_inp_shexp|exp_probs_b)(\.weight|\.bias)?$")
_DENSE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)\.weight$")
_SEQMIX = re.compile(r"^blk\.\d+\.(attn_|ssm_|time_mix|shortconv|conv|mamba|linear_attn|indexer)")


def _work_root():
    """Probe GGUFs and base logits (model-sized) go in the workspace cache/, not the system temp dir."""
    import pollard_workspace as ws
    return ws.cache_dir(create=True)


def type_rank(t):
    """position of a type on the low -> high byte scale (unknown types sort by their bpw)."""
    return bpw_of(t)


def bpw_of(t):
    bs, ts = QUANT_SIZES[t]
    return 8.0 * ts / bs


def type_bytes(t, numel):
    bs, ts = QUANT_SIZES[t]
    return int(numel // bs * ts)


def role_of(name, ndim=2):
    """the allocation role of a GGUF tensor name."""
    if name == "token_embd.weight":
        return ROLE_EMBD
    if name == "output.weight":
        return ROLE_HEAD
    if ndim < 2 or not name.endswith(".weight"):
        return ROLE_OTHER
    if _ROUTER.search(name):
        return ROLE_ROUTER
    if _EXPERT.search(name):
        return ROLE_EXPERT
    if _SHARED.search(name):
        return ROLE_SHARED
    if _DENSE.search(name):
        return ROLE_DENSE
    if _BLK.search(name) and _SEQMIX.search(name):
        return ROLE_ATTN
    if _BLK.search(name):
        return ROLE_DENSE                      # any other block matrix: treated like a dense FFN (floored)
    return ROLE_OTHER


def layer_of(name):
    m = _BLK.search(name)
    return int(m.group(1)) if m else None


def group_of(role):
    """pollard-sensitivity's group for a role: sequence mixing (attn) or channel mixing (ffn)."""
    return "attn" if role == ROLE_ATTN else "ffn"


# ------------------------------------------------------------------------------------------------ imatrix / routing

def read_imatrix(path):
    """{tensor name: (in_sum2 [n_mat, n_in] float64, counts [n_mat] float64)} from a GGUF-format imatrix
    (llama-imatrix's default since 2025: `<name>.in_sum2` + `<name>.counts`). For a merged expert tensor n_mat is
    the expert count and counts[x] = tokens the router sent to expert x on the calibration corpus.

    A legacy .dat imatrix has ONE call count per tensor, not per expert, so it cannot say which experts are hot:
    it is read with counts = the single ncall for every expert (equal traffic) and flagged by the caller."""
    import gguf
    try:
        r = gguf.GGUFReader(path)
    except Exception:
        return _read_imatrix_legacy(path), "legacy"
    sums, cnts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-len(".in_sum2")]] = np.asarray(t.data, dtype=np.float64)
        elif t.name.endswith(".counts"):
            cnts[t.name[:-len(".counts")]] = np.asarray(t.data, dtype=np.float64).reshape(-1)
    out = {}
    for k, s in sums.items():
        s = s.reshape(-1, s.shape[-1]) if s.ndim > 1 else s.reshape(1, -1)
        c = cnts.get(k, np.ones(s.shape[0]))
        if c.size == 1 and s.shape[0] > 1:
            c = np.full(s.shape[0], float(c[0]))
        out[k] = (s, c)
    return out, "gguf"


def _read_imatrix_legacy(path):
    import struct
    d = open(path, "rb").read()
    out, off = {}, 0
    n = struct.unpack_from("<i", d, off)[0]
    off += 4
    for _ in range(n):
        ln = struct.unpack_from("<i", d, off)[0]
        off += 4
        name = d[off:off + ln].decode("utf-8", "replace")
        off += ln
        ncall = struct.unpack_from("<i", d, off)[0]
        off += 4
        nval = struct.unpack_from("<i", d, off)[0]
        off += 4
        v = np.frombuffer(d, dtype="<f4", count=nval, offset=off).astype(np.float64)
        off += 4 * nval
        out[name] = (v.reshape(1, -1), np.array([float(max(ncall, 1))]))   # sums are already means x ncall
    return out


def routing_profile(imx):
    """{layer: {"counts": [E], "share": [E]}} from the gate/up expert entries of an imatrix (gate and up see the
    same routed tokens, so their counts agree; up is taken when both exist)."""
    prof = {}
    for name, (s, c) in imx.items():
        m = re.match(r"^blk\.(\d+)\.ffn_(up|gate|gate_up)_exps\.weight$", name)
        if not m or c.size < 2:
            continue
        L = int(m.group(1))
        if L in prof and m.group(2) != "up":
            continue
        tot = float(c.sum()) or 1.0
        prof[L] = {"counts": [float(x) for x in c], "share": [float(x) / tot for x in c]}
    return dict(sorted(prof.items()))


def routing_summary(prof):
    """per layer: experts, normalised entropy (1 = perfectly balanced), the share of the top 25% of experts, and
    the dead experts (never routed on the corpus)."""
    rows = []
    for L, p in prof.items():
        sh = np.asarray(p["share"], dtype=np.float64)
        E = sh.size
        nz = sh[sh > 0]
        ent = float(-(nz * np.log(nz)).sum() / math.log(E)) if E > 1 else 1.0
        top = float(np.sort(sh)[::-1][:max(1, E // 4)].sum())
        rows.append({"layer": L, "experts": E, "entropy": round(ent, 4), "top25_share": round(top, 4),
                     "dead": int((sh == 0).sum())})
    return rows


# ------------------------------------------------------------------------------------------------ the error map

def weighted_rel_err(W, Wq, act):
    """per matrix x: (err_x, energy_x) with err = sum_c a_c sum_r (W - Wq)^2, energy = sum_c a_c sum_r W^2.
    W, Wq: [n_mat, rows, n_in] (or [rows, n_in]); act: [n_mat, n_in] (or [n_in]) mean squared input activation."""
    W = np.asarray(W, dtype=np.float64)
    Wq = np.asarray(Wq, dtype=np.float64)
    if W.ndim == 2:
        W, Wq = W[None], Wq[None]
    a = np.asarray(act, dtype=np.float64).reshape(W.shape[0] if np.size(act) != W.shape[-1] else 1, -1)
    if a.shape[0] == 1 and W.shape[0] > 1:
        a = np.repeat(a, W.shape[0], axis=0)
    d2 = ((W - Wq) ** 2).sum(axis=1)                  # [n_mat, n_in]
    w2 = (W ** 2).sum(axis=1)
    return (d2 * a).sum(axis=1), (w2 * a).sum(axis=1)


def aggregate(err, energy, counts):
    """a tensor's relative error: count-weighted output-error energy over count-weighted output energy."""
    err, energy, counts = (np.asarray(x, dtype=np.float64) for x in (err, energy, counts))
    den = float((counts * energy).sum())
    return float((counts * err).sum() / den) if den > 0 else 0.0


def _act(imx, name, n_mat, n_in):
    """mean squared input activation per matrix [n_mat, n_in] and token counts [n_mat]; ones if uncovered."""
    ent = imx.get(name)
    if ent is None:
        return np.ones((n_mat, n_in)), np.ones(n_mat), False
    s, c = ent
    if s.shape[-1] != n_in:
        s = s.reshape(-1)[: n_mat * n_in].reshape(n_mat, n_in) if s.size >= n_mat * n_in else np.ones((n_mat, n_in))
    if s.shape[0] != n_mat:
        s = np.repeat(s[:1], n_mat, axis=0)
    c = c if c.size == n_mat else np.full(n_mat, float(c.reshape(-1)[0]))
    a = s / np.maximum(c, 1.0)[:, None]
    return a, c, True


def _tensor_array(t):
    """a GGUFReader tensor -> float32 numpy in its logical shape (dequantized when quantized)."""
    import gguf
    from gguf import quants
    qt = t.tensor_type
    if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(t.data, dtype=np.float32)
    return quants.dequantize(np.asarray(t.data), qt).astype(np.float32)        # bf16 and every quant type


def candidate_tensors(reader):
    """[(name, role, shape(numpy order), numel)] -- the block matrices a GGUF rung allocates (llama-quantize can
    store them in any k/iq type: 2-D or 3-D, input dim a multiple of 256)."""
    out = []
    for t in reader.tensors:
        ne = [int(x) for x in t.shape]                 # ggml order: ne[0] = input dim
        role = role_of(t.name, len(ne))
        if role not in (ROLE_EXPERT, ROLE_SHARED, ROLE_ATTN, ROLE_DENSE):
            continue
        if ne[0] % 256:
            continue                                   # no k/iq type fits a row: left to llama-quantize's fallback
        out.append((t.name, role, tuple(reversed(ne)), int(np.prod(ne))))
    return out


def probe_type_file(names, qtype, path):
    with open(path, "w") as f:
        for n in names:
            f.write(f"^{re.escape(n)}$={qtype}\n")
    return path


def measure_probe(src_reader, probe_reader, cands, imx, store, qtype, chunk=64):
    """fill store[name]["rel"][qtype] (and per-expert errors) for one probe build."""
    src = {t.name: t for t in src_reader.tensors}
    prb = {t.name: t for t in probe_reader.tensors}
    import gguf
    for name, role, shape, numel in cands:
        tq = prb.get(name)
        if tq is None:
            continue
        got = gguf.GGMLQuantizationType(tq.tensor_type).name.lower()
        rec = store[name]
        if got != qtype.lower():
            rec.setdefault("type_mismatch", {})[qtype] = got     # llama-quantize fell back: record what it built
        rec.setdefault("bytes", {})[qtype] = int(tq.n_bytes)      # the exact bytes of this tensor at this type
        n_mat = shape[0] if len(shape) == 3 else 1
        n_in = shape[-1]
        act, cnt, covered = _act(imx, name, n_mat, n_in)
        ts = src[name]
        err = np.zeros(n_mat)
        en = np.zeros(n_mat)
        if n_mat == 1:
            W = _tensor_array(ts).reshape(shape[-2], n_in)
            Wq = _tensor_array(tq).reshape(shape[-2], n_in)
            err, en = weighted_rel_err(W, Wq, act[0])
        else:
            rs, rq = np.asarray(ts.data), np.asarray(tq.data)
            for a0 in range(0, n_mat, chunk):
                b0 = min(n_mat, a0 + chunk)
                W = _dequant_slice(ts, rs, a0, b0).reshape(b0 - a0, shape[1], n_in)
                Wq = _dequant_slice(tq, rq, a0, b0).reshape(b0 - a0, shape[1], n_in)
                e, g = weighted_rel_err(W, Wq, act[a0:b0])
                err[a0:b0], en[a0:b0] = e, g
        rec["rel"][qtype] = aggregate(err, en, cnt)
        rec["covered"] = bool(covered)
        if n_mat > 1:
            rec.setdefault("experts", {"count": [float(x) for x in cnt], "energy": [float(x) for x in en], "err": {}})
            rec["experts"]["err"][qtype] = [float(x) for x in (err / np.maximum(en, 1e-300))]


def _dequant_slice(t, raw, a, b):
    import gguf
    from gguf import quants
    qt = t.tensor_type
    if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return np.asarray(raw[a:b], dtype=np.float32)
    return quants.dequantize(np.asarray(raw[a:b]), qt).astype(np.float32)


def errmap(gguf_path, imatrix, out, types=DEFAULT_TYPES, quantize="llama-quantize", workdir=None,
           keep_probes=False, threads=None, log=print, allow_requantize=False):
    """Measure e(t, q) for every block matrix and every candidate type: one llama-quantize per type, then a
    dequantize-and-compare pass (streaming, one tensor -- or one chunk of experts -- in memory at a time)."""
    import gguf
    src = gguf.GGUFReader(gguf_path)
    imx, kind = read_imatrix(imatrix)
    if kind == "legacy":
        log("WARNING: legacy .dat imatrix -- no per-expert token counts; every expert weighs the same. Rebuild the "
            "imatrix with a current llama-imatrix (GGUF output) to get the routing counts.")
    cands = candidate_tensors(src)
    if not cands:
        raise SystemExit("no block matrices to allocate -- is this a GGUF of a transformer?")
    names = [c[0] for c in cands]
    store = {n: {"role": r, "layer": layer_of(n), "shape": list(s), "numel": int(k),
                 "bytes": {q: type_bytes(q, k) for q in types}, "rel": {}} for n, r, s, k in cands}
    for t in src.tensors:                                  # what the source stores (a QAT GGUF: the kept option)
        if t.name in store:
            st = gguf.GGMLQuantizationType(t.tensor_type).name.lower()
            store[t.name]["src_type"] = st
            store[t.name]["src_bytes"] = int(t.n_bytes)
            if st in QUANT_SIZES:
                store[t.name]["bytes"].setdefault(st, int(t.n_bytes))
    fixed = {}
    for t in src.tensors:
        if t.name not in store:
            fixed[t.name] = {"role": role_of(t.name, len(t.shape)), "numel": int(np.prod([int(x) for x in t.shape])),
                             "src_bytes": int(t.n_bytes)}
    work = workdir or tempfile.mkdtemp(prefix="pollard_moefit_", dir=_work_root())
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    for q in types:
        probe = os.path.join(work, f"probe-{q}.gguf")
        tt = probe_type_file(names, q, os.path.join(work, f"probe-{q}.txt"))
        if not (os.path.exists(probe) and _gguf_ok(probe)):
            cmd = [quantize] + (["--allow-requantize"] if allow_requantize else []) + [
                "--imatrix", imatrix, "--tensor-type-file", tt, gguf_path, probe, "Q8_0"]
            if threads:
                cmd.append(str(threads))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not _gguf_ok(probe):
                tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-6:])
                log(f"  {q:8} probe build FAILED -- skipped\n{tail}")
                continue
        measure_probe(src, gguf.GGUFReader(probe), cands, imx, store, q)
        if not keep_probes:
            os.remove(probe)
        med = np.median([store[n]["rel"].get(q, np.nan) for n in names])
        log(f"  {q:8} {bpw_of(q):5.2f} bpw   median rel err {med:.5f}   [{time.time() - t0:5.0f}s]")
    if not keep_probes and workdir is None:
        shutil.rmtree(work, ignore_errors=True)
    payload = {"format": FORMAT_ERRMAP, "source": os.path.abspath(gguf_path), "imatrix": os.path.abspath(imatrix),
               "imatrix_kind": kind, "types": [q for q in types if any(q in store[n]["rel"] for n in names)],
               "passthrough": bool(allow_requantize),
               "routing": {str(k): v for k, v in routing_profile(imx).items()},
               "tensors": store, "fixed": fixed}
    if out:
        with open(out, "w") as f:
            json.dump(payload, f)
    return payload


def _gguf_ok(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF" and os.path.getsize(path) > 1024
    except OSError:
        return False


# ------------------------------------------------------------------------------------------------ the plan

def tensor_sensitivity(tensors, sens):
    """{name: S_t}: the measured group KL per (layer, attn|ffn) split across the group's tensors by size. No profile:
    the tensor's share of the allocated weights."""
    numel = {n: t["numel"] for n, t in tensors.items()}
    if not sens:
        tot = float(sum(numel.values())) or 1.0
        return {n: k / tot for n, k in numel.items()}
    grp = {g: {int(k): float(v) for k, v in (sens.get(g) or {}).items() if v is not None} for g in ("attn", "ffn")}
    allv = sorted(v for d in grp.values() for v in d.values())
    fallback = allv[len(allv) // 2] if allv else 1.0
    gsize = {}
    for n, t in tensors.items():
        key = (group_of(t["role"]), t["layer"])
        gsize[key] = gsize.get(key, 0) + numel[n]
    out = {}
    for n, t in tensors.items():
        g, L = group_of(t["role"]), t["layer"]
        other = "ffn" if g == "attn" else "attn"
        sg = grp[g].get(L, grp[other].get(L, fallback)) if L is not None else fallback
        out[n] = max(sg, 0.0) * numel[n] / max(gsize[(g, L)], 1)
    return out


def allowed_types(role, types, floors=None, ceil=DEFAULT_CEIL, allow_1bit=True):
    floors = {**DEFAULT_FLOORS, **(floors or {})}
    lo = floors.get(role, "iq4_xs")
    if role == ROLE_EXPERT and not allow_1bit and bpw_of(lo) < bpw_of("iq2_xxs"):
        lo = "iq2_xxs"
    return [q for q in types if bpw_of(lo) - 1e-9 <= bpw_of(q) <= bpw_of(ceil) + 1e-9]


def fixed_bytes(em, head=DEFAULT_HEAD, embd=DEFAULT_EMBD):
    """bytes of everything the knapsack does not choose: head / embeddings at their fixed types, the rest (norms,
    router, odd-shaped matrices) as the source stores them -- f32 router / norms stay f32 in a llama-quantize build;
    other matrices fall back to q8_0, the build's base preset."""
    tot = 0
    for n, f in em.get("fixed", {}).items():
        r = f["role"]
        if r == ROLE_HEAD:
            tot += type_bytes(head, f["numel"])
        elif r == ROLE_EMBD:
            tot += type_bytes(embd, f["numel"])
        elif r in (ROLE_ROUTER, ROLE_OTHER):
            tot += int(f["src_bytes"]) if f["src_bytes"] <= f["numel"] * 4 else f["numel"] * 4
        else:
            tot += type_bytes("q8_0", f["numel"]) if f["numel"] % 32 == 0 else f["numel"] * 2
    return tot


def plan(em, budget_bytes, sens=None, probe=None, floors=None, ceil=DEFAULT_CEIL, allow_1bit=True,
         head=DEFAULT_HEAD, embd=DEFAULT_EMBD, types=None, max_cells=200_000_000, lowbit=None):
    """the per-tensor type choice: exact multiple-choice knapsack on KL(t, q) = S_t * e(t, q) / e(t, probe) under
    the whole file's bytes <= budget_bytes.

    lowbit (pollard-moefit lowbit, or {"tensor_costs": ...}): the routed-expert tensors' KL MEASURED at the low
    types (iq1_s / iq1_m / iq2_xxs, one class crushed at a time) replaces the extrapolated law for those tensors:
    a measured type costs what it measured, any other type is scaled from the nearest measured anchor at or above
    it, KL(t, q) = KL(t, a) * e(t, q) / e(t, a) -- so a 1-bit choice is priced by its own measurement, never by a
    straight line drawn from the 2-bit probe."""
    from pollard_exlr8 import knapsack
    tensors = em["tensors"]
    types = [q for q in (types or em["types"]) if q in QUANT_SIZES]
    probe = probe or (sens or {}).get("probe") or "iq2_xxs"
    S = tensor_sensitivity(tensors, sens)
    if lowbit and not sens:
        raise ValueError("measured low-bit costs are KL: they need the measured sensitivity profile (--sensitivity) "
                         "so every tensor is priced in the same unit")
    LB = resolve_lowbit(lowbit, tensors) if lowbit else {}
    fixed = fixed_bytes(em, head, embd)
    names = sorted(tensors, key=lambda n: (tensors[n]["layer"] if tensors[n]["layer"] is not None else -1, n))
    items = []
    for n in names:
        t = tensors[n]
        ref = float(t["rel"].get(probe) or 0.0) if sens else 1.0
        if sens and ref <= 0:
            ref = _interp_rel(t["rel"], probe) or 1.0
        opts = []
        passthrough = em.get("passthrough") and t.get("src_type") not in (None, "f32", "f16", "bf16")
        for q in allowed_types(t["role"], types, floors, ceil, allow_1bit):
            e = t["rel"].get(q)
            if e is None:
                continue
            b = int(t["bytes"][q] if q in t["bytes"] else type_bytes(q, t["numel"]))
            if passthrough and (b > t["src_bytes"] or q == t["src_type"]):
                continue                                 # never "upgrade" a QAT tensor; its own type is below
            cost = lowbit_cost(LB[n], t, q) if n in LB else S[n] * float(e) / ref
            opts.append((b, cost, q))
        if passthrough:                                  # the source's own bytes, copied verbatim: zero error
            opts.append((int(t["src_bytes"]), 0.0, t["src_type"]))
        if not opts:
            raise ValueError(f"{n}: no measured type within its role's range")
        items.append(opts)
    budget = int(budget_bytes) - fixed
    if budget <= 0:
        raise ValueError(f"budget {budget_bytes} B is below the fixed tensors' {fixed} B")
    res = knapsack(items, budget, max_cells=max_cells)
    chosen = dict(zip(names, res["choice"]))
    out = summarize(em, chosen, fixed, res, head, embd, sens is not None, probe)
    if LB:
        out["estimate"] = "measured+lowbit"
    return out


def _interp_rel(rel, q):
    """log-linear interpolation of a missing type's error in bpw."""
    pts = sorted((bpw_of(k), math.log(v)) for k, v in rel.items() if v and k in QUANT_SIZES)
    if len(pts) < 2:
        return None
    x = bpw_of(q)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if x < pts[0][0] else (pts[-2], pts[-1])
    return math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def summarize(em, chosen, fixed, res, head, embd, measured, probe):
    tensors = em["tensors"]
    hist = {}
    by_role = {}
    for n, q in chosen.items():
        r = tensors[n]["role"]
        hist.setdefault(r, {}).setdefault(q, 0)
        hist[r][q] += 1
        br = by_role.setdefault(r, [0, 0])
        br[0] += tensors[n]["numel"]
        br[1] += tensors[n]["bytes"].get(q, type_bytes(q, tensors[n]["numel"]))
    tot_numel = sum(t["numel"] for t in tensors.values()) + sum(f["numel"] for f in em.get("fixed", {}).values())
    size = int(res["size"]) + int(fixed)
    return {"format": FORMAT_PLAN, "types": chosen, "bytes": size, "gb": round(size / 1e9, 4),
            "bpw_file": round(8.0 * size / tot_numel, 4),
            "bpw_role": {r: round(8.0 * b / k, 4) for r, (k, b) in by_role.items()},
            "hist": hist, "predicted_kld": round(float(res["cost"]), 6), "estimate": "measured" if measured else "relerr",
            "probe": probe, "head": head, "embd": embd, "granularity": res["granularity"], "exact": res["exact"],
            "source": em.get("source")}


def write_tensor_types(p, path):
    """the plan as a llama-quantize --tensor-type-file: one anchored, escaped pattern per tensor (llama-quantize's
    patterns are unanchored first-match searches -- an anchor keeps blk.1 off blk.11 and off dec.blk.1)."""
    with open(path, "w") as f:
        for n, q in sorted(p["types"].items()):
            f.write(f"^{re.escape(n)}$={q}\n")
    return path


def uniform_plan(em, qtype, head=DEFAULT_HEAD, embd=DEFAULT_EMBD, roles=(ROLE_EXPERT,), floors=None):
    """a reference plan: every tensor of `roles` at qtype, everything else at its role floor -- the plain
    "experts at X" build to compare a knapsack rung against at matched bytes."""
    floors = {**DEFAULT_FLOORS, **(floors or {})}
    chosen = {}
    for n, t in em["tensors"].items():
        chosen[n] = qtype if t["role"] in roles else floors.get(t["role"], "iq4_xs")
    size = sum(em["tensors"][n]["bytes"].get(q, type_bytes(q, em["tensors"][n]["numel"])) for n, q in chosen.items())
    fixed = fixed_bytes(em, head, embd)
    return summarize(em, chosen, fixed, {"size": size, "cost": predicted_cost(em, chosen), "granularity": 1,
                                         "exact": True}, head, embd, False, None)


def predicted_cost(em, chosen, sens=None, probe="iq2_xxs"):
    S = tensor_sensitivity(em["tensors"], sens)
    c = 0.0
    for n, q in chosen.items():
        t = em["tensors"][n]
        e = t["rel"].get(q) or _interp_rel(t["rel"], q) or 0.0
        ref = (t["rel"].get(probe) or _interp_rel(t["rel"], probe) or 1.0) if sens else 1.0
        c += S[n] * e / ref
    return c


# ------------------------------------------------------------------------------------------------ build

def build(gguf_path, imatrix, p, out, quantize="llama-quantize", threads=None, log=print, allow_requantize=False):
    """llama-quantize with the plan's per-tensor types (base preset Q8_0 for anything unlisted).

    allow_requantize: the source is ALREADY quantized (a vendor QAT GGUF -- Bolt's IQ2_XXS experts). llama-quantize
    copies a tensor whose planned type equals its stored type byte-for-byte, so a plan that keeps a QAT tensor at its
    own type keeps the QAT bytes; only the tensors the plan moves are requantized."""
    tt = out + ".tensor-types.txt"
    write_tensor_types(p, tt)
    cmd = [quantize] + (["--allow-requantize"] if allow_requantize else []) + ["--imatrix", imatrix, "--tensor-type-file", tt,
           "--token-embedding-type", p.get("embd", DEFAULT_EMBD), "--output-tensor-type", p.get("head", DEFAULT_HEAD),
           gguf_path, out, "Q8_0"]
    if threads:
        cmd.append(str(threads))
    log("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not _gguf_ok(out):
        raise SystemExit("llama-quantize failed:\n" + "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:]))
    return check_build(out, p)


def check_build(path, p):
    """read the built file back: every planned tensor carries the planned type, and the size matches the plan."""
    import gguf
    r = gguf.GGUFReader(path)
    got = {t.name: gguf.GGMLQuantizationType(t.tensor_type).name.lower() for t in r.tensors}
    wrong = {n: (q, got.get(n)) for n, q in p["types"].items() if got.get(n, "").lower() != q.lower()}
    size = os.path.getsize(path)
    return {"path": path, "bytes": size, "planned_bytes": p["bytes"], "mismatched": wrong,
            "delta_pct": round(100.0 * (size - p["bytes"]) / max(p["bytes"], 1), 3)}


# ------------------------------------------------------------------------------------------------ EXL3 / EXLR8 units

def exl3_expert_units(budgeted, routing, sens, probe_bits=4):
    """[(key, numel, S)] for EXL3's budgeted Linears with each routed expert's S scaled by its traffic.

    budgeted: [(module key, numel)] (pollard_exl3.exl3_budgeted_keys). routing: {layer: {"share": [E]}} (from the
    GGUF imatrix of the same checkpoint -- llama.cpp keeps HF's expert order). An expert key looks like
    model.layers.L.mlp.experts.X.gate_proj; its S = S_group(L) * numel share * E * share_x, so a layer's experts
    together keep the layer's measured sensitivity and an expert that never fires gets ~0."""
    import pollard_exl3 as X
    keys = [k for k, _ in budgeted]
    numel = dict(budgeted)
    grp = {g: {int(k): float(v) for k, v in ((sens or {}).get(g) or {}).items() if v is not None}
           for g in ("attn", "ffn")}
    allv = sorted(v for d in grp.values() for v in d.values())
    fallback = allv[len(allv) // 2] if allv else 1.0
    gsize = {}
    for k in keys:
        key = (X._group_of(k), X._layer_of(k))
        gsize[key] = gsize.get(key, 0) + numel[k]
    out = []
    for k in keys:
        g, L = X._group_of(k), X._layer_of(k)
        sg = grp[g].get(L, grp["ffn" if g == "attn" else "attn"].get(L, fallback)) if (sens and L is not None) else 1.0
        s = max(sg, 0.0) * numel[k] / max(gsize[(g, L)], 1)
        m = re.search(r"\.experts\.(\d+)\.", k)
        prof = ((routing or {}).get(str(L)) or (routing or {}).get(L)) if m else None
        if prof:
            sh = prof["share"]
            x = int(m.group(1))
            if x < len(sh):
                s *= len(sh) * float(sh[x])
        out.append((k, numel[k], s))
    return out


def exl3_kld(S, K, probe_bits=4):
    """pollard-exl3's error law (sc_optimize's, steeper below K2)."""
    import pollard_exl3 as X
    return X.kld_law(S, K, probe_bits)


def exl3_recipe(budgeted, routing, sens, target_bpw, head_bits=8, min_bits=1, max_bits=8, granularity="layer",
                expert_floor=None, nonexpert_floor=3, probe_bits=4, tie=None):
    """EXL3 -rcp recipe at target_bpw (mean bits over EXL3's budgeted Linears), exact knapsack over integer K.

    granularity "layer": one K per (layer, projection) across all its experts -- what STOCK exllamav3 loads: its MoE
    fast path builds a MultiLinear per projection that asserts one K for every expert (modules/multilinear.py). The
    layer's choice is still driven by per-expert traffic (the summed S). "expert": one K per expert matrix (the SAGE
    shape) -- needs an exllamav3 that drops mixed-K layers to the per-expert path; flagged in the recipe.
    Gate and up of one expert always share K (EXL3's gate_proj+up_proj tie)."""
    import pollard_exl3 as X
    from pollard_exlr8 import knapsack
    tie = tie or X.EXL3_TIE
    units = exl3_expert_units(budgeted, routing, sens, probe_bits)
    expert = lambda k: re.search(r"\.experts\.\d+\.", k) is not None
    # group: tie sets first (gate+up of one module), then the layer granularity merges experts of one projection
    groups = X._ties([k for k, _, _ in units], tie)
    if granularity == "layer":
        merged, index = [], {}
        for g in groups:
            k0 = g[0]
            if expert(k0):
                proj = "gu" if re.search(r"(gate_proj|up_proj|w1|w3)$", k0) else "down"
                gk = (re.sub(r"\.experts\.\d+\..*$", "", k0), proj)
                if gk not in index:
                    index[gk] = len(merged)
                    merged.append([])
                merged[index[gk]].extend(g)
            else:
                merged.append(list(g))
        groups = merged
    info = {k: (n, s) for k, n, s in units}
    total = sum(n for _, n, _ in units)
    items = []
    for g in groups:
        n = sum(info[k][0] for k in g)
        s = sum(info[k][1] for k in g)
        lo = (expert_floor if expert(g[0]) else nonexpert_floor) or min_bits
        lo = max(min_bits, lo)
        opts = [(n * K, exl3_kld(s, K, probe_bits), K) for K in range(lo, max_bits + 1)]
        items.append(opts)
    unit = 16 * 16                                     # one trellis tile per bit: keeps the DP small, exact on tiles
    items_u = [[(-(-sz // unit), c, K) for sz, c, K in opts] for opts in items]
    res = knapsack(items_u, int(target_bpw * total) // unit)
    bits = {}
    for g, K in zip(groups, res["choice"]):
        for k in g:
            bits[k] = int(K)
    used = sum(info[k][0] * bits[k] for k in bits)
    hist = {}
    for k, K in bits.items():
        if expert(k):
            hist[K] = hist.get(K, 0) + 1
    return {"model": None, "source": "pollard-moefit", "target_bpw": float(target_bpw),
            "achieved_bpw": round(used / total, 4), "predicted_kld": round(float(res["cost"]), 6),
            "head_bits": head_bits, "tie": tie, "granularity": granularity,
            "needs_mixed_k_moe": granularity == "expert" and _mixed_within_layer(bits),
            "expert_k_hist": dict(sorted(hist.items())),
            "tensors": {k: bits[k] for k, _ in budgeted}}


def _mixed_within_layer(bits):
    seen = {}
    for k, K in bits.items():
        m = re.search(r"^(.*)\.experts\.\d+\.(\w+)$", k)
        if m:
            seen.setdefault((m.group(1), m.group(2)), set()).add(K)
    return any(len(v) > 1 for v in seen.values())


# ------------------------------------------------------------------------------------------------ measured low-bit cost

FORMAT_LOWBIT = "pollard-moefit-lowbit"
FORMAT_AUTO = "pollard-moefit-auto"
#: the types whose cost is MEASURED per class (the 1-bit pair + the 2-bit anchor every higher type scales from)
LOWBIT_TYPES = ("iq2_xxs", "iq1_m", "iq1_s")


def expert_class(name, granularity="proj"):
    """the class a routed-expert tensor is measured in: "<layer>.<proj>" (gate / up / down / gate_up) or "<layer>"."""
    m = re.match(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$", name)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}" if granularity == "proj" else m.group(1)


def resolve_lowbit(lb, tensors):
    """{tensor: {type: KL}} from a lowbit measurement: a class of one tensor is that tensor's cost; a class of
    several (granularity "layer") is split by each tensor's share of the class's measured output error at that
    type, w_t(q) = numel_t * e(t, q)."""
    if not lb:
        return {}
    if "tensor_costs" in lb:
        return {n: dict(c) for n, c in lb["tensor_costs"].items() if n in tensors}
    out = {}
    for cls in (lb.get("classes") or {}).values():
        names = [n for n in cls.get("tensors", []) if n in tensors]
        for q, kl in (cls.get("kl") or {}).items():
            if kl is None or not names:
                continue
            w = {n: tensors[n]["numel"] * float(tensors[n]["rel"].get(q) or _interp_rel(tensors[n]["rel"], q) or 0)
                 for n in names}
            tot = sum(w.values())
            for n in names:
                share = w[n] / tot if tot > 0 else 1.0 / len(names)
                out.setdefault(n, {})[q] = max(0.0, float(kl)) * share
    return out


def lowbit_cost(meas, t, q):
    """KL of tensor t at type q from its measured anchors: the measurement itself, else the nearest measured type at
    or above q (the highest anchor when q is above them all) scaled by the measured output-error ratio."""
    if q in meas:
        return float(meas[q])
    anchors = sorted(meas, key=bpw_of)
    above = [a for a in anchors if bpw_of(a) >= bpw_of(q)]
    a = above[0] if above else anchors[-1]
    ea = float(t["rel"].get(a) or _interp_rel(t["rel"], a) or 0.0)
    eq = float(t["rel"].get(q) or _interp_rel(t["rel"], q) or 0.0)
    if ea <= 0:
        return float(meas[a])
    return float(meas[a]) * eq / ea


def _kld(perplexity, model, eval_path, base, ngl=99, chunks=None):
    """llama-perplexity --kl-divergence against stored base logits -> {mean_kld, median_kld, p99_kld, top1, ppl,
    ref_ppl}; every value None when the run failed."""
    cmd = [perplexity, "-m", model, "-f", eval_path, "--kl-divergence", "--kl-divergence-base", base,
           "-ngl", str(ngl), "-c", "512"] + (["--chunks", str(chunks)] if chunks else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")

    def g(pat):
        m = re.search(pat, out, re.M)
        return float(m.group(1)) if m else None
    return {"mean_kld": g(r"^Mean\s+KLD:\s*(-?[0-9.]+)"), "median_kld": g(r"^Median\s+KLD:\s*(-?[0-9.]+)"),
            "p99_kld": g(r"^99\.0%\s+KLD:\s*([0-9.]+)"), "top1": g(r"^Same top p:\s*([0-9.]+)"),
            "ppl": g(r"^Mean PPL\(Q\)\s*:\s*([0-9.]+)"), "ref_ppl": g(r"^Mean PPL\(base\)\s*:\s*([0-9.]+)")}


def _base_logits(perplexity, model, eval_path, out, ngl=99, chunks=None):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    cmd = [perplexity, "-m", model, "-f", eval_path, "--kl-divergence-base", out, "-ngl", str(ngl), "-c", "512"] + \
        (["--chunks", str(chunks)] if chunks else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise SystemExit("could not write the base logits:\n" + "\n".join((r.stderr or r.stdout).strip().splitlines()[-8:]))
    return out


def _quantized_source(path):
    import gguf
    r = gguf.GGUFReader(path)
    for t in r.tensors:
        if role_of(t.name, len(t.shape)) == ROLE_EXPERT:
            return gguf.GGMLQuantizationType(t.tensor_type).name.lower() not in ("f32", "f16", "bf16")
    return False


def _quantize(quantize, src, imatrix, out, lines, requant, threads=None):
    tt = out + ".tensor-types.txt"
    with open(tt, "w") as f:
        f.write("".join(x + "\n" for x in lines))
    cmd = [quantize] + (["--allow-requantize"] if requant else []) + ["--imatrix", imatrix] + \
        (["--tensor-type-file", tt] if lines else []) + [src, out, "Q8_0"] + ([str(threads)] if threads else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.path.exists(tt) and os.remove(tt)
    if r.returncode != 0 or not _gguf_ok(out):
        return "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-6:])
    return None


def lowbit(gguf_path, imatrix, eval_path, out, types=LOWBIT_TYPES, granularity="proj", quantize="llama-quantize",
           perplexity="llama-perplexity", ngl=99, workdir=None, threads=None, log=print, chunks=None):
    """MEASURE each routed-expert class's KL at each low type: the class alone is quantized to q (everything else
    as the reference build), llama-perplexity --kl-divergence on the held-out eval against the source's logits, minus
    the reference build's own KL. One build + one forward pass per (class, type); resumable (the out file is
    rewritten after every probe and a re-run skips what it holds).

    This is the 1-bit cost the allocator uses instead of extrapolating the 2-bit probe's sensitivity down the
    relative-error curve: on granite-3.1-1b-a400m that extrapolation priced iq1_s experts below what they cost, and
    the knapsack bought 1-bit where 2-bit measured better."""
    import gguf
    src = gguf.GGUFReader(gguf_path)
    classes = {}
    for t in src.tensors:
        c = expert_class(t.name, granularity)
        if c is not None:
            classes.setdefault(c, []).append(t.name)
    if not classes:
        raise SystemExit("no routed-expert tensors (ffn_*_exps) in this GGUF -- lowbit is for a MoE")
    order = sorted(classes, key=lambda c: tuple(int(x) if x.isdigit() else x for x in c.split(".")))
    requant = _quantized_source(gguf_path)
    state = {}
    if out and os.path.exists(out):
        try:
            state = json.load(open(out))
        except ValueError:
            state = {}
    same = (state.get("format") == FORMAT_LOWBIT and state.get("source") == os.path.abspath(gguf_path)
            and state.get("eval") == os.path.abspath(eval_path) and state.get("granularity") == granularity)
    if not same:
        state = {"format": FORMAT_LOWBIT, "source": os.path.abspath(gguf_path), "imatrix": os.path.abspath(imatrix),
                 "eval": os.path.abspath(eval_path), "granularity": granularity, "types": list(types),
                 "ref_kl": None, "classes": {c: {"tensors": classes[c], "kl": {}} for c in order}}
    state["types"] = sorted(set(state.get("types", [])) | set(types), key=bpw_of)
    work = workdir or tempfile.mkdtemp(prefix="pollard_lowbit_", dir=_work_root())
    os.makedirs(work, exist_ok=True)
    base = _base_logits(perplexity, gguf_path, eval_path, os.path.join(work, "lowbit-base.kld"), ngl, chunks)
    probe = os.path.join(work, "lowbit-probe.gguf")

    def save():
        if out:
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=1)
            os.replace(tmp, out)
    if state.get("ref_kl") is None:
        err = _quantize(quantize, gguf_path, imatrix, probe, [], requant, threads)
        if err:
            raise SystemExit(f"reference build failed:\n{err}")
        state["ref_kl"] = _kld(perplexity, probe, eval_path, base, ngl, chunks)["mean_kld"] or 0.0
        os.remove(probe)
        save()
    ref = float(state["ref_kl"])
    n_total = len(order) * len(types)
    done, t0 = 0, time.time()
    log(f"  lowbit: {len(order)} classes x {list(types)}  reference build KL {ref:.5f}")
    for c in order:
        rec = state["classes"].setdefault(c, {"tensors": classes[c], "kl": {}})
        for q in types:
            done += 1
            if rec["kl"].get(q) is not None:
                continue
            err = _quantize(quantize, gguf_path, imatrix, probe,
                            [f"^{re.escape(n)}$={q}" for n in classes[c]], requant, threads)
            if err:
                log(f"    {c} {q}: build FAILED ({err.splitlines()[-1] if err else ''}) -- left unmeasured")
                continue
            k = _kld(perplexity, probe, eval_path, base, ngl, chunks)["mean_kld"]
            os.remove(probe)
            rec["kl"][q] = None if k is None else round(max(0.0, k - ref), 6)
            save()
        el = time.time() - t0
        log(f"    {c:10} " + "  ".join(f"{q}={rec['kl'].get(q)}" for q in types)
            + f"   [{done}/{n_total}  {el:5.0f}s]")
    if workdir is None:
        shutil.rmtree(work, ignore_errors=True)
    save()
    return state


# ------------------------------------------------------------------------------------------------ pruned candidates

def prune_errmap(em, keep):
    """the error map of the model with only `keep` = {layer: [expert indices]} left, derived from the FULL model's
    per-expert measurement: each routed-expert tensor keeps its kept experts' rows (bytes scale exactly -- one
    expert is a whole number of blocks), its relative error is re-aggregated over the kept experts with their
    traffic, the router shrinks with it. Nothing is re-measured: it is the screening estimate the auto search
    ranks candidates with before it builds and measures the best of them."""
    import copy
    out = copy.deepcopy(em)
    for n, t in out["tensors"].items():
        if t["role"] != ROLE_EXPERT or t.get("layer") not in keep and str(t.get("layer")) not in keep:
            continue
        idx = list(keep.get(t["layer"], keep.get(str(t["layer"]))))
        E = int(t["shape"][0])
        k = len(idx)
        f = k / E
        t["shape"] = [k] + list(t["shape"][1:])
        t["numel"] = int(t["numel"] * f)
        t["bytes"] = {q: int(b * k // E) for q, b in t["bytes"].items()}
        if "src_bytes" in t:
            t["src_bytes"] = int(t["src_bytes"] * k // E)
        ex = t.get("experts")
        if ex:
            cnt = np.asarray(ex["count"], dtype=np.float64)[idx]
            en = np.asarray(ex["energy"], dtype=np.float64)[idx]
            errs = {q: list(np.asarray(v, dtype=np.float64)[idx]) for q, v in ex["err"].items()}
            t["experts"] = {"count": [float(x) for x in cnt], "energy": [float(x) for x in en],
                            "err": {q: [float(x) for x in v] for q, v in errs.items()}}
            for q, v in errs.items():
                t["rel"][q] = aggregate(np.asarray(v) * en, en, cnt)
    for n, f_ in out.get("fixed", {}).items():
        L = layer_of(n)
        if f_["role"] == ROLE_ROUTER and L is not None and (L in keep or str(L) in keep):
            idx = keep.get(L, keep.get(str(L)))
            E = int(em.get("n_experts") or 0) or _n_experts(em, L)
            if E:
                f_["numel"] = int(f_["numel"] * len(idx) // E)
                f_["src_bytes"] = int(f_["src_bytes"] * len(idx) // E)
    out["routing"] = {L: {"counts": [p["counts"][i] for i in keep.get(int(L), keep.get(str(L), range(len(p["counts"]))))],
                          "share": None} for L, p in (em.get("routing") or {}).items()}
    for p in out["routing"].values():
        tot = sum(p["counts"]) or 1.0
        p["share"] = [c / tot for c in p["counts"]]
    out["pruned_from"] = em.get("source")
    return out


def _n_experts(em, L):
    for t in em["tensors"].values():
        if t["role"] == ROLE_EXPERT and t.get("layer") == L:
            return int(t["shape"][0])
    return 0


def prune_lowbit(lb, em_full, em_pruned):
    """the measured class costs carried onto a pruned candidate: each tensor's measured KL scaled by the ratio of
    its kept experts' relative error to the full tensor's at that type (screening estimate; the build is judged by
    its own measurement)."""
    if not lb:
        return None
    full = resolve_lowbit(lb, em_full["tensors"])
    out = {}
    for n, costs in full.items():
        tf, tp = em_full["tensors"][n], em_pruned["tensors"].get(n)
        if tp is None:
            continue
        out[n] = {}
        for q, kl in costs.items():
            ef = float(tf["rel"].get(q) or 0.0)
            ep = float(tp["rel"].get(q) or 0.0)
            out[n][q] = kl * ep / ef if ef > 0 else kl
    return {"tensor_costs": out}


def slice_imatrix(imatrix, keep, out):
    """a GGUF imatrix for the pruned model: every routed-expert entry keeps only the kept experts' rows (in_sum2 and
    counts), so llama-quantize finds an entry of the right size for each pruned expert tensor; everything else, and
    the datasets / chunk metadata, is copied."""
    import gguf
    from pollard_prune import _copy_field
    r = gguf.GGUFReader(imatrix)
    w = gguf.GGUFWriter(out, "imatrix")
    for field in r.fields.values():
        if field.name in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        _copy_field(w, field)
    for t in r.tensors:
        arr = np.asarray(t.data)
        base = re.sub(r"\.(in_sum2|counts)$", "", t.name)
        m = re.match(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$", base)
        if m:
            L = int(m.group(1))
            idx = keep.get(L, keep.get(str(L)))
            if idx is not None and arr.ndim >= 1 and arr.shape[0] > len(idx):
                arr = np.take(arr, list(idx), axis=0)
        w.add_tensor(t.name, np.ascontiguousarray(arr.astype(np.float32)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def _gguf_experts(path):
    import gguf
    r = gguf.GGUFReader(path)
    n_exp = n_used = None
    for f in r.fields.values():
        if f.name.endswith(".expert_count"):
            n_exp = int(f.parts[f.data[0]][0])
        elif f.name.endswith(".expert_used_count"):
            n_used = int(f.parts[f.data[0]][0])
    return n_exp, n_used


def keep_sets(n_exp, n_used, prunes, scores):
    """{fraction: (K, {layer: kept indices})} for each prune fraction, K = round((1 - p) * E) experts per layer, the
    top-K by score in each layer; fractions that would leave fewer than the router's top-k (or repeat a K) dropped."""
    out, seen = {}, set()
    for p in sorted(set(float(x) for x in prunes)):
        K = int(round((1.0 - p) * n_exp))
        if K < max(1, n_used or 1) or K > n_exp or K in seen:
            continue
        seen.add(K)
        keep = {int(L): sorted(int(i) for i in np.argsort(np.asarray(sc))[::-1][:K]) for L, sc in scores.items()}
        out[round(p, 4)] = (K, keep)
    return out


def expert_scores(reap=None, imx=None):
    """{layer: per-expert keep score}: REAP saliency from a pollard-reap file when given, else the router's token
    counts on the calibration corpus (the imatrix's per-expert counts) -- frequency, the cheap REAP proxy."""
    if reap:
        d = json.load(open(reap)) if isinstance(reap, str) else reap
        return {int(L): list(v["reap"]) for L, v in d["layers"].items()}, "reap"
    prof = routing_profile(imx or {})
    if not prof:
        raise SystemExit("no expert scores: give --reap (pollard-reap) or an imatrix with per-expert counts")
    return {int(L): p["counts"] for L, p in prof.items()}, "router-count"


def _min_bytes(em, allow_1bit, floors=None, head=DEFAULT_HEAD, embd=DEFAULT_EMBD):
    b = fixed_bytes(em, head, embd)
    for t in em["tensors"].values():
        al = [q for q in allowed_types(t["role"], em["types"], floors, DEFAULT_CEIL, allow_1bit) if q in t["rel"]]
        b += min(t["bytes"].get(q, type_bytes(q, t["numel"])) for q in al) if al else t.get("src_bytes", 0)
    return b


def _prune_gguf(src, K, keep_list_path, scores_path, out, log=print):
    """the pruned GGUF via pollard-prune (same writer the release uses)."""
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pollard_prune.py"),
           "--gguf", src, "--keep-experts", str(K), "--score", "reap", "--scores", scores_path, "--out", out,
           "--keep-list", keep_list_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not _gguf_ok(out):
        raise SystemExit("pollard-prune failed:\n" + "\n".join((r.stderr or r.stdout).strip().splitlines()[-8:]))
    return out


def _toolgate(rung, ref, ngl, out_json, log=print):
    """pollard-toolcall on the rung vs the reference (served llama-server, Pollard's reader on the content)."""
    if not os.path.exists(out_json):
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pollard_toolcall.py"),
               "--gguf", rung, "--ref", ref, "--ngl", str(ngl), "--out", out_json]
        subprocess.run(cmd, capture_output=True, text=True)
    try:
        d = json.load(open(out_json))
    except (OSError, ValueError):
        return {"tool_status": "not-measured"}
    sc, rs = d.get("score") or {}, d.get("ref_score") or {}
    ret = d.get("retention")
    return {"tool_status": d.get("status", "measured"), "tool_rate": sc.get("rate"), "tool_ref_rate": rs.get("rate"),
            "tool_parse": sc.get("parse_rate"), "tool_ref_parse": rs.get("parse_rate"),
            "tool_retention": round(100 * ret, 1) if ret is not None else None, "tool_n": sc.get("n"),
            "tool_verdict": d.get("verdict")}


def auto(src, imatrix, em, sens, lb, budget_bytes, prunes=(0.0, 0.10, 0.25, 0.40, 0.50), reap=None,
         screen_eval=None, judge_eval=None, workdir=None, out=None, quantize="llama-quantize",
         perplexity="llama-perplexity", ngl=99, top=3, toolcall=True, tool_ref=None, head=DEFAULT_HEAD,
         embd=DEFAULT_EMBD, controls=True, min_tool_retention=90.0, judge_chunks=None, keep_candidates=False,
         log=print):
    """THE AUTO-AWARE MoE ALLOCATION: prune fraction x bit mix at ONE byte budget, chosen by measurement.

      1. screen every prune fraction p (0 = no pruning): K = round((1-p) E) experts kept per layer by REAP score;
         the pruned model is written and its OWN damage measured (KL vs the full model's logits on the screening
         eval -- the allocation eval, not the judge's); the bit mix for its remaining experts is the exact
         knapsack at the budget on the full model's error map restricted to the kept experts, with the measured
         low-bit costs. Screening score = prune KL + the knapsack's predicted KL.
      2. measure the `top` best for real: build at the budget, llama-perplexity --kl-divergence vs the FULL
         model on the held-out judge eval (mean / p99 KLD, top-1, PPL) and the tool gate vs the full model. When a
         measured plan uses 1-bit experts, its no-1-bit twin at the same bytes is measured beside it (`controls`).
      3. the winner = lowest measured mean KLD among the candidates whose tool-call retention holds
         (>= min_tool_retention, or not measurable); its file goes to `out`.

    Returns the report (every candidate's screening + measured numbers, the winner and why)."""
    from pollard_prune import reap_layers  # noqa: F401 -- same scorer the CLI prune uses
    work = workdir or tempfile.mkdtemp(prefix="pollard_moeauto_", dir=_work_root())
    os.makedirs(work, exist_ok=True)
    screen_eval = screen_eval or judge_eval
    judge_eval = judge_eval or screen_eval
    if not screen_eval:
        raise SystemExit("auto needs held-out text: --screen-eval (allocation) and --eval (judge)")
    n_exp, n_used = _gguf_experts(src)
    if not n_exp or n_exp < 2:
        raise SystemExit("not a MoE: no expert_count in the GGUF")
    imx, _ = read_imatrix(imatrix)
    scores, score_kind = expert_scores(reap, imx)
    scores_path = os.path.join(work, "expert-scores.json")
    json.dump({"format": "pollard-reap", "score": score_kind,
               "layers": {str(L): {"reap": v, "count": v, "ean": v} for L, v in scores.items()}},
              open(scores_path, "w"))
    requant = _quantized_source(src)
    sbase = _base_logits(perplexity, src, screen_eval, os.path.join(work, "screen-base.kld"), ngl)
    jbase = _base_logits(perplexity, src, judge_eval, os.path.join(work, "judge-base.kld"), ngl, judge_chunks)
    sets = keep_sets(n_exp, n_used, prunes, scores)
    log(f"== pollard-moefit auto :: {os.path.basename(src)}  {n_exp} experts (top-{n_used})  budget "
        f"{budget_bytes / 1e9:.4f} GB  prune fractions {sorted(sets)}  scores {score_kind}")
    cands, models = [], {}
    for p, (K, keep) in sorted(sets.items()):
        tag = f"p{int(round(100 * p)):02d}"
        if p == 0:
            model, im, emp, lbp, pkl = src, imatrix, em, lb, 0.0
        else:
            model = os.path.join(work, f"{tag}-src.gguf")
            klp = os.path.join(work, f"{tag}.keep.json")
            if not _gguf_ok(model):
                _prune_gguf(src, K, klp, scores_path, model, log)
            im = os.path.join(work, f"{tag}.imatrix.gguf")
            if not _gguf_ok(im):
                slice_imatrix(imatrix, keep, im)
            emp = prune_errmap(em, keep)
            lbp = prune_lowbit(lb, em, emp)
            pkl = _kld(perplexity, model, screen_eval, sbase, ngl)["mean_kld"]
        models[p] = (model, im, emp, lbp, K, tag)
        for one in (True, False):
            row = {"prune": p, "keep": K, "experts": n_exp, "tag": tag, "allow_1bit": one, "prune_kl": pkl}
            if _min_bytes(emp, one, head=head, embd=embd) > budget_bytes:
                row.update(feasible=False)
                cands.append(row)
                continue
            try:
                pl = plan(emp, budget_bytes, sens, head=head, embd=embd, allow_1bit=one, lowbit=lbp)
            except ValueError as e:
                row.update(feasible=False, why=str(e)[:120])
                cands.append(row)
                continue
            n1 = sum(v for q, v in pl["hist"].get(ROLE_EXPERT, {}).items() if q in TYPES_1BIT)
            row.update(feasible=True, predicted=pl["predicted_kld"], plan=pl, bytes=pl["bytes"],
                       expert_bpw=pl["bpw_role"].get(ROLE_EXPERT), n_1bit=n1,
                       score=(pkl or 0.0) + pl["predicted_kld"])
            cands.append(row)
        log(f"   {tag}  keep {K}/{n_exp}  prune KL {pkl if pkl is not None else '-'}  " + "  ".join(
            f"{'1bit' if r['allow_1bit'] else 'no1bit'}: " + (f"pred {r['predicted']:.4f} score {r['score']:.4f}"
                                                              f" ({r['n_1bit']} 1-bit)" if r.get("feasible") else "infeasible")
            for r in cands if r["tag"] == tag))
    # a no-1-bit row whose plan equals the 1-bit row's is the same build: keep one
    feas = [r for r in cands if r.get("feasible")]
    uniq = {}
    for r in sorted(feas, key=lambda r: r["score"]):
        key = (r["prune"], json.dumps(r["plan"]["types"], sort_keys=True))
        uniq.setdefault(key, r)
    ranked = sorted(uniq.values(), key=lambda r: r["score"])
    for i, r in enumerate(ranked):
        r["screen_rank"] = i + 1
    chosen = ranked[:max(1, top)]
    if controls:
        for r in list(chosen):
            if r["n_1bit"]:
                twin = next((x for x in ranked if x["prune"] == r["prune"] and not x["allow_1bit"]), None)
                if twin is not None and twin not in chosen:
                    twin["control_for"] = r["screen_rank"]
                    chosen.append(twin)
    log(f"   measuring {len(chosen)} candidate(s) on the judge eval" + (" + the tool gate" if toolcall else ""))
    for r in chosen:
        model, im, _, _, K, tag = models[r["prune"]]
        name = f"{tag}-{'1bit' if r['allow_1bit'] else 'no1bit'}"
        path = os.path.join(work, f"cand-{name}-{int(budget_bytes)}.gguf")       # one work dir, many budgets
        pj = path + ".plan.json"
        json.dump(r["plan"], open(pj, "w"), indent=1)
        if not _gguf_ok(path):
            build(model, im, r["plan"], path, quantize, allow_requantize=requant)
        r["file"] = path
        r["bytes_built"] = os.path.getsize(path)
        r.update({k: v for k, v in _kld(perplexity, path, judge_eval, jbase, ngl, judge_chunks).items()})
        if toolcall:
            r.update(_toolgate(path, tool_ref or src, ngl, path + ".toolcall.json", log))
        log(f"   {name:12} {r['bytes_built'] / 1e9:.4f} GB  KLD {r.get('mean_kld')}  p99 {r.get('p99_kld')}  "
            f"top-1 {r.get('top1')}  PPL {r.get('ppl')}" + (f"  tools {r.get('tool_retention')}% "
                                                             f"(rate {r.get('tool_rate')})" if toolcall else ""))
    measured = [r for r in chosen if r.get("mean_kld") is not None]

    def tool_ok(r):
        ret = r.get("tool_retention")
        return ret is None or ret >= min_tool_retention
    pool = [r for r in measured if tool_ok(r)] or measured
    if not pool:
        raise SystemExit("no candidate could be measured")
    win = min(pool, key=lambda r: r["mean_kld"])
    win["winner"] = True
    others = sorted((r for r in measured if r is not win), key=lambda r: r["mean_kld"])
    why = (f"{int(round(100 * win['prune']))}% of routed experts pruned (keep {win['keep']}/{n_exp} per layer), "
           f"{'1-bit experts allowed' if win['allow_1bit'] else 'experts >= 2-bit'}: measured mean KLD "
           f"{win['mean_kld']:.4f}")
    if others:
        o = others[0]
        why += (f", next best {o['mean_kld']:.4f} at {int(round(100 * o['prune']))}% pruned"
                f"{' (1-bit)' if o['allow_1bit'] and o.get('n_1bit') else ''}")
    if win.get("tool_retention") is not None:
        why += f"; tool-call retention {win['tool_retention']}%"
    if out:
        shutil.copyfile(win["file"], out)
    rows = []
    for r in sorted(cands, key=lambda r: (not r.get("feasible"), r.get("score") or 9e9)):
        rows.append({k: v for k, v in r.items() if k not in ("plan",)} |
                    ({"hist": r["plan"]["hist"].get(ROLE_EXPERT, {})} if r.get("plan") else {}))
    rep = {"format": FORMAT_AUTO, "source": os.path.abspath(src), "budget_bytes": int(budget_bytes),
           "budget_gb": round(budget_bytes / 1e9, 4), "experts": n_exp, "top_k": n_used, "scores": score_kind,
           "screen_eval": os.path.abspath(screen_eval), "judge_eval": os.path.abspath(judge_eval),
           "lowbit": bool(lb), "candidates": rows, "winner": {k: v for k, v in win.items() if k != "plan"},
           "chosen_prune": win["prune"], "why": why, "out": os.path.abspath(out) if out else None}
    rep["winner"]["hist"] = win["plan"]["hist"].get(ROLE_EXPERT, {})
    if not keep_candidates:
        for r in chosen:
            if r.get("file") and os.path.exists(r["file"]) and (out or r is not win):
                os.remove(r["file"])
        if workdir is None:
            shutil.rmtree(work, ignore_errors=True)
    return rep


def auto_table(rep):
    """the report's candidate table (markdown), screening and measured side by side -- what the card shows."""
    lines = ["| prune | experts kept | 1-bit | screened KL (prune + mix) | measured mean KLD | top-1 % | PPL | "
             "tool retention | expert types |", "|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in rep["candidates"]:
        if not r.get("feasible"):
            continue
        h = " ".join(f"{q}:{n}" for q, n in sorted((r.get("hist") or {}).items(), key=lambda kv: bpw_of(kv[0])))
        mk = f"**{r['mean_kld']:.4f}**" if r.get("winner") else (f"{r['mean_kld']:.4f}" if r.get("mean_kld")
                                                                 is not None else "not built")
        tr = r.get("tool_retention")
        lines.append(f"| {int(round(100 * r['prune']))}% | {r['keep']}/{r['experts']} | "
                     f"{'allowed' if r['allow_1bit'] else 'no'} ({r.get('n_1bit', 0)} tensors) | "
                     f"{r['score']:.4f} | {mk} | {r.get('top1') if r.get('top1') is not None else '-'} | "
                     f"{r.get('ppl') if r.get('ppl') is not None else '-'} | "
                     f"{f'{tr}%' if tr is not None else '-'} | {h} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ CLI

def _budget(a, em):
    tot = sum(t["numel"] for t in em["tensors"].values()) + sum(f["numel"] for f in em.get("fixed", {}).values())
    if a.gb:
        return int(a.gb * 1e9)
    if a.bytes:
        return int(a.bytes)
    if a.bpw:
        return int(a.bpw * tot / 8)
    raise SystemExit("give a budget: --gb, --bytes or --bpw (over all weights)")


def _floors(spec):
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _engine(a, *names):
    """The llama.cpp binaries for a subcommand, resolved the way every other pipeline step resolves them:
    find_llama_bin with the model's architecture (set_model_arch from --gguf, as pollard / pollard-release /
    pollard-verify do), so a clef-omni run gets the engine tree that carries its patches -- not whichever
    llama-quantize comes first. An explicit path is honored as-is; an unresolved name is passed through."""
    from pollard_calc import find_llama_bin, set_model_arch
    if getattr(a, "gguf", None):
        set_model_arch(a.gguf)
    return tuple(find_llama_bin(n) or n for n in names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("routing", help="per-layer expert traffic from a GGUF imatrix")
    r.add_argument("--imatrix", required=True)
    r.add_argument("--out", help="write the routing profile JSON (pollard-moefit exl3-recipe / pollard-exlr8 --routing)")

    e = sub.add_parser("errmap", help="measure e(t, q) per tensor and per expert, one encode per type")
    e.add_argument("--gguf", required=True, help="source GGUF (f16/bf16)")
    e.add_argument("--imatrix", required=True, help="GGUF-format imatrix (per-expert counts)")
    e.add_argument("--out", required=True)
    e.add_argument("--types", default=",".join(DEFAULT_TYPES))
    e.add_argument("--llama-quantize", default="llama-quantize")
    e.add_argument("--workdir", help="where the probe builds go (default a temp dir; one probe on disk at a time)")
    e.add_argument("--keep-probes", action="store_true")
    e.add_argument("--threads", type=int)
    e.add_argument("--allow-requantize", action="store_true",
                   help="the source is already quantized (a QAT GGUF): errors are measured against ITS weights")

    p = sub.add_parser("plan", help="exact knapsack -> per-tensor types at a byte budget")
    p.add_argument("--errmap", required=True)
    p.add_argument("--sensitivity", help="pollard-sensitivity profile (measured group KL); without it: relative error")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gb", type=float, help="target file size, GB (1e9)")
    g.add_argument("--bytes", type=int)
    g.add_argument("--bpw", type=float, help="target bits per weight over ALL weights (the file's bpw)")
    p.add_argument("--no-1bit", action="store_true", help="routed experts stop at iq2_xxs")
    p.add_argument("--floors", help="role floors, e.g. expert=iq1_s,attn=iq4_xs,shared=q5_K,dense_ffn=iq4_xs")
    p.add_argument("--ceil", default=DEFAULT_CEIL)
    p.add_argument("--types", help="restrict the candidate types (comma list)")
    p.add_argument("--head", default=DEFAULT_HEAD)
    p.add_argument("--embd", default=DEFAULT_EMBD)
    p.add_argument("--uniform", help="instead of the knapsack: experts at this type, the rest at their floors")
    p.add_argument("--out", required=True)
    p.add_argument("--tensor-types", help="also write the llama-quantize --tensor-type-file")

    p.add_argument("--lowbit", help="pollard-moefit lowbit output: the experts' MEASURED KL at iq1_s / iq1_m / iq2_xxs")

    lb = sub.add_parser("lowbit", help="MEASURE each expert class's KL at the 1-bit types (+ the 2-bit anchor)")
    lb.add_argument("--gguf", required=True, help="source GGUF (f16/bf16, or the Q8_0 the errmap was taken from)")
    lb.add_argument("--imatrix", required=True)
    lb.add_argument("--eval", required=True, help="held-out text (the allocation eval: not the calibration, not "
                                                  "the text the rungs are judged on)")
    lb.add_argument("--out", required=True, help="cost file (resumable: a re-run skips what it holds)")
    lb.add_argument("--types", default=",".join(LOWBIT_TYPES))
    lb.add_argument("--granularity", choices=("proj", "layer"), default="proj",
                    help="proj = one class per layer x gate/up/down (3 x layers probes per type); layer = the "
                         "layer's experts together, split by measured error (for a big MoE)")
    lb.add_argument("--llama-quantize", default="llama-quantize")
    lb.add_argument("--llama-perplexity", default="llama-perplexity")
    lb.add_argument("--ngl", default="99")
    lb.add_argument("--workdir")
    lb.add_argument("--threads", type=int)

    au = sub.add_parser("auto", help="prune fraction x bit mix at one byte budget, chosen by measurement")
    au.add_argument("--gguf", required=True, help="source GGUF (the full model; the KL reference)")
    au.add_argument("--imatrix", required=True)
    au.add_argument("--errmap", required=True)
    au.add_argument("--sensitivity", required=True, help="pollard-sensitivity profile of the source")
    au.add_argument("--lowbit", help="pollard-moefit lowbit costs (strongly recommended: 1-bit priced by measurement)")
    au.add_argument("--reap", help="pollard-reap scores (else the imatrix's per-expert router counts)")
    g2 = au.add_mutually_exclusive_group(required=True)
    g2.add_argument("--gb", type=float)
    g2.add_argument("--bytes", type=int)
    g2.add_argument("--bpw", type=float)
    au.add_argument("--prunes", default="0,0.1,0.25,0.4,0.5", help="fractions of routed experts to remove")
    au.add_argument("--screen-eval", required=True, help="held-out text for the screening (allocation) KL")
    au.add_argument("--eval", required=True, help="held-out text the measured candidates are judged on")
    au.add_argument("--top", type=int, default=3, help="candidates built and measured for real")
    au.add_argument("--judge-chunks", type=int, help="llama-perplexity --chunks for the judge eval")
    au.add_argument("--no-controls", action="store_true", help="do not measure a 1-bit plan's no-1-bit twin")
    au.add_argument("--no-toolcall", action="store_true")
    au.add_argument("--tool-ref", help="tool-gate reference (default the source GGUF)")
    au.add_argument("--min-tool-retention", type=float, default=90.0)
    au.add_argument("--head", default=DEFAULT_HEAD)
    au.add_argument("--embd", default=DEFAULT_EMBD)
    au.add_argument("--out", required=True, help="the winning GGUF")
    au.add_argument("--report", help="the report JSON (default <out>.moefit.json)")
    au.add_argument("--workdir")
    au.add_argument("--keep-candidates", action="store_true")
    au.add_argument("--llama-quantize", default="llama-quantize")
    au.add_argument("--llama-perplexity", default="llama-perplexity")
    au.add_argument("--ngl", default="99")

    b = sub.add_parser("build", help="llama-quantize the plan")
    b.add_argument("--gguf", required=True)
    b.add_argument("--imatrix", required=True)
    b.add_argument("--plan", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--llama-quantize", default="llama-quantize")
    b.add_argument("--threads", type=int)
    b.add_argument("--allow-requantize", action="store_true", help="the source is already quantized (QAT GGUF)")

    x = sub.add_parser("exl3-recipe", help="EXL3 -rcp recipe with K = 1..8 per expert (or per layer projection)")
    x.add_argument("--model", required=True, help="HF checkpoint dir (EXL3's tensor list)")
    x.add_argument("--routing", help="routing profile JSON (pollard-moefit routing --out)")
    x.add_argument("--sensitivity")
    x.add_argument("--bpw", type=float, required=True)
    x.add_argument("--granularity", choices=("layer", "expert"), default="layer")
    x.add_argument("--min-bits", type=int, default=1)
    x.add_argument("--max-bits", type=int, default=8)
    x.add_argument("--expert-floor", type=int, default=1)
    x.add_argument("--nonexpert-floor", type=int, default=3, help="attention / shared / dense floor (bits)")
    x.add_argument("--head-bits", type=int, default=8)
    x.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "routing":
        imx, kind = read_imatrix(a.imatrix)
        prof = routing_profile(imx)
        if not prof:
            raise SystemExit("no per-expert entries in this imatrix (dense model, or a legacy .dat)")
        for row in routing_summary(prof):
            print(f"  layer {row['layer']:>3}  experts {row['experts']:>4}  entropy {row['entropy']:.3f}  "
                  f"top-25% share {row['top25_share']:.3f}  dead {row['dead']}")
        if a.out:
            json.dump({str(k): v for k, v in prof.items()}, open(a.out, "w"))
            print(f"wrote {a.out}")
        return 0

    if a.cmd == "errmap":
        q, = _engine(a, a.llama_quantize)
        types = [t.strip() for t in a.types.split(",") if t.strip()]
        bad = [t for t in types if t not in QUANT_SIZES]
        if bad:
            raise SystemExit(f"unknown types {bad}; known: {sorted(QUANT_SIZES)}")
        print(f"== pollard-moefit errmap :: {a.gguf}  types {types}")
        errmap(a.gguf, a.imatrix, a.out, types, q, a.workdir, a.keep_probes, a.threads,
               allow_requantize=a.allow_requantize)
        print(f"wrote {a.out}")
        return 0

    if a.cmd == "plan":
        em = json.load(open(a.errmap))
        sens = json.load(open(a.sensitivity)) if a.sensitivity else None
        floors = _floors(a.floors)
        if a.uniform:
            res = uniform_plan(em, a.uniform, a.head, a.embd, floors=floors)
        else:
            types = [t.strip() for t in a.types.split(",")] if a.types else None
            res = plan(em, _budget(a, em), sens, floors=floors, ceil=a.ceil, allow_1bit=not a.no_1bit,
                       head=a.head, embd=a.embd, types=types,
                       lowbit=json.load(open(a.lowbit)) if a.lowbit else None)
        json.dump(res, open(a.out, "w"), indent=1)
        if a.tensor_types:
            write_tensor_types(res, a.tensor_types)
        print(f"plan: {res['gb']} GB  {res['bpw_file']} bpw (file)  role bpw {res['bpw_role']}  "
              f"predicted {res['predicted_kld']} ({res['estimate']})")
        for role, h in res["hist"].items():
            print(f"  {role:10} " + "  ".join(f"{q}:{n}" for q, n in sorted(h.items(), key=lambda kv: bpw_of(kv[0]))))
        return 0

    if a.cmd == "lowbit":
        q, ppl = _engine(a, a.llama_quantize, a.llama_perplexity)
        types = [t.strip() for t in a.types.split(",") if t.strip()]
        print(f"== pollard-moefit lowbit :: {a.gguf}  types {types}  granularity {a.granularity}")
        lowbit(a.gguf, a.imatrix, a.eval, a.out, types, a.granularity, q, ppl, a.ngl, a.workdir, a.threads)
        print(f"wrote {a.out}")
        return 0

    if a.cmd == "auto":
        q, ppl = _engine(a, a.llama_quantize, a.llama_perplexity)
        em = json.load(open(a.errmap))
        rep = auto(a.gguf, a.imatrix, em, json.load(open(a.sensitivity)),
                   json.load(open(a.lowbit)) if a.lowbit else None, _budget(a, em),
                   [float(x) for x in a.prunes.split(",") if x.strip()], a.reap, a.screen_eval, a.eval, a.workdir,
                   a.out, q, ppl, a.ngl, a.top, not a.no_toolcall, a.tool_ref, a.head, a.embd, not a.no_controls,
                   a.min_tool_retention, judge_chunks=a.judge_chunks, keep_candidates=a.keep_candidates)
        rp = a.report or a.out + ".moefit.json"
        json.dump(rep, open(rp, "w"), indent=1)
        print(auto_table(rep))
        print(f"\nwinner: {rep['why']}\nwrote {a.out} and {rp}")
        return 0

    if a.cmd == "build":
        q, = _engine(a, a.llama_quantize)
        res = build(a.gguf, a.imatrix, json.load(open(a.plan)), a.out, q, a.threads,
                    allow_requantize=a.allow_requantize)
        print(json.dumps({k: v for k, v in res.items() if k != "mismatched"}), f"mismatched={len(res['mismatched'])}")
        if res["mismatched"]:
            for n, (want, got) in list(res["mismatched"].items())[:12]:
                print(f"  {n}: planned {want}, built {got}")
            return 2
        return 0

    if a.cmd == "exl3-recipe":
        import pollard_exl3 as X
        budgeted, how = X.exl3_budgeted_keys(a.model)
        routing = json.load(open(a.routing)) if a.routing else None
        sens = json.load(open(a.sensitivity)) if a.sensitivity else None
        rec = exl3_recipe(budgeted, routing, sens, a.bpw, a.head_bits, a.min_bits, a.max_bits, a.granularity,
                          a.expert_floor, a.nonexpert_floor)
        X.write_recipe(rec, a.out)
        print(f"recipe ({how}): target {a.bpw} -> {rec['achieved_bpw']} bpw, expert K histogram {rec['expert_k_hist']}"
              + ("  [mixed K within a layer: needs an exllamav3 whose MoE path accepts it]"
                 if rec["needs_mixed_k_moe"] else ""))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
