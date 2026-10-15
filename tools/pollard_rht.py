#!/usr/bin/env python3
"""pollard-rht -- online randomized-Hadamard (RHT) GGUF rungs that STOCK llama.cpp loads.

llama.cpp PR #29600 (merged 2026-10-11, release b11556, commit a9d27ac) taught the stock runtime a
`prism.hadamard.*` manifest: a weight listed there is stored FOLDED, W_f = W.D.H, and the graph
applies D then H to that matmul's INPUT at run time, so W_f.(H.(D.x)) = W.D.H.H.D.x = W.x. D is a
+-1 sign vector, H the normalized Sylvester Walsh-Hadamard matrix applied block-diagonally along the
input dimension (block B, a power of two). H.H = I and D.D = I, so the f16 fold is exact up to f16
rounding (PR: mean KLD 0.000000).

Why that matters for quantization: the quantizer now sees W_f, whose rows are mixed inside every
B-wide input block -- outliers smeared, the block distribution near-Gaussian -- which is what the
IQ codebook / K-quant blocks pack best. Unlike pollard-rotate (a residual-stream rotation folded
into neighbours, which smears the imatrix across the whole hidden size and measured a LOSS on 7B),
this is per-weight and block-local, and the imatrix is collected on the ROTATED inputs (the runtime
hook feeds llama-imatrix H.D.x, keyed by the folded weight's name), so the importance signal is in
the basis the quantizer actually rounds in.

Measured (notes/rht-rungs.md, Qwen3-0.6B): the fold is exact (KLD 0.000005 vs the f16), but block RHT
flattens the diagonal imatrix (peak/mean 166 -> 13) that llama.cpp's rounding relies on, and the rungs
LOST to uniform+imatrix at IQ2_XXS..Q4_K (ffn_down-only lost least); it won only without an imatrix.
So it is a pollard-precondition CANDIDATE -- measured per model, never worse than "none" -- not a
default. The 7-8B board decides.

  pollard-rht fold  --gguf m-f16.gguf --out m-rht128-f16.gguf --block 128
  pollard-rht fold  --gguf m-f16.gguf --out m-rhtdown-f16.gguf --scope down
  pollard-rht rungs --gguf m-rht128-f16.gguf --calib calib.txt --outdir rungs/ \\
                    --targets IQ2_XXS,IQ2_S,Q2_K,Q3_K,IQ3_S,IQ4_XS
  pollard-rht info  m-rht128-IQ2_S.gguf

Runtime requirement (say it on the card -- pollard-ggufcheck does): llama.cpp >= b11556. Archs the
runtime accepts the manifest for: llama, qwen3, qwen3moe, qwen35, qwen35moe, qwen3next (others are
REFUSED at load -- this tool refuses them at fold time). ik_llama.cpp is NOT a target: its loader
reads the keys but its default fused up/gate path skips the transform for non-Prism atoms.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the runtime contract (llama.cpp src/llama-model.cpp load_hparams_hadamard, b11556) ---------
RUNTIME_MIN = "b11556"                       # the release that carries #29600
RUNTIME_COMMIT = "a9d27ac"
TRANSFORM = "normalized-sylvester-walsh-hadamard"
AXIS = "input-last-dimension"
ARCHS = {"llama", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "qwen3next"}
FOLDABLE_KINDS = ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
                  "ffn_gate", "ffn_up", "ffn_down",
                  "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
                  "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ssm_out")
_FOLDABLE = re.compile(r"^blk\.\d+\.(" + "|".join(FOLDABLE_KINDS) + r")\.weight$")
KEY = "prism.hadamard."

# which tensor kinds each --scope folds
SCOPES = {
    "all":  set(FOLDABLE_KINDS),
    "down": {"ffn_down", "ffn_down_exps", "ffn_down_shexp"},
    "ffn":  {"ffn_gate", "ffn_up", "ffn_down", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
             "ffn_gate_up_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"},
    "attn": {"attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output"},
}

DEFAULT_RUNGS = ["IQ2_XXS", "IQ2_S", "Q2_K", "Q3_K", "IQ3_S", "IQ4_XS"]


def _hms(s):
    s = int(s); h, m = divmod(s, 3600); m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


# ------------------------------------------------------------------------------------------------
# math
# ------------------------------------------------------------------------------------------------
def sylvester(B):
    """Normalized Sylvester Walsh-Hadamard matrix, entry (r, c) = (-1)^popcount(r & c) / sqrt(B).
    Exactly the matrix load_tensors_hadamard builds (same row/column order). Symmetric, H.H = I."""
    if B <= 0 or B & (B - 1):
        raise ValueError(f"block size must be a power of two, got {B}")
    r = np.arange(B)
    par = np.zeros((B, B), dtype=np.int64)
    rc = r[:, None] & r[None, :]
    while rc.any():
        par ^= rc & 1
        rc >>= 1
    return np.where(par == 1, -1.0, 1.0) / np.sqrt(B)


def signs_for(width, seed):
    """Deterministic +-1 vector for one input width. Seeded per (seed, width) so the same model gets
    the same signs on every machine, and two widths never share a stream."""
    rng = np.random.default_rng([int(seed), int(width)])
    return (rng.integers(0, 2, width, dtype=np.int8) * 2 - 1).astype(np.int8)


def fold_rows(W, d, H, chunk_rows=8192):
    """W_f = W . diag(d) . blockdiag(H) along the LAST (input) axis. Any leading shape (experts
    included). Returns float32. Done in row chunks so a 150k-row output head stays small."""
    B = H.shape[0]
    shp = W.shape
    n = shp[-1]
    if n % B:
        raise ValueError(f"block {B} does not divide input width {n}")
    flat = W.reshape(-1, n)
    out = np.empty(flat.shape, dtype=np.float32)
    Hf = H.astype(np.float32)
    df = d.astype(np.float32)
    for r0 in range(0, flat.shape[0], chunk_rows):
        x = flat[r0:r0 + chunk_rows].astype(np.float32) * df[None, :]
        out[r0:r0 + chunk_rows] = (x.reshape(-1, n // B, B) @ Hf).reshape(-1, n)
    return out.reshape(shp)


def runtime_input(x, d, H):
    """What the runtime feeds the folded weight: H(D x), block-wise (build_hadamard_input)."""
    B = H.shape[0]
    y = x * d
    return (y.reshape(-1, B) @ H).reshape(x.shape)


# ------------------------------------------------------------------------------------------------
# GGUF helpers
# ------------------------------------------------------------------------------------------------
def _field_str(reader, name, default=""):
    f = reader.fields.get(name)
    if f is None:
        return default
    try:
        return str(bytes(f.parts[-1]), "utf-8")
    except Exception:
        return default


def _logical(t):
    from gguf.quants import dequantize
    data = dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
    return data.reshape(tuple(int(x) for x in t.shape)[::-1])


def plan(reader, scope="all", block=128, output="auto", only=None):
    """Which tensors get folded, and the manifest shape. Returns a dict, raises ValueError with a
    reader-facing message for anything the runtime would refuse."""
    arch = _field_str(reader, "general.architecture")
    if arch not in ARCHS:
        raise ValueError(f"architecture '{arch}' is not one the #29600 runtime applies the transform "
                         f"for ({', '.join(sorted(ARCHS))}); stock llama.cpp would REFUSE the file")
    if any(k.startswith(KEY) for k in reader.fields):
        raise ValueError("this GGUF already carries a prism.hadamard manifest -- fold the original f16")
    kinds = SCOPES[scope] if only is None else set(only)
    tensors = {t.name: t for t in reader.tensors}
    names, skipped = [], []
    for t in reader.tensors:
        m = _FOLDABLE.match(t.name)
        if not m or m.group(1) not in kinds:
            continue
        width = int(t.shape[0])                       # ne0 = input dimension
        if width % block:
            skipped.append((t.name, f"input width {width} not a multiple of {block}"))
            continue
        names.append(t.name)
    tied = "output.weight" not in tensors and "token_embd.weight" in tensors
    fold_output = output == "fold" or (output == "auto" and scope == "all")
    inverse = []
    if fold_output:
        if not tied:
            w = int(tensors["output.weight"].shape[0])
            if w % block == 0:
                names.append("output.weight")
            else:
                skipped.append(("output.weight", f"input width {w} not a multiple of {block}"))
        else:
            w = int(tensors["token_embd.weight"].shape[0])
            if w % block == 0:
                inverse.append("token_embd.weight")   # v2: latent embedding + tied output
            else:
                skipped.append(("token_embd.weight", f"width {w} not a multiple of {block}"))
    if not names:
        raise ValueError("nothing to fold (scope matched no tensor whose input width divides the block)")
    widths = sorted({int(tensors[n].shape[0]) for n in names + inverse})
    return {"arch": arch, "names": names, "inverse": inverse, "tied": bool(inverse),
            "widths": widths, "block": block, "skipped": skipped}


def manifest_kv(p, seed, sign_mode="explicit"):
    """The prism.hadamard key/values, exactly as load_hparams_hadamard reads them."""
    from gguf.constants import GGUFValueType as VT
    kv = [
        (KEY + "version", 2 if p["tied"] else 1, VT.UINT32, None),
        (KEY + "block_size", int(p["block"]), VT.UINT32, None),
        (KEY + "transform", TRANSFORM, VT.STRING, None),
        (KEY + "axis", AXIS, VT.STRING, None),
        (KEY + "sign_mode", sign_mode, VT.STRING, None),
        (KEY + "weight_names", list(p["names"]), VT.ARRAY, VT.STRING),
    ]
    if p["tied"]:
        kv.insert(1, (KEY + "tied_output", True, VT.BOOL, None))
        kv.append((KEY + "inverse_weight_names", list(p["inverse"]), VT.ARRAY, VT.STRING))
    if sign_mode == "explicit":
        vals = []
        for w in p["widths"]:
            vals.extend(int(v) for v in signs_for(w, seed))
        kv.append((KEY + "sign_widths", [int(w) for w in p["widths"]], VT.ARRAY, VT.INT32))
        kv.append((KEY + "sign_values", vals, VT.ARRAY, VT.INT32))
    return kv


def read_manifest(path):
    """{key: value} of the prism.hadamard.* fields of a GGUF (empty dict = not an RHT file)."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {}
    for k, f in r.fields.items():
        if not k.startswith(KEY):
            continue
        try:
            out[k[len(KEY):]] = f.contents()
        except Exception:
            out[k[len(KEY):]] = None
    return out


def fold(src, out, block=128, scope="all", seed=1, output="auto", sign_mode="explicit",
         canary_tol=2e-2, quiet=False, only=None):
    """Write `out` = `src` with the selected weights folded and the manifest added. Streaming: holds
    one tensor at a time. Returns the plan dict plus canary stats."""
    import gguf
    from gguf import GGUFReader, GGUFWriter
    from gguf.constants import GGMLQuantizationType as T, GGUFValueType as VT
    from gguf.quants import quantize, dequantize

    say = (lambda *a, **k: None) if quiet else print
    reader = GGUFReader(src)
    p = plan(reader, scope=scope, block=block, output=output, only=only)
    H = sylvester(block)
    signs = {w: (signs_for(w, seed) if sign_mode == "explicit" else np.ones(w, dtype=np.int8))
             for w in p["widths"]}
    fold_set = set(p["names"]) | set(p["inverse"])
    say(f"== pollard-rht fold :: {src}  [{p['arch']}]  block={block}  scope={scope}  "
        f"{len(p['names'])} weight(s){' + tied latent embedding (v2)' if p['tied'] else ''}")
    for n, why in p["skipped"]:
        say(f"   skip {n}: {why}")

    w = GGUFWriter(out, p["arch"])
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue
        vtype = f.types[0]
        sub = f.types[1] if vtype == VT.ARRAY and len(f.types) > 1 else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    for k, v, vt, sub in manifest_kv(p, seed, sign_mode):
        w.add_key_value(k, v, vt, sub_type=sub)

    # tensor table first (shapes/types unchanged), then stream the data
    for t in reader.tensors:
        d = t.data
        if t.tensor_type in (T.F32, T.F16):
            w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes)
        else:
            w.add_tensor_info(t.name, d.shape, np.uint8, d.nbytes, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    rng = np.random.default_rng(0)
    worst, t0, nt = 0.0, time.time(), len(reader.tensors)
    for i, t in enumerate(reader.tensors):
        if t.name not in fold_set:
            w.write_tensor_data(np.asarray(t.data))
            continue
        W = _logical(t)
        width = W.shape[-1]
        Wf = fold_rows(W, signs[width], H)
        if t.tensor_type == T.F32:
            raw = Wf.astype(np.float32)
        elif t.tensor_type == T.F16:
            raw = Wf.astype(np.float16)
        else:
            raw = quantize(Wf, t.tensor_type)
            if raw.nbytes != t.data.nbytes:
                raise RuntimeError(f"{t.name}: re-encoded size changed ({raw.nbytes} != {t.data.nbytes})")
        # identity canary on the STORED weights: W x  vs  W_f (H D x), a few rows
        rows = W.reshape(-1, width)
        stored = (raw.astype(np.float32) if t.tensor_type in (T.F32, T.F16)
                  else dequantize(raw, t.tensor_type).astype(np.float32)).reshape(-1, width)
        idx = rng.choice(rows.shape[0], size=min(64, rows.shape[0]), replace=False)
        x = rng.standard_normal(width).astype(np.float32)
        ref = rows[idx] @ x
        got = stored[idx] @ runtime_input(x, signs[width].astype(np.float32), H.astype(np.float32))
        rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))
        worst = max(worst, rel)
        if rel > canary_tol:
            w.close()
            os.remove(out)
            raise RuntimeError(f"identity canary FAILED on {t.name}: rel err {rel:.2e} > {canary_tol}")
        w.write_tensor_data(raw.reshape(t.data.shape) if raw.shape != t.data.shape else raw)
        if not quiet and ((i + 1) % 64 == 0):
            say(f"   [{(i+1)*100//nt:3d}%  {_hms(time.time()-t0)}] {i+1}/{nt} tensors")
    w.close()
    say(f"   identity canary (stored weights, worst tensor): rel err {worst:.2e}")
    say(f"   wrote {out}  ({os.path.getsize(out)/1e9:.2f} GB)  -- needs llama.cpp >= {RUNTIME_MIN}")
    p["canary_worst"] = worst
    return p


# ------------------------------------------------------------------------------------------------
# binaries
# ------------------------------------------------------------------------------------------------
def binary_supports_prism(bin_path):
    """Does this llama.cpp build apply the manifest? The loader's key names are string literals in
    libllama (or in the static binary); an older build has none and would load a folded file
    WITHOUT the transform -- garbage, no error. Unreadable -> False (never assume)."""
    try:
        from pollard_calc import _llama_libs
        libs = list(_llama_libs(bin_path))
    except Exception:
        libs = [bin_path]
    if bin_path not in libs:
        libs.append(bin_path)
    for lib in libs:
        try:
            with open(lib, "rb") as fh:
                if b"prism.hadamard.version\x00" in fh.read():
                    return True
        except OSError:
            continue
    return False


def find_prism_bin(name, bin_dir=None):
    """A `name` binary that applies the manifest: --bin, $POLLARD_PRISM_BIN, then the usual places."""
    cands = []
    for d in (bin_dir, os.environ.get("POLLARD_PRISM_BIN")):
        if d:
            cands.append(os.path.join(os.path.expanduser(d), name))
    try:
        from pollard_calc import find_llama_bin
        p = find_llama_bin(name)
        if p:
            cands.append(p)
    except Exception:
        pass
    from shutil import which
    if which(name):
        cands.append(which(name))
    for c in cands:
        for cc in (c, c + ".exe"):
            if os.path.exists(cc) and binary_supports_prism(cc):
                return cc
    return None


def _need(name, bin_dir):
    p = find_prism_bin(name, bin_dir)
    if not p:
        sys.exit(f"ERROR: no {name} that understands prism.hadamard (llama.cpp >= {RUNTIME_MIN}). "
                 f"Pass --bin <llama.cpp build/bin> or set POLLARD_PRISM_BIN. An older build would "
                 f"load the folded weights WITHOUT the transform and produce garbage.")
    return p


# ------------------------------------------------------------------------------------------------
# imatrix basis check
# ------------------------------------------------------------------------------------------------
def imatrix_stats(path):
    """{weight name: per-channel mean squared activation} from a GGUF-format imatrix."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    sums, counts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-8]] = np.asarray(t.data, dtype=np.float64).reshape(-1)
        elif t.name.endswith(".counts"):
            counts[t.name[:-7]] = float(np.asarray(t.data).reshape(-1)[0])
    return {k: v / max(counts.get(k, 1.0), 1.0) for k, v in sums.items()}


def imatrix_basis_check(ref, rot, names):
    """Evidence that llama-imatrix collected the ROTATED inputs for the folded weights.
    E[(HDx)^2] summed over channels = E[|x|^2] (H, D orthogonal), so the per-tensor TRACE must match
    the original model's imatrix, while the per-channel profile flattens (peak/mean drops). A hook
    that saw unrotated inputs would give an identical profile; one that missed the weight gives
    nothing at all."""
    a, b = imatrix_stats(ref), imatrix_stats(rot)
    rows = []
    for n in names:
        if n not in a or n not in b or a[n].shape != b[n].shape:
            continue
        ta, tb = a[n].sum(), b[n].sum()
        pa, pb = a[n].max() / max(a[n].mean(), 1e-30), b[n].max() / max(b[n].mean(), 1e-30)
        same = float(np.max(np.abs(a[n] - b[n])) / max(np.abs(a[n]).max(), 1e-30))
        rows.append((n, ta, tb, pa, pb, same))
    return rows


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _run(cmd, log=None):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if log:
        with open(log, "a") as fh:
            fh.write("$ " + " ".join(cmd) + "\n" + r.stdout + r.stderr + "\n")
    return r


def cmd_fold(a):
    if not os.path.exists(a.gguf):
        sys.exit(f"ERROR: not found: {a.gguf}")
    suffix = "" if a.scope == "all" else "-" + a.scope
    out = a.out or f"{a.gguf.rsplit('.gguf', 1)[0]}-rht{a.block}{suffix}.gguf"
    try:
        fold(a.gguf, out, block=a.block, scope=a.scope, seed=a.seed, output=a.output,
             sign_mode=a.sign_mode, canary_tol=a.canary_tol)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"ERROR: {e}")


def cmd_rungs(a):
    """imatrix on the FOLDED model (the runtime hook feeds it H.D.x), then the normal rungs."""
    if not read_manifest(a.gguf):
        sys.exit(f"ERROR: {a.gguf} has no prism.hadamard manifest -- run `pollard-rht fold` first")
    quant = _need("llama-quantize", a.bin)
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, os.path.basename(a.gguf).rsplit(".gguf", 1)[0].replace("-f16", "")
                        .replace("-bf16", ""))
    imat = a.imatrix
    if not imat:
        imb = _need("llama-imatrix", a.bin)
        imat = stem + ".imatrix"
        print(f"   imatrix on the folded model -> {imat}")
        r = _run([imb, "-m", a.gguf, "-f", a.calib, "-o", imat, "-ngl", str(a.ngl),
                  "--chunks", str(a.chunks), "--output-format", "gguf"]
                 + (["-t", str(a.threads)] if a.threads else []))
        if r.returncode != 0 or not os.path.exists(imat):
            sys.exit("ERROR: llama-imatrix failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-8:]))
        if a.ref_imatrix:
            names = read_manifest(a.gguf).get("weight_names") or []
            rows = imatrix_basis_check(a.ref_imatrix, imat, names)
            if rows:
                dt = max(abs(r_[2] / r_[1] - 1) for r_ in rows if r_[1] > 0)
                flat = sum(1 for r_ in rows if r_[4] < r_[3]) / len(rows)
                print(f"   imatrix basis check: {len(rows)} folded weights, max |trace ratio - 1| "
                      f"{dt:.2e} (orthogonal => ~0), peak/mean flattened on {flat:.0%}")
    built = []
    if a.allocation == "gold":
        # the gold ladder's planner (pollard_auto.ladder_points/pick_ladder -> pollard-fit), on the
        # folded weights with the ROTATED imatrix. No ik flagship: ik_llama does not run RHT files.
        from pollard_auto import ladder_points, pick_ladder
        picks = pick_ladder(ladder_points(a.gguf, imat, a.sensitivity),
                            os.path.getsize(a.gguf) / 1e9, ik_flagship=False)
        for label, (ram, preset, gb) in picks.items():
            out = f"{stem}-gold-{label}.gguf"
            cmd = [sys.executable, os.path.join(HERE, "pollard_fit.py"), "--gguf", a.gguf,
                   "--ram", str(ram), "--reserve", "0", "--imatrix", imat, "--llama-quantize", quant,
                   "--out", out]
            if a.sensitivity:
                cmd += ["--sensitivity", a.sensitivity]
            if a.threads:
                cmd += ["--threads", str(a.threads)]
            r = _run(cmd)
            if r.returncode != 0 or not os.path.exists(out):
                print(f"   gold {label}: pollard-fit FAILED\n     "
                      + "\n     ".join((r.stdout + r.stderr).splitlines()[-4:]))
                continue
            built.append(out)
            print(f"   gold {label:7s} (--ram {ram}, {preset}) -> {out}  ({os.path.getsize(out)/1e9:.3f} GB)")
    for tgt in ([] if a.allocation == "gold" else [x.strip() for x in a.targets.split(",") if x.strip()]):
        out = f"{stem}-{tgt}.gguf"
        r = _run([quant, "--imatrix", imat, a.gguf, out, tgt] + ([str(a.threads)] if a.threads else []))
        if r.returncode != 0 or not os.path.exists(out):
            print(f"   {tgt}: quantize FAILED\n     " + "\n     ".join((r.stdout + r.stderr).splitlines()[-4:]))
            continue
        built.append(out)
        print(f"   {tgt:8s} -> {out}  ({os.path.getsize(out)/1e9:.3f} GB)")
    if not built:
        sys.exit("ERROR: no rung built")
    print(f"\n   every file above needs llama.cpp >= {RUNTIME_MIN} (#29600); not ik_llama.cpp.")


def cmd_info(a):
    rc = 0
    for f in a.files:
        m = read_manifest(f)
        if not m:
            print(f"  {f}: no prism.hadamard manifest")
            continue
        n = len(m.get("weight_names") or [])
        inv = m.get("inverse_weight_names") or []
        print(f"  {f}: v{m.get('version')} block {m.get('block_size')} signs {m.get('sign_mode')} "
              f"widths {m.get('sign_widths')}  {n} folded weight(s)"
              f"{' + tied latent ' + ','.join(inv) if inv else ''}  -> llama.cpp >= {RUNTIME_MIN}")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)

    f = sp.add_parser("fold", help="fold W.D.H into an f16/bf16 GGUF and write the manifest")
    f.add_argument("--gguf", required=True)
    f.add_argument("--out")
    f.add_argument("--block", type=int, default=128, choices=[32, 64, 128, 256, 512])
    f.add_argument("--scope", default="all", choices=sorted(SCOPES))
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--output", default="auto", choices=["auto", "fold", "skip"],
                   help="the LM head: auto = fold it with --scope all (tied models: latent embedding, "
                        "manifest v2); skip = leave it alone")
    f.add_argument("--sign-mode", default="explicit", choices=["explicit", "identity"])
    f.add_argument("--canary-tol", type=float, default=2e-2,
                   help="max relative error of W.x vs W_f.(H.D.x) on the stored weights")
    f.set_defaults(fn=cmd_fold)

    r = sp.add_parser("rungs", help="imatrix on the folded model, then the normal ladder")
    r.add_argument("--gguf", required=True, help="folded f16 GGUF (from `fold`)")
    r.add_argument("--calib", help="calibration text for the imatrix")
    r.add_argument("--imatrix", help="reuse an imatrix already computed ON THE FOLDED MODEL")
    r.add_argument("--ref-imatrix", help="the ORIGINAL model's imatrix (GGUF format): prints the basis check")
    r.add_argument("--targets", default=",".join(DEFAULT_RUNGS))
    r.add_argument("--allocation", default="uniform", choices=["uniform", "gold"],
                   help="uniform = one llama-quantize preset per --targets entry; gold = the Pollard "
                        "ladder planner + pollard-fit (measured allocation) on the folded weights")
    r.add_argument("--sensitivity", help="sensitivity.json for --allocation gold (the gold run's profile)")
    r.add_argument("--outdir", default=".")
    r.add_argument("--bin", help="llama.cpp build/bin carrying #29600")
    r.add_argument("--ngl", type=int, default=99)
    r.add_argument("--chunks", type=int, default=200)
    r.add_argument("--threads", type=int, help="CPU threads (the box: 19, with --ngl 0)")
    r.set_defaults(fn=cmd_rungs)

    i = sp.add_parser("info", help="print a file's prism.hadamard manifest")
    i.add_argument("files", nargs="+")
    i.set_defaults(fn=cmd_info)

    a = ap.parse_args(argv)
    if a.cmd == "rungs" and not (a.calib or a.imatrix):
        ap.error("rungs needs --calib (or --imatrix computed on the folded model)")
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
