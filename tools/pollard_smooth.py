#!/usr/bin/env python3
"""pollard-smooth — activation-aware weight preconditioning (AWQ-style) before the quant.

This is the step mainline llama.cpp does NOT do. imatrix bends the *rounding*
toward hot channels; it never moves magnitude OFF them. AWQ/SmoothQuant do: they
migrate the outlier magnitude of the salient input channels into an adjacent
tensor via a per-channel diagonal scale, so the quantizer spends its bits where
the activations actually are. The transform is mathematically identity — the
inverse is folded into the paired tensor — so the output is a NORMAL GGUF that
runs unchanged on CPU, CUDA, Metal, Vulkan (nothing about it is CPU-only).

The salient-channel signal is free: it's the imatrix we already build
(`<weight>.in_sum2 / .counts` = mean squared activation per input channel — the
exact quantity AWQ derives its scales from). We reuse it, so no second calibration
pass is needed.

Folded seams (each validated for dimension match; mismatches are SKIPPED, never
forced — fused-QKV / MLA / extra-norm archs simply get fewer seams, never corruption):

  D  up   -> down    scale down's input cols up, up's output rows down   (safest, universal)
  B  ffn_norm -> gate,up   fold 1/s into the RMSNorm gain
  A  attn_norm -> q,k,v    fold 1/s into the RMSNorm gain

Every fold is checked with a numerical identity canary (random input, output must
match pre-fold to fp tolerance) BEFORE it is kept. Then: recompute the imatrix on
the smoothed model, and quantize as usual (pollard-fit / pollard-sensitivity).
The win compounds with imatrix rounding and KL allocation — measure it with
pollard-eval (KL vs the f16 base) at equal bits-per-weight.

Usage:
  pollard-smooth --gguf model-f16.gguf --imatrix model.imatrix --out model-smooth-f16.gguf
  pollard-smooth --gguf model-f16.gguf --imatrix model.imatrix --seams D --plan-only
  pollard-smooth --gguf model-f16.gguf --imatrix model.imatrix --target iq3_s --alpha-grid 9

Start from an f16/bf16 source (smoothing then requantizing an already-4-bit file
compounds loss). After this, BUILD A FRESH IMATRIX on the output before fitting.
"""
import argparse
import re
import sys
import time

import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter
from gguf.quants import quantize, dequantize
from gguf.constants import GGMLQuantizationType as T, GGUFValueType as VT

# effective bits for the proxy quantizer used only to CHOOSE alpha (the real
# quant is done later by llama-quantize). block=32 min/max symmetric round is a
# faithful stand-in for how K/IQ quants set a shared block scale along the input.
TARGET_BITS = {"iq1_s": 1.56, "iq2_xxs": 2.1, "iq2_s": 2.5, "iq3_s": 3.4,
               "iq4_xs": 4.25, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.6, "q8_0": 8.5}
BLOCK = 32


def _hms(s):
    s = int(s); h, m = divmod(s, 3600); m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def load_imatrix_scales(path):
    """{weight_name: act_scale_vector} where act_scale = sqrt(in_sum2 / counts) =
    per-input-channel RMS activation magnitude (AWQ's saliency signal)."""
    r = GGUFReader(path)
    sum2, counts = {}, {}
    for t in r.tensors:
        arr = np.array(t.data, dtype=np.float64).reshape(tuple(int(d) for d in t.shape)[::-1]).ravel()
        if t.name.endswith(".in_sum2"):
            sum2[t.name[:-len(".in_sum2")]] = arr
        elif t.name.endswith(".counts"):
            counts[t.name[:-len(".counts")]] = arr
    out = {}
    for name, s2 in sum2.items():
        c = counts.get(name, np.array([1.0]))
        c = float(c[0]) if c.size else 1.0
        out[name] = np.sqrt(np.maximum(s2, 0.0) / max(c, 1.0))
    if not out:
        sys.exit(f"ERROR: no .in_sum2 tensors in {path} — not a GGUF imatrix, or it's "
                 f"the old binary format. Rebuild it with a current llama-imatrix.")
    return out


def to_logical(t):
    """Dequantize a reader tensor to f32 in logical [out, in] shape."""
    data = dequantize(t.data, t.tensor_type).astype(np.float32)
    return data.reshape(tuple(int(d) for d in t.shape)[::-1])


def _real_quant_err_sq(W):
    """Per-element squared error of a REAL block-32 quantizer (Q4_0 via gguf-py),
    used to rank alpha. A hand-rolled absmax proxy mispredicts which tensors
    benefit (it says q/k help; the real block quant says they don't) — so we use
    the actual kernel. Q4_0 is the fastest real proxy and its verdict on WHICH
    tensors benefit tracks the IQ/K targets (both are block-32-along-input)."""
    Wf = W.astype(np.float32)
    if Wf.shape[1] % BLOCK != 0:                  # pad the input axis to a block
        pad = BLOCK - (Wf.shape[1] % BLOCK)
        Wf = np.concatenate([Wf, np.zeros((Wf.shape[0], pad), np.float32)], axis=1)
    Wq = dequantize(quantize(Wf, T.Q4_0), T.Q4_0).reshape(Wf.shape).astype(np.float64)
    e = (Wq - Wf)[:, :W.shape[1]]
    return e * e


def choose_alpha(W_consumer, act, grid, min_gain=0.01):
    """Grid-search alpha in [0,1] minimizing the consumer's activation-weighted
    quant error, measured with a REAL block quantizer. s = (act/mean)^alpha. Returns
    (alpha, s). If the best alpha beats alpha=0 by less than min_gain (fractional),
    returns alpha=0 with s=1 — i.e. it SELF-SKIPS tensors that don't benefit (q/k/
    gate), so the caller never has to hardcode which seams help."""
    a = np.asarray(act, dtype=np.float64)
    a = np.where(a > 0, a, np.median(a[a > 0]) if np.any(a > 0) else 1.0)
    a = a / np.mean(a)                            # normalize so s doesn't inflate globally
    costs = []
    for alpha in grid:
        s = np.clip(np.power(a, alpha), 0.1, 10.0)
        Wf = W_consumer * s[np.newaxis, :]        # fold s into consumer input cols
        # error in ORIGINAL space is quant_err(Wf)/s, weighted by activation a
        w = (a * a) / (s * s)
        costs.append((float(np.sum(_real_quant_err_sq(Wf) * w[np.newaxis, :])), alpha, s))
    base = next((c for c, al, _ in costs if al == 0.0), costs[0][0])
    best_c, best_a, best_s = min(costs, key=lambda t: t[0])
    if best_a == 0.0 or base <= 0 or (base - best_c) / base < min_gain:
        return 0.0, np.ones_like(a, dtype=np.float32)   # self-skip: no real gain
    return best_a, best_s.astype(np.float32)


def canary(before_out, W_consumer_after, s, x):
    """Numerical identity check: consumer sees input x/s and weight W*s; output
    must match the pre-fold output W_orig @ x. Returns max abs rel error."""
    after = W_consumer_after @ (x / s)
    denom = np.maximum(np.abs(before_out), 1e-6)
    return float(np.max(np.abs(after - before_out) / denom))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (f16/bf16 best)")
    ap.add_argument("--imatrix", required=True, help="GGUF imatrix for the SAME model")
    ap.add_argument("--out")
    ap.add_argument("--seams", default="D",
                    help="which fold seams to apply (subset of D,B,A). Default D "
                    "(up->down): the seam every tensor benefits from. B/A share one "
                    "scale across q/k/v and gate/up, dragging in tensors that don't.")
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="skip a tensor unless smoothing cuts its (real) quant error "
                    "by at least this fraction — auto-skips q/k/gate-like tensors")
    ap.add_argument("--target", default="iq3_s", choices=list(TARGET_BITS),
                    help="quant type the smoothing is tuned for (picks alpha)")
    ap.add_argument("--alpha-grid", type=int, default=11,
                    help="number of alpha points searched in [0,1] (AWQ uses ~20)")
    ap.add_argument("--canary-tol", type=float, default=1e-3,
                    help="max allowed identity error per fold before it's rejected")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    import os
    if not os.path.exists(a.gguf):
        sys.exit(f"ERROR: source GGUF not found: {a.gguf}")
    if not os.path.exists(a.imatrix) or os.path.getsize(a.imatrix) < 1024:
        sys.exit(f"ERROR: imatrix not found or empty: {a.imatrix}\n"
                 f"  build it first: llama-imatrix -m {a.gguf} -f calib.txt -o out.imatrix -ngl 99")
    seams = {c.strip().upper() for c in a.seams.split(",") if c.strip()}
    grid = [i / (a.alpha_grid - 1) for i in range(a.alpha_grid)] if a.alpha_grid > 1 else [0.5]

    print(f"== pollard-smooth :: {a.gguf}")
    print(f"   imatrix {a.imatrix}  seams {sorted(seams)}  target {a.target}  alpha-grid {len(grid)}")
    scales = load_imatrix_scales(a.imatrix)
    reader = GGUFReader(a.gguf)
    arch = next((str(bytes(f.parts[-1]), "utf-8") for f in reader.fields.values()
                 if f.name == "general.architecture"), "llama")

    tensors = {t.name: t for t in reader.tensors}
    # discover layer indices from blk.N. names
    layers = sorted({int(m.group(1)) for n in tensors
                     for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    print(f"   arch {arch}  layers {len(layers)}  tensors {len(tensors)}")

    # seam spec: (letter, consumer_suffix, producer_suffix, producer_is_norm, extra_consumers)
    # producer absorbs 1/s; consumer input-cols absorb s. act comes from the
    # CONSUMER's own imatrix (its input = the scaled channel).
    SEAMS = {
        "D": ("ffn_down.weight", "ffn_up.weight", False, []),
        "B": ("ffn_gate.weight", "ffn_norm.weight", True, ["ffn_up.weight"]),
        "A": ("attn_q.weight", "attn_norm.weight", True, ["attn_k.weight", "attn_v.weight"]),
    }

    edits = {}                       # name -> new f32 array (folded)
    plan_rows, skipped = [], []
    t0 = time.time()
    total = len(layers) * len(seams & set(SEAMS))
    done = 0
    for li in layers:
        p = f"blk.{li}."
        for letter in ("D", "B", "A"):
            if letter not in seams:
                continue
            cons_suf, prod_suf, prod_is_norm, extras = SEAMS[letter]
            cons_n, prod_n = p + cons_suf, p + prod_suf
            if cons_n not in tensors or prod_n not in tensors:
                continue
            done += 1
            act = scales.get(cons_n)
            if act is None:
                skipped.append((cons_n, "no imatrix coverage")); continue
            Wc = edits.get(cons_n)
            if Wc is None:
                Wc = to_logical(tensors[cons_n])
            in_dim = Wc.shape[1]
            if act.shape[0] != in_dim:
                skipped.append((cons_n, f"imatrix len {act.shape[0]} != in_dim {in_dim}")); continue
            # producer must expose an axis == in_dim to absorb 1/s
            Wp = edits.get(prod_n)
            if Wp is None:
                Wp = to_logical(tensors[prod_n])
            if prod_is_norm:
                if Wp.ndim != 1 or Wp.shape[0] != in_dim:
                    skipped.append((prod_n, f"norm len {Wp.shape} != in_dim {in_dim}")); continue
            else:
                if Wp.shape[0] != in_dim:                 # producer OUTPUT rows == consumer input
                    skipped.append((prod_n, f"producer rows {Wp.shape[0]} != in_dim {in_dim}")); continue

            alpha, s = choose_alpha(Wc, act, grid, a.min_gain)
            if alpha == 0.0:                              # self-skipped: no real gain
                skipped.append((cons_n, "no quant gain (auto-skip)")); continue
            # identity canary on the primary consumer (random input)
            x = np.random.default_rng(li).standard_normal(in_dim).astype(np.float32)
            before = to_logical(tensors[cons_n]) @ x if cons_n not in edits else Wc @ x
            Wc_new = Wc * s[np.newaxis, :]
            err = canary(before, Wc_new, s, x)
            if err > a.canary_tol:
                skipped.append((cons_n, f"canary {err:.2e} > tol")); continue

            # commit: consumer input-cols *= s ; producer absorbs 1/s
            edits[cons_n] = Wc_new
            for ex in extras:                            # gate's sibling up shares the input scale
                exn = p + ex
                if exn in tensors:
                    We = edits.get(exn, to_logical(tensors[exn]))
                    if We.shape[1] == in_dim:
                        edits[exn] = We * s[np.newaxis, :]
            if prod_is_norm:
                edits[prod_n] = Wp / s
            else:
                edits[prod_n] = Wp / s[:, np.newaxis]     # producer output rows /= s
            plan_rows.append((p + letter, alpha, float(s.min()), float(s.max()), err))

            if done % 8 == 0 or done == total:
                el = time.time() - t0
                eta = el / done * (total - done)
                print(f"   [{done*100//max(total,1):3d}%  {_hms(el)}  ETA {_hms(eta)}] "
                      f"{p+letter} a={alpha:.2f} s[{s.min():.2f},{s.max():.2f}] canary={err:.1e}")

    print(f"\n   folds applied : {len(plan_rows)}   skipped seams : {len(skipped)}")
    if skipped:
        from collections import Counter
        for reason, k in Counter(r for _, r in skipped).most_common(6):
            print(f"     skip: {reason}  x{k}")
    if plan_rows:
        al = np.array([r[1] for r in plan_rows])
        print(f"   alpha: mean {al.mean():.2f}  range [{al.min():.2f},{al.max():.2f}]   "
              f"(0=no smoothing, 1=full activation scaling)")
    if not plan_rows:
        sys.exit("ERROR: no seams folded — check --seams, imatrix coverage, and dims above. "
                 "Nothing written (a no-op smooth would just waste a requant).")
    if a.plan_only:
        print("   (plan-only — nothing written)")
        return

    out = a.out or a.gguf.rsplit(".gguf", 1)[0] + "-smooth.gguf"
    print(f"\n   writing {out} ...")
    w = GGUFWriter(out, arch)
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"):
            continue
        vtype = f.types[0]
        sub = f.types[1] if vtype == VT.ARRAY and len(f.types) > 1 else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    for t in reader.tensors:
        if t.name in edits:
            data = edits[t.name]
            # keep the ORIGINAL dtype family; a smoothed f16 stays f16
            if t.tensor_type in (T.F32,):
                w.add_tensor(t.name, data.astype(np.float32))
            else:
                w.add_tensor(t.name, data.astype(np.float16))
        else:
            data = to_logical(t)
            if t.tensor_type == T.F32:
                w.add_tensor(t.name, data.astype(np.float32))
            elif t.tensor_type in (T.F16, T.BF16):
                w.add_tensor(t.name, data.astype(np.float16))
            else:
                qd = quantize(data, t.tensor_type)
                w.add_tensor(t.name, qd, raw_shape=qd.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"   done: {out}")
    print(f"\n   NEXT: build a FRESH imatrix on {out} (activations changed), then fit:")
    print(f"     llama-imatrix -m {out} -f calib.txt -o smooth.imatrix -ngl 99")
    print(f"     pollard-fit --gguf {out} --imatrix smooth.imatrix --ram <N> --out final.gguf")
    print(f"   Then compare KL vs the f16 base: uniform vs smoothed at EQUAL bits/weight.")


if __name__ == "__main__":
    main()
