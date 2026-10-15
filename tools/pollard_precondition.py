#!/usr/bin/env python3
"""pollard-precondition -- measure which weight-preconditioner wins for THIS model +
target quant, and emit the winning build. The "dynamic" front door.

There is no single best preconditioner -- it depends on the target bit-width and the
model (measured, Qwen2.5-0.5B): diagonal smoothing helps SCALAR quants but hurts IQ
codebook quants; rotation helps IQ and its win GROWS as bits drop (a wash at 3-bit,
-5.7% at IQ2_S, -9.7% at IQ2_XXS); block-diagonal rotation keeps the imatrix useful.
So instead of guessing, this tool BUILDS each candidate, scores KL vs the f16 base on
a held-out slice, and keeps the winner. By construction the result is NEVER worse than
uniform (candidate 'none' is always in the running).

Candidates (auto-filtered by target family):
  none            uniform + imatrix (the baseline everyone ships)
  smooth          AWQ-style diagonal (SCALAR targets only -- hurts IQ)
  rot-block32     block-diagonal rotation (keeps imatrix; best at the low-bit frontier)
  rot-dense       dense rotation (strongest incoherence; best ~2.5-bit)
  rht-B64/128/256 ONLINE block RHT (pollard-rht): W.D.H folded per weight, the runtime applies
                  H.D to the matmul input (llama.cpp >= b11556, #29600). Every q/k/v/o/gate/up/
                  down + the LM head. Archs: llama, qwen3(moe), qwen35(moe), qwen3next.
  rht-down-only   the same, ffn_down only (block 128; rht-down-B<n> for another block)
  rht-attn-B<n> / rht-ffn-B<n>   attention-only / FFN-only

The rht candidates run only when the llama.cpp in use carries #29600 (--bin, else auto-found) --
an older build loads the folded weights WITHOUT the transform and scores garbage -- and the
imatrix for them is computed ON THE FOLDED MODEL, where the runtime hook feeds llama-imatrix the
rotated inputs. Their rungs need llama.cpp >= b11556 to run; pollard-ggufcheck says so.

Disk-careful: each candidate's transformed f16 + imatrix are deleted right after it's
scored, so only the small quantized files persist.

Usage:
  pollard-precondition --gguf m-f16.gguf --imatrix base.imatrix --calib calib.txt \\
      --eval held_out.txt --target iq2_xxs --out m-pollard.gguf
  pollard-precondition ... --target q4_k --candidates none,smooth        # scalar target
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCALAR = re.compile(r"^(q\d|iq4_nl)", re.I)          # scalar/K/_0 families (smooth helps)


def _hms(s):
    s = int(s); h, m = divmod(s, 3600); m, s = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def find_bin(name, bin_dir=None):
    if bin_dir:
        for c in (os.path.join(bin_dir, name), os.path.join(bin_dir, name + ".exe")):
            if os.path.exists(c):
                return os.path.abspath(c)
        sys.exit(f"ERROR: {name} not in --bin {bin_dir}")
    for c in (os.path.join(HERE, "..", "runtime", "llama.cpp", "build", "bin", name),):
        if os.path.exists(c):
            return os.path.abspath(c)
    from shutil import which
    p = which(name)
    if not p:
        sys.exit(f"ERROR: {name} not found -- build the runtime (install.sh) or add it to PATH.")
    return p


RHT_RE = re.compile(r"^rht(?:-(down|attn|ffn))?(?:-only)?(?:-B(\d+))?$")


def rht_spec(cand):
    """'rht-B128' -> ('all', 128); 'rht-down-only' -> ('down', 128); None if not an rht candidate."""
    m = RHT_RE.match(cand)
    if not m:
        return None
    return (m.group(1) or "all"), int(m.group(2) or 128)


def gguf_arch(path):
    try:
        from gguf import GGUFReader
        f = GGUFReader(path).fields.get("general.architecture")
        return str(bytes(f.parts[-1]), "utf-8") if f else ""
    except Exception:
        return ""


def kl_mean(perplexity_bin, model, eval_f, base_kld, ngl, extra=(), ctx=2048):
    r = sh([perplexity_bin, "-m", model, "-f", eval_f, "-c", str(ctx),
            "--kl-divergence-base", base_kld, "--kl-divergence", "-ngl", str(ngl)] + list(extra))
    m = re.search(r"Mean\s+KLD:\s*([0-9.]+)", r.stdout + r.stderr)
    t = re.search(r"Same top p:\s*([0-9.]+)", r.stdout + r.stderr)
    if not m:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
        return None, None, f"KL scoring failed:\n{tail}"
    return float(m.group(1)), (float(t.group(1)) if t else None), None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="f16/bf16 source GGUF")
    ap.add_argument("--imatrix", required=True, help="imatrix for the ORIGINAL model")
    ap.add_argument("--calib", required=True, help="calibration text (for per-candidate imatrix)")
    ap.add_argument("--eval", required=True, help="held-out eval text (for KL scoring)")
    ap.add_argument("--target", required=True, help="quant type, e.g. iq2_xxs, iq3_s, q4_k")
    ap.add_argument("--out", help="write the winning build here")
    ap.add_argument("--candidates", help="comma list; default auto by target family")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--bin", help="llama.cpp build/bin to use for every step (needs >= b11556 for rht-*)")
    ap.add_argument("--ctx", type=int, default=2048,
                    help="KL context per chunk (the base-logits file is tokens x vocab x 2 bytes -- "
                         "a 152k-vocab model writes ~0.3 MB per token, so keep the eval slice small)")
    ap.add_argument("--threads", type=int, help="CPU threads for imatrix/perplexity (box: 19)")
    ap.add_argument("--keep", action="store_true", help="keep every candidate build, not just the winner")
    a = ap.parse_args()

    for f in (a.gguf, a.imatrix, a.calib, a.eval):
        if not os.path.exists(f):
            sys.exit(f"ERROR: not found: {f}")
    quantize = find_bin("llama-quantize", a.bin)
    imatrix = find_bin("llama-imatrix", a.bin)
    perplexity = find_bin("llama-perplexity", a.bin)
    py = sys.executable
    scalar = bool(SCALAR.match(a.target))
    thr = ["-t", str(a.threads)] if a.threads else []

    # online RHT needs a runtime that applies the manifest, on an arch it accepts it for
    sys.path.insert(0, HERE)
    import pollard_rht
    rht_ok_bins = all(pollard_rht.binary_supports_prism(b) for b in (imatrix, perplexity))
    arch = gguf_arch(a.gguf)
    rht_ok_arch = arch in pollard_rht.ARCHS
    if a.candidates:
        cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
    else:
        cands = ["none", "rot-block32", "rot-dense"] + (["smooth"] if scalar else [])
        if rht_ok_bins and rht_ok_arch:
            cands += ["rht-B128", "rht-down-only"]
    if any(rht_spec(c) for c in cands) and not (rht_ok_bins and rht_ok_arch):
        why = (f"arch '{arch}' is not one #29600 applies the transform for" if not rht_ok_arch else
               f"{perplexity} predates llama.cpp {pollard_rht.RUNTIME_MIN} (#29600) -- it would load the "
               "folded weights WITHOUT the transform; pass --bin <newer build/bin>")
        print(f"   note: dropping the rht candidates: {why}")
        cands = [c for c in cands if not rht_spec(c)]
    if "smooth" in cands and not scalar:
        print("   note: 'smooth' distorts IQ codebook quants -- dropping it for this IQ target.")
        cands = [c for c in cands if c != "smooth"]

    work = os.path.dirname(os.path.abspath(a.out or a.gguf)) or "."
    stem = os.path.join(work, "_precond")
    base_kld = stem + ".base.kld"

    print(f"== pollard-precondition :: {a.gguf}  target {a.target}")
    print(f"   candidates: {cands}")
    # KL reference from the f16 base (once)
    if not (os.path.exists(base_kld) and os.path.getsize(base_kld) > 1024):
        print("   building KL base logits from f16 ...")
        r = sh([perplexity, "-m", a.gguf, "-f", a.eval, "-c", str(a.ctx),
                "--kl-divergence-base", base_kld, "-ngl", str(a.ngl)] + thr)
        if not (os.path.exists(base_kld) and os.path.getsize(base_kld) > 1024):
            sys.exit("ERROR: could not build KL base logits:\n" +
                     "\n".join((r.stdout + r.stderr).splitlines()[-8:]))

    results, t0 = [], time.time()
    for ci, cand in enumerate(cands):
        print(f"\n-- [{ci+1}/{len(cands)}] candidate: {cand}  (elapsed {_hms(time.time()-t0)}) --")
        f16 = a.gguf
        imat = a.imatrix
        tmp_f16, tmp_imat = None, None
        try:
            if cand != "none":
                tmp_f16 = f"{stem}.{cand}.f16.gguf"
                if cand == "smooth":
                    cmd = [py, os.path.join(HERE, "pollard_smooth.py"), "--gguf", a.gguf,
                           "--imatrix", a.imatrix, "--target", a.target, "--out", tmp_f16]
                elif cand.startswith("rot-block"):
                    bs = cand.replace("rot-block", "") or "32"
                    cmd = [py, os.path.join(HERE, "pollard_rotate.py"), "--gguf", a.gguf,
                           "--kind", "block", "--block", bs, "--out", tmp_f16]
                elif cand == "rot-dense":
                    cmd = [py, os.path.join(HERE, "pollard_rotate.py"), "--gguf", a.gguf,
                           "--kind", "orthogonal", "--out", tmp_f16]
                elif rht_spec(cand):
                    scope, blk = rht_spec(cand)
                    cmd = [py, os.path.join(HERE, "pollard_rht.py"), "fold", "--gguf", a.gguf,
                           "--scope", scope, "--block", str(blk), "--out", tmp_f16]
                else:
                    print(f"   unknown candidate '{cand}' -- skipping"); continue
                r = sh(cmd)
                if not (os.path.exists(tmp_f16) and os.path.getsize(tmp_f16) > 1024):
                    print("   transform FAILED:\n     " +
                          "\n     ".join((r.stdout + r.stderr).splitlines()[-4:]))
                    continue
                # fresh imatrix on the transformed weights (basis/scale changed). For rht the
                # runtime hook hands llama-imatrix H.D.x, so this IS the rotated-basis imatrix.
                tmp_imat = f"{stem}.{cand}.imatrix"
                sh([imatrix, "-m", tmp_f16, "-f", a.calib, "-o", tmp_imat, "-ngl", str(a.ngl)] + thr)
                if not (os.path.exists(tmp_imat) and os.path.getsize(tmp_imat) > 1024):
                    print("   imatrix build FAILED -- skipping candidate"); continue
                f16, imat = tmp_f16, tmp_imat

            out_q = f"{stem}.{cand}.{a.target}.gguf"
            rq = sh([quantize, "--imatrix", imat, f16, out_q, a.target])
            if not (os.path.exists(out_q) and os.path.getsize(out_q) > 1024):
                print("   quantize FAILED:\n     " +
                      "\n     ".join((rq.stdout + rq.stderr).splitlines()[-4:]))
                continue
            kl, top1, err = kl_mean(perplexity, out_q, a.eval, base_kld, a.ngl, thr, a.ctx)
            if kl is None:
                print(f"   {err}"); continue
            size = os.path.getsize(out_q)
            bpw = size * 8 / _nparams(a.gguf)
            print(f"   KL={kl:.5f}  top1={top1:.2f}%  size={size/1e6:.0f}MB  ~{bpw:.2f} bpw")
            results.append({"cand": cand, "kl": kl, "top1": top1, "path": out_q, "size": size})
        finally:
            for f in (tmp_f16, tmp_imat):
                if f and os.path.exists(f):
                    os.remove(f)

    if not results:
        sys.exit("ERROR: no candidate produced a score -- see failures above.")
    results.sort(key=lambda r: r["kl"])
    win = results[0]
    base = next((r for r in results if r["cand"] == "none"), None)
    print("\n== RESULT (sorted by KL, lower = better) ==")
    for r in results:
        tag = "  <-- WINNER" if r is win else ""
        delta = ""
        if base and r is not base and base["kl"] > 0:
            delta = f"  ({100*(base['kl']-r['kl'])/base['kl']:+.1f}% vs uniform)"
        print(f"   {r['cand']:12s} KL {r['kl']:.5f}  top1 {r['top1']:.2f}%{delta}{tag}")

    if a.out:
        import shutil
        shutil.move(win["path"], a.out)
        print(f"\n   winner '{win['cand']}' -> {a.out}")
        if not a.keep:
            for r in results:
                if r is not win and os.path.exists(r["path"]):
                    os.remove(r["path"])
    if os.path.exists(base_kld):
        os.remove(base_kld)               # the big file -- always clean it


def _nparams(gguf):
    """Rough param count from the f16 GGUF file size (bytes/2). Good enough for a bpw label."""
    try:
        return max(os.path.getsize(gguf) / 2, 1)
    except OSError:
        return 1


if __name__ == "__main__":
    main()
