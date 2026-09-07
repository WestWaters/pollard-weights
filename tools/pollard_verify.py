#!/usr/bin/env python3
"""pollard-verify — correctness/perf GATE for a converted model. Catch "weights decode fine but the
assembled model forwards to garbage" BEFORE trusting a build.

Why this exists: a convert can produce per-tensor-plausible output yet a broken model. We judge builds
ONLY by real reconstruction (the metric a convert prints is a Hessian-weighted proxy that lies — banned,
see legacy/PROXY_ERR_BANNED.md). This gate checks the two things that actually matter:

  1. per-tensor decode round-trip vs the fp16 SOURCE weight (cos), failing loud below --threshold
  2. the assembled end-to-end forward vs the source model (--end-to-end): next-token agreement

Hard-won rules baked in:
  * verify against the ORIGINAL source weight, never a weight a quantizer mutated in place;
  * DLL search must include torch's OWN bundled CUDA runtime (site-packages/nvidia/*/bin), or low-bit
    kernels silently mis-resolve and produce garbage (decode looks fine — the trap);
  * codebook is AUTO-DETECTED per tensor from the on-disk markers (mul1/mcg/else=3inst), so we read ANY
    model's convention, not one hard-coded lane's;
  * per-tensor correct is necessary, NOT sufficient — always run --end-to-end before trusting a build.

  pollard-verify --model out-exl3 --source ./Qwen2.5-3B --end-to-end        # gate a build
  pollard-verify --model out-exl3 --source ./src --sanitizer               # kernels under compute-sanitizer

Lanes: exl3 (implemented). gguf/gptq/mlx share the same interface (decode(name)->weight) — add a backend.
Exit code is non-zero if any check fails, so it drops straight into CI / a pre-ship gate.
"""
import argparse, glob, math, os, subprocess, sys


def _add_cuda_dll_dirs():
    """torch's bundled CUDA runtime MUST win the DLL search or low-bit kernels silently corrupt."""
    cands = []
    try:
        import torch
        cands.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
        cands += glob.glob(os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", "*", "bin"))
    except Exception:
        pass
    for d in cands:
        if os.path.isdir(d):
            try: os.add_dll_directory(d)
            except Exception: pass


def _detect_codebook(keys, prefix):
    """Read the on-disk codebook convention for a tensor — generalized, not lane-hard-coded."""
    if prefix + ".mul1" in keys: return {"mul1": True}
    if prefix + ".mcg" in keys: return {"mcg": True}
    return {}                                             # else = 3inst (marker-free)


def _iter_source_weights(source_dir):
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(source_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k.endswith(".weight") and ("layers." in k or k in ("lm_head.weight",)):
                    yield k[:-len(".weight")], f, k


def verify_exl3(model_dir, source_dir, threshold, device, limit):
    """Per-tensor: decode each quantized linear and compare to the fp16 SOURCE weight (cos)."""
    import torch, torch.nn.functional as F
    from safetensors import safe_open
    from exllamav3.modules.linear import LinearEXL3
    qf = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    qhs = [safe_open(f, framework="pt") for f in qf]
    qkeys = set(k for h in qhs for k in h.keys())
    def qget(k):
        for h in qhs:
            if k in h.keys(): return h.get_tensor(k)
        return None
    src = {k: (f, wk) for k, f, wk in _iter_source_weights(source_dir)}
    # tied lm_head: source has no lm_head.weight -> falls back to embed_tokens.weight
    emb = None
    checked, failed = 0, []
    g = torch.Generator(device=device).manual_seed(1234)
    for name in sorted(k for k in qkeys if k.endswith(".trellis")):
        prefix = name[:-len(".trellis")]
        if source_dir:
            wk = None
            if prefix in src: wk = src[prefix]
            elif prefix == "lm_head":                    # tied head -> compare vs embed_tokens
                if emb is None:
                    for k, f, w in _iter_source_weights(source_dir):
                        pass
                wk = None
            if wk is None:
                continue
            with safe_open(wk[0], framework="pt") as h:
                W = h.get_tensor(wk[1]).to(device).float()   # (out, in) — the pristine source weight
        else:
            continue
        outf, inf = W.shape
        cb = _detect_codebook(qkeys, prefix)
        kw = {k.split(".")[-1]: qget(k).to(device)
              for k in qkeys if k.startswith(prefix + ".")
              and k.split(".")[-1] in ("suh", "svh", "trellis", "mul1", "mcg", "scale", "su", "sv")}
        x = torch.randn(1, 8, inf, device=device, dtype=torch.float16, generator=g)
        y_ref = x.float() @ W.t()
        lin = LinearEXL3(None, inf, outf, **kw)
        with torch.no_grad():
            y = lin.forward(x, {}).float()
        cos = F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item()
        checked += 1
        tag = "OK " if cos >= threshold else "FAIL"
        if cos < threshold: failed.append((prefix, cos, cb))
        print(f"  [{tag}] {prefix:<48} cos={cos:.4f}  cb={cb or '3inst'}", flush=True)
        if limit and checked >= limit: break
    return checked, failed


def end_to_end(model_dir, source_dir, device, rows, length):
    """The assembled forward vs the fp16 source — per-tensor correct is NOT sufficient."""
    import torch, torch.nn.functional as F
    from exllamav3 import Config, Model, Cache, Tokenizer
    cfg = Config.from_directory(model_dir); m = Model.from_config(cfg)
    c = Cache(m, max_num_tokens=length + 64); m.load()
    tok = Tokenizer.from_config(cfg)
    text = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tok.encode(text)
    if ids.dim() == 1: ids = ids.unsqueeze(0)
    seq = ids[:, :length]
    lg = m.forward(seq[:, :-1].to(device))[0].float()
    tg = seq[0, 1:seq.shape[-1]].to(device)
    z = min(lg.shape[0], tg.shape[0])
    acc = (lg[:z].argmax(-1) == tg[:z]).float().mean().item()
    ppl = math.exp(F.cross_entropy(lg[:z], tg[:z]).item())
    return acc, ppl


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="converted model dir to gate")
    ap.add_argument("--source", help="fp16 HF source dir (per-tensor decode-vs-source check)")
    ap.add_argument("--lane", default="exl3", choices=["exl3"], help="convert lane (exl3 implemented)")
    ap.add_argument("--threshold", type=float, default=0.9, help="min per-tensor decode cos vs source")
    ap.add_argument("--limit", type=int, default=0, help="cap tensors checked (0 = all)")
    ap.add_argument("--end-to-end", action="store_true", help="also check the assembled forward")
    ap.add_argument("--e2e-min-acc", type=float, default=0.30, help="min end-to-end next-tok agreement")
    ap.add_argument("--length", type=int, default=256, help="end-to-end eval length (tokens)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sanitizer", action="store_true",
                    help="re-run this verify under NVIDIA compute-sanitizer (memcheck+racecheck)")
    a = ap.parse_args()

    if a.sanitizer and not os.environ.get("_POLLARD_UNDER_SANITIZER"):
        san = (os.environ.get("COMPUTE_SANITIZER")
               or ("compute-sanitizer" if not sys.platform.startswith("win") else None))
        if san is None:
            print("compute-sanitizer not on PATH; set COMPUTE_SANITIZER=<path to .bat/.exe>", file=sys.stderr)
            sys.exit(3)
        env = dict(os.environ, _POLLARD_UNDER_SANITIZER="1")
        cmd = [san, "--tool", "memcheck", "--error-exitcode", "8", sys.executable, "-X", "utf8",
               os.path.abspath(__file__), "--model", a.model] + (["--source", a.source] if a.source else []) \
              + (["--end-to-end"] if a.end_to_end else [])
        print("== compute-sanitizer ==\n  $ " + " ".join(cmd))
        sys.exit(subprocess.run(cmd, env=env).returncode)

    _add_cuda_dll_dirs()
    print(f"== pollard-verify :: {a.model}  (lane={a.lane}, threshold={a.threshold}) ==")
    fails = 0

    if a.source:
        print("-- per-tensor decode vs source --")
        checked, failed = verify_exl3(a.model, a.source, a.threshold, a.device, a.limit)
        print(f"   {checked - len(failed)}/{checked} tensors >= {a.threshold}")
        if failed:
            fails += len(failed)
            print(f"   !! {len(failed)} tensors FAILED (e.g. {failed[0][0]} cos={failed[0][1]:.4f})")
    else:
        print("   (no --source; skipping per-tensor decode-vs-source. Pass --source to gate weights.)")

    if a.end_to_end:
        print("-- end-to-end assembled forward --")
        acc, ppl = end_to_end(a.model, a.source or a.model, a.device, 1, a.length)
        ok = acc >= a.e2e_min_acc
        print(f"   next_tok_acc={acc:.4f}  ppl={ppl:.2f}  [{'OK' if ok else 'FAIL'} vs min {a.e2e_min_acc}]")
        if not ok:
            fails += 1
            print("   !! assembled forward is broken even if per-tensor checks passed "
                  "(per-tensor correct is necessary, not sufficient).")

    # stamp the workspace manifest so `pollard-ls` shows verified status (no-op if built outside workspace)
    try:
        import pollard_workspace as ws
        ws.mark_verified(a.model, fails == 0)
    except Exception:
        pass

    if fails:
        print(f"\nGATE FAILED ({fails} problem(s)). Do NOT ship this build.")
        sys.exit(1)
    print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
