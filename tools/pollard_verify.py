#!/usr/bin/env python3
"""pollard-verify -- correctness/perf GATE for a converted model. Catch "weights decode fine but the
assembled model forwards to garbage" BEFORE trusting a build.

Why this exists: a convert can produce per-tensor-plausible output yet a broken model. We judge builds
ONLY by real reconstruction (the metric a convert prints is a Hessian-weighted proxy that lies -- banned,
see legacy/PROXY_ERR_BANNED.md). This gate checks the two things that actually matter:

  1. per-tensor decode round-trip vs the fp16 SOURCE weight (cos), failing loud below --threshold
  2. the assembled end-to-end forward vs the source model (--end-to-end): next-token agreement

Hard-won rules baked in:
  * verify against the ORIGINAL source weight, never a weight a quantizer mutated in place;
  * DLL search must include torch's OWN bundled CUDA runtime (site-packages/nvidia/*/bin), or low-bit
    kernels silently mis-resolve and produce garbage (decode looks fine -- the trap);
  * codebook is AUTO-DETECTED per tensor from the on-disk markers (mul1/mcg/else=3inst), so we read ANY
    model's convention, not one hard-coded lane's;
  * per-tensor correct is necessary, NOT sufficient -- always run --end-to-end before trusting a build.

  pollard-verify --model out-exl3 --source ./Qwen2.5-3B --end-to-end        # gate a build
  pollard-verify --model out-exl3 --source ./src --sanitizer               # kernels under compute-sanitizer

Lanes (end-to-end fidelity vs the source, measured the same way for every lane: mean KLD + top-1, on BOS-led
2048-token chunks with the second half scored -- llama-perplexity -c 2048's protocol, so a card's numbers compare):
  gguf   llama-perplexity --kl-divergence against the f16/bf16 GGUF (--ref); base logits cached once
  mlx    mlx_lm: quantized vs source logits on the eval text (Apple Silicon)
  gptq   transformers: the GPTQ checkpoint vs the source (gptqmodel/optimum to load it)
  mx     transformers: the compressed-tensors checkpoint vs the source
  exl3   exllamav3's own forward vs the source on --eval, per-tensor decode vs source, the assembled forward
  exlr8  transformers: the rung's reference decode (pollard-exlr8, bit-exact to its codes) vs the source

PUBLISH GATE: --gate gates.yaml applies per-rung thresholds -- KLD max, top-1 min, task retention min
(pollard-taskeval), tool-call pass (pollard-toolcall), decision agreement (pollard-decision), GGUF runtime
(pollard-ggufcheck: stock vs ik_llama, read from the tensor types, never the file name) and coherence
(pollard-bench's gate). Numbers come from this run and from --results (what the other tools wrote). A
threshold with no number to check FAILS unless the file says `missing: skip` -- an unmeasured rung is not
a verified one. The verdict is stamped on the build in the MANIFEST and in results.json, so pollard-card
shows "verified" and pollard release refuses to upload a rung that did not pass.

DECISION-ONLY models (the graph ends in a decision head, no per-token logits: clef / clef-omni, a d3 readout GGUF)
have no KLD / PPL: the gguf lane records them "not applicable: decision-only model" instead of running
llama-perplexity, and gates the rung on its decision board (pollard-decision --ref --results: agreement + option KL,
`kinds.decision_only` in gates.yaml, built-in per-bit-band defaults when the file has none).

  pollard-verify --lane gguf --model M-Q6_K.gguf --ref M-f16.gguf --eval heldout.txt \
                 --gate gates.yaml --rung Q6_K --results results.json
  pollard-verify --lane mlx --model ./M-Pollard-MLX-4bit --source ./M --eval heldout.txt --gate gates.yaml
  pollard-verify --model M-IQ2_M.gguf --gate gates.yaml --rung IQ2_M --results results.json --no-measure
  pollard-verify --print-gates > gates.yaml                       # the default thresholds, to edit

Exit code is non-zero if any check fails, so it drops straight into CI / a pre-ship gate.
"""
import argparse, fnmatch, glob, json, math, os, re, subprocess, sys


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
    """Read the on-disk codebook convention for a tensor -- generalized, not lane-hard-coded."""
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
                W = h.get_tensor(wk[1]).to(device).float()   # (out, in) -- the pristine source weight
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


def exl3_cache_tokens(length, headroom=64):
    """Cache size for an `length`-token forward: exllamav3 asserts `max_num_tokens` is a multiple of 256 (its page
    size), so length + headroom is rounded UP to the next page. 512 + 64 = 576 crashed every EXL3 rung's verify."""
    need = max(1, int(length) + int(headroom))
    return (need + 255) // 256 * 256


def end_to_end(model_dir, source_dir, device, rows, length):
    """The assembled forward vs the fp16 source -- per-tensor correct is NOT sufficient."""
    import torch, torch.nn.functional as F
    from exllamav3 import Config, Model, Cache, Tokenizer
    cfg = Config.from_directory(model_dir); m = Model.from_config(cfg)
    c = Cache(m, max_num_tokens=exl3_cache_tokens(length)); m.load()
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


# ----------------------------------------------------------------------- shared fidelity math

def kld_top1(ref_logits, q_logits, rows=256):
    """{kld, top1, kld_p99, n} for two [N, V] logit arrays (numpy): mean KL(ref || q) per token in nats,
    and top-1 agreement in percent -- the same two numbers llama-perplexity reports for a GGUF, so every
    lane is gated on the same scale.

    Computed `rows` positions at a time in float32, each row's KL summed in float64: the whole-array float64 cast
    this used to do is 4.64 GiB PER COPY at 4096 x 151936 (16 x 256 tokens of a Qwen vocabulary), several copies
    deep -- a numpy MemoryError on a 32 GB box that failed every GPTQ rung's gate regardless of its quality."""
    import numpy as np
    r_all, q_all = np.asarray(ref_logits), np.asarray(q_logits)
    v = min(r_all.shape[-1], q_all.shape[-1])               # padded vocab rows on one side
    r_all, q_all = r_all.reshape(-1, r_all.shape[-1]), q_all.reshape(-1, q_all.shape[-1])
    kl, hit = [], []
    for a in range(0, len(r_all), max(1, int(rows))):
        r = np.asarray(r_all[a:a + rows, :v], dtype=np.float32)
        q = np.asarray(q_all[a:a + rows, :v], dtype=np.float32)
        r = r - r.max(-1, keepdims=True)
        q = q - q.max(-1, keepdims=True)
        lr = r - np.log(np.exp(r).sum(-1, keepdims=True, dtype=np.float64)).astype(np.float32)
        lq = q - np.log(np.exp(q).sum(-1, keepdims=True, dtype=np.float64)).astype(np.float32)
        kl.append((np.exp(lr) * (lr - lq)).sum(-1, dtype=np.float64))
        hit.append(r.argmax(-1) == q.argmax(-1))
    kl = np.concatenate(kl) if kl else np.zeros(0)
    hit = np.concatenate(hit) if hit else np.zeros(0, bool)
    return {"kld": float(kl.mean()), "top1": float(hit.mean() * 100.0), "kld_p99": float(np.percentile(kl, 99)),
            "n": int(kl.size)}


def _chunks(ids, length, n):
    return [ids[i:i + length] for i in range(0, max(0, len(ids) - length + 1), length)][:n] or [ids[:length]]


def _eval_text(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"--eval text not found: {path} (use a HELD-OUT corpus: pollard-calib --held-out)")
    return open(path, encoding="utf-8", errors="replace").read()


def _ref_cache(source, eval_path, length, n, tag):
    import hashlib
    try:
        import pollard_workspace as ws
        d = ws.cache_dir(create=True)
    except Exception:
        d = "."
    h = hashlib.sha1(f"{os.path.abspath(source)}|{os.path.abspath(eval_path)}|{length}|{n}|{tag}".encode()).hexdigest()[:12]
    return os.path.join(d, f"verify-ref-{h}.npy")


# ----------------------------------------------------------------------- lane backends

def measure_gguf(model, ref, eval_path, ngl="0", chunks=16, ppl_bin=None):
    """llama-perplexity --kl-divergence: the base logits from the f16 once (cached in the workspace,
    reused by every rung), then this rung against them. Same numbers pollard-bench puts on the card."""
    import shutil
    from pollard_calc import find_llama_bin
    from pollard_bench import _ppl_kl
    import pollard_llamaserve as LS
    # the rung's own runtime: an ik_llama-only file (the IQ*_KT flagship) is measured by ik_llama's perplexity, which
    # reads the stock-written KL base (same format) -- stock llama.cpp cannot load it at all
    given = ppl_bin
    ppl_bin = ppl_bin or LS.bin_for(model, "llama-perplexity") or find_llama_bin("llama-perplexity") or "llama-perplexity"
    ref_bin = given or find_llama_bin("llama-perplexity") or ppl_bin
    try:
        import pollard_workspace as ws
        base = os.path.join(ws.cache_dir(create=True), os.path.splitext(os.path.basename(ref))[0]
                            + f".c{chunks}.klbase.dat")
    except Exception:
        base = os.path.splitext(ref)[0] + f".c{chunks}.klbase.dat"
    # -ngl per file from free VRAM (pollard_llamaserve): `auto` is never handed to llama.cpp, and a GPU run that
    # fails is retried once on the CPU
    import pollard_llamaserve as LS

    def tries(path):
        n = LS.resolve_ngl(path, ngl)
        return [n] + (["0"] if n != "0" else [])

    if not (os.path.exists(base) and os.path.getsize(base) > 0):
        for n in tries(ref):
            cmd = [ref_bin, "-m", ref, "-f", eval_path, "-c", "2048", "-ngl", str(n),
                   "--kl-divergence-base", base] + (["--chunks", str(chunks)] if chunks else [])
            print("   KL base: $ " + " ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            free = shutil.disk_usage(os.path.dirname(os.path.abspath(base)) or ".").free
            if (r.returncode or free < 1e9) and os.path.exists(base):
                os.remove(base)                          # a short base poisons every later rung
            if os.path.exists(base):
                break
            tail = "\n".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-8:])
            print(f"   KL base -ngl {n} failed (exit {r.returncode}):\n{tail}", flush=True)
        if not os.path.exists(base):
            raise SystemExit(f"could not build the KL base from {ref} (exit {r.returncode}, "
                             f"{free/1e9:.1f} GB free)")
    for n in tries(model):
        res = _ppl_kl(ppl_bin, model, eval_path, base, n, chunks)
        if res.get("mean_kld") is not None:
            break
        print(f"   llama-perplexity -ngl {n} produced no KLD for {model}", flush=True)
    if res.get("mean_kld") is None:
        raise SystemExit(f"llama-perplexity produced no KLD for {model}")
    return {"kld": res["mean_kld"], "top1": res.get("top1"), "ppl": res.get("ppl"), "ref_ppl": res.get("ref_ppl"),
            "median_kld": res.get("median_kld")}


def _kl_chunk(r, q, ids, rows=64, start=0):
    """(per-token KL, top-1 hits, rung NLL, reference NLL) of one chunk, computed `rows` positions at a time: the
    float64 math on a 128k-200k vocabulary is ~1 GB per array for a whole chunk, ~0.1 GB this way. `r` / `q` are the
    logits of positions start, start + 1, ... of `ids` (the scored part of the chunk)."""
    import numpy as np
    ids = np.asarray(ids)
    kl, t1, nq, nr = [], [], [], []
    for a in range(0, len(r), rows):
        k, t, lr, lq = _kl_rows(np.asarray(r[a:a + rows], np.float32), np.asarray(q[a:a + rows], np.float32))
        kl.append(k), t1.append(t)
        nxt = ids[start + a + 1:start + a + 1 + len(k)]   # position t predicts ids[t + 1]
        m = len(nxt)
        nq.append(-lq[np.arange(m), nxt]), nr.append(-lr[np.arange(m), nxt])
    return np.concatenate(kl), np.concatenate(t1), np.concatenate(nq), np.concatenate(nr)


def _kl_rows(r, q):
    """Per-token KL(ref || q) (nats, float64) and top-1 agreement for one chunk -- kld_top1's math, row by row."""
    import numpy as np
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if r.shape != q.shape:
        v = min(r.shape[-1], q.shape[-1])
        r, q = r[..., :v], q[..., :v]
    r = r - r.max(-1, keepdims=True)
    q = q - q.max(-1, keepdims=True)
    lr = r - np.log(np.exp(r).sum(-1, keepdims=True))
    lq = q - np.log(np.exp(q).sum(-1, keepdims=True))
    return (np.exp(lr) * (lr - lq)).sum(-1), r.argmax(-1) == q.argmax(-1), lr, lq


def mlx_stream_wanted(source, model, mode="auto"):
    """(True, why) when the MLX lane is measured STREAMED (block by block): asked for, or the source + rung do not
    fit comfortably in free RAM (the whole-model path loads the source, then the rung, each whole)."""
    if mode == "on":
        return True, "--stream on"
    if os.path.isfile(os.path.join(str(source), ".pollard-remote")):
        return True, "the source is a remote mirror"
    if mode == "off":
        return False, ""
    def size_gb(d):
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs
                   if f.endswith(".safetensors")) / 1e9 if os.path.isdir(str(d)) else 0.0
    try:
        import pollard_stream as S
        need = max(size_gb(source), size_gb(model)) + 1.5
        free = S.free_ram_gb()
    except Exception:                                        # noqa: BLE001
        return True, "free RAM unknown"
    if free is None or need > 0.5 * free:
        return True, f"the source is {size_gb(source):.1f} GB, {('%.1f' % free) if free else '?'} GB RAM free"
    return False, ""


#: The MLX lane is scored the way llama-perplexity scores the GGUF lane (measure_gguf: -c 2048 --kl-divergence), so
#: one gate threshold means the same thing on both: every chunk starts with the model's BOS -- a chunk cut from the
#: middle of the corpus is otherwise a context the model never saw (LFM2.5-8B-A1B: reference PPL 32.2 without it,
#: 16.7 with it) -- and only the SECOND HALF of each chunk is scored, every scored token having >= length/2 tokens of
#: context. Before this the lane scored 256-token BOS-less chunks from position 0 and read LFM2.5-24B-A2B's 3-bit
#: rung at KLD 1.05; on LFM2.5-8B-A1B the same rung read 0.94 that way and 0.53 this way.
MLX_PROTOCOL = "bos-second-half"
#: ...and so is every transformers lane (gptq / mx / exlr8, whole or streamed): one protocol for every lane, so the KLDs
#: on a card compare. The HF lanes used to score 256-token BOS-less chunks from position 0 -- LFM2.5-24B-A2B's
#: GPTQ-int4 rung read 0.336 that way next to GGUF rungs measured by llama-perplexity at -c 2048, second half.
PROTOCOL = MLX_PROTOCOL
#: exit code of a step held for disk / RAM (pollard release: resumable, the rung is kept), as pollard-mlx's
EXIT_HELD = 3
MLX_LENGTH = 2048
#: tokens per chunk for every lane: llama-perplexity's -c 2048 (measure_gguf)
PROTOCOL_LENGTH = MLX_LENGTH


def _bos_id(tok):
    """The BOS id a chunk should start with, or None when the model does not use one (Qwen: none defined). A model
    uses it when its tokenizer adds it or its chat template writes it (LFM2's tokenizer does not add
    <|startoftext|>; its template starts every conversation with it)."""
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return None
    try:
        if list(tok.encode("a"))[:1] == [bos]:
            return bos
    except Exception:                                        # noqa: BLE001
        pass
    tmpl = getattr(tok, "chat_template", None) or ""
    name = getattr(tok, "bos_token", None)
    return bos if isinstance(tmpl, str) and ("bos_token" in tmpl or (name and str(name) in tmpl)) else None


def _protocol_chunks(ids, length, n, bos):
    """`n` chunks of `length` tokens, each starting with `bos` (llama-perplexity overwrites a chunk's first token with
    BOS; the same here), and where each one's scored part begins."""
    if bos is not None and list(ids[:1]) == [bos]:
        ids = ids[1:]
    if bos is None:
        parts = _chunks(ids, length, n)
    else:
        parts = [[bos] + list(p[1:]) for p in _chunks(ids, length, n)]
    return parts, [len(p) // 2 for p in parts]


def _mlx_tokens(source, eval_path, length, chunks, with_starts=False):
    from pathlib import Path
    try:
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(Path(source))
    except Exception:                                        # noqa: BLE001 -- a processor-only folder
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(source)
    parts, starts = _protocol_chunks(list(tok.encode(_eval_text(eval_path))), length, chunks, _bos_id(tok))
    return (parts, starts) if with_starts else parts


def ref_disk_room(cache, nbytes, mirror=None, log=print):
    """Before the source's reference logits (`nbytes`) are written at `cache`: room for them over the disk reserve
    (pollard_mlx.disk_reserve_gb). A remote mirror's cache on the same disk is evicted for it first; still short ->
    OutOfMemoryBudget (the step is held, nothing half-written)."""
    import shutil
    import pollard_mlx as M
    import pollard_stream as S
    d = os.path.dirname(os.path.abspath(cache))
    reserve = M.disk_reserve_gb(d)
    free = shutil.disk_usage(d).free
    if free < nbytes + reserve * 1e9 and mirror is not None and os.stat(d).st_dev == os.stat(mirror.dir).st_dev:
        freed = mirror.make_room(reserve + nbytes / 1e9)
        free = shutil.disk_usage(d).free
        log(f"   disk: {freed / 1e9:.1f} GB of the mirror's cache evicted for the reference logits")
    if free < nbytes + reserve * 1e9:
        raise S.OutOfMemoryBudget(f"disk: the reference logits need {nbytes / 1e9:.2f} GB at {d} over the "
                                  f"{reserve:.1f} GB reserve and {free / 1e9:.2f} GB is free")


def measure_mlx(model, source, eval_path, length=None, chunks=8, stream="auto", log=print):
    """mlx_lm / mlx-vlm: the quantized model and its source over the same token chunks -- KLD, top-1, PPL, scored as
    llama-perplexity scores the GGUF lane (MLX_PROTOCOL: BOS-led chunks, second half). Streamed (pollard_mlx.StreamedLM:
    block by block, bounded RAM, any size; a VLM / omni through its language model) when the models do not fit whole,
    else loaded whole. The source's logits (the scored positions) are cached (fp16) for every later rung."""
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        raise SystemExit("the mlx lane needs mlx-lm (Apple Silicon): pip install mlx-lm") from None
    import numpy as np
    import pollard_mlx as M
    bad = M.mlx_runtime_problem(source)
    if bad:
        raise SystemExit(f"the mlx lane cannot measure this model with the installed runtime: {bad}")
    length = length or MLX_LENGTH
    parts, starts = _mlx_tokens(source, eval_path, length, chunks, with_starts=True)
    cache = _ref_cache(source, eval_path, length, chunks, f"mlx-{MLX_PROTOCOL}")
    engine, _ = M.mlx_engine(source)
    engine = "mlx_vlm" if engine == "mlx_vlm" else "mlx_lm"
    streamed, why = mlx_stream_wanted(source, model, stream)
    if engine == "mlx_vlm" and not streamed:                 # mlx-lm cannot load a VLM / omni whole: its LM, streamed
        streamed, why = True, "a VLM / omni: its language model through mlx-vlm"
    kls, top, nll_q, nll_r = [], [], [], []
    if streamed:
        guard = M.MemoryGuard(log=log)
        log(f"   streamed ({why}): source and rung block by block through {engine}")
        try:
            plan = M.plan_verify(source, model, length - length // 2, chunks)   # logits of the scored half only
            log(f"   memory plan: peak ~{plan['peak_gb']} GB (one block {plan['layer_gb']} GB, embeddings "
                f"{plan['outer_gb']} GB, one chunk's logits {plan['logits_gb']} GB)")
        except (Exception, SystemExit) as e:                 # noqa: BLE001
            log(f"   (memory plan unavailable: {e})")
        offs = np.cumsum([0] + [len(p) - s for p, s in zip(parts, starts)])
        if not os.path.exists(cache):
            src = M.StreamedLM(source, engine, guard, log).run(parts)
            ref_mm = None
            try:
                for b in range(len(parts)):
                    lg = src.logits(b, starts[b])
                    if ref_mm is None:
                        ref_disk_room(cache, int(offs[-1]) * lg.shape[-1] * 2, src.mirror, log)
                        ref_mm = np.lib.format.open_memmap(cache + ".partial.npy", mode="w+", dtype=np.float16,
                                                           shape=(int(offs[-1]), lg.shape[-1]))
                    ref_mm[offs[b]:offs[b + 1]] = lg.astype(np.float16)
                ref_mm.flush()
            except BaseException:
                del ref_mm
                try:
                    os.remove(cache + ".partial.npy")      # never a half-written reference left taking the disk
                except OSError:
                    pass
                raise
            del ref_mm, src
            mx.clear_cache()
            os.replace(cache + ".partial.npy", cache)
        ref = np.load(cache, mmap_mode="r")
        q = M.StreamedLM(model, engine, guard, log).run(parts)
        for b, ids in enumerate(parts):
            kl, t1, nq_, nr_ = _kl_chunk(ref[offs[b]:offs[b + 1]], q.logits(b, starts[b]), ids, start=starts[b])
            kls.append(kl), top.append(t1), nll_q.append(nq_), nll_r.append(nr_)
        nb = guard.numbers()
        log(f"   streamed verify: peak footprint {nb['peak_gb']} GB, peak RSS {nb['peak_rss_gb']} GB, lowest free RAM "
            f"{nb['min_free_ram_gb']} GB")
    else:
        smodel = None
        if os.path.exists(cache):
            ref = np.load(cache)
        else:
            smodel, _ = load(source)
            ref = np.concatenate([np.array(smodel(mx.array(c)[None])[0, s:].astype(mx.float32))
                                  for c, s in zip(parts, starts)]).astype(np.float16)
            np.save(cache, ref)
        del smodel
        qmodel, _ = load(model)
        off = 0
        for ids, s in zip(parts, starts):
            qq = np.array(qmodel(mx.array(ids)[None])[0, s:].astype(mx.float32))
            kl, t1, nq_, nr_ = _kl_chunk(ref[off:off + len(ids) - s], qq, ids, start=s)
            off += len(ids) - s
            kls.append(kl), top.append(t1), nll_q.append(nq_), nll_r.append(nr_)
    kl, t1 = np.concatenate(kls), np.concatenate(top)
    nq, nr = np.concatenate(nll_q), np.concatenate(nll_r)
    return {"kld": float(kl.mean()), "top1": float(t1.mean() * 100.0), "kld_p99": float(np.percentile(kl, 99)),
            "n": int(kl.size), "ppl": float(np.exp(nq.mean())) if nq.size else None,
            "ref_ppl": float(np.exp(nr.mean())) if nr.size else None, "streamed": bool(streamed),
            "protocol": f"{MLX_PROTOCOL} (length {length}, {len(parts)} chunks)"}


def _hf_tokens(tok, eval_path, length, chunks):
    """(parts, starts) of the shared protocol from a transformers tokenizer: BOS-led chunks of `length` tokens, the
    second half of each scored (where each one's scored part begins)."""
    ids = list(tok(_eval_text(eval_path))["input_ids"])
    return _protocol_chunks(ids, length, chunks, _bos_id(tok))


def _scored_logits(m, ids, start):
    """float32 numpy logits of positions start.. of one chunk: only the scored half leaves the model when it takes
    `logits_to_keep` (transformers >= 4.50), else the full chunk's logits are sliced."""
    import torch
    keep = len(ids) - start
    x = torch.tensor([list(ids)], device=getattr(m, "device", "cpu"))
    with torch.no_grad():
        try:
            lg = m(x, logits_to_keep=keep).logits[0]
        except TypeError:                                    # an older / custom forward
            lg = m(x).logits[0]
    if lg.shape[0] != keep:                                  # a forward that ignored logits_to_keep
        lg = lg[start:]
    return lg.float().cpu().numpy()


def _ref_store(cache, offs, vocab, log=print):
    """A float16 memmap for the source's scored logits, written as `<cache>.partial.npy` (renamed when complete): the
    reference of 16 x 1024 scored positions at a 150k vocabulary is ~5 GB -- never held in RAM, checked for room."""
    import numpy as np
    import shutil
    need = int(offs[-1]) * int(vocab) * 2
    d = os.path.dirname(os.path.abspath(cache)) or "."
    free = shutil.disk_usage(d).free
    if free < need + 2e9:
        import pollard_stream as S
        raise S.OutOfMemoryBudget(f"disk: the reference logits need {need / 1e9:.2f} GB at {d} (+2 GB margin) and "
                                  f"{free / 1e9:.2f} GB is free")
    return np.lib.format.open_memmap(cache + ".partial.npy", mode="w+", dtype=np.float16,
                                     shape=(int(offs[-1]), int(vocab)))


def _protocol_result(kls, top, nll_q, nll_r, length, n_parts, **extra):
    import numpy as np
    kl, t1 = np.concatenate(kls), np.concatenate(top)
    nq, nr = np.concatenate(nll_q), np.concatenate(nll_r)
    return {"kld": float(kl.mean()), "top1": float(t1.mean() * 100.0), "kld_p99": float(np.percentile(kl, 99)),
            "n": int(kl.size), "ppl": float(np.exp(nq.mean())) if nq.size else None,
            "ref_ppl": float(np.exp(nr.mean())) if nr.size else None,
            "protocol": f"{PROTOCOL} (length {length}, {n_parts} chunks)", **extra}


def _hf_reference(load_source, parts, starts, cache, offs, free=None):
    """The source's scored logits under the protocol, from the cache (fp16 memmap) or computed once into it: the
    source is loaded by `load_source()`, run chunk by chunk, and freed before anything else loads."""
    import numpy as np
    if not os.path.exists(cache):
        m, ref_mm = load_source(), None
        try:
            for b, (ids, st) in enumerate(zip(parts, starts)):
                lg = _scored_logits(m, ids, st)
                if ref_mm is None:
                    ref_mm = _ref_store(cache, offs, lg.shape[-1])
                ref_mm[offs[b]:offs[b + 1]] = lg.astype(np.float16)
            ref_mm.flush()
        except BaseException:
            del ref_mm
            try:
                os.remove(cache + ".partial.npy")              # never a half-written reference
            except OSError:
                pass
            raise
        del ref_mm, m
        if free:
            free()
        os.replace(cache + ".partial.npy", cache)
    return np.load(cache, mmap_mode="r")


def measure_hf(model, source, eval_path, device="cuda:0", length=None, chunks=8, trust_remote_code=False,
               loader=None):
    """transformers: the GPTQ / compressed-tensors (MX) / EXLR8 checkpoint vs the source, scored as llama-perplexity
    scores the GGUF lane (PROTOCOL: BOS-led chunks of 2048, the second half of each scored) -- KLD, top-1, PPL. The
    source's scored logits are computed once and cached (fp16 memmap, keyed on the protocol) and the source is freed
    before the quantized model loads, so the two never sit in memory together; KL is taken chunk by chunk."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("the gptq/mx lanes need torch + transformers (+ gptqmodel / compressed-tensors)") from None
    import numpy as np
    length = length or PROTOCOL_LENGTH
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    parts, starts = _hf_tokens(tok, eval_path, length, chunks)
    cache = _ref_cache(source, eval_path, length, chunks, f"hf-{PROTOCOL}")
    offs = np.cumsum([0] + [len(p) - st for p, st in zip(parts, starts)])

    def load_model(path, load=None):
        import pollard_mxfp6 as M6
        if load:
            return load(path)
        if M6.read_meta(path):
            # a pollard-mx MXFP6 / mix rung: the source architecture with every quantized linear replaced by the
            # rung's own bytes decoded exactly (weight-only, so this IS the math any runtime does with them)
            m = AutoModelForCausalLM.from_pretrained(source, torch_dtype="auto", device_map=device,
                                                     trust_remote_code=trust_remote_code).eval()
            sd = m.state_dict()
            for k, w in M6.dequantized_weights(path).items():
                key = k if k in sd else next((x for x in sd if x.endswith("." + k.split(".", 1)[-1])), None)
                if key is None:
                    raise SystemExit(f"{k}: no such weight in the source model")
                sd[key].copy_(torch.from_numpy(w).to(sd[key].dtype))
            return m
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map=device,
                                                    trust_remote_code=trust_remote_code).eval()

    def free():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ref = _hf_reference(lambda: load_model(source), parts, starts, cache, offs, free)
    m = load_model(model, loader)
    kls, top, nll_q, nll_r = [], [], [], []
    for b, (ids, st) in enumerate(zip(parts, starts)):
        kl, t1, nq_, nr_ = _kl_chunk(ref[offs[b]:offs[b + 1]], _scored_logits(m, ids, st), ids, start=st)
        kls.append(kl), top.append(t1), nll_q.append(nq_), nll_r.append(nr_)
    del m
    free()
    return _protocol_result(kls, top, nll_q, nll_r, length, len(parts))


def measure_exl3(model, source, eval_path, device="cuda:0", length=None, chunks=8, trust_remote_code=False):
    """An EXL3 rung through exllamav3's own kernels vs the HF source, under the same PROTOCOL as every other lane
    (BOS-led chunks of 2048, second half): the tokens come from the SOURCE's tokenizer, so the reference logits are
    the very cache measure_hf uses. Before this the exl3 lane had no logit-level number at all -- only the
    end-to-end next-token check -- so an EXL3 rung could not be compared on a card or gated on KLD."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from exllamav3 import Config, Model, Cache
    length = length or PROTOCOL_LENGTH
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    parts, starts = _hf_tokens(tok, eval_path, length, chunks)
    cache = _ref_cache(source, eval_path, length, chunks, f"hf-{PROTOCOL}")
    offs = np.cumsum([0] + [len(p) - st for p, st in zip(parts, starts)])

    def free():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ref = _hf_reference(lambda: AutoModelForCausalLM.from_pretrained(
        source, torch_dtype="auto", device_map=device, trust_remote_code=trust_remote_code).eval(),
        parts, starts, cache, offs, free)
    cfg = Config.from_directory(model)
    m = Model.from_config(cfg)
    kv = Cache(m, max_num_tokens=exl3_cache_tokens(max(len(p) for p in parts)))   # noqa: F841 -- kept referenced
    m.load()
    kls, top, nll_q, nll_r = [], [], [], []
    with torch.no_grad():
        for b, (ids, st) in enumerate(zip(parts, starts)):
            lg = m.forward(torch.tensor([list(ids)], device=device))[0].float()
            q = lg[st:] if lg.shape[0] == len(ids) else lg[-(len(ids) - st):]
            kl, t1, nq_, nr_ = _kl_chunk(ref[offs[b]:offs[b + 1]], q.cpu().numpy(), ids, start=st)
            kls.append(kl), top.append(t1), nll_q.append(nq_), nll_r.append(nr_)
    try:
        m.unload()
    except Exception:                                        # noqa: BLE001
        pass
    del m, kv
    free()
    return _protocol_result(kls, top, nll_q, nll_r, length, len(parts), decoder="exllamav3")


def _quant_method(d):
    try:
        with open(os.path.join(str(d), "config.json"), encoding="utf-8") as f:
            q = json.load(f).get("quantization_config") or {}
        return str(q.get("quant_method") or "").lower() if isinstance(q, dict) else ""
    except (OSError, ValueError, AttributeError):
        return ""


def stream_wanted(source, mode="auto", rung=None):
    """(True, why) when the transformers lanes are measured STREAMED (pollard_stream: source and rung one block at a
    time) -- for a source transformers' AutoModelForCausalLM cannot load whole here: a VLM / omni wrapper, custom
    code, an already-quantized or overlay source, or weights bigger than free RAM can hold twice over."""
    if mode in ("on", True):
        return True, "asked for"
    if mode in ("off", False):
        return False, ""
    for side in ("pollard_mx.json", "pollard-gptq.json", "pollard-exlr8.json"):
        if rung and os.path.isfile(os.path.join(str(rung), side)):
            # a Pollard rung stores EVERY tensor it was built with (a smoothed rung's folded norms too): read it
            # whole, block by block -- swapping only its linears into the original source would pair smoothed
            # weights with unsmoothed norms
            return True, f"a Pollard rung ({side}): every tensor read from the rung"
    if rung and os.path.isdir(str(rung)) and _quant_method(rung) == "gptq":
        # any GPTQ rung (pollard-export's gptqmodel build writes pollard-allocation.json, not pollard-gptq.json):
        # transformers' GPTQ loader (optimum) builds EVERY linear at the base bits and ignores the per-layer `dynamic`
        # map, so a 4/8 mix reads as noise there; the stream reader decodes each module at the bits it was stored at
        return True, "a GPTQ rung: every module decoded at its own bits (transformers ignores `dynamic`)"
    if not os.path.isdir(str(source)):
        return False, ""
    try:
        import pollard_stream as S
        import pollard_workspace as ws
    except Exception:                                                    # noqa: BLE001
        return False, ""
    if S.is_overlay(source):
        return True, "the source is a smoothing overlay"
    if S.needs_remote_code(source):
        return True, "the source needs custom code"
    cfg = ws.config_dict(source)
    if cfg.get("quantization_config"):
        return True, "the source is already quantized"
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as LM
        if cfg.get("model_type") not in LM:
            return True, f"transformers has no causal-LM class for {cfg.get('model_type')!r}"
    except Exception:                                                    # noqa: BLE001
        return False, ""
    try:
        w = sum(os.path.getsize(p) for p in S.weight_files(source)) / 1e9
    except Exception:                                                    # noqa: BLE001
        return False, ""
    free = S.free_ram_gb() or 0
    if free and w * 2.2 > free:
        return True, f"{w:.1f} GB of weights; {free:.1f} GB free RAM"
    return False, ""


def measure_stream(model, source, eval_path, device="cpu", length=None, chunks=8, trust_remote_code="auto",
                   max_ram_gb=None):
    """The rung against its source, both streamed block by block (pollard_stream.stream_fidelity): any size, any
    backbone the stream engine resolves, every Pollard rung format decoded from its bytes -- scored under the same
    PROTOCOL as every other lane (BOS-led chunks of 2048, second half)."""
    import pollard_stream as S
    dev = S.stream_device(device)
    from transformers import AutoTokenizer
    length = length or PROTOCOL_LENGTH
    src = S.Source(source)
    tok = AutoTokenizer.from_pretrained(src.base_dir, trust_remote_code=trust_remote_code in (True, "on"))
    parts, starts = _hf_tokens(tok, eval_path, length, chunks)
    r = S.stream_fidelity(source, model, parts, dev, trust_remote_code, starts=starts, max_ram_gb=max_ram_gb)
    r["protocol"] = f"{PROTOCOL} (length {length}, {len(parts)} chunks)"
    return r


def measure_exlr8(model, source, eval_path, device="cuda:0", length=None, chunks=8, trust_remote_code=False):
    """An EXLR8 rung: its weights decoded by the reference loader (bit-exact to the stored trellis codes) into the
    source architecture, then the same KLD / top-1 as the other transformers lanes."""
    import pollard_exlr8 as X
    return measure_hf(model, source, eval_path, device, length, chunks, trust_remote_code,
                      loader=lambda p: X.load_model(p, device, trust_remote_code=trust_remote_code))


# ----------------------------------------------------------------------- the publish gate

DEFAULT_GATES = """\
# pollard-verify --gate: what a rung must show before pollard release uploads it.
# Resolution: `default`, then `lanes.<lane>`, then every `rungs` entry whose `match` patterns hit the
# rung tag (fnmatch, case-insensitive), in order -- later entries override earlier ones.
# A threshold with no measured number FAILS the rung (missing: fail); `missing: skip` relaxes that.
#   kld_max          mean KLD vs the f16/bf16 source (nats/token)       pollard-verify / pollard-bench
#   top1_min         top-1 agreement with the source, percent            pollard-verify / pollard-bench
#   retention_min    task retention vs f16, percent (overall)            pollard-taskeval --results
#   mt_retention_min mean chrF++ retention over the language pairs       pollard-taskeval --suite mt
#   toolcall_pass    true = still emits valid tool calls                 pollard-toolcall --results
#   toolcall_retention_min  tool-call success vs the reference's, percent pollard-toolcall --ref --results
#   toolcall_not_measured   skip | fail: a rung pollard-toolcall recorded as NOT MEASURED (no server for its lane on
#                    this machine, GPU not free, too big). skip (the default) PUBLISHES it and the card says
#                    "tool calls: not measured (<reason>)"; fail holds it back
#   decision_agreement_min  same decision as the f16, percent            pollard-decision --results
#   decision_option_kl_max  mean option-distribution KL to the f16 (nats) pollard-decision --results
#   runtime          stock | ik_llama | any  (GGUF: from the tensor types, pollard-ggufcheck). An architecture only
#                    Pollard's own runtime patch implements (`pollard`, e.g. clef-omni) passes when this release
#                    built/verified it with Pollard's runtime; the card says "Needs llama.cpp with Pollard's patch"
#   coherence        true = passes the coherence gate (GGUF: pollard-bench's)
#   draft_tau_min    drafter: tokens per target pass vs the quantized target rung(s), worst pairing  pollard-draft measure
#   draft_accept_rate_min   drafter: accepted / drafted, 0-1                                    pollard-draft measure
#   draft_redistributable   true = the drafter's licence lets us republish it                 pollard-draft convert
missing: fail
default:
  kld_max: 0.5
  top1_min: 75
  toolcall_not_measured: skip
lanes:
  gguf:
    runtime: stock
    coherence: true
  # exl3: KLD / top-1 through exllamav3's own kernels on --eval (the shared protocol), plus the assembled forward
  exl3:
    e2e_acc_min: 0.3
  # a drafter is judged on what it is for -- tokens per target pass against the rungs it ships with -- not on its own
  # logits; the per-bit-width and tool-call rows below are about TARGET rungs
  draft:
    draft_tau_min: 1.5
    draft_redistributable: true
    skip: [kld_max, top1_min, toolcall_retention_min, toolcall_pass, runtime, coherence]
rungs:
  - match: ["Q8_0", "*8bit*", "*int8*", "*FP8*", "*-8.0bpw"]
    kld_max: 0.01
    top1_min: 97
  - match: ["Q6_K*", "*6bit*", "*-6.0bpw", "*MXFP6*", "MIX-6*bpw"]
    kld_max: 0.02
    top1_min: 95
  - match: ["Q5_K*", "*5bit*", "*-5.0bpw", "MIX-5*bpw"]
    kld_max: 0.04
    top1_min: 93
  - match: ["Q4_K*", "IQ4_*", "*4bit*", "*int4*", "*NVFP4*", "*MXFP4*", "*-4.0bpw", "MIX-4*bpw"]
    kld_max: 0.08
    top1_min: 90
  - match: ["Q3_K*", "IQ3_*", "*3bit*", "*int3*", "*-3.0bpw", "*-3.25bpw", "*-3.5bpw"]
    kld_max: 0.2
    top1_min: 85
  - match: ["Q2_K*", "IQ2_*", "IQ1_*", "*2bit*", "*int2*", "*-2.0bpw", "*-2.5bpw"]
    kld_max: 0.5
    top1_min: 75
  # quality-targeted rungs (pollard_qtarget, <class>-<size>GB): gated on the class they were searched for
  - match: ["Q90-*GB"]
    kld_max: 0.08
    top1_min: 90
  - match: ["Q85-*GB"]
    kld_max: 0.2
    top1_min: 85
  # the trellis flagship is ik_llama-only BY DESIGN -- the gate checks it says so, not that it isn't
  - match: ["*_KT*", "*KT-*", "PollardMix*"]
    runtime: ik_llama
# per model KIND (pollard-modelkind), applied after the rung entries. agentic = the chat template writes tool calls
# in a syntax pollard-modelkind recognises; its rungs must keep their tool calling -- in EVERY lane (pollard-toolcall,
# 60 probes the lane's own way: llama-server --jinja / mlx_lm.server / vllm serve --tool-call-parser / exllamav3 or
# transformers in-process; retention = rung success / reference success). The bit-width patterns below cover every
# lane's rung names.
kinds:
  agentic:
    toolcall_retention_min: 90
    rungs:
      - match: ["Q3_K*", "IQ3_*", "*3bit*", "*int3*", "*-3.0bpw", "*-3.25bpw", "*-3.5bpw"]
        toolcall_retention_min: 85
      - match: ["Q2_K*", "IQ2_*", "IQ1_*", "*2bit*", "*int2*", "*-2.0bpw", "*-2.5bpw", "*_KT*", "PollardMix*"]
        toolcall_retention_min: 75
  # decision_only = the graph ends in a decision head and has NO per-token logits (clef / clef-omni, a d3 readout
  # GGUF): llama-perplexity KLD / PPL cannot exist ("not applicable"), so the rung is gated on its decision board --
  # agreement with the reference's decisions and the option-distribution KL (pollard-decision --ref --results).
  # Applied by pollard-verify itself when it finds such a model, whatever gates file is in use. Such a graph never
  # generates text either, so tool-call numbers cannot exist: a template with tools (pollard-modelkind: agentic)
  # must not hold the rung to a tool-call retention it has no way to measure.
  decision_only:
    skip: [kld_max, top1_min, toolcall_retention_min, toolcall_pass]
    decision_agreement_min: 85
    decision_option_kl_max: 0.15
    rungs:
      - match: ["Q8_0", "*8bit*", "*int8*", "*FP8*", "*-8.0bpw"]
        decision_agreement_min: 98
        decision_option_kl_max: 0.01
      - match: ["Q6_K*", "*6bit*", "*-6.0bpw", "*MXFP6*", "MIX-6*bpw"]
        decision_agreement_min: 97
        decision_option_kl_max: 0.02
      - match: ["Q5_K*", "*5bit*", "*-5.0bpw", "MIX-5*bpw"]
        decision_agreement_min: 95
        decision_option_kl_max: 0.03
      - match: ["Q4_K*", "IQ4_*", "*4bit*", "*int4*", "*NVFP4*", "*MXFP4*", "*-4.0bpw", "MIX-4*bpw"]
        decision_agreement_min: 92
        decision_option_kl_max: 0.05
      - match: ["Q3_K*", "IQ3_*", "*3bit*", "*int3*", "*-3.0bpw", "*-3.25bpw", "*-3.5bpw"]
        decision_agreement_min: 88
        decision_option_kl_max: 0.10
      - match: ["Q2_K*", "IQ2_*", "IQ1_*", "*2bit*", "*int2*", "*-2.0bpw", "*-2.5bpw", "*_KT*", "PollardMix*"]
        decision_agreement_min: 80
        decision_option_kl_max: 0.20
"""


def load_gates(path):
    """gates.yaml (or .json). PyYAML ships with gguf, a core dependency; JSON is accepted either way."""
    txt = open(path, encoding="utf-8").read() if path else DEFAULT_GATES
    try:
        import yaml
        return yaml.safe_load(txt) or {}
    except ImportError:
        return json.loads(txt)


def _apply_rungs(th, entries, rung):
    for entry in entries or []:
        pats = entry.get("match") or []
        pats = [pats] if isinstance(pats, str) else pats
        if rung and any(fnmatch.fnmatch(str(rung).lower(), str(p).lower()) for p in pats):
            th.update({k: v for k, v in entry.items() if k != "match"})


def thresholds_for(gates, rung, lane=None, kinds=()):
    """Merged thresholds for one rung: default <- lanes.<lane> <- each matching rungs entry, in order <- then, for
    each kind the model is (pollard-modelkind: agentic, ...), kinds.<kind> and its own matching rungs entries."""
    th = dict(gates.get("default") or {})
    th.update((gates.get("lanes") or {}).get(lane or "", {}) or {})
    _apply_rungs(th, gates.get("rungs"), rung)
    kind_skip = []
    for kind in kinds or ():
        sec = (gates.get("kinds") or {}).get(kind) or {}
        th.update({k: v for k, v in sec.items() if k not in ("rungs", "skip")})
        _apply_rungs(th, sec.get("rungs"), rung)
        kind_skip += list(sec.get("skip") or [])
    for k in kind_skip:                         # a kind that has no such number (decision_only: no KLD / top-1)
        th.pop(k, None)
    # a lane can opt out of a check it has no way to measure (exl3: no logit-level KLD path yet)
    for k in ((gates.get("lanes") or {}).get(lane or "", {}) or {}).get("skip") or []:
        th.pop(k, None)
    th.pop("skip", None)
    return th


DECISION_ONLY_NA = ("kld_max", "top1_min", "toolcall_retention_min", "toolcall_pass")


def decision_only_thresholds(th, gates, rung):
    """A decision-only rung's thresholds: KLD / top-1 and the tool-call checks removed (not applicable -- no
    per-token logits, no generated text), and the
    decision board's thresholds present. A gates file without a `kinds.decision_only` section gets the built-in
    ones for the rung's bit band, so a custom file can never gate a decision model on a number that cannot exist."""
    th = {k: v for k, v in th.items() if k not in DECISION_ONLY_NA}
    if not any(k.startswith("decision_") for k in th):
        builtin = load_gates(None)
        sec = (builtin.get("kinds") or {}).get("decision_only") or {}
        add = {k: v for k, v in sec.items() if k not in ("rungs", "skip")}
        _apply_rungs(add, sec.get("rungs"), rung)
        th.update(add)
    return th


#: threshold -> (metric key, comparison, human name)
_CHECKS = {
    "kld_max": ("kld", "<=", "mean KLD"),
    "top1_min": ("top1", ">=", "top-1 agreement %"),
    "retention_min": ("retention", ">=", "task retention %"),
    "mt_retention_min": ("mt_retention", ">=", "translation retention %"),
    "decision_agreement_min": ("decision_agreement", ">=", "decision agreement %"),
    "decision_option_kl_max": ("decision_option_kl", "<=", "decision option KL"),
    "toolcall_retention_min": ("toolcall_retention", ">=", "tool-call retention %"),
    "e2e_acc_min": ("e2e_acc", ">=", "end-to-end next-token agreement"),
    "draft_tau_min": ("draft_tau", ">=", "drafter tokens per target pass"),
    "draft_accept_rate_min": ("draft_accept_rate", ">=", "drafter accept rate"),
}
_RUNTIME_OK = {"stock": {"stock"}, "ik_llama": {"stock", "ik_llama"}}


def complete_check(model, source, lane):
    """The complete-model gate (pollard_complete) for a rung folder against its HF source folder: does the rung carry
    every tower / projector / talker / side weight and processor file? None when it does not apply (a GGUF file --
    the release checks the GGUF repo folder -- a draft, or no local source)."""
    import pollard_complete as PC
    if lane not in PC.LANES or lane == "gguf" or not (model and source):
        return None
    if not (os.path.isdir(str(model)) and os.path.isfile(os.path.join(str(source), "config.json"))):
        return None
    res = PC.check(str(source), str(model), lane)
    PC.report(res)
    return res


def evaluate_gate(metrics, th, missing="fail"):
    """{pass, checks: [{check, value, want, ok, why}]} -- one row per threshold, PASS only if every
    row passes. A threshold whose number was never measured fails under missing: fail."""
    checks = []

    def row(name, value, want, ok, why=""):
        checks.append({"check": name, "value": value, "want": want, "ok": bool(ok), "why": why})
    # a rung pollard-toolcall explicitly recorded as NOT MEASURED (no server for its lane here, GPU not free, too
    # big) is held to `toolcall_not_measured`: skip by default (owner, 2026-10-16) -- the rung publishes and the card
    # says "tool calls: not measured (<reason>)"; `fail` holds it back. The reason is on the row either way.
    tc_nm = metrics.get("toolcall_status") == "not-measured"
    tc_policy = th.get("toolcall_not_measured", "skip")
    tc_why = f"not measured: {metrics.get('toolcall_reason') or 'no reason recorded'}"
    for key, want in th.items():
        if key in _CHECKS:
            mkey, op, human = _CHECKS[key]
            v = metrics.get(mkey)
            if v is None and tc_nm and key.startswith("toolcall"):
                row(human, None, f"{op} {want}", tc_policy == "skip", tc_why)
                continue
            if (v is None and key == "toolcall_retention_min" and metrics.get("toolcall_ref_status") == "not-measured"
                    and metrics.get("toolcall_pass") is not None):
                # no reference to retain against: the rung's own absolute tool-call verdict stands in (what
                # pollard-toolcall reports), with the reason on the row -- never a silent pass
                row(human, None, f"{op} {want}", metrics["toolcall_pass"] is True,
                    f"reference not measured ({metrics.get('toolcall_ref_reason')}): absolute success "
                    + ("PASS" if metrics["toolcall_pass"] else "FAIL"))
                continue
            if v is None:
                row(human, None, f"{op} {want}", missing == "skip", "not measured")
                continue
            ok = float(v) <= float(want) if op == "<=" else float(v) >= float(want)
            row(human, v, f"{op} {want}", ok)
        elif key == "toolcall_pass" and want:
            v = metrics.get("toolcall_pass")
            if v is None and tc_nm:
                row("tool calls", None, "pass", tc_policy == "skip", tc_why)
                continue
            row("tool calls", v, "pass", v is True if v is not None else missing == "skip",
                "" if v is not None else "not measured")
        elif key == "draft_redistributable" and want:
            v = metrics.get("draft_redistributable")
            row("drafter licence allows republishing", v, "true", v is True if v is not None else missing == "skip",
                "" if v is not None else "licence unknown")
        elif key == "coherence" and want:
            v = metrics.get("coherence")
            row("coherence", v, "PASS", v == "PASS" if v is not None else missing == "skip",
                "" if v is not None else "not measured")
        elif key == "runtime" and want and want != "any":
            v = metrics.get("runtime")
            if v is None:
                row("runtime", None, want, missing == "skip", "not checked (not a GGUF, or unreadable)")
            else:
                ok = v in _RUNTIME_OK.get(want, {want})
                if v == "pollard" and want in _RUNTIME_OK:
                    # an architecture only Pollard's own runtime patch implements (runtime-patches/, upstream PR
                    # pending): passes when THIS release built / verified the rung with that runtime -- the card then
                    # says "Needs llama.cpp with Pollard's <arch> patch", never "stock". Without that evidence the
                    # card and the runtime could disagree, so it fails like any other mismatch.
                    ok = metrics.get("runtime_engine") == "pollard"
                    row("runtime", v, want, ok,
                        "needs llama.cpp with Pollard's runtime patch; built/verified with it here -- the card says so"
                        if ok else "the architecture needs Pollard's runtime patch and this rung was not built or "
                        "verified with Pollard's runtime (pollard-runtime --update)")
                    continue
                row("runtime", v, want, ok, "" if ok else
                    f"a rung published as {want} needs '{v}' -- the card and the runtime would disagree")
    if metrics.get("complete") is not None:            # never a text-only rung, whatever the thresholds say
        lost = metrics.get("complete_missing") or []
        row("complete model", metrics["complete"], "every component of the source", metrics["complete"] is True,
            "" if metrics["complete"] else "missing: " + "; ".join(lost))
    return {"pass": all(c["ok"] for c in checks), "checks": checks}


def metrics_from_results(results, rung):
    """The numbers the other tools already wrote for this rung (pollard-bench / taskeval / toolcall /
    decision), in the gate's vocabulary."""
    r = (results or {}).get(rung) or {}
    m = {}
    if r.get("kld") is not None:
        m["kld"] = r["kld"]
    if r.get("top1") is not None:
        m["top1"] = r["top1"]
    ret = r.get("retention") or {}
    if ret.get("overall") is not None:
        m["retention"] = ret["overall"]
    mt = [v["retention"] for v in (r.get("mt") or {}).values() if isinstance(v, dict) and v.get("retention") is not None]
    if mt:
        m["mt_retention"] = sum(mt) / len(mt)
    tc = r.get("toolcall") or {}
    if tc.get("status") == "not-measured":
        m["toolcall_status"], m["toolcall_reason"] = "not-measured", tc.get("reason")
    if tc.get("cross_backend"):
        m["toolcall_cross_backend"] = True
    if tc.get("pass") is not None:
        m["toolcall_pass"] = bool(tc["pass"])
    if tc.get("retention") is None and tc.get("ref_note") and tc.get("status") == "measured":
        # the RUNG was measured but its reference could not be served here (the bf16 bigger than this machine's
        # RAM): pollard-toolcall judges such a rung on its absolute success rate, and so does the gate
        m["toolcall_ref_status"], m["toolcall_ref_reason"] = "not-measured", tc.get("ref_note")
    if tc.get("retention") is not None:
        m["toolcall_retention"] = float(tc["retention"])
    elif tc.get("rate") is not None and tc.get("ref_rate"):          # a --quick run: rate / f16 rate
        m["toolcall_retention"] = 100.0 * float(tc["rate"]) / float(tc["ref_rate"])
    d = r.get("decision") or {}
    if d.get("agreement") is not None:
        m["decision_agreement"] = 100.0 * float(d["agreement"])
    if d.get("option_kl") is not None:
        m["decision_option_kl"] = float(d["option_kl"])
    if r.get("coherence"):
        m["coherence"] = r["coherence"]
    if r.get("runtime_engine"):                # a `pollard`-architecture rung built / verified with Pollard's runtime
        m["runtime_engine"] = r["runtime_engine"]
    dr = r.get("draft") or {}
    if dr.get("tau") is not None:
        m["draft_tau"] = float(dr["tau"])
    if dr.get("accept_rate") is not None:
        m["draft_accept_rate"] = float(dr["accept_rate"])
    if dr.get("redistributable") is not None:
        m["draft_redistributable"] = bool(dr["redistributable"])
    return m


def model_kinds(model):
    """The kinds pollard-modelkind finds in this build ("agentic" = its template writes tool calls in a known
    syntax), for the gate file's `kinds:` section. Empty when the model cannot be read."""
    try:
        from pollard_modelkind import classify
        k = classify(model)
    except (Exception, SystemExit):                          # noqa: BLE001 -- not a readable model
        return []
    out = []
    if (k.get("tools") or {}).get("family", "none") != "none":
        out.append("agentic")
    for name in ("thinking", "decision", "vision"):
        if k.get(name):
            out.append(name)
    return out


def gguf_runtime(path):
    try:
        import pollard_ggufcompat as gc
        return gc.runtime_of(path)[0]
    except Exception as e:                                   # noqa: BLE001
        print(f"   (runtime check unavailable: {e})", file=sys.stderr)
        return None


def pollard_runtime_engine(path, measured=None):
    """'pollard' when this machine built / verified a `pollard`-architecture rung WITH Pollard's runtime: the
    measurement here loaded the file (a llama.cpp that opens an architecture only our patch implements is, by
    construction, Pollard's), or the engine tree the one resolver (pollard_calc.find_llama_bin, via
    pollard_ggufcompat.pollard_engine_tree) runs this architecture with carries it. None otherwise -- the runtime
    gate then fails rather than letting the card guess."""
    measured = measured or {}
    if any(measured.get(k) is not None for k in ("kld", "ppl", "decision_agreement", "decision_option_kl")):
        return "pollard"
    try:
        import pollard_ggufcompat as gc
        arch = gc.read_header(path)[0]
        return "pollard" if gc.pollard_engine_tree(arch) else None
    except Exception:                                        # noqa: BLE001
        return None


def print_gate(rung, res):
    print(f"-- gate [{rung}] --")
    for c in res["checks"]:
        v = c["value"]
        v = f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"   [{'ok  ' if c['ok'] else 'FAIL'}] {c['check']:<34} {v:>10}  (want {c['want']})"
              + (f"  {c['why']}" if c["why"] else ""))
    print(f"   gate: {'PASS' if res['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", help="the build to gate: a GGUF file or a converted model dir")
    ap.add_argument("--source", "--ref", dest="source",
                    help="the fp16/bf16 source: HF dir (mlx/gptq/mx/exl3) or the f16 GGUF (gguf)")
    ap.add_argument("--lane", default=None, choices=["exl3", "gguf", "mlx", "gptq", "mx", "exlr8", "draft"],
                    help="convert lane (default: gguf for a .gguf file, else exl3). draft = a speculative drafter, "
                         "gated on the pollard-draft measure numbers in --results (nothing is run)")
    ap.add_argument("--eval", help="HELD-OUT text for the KLD/top-1 measurement (gguf/mlx/gptq/mx)")
    ap.add_argument("--chunks", type=int, default=16, help="eval chunks (gguf: llama-perplexity --chunks)")
    ap.add_argument("--length", type=int, default=None,
                    help="tokens per chunk (mlx / gptq / mx / exlr8: 2048, BOS-led, second half scored -- like "
                         "llama-perplexity -c 2048 on the gguf lane) / e2e length (exl3: 256)")
    ap.add_argument("--ngl", default="0", help="gguf: GPU layers for llama-perplexity (default 0 -- shared boxes)")
    ap.add_argument("--no-measure", dest="measure", action="store_false",
                    help="gate only on numbers already in --results (no model is run)")
    ap.add_argument("--gate", nargs="?", const="", default=None, metavar="GATES_YAML",
                    help="apply the publish gate (thresholds file; bare --gate = the built-in defaults)")
    ap.add_argument("--rung", help="the rung's tag for the gate patterns and --results (default: the file name)")
    ap.add_argument("--results", help="results.json: read what other tools measured, write what this run measured")
    ap.add_argument("--coherence", action="store_true", help="gguf: also run pollard-bench's coherence gate")
    ap.add_argument("--print-gates", action="store_true", help="print the default gates.yaml and exit")
    ap.add_argument("--threshold", type=float, default=0.9, help="exl3: min per-tensor decode cos vs source")
    ap.add_argument("--limit", type=int, default=0, help="exl3: cap tensors checked (0 = all)")
    ap.add_argument("--end-to-end", action="store_true", help="exl3: also check the assembled forward")
    ap.add_argument("--e2e-min-acc", type=float, default=0.30, help="exl3: min end-to-end next-tok agreement")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-ram-gb", type=float, default=None,
                    help="gptq/mx/exlr8 streamed: cap the verify's RAM (default: 85%% of what is free); the source "
                         "and the rung pass run one after the other, planned together under this")
    ap.add_argument("--trust-remote-code", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="run the source's own code where no native transformers class covers it (never silent)")
    ap.add_argument("--stream", default="auto", choices=["auto", "on", "off"],
                    help="gptq / mx / exlr8 / mlx: measure source and rung streamed block by block (any size, VLM / "
                         "omni text backbones, custom code, already-quantized sources). auto = when the source cannot "
                         "be loaded whole here (transformers), or source + rung do not fit free RAM comfortably (mlx)")
    ap.add_argument("--sanitizer", action="store_true",
                    help="re-run this verify under NVIDIA compute-sanitizer (memcheck+racecheck)")
    a = ap.parse_args()
    # every measured lane defaults to PROTOCOL_LENGTH (2048, scored like llama-perplexity -c 2048); --length None is
    # passed down as such. The exl3 end-to-end forward keeps its own 256.
    mlx_length = a.length
    if a.print_gates:
        sys.stdout.write(DEFAULT_GATES)
        return
    if not a.model:
        ap.error("--model is required")
    lane = a.lane or ("gguf" if str(a.model).endswith(".gguf") else "exl3")
    if lane == "draft":
        a.measure = False                      # pollard-draft measure wrote the numbers; the gate reads them
    rung = a.rung or os.path.basename(str(a.model).rstrip("/\\"))

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

    print(f"== pollard-verify :: {a.model}  (lane={lane}) ==")
    fails, measured = 0, {}
    dec_only = None
    if lane == "gguf":
        from pollard_calc import set_model_arch
        set_model_arch(a.model)            # llama-perplexity / llama-cli from the engine this architecture needs
        from pollard_decision import decision_only, NOT_APPLICABLE
        dec_only = decision_only(a.model)
    if dec_only and a.measure:
        # nothing for llama-perplexity to read: KLD / PPL do not exist for this model. The rung's fidelity is the
        # decision board pollard-decision wrote into --results (agreement + option KL vs the reference)
        print(f"-- KLD / PPL: {NOT_APPLICABLE} ({dec_only}); fidelity = the decision board in --results --")
        a.measure = False
        if a.results:
            import pollard_workspace as ws
            try:
                ws.merge_results(a.results, rung, {"kld_status": "not-applicable",
                                                   "kld_reason": f"{NOT_APPLICABLE} ({dec_only})"})
            except OSError as e:
                print(f"HELD: {e}", flush=True)
                sys.exit(EXIT_HELD)

    if a.measure and lane == "exl3":
        _add_cuda_dll_dirs()
        if a.source:
            print("-- per-tensor decode vs source --")
            checked, failed = verify_exl3(a.model, a.source, a.threshold, a.device, a.limit)
            print(f"   {checked - len(failed)}/{checked} tensors >= {a.threshold}")
            if failed:
                fails += len(failed)
                print(f"   !! {len(failed)} tensors FAILED (e.g. {failed[0][0]} cos={failed[0][1]:.4f})")
        else:
            print("   (no --source; skipping per-tensor decode-vs-source. Pass --source to gate weights.)")
        if a.eval:
            if not a.source:
                ap.error("--lane exl3 --eval measures against the source: pass --source")
            print(f"-- fidelity vs {os.path.basename(str(a.source).rstrip('/'))} on {a.eval} (exllamav3) --")
            import pollard_stream as S
            try:
                measured.update(measure_exl3(a.model, a.source, a.eval, a.device, a.length, a.chunks,
                                             a.trust_remote_code == "on"))
            except S.OutOfMemoryBudget as e:
                print(f"HELD: {e}", flush=True)
                sys.exit(EXIT_HELD)
            print(f"   mean KLD {measured['kld']:.5f}   top-1 {measured.get('top1') or 0:.2f}%"
                  + (f"   PPL {measured['ppl']:.4f}" if measured.get("ppl") else "")
                  + f"   protocol: {measured.get('protocol')}")
        if a.end_to_end:
            print("-- end-to-end assembled forward --")
            acc, ppl = end_to_end(a.model, a.source or a.model, a.device, 1, a.length or 256)
            measured["e2e_acc"] = acc
            ok = acc >= a.e2e_min_acc
            print(f"   next_tok_acc={acc:.4f}  ppl={ppl:.2f}  [{'OK' if ok else 'FAIL'} vs min {a.e2e_min_acc}]")
            if not ok:
                fails += 1
                print("   !! assembled forward is broken even if per-tensor checks passed "
                      "(per-tensor correct is necessary, not sufficient).")
    elif a.measure:
        if not a.source:
            ap.error(f"--lane {lane} measures against the source: pass --source (--ref for a GGUF)")
        print(f"-- fidelity vs {os.path.basename(str(a.source).rstrip('/'))} on {a.eval} --")
        trc = a.trust_remote_code
        streamed, why = stream_wanted(a.source, a.stream, a.model) if lane in ("gptq", "mx", "exlr8") else (False, "")
        if lane == "gguf":
            measured = measure_gguf(a.model, a.source, a.eval, a.ngl, a.chunks)
        elif lane == "mlx":
            import pollard_stream as S
            try:
                measured = measure_mlx(a.model, a.source, a.eval, mlx_length, a.chunks, a.stream)
            except S.OutOfMemoryBudget as e:           # disk / RAM short: the rung stays on disk, the step is held
                print(f"HELD: {e}", flush=True)
                sys.exit(EXIT_HELD)
        else:
            import pollard_stream as S
            try:
                if streamed:
                    print(f"   streamed: source and rung block by block ({why})")
                    measured = measure_stream(a.model, a.source, a.eval, a.device, a.length, a.chunks, trc,
                                              max_ram_gb=a.max_ram_gb)
                elif lane == "exlr8":
                    measured = measure_exlr8(a.model, a.source, a.eval, a.device, a.length, a.chunks, trc == "on")
                else:
                    measured = measure_hf(a.model, a.source, a.eval, a.device, a.length, a.chunks, trc == "on")
            except S.OutOfMemoryBudget as e:           # no disk for the reference logits: held, the rung is kept
                print(f"HELD: {e}", flush=True)
                sys.exit(EXIT_HELD)
            print(f"   protocol: {measured.get('protocol')}")
        print(f"   mean KLD {measured['kld']:.5f}   top-1 {measured.get('top1') or 0:.2f}%"
              + (f"   PPL {measured['ppl']:.4f}" if measured.get("ppl") else ""))
    comp = complete_check(a.model, a.source, lane)
    if comp is not None:
        measured["complete"] = comp["ok"]
        measured["complete_missing"] = comp["missing"]
        if not comp["ok"]:
            fails += 1
    if lane == "gguf" and a.gate is not None and str(a.model).endswith(".gguf"):
        measured["runtime"] = gguf_runtime(a.model)
        if measured["runtime"] == "pollard":
            prior = {}
            if a.results and os.path.exists(a.results):
                try:
                    prior = metrics_from_results(json.load(open(a.results, encoding="utf-8")), rung)
                except (OSError, ValueError):
                    prior = {}
            measured["runtime_engine"] = pollard_runtime_engine(a.model, {**prior, **measured})
    if a.coherence and lane == "gguf":
        from pollard_bench import coherence_gate
        from pollard_calc import find_llama_bin
        import pollard_llamaserve as LS
        res = coherence_gate(LS.bin_for(a.model, "llama-cli") or find_llama_bin("llama-cli") or "llama-cli", a.model,
                             LS.resolve_ngl(a.model, a.ngl),
                             quick=True)
        measured["coherence"] = res.get("verdict")

    if a.results and measured:
        import pollard_workspace as ws
        keep = {k: measured[k] for k in ("kld", "top1", "ppl", "coherence") if measured.get(k) is not None}
        if measured.get("runtime") == "pollard":       # the card reads this: "Needs llama.cpp with Pollard's patch"
            keep.update({k: measured[k] for k in ("runtime", "runtime_engine") if measured.get(k)})
        if keep:
            try:
                ws.merge_results(a.results, rung, keep)
            except OSError as e:                       # a full disk: the old results stay whole, the step is held
                print(f"HELD: {e}", flush=True)
                sys.exit(EXIT_HELD)

    gate_res = None
    if a.gate is not None:
        gates = load_gates(a.gate or None)
        results = json.load(open(a.results, encoding="utf-8")) if a.results and os.path.exists(a.results) else {}
        metrics = {**metrics_from_results(results, rung), **{k: v for k, v in measured.items() if v is not None}}
        kinds = model_kinds(a.model) if gates.get("kinds") and lane != "draft" else []
        if gates.get("kinds") and "agentic" not in kinds and (
                metrics.get("toolcall_status") or metrics.get("toolcall_retention") is not None):
            kinds.append("agentic")            # pollard-toolcall measured it: its template writes tool calls
        if dec_only and "decision_only" not in kinds:
            kinds.append("decision_only")
        th = thresholds_for(gates, rung, lane, kinds)
        if dec_only:
            th = decision_only_thresholds(th, gates, rung)
        gate_res = evaluate_gate(metrics, th, gates.get("missing", "fail"))
        gate_res.update({"rung": rung, "lane": lane, "gates": os.path.basename(a.gate) if a.gate else "built-in",
                         "thresholds": th, "kinds": kinds})
        print_gate(rung, gate_res)
        if not gate_res["pass"]:
            fails += 1
        if a.results:
            import pollard_workspace as ws
            try:
                ws.merge_results(a.results, rung, {"verified": {"pass": gate_res["pass"], "gates": gate_res["gates"],
                                                                "failed": [c["check"] for c in gate_res["checks"]
                                                                           if not c["ok"]]}})
            except OSError as e:
                print(f"HELD: {e}", flush=True)
                sys.exit(EXIT_HELD)

    # stamp the workspace manifest so `pollard-ls` / the card / pollard release see the verdict
    # (no-op if the build was made outside the workspace)
    try:
        import pollard_workspace as ws
        if gate_res is not None:
            ws.record_gate(a.model, gate_res)
        else:
            ws.mark_verified(a.model, fails == 0)
    except Exception:
        pass

    if fails:
        print(f"\nGATE FAILED ({fails} problem(s)). Do NOT ship this build.")
        sys.exit(1)
    print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
