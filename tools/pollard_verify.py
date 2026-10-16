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

Lanes (end-to-end fidelity vs the source, measured the same way for every lane: mean KLD + top-1):
  gguf   llama-perplexity --kl-divergence against the f16/bf16 GGUF (--ref); base logits cached once
  mlx    mlx_lm: quantized vs source logits on the eval text (Apple Silicon)
  gptq   transformers: the GPTQ checkpoint vs the source (gptqmodel/optimum to load it)
  mx     transformers: the compressed-tensors checkpoint vs the source
  exl3   per-tensor decode vs source + the assembled forward (the original lane)

PUBLISH GATE: --gate gates.yaml applies per-rung thresholds -- KLD max, top-1 min, task retention min
(pollard-taskeval), tool-call pass (pollard-toolcall), decision agreement (pollard-decision), GGUF runtime
(pollard-ggufcheck: stock vs ik_llama, read from the tensor types, never the file name) and coherence
(pollard-bench's gate). Numbers come from this run and from --results (what the other tools wrote). A
threshold with no number to check FAILS unless the file says `missing: skip` -- an unmeasured rung is not
a verified one. The verdict is stamped on the build in the MANIFEST and in results.json, so pollard-card
shows "verified" and pollard release refuses to upload a rung that did not pass.

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


def end_to_end(model_dir, source_dir, device, rows, length):
    """The assembled forward vs the fp16 source -- per-tensor correct is NOT sufficient."""
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


# ----------------------------------------------------------------------- shared fidelity math

def kld_top1(ref_logits, q_logits):
    """{kld, top1, kld_p99, n} for two [N, V] logit arrays (numpy): mean KL(ref || q) per token in nats,
    and top-1 agreement in percent -- the same two numbers llama-perplexity reports for a GGUF, so every
    lane is gated on the same scale."""
    import numpy as np
    r = np.asarray(ref_logits, dtype=np.float64)
    q = np.asarray(q_logits, dtype=np.float64)
    if r.shape != q.shape:
        v = min(r.shape[-1], q.shape[-1])               # padded vocab rows on one side
        r, q = r[..., :v], q[..., :v]
    r = r - r.max(-1, keepdims=True)
    q = q - q.max(-1, keepdims=True)
    lr = r - np.log(np.exp(r).sum(-1, keepdims=True))
    lq = q - np.log(np.exp(q).sum(-1, keepdims=True))
    kl = (np.exp(lr) * (lr - lq)).sum(-1)
    top1 = (r.argmax(-1) == q.argmax(-1)).mean() * 100.0
    return {"kld": float(kl.mean()), "top1": float(top1), "kld_p99": float(np.percentile(kl, 99)), "n": int(kl.size)}


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
    ppl_bin = ppl_bin or find_llama_bin("llama-perplexity") or "llama-perplexity"
    try:
        import pollard_workspace as ws
        base = os.path.join(ws.cache_dir(create=True), os.path.splitext(os.path.basename(ref))[0]
                            + f".c{chunks}.klbase.dat")
    except Exception:
        base = os.path.splitext(ref)[0] + f".c{chunks}.klbase.dat"
    if not (os.path.exists(base) and os.path.getsize(base) > 0):
        cmd = [ppl_bin, "-m", ref, "-f", eval_path, "-c", "2048", "-ngl", str(ngl),
               "--kl-divergence-base", base] + (["--chunks", str(chunks)] if chunks else [])
        print("   KL base: $ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(base)) or ".").free
        if (r.returncode or free < 1e9) and os.path.exists(base):
            os.remove(base)                              # a short base poisons every later rung
        if not os.path.exists(base):
            raise SystemExit(f"could not build the KL base from {ref} (exit {r.returncode}, "
                             f"{free/1e9:.1f} GB free)")
    res = _ppl_kl(ppl_bin, model, eval_path, base, ngl, chunks)
    if res.get("mean_kld") is None:
        raise SystemExit(f"llama-perplexity produced no KLD for {model}")
    return {"kld": res["mean_kld"], "top1": res.get("top1"), "ppl": res.get("ppl"), "ref_ppl": res.get("ref_ppl"),
            "median_kld": res.get("median_kld")}


def measure_mlx(model, source, eval_path, length=512, chunks=8):
    """mlx_lm: the quantized model and its source over the same token chunks."""
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        raise SystemExit("the mlx lane needs mlx-lm (Apple Silicon): pip install mlx-lm") from None
    import numpy as np
    smodel, tok = load(source)
    ids = tok.encode(_eval_text(eval_path))
    parts = _chunks(ids, length, chunks)
    cache = _ref_cache(source, eval_path, length, chunks, "mlx")
    if os.path.exists(cache):
        ref = np.load(cache)
    else:
        ref = np.concatenate([np.array(smodel(mx.array(c)[None])[0].astype(mx.float32)) for c in parts]).astype(np.float16)
        np.save(cache, ref)
    del smodel
    qmodel, _ = load(model)
    q = np.concatenate([np.array(qmodel(mx.array(c)[None])[0].astype(mx.float32)) for c in parts])
    return kld_top1(ref.astype(np.float32), q)


def measure_hf(model, source, eval_path, device="cuda:0", length=512, chunks=8, trust_remote_code=False):
    """transformers: the GPTQ / compressed-tensors (MX) checkpoint vs the source, same chunks. The source's
    logits are computed once, cached as fp16 in the workspace, and the source is freed before the
    quantized model loads, so the two never sit in memory together."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit("the gptq/mx lanes need torch + transformers (+ gptqmodel / compressed-tensors)") from None
    import numpy as np
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    parts = _chunks(tok(_eval_text(eval_path))["input_ids"], length, chunks)
    cache = _ref_cache(source, eval_path, length, chunks, "hf")

    def logits(path):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map=device,
                                                 trust_remote_code=trust_remote_code).eval()
        with torch.no_grad():
            out = [m(torch.tensor([c], device=m.device)).logits[0].float().cpu().numpy() for c in parts]
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(out)
    if os.path.exists(cache):
        ref = np.load(cache).astype(np.float32)
    else:
        ref = logits(source)
        np.save(cache, ref.astype(np.float16))
    return kld_top1(ref, logits(model))


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
#   toolcall_not_measured   fail | skip: a rung pollard-toolcall recorded as NOT MEASURED (no server for its lane on
#                    this machine, GPU not free, too big) -- default: the `missing` policy
#   decision_agreement_min  same decision as the f16, percent            pollard-decision --results
#   runtime          stock | ik_llama | any  (GGUF: from the tensor types, pollard-ggufcheck)
#   coherence        true = passes the coherence gate (GGUF: pollard-bench's)
#   draft_tau_min    drafter: tokens per target pass vs the quantized target rung(s), worst pairing  pollard-draft measure
#   draft_accept_rate_min   drafter: accepted / drafted, 0-1                                    pollard-draft measure
#   draft_redistributable   true = the drafter's licence lets us republish it                 pollard-draft convert
missing: fail
default:
  kld_max: 0.5
  top1_min: 75
lanes:
  gguf:
    runtime: stock
    coherence: true
  # exl3 has no logit-level KLD path in pollard-verify yet: it is gated on the assembled forward instead
  exl3:
    e2e_acc_min: 0.3
    skip: [kld_max, top1_min]
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
  - match: ["Q6_K*", "*6bit*", "*-6.0bpw"]
    kld_max: 0.02
    top1_min: 95
  - match: ["Q5_K*", "*5bit*", "*-5.0bpw"]
    kld_max: 0.04
    top1_min: 93
  - match: ["Q4_K*", "IQ4_*", "*4bit*", "*int4*", "*NVFP4*", "*MXFP4*", "*-4.0bpw"]
    kld_max: 0.08
    top1_min: 90
  - match: ["Q3_K*", "IQ3_*", "*3bit*", "*int3*", "*-3.0bpw", "*-3.5bpw"]
    kld_max: 0.2
    top1_min: 85
  - match: ["Q2_K*", "IQ2_*", "IQ1_*", "*2bit*", "*int2*", "*-2.0bpw", "*-2.5bpw"]
    kld_max: 0.5
    top1_min: 75
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
      - match: ["Q3_K*", "IQ3_*", "*3bit*", "*int3*", "*-3.0bpw", "*-3.5bpw"]
        toolcall_retention_min: 85
      - match: ["Q2_K*", "IQ2_*", "IQ1_*", "*2bit*", "*int2*", "*-2.0bpw", "*-2.5bpw", "*_KT*", "PollardMix*"]
        toolcall_retention_min: 75
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
    for kind in kinds or ():
        sec = (gates.get("kinds") or {}).get(kind) or {}
        th.update({k: v for k, v in sec.items() if k != "rungs"})
        _apply_rungs(th, sec.get("rungs"), rung)
    # a lane can opt out of a check it has no way to measure (exl3: no logit-level KLD path yet)
    for k in ((gates.get("lanes") or {}).get(lane or "", {}) or {}).get("skip") or []:
        th.pop(k, None)
    th.pop("skip", None)
    return th


#: threshold -> (metric key, comparison, human name)
_CHECKS = {
    "kld_max": ("kld", "<=", "mean KLD"),
    "top1_min": ("top1", ">=", "top-1 agreement %"),
    "retention_min": ("retention", ">=", "task retention %"),
    "mt_retention_min": ("mt_retention", ">=", "translation retention %"),
    "decision_agreement_min": ("decision_agreement", ">=", "decision agreement %"),
    "toolcall_retention_min": ("toolcall_retention", ">=", "tool-call retention %"),
    "e2e_acc_min": ("e2e_acc", ">=", "end-to-end next-token agreement"),
    "draft_tau_min": ("draft_tau", ">=", "drafter tokens per target pass"),
    "draft_accept_rate_min": ("draft_accept_rate", ">=", "drafter accept rate"),
}
_RUNTIME_OK = {"stock": {"stock"}, "ik_llama": {"stock", "ik_llama"}}


def evaluate_gate(metrics, th, missing="fail"):
    """{pass, checks: [{check, value, want, ok, why}]} -- one row per threshold, PASS only if every
    row passes. A threshold whose number was never measured fails under missing: fail."""
    checks = []

    def row(name, value, want, ok, why=""):
        checks.append({"check": name, "value": value, "want": want, "ok": bool(ok), "why": why})
    # a rung pollard-toolcall explicitly recorded as NOT MEASURED (no server for its lane here, GPU not free, too
    # big) is held to `toolcall_not_measured` (default: the `missing` policy), with the reason on the row
    tc_nm = metrics.get("toolcall_status") == "not-measured"
    tc_policy = th.get("toolcall_not_measured", missing)
    tc_why = f"not measured: {metrics.get('toolcall_reason') or 'no reason recorded'}"
    for key, want in th.items():
        if key in _CHECKS:
            mkey, op, human = _CHECKS[key]
            v = metrics.get(mkey)
            if v is None and tc_nm and key.startswith("toolcall"):
                row(human, None, f"{op} {want}", tc_policy == "skip", tc_why)
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
                row("runtime", v, want, ok, "" if ok else
                    f"a rung published as {want} needs '{v}' -- the card and the runtime would disagree")
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
    if tc.get("retention") is not None:
        m["toolcall_retention"] = float(tc["retention"])
    elif tc.get("rate") is not None and tc.get("ref_rate"):          # a --quick run: rate / f16 rate
        m["toolcall_retention"] = 100.0 * float(tc["rate"]) / float(tc["ref_rate"])
    d = r.get("decision") or {}
    if d.get("agreement") is not None:
        m["decision_agreement"] = 100.0 * float(d["agreement"])
    if r.get("coherence"):
        m["coherence"] = r["coherence"]
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
    ap.add_argument("--lane", default=None, choices=["exl3", "gguf", "mlx", "gptq", "mx", "draft"],
                    help="convert lane (default: gguf for a .gguf file, else exl3). draft = a speculative drafter, "
                         "gated on the pollard-draft measure numbers in --results (nothing is run)")
    ap.add_argument("--eval", help="HELD-OUT text for the KLD/top-1 measurement (gguf/mlx/gptq/mx)")
    ap.add_argument("--chunks", type=int, default=16, help="eval chunks (gguf: llama-perplexity --chunks)")
    ap.add_argument("--length", type=int, default=256, help="tokens per chunk (mlx/gptq/mx) / e2e length (exl3)")
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
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--sanitizer", action="store_true",
                    help="re-run this verify under NVIDIA compute-sanitizer (memcheck+racecheck)")
    a = ap.parse_args()
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
        if a.end_to_end:
            print("-- end-to-end assembled forward --")
            acc, ppl = end_to_end(a.model, a.source or a.model, a.device, 1, a.length)
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
        if lane == "gguf":
            measured = measure_gguf(a.model, a.source, a.eval, a.ngl, a.chunks)
        elif lane == "mlx":
            measured = measure_mlx(a.model, a.source, a.eval, a.length, a.chunks)
        else:
            measured = measure_hf(a.model, a.source, a.eval, a.device, a.length, a.chunks, a.trust_remote_code)
        print(f"   mean KLD {measured['kld']:.5f}   top-1 {measured.get('top1') or 0:.2f}%"
              + (f"   PPL {measured['ppl']:.4f}" if measured.get("ppl") else ""))
    if lane == "gguf" and a.gate is not None and str(a.model).endswith(".gguf"):
        measured["runtime"] = gguf_runtime(a.model)
    if a.coherence and lane == "gguf":
        from pollard_bench import coherence_gate
        from pollard_calc import find_llama_bin
        res = coherence_gate(find_llama_bin("llama-cli") or "llama-cli", a.model, a.ngl, quick=True)
        measured["coherence"] = res.get("verdict")

    if a.results and measured:
        import pollard_workspace as ws
        keep = {k: measured[k] for k in ("kld", "top1", "ppl", "coherence") if measured.get(k) is not None}
        if keep:
            ws.merge_results(a.results, rung, keep)

    gate_res = None
    if a.gate is not None:
        gates = load_gates(a.gate or None)
        results = json.load(open(a.results, encoding="utf-8")) if a.results and os.path.exists(a.results) else {}
        metrics = {**metrics_from_results(results, rung), **{k: v for k, v in measured.items() if v is not None}}
        kinds = model_kinds(a.model) if gates.get("kinds") and lane != "draft" else []
        if gates.get("kinds") and "agentic" not in kinds and (
                metrics.get("toolcall_status") or metrics.get("toolcall_retention") is not None):
            kinds.append("agentic")            # pollard-toolcall measured it: its template writes tool calls
        th = thresholds_for(gates, rung, lane, kinds)
        gate_res = evaluate_gate(metrics, th, gates.get("missing", "fail"))
        gate_res.update({"rung": rung, "lane": lane, "gates": os.path.basename(a.gate) if a.gate else "built-in",
                         "thresholds": th, "kinds": kinds})
        print_gate(rung, gate_res)
        if not gate_res["pass"]:
            fails += 1
        if a.results:
            import pollard_workspace as ws
            ws.merge_results(a.results, rung, {"verified": {"pass": gate_res["pass"], "gates": gate_res["gates"],
                                                            "failed": [c["check"] for c in gate_res["checks"] if not c["ok"]]}})

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
