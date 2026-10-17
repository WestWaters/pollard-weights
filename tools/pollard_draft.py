#!/usr/bin/env python3
"""pollard-draft -- ship a speculative-decoding drafter matched to the QUANTIZED target, in every runtime.

A drafter proposes a few tokens; the target checks them all in one pass and keeps the prefix it agrees with. The
output is the target's own (greedy: identical except where batched arithmetic flips a near-tie), and every accepted
token is a target pass saved. So the number that matters is how many tokens one target pass yields with the drafter
attached -- measured against the rung people will actually run, not the bf16 original.

    pollard-draft find    --target Qwen/Qwen3-4B                          # which published drafter fits, and why
    pollard-draft convert --target Qwen/Qwen3-4B --rungs Q8_0,Q4_K_M      # GGUF ladder + the HF folder (vLLM/MLX)
    pollard-draft measure --target Qwen3-4B-Pollard-Q6_K.gguf --target-tag Q6_K \\
                          --drafter Qwen3-4B-Pollard-DRAFT-DFlash-Q4_K_M.gguf --rung DFlash-Q4_K_M \\
                          --results results-draft.json                    # tokens per target pass, per prompt kind
    pollard-draft card    --target Qwen/Qwen3-4B --results results-draft.json --repo PollardWeights/Qwen3-4B-Pollard-DRAFT
    pollard-draft train   --target Qwen/Qwen3-4B --rung Qwen3-4B-Pollard-Q4_K_M.gguf --rung-tag Q4_K_M --run
                                                                          # Pollard's own QUANT-AWARE head (CUDA box)
    pollard-draft serve   --target Qwen/Qwen3-4B --method dflash          # the serve line per runtime

Drafter kinds (what each runtime runs; llama.cpp from b9606 on, vLLM, SGLang, MLX via dflash-mlx / mlx-lm):

  dflash  block-parallel drafter (z-lab DFlash / DFlash2, MIT): ~1B, 5 layers, reads 5 target layers' hidden states.
          llama.cpp --spec-type draft-dflash | vLLM method dflash | SGLang DFLASH | MLX: dflash-mlx
  dspark  DFlash + Markov/confidence heads (DeepSeek DSpark). llama.cpp draft-dspark | vLLM dspark | MLX: mlx-dspark
  eagle3  one-layer head on 3 target layers (EAGLE-3: yuhuili, RedHatAI speculators, AngelSlim, lmsys SpecForge).
          llama.cpp draft-eagle3 | vLLM method eagle3 | SGLang EAGLE3 | MLX: no runtime serves it
  simple  a small model of the same family and tokenizer (Qwen3-0.6B for Qwen3-8B) -- runs everywhere.
          llama.cpp -md (draft-simple) | vLLM method draft_model | MLX: mlx_lm --draft-model

Layout of the drafter repo (<M>-Pollard-DRAFT, the PollardWeights lane naming): the drafter in its native HF format
at the ROOT -- so `"model": "<org>/<M>-Pollard-DRAFT"` works as-is in vLLM's --speculative-config, SGLang and
dflash-mlx -- and its GGUF ladder beside it, <M>-Pollard-DRAFT-<Kind>-<QUANT>.gguf, for llama.cpp's -md / -hfd.

The numbers (results-draft.json, one entry per drafter rung, per target rung):
  tau          tokens per target pass = generated / (generated - accepted); 1.0 = the drafter never helps
  accept_len   accepted draft tokens per target pass (= tau - 1)
  accept_rate  accepted / drafted
llama.cpp reports the counters itself (timings.draft_n / draft_n_accepted); vLLM's /metrics spec_decode counters;
dflash-mlx's /metrics tokens_per_cycle. Nothing is estimated: a backend this machine cannot run is recorded as
NOT MEASURED with the reason, and the gate (pollard-verify --lane draft) holds that rung back.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HF = "https://huggingface.co"

#: kind -> how each runtime names it, and the draft width to start from. dflash 7: on an M4 the verify pass is flat
#: to 8 rows and jumps 1.6x at 9 (Graft fastest-format.md section 7: 7 beat 15 for every target); on a big GPU
#: raise it (vLLM auto-fills it from the drafter's config when omitted).
METHODS = {
    "dflash": {"label": "DFlash", "spec_type": "draft-dflash", "vllm": "dflash", "sglang": "DFLASH",
               "mlx": "dflash-mlx", "n_max": 7, "needs_target_dir": True},
    "dspark": {"label": "DSpark", "spec_type": "draft-dspark", "vllm": "dspark", "sglang": None,
               "mlx": "mlx-dspark", "n_max": 6, "needs_target_dir": True},
    "eagle3": {"label": "EAGLE3", "spec_type": "draft-eagle3", "vllm": "eagle3", "sglang": "EAGLE3",
               "mlx": None, "n_max": 3, "needs_target_dir": True},
    "simple": {"label": "Draft", "spec_type": "draft-simple", "vllm": "draft_model", "sglang": "STANDALONE",
               "mlx": "mlx_lm", "n_max": 8, "needs_target_dir": False},
}
DFLASH_ARCHS = {"DFlashDraftModel", "DFlash2DraftModel"}
EAGLE3_ARCHS = {"LlamaForCausalLMEagle3", "Eagle3LlamaForCausalLM", "Eagle3Speculator", "Eagle3DraftModel"}
#: GGUF general.architecture of a converted drafter -> kind (llama.cpp's LLM_ARCH_DFLASH / LLM_ARCH_EAGLE3; a DSpark
#: drafter converts to the dflash arch and is told apart by its Markov tensors, so the kind is passed explicitly)
GGUF_ARCH_KIND = {"dflash": "dflash", "eagle3": "eagle3"}
#: licences that let Pollard redistribute a third party's drafter (re-quantized) under its own terms
REDISTRIBUTABLE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "unlicense"}
#: who publishes drafters people actually serve -- a tie-breaker, never a reason to accept an incompatible one
KNOWN_ORGS = ("z-lab", "RedHatAI", "deepseek-ai", "yuhuili", "lmsys", "AngelSlim", "nvidia", "Tengyunw",
              "openbmb", "LiquidAI", "RadixArk")
_DRAFT_WORDS = re.compile(r"dflash|dspark|eagle|speculator|draft|mtp", re.I)
#: copies of a drafter in someone else's runtime format -- we convert from the source checkpoint, not from these
_DERIVED = re.compile(r"gguf|[-_]ov$|-int[48]|nvfp4|mlx|-awq|-gptq|w4a16|quantized", re.I)
DEFAULT_RUNGS = ["Q8_0", "Q4_K_M"]


class NotMeasured(Exception):
    pass


# ----------------------------------------------------------------------------- hub access (stdlib; faked in tests)

def hf_json(path, timeout=20):
    """GET <hub>/api/... or a resolve/main/config.json -- stdlib only, so `find` runs in a bare install."""
    url = path if path.startswith("http") else HF + path
    req = urllib.request.Request(url, headers={"User-Agent": "pollard-draft"})
    tok = os.environ.get("HF_TOKEN") or _hf_token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _hf_token():
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:                                            # noqa: BLE001 -- no hub client: anonymous
        return None


def _get(path, fetch=None):
    try:
        return (fetch or hf_json)(path)
    except Exception:                                            # noqa: BLE001 -- missing file / offline
        return None


def repo_config(repo, fetch=None):
    if os.path.isdir(repo):
        p = os.path.join(repo, "config.json")
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            return None
    return _get(f"{HF}/{repo}/resolve/main/config.json", fetch)


def repo_info(repo, fetch=None):
    return _get(f"/api/models/{repo}", fetch) or {}


def search(query, fetch=None, limit=100, base_model=None):
    q = {"search": query, "limit": str(limit), "sort": "downloads"}
    if base_model:
        q["filter"] = f"base_model:{base_model}"
    return _get("/api/models?" + urllib.parse.urlencode(q), fetch) or []


# ----------------------------------------------------------------------------- what a checkpoint is

def text_cfg(cfg):
    cfg = cfg or {}
    return cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) and not cfg.get("hidden_size") else cfg


def target_shape(cfg):
    t = text_cfg(cfg)
    return {"hidden_size": t.get("hidden_size"), "num_hidden_layers": t.get("num_hidden_layers"),
            "vocab_size": t.get("vocab_size"), "model_type": t.get("model_type"),
            "architectures": list((cfg or {}).get("architectures") or [])}


def classify(cfg):
    """The drafter kind a config.json declares, or None for an ordinary causal LM (a `simple` candidate)."""
    cfg = cfg or {}
    archs = set(cfg.get("architectures") or [])
    spec = str(cfg.get("speculators_model_type") or (cfg.get("speculators_config") or {}).get("algorithm") or "")
    if archs & DFLASH_ARCHS or spec.startswith("dflash") or "dflash_config" in cfg:
        return "dflash"
    if any("dspark" in a.lower() for a in archs) or spec == "dspark":
        return "dspark"
    if archs & EAGLE3_ARCHS or spec == "eagle3" or any("eagle3" in a.lower() for a in archs):
        return "eagle3"
    # yuhuili/EAGLE3-* declare a plain LlamaForCausalLM: one layer + a reduced draft vocab is the tell
    if cfg.get("draft_vocab_size") and int(cfg.get("num_hidden_layers") or 0) == 1:
        return "eagle3"
    return None


def _layer_cfg(cfg):
    return cfg.get("transformer_layer_config") if isinstance(cfg.get("transformer_layer_config"), dict) else cfg


def compat(dcfg, tcfg, kind):
    """(ok, problems, notes): can this drafter run against this target? Checked on the numbers the runtime needs --
    the hidden size it reads, the vocabulary it predicts in, the target layers it taps."""
    dcfg, t = dcfg or {}, target_shape(tcfg)
    problems, notes = [], []
    lc = _layer_cfg(dcfg)
    dv = lc.get("vocab_size") or dcfg.get("vocab_size")
    if t["vocab_size"] and dv and int(dv) != int(t["vocab_size"]):
        problems.append(f"vocab {dv} != target {t['vocab_size']}")
    th, tl = t["hidden_size"], t["num_hidden_layers"]
    if kind in ("dflash", "dspark"):
        if th and lc.get("hidden_size") and int(lc["hidden_size"]) != int(th):
            problems.append(f"hidden {lc['hidden_size']} != target {th}")
        ntl = dcfg.get("num_target_layers")
        if tl and ntl and int(ntl) != int(tl):
            problems.append(f"trained on a {ntl}-layer target, this one has {tl}")
        ids = (dcfg.get("dflash_config") or {}).get("target_layer_ids") or dcfg.get("target_layer_ids") or []
        if tl and ids and max(ids) >= int(tl):
            problems.append(f"taps layer {max(ids)}, target has {tl}")
    elif kind == "eagle3":
        want = dcfg.get("target_hidden_size") or lc.get("hidden_size")
        if th and want and int(want) != int(th):
            problems.append(f"reads hidden {want}, target has {th}")
        ids = dcfg.get("eagle_aux_hidden_state_layer_ids") or []
        if tl and ids and max(ids) >= int(tl):
            problems.append(f"taps layer {max(ids)}, target has {tl}")
        if dcfg.get("draft_vocab_size"):
            notes.append(f"draft vocab {dcfg['draft_vocab_size']} (d2t map to the full vocab)")
    elif kind == "simple":
        if t["model_type"] and dcfg.get("model_type") and dcfg["model_type"] != t["model_type"]:
            problems.append(f"model_type {dcfg['model_type']} != target {t['model_type']}")
        dl, dh = dcfg.get("num_hidden_layers") or 0, dcfg.get("hidden_size") or 0
        if th and tl and dl and dh and dl * dh * dh >= tl * th * th:
            problems.append("not smaller than the target")
    ver = ((dcfg.get("speculators_config") or {}).get("verifier") or {}).get("name_or_path")
    if ver:
        notes.append(f"trained against {ver}")
    return not problems, problems, notes


#: drafter archs llama.cpp's converter registers (b9606+; pinned Graft tree f2cc7282c, 2026-10-11)
GGUF_CONVERTIBLE = DFLASH_ARCHS | EAGLE3_ARCHS | {"Qwen3DSparkModel", "DSparkDraftModel", "DSparkSpeculator",
                                                  "Lfm2DSparkDraftModel", "LingDSparkModel", "Gemma4DSparkModel"}


def gguf_convertible(cfg, kind, converter=None):
    """Can llama.cpp's converter turn this drafter into a GGUF? True / False / None (unknown). A speculators-format
    DFlash checkpoint (nested transformer_layer_config) is not one the DFlash converter flattens -- DSpark's does."""
    cfg = cfg or {}
    archs = list(cfg.get("architectures") or [])
    if kind == "simple":
        return True if not converter else any(_registers(converter, x) for x in archs)
    if kind == "dflash" and "transformer_layer_config" in cfg:
        return False
    if kind == "eagle3" and not set(archs) & EAGLE3_ARCHS:
        # yuhuili-style: a plain LlamaForCausalLM with a draft vocab, which the llama converter detects
        return "LlamaForCausalLM" in archs or None
    if converter:
        return any(_registers(converter, x) for x in archs)
    return bool(set(archs) & GGUF_CONVERTIBLE)


def _registers(converter, arch):
    try:
        from pollard_convert import converter_registers
        return converter_registers(converter, arch)
    except Exception:                                            # noqa: BLE001
        return False


def _basename(model):
    return str(model).rstrip("/\\").split("/")[-1]


def name_score(repo, target):
    """2 = the drafter is named for exactly this model, 1 = it mentions it (maybe a variant), 0 = neither."""
    base = _basename(target).lower()
    name = _basename(repo).lower()
    if base not in name:
        return 0
    rest = name.replace(base, " ")
    rest = re.sub(r"speculator|eagle3?|dflash2?|dspark|draft|[-_.]b\d+|block\d+|\bv\d+\b|[-_.]", " ", rest)
    return 2 if not rest.strip() else 1


def licence_of(info):
    lic = ((info or {}).get("cardData") or {}).get("license")
    if not lic:
        for t in (info or {}).get("tags") or []:
            if str(t).startswith("license:"):
                lic = t.split(":", 1)[1]
    return str(lic).lower() if lic else None


# ----------------------------------------------------------------------------- find

def candidates_for(target, fetch=None):
    """Hub repos that look like drafters for `target`: searched by its name and by base_model tag."""
    base = _basename(target)
    seen, out = set(), []
    hits = list(search(base, fetch))
    for kw in ("dflash", "eagle", "speculator", "dspark"):     # drafters are rarely the most-downloaded '<base>' hits
        for q in (f"{base}-{kw}", f"{base}_{kw}", f"{kw}-{base}"):
            hits += search(q, fetch, limit=30)
    if "/" in str(target) and not os.path.exists(str(target)):
        for kw in ("dflash", "eagle", "speculator", "dspark"):
            hits += search(kw, fetch, limit=50, base_model=target)
    for m in hits:
        rid = m.get("id") or m.get("modelId")
        if not rid or rid in seen or not _DRAFT_WORDS.search(rid) or "mtp" in rid.lower():
            continue
        seen.add(rid)
        out.append({"id": rid, "downloads": m.get("downloads") or 0})
    return out


def simple_candidates(target, tcfg, fetch=None):
    """A same-family smaller model (Qwen3-0.6B for Qwen3-8B) -- the drafter every runtime can run."""
    if "/" not in str(target) or os.path.exists(str(target)):
        return []
    org, base = str(target).split("/", 1)
    m = re.match(r"^(.*?)(\d+(?:\.\d+)?)B(.*)$", base, re.I)
    if not m:
        return []
    fam, size, suffix = m.group(1), float(m.group(2)), m.group(3)
    out = []
    for hit in search(f"{org}/{fam}", fetch, limit=100):
        rid = hit.get("id") or ""
        if not rid.startswith(org + "/") or _DERIVED.search(rid) or _DRAFT_WORDS.search(rid):
            continue
        mm = re.match(rf"^{re.escape(fam)}(\d+(?:\.\d+)?)B{re.escape(suffix)}$", rid.split("/", 1)[1], re.I)
        if mm and float(mm.group(1)) < size / 3:
            out.append({"id": rid, "downloads": hit.get("downloads") or 0, "size_b": float(mm.group(1))})
    return sorted(out, key=lambda x: -x["size_b"])                 # the biggest that is still < 1/3 drafts best


def assess(rid, target, tcfg, fetch=None, kind=None, downloads=0, converter=None):
    cfg = repo_config(rid, fetch)
    if cfg is None:
        return {"id": rid, "ok": False, "problems": ["no config.json (a converted copy, or private)"], "kind": kind}
    k = kind or classify(cfg)
    if k is None:
        return {"id": rid, "ok": False, "problems": ["not a drafter checkpoint this tool knows"], "kind": None}
    ok, problems, notes = compat(cfg, tcfg, k)
    info = repo_info(rid, fetch) if not os.path.isdir(rid) else {}
    lic = licence_of(info)
    files = [s.get("rfilename", "") for s in info.get("siblings") or []]
    if files and not any(f.endswith((".safetensors", ".bin")) for f in files):
        ok, problems = False, problems + ["no weights in the repo"]
    if any("mxq" in str(x) for x in cfg) or cfg.get("base_mxq_path"):
        ok, problems = False, problems + ["an NPU (mxq) export, not a runtime checkpoint"]
    return {"id": rid, "kind": k, "ok": ok, "problems": problems, "notes": notes, "license": lic,
            "gguf": gguf_convertible(cfg, k, converter),
            "redistributable": (lic in REDISTRIBUTABLE) if lic else False,
            "downloads": info.get("downloads", downloads) or 0,
            "name_score": name_score(rid, target), "derived": bool(_DERIVED.search(rid)),
            "org_known": rid.split("/")[0] in KNOWN_ORGS}


def rank_key(c):
    order = {"dflash": 3, "dspark": 2, "eagle3": 1, "simple": 0}   # tokens per pass, highest first (measured, Graft)
    return (c.get("ok", False), not c.get("derived"), c.get("name_score", 0), c.get("gguf") is True,
            order.get(c.get("kind"), -1),
            c.get("redistributable", False), c.get("org_known", False), c.get("downloads", 0))


def find(target, target_cfg=None, fetch=None, drafter=None, max_check=12, with_simple=True):
    """Every drafter candidate for `target`, checked against its config, best first. `drafter` pins one repo."""
    tcfg = target_cfg or repo_config(target, fetch) or {}
    try:
        from pollard_convert import find_converter
        converter = find_converter(None)[0]
    except Exception:                                            # noqa: BLE001
        converter = None
    if drafter:
        cands = [assess(drafter, target, tcfg, fetch, converter=converter)]
        if cands[0].get("kind") is None and cands[0].get("problems") == ["not a drafter checkpoint this tool knows"]:
            cands = [assess(drafter, target, tcfg, fetch, kind="simple", converter=converter)]
    else:
        raw = sorted(candidates_for(target, fetch),
                     key=lambda m: (name_score(m["id"], target), not _DERIVED.search(m["id"]), m["downloads"]),
                     reverse=True)
        cands = [assess(m["id"], target, tcfg, fetch, downloads=m["downloads"], converter=converter)
                 for m in raw[:max_check]]
        if with_simple:
            for s in simple_candidates(target, tcfg, fetch)[:2]:
                cands.append(assess(s["id"], target, tcfg, fetch, kind="simple", downloads=s["downloads"],
                                    converter=converter))
    cands.sort(key=rank_key, reverse=True)
    pick = next((c for c in cands if c.get("ok") and not c.get("derived")), None)
    unknown = not target_shape(tcfg).get("hidden_size")
    return {"target": str(target), "target_shape": target_shape(tcfg), "pick": pick, "candidates": cands,
            "warning": "the target's config.json could not be read (gated? set HF_TOKEN) -- shapes NOT checked"
            if unknown else None,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def print_find(res, out=print):
    out(f"== pollard-draft find :: {res['target']}  {res['target_shape']}")
    for c in res["candidates"]:
        mark = "OK " if c.get("ok") else "-- "
        extra = "; ".join((c.get("problems") or []) + (c.get("notes") or []))
        g = {True: "gguf", False: "no-gguf", None: "gguf?"}.get(c.get("gguf"), "")
        out(f"   {mark}{c['id']:58s} {str(c.get('kind')):7s} {g:7s} lic={c.get('license')}  "
            f"dl={c.get('downloads', 0)}" + (f"  ({extra})" if extra else ""))
    if res.get("warning"):
        out(f"   WARNING: {res['warning']}")
    p = res.get("pick")
    if p:
        red = "" if p.get("redistributable") else "  -- licence does not allow redistribution: card links it instead"
        out(f"   pick: {p['id']} ({p['kind']}){red}")
    else:
        out("   no compatible drafter published -- `pollard-draft train` has the recipe")


# ----------------------------------------------------------------------------- convert / quantize

def build_tag(kind, quant, simple_name=None):
    label = METHODS[kind]["label"] if kind != "simple" else (simple_name or "Draft")
    return f"{label}-{quant}"


def gguf_name(target, kind, quant, simple_name=None):
    import pollard_workspace as ws
    return f"{ws.model_basename(target)}-Pollard-DRAFT-{build_tag(kind, quant, simple_name)}.gguf"


def tokenizer_dir(target, fetch_dir=None):
    """The target's config + tokenizer (what a standalone drafter's GGUF needs) -- a local checkout as-is, else
    those files alone from the Hub into the workspace cache (KBs, not the weights)."""
    if os.path.isdir(str(target)):
        return str(target)
    import pollard_workspace as ws
    dest = fetch_dir or os.path.join(ws.cache_dir(create=True), ws.model_slug(target) + "-tokenizer")
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    snapshot_download(target, local_dir=dest, allow_patterns=["config.json", "generation_config.json", "tokenizer*",
                                                               "vocab*", "merges.txt", "special_tokens_map.json",
                                                               "chat_template*", "*.tiktoken"])
    return dest


def convert_cmds(drafter_dir, target_dir, kind, out_dir, target, rungs, python=None, converter=None,
                 quantize=None, simple_name=None):
    """The commands that make the GGUF ladder: convert to BF16 once (the converter needs the target's tokenizer for a
    standalone drafter), then llama-quantize each rung from it. Returns [(tag, path, cmd)], BF16 first."""
    bf16 = os.path.join(out_dir, gguf_name(target, kind, "BF16", simple_name))
    conv = [python or sys.executable, str(converter), str(drafter_dir), "--outtype", "bf16", "--outfile", bf16]
    if METHODS[kind]["needs_target_dir"]:
        conv += ["--target-model-dir", str(target_dir)]
    steps = [(build_tag(kind, "BF16", simple_name), bf16, conv)]
    for q in rungs:
        if q.upper() == "BF16":
            continue
        path = os.path.join(out_dir, gguf_name(target, kind, q, simple_name))
        steps.append((build_tag(kind, q, simple_name), path, [str(quantize), bf16, path, q]))
    return steps


_HF_KEEP = re.compile(r"\.(json|safetensors|py|txt|model|tiktoken|jinja)$|^LICENSE", re.I)


def stage_hf(drafter_dir, dest, kind, target=None, speculators_bin=None, run=subprocess.run):
    """The drafter in its HF format at `dest` -- what vLLM / SGLang / dflash-mlx load. An EAGLE-3 research checkpoint
    (yuhuili / AngelSlim layout) becomes the speculators format vLLM reads when `speculators` is installed; anything
    vLLM already reads (z-lab DFlash, a speculators checkpoint) is copied as published."""
    cfg = repo_config(drafter_dir) or {}
    if kind == "eagle3" and "speculators_config" not in cfg and speculators_bin and target:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        r = run([speculators_bin, "convert", str(drafter_dir), "--algorithm", "eagle3", "--verifier", str(target),
                 "--output-path", dest])
        if getattr(r, "returncode", 0) == 0:
            return dest, "speculators convert"
    os.makedirs(dest, exist_ok=True)
    for f in sorted(os.listdir(drafter_dir)):
        src = os.path.join(drafter_dir, f)
        if os.path.isfile(src) and _HF_KEEP.search(f):
            shutil.copy2(src, os.path.join(dest, f))
    return dest, "copied as published"


def record(key, path, tag, fmt, kind, meta):
    """A drafter rung in the model's MANIFEST, lane `draft`, with what it is and where it came from."""
    import pollard_workspace as ws
    ws.record_build(key, "draft", path, tag=tag)
    man = ws.read_manifest(key)
    for b in man.get("builds", []):
        if b.get("path") == os.path.abspath(path):
            b.update({"format": fmt, "kind": kind, **{k: v for k, v in meta.items() if v is not None}})
    ws._write_manifest(key, man)


def draft_builds(key):
    import pollard_workspace as ws
    return [b for b in ws.read_manifest(key).get("builds", []) if b.get("lane") == "draft"]


def _load_pick(path):
    res = json.load(open(path, encoding="utf-8"))
    p = res.get("pick")
    if not p:
        raise SystemExit(f"{path} names no compatible drafter -- nothing to convert (see `pollard-draft train`)")
    return p


def trained_builds(key):
    """Drafter rungs Pollard trained itself (pollard-draft train), from the MANIFEST."""
    return [b for b in draft_builds(key) if b.get("source") == "pollard-trained"]


def cmd_convert(a):
    import pollard_workspace as ws
    from pollard_calc import find_llama_bin
    from pollard_convert import find_converter
    if getattr(a, "skip_if_trained", False) and trained_builds(a.key or a.target):
        print(f"   Pollard trained its own head for {a.target} (pollard-draft train) -- nothing to convert")
        return 0
    pick = _load_pick(a.from_) if a.from_ else None
    drafter = a.drafter or (pick or {}).get("id")
    if not drafter:
        raise SystemExit("pass --drafter <repo-or-dir> or --from drafter.json (pollard-draft find --out)")
    ddir = ws.fetch_source(drafter)
    dcfg = repo_config(ddir) or {}
    kind = a.kind or (pick or {}).get("kind") or classify(dcfg) or "simple"
    simple_name = _basename(drafter) if kind == "simple" else None
    key = a.key or a.target
    out_dir = a.out_dir or os.path.join(ws.model_dir(key, create=True), "draft")
    os.makedirs(out_dir, exist_ok=True)
    meta = {"source": drafter, "license": (pick or {}).get("license") or licence_of(repo_info(drafter)),
            "spec_type": METHODS[kind]["spec_type"]}
    meta["redistributable"] = (meta["license"] in REDISTRIBUTABLE) if meta["license"] else False
    meta.update(getattr(a, "meta", None) or {})                # a Pollard-trained head: its own licence + provenance
    rungs = [r.strip() for r in a.rungs.split(",") if r.strip()]
    if not a.no_gguf:
        conv, why = find_converter(ddir)
        if not conv:
            raise SystemExit(f"cannot convert {drafter} to GGUF here: {why}")
        qbin = find_llama_bin("llama-quantize")
        if not qbin and any(r.upper() != "BF16" for r in rungs):
            raise SystemExit("llama-quantize not found (install llama.cpp, or POLLARD_HOME/bin)")
        tdir = tokenizer_dir(a.target_dir or a.target) if METHODS[kind]["needs_target_dir"] else None
        steps = convert_cmds(ddir, tdir, kind, out_dir, a.target, rungs, a.python, conv, qbin, simple_name)
        bf16 = steps[0][1]
        for tag, path, cmd in steps:
            if os.path.exists(path) and not a.force:
                print(f"   [exists] {path}")
            else:
                print("   $ " + " ".join(map(str, cmd)))
                r = subprocess.run(cmd)
                if r.returncode != 0:
                    raise SystemExit(f"{tag}: exit {r.returncode}")
            if tag.endswith("-BF16") and "BF16" not in [x.upper() for x in rungs]:
                record(key, path, tag, "gguf", kind, {**meta, "publish": False})
            else:
                record(key, path, tag, "gguf", kind, {**meta, "via": bf16})
    if not a.no_hf:
        dest = os.path.join(out_dir, "hf")
        dest, how = stage_hf(ddir, dest, kind, a.target, shutil.which("speculators"))
        bf16 = os.path.join(out_dir, gguf_name(a.target, kind, "BF16", simple_name))
        record(key, dest, build_tag(kind, "hf", simple_name), "hf", kind,
               {**meta, "how": how, "via": bf16 if os.path.exists(bf16) else None})
        print(f"   HF format ({how}) -> {dest}")
    if a.mlx and kind == "simple":
        dest = os.path.join(out_dir, f"mlx-{a.mlx_bits}bit")
        cmd = [os.environ.get("POLLARD_MLX_PYTHON") or sys.executable, "-m", "mlx_lm", "convert", "--hf-path", ddir,
               "--mlx-path", dest, "-q", "--q-bits", str(a.mlx_bits)]
        print("   $ " + " ".join(cmd))
        if subprocess.run(cmd).returncode == 0:
            record(key, dest, build_tag(kind, f"mlx-{a.mlx_bits}bit", simple_name), "mlx", kind, meta)
    print(f"   {len(draft_builds(key))} drafter rung(s) in the MANIFEST ({ws.model_dir(key)})")
    return 0


# ----------------------------------------------------------------------------- measure

def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_WEATHER = _tool("get_weather", "Current weather for a city.",
                 {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, ["city"])
_SEARCH = _tool("search_files", "Search the repository for a pattern.",
                {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"])
_SHELL = _tool("run_shell", "Run a shell command and return its output.", {"command": {"type": "string"}}, ["command"])

#: fixed, short, mixed: what people run a drafter for -- chat, code, step-by-step, and AGENT turns in the model's own
#: tool-call format (a call to emit, and a turn after a tool result). Prompt kind is reported separately: tau is
#: ~2 on chat and ~4 on reasoning for the same pair, so one pooled number hides what a user will see.
PROMPTS = [
    {"id": "chat-hashmap", "kind": "chat", "messages": [{"role": "user", "content":
        "Explain how a hash map works to a new programmer, in two short paragraphs."}]},
    {"id": "chat-email", "kind": "chat", "messages": [{"role": "user", "content":
        "Write a polite three-sentence email asking a colleague to review my pull request by Friday."}]},
    {"id": "chat-history", "kind": "chat", "messages": [{"role": "user", "content":
        "Summarise the causes of the French Revolution in five bullet points."}]},
    {"id": "code-python", "kind": "code", "messages": [{"role": "user", "content":
        "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests."}]},
    {"id": "code-sql", "kind": "code", "messages": [{"role": "user", "content":
        "Write a SQL query returning the top 5 customers by total order value from tables customers(id, name) and "
        "orders(id, customer_id, amount)."}]},
    {"id": "code-js", "kind": "code", "messages": [{"role": "user", "content":
        "Write a JavaScript debounce function and show how to use it on a search input."}]},
    {"id": "reason-trains", "kind": "reasoning", "messages": [{"role": "user", "content":
        "A train leaves at 9:00 at 60 km/h, another at 10:00 at 90 km/h on the same track. When does the second "
        "catch up? Reason step by step."}]},
    {"id": "reason-primes", "kind": "reasoning", "messages": [{"role": "user", "content":
        "Is 391 prime? Work it out step by step."}]},
    {"id": "tool-weather", "kind": "tool", "tools": [_WEATHER], "messages": [{"role": "user", "content":
        "What's the weather in Lisbon in celsius?"}]},
    {"id": "tool-search", "kind": "tool", "tools": [_SEARCH, _SHELL], "messages": [{"role": "user", "content":
        "Find where the function parse_config is defined in this repo."}]},
    {"id": "tool-shell", "kind": "tool", "tools": [_SEARCH, _SHELL], "messages": [{"role": "user", "content":
        "List the files in the current directory, then tell me which is largest."}]},
    {"id": "tool-after-result", "kind": "tool", "tools": [_WEATHER], "messages": [
        {"role": "user", "content": "Is it warm enough for a swim in Nice today?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": "Nice", "unit": "c"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"temp_c": 24, "sky": "sunny",
                                                                           "water_c": 21})}]},
]


def pick_prompts(limit=0):
    """`limit` prompts spread over the kinds (round-robin), so a short run still covers agent turns."""
    if not limit or limit >= len(PROMPTS):
        return list(PROMPTS)
    by = {}
    for p in PROMPTS:
        by.setdefault(p["kind"], []).append(p)
    out, i = [], 0
    while len(out) < limit:
        for k in by:
            if i < len(by[k]) and len(out) < limit:
                out.append(by[k][i])
        i += 1
    return out


def stats(rows):
    """tau / accept_len / accept_rate over measured rows ({n, drafted, accepted}) -- pooled over tokens, not a mean
    of per-prompt ratios (a 12-token answer must not weigh like a 128-token one)."""
    n = sum(r["n"] for r in rows)
    acc = sum(r["accepted"] for r in rows)
    dr = sum(r["drafted"] for r in rows)
    steps = n - acc
    if n <= 0 or steps <= 0:
        return None
    out = {"tau": round(n / steps, 3), "accept_len": round(acc / steps, 3),
           "accept_rate": round(acc / dr, 4) if dr else 0.0, "tokens": n, "prompts": len(rows)}
    tps = [r["tps"] for r in rows if r.get("tps")]
    if tps and all(r.get("ms") for r in rows):
        out["tok_s"] = round(n / (sum(r["ms"] for r in rows) / 1000.0), 2)
    return out


def summarise(rows):
    s = stats(rows) or {}
    kinds = {}
    for k in sorted({r["kind"] for r in rows}):
        ks = stats([r for r in rows if r["kind"] == k])
        if ks:
            kinds[k] = {x: ks[x] for x in ("tau", "accept_rate", "tokens") if x in ks}
    if kinds:
        s["by_kind"] = kinds
    return s


def row_from_timings(prompt, t):
    """One prompt's counters from llama-server's timings (draft_n / draft_n_accepted appear once it drafted)."""
    return {"id": prompt["id"], "kind": prompt["kind"], "n": int(t.get("predicted_n") or 0),
            "drafted": int(t.get("draft_n") or 0), "accepted": int(t.get("draft_n_accepted") or 0),
            "ms": float(t.get("predicted_ms") or 0.0), "tps": float(t.get("predicted_per_second") or 0.0)}


def request_body(prompt, max_tokens, no_think=True, model=None):
    body = {"messages": prompt["messages"], "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "cache_prompt": False}
    if prompt.get("tools"):
        body["tools"] = prompt["tools"]
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if model:
        body["model"] = model
    return body


def _post(base, body, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions", method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def gguf_server_cmd(binary, target, drafter, kind, port, ngl="0", ngld=None, n_max=None, ctx=4096, threads=None):
    cmd = [binary, "-m", target, "--host", "127.0.0.1", "--port", str(port), "-ngl", str(ngl), "-c", str(ctx),
           "-np", "1", "--jinja", "--reasoning-format", "deepseek"]
    if drafter:
        cmd += ["-md", drafter, "--spec-type", METHODS[kind]["spec_type"],
                "--spec-draft-n-max", str(n_max or METHODS[kind]["n_max"]),
                "-ngld", str(ngl if ngld is None else ngld)]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def gguf_kind(path):
    """The drafter kind a GGUF declares (general.architecture), else None (an ordinary model = `simple`)."""
    try:
        from gguf import GGUFReader
        f = GGUFReader(path)
        fld = f.fields.get("general.architecture")
        arch = bytes(fld.parts[fld.data[0]]).decode() if fld else ""
        return GGUF_ARCH_KIND.get(arch)
    except Exception:                                            # noqa: BLE001
        return None


def _gb(path):
    try:
        if os.path.isdir(path):
            return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e9
        return os.path.getsize(path) / 1e9
    except OSError:
        return 0.0


def measure_gguf(target, drafter, kind, prompts, max_tokens=128, ngl="0", ngld=None, n_max=None, ctx=4096,
                 threads=None, server=None, no_think=True, log_path=None, baseline=False, progress=print):
    """Serve target + drafter in llama-server and read its own draft counters per prompt."""
    import pollard_toolcall as T
    from pollard_calc import find_llama_bin
    binary = server or find_llama_bin("llama-server") or shutil.which("llama-server")
    if not binary:
        raise NotMeasured("llama-server not found (install llama.cpp or pass --llama-server)")
    need = _gb(target) + _gb(drafter)
    if str(ngl) in ("0", "") or sys.platform == "darwin":        # unified memory: the GPU's share is RAM too
        try:
            T.check_ram(need)
        except T.NotMeasured as e:
            raise NotMeasured(str(e)) from None
    out = {}
    runs = [("spec", drafter)] + ([("plain", None)] if baseline else [])
    for label, d in runs:
        port = T._free_port()
        cmd = gguf_server_cmd(binary, target, d, kind, port, ngl, ngld, n_max, ctx, threads)
        rows = []
        with T._launch(cmd, f"llama-server ({label}) on {os.path.basename(target)}",
                       log_path or "draft-server.log", port=port) as base:
            for p in prompts:
                r = _post(base, request_body(p, max_tokens, no_think))
                row = row_from_timings(p, r.get("timings") or {})
                rows.append(row)
                if progress:
                    progress(f"     {label:5s} {p['id']:18s} n={row['n']:4d} drafted={row['drafted']:4d} "
                             f"accepted={row['accepted']:4d}  {row['tps']:.1f} tok/s")
        out[label] = rows
    return out


_PROM = re.compile(r"^(vllm:spec_decode_num_(?:accepted_tokens|draft_tokens|drafts))_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
                   re.M)


def vllm_counters(text):
    """vLLM's Prometheus spec-decode counters, summed over label sets."""
    out = {}
    for name, v in _PROM.findall(text or ""):
        key = name.rsplit("_", 1)[-1] if name.endswith("drafts") else name.split("num_")[-1]
        out[key] = out.get(key, 0.0) + float(v)
    return {"accepted": out.get("accepted_tokens", 0.0), "drafted": out.get("draft_tokens", 0.0),
            "drafts": out.get("drafts", 0.0)}


def vllm_stats(before, after):
    acc = after["accepted"] - before["accepted"]
    dr = after["drafted"] - before["drafted"]
    steps = after["drafts"] - before["drafts"]
    if steps <= 0:
        return None
    return {"tau": round(1 + acc / steps, 3), "accept_len": round(acc / steps, 3),
            "accept_rate": round(acc / dr, 4) if dr else 0.0}


def mlx_tokens_per_cycle(metrics):
    """dflash-mlx's /metrics: every tokens_per_cycle it reports (recent requests), wherever it nests them."""
    vals = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "tokens_per_cycle" and isinstance(v, (int, float)):
                    vals.append(float(v))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(metrics)
    return vals


def vllm_spec_config(kind, model, n_max=None):
    cfg = {"method": METHODS[kind]["vllm"], "model": model}
    if n_max:
        cfg["num_speculative_tokens"] = int(n_max)
    return cfg


def measure_vllm(target, drafter, kind, prompts, max_tokens=128, n_max=None, server=None, progress=print):
    import pollard_toolcall as T
    binary = server or shutil.which("vllm")
    if not binary:
        raise NotMeasured("vLLM is not installed here")
    try:
        T.check_gpu("vllm", _gb(target) + _gb(drafter))
    except T.NotMeasured as e:
        raise NotMeasured(str(e)) from None
    port = T._free_port()
    cmd = [binary, "serve", target, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "draft",
           "--max-model-len", "4096", "--speculative-config", json.dumps(vllm_spec_config(kind, drafter, n_max))]
    with T._launch(cmd, f"vllm serve {os.path.basename(target)} + drafter", "draft-vllm.log", port=port) as base:
        def metrics():
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                return vllm_counters(r.read().decode())
        before = metrics()
        for p in prompts:
            _post(base, request_body(p, max_tokens, True, "draft"))
        s = vllm_stats(before, metrics())
    if not s:
        raise NotMeasured("vLLM reported no speculative steps")
    return s


def measure_mlx(target, drafter, prompts, max_tokens=128, progress=print):
    import pollard_toolcall as T
    binary = shutil.which("dflash")
    if not binary:
        raise NotMeasured("dflash-mlx is not installed here (pip install dflash-mlx; Apple Silicon)")
    port = T._free_port()
    cmd = [binary, "serve", "--model", target, "--draft", drafter, "--port", str(port)]
    with T._launch(cmd, "dflash serve", "draft-mlx.log", port=port, health="/metrics") as base:
        for p in prompts:
            _post(base, request_body(p, max_tokens, True, target))
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            vals = mlx_tokens_per_cycle(json.loads(r.read().decode()))
    if not vals:
        raise NotMeasured("dflash-mlx reported no tokens_per_cycle")
    tau = sum(vals) / len(vals)
    return {"tau": round(tau, 3), "accept_len": round(tau - 1, 3)}


def _hw():
    import platform
    try:
        if sys.platform == "darwin":
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
            return chip or platform.machine()
    except Exception:                                            # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def record_measure(results, rung, target_tag, entry, meta):
    """Merge one (drafter rung, target rung) measurement into results-draft.json. The rung's headline tau and
    accept_rate are the WORST over the target rungs measured so far -- the gate holds the weakest pairing."""
    import pollard_workspace as ws
    try:
        cur = json.load(open(results, encoding="utf-8")) if os.path.exists(results) else {}
    except Exception:                                            # noqa: BLE001
        cur = {}
    d = dict(((cur.get(rung) or {}).get("draft") or {}))
    targets = dict(d.get("targets") or {})
    targets[target_tag] = entry
    d.update(meta)
    d["targets"] = targets
    meas = [v for v in targets.values() if isinstance(v, dict) and v.get("tau") is not None]
    for k in ("tau", "accept_rate", "accept_len", "reason"):
        d.pop(k, None)
    if meas:
        d["tau"] = min(v["tau"] for v in meas)
        rates = [v["accept_rate"] for v in meas if v.get("accept_rate") is not None]
        if rates:
            d["accept_rate"] = min(rates)
        d["accept_len"] = round(d["tau"] - 1, 3)
        d["status"] = "measured"
    else:
        d["status"] = "not-measured"
        d["reason"] = next((v.get("reason") for v in targets.values() if isinstance(v, dict) and v.get("reason")),
                           None)
    return ws.merge_results(results, rung, {"draft": d})


def build_meta(drafter_path):
    """What the MANIFEST knows about this drafter rung (kind, source, licence)."""
    try:
        import pollard_workspace as ws
        _, b = ws.find_build(drafter_path)
    except Exception:                                            # noqa: BLE001
        b = None
    b = b or {}
    return {k: b.get(k) for k in ("kind", "source", "license", "redistributable", "spec_type", "format", "via",
                                  "quant_aware_vs", "trained_by", "trainer_url", "taps")}


def cmd_measure(a):
    targets = a.target or []
    tags = (a.target_tag or []) + [None] * len(targets)
    if not targets:
        raise SystemExit("pass --target <quantized target rung> (repeatable, with --target-tag)")
    meta = build_meta(a.drafter)
    path = a.via or (meta.get("via") if meta.get("format") == "hf" else None) or a.drafter
    kind = a.kind or meta.get("kind") or (gguf_kind(path) if path.endswith(".gguf") else None) or "simple"
    rung = a.rung or os.path.basename(a.drafter.rstrip("/\\"))
    prompts = pick_prompts(a.limit)
    n_max = a.n_max or METHODS[kind]["n_max"]
    rec = {"kind": kind, "n_max": n_max, "backend": a.backend, "hw": a.hw or _hw(), "max_tokens": a.max_tokens,
           "prompt_set": f"pollard-draft/{len(PROMPTS)}" + (f" (first {len(prompts)})" if a.limit else ""),
           **{k: v for k, v in meta.items() if v is not None and k not in ("via", "format")}}
    if meta.get("redistributable") is None and a.redistributable is not None:
        rec["redistributable"] = a.redistributable
    if path != a.drafter:
        rec["measured_via"] = os.path.basename(path)
    print(f"== pollard-draft measure :: {rung} ({kind}, draft {n_max}) on {len(targets)} target rung(s), "
          f"{len(prompts)} prompts x {a.max_tokens} tokens")
    for t, tag in zip(targets, tags):
        tag = tag or os.path.basename(t)
        try:
            if a.backend == "gguf":
                runs = measure_gguf(t, path, kind, prompts, a.max_tokens, a.ngl, a.ngld, n_max, a.ctx, a.threads,
                                    a.llama_server, not a.think, baseline=a.baseline)
                entry = summarise(runs["spec"])
                if not entry:
                    raise NotMeasured("the server reported no generated tokens")
                if "plain" in runs:
                    plain = stats([{**r, "accepted": 0, "drafted": 0} for r in runs["plain"]]) or {}
                    if plain.get("tok_s") and entry.get("tok_s"):
                        entry["plain_tok_s"] = plain["tok_s"]
                        entry["speedup"] = round(entry["tok_s"] / plain["tok_s"], 3)
                entry["rows"] = runs["spec"] if a.keep_rows else None
            elif a.backend == "vllm":
                entry = measure_vllm(t, path, kind, prompts, a.max_tokens, n_max, a.vllm)
            else:
                entry = measure_mlx(t, path, prompts, a.max_tokens)
            entry = {k: v for k, v in entry.items() if v is not None}
            print(f"   {tag}: tau {entry['tau']:.2f} tokens/pass, accept rate "
                  f"{100 * entry.get('accept_rate', 0):.1f}%" + (f", {entry['tok_s']:.1f} tok/s" if entry.get('tok_s')
                                                                  else ""))
        except NotMeasured as e:
            entry = {"status": "not-measured", "reason": str(e)}
            print(f"   {tag}: NOT MEASURED -- {e}")
        if a.results:                                            # NOT MEASURED is recorded, not a crash:
            record_measure(a.results, rung, tag, entry, rec)     # the gate decides what ships
    return 0


# ----------------------------------------------------------------------------- serve lines + cards

def serve_lines(kind, target_repos, draft_repo, gguf_target=None, gguf_draft=None, n_max=None):
    """The exact serve command per runtime, for this kind of drafter. target_repos: {lane: repo}."""
    m = METHODS[kind]
    n = n_max or m["n_max"]
    lines = {}
    if target_repos.get("gguf"):
        t = gguf_target or "<target rung>.gguf"
        d = gguf_draft or "<drafter rung>.gguf"
        lines["llama.cpp"] = (f"# files from {target_repos['gguf']} and {draft_repo}\n"
                              f"llama-server -m {t} -md {d} \\\n  --spec-type {m['spec_type']} "
                              f"--spec-draft-n-max {n} -ngl 99 -ngld all --jinja")
    vtarget = target_repos.get("gptq") or target_repos.get("mx") or target_repos.get("base")
    if vtarget and m["vllm"]:
        sc = json.dumps(vllm_spec_config(kind, draft_repo, n))
        lines["vLLM"] = f"vllm serve {vtarget} \\\n  --speculative-config '{sc}'"
        if m["sglang"] and kind != "simple":
            extra = ("--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
                     if kind == "eagle3" else f"--speculative-num-draft-tokens {n + 1}")
            lines["SGLang"] = (f"python -m sglang.launch_server --model-path {vtarget} \\\n"
                               f"  --speculative-algorithm {m['sglang']} --speculative-draft-model-path {draft_repo} "
                               f"\\\n  {extra}")
    mt = target_repos.get("mlx")
    if mt:
        if m["mlx"] == "dflash-mlx":
            lines["MLX"] = f"pip install dflash-mlx\ndflash serve --model {mt} --draft {draft_repo} --port 8000"
        elif m["mlx"] == "mlx-dspark":
            lines["MLX"] = f"# mlx-dspark (MIT): see github.com/ARahim3/mlx-dspark\n# target {mt}, drafter {draft_repo}"
        elif m["mlx"] == "mlx_lm":
            lines["MLX"] = (f"mlx_lm.server --model {mt} --draft-model {draft_repo} --num-draft-tokens {n}")
        else:
            lines["MLX"] = "# no MLX runtime serves EAGLE-3 heads yet (mlx-lm runs classic draft models only)"
    return lines


def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "--"


def measured_table(results, only_rungs=None):
    """drafter rung x target rung -> tau / accept rate / tok/s, from results-draft.json."""
    rows = ["| drafter | target rung | tokens / target pass (tau) | accepted / pass | accept rate | chat / code / "
            "reasoning / tool tau | tok/s (single stream) |", "|---|---|---:|---:|---:|---|---:|"]
    hw = set()
    for rung, r in sorted((results or {}).items()):
        if rung.startswith("_") or not isinstance(r, dict) or not isinstance(r.get("draft"), dict):
            continue
        if only_rungs is not None and rung not in only_rungs:
            continue
        d = r["draft"]
        if d.get("hw"):
            hw.add(d["hw"])
        for tag, t in sorted((d.get("targets") or {}).items()):
            if not isinstance(t, dict):
                continue
            if t.get("status") == "not-measured":
                rows.append(f"| {rung} | {tag} | not measured: {t.get('reason')} | | | | |")
                continue
            bk = t.get("by_kind") or {}
            kinds = " / ".join(_fmt((bk.get(k) or {}).get("tau")) for k in ("chat", "code", "reasoning", "tool"))
            tps = _fmt(t.get("tok_s"), 1) + (f" ({t['speedup']:.2f}x plain)" if t.get("speedup") else "")
            rows.append(f"| {rung} | {tag} | **{_fmt(t.get('tau'))}** | {_fmt(t.get('accept_len'))} | "
                        f"{_fmt(100 * t['accept_rate'], 1) + '%' if t.get('accept_rate') is not None else '--'} | "
                        f"{kinds} | {tps} |")
    return rows if len(rows) > 2 else [], sorted(hw)


TRAINED = "pollard-trained"


def trained_phrase(meta, md=True):
    """"Pollard-trained, quant-aware vs Q4_K_M" + how, for a head `pollard-draft train` made; None otherwise."""
    if (meta or {}).get("source") != TRAINED:
        return None
    vs = meta.get("quant_aware_vs")
    by = meta.get("trained_by") or "vLLM speculators (Apache-2.0)"
    url = meta.get("trainer_url") or "https://github.com/vllm-project/speculators"
    trainer = f"[{by}]({url})" if md else by
    if not vs:
        return f"Pollard-trained against the bf16 target with {trainer}"
    taps = " and hidden-state taps from that rung's own weights" if meta.get("taps") == "rung" else ""
    return (f"Pollard-trained, quant-aware vs {vs}: trained with {trainer} on answers regenerated by the {vs} "
            f"rung{taps}")


def card_section(results, draft_repo, target_repos, lane=None, kind=None, folder_of=None):
    """`## Speculative decoding` for a TARGET lane's card: the matched drafter, its measured acceptance against this
    model's quantized rungs, and the serve line for this lane's runtime. `folder_of` = the ONE repo that holds the
    drafter in its DRAFT/ folder; `draft_repo` is then the local path that folder downloads to."""
    if not results:
        return []
    rungs = {k: v for k, v in results.items() if not k.startswith("_") and isinstance(v, dict)
             and isinstance(v.get("draft"), dict) and (v.get("verified") or {}).get("pass") is not False}
    if not rungs:
        return []
    kind = kind or next((v["draft"].get("kind") for v in rungs.values() if v["draft"].get("kind")), "dflash")
    src = next((v["draft"].get("source") for v in rungs.values() if v["draft"].get("source")), None)
    own = next((trained_phrase(v["draft"]) for v in rungs.values() if trained_phrase(v["draft"])), None)
    table, hw = measured_table(rungs)
    where = (f"this repo's `DRAFT/` folder (`hf download {folder_of} --include \"DRAFT/*\" --local-dir "
             f"{os.path.dirname(draft_repo) or '.'}`)" if folder_of else
             f"[{draft_repo}](https://huggingface.co/{draft_repo})")
    out = ["## Speculative decoding", "",
           f"A drafter matched to these quantized rungs ships in {where}"
           + (f" ({METHODS[kind]['label']}; {own})" if own else
              f" ({METHODS[kind]['label']}, from [{src}](https://huggingface.co/{src}))" if src else "") + ". "
           "The target verifies every drafted token, so the output is the target's own; each accepted token is a "
           "target pass saved.", ""]
    if table:
        out += table + ["", "_tau = tokens produced per target pass (1.0 = no gain), measured with the drafter "
                            "against the quantized rung, greedy, on pollard-draft's fixed chat / code / reasoning / "
                            "tool-call prompts" + (f" on {', '.join(hw)}" if hw else "") + "._", ""]
    lines = serve_lines(kind, target_repos, draft_repo)
    runtime = {"gguf": "llama.cpp", "mlx": "MLX", "gptq": "vLLM", "mx": "vLLM", "exl3": None}.get(lane or "", None)
    pick = {runtime: lines[runtime]} if runtime and runtime in lines else lines
    for name, cmd in pick.items():
        out += [f"**{name}**", "", "```bash", cmd, "```", ""]
    return out


def draft_card(target, results, repo, target_repos, builds, verified_only=True, license_=None, source=None,
               kind=None, base_license=None):
    """README for <M>-Pollard-DRAFT: what the drafter is, the measured acceptance per target rung, the files, and the
    serve line per runtime."""
    res = results or {}

    def ok(b):
        v = (res.get(b.get("tag")) or {}).get("verified") or {}
        return v.get("pass") is True or (not verified_only)
    shown = [b for b in builds if b.get("publish", True) and ok(b)]
    kind = kind or next((b.get("kind") for b in shown if b.get("kind")), "dflash")
    source = source or next((b.get("source") for b in shown if b.get("source")), None)
    lic = license_ or next((b.get("license") for b in shown if b.get("license")), None) or "other"
    own = next((trained_phrase(b) for b in shown if trained_phrase(b)), None)
    if own:
        own_meta = next(b for b in shown if trained_phrase(b))
        source = None
    base = _basename(target)
    fm = ["---", f"license: {lic}", "base_model:"] + ([f"- {source}"] if source else []) + [
        f"- {target}" if "/" in str(target) else f"- {base}", "tags:", "- pollard", "- speculative-decoding",
        f"- {kind}"] + (["- pollard-trained", "- quant-aware"] if own else []) + [
        "- gguf", "library_name: gguf", "pipeline_tag: text-generation", "---", ""]
    out = fm + [f"# {base} -- Pollard drafter ({METHODS[kind]['label']})", "",
                f"A speculative-decoding drafter for **{target}**, matched to the Pollard quantized rungs and measured "
                "against them. Pair it with any rung of "
                + ", ".join(f"[{r}](https://huggingface.co/{r})" for lane_, r in sorted(target_repos.items())
                            if lane_ != "base") + ".", ""]
    if own:
        vs = own_meta.get("quant_aware_vs")
        out += [f"**{own}.** "
                + (f"A head trained against bf16 loses acceptance on a quantized rung: every token where {vs} and "
                   f"bf16 disagree is a rejected draft. This one learned what {vs} says. " if vs else "")
                + "The table lists acceptance against the quantized rung and against bf16, so the difference is "
                  "on the page. The drafter changes speed, never the text: the target verifies every token.", ""]
    if source:
        out += [f"Drafter weights: [{source}](https://huggingface.co/{source}) ({lic}), converted and re-quantized "
                "with [Pollard Weights](https://github.com/WestWaters/pollard-weights). The drafter changes speed, "
                "never the text: the target verifies every token.", ""]
    table, hw = measured_table(res, {b.get("tag") for b in shown} | {
        k for k, v in res.items() if isinstance(v, dict) and (v.get("draft") or {}).get("measured_via")})
    if table:
        out += ["## Measured acceptance", ""] + table + [
            "", "_tau = tokens produced per target pass = generated / (generated - accepted draft tokens), from the "
                "runtime's own counters, greedy, pollard-draft's fixed 12-prompt set (chat, code, step-by-step, and "
                "agent turns in the model's tool-call format)" + (f"; {', '.join(hw)}" if hw else "") +
                ". Every rung listed passed `pollard-verify --lane draft --gate`._", ""]
    files = [b for b in shown if b.get("format") == "gguf"]
    hf = [b for b in shown if b.get("format") == "hf"]
    if files or hf:
        out += ["## Files", "", "| file | for | size |", "|---|---|---:|"]
        for b in sorted(files, key=lambda x: -(x.get("bytes") or 0)):
            out.append(f"| `{b.get('name')}` | llama.cpp `-md` | {(b.get('bytes') or 0) / 1e9:.2f} GB |")
        for b in hf:
            out.append(f"| `config.json` + `*.safetensors` (repo root) | vLLM / SGLang / MLX | "
                       f"{(b.get('bytes') or 0) / 1e9:.2f} GB |")
        out.append("")
    small = min(files, key=lambda x: x.get("bytes") or 0).get("name") if files else None
    out += ["## Run it", ""]
    for name, cmd in serve_lines(kind, target_repos, repo, f"{base}-Pollard-<RUNG>.gguf", small).items():
        out += [f"**{name}**", "", "```bash", cmd, "```", ""]
    out += ["## Credits", "", f"- Base model: {target}" + (f" ({base_license})" if base_license else ""),
            (f"- Drafter: trained by Pollard ({lic}) with {own_meta.get('trained_by') or 'vLLM speculators (Apache-2.0)'}"
             f" -- {own_meta.get('trainer_url') or 'https://github.com/vllm-project/speculators'}") if own else
            f"- Drafter: {source or 'see above'}", "- Runtimes: llama.cpp, vLLM, SGLang, dflash-mlx / mlx-lm",
            "- Built and measured with [Pollard Weights](https://github.com/WestWaters/pollard-weights)", ""]
    return "\n".join(out)


def target_repos_for(org, target, lanes=("gguf", "mlx", "gptq", "mx")):
    from pollard_release import repo_name
    out = {lane: repo_name(org, target, lane) for lane in lanes}
    out["base"] = target if "/" in str(target) and not os.path.exists(str(target)) else None
    return {k: v for k, v in out.items() if v}


def cmd_card(a):
    res = json.load(open(a.results, encoding="utf-8")) if a.results and os.path.exists(a.results) else {}
    builds = draft_builds(a.key or a.target)
    lanes = [x.strip() for x in a.lanes.split(",")] if a.lanes else ["gguf", "mlx", "gptq", "mx"]
    trepos = target_repos_for(a.org, a.target, lanes)
    from pollard_release import repo_name
    repo = a.repo or repo_name(a.org, a.target, "draft")
    md = draft_card(a.target, res, repo, trepos, builds, a.verified_only)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(md)
    print(f"   card -> {a.out}")
    return 0


# ----------------------------------------------------------------------------- train (recipe; not run here)

TRAINERS = {
    # vLLM speculators: Apache-2.0; trains eagle3 / dflash / dflash2 / dspark / peagle / mtp; output serves in vLLM
    # as-is and converts to GGUF with llama.cpp's converter (--target-model-dir)
    "speculators": {"licence": "Apache-2.0", "url": "https://github.com/vllm-project/speculators"},
    # SpecForge (sgl-project): MIT; EAGLE-3 / DFlash / DFlash2 / Domino training for SGLang
    "specforge": {"licence": "MIT", "url": "https://github.com/sgl-project/SpecForge"},
}


def train_recipe(target, kind="dflash", data="./draft-data", out="./draft-ckpt", trainer="speculators",
                 regen_url=None, epochs=None):
    """The documented commands to train a drafter for a model nobody has published one for. Online training: a vLLM
    server feeds the target's hidden states while the drafter trains (one CUDA box; 8B target ~ 1 GPU-day).
    `regen_url` = an OpenAI-compatible server running the QUANTIZED rung: the training answers are regenerated by
    the model the drafter will serve (arXiv 2607.04244's adaptation; Graft measured Q8_0->Q3_K_M costs a stock
    DFlash only ~3% tau on Qwen3-4B, so this is the second step, not the first)."""
    if trainer != "speculators":
        return [f"# {trainer} ({TRAINERS[trainer]['licence']}): follow {TRAINERS[trainer]['url']} "
                f"examples/configs for {kind}; then pollard-draft convert --drafter {out}"]
    t = "dflash" if kind in ("dflash", "simple") else kind
    gen = regen_url or "http://127.0.0.1:8000"
    return [
        "pip install speculators vllm   # Apache-2.0, Linux + CUDA; git clone the repo for scripts/launch_vllm.py",
        f"python scripts/launch_vllm.py {target} -- --port 8000 &   # hidden-state + render server for the target",
        f"speculators regenerate-responses --dataset ultrachat --limit 20000 --endpoint {gen}/v1/chat/completions "
        f"--outfile {data}/regen.jsonl   # on-policy answers" + (", from the QUANTIZED rung" if regen_url else ""),
        f"speculators prepare-data --model {target} --data {data}/regen.jsonl "
        f"--render-endpoint http://127.0.0.1:8000 --output {data}",
        f"speculators train --verifier-name-or-path {target} --speculator-type {t} --data-path {data} "
        f"--save-path {out} --vllm-endpoint http://127.0.0.1:8000/v1" + (f" --epochs {epochs}" if epochs else "")
        + (" --draft-vocab-size 32000" if t == "eagle3" else ""),
        f"pollard-draft convert --target {target} --drafter {out}/<best checkpoint> --kind {t}"]


def cmd_train(a):
    """--run (or --if-needed, the release step): Pollard's own quant-aware head, end to end (pollard_draft_train).
    Otherwise: print the plain upstream recipe -- what the trainer's own docs would have you run."""
    if a.run or a.if_needed:
        import pollard_draft_train as DT
        return DT.cmd_train(a)
    cmds = train_recipe(a.target, a.kind, a.data, a.out, a.trainer, a.regen_url, a.epochs)
    print(f"== pollard-draft train :: {a.target} ({a.kind}, {a.trainer}, "
          f"{TRAINERS[a.trainer]['licence']} -- {TRAINERS[a.trainer]['url']})")
    for c in cmds:
        print("   " + c)
    print("\n   recipe only -- `--run` trains Pollard's own QUANT-AWARE head instead: answers regenerated by the "
          "quantized rung (--rung <.gguf>), taps from its weights, speculators on a free CUDA GPU (--smoke: a tiny "
          "local proof).")
    return 0


# ----------------------------------------------------------------------------- CLI

def cmd_find(a):
    res = find(a.target, repo_config(a.target_dir) if a.target_dir else None, drafter=a.drafter,
               max_check=a.max_check, with_simple=not a.no_simple)
    print_find(res)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"   -> {a.out}")
    return 0 if res.get("pick") else 1


def cmd_serve(a):
    trepos = target_repos_for(a.org, a.target)
    from pollard_release import repo_name
    for name, cmd in serve_lines(a.kind, trepos, a.repo or repo_name(a.org, a.target, "draft"),
                                 n_max=a.n_max).items():
        print(f"# {name}\n{cmd}\n")
    return 0


def main(argv=None):
    for p in (os.path.dirname(os.path.abspath(__file__)),):
        if p not in sys.path:
            sys.path.insert(0, p)
    ap = argparse.ArgumentParser(prog="pollard-draft", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="locate a published drafter for the target and check it fits")
    f.add_argument("--target", required=True, help="the target's HF id (or a local checkout)")
    f.add_argument("--target-dir", help="local checkout of the target (its config.json is read from here)")
    f.add_argument("--drafter", help="check this drafter repo/dir instead of searching")
    f.add_argument("--max-check", type=int, default=12, help="candidate configs to fetch (default 12)")
    f.add_argument("--no-simple", action="store_true", help="do not propose a same-family small model")
    f.add_argument("--out", help="write the result (drafter.json) for `convert --from`")
    c = sub.add_parser("convert", help="GGUF ladder + HF folder (+ MLX for a simple drafter)")
    c.add_argument("--target", required=True)
    c.add_argument("--target-dir", help="local target checkout (its tokenizer goes into the drafter GGUF)")
    c.add_argument("--from", dest="from_", help="drafter.json from `find --out`")
    c.add_argument("--drafter", help="drafter repo or dir (instead of --from)")
    c.add_argument("--kind", choices=sorted(METHODS))
    c.add_argument("--rungs", default=",".join(DEFAULT_RUNGS), help="GGUF quants (default Q8_0,Q4_K_M)")
    c.add_argument("--out-dir", help="default: <workspace>/models/<target>/draft")
    c.add_argument("--key", help="MANIFEST key (default --target)")
    c.add_argument("--python", help="interpreter for convert_hf_to_gguf.py")
    c.add_argument("--no-gguf", action="store_true")
    c.add_argument("--no-hf", action="store_true")
    c.add_argument("--mlx", action="store_true", help="simple drafters: also an MLX copy (mlx_lm convert)")
    c.add_argument("--mlx-bits", type=int, default=8)
    c.add_argument("--force", action="store_true", help="rebuild files that exist")
    c.add_argument("--skip-if-trained", action="store_true",
                   help="do nothing when `pollard-draft train` already shipped Pollard's own head for this target")
    m = sub.add_parser("measure", help="tokens per target pass, drafter vs the quantized target rung(s)")
    m.add_argument("--target", action="append", help="target rung (GGUF file / model dir); repeatable")
    m.add_argument("--target-tag", action="append", help="its rung tag, in --target order")
    m.add_argument("--drafter", required=True, help="drafter rung (GGUF, or the HF folder for vllm/mlx)")
    m.add_argument("--via", help="measure this file in place of --drafter (the hf rung through its BF16 GGUF)")
    m.add_argument("--kind", choices=sorted(METHODS))
    m.add_argument("--rung", help="the drafter rung's tag in --results (default: file name)")
    m.add_argument("--results", help="results-draft.json to merge into")
    m.add_argument("--backend", choices=["gguf", "vllm", "mlx"], default="gguf")
    m.add_argument("--limit", type=int, default=0, help="first N prompts, spread over the kinds (0 = all 12)")
    m.add_argument("--max-tokens", type=int, default=128)
    m.add_argument("--n-max", type=int, help="draft width (default per kind: dflash 7, dspark 6, eagle3 3)")
    m.add_argument("--ngl", default="0", help="target GPU layers (default 0 -- shared boxes)")
    m.add_argument("--ngld", help="drafter GPU layers (default: same as --ngl)")
    m.add_argument("--ctx", type=int, default=4096)
    m.add_argument("--threads", type=int)
    m.add_argument("--llama-server", help="llama-server binary")
    m.add_argument("--vllm", help="vllm binary")
    m.add_argument("--think", action="store_true", help="leave the template's thinking on (default off)")
    m.add_argument("--baseline", action="store_true", help="also the plain target, for a tok/s speedup")
    m.add_argument("--keep-rows", action="store_true", help="keep per-prompt rows in the results")
    m.add_argument("--hw", help="machine name for the card (default: detected)")
    m.add_argument("--redistributable", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                   help="licence verdict when the drafter is not in a MANIFEST")
    k = sub.add_parser("card", help="README for <M>-Pollard-DRAFT")
    k.add_argument("--target", required=True)
    k.add_argument("--results")
    k.add_argument("--repo")
    k.add_argument("--org", default="PollardWeights")
    k.add_argument("--lanes", help="target lanes released (default gguf,mlx,gptq,mx)")
    k.add_argument("--key", help="MANIFEST key (default --target)")
    k.add_argument("--out", default="README.md")
    k.add_argument("--verified-only", action="store_true")
    t = sub.add_parser("train", help="train Pollard's own quant-aware drafter head (or print the upstream recipe)")
    t.add_argument("--target", required=True)
    t.add_argument("--kind", choices=["dflash", "dspark", "eagle3"], default="eagle3",
                   help="eagle3 (default: converts to GGUF, fits a 16GB card), dspark (GGUF too), dflash (HF only)")
    t.add_argument("--trainer", choices=sorted(TRAINERS), default="speculators")
    t.add_argument("--data", default="./draft-data", help="(recipe printout only)")
    t.add_argument("--out", default="./draft-ckpt", help="(recipe printout only)")
    t.add_argument("--epochs", type=int)
    t.add_argument("--run", action="store_true", help="train (CUDA box with a free GPU; --smoke anywhere)")
    import pollard_draft_train as DT
    DT.add_args(t)
    s = sub.add_parser("serve", help="print the serve line per runtime")
    s.add_argument("--target", required=True)
    s.add_argument("--kind", choices=sorted(METHODS), default="dflash")
    s.add_argument("--org", default="PollardWeights")
    s.add_argument("--repo")
    s.add_argument("--n-max", type=int)
    a = ap.parse_args(argv)
    return {"find": cmd_find, "convert": cmd_convert, "measure": cmd_measure, "card": cmd_card,
            "train": cmd_train, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
