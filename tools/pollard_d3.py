#!/usr/bin/env python3
"""pollard-d3 -- d3 code-readout decision models (vllm-sr/d3 family): detect, ship whole, gate every rung on its readout.

A d3 model is a Qwen3.5 BODY (`Qwen3_5Model`, no lm_head) plus a separate FP32 readout. `decision_config.json` says
`format_id: d3-code-readout-v1`: every option of a question is listed under a single-token answer code (A..Z, AA..JT,
255 of them), the body reads the whole prompt with NON-CAUSAL full-attention layers (the Gated DeltaNet layers stay
causal), the hidden state of the LAST prompt token goes through `readout.safetensors` ([255, hidden], FP32), and the
answer is the softmax over the question's own codes (temperature from the config). No text is ever generated.

So the first-position letter readout of pollard-decision measures the wrong thing for d3, and a d3 rung is gated
here instead: the reference is the bf16 checkpoint through the package's own runtime (transformers, CPU is enough for
the small sizes), every rung is run through ITS runtime and read out the same way, and the board is the readout
distribution's KL to the reference, top-choice agreement and calibration (ECE / Brier / NLL, and the change from the
reference), on the typed-decisions set (2000 decisions; choice, noul and score questions) or the 20 built-ins.

Runtimes (auto-picked from the rung):
  * GGUF       llama-server --embeddings --pooling last, non-causal attention (baked into the GGUF by `convert`, and
               forced with --override-kv for any other GGUF); the readout is applied to the pooled hidden state here
  * MLX        mlx-lm's Qwen3.5 text layers run without a mask (full attention sees the whole prompt), last token
  * HF lanes   the package's own runtime with the rung's weights decoded to bf16 (GPTQ / MX / compressed-tensors) --
               the numbers of the rung's weights on a box without vLLM; `--systemone URL` measures a served rung
               (vLLM `/v1/systemone`, the rung's own runtime) instead
Every lane quantizes the BODY only: `decision_config.json` and `readout.safetensors` ship byte-exact (pollard-complete
fails a rung without them), and a rung gets its own MODEL_MANIFEST.json (the source's lists the source's shard hashes,
so the d3 runtime would refuse the rung).

    pollard-d3 info --model vllm-sr/d3-edge-dir
    pollard-d3 convert --hf <d3 dir> --outfile d3-edge-f16.gguf            # body -> GGUF, non-causal + last pooling
    pollard-d3 gate --rung d3-edge-Q4_K_M.gguf --source <d3 dir> --results results.json --rung-name Q4_K_M
    pollard-d3 gate --rung <MLX or GPTQ dir> --source <d3 dir> [--set builtin] [--limit 40]
    pollard-d3 ship --source <d3 dir> --out <rung dir> --lane gptq        # decision files + the rung's manifest
    pollard-d3 serve --rung <any rung> --source <d3 dir> --port 8000      # /v1/systemone for a GGUF / MLX / HF rung
    pollard-d3 jev-index --rung <rung> --source <d3 dir> --kit <decision-index checkout> --out runs/<rung>
                                                                          # opt-in, overnight: never in the publish path
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, shutil, struct, sys

FORMAT_ID = "d3-code-readout-v1"
CONFIG = "decision_config.json"
READOUT = "readout.safetensors"
#: what makes a d3 checkpoint a decision model: every lane ships these byte for byte
DECISION_FILES = (CONFIG, READOUT)
MANIFEST = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "d3-package-manifest/1"
#: the package's own runtime (Apache-2.0, part of the model): an HF-format rung carries it so AutoModel / pipeline /
#: d3_server.py load the rung exactly as they load the source
RUNTIME_FILES = ("d3_runtime.py", "d3_format.py", "d3_fast.py", "d3_kernels.py", "d3_engine.py", "d3_server.py",
                 "d3_compose.py", "modeling_d3.py", "pipeline_d3.py", "requirements.txt")
#: lanes whose rung is an HF-format checkpoint the d3 runtime / vLLM load (they carry the runtime files too)
HF_FORMAT_LANES = ("gptq", "mx", "hf")
#: the converter class a d3 body converts as (its config says `Qwen3_5Model`, which no converter registers)
CONVERT_ARCH = {"qwen3_5": "Qwen3_5ForConditionalGeneration"}
#: llama-server for a d3 GGUF: embeddings of the last token, every attention layer non-causal
LLAMA_EMBED_ARGS = ("--embeddings", "--pooling", "last", "--embd-normalize", "-1")
#: why the lanes without a d3 runtime are held (the release plan prints this)
HOLD_REASON = {
    "exl3": "exllamav3 has no non-causal body / last-token pooling / readout path: no runtime would serve a d3 rung",
    "exlr8": "the EXLR8 loaders decode a causal LM only: no non-causal body / last-token pooling / readout path",
}
LANE_RUNTIME = {"gguf": "llama.cpp (llama-server --embeddings --pooling last, non-causal)",
                "mlx": "MLX (mlx-lm Qwen3.5 layers, no mask, last token)",
                "hf": "transformers (d3 runtime, rung weights decoded to bf16)",
                "systemone": "/v1/systemone (served rung)",
                "vllm": "vLLM (vllm serve, /v1/systemone)",
                "reference": "transformers (d3 runtime, bf16)"}


# ------------------------------------------------------------------------------------------------- detection

def package_dir(path):
    """The folder holding decision_config.json for `path`: the dir itself, a file's folder, or None."""
    if not path:
        return None
    p = str(path)
    d = p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))
    return d if os.path.isfile(os.path.join(d, CONFIG)) else None


def decision_config(path):
    """decision_config.json of a d3 package (or of the folder beside a GGUF), None for anything else."""
    d = package_dir(path)
    if not d:
        return None
    try:
        cfg = json.load(open(os.path.join(d, CONFIG), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) and cfg.get("format_id") == FORMAT_ID else None


def gguf_is_d3(meta):
    """A GGUF converted by `convert` carries the format id in general.tags."""
    tags = (meta or {}).get("general.tags")
    return isinstance(tags, (list, tuple)) and FORMAT_ID in [str(t) for t in tags]


def is_d3(path, meta=None):
    return decision_config(path) is not None or gguf_is_d3(meta)


def info(path):
    """What `path` is, as a d3 package: the facts a release / card / gate needs."""
    cfg = decision_config(path)
    if cfg is None:
        return None
    d = package_dir(path)
    try:
        conf = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    except (OSError, ValueError):
        conf = {}
    man = _json(os.path.join(d, MANIFEST)) or {}
    hidden = (conf.get("text_config") or conf).get("hidden_size")
    return {"format_id": FORMAT_ID, "dir": d, "model_name": man.get("model_name") or os.path.basename(d),
            "repo_id": man.get("repo_id"), "base_model": cfg.get("base_model"),
            "attention_mode": cfg.get("attention_mode"), "pooling": cfg.get("pooling"),
            "temperature": float(cfg.get("temperature", 1.0)), "codes": len(cfg.get("codes") or []),
            "readout_dtype": cfg.get("readout_dtype", "float32"), "hidden_size": hidden,
            "model_type": conf.get("model_type"), "architectures": conf.get("architectures") or [],
            "runtime_files": [f for f in RUNTIME_FILES if os.path.isfile(os.path.join(d, f))],
            "readout": os.path.isfile(os.path.join(d, READOUT))}


def _json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- shipping

def check_files(src, out, lane=None):
    """(carried, missing) for the d3 parts of a rung folder: the decision files byte-exact, and no stale manifest.
    Empty lists when `src` is not a d3 package."""
    sd = package_dir(src)
    if decision_config(src) is None:
        return [], []
    carried, missing = [], []
    have = {f: os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs if f in DECISION_FILES + (MANIFEST,)}
    for f in DECISION_FILES:
        s = os.path.join(sd, f)
        if f not in have:
            missing.append(f"d3 {f}: absent (every lane ships it byte-exact; run pollard-d3 ship)")
        elif sha256_file(have[f]) != sha256_file(s):
            missing.append(f"d3 {f}: differs from the source (must ship byte-exact -- the body is quantized, "
                           "the readout never)")
        else:
            carried.append(f"d3 {f} byte-exact")
    sm = os.path.join(sd, MANIFEST)
    if lane != "gguf" and MANIFEST in have and os.path.isfile(sm) and sha256_file(have[MANIFEST]) == sha256_file(sm):
        missing.append(f"d3 {MANIFEST}: the SOURCE manifest (its shard hashes) -- the d3 runtime would refuse this "
                       "rung; run pollard-d3 ship to write the rung's own")
    return carried, missing


def ship(src, out, lane, rung=None, repo_id=None, log=print):
    """Make a rung folder a whole d3 package: the decision files copied byte for byte, the package's runtime (HF-format
    lanes), and a MODEL_MANIFEST.json describing THIS rung's files. Returns what was written."""
    sd = package_dir(src)
    if decision_config(src) is None:
        raise SystemExit(f"{src}: not a {FORMAT_ID} package (no {CONFIG} with that format_id)")
    os.makedirs(out, exist_ok=True)
    wrote = []
    files = list(DECISION_FILES) + (list(RUNTIME_FILES) if lane in HF_FORMAT_LANES else [])
    for f in files:
        s, d = os.path.join(sd, f), os.path.join(out, f)
        if not os.path.isfile(s):
            continue
        if not os.path.isfile(d) or sha256_file(d) != sha256_file(s):
            shutil.copyfile(s, d)
            wrote.append(f)
    if lane != "gguf":
        write_manifest(src, out, lane, rung=rung, repo_id=repo_id)
        wrote.append(MANIFEST)
    for w in wrote:
        log(f"   d3 ship [{lane}] {w}")
    return wrote


def write_manifest(src, out, lane, rung=None, repo_id=None):
    """The rung's MODEL_MANIFEST.json, in the source's schema: every file of the rung hashed (so `verify="fast"` in the
    d3 runtime checks the rung, not the source's shards), the decision block and licence carried, and where it came
    from. The body is the rung's; readout and decision config are the source's bytes."""
    sd = package_dir(src)
    sm = _json(os.path.join(sd, MANIFEST)) or {}
    man = {k: v for k, v in sm.items() if k not in ("files_sha256", "files_bytes", "built_utc")}
    man["schema"] = MANIFEST_SCHEMA
    man["model_name"] = os.path.basename(os.path.abspath(out).rstrip("/"))
    man["repo_id"] = repo_id
    files = {}
    for root, dirs, fs in os.walk(out):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in fs:
            if f.startswith(".") or f == MANIFEST:
                continue
            p = os.path.join(root, f)
            files[os.path.relpath(p, out).replace(os.sep, "/")] = p
    man["files_sha256"] = {k: sha256_file(p) for k, p in sorted(files.items())}
    man["files_bytes"] = {k: os.path.getsize(p) for k, p in sorted(files.items())}
    ident = dict(sm.get("identity") or {})
    loaded = [k for k in man["files_sha256"] if re.match(r"^model.*\.safetensors$", k) or k in
              ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json") + DECISION_FILES]
    ident["files_sha256"] = {k: man["files_sha256"][k] for k in sorted(loaded)}
    ident["model_sha256"] = None                     # the source's id names the bf16 body; this body is the rung's
    man["identity"] = ident
    man["pollard"] = {"lane": lane, "rung": rung, "source_repo": sm.get("repo_id"),
                      "source_model_sha256": (sm.get("identity") or {}).get("model_sha256"),
                      "body": "quantized by Pollard", "readout": "source bytes (byte-exact)",
                      "decision_config": "source bytes (byte-exact)"}
    with open(os.path.join(out, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    return man


# ------------------------------------------------------------------------------------------------- GGUF convert

def convert_view(hf, view):
    """A converter-ready view of a d3 checkpoint: everything symlinked, config.json naming the converter's class."""
    conf = _json(os.path.join(hf, "config.json")) or {}
    arch = CONVERT_ARCH.get(conf.get("model_type"))
    if not arch:
        raise SystemExit(f"{hf}: model_type {conf.get('model_type')!r} -- the GGUF lane converts d3 on a Qwen3.5 body "
                         f"({', '.join(CONVERT_ARCH)}) only")
    os.makedirs(view, exist_ok=True)
    for f in os.listdir(hf):
        if f == "config.json":
            continue
        d = os.path.join(view, f)
        if not os.path.lexists(d):
            os.symlink(os.path.abspath(os.path.join(hf, f)), d)
    conf = dict(conf, architectures=[arch])
    json.dump(conf, open(os.path.join(view, "config.json"), "w", encoding="utf-8"), indent=1)
    return view


#: runs the converter with three d3 facts it does not know: the body's tensors are stored as `language_model.*`
#: (a bare Qwen3_5Model), the body attends non-causally, and the answer is read from the last token
_SHIM = r'''
import os, runpy, sys
conv = sys.argv.pop(1)
root = os.path.dirname(os.path.abspath(conv))
sys.path[:0] = [root, os.path.join(root, "gguf-py")]
import gguf
from conversion import base as B
_ft = B.ModelBase.filter_tensors.__func__
def _filter(cls, item):
    name, gen = item
    if name.startswith("language_model."):
        name = "model." + name
    return _ft(cls, (name, gen))
B.ModelBase.filter_tensors = classmethod(_filter)
_sgp = B.TextModel.set_gguf_parameters
def _params(self):
    _sgp(self)
    self.gguf_writer.add_causal_attention(False)
    self.gguf_writer.add_pooling_type(gguf.PoolingType.LAST)
    self.gguf_writer.add_tags(["decision-model", "%s"])
B.TextModel.set_gguf_parameters = _params
sys.argv[0] = conv
runpy.run_path(conv, run_name="__main__")
''' % FORMAT_ID


def convert_cmd(hf, outfile, converter, view=None, outtype="f16", python=None):
    """The command that converts a d3 body to GGUF (text body only; --no-mtp: the checkpoint has no MTP weights even
    where its config counts an MTP layer). Writes the view + shim next to the output."""
    out_dir = os.path.dirname(os.path.abspath(outfile)) or "."
    view = view or os.path.join(out_dir, ".pollard-d3-view")
    convert_view(hf, view)
    shim = os.path.join(out_dir, ".pollard-d3-convert.py")
    with open(shim, "w", encoding="utf-8") as f:
        f.write(_SHIM)
    return [python or sys.executable, shim, converter, view, "--outtype", outtype, "--no-mtp", "--outfile", outfile]


def mmproj_cmd(hf, outfile, converter, view=None, outtype="f16", python=None):
    """The vision tower as an mmproj next to the body (the stock Qwen3.5 mmproj path reads `visual.*` as stored)."""
    out_dir = os.path.dirname(os.path.abspath(outfile)) or "."
    view = view or os.path.join(out_dir, ".pollard-d3-view")
    convert_view(hf, view)
    return [python or sys.executable, converter, view, "--mmproj", "--outtype", outtype, "--outfile", outfile]


#: transformers' causal-LM class for the d3 body reads the bare Qwen3_5Model names as `model.*` (the tied embedding is
#: the LM head). Without it AutoModelForCausalLM loads a d3 checkpoint with EVERY body weight missing -- randomly
#: initialised, with only a warning -- and a sensitivity probe would rank the layers of a random model.
CAUSAL_KEY_MAPPING = {r"^language_model\.": "model."}


def causal_lm_kwargs(model_id):
    """from_pretrained kwargs that load a d3 body as transformers' causal LM (the sensitivity probe's view of it)."""
    return {"key_mapping": dict(CAUSAL_KEY_MAPPING)} if decision_config(model_id) is not None else {}


def llama_server_args(meta=None, ctx=8192):
    """llama-server flags for a d3 GGUF: embeddings of the last token, one ubatch per prompt (non-causal attention
    needs the whole prompt in one ubatch), and non-causal attention forced for a GGUF that does not say so."""
    arch = str((meta or {}).get("general.architecture") or "qwen35")
    return list(LLAMA_EMBED_ARGS) + ["-b", str(ctx), "-ub", str(ctx),
                                     "--override-kv", f"{arch}.attention.causal=bool:false"]


# ------------------------------------------------------------------------------------------------- readout + prompts

def read_safetensors(path):
    """{name: float32 ndarray} of a small safetensors file (the readout), numpy only."""
    import numpy as np
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            dt = v["dtype"]
            if dt == "F32":
                arr = np.frombuffer(raw, np.float32)
            elif dt == "BF16":
                arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, np.float16).astype(np.float32)
            else:
                raise ValueError(f"{path}: {k} has dtype {dt}")
            out[k] = arr.reshape(v["shape"]).astype(np.float32)
    return out


def load_readout(src):
    return read_safetensors(os.path.join(package_dir(src), READOUT))["weight"]


def softmax_t(z, T=1.0):
    m = max(z)
    e = [math.exp((v - m) / T) for v in z]
    s = sum(e)
    return [v / s for v in e]


def readout_probs(hidden, W, n, T=1.0):
    """The answer distribution over a question's n codes from the last token's final-norm hidden state:
    logits = W[:n] @ h in FP32 (the checkpoint's readout dtype), softmax at temperature T."""
    import numpy as np
    z = (np.asarray(W[:n], np.float32) @ np.asarray(hidden, np.float32)).astype(np.float64)
    return softmax_t(z.tolist(), T)


def load_runtime(src):
    """The package's own d3_runtime module (its prompt + question contract), imported from the source folder."""
    import importlib.util
    d = package_dir(src)
    path = os.path.join(d, "d3_runtime.py")
    if not os.path.isfile(path):
        raise SystemExit(f"{d}: no d3_runtime.py -- the source package is needed for the prompt contract")
    name = "d3_runtime_" + hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    if d not in sys.path:
        sys.path.insert(0, d)                      # its own `from d3_format import ...` fallback
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Prompts:
    """Questions -> token ids exactly as the d3 runtime renders them (its normalize_question + render + tokenizer)."""

    def __init__(self, src):
        from transformers import AutoTokenizer
        self.dir = package_dir(src)
        self.cfg = decision_config(src)
        self.rt = load_runtime(src)
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.codes = self.cfg["codes"]
        self.T = float(self.cfg.get("temperature", 1.0))

    def build(self, state, question):
        """(ids, option keys, kind) for one question."""
        q = self.rt.normalize_question(question)
        text = self.rt.render(self.tok, self.cfg.get("prompt", "d3"), state, q.rendered, self.codes)
        ids = self.tok([text], add_special_tokens=False)["input_ids"][0]
        return list(ids), list(q.keys), q.kind


# ------------------------------------------------------------------------------------------------- runtimes

class TorchRunner:
    """The package's runtime (transformers, SDPA, bf16 body, FP32 readout): the reference, and -- with `rung` -- an
    HF-format rung whose weights are decoded to bf16 into the same model (GPTQ / MX / compressed-tensors)."""
    kind = "reference"

    def __init__(self, src, rung=None, device="auto", batch=8, log=print):
        rt = load_runtime(src)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pollard_device as PD
        if device == "cpu":
            PD.hide_cuda()
        self.device = PD.pick(device)
        self.m = rt.D3.from_pretrained(package_dir(src), device=self.device, verify="none", batch_size=batch)
        self.batch = batch
        self.replaced = 0
        if rung:
            self.kind = "hf"
            self.replaced = load_rung_weights(self.m.backbone, rung, log=log)

    def probs(self, batch):
        """[(ids, n)] -> [[p over the n options]]"""
        out = []
        for i in range(0, len(batch), self.batch):
            chunk = batch[i:i + self.batch]
            z = self.m.logits([b[0] for b in chunk], [b[1] for b in chunk]).cpu().tolist()
            out += [softmax_t(row[:n], self.m.temperature) for row, (_, n) in zip(z, chunk)]
        return out

    def close(self):
        pass


def load_rung_weights(backbone, rung, log=print):
    """Overwrite the backbone's parameters with an HF-format rung's weights, decoded to float by pollard_stream's
    RungReader (GPTQ qweight/scales, Pollard MX atoms, compressed-tensors NVFP4 / FP8, plain tensors as stored).
    Returns how many tensors changed. A parameter the rung lacks is an error: a rung is the whole body."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_stream import RungReader
    rd = RungReader(rung)
    logical = set(getattr(getattr(rd, "src", None), "logical", {}) or {})
    changed, missing = 0, []
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            key = name if name in logical else next((k for k in (f"model.{name}",) if k in logical), None)
            if key is None:
                missing.append(name)
                continue
            arr = torch.from_numpy(rd.read(key)).reshape(p.shape).to(p.dtype)
            if not torch.equal(arr, p.data):
                changed += 1
            p.data.copy_(arr)
    if missing:
        raise SystemExit(f"{rung}: the rung lacks {len(missing)} body tensors (e.g. {missing[0]}) -- not a whole body")
    log(f"   rung weights ({rd.kind}): {changed} tensors differ from the bf16 body")
    return changed


class LlamaRunner:
    """A GGUF rung through llama-server: last-token embeddings with non-causal attention, readout applied here."""
    kind = "gguf"

    def __init__(self, gguf, src, ngl=0, ctx=8192, log=print):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pollard_bench as PB
        try:
            from pollard_calc import read_gguf_meta
            meta = read_gguf_meta(gguf)
        except Exception:                                    # noqa: BLE001
            meta = {}
        self.W = load_readout(src)
        self.T = float(decision_config(src).get("temperature", 1.0))
        self.PB = PB
        if str(ngl) == "auto":                  # Metal on a Mac; never a GPU that may be someone else's (-ngl 0)
            ngl = "99" if sys.platform == "darwin" else "0"
        self._cm = PB._served(gguf, ngl, ctx, extra=llama_server_args(meta, ctx))
        self.base = self._cm.__enter__()
        if self.base is None:
            raise SystemExit(f"llama-server would not start for {gguf} (log: {getattr(PB, '_SERVER_LOG', '?')})")

    def hidden(self, ids):
        got = self.PB._post(self.base, "/embeddings", {"content": ids, "embd_normalize": -1})
        v = got
        while isinstance(v, (list, dict)):
            if isinstance(v, dict):
                v = v.get("embedding", v.get("data"))
            elif v and isinstance(v[0], (list, dict)):
                v = v[-1]
            else:
                break
        return v

    def probs(self, batch):
        return [readout_probs(self.hidden(ids), self.W, n, self.T) for ids, n in batch]

    def close(self):
        try:
            self._cm.__exit__(None, None, None)
        except Exception:                                    # noqa: BLE001
            pass


class MlxRunner:
    """An MLX rung: mlx-lm's Qwen3.5 text layers run with no mask (full attention over the whole prompt, the Gated
    DeltaNet scan causal as always), final norm, last token, readout here."""
    kind = "mlx"

    def __init__(self, rung, src):
        from pathlib import Path
        import mlx.core as mx
        from mlx_lm.utils import load_model
        self.mx = mx
        with mlx_compat():
            self.model, _ = load_model(Path(rung), strict=False)
        self.lm = _mlx_text_model(self.model)
        self.W = load_readout(src)
        self.T = float(decision_config(src).get("temperature", 1.0))

    def hidden(self, ids):
        import numpy as np
        mx, lm = self.mx, self.lm
        h = lm.embed_tokens(mx.array([ids]))
        for layer in lm.layers:
            h = layer(h, mask=None, cache=None)
        h = lm.norm(h)[0, -1].astype(mx.float32)
        mx.eval(h)
        return np.array(h)

    def probs(self, batch):
        return [readout_probs(self.hidden(ids), self.W, n, self.T) for ids, n in batch]

    def close(self):
        pass


def mlx_compat():
    """Context manager: what mlx-lm / mlx-vlm's Qwen3.5 code needs to build a d3 body.

      * names: a d3 body is stored as a bare `Qwen3_5Model` (`language_model.*`, `visual.*`); mlx-vlm's key mapping
        knows the ForConditionalGeneration layout (`model.language_model.*`, `model.visual.*`) -- it takes both here
      * layer types: both build every layer from `full_attention_interval` and drop `layer_types` from the config.
        d3-edge is `llFllFllFFFF` (not every 4th), so its layers would be built as the wrong kind and its weights
        refused ("parameters not in model"). Here `layer_types` is kept and decides each layer, as transformers,
        vLLM and llama.cpp do
    Patches are undone on exit; a model built inside keeps the right layers."""
    import contextlib, copy, importlib

    @contextlib.contextmanager
    def patched():
        saved = []

        missing = object()

        def swap(owner, attr, new):
            saved.append((owner, attr, owner.__dict__.get(attr, missing) if isinstance(owner, type)
                          else getattr(owner, attr)))
            setattr(owner, attr, new)

        def mod(name):
            try:
                return importlib.import_module(name)
            except Exception:                                # noqa: BLE001 -- that package / family is absent
                return None

        for name in ("mlx_vlm.models.qwen3_5.qwen3_5", "mlx_vlm.models.qwen3_5_moe.qwen3_5_moe"):
            m = mod(name)
            if m is not None and hasattr(m, "sanitize_key"):
                def bare(key, _orig=m.sanitize_key):
                    if (key.startswith("language_model.") and not key.startswith("language_model.model.")) \
                            or key.startswith("visual."):
                        key = "model." + key
                    return _orig(key)
                swap(m, "sanitize_key", bare)
        for mname, args_cls, layer_cls, text_cls in (
                ("mlx_lm.models.qwen3_5", "TextModelArgs", "DecoderLayer", "Qwen3_5TextModel"),
                ("mlx_vlm.models.qwen3_5.config", "TextConfig", None, None),
                ("mlx_vlm.models.qwen3_5.language", None, "Qwen3_5DecoderLayer", "Qwen3_5Model")):
            m = mod(mname)
            if m is None:
                continue
            A = getattr(m, args_cls, None) if args_cls else None
            if A is not None and hasattr(A, "from_dict"):
                def from_dict(cls, params, _orig=A.from_dict.__func__):
                    obj = _orig(cls, params)
                    lt = params.get("layer_types") if isinstance(params, dict) else None
                    if lt:
                        obj.layer_types = list(lt)
                    return obj
                swap(A, "from_dict", classmethod(from_dict))
            if hasattr(m, "_maybe_deserialize_config"):      # mlx-vlm builds its TextConfig without from_dict
                def deser(config_cls, params, *a, _orig=m._maybe_deserialize_config, **k):
                    obj = _orig(config_cls, params, *a, **k)
                    lt = params.get("layer_types") if isinstance(params, dict) else None
                    if lt and not isinstance(obj, dict):
                        obj.layer_types = list(lt)
                    return obj
                swap(m, "_maybe_deserialize_config", deser)
            L = getattr(m, layer_cls, None) if layer_cls else None
            if L is not None:
                def init(self, args, layer_idx, *a, _orig=L.__dict__["__init__"], **k):
                    lt = getattr(args, "layer_types", None)
                    if lt and layer_idx < len(lt):
                        args = copy.copy(args)
                        args.full_attention_interval = layer_idx + 1 if lt[layer_idx] == "full_attention" else 1 << 30
                    return _orig(self, args, layer_idx, *a, **k)
                swap(L, "__init__", init)
            T = getattr(m, text_cls, None) if text_cls else None
            if T is not None:
                def tinit(self, *a, _orig=T.__dict__["__init__"], **k):
                    _orig(self, *a, **k)
                    layers = list(getattr(self, "layers", []) or [])
                    lin = [i for i, l in enumerate(layers) if getattr(l, "is_linear", False)]
                    full = [i for i, l in enumerate(layers) if not getattr(l, "is_linear", True)]
                    if lin and hasattr(self, "ssm_idx"):
                        self.ssm_idx = lin[0]
                    if full and hasattr(self, "fa_idx"):
                        self.fa_idx = full[0]
                swap(T, "__init__", tinit)
        try:
            yield
        finally:
            for owner, attr, orig in reversed(saved):
                if orig is missing:
                    delattr(owner, attr)
                else:
                    setattr(owner, attr, orig)
    return patched()


def _mlx_text_model(model):
    """The Qwen3_5TextModel (embed_tokens / layers / norm) inside an mlx-lm or mlx-vlm model."""
    seen, todo = set(), [model]
    while todo:
        m = todo.pop(0)
        if id(m) in seen:
            continue
        seen.add(id(m))
        if all(hasattr(m, a) for a in ("embed_tokens", "layers", "norm")):
            return m
        for a in ("language_model", "model"):
            if hasattr(m, a):
                todo.append(getattr(m, a))
    raise SystemExit("MLX rung: no text model with embed_tokens / layers / norm found")


class SystemOneRunner:
    """A served rung on /v1/systemone (vLLM nightly, `vllm serve <rung> ...`): its own runtime end to end."""
    kind = "systemone"

    def __init__(self, base, model=None):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pollard_bench as PB
        self.PB, self.base, self.model = PB, base.rstrip("/"), model

    def ask(self, state, question):
        body = {"state": state, "questions": {"q": question}}
        if self.model:
            body["model"] = self.model
        got = self.PB._post(self.base, "/v1/systemone", body)
        return ((got.get("answers") or {}).get("q")) or {}


class VllmRunner(SystemOneRunner):
    """An HF-format rung through `vllm serve <rung>` and its /v1/systemone (vLLM nightly, vllm-project/vllm#61565):
    the runtime the GPTQ / MX rungs are published for. Started here, measured, stopped."""
    kind = "vllm"

    def __init__(self, rung, port=0, extra=(), log=print, timeout=1800):
        import socket, subprocess, time, urllib.request
        if not port:
            s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        name = os.path.basename(str(rung).rstrip("/"))
        cmd = ["vllm", "serve", str(rung), "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name] \
            + list(extra)
        log("   " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd)
        base = f"http://127.0.0.1:{port}"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited ({self.proc.returncode}) before it was ready")
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:                                # noqa: BLE001
                time.sleep(2)
        else:
            self.close()
            raise RuntimeError("vllm serve was not ready in time")
        super().__init__(base, name)

    def close(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except Exception:                                # noqa: BLE001
                self.proc.kill()


def vllm_here():
    """vLLM importable and a CUDA device usable: the HF lanes' own runtime is on this machine."""
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        return False
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_device as PD
    return PD.best() == "cuda"


def lane_of(rung):
    """gguf | mlx | hf for a rung path."""
    p = str(rung)
    if p.endswith(".gguf"):
        return "gguf"
    conf = _json(os.path.join(p, "config.json")) or {}
    q = conf.get("quantization") or (conf.get("text_config") or {}).get("quantization")
    if isinstance(q, dict) and "group_size" in q and "quant_method" not in q:
        return "mlx"
    if os.path.isfile(os.path.join(p, "pollard-mlx.json")):
        return "mlx"
    return "hf"


def open_runner(rung, src, runtime="auto", ngl=0, ctx=8192, device="auto", systemone=None, log=print):
    if systemone:
        return SystemOneRunner(systemone)
    lane = lane_of(rung) if runtime == "auto" else runtime
    if lane in ("gguf", "llama"):
        return LlamaRunner(rung, src, ngl=ngl, ctx=ctx, log=log)
    if lane == "mlx":
        return MlxRunner(rung, src)
    if os.path.abspath(str(rung)) == os.path.abspath(package_dir(src) or ""):
        return TorchRunner(src, device=device, log=log)
    if lane == "vllm" or (lane == "hf" and runtime == "auto" and device != "cpu" and vllm_here()):
        try:
            return VllmRunner(rung, log=log)
        except Exception as e:                               # noqa: BLE001
            if lane == "vllm":
                raise SystemExit(f"vllm serve {rung}: {e}")
            log(f"   vLLM could not serve the rung ({e}) -- measuring its decoded weights through the d3 runtime")
    return TorchRunner(src, rung=rung, device=device, log=log)


# ------------------------------------------------------------------------------------------------- question sets

def builtin_items():
    """pollard-decision's 20 typed questions as d3 choice questions (gold = the intended option)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_decision import DECISION_SET
    items = []
    for i, (state, q, opts, ans) in enumerate(DECISION_SET):
        items.append({"row": f"builtin-{i}", "workflow": "builtin", "qid": "q", "state": state,
                      "question": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}},
                      "gold": {"label": opts[ans], "probabilities": {o: float(j == ans) for j, o in enumerate(opts)}}})
    return items, "pollard-decision built-ins (20)"


def typed_items(limit=0):
    """typed-decisions (Apache-2.0): 400 states x 5 typed questions, each already a /v1/systemone request."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_decision as D
    data = D.load_typed()
    rows = data["rows"]
    if limit and limit < len(rows):
        # evenly spaced states, so a smoke run still covers every workflow (the split is ordered by workflow)
        rows = [rows[int(i * len(rows) / limit)] for i in range(limit)]
    items = []
    for r in rows:
        for qid, q in r["questions"].items():
            items.append({"row": r["id"], "workflow": r["workflow"], "qid": qid, "state": r["state"],
                          "question": q, "gold": r["gold"].get(qid) or {}})
    label = f"{data['dataset']}@{(data.get('revision') or '')[:8]}" + (f" ({limit} of 400 states, evenly spaced)"
                                                                        if limit else "")
    return items, label


def question_items(qset="typed-decisions", limit=0):
    return builtin_items() if qset == "builtin" else typed_items(limit)


def _dist(keys, kind, probs):
    """{option: p} keyed the way typed-decisions' gold is: choice by key, noul true/false, score by level index."""
    return {str(k): float(p) for k, p in zip(keys, probs)}


def run_items(runner, prompts, items, log=print, every=200):
    """[{row, workflow, qid, type, dist, gold}] -- one decision per question, as the d3 runtime decides them."""
    rows = []
    if isinstance(runner, SystemOneRunner):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pollard_decision as D
        for i, it in enumerate(items):
            ans = runner.ask(it["state"], it["question"])
            rows.append({"row": it["row"], "workflow": it["workflow"], "qid": it["qid"],
                         "type": it["question"].get("type"), "dist": D._answer_dist(it["question"], ans),
                         "gold": it["gold"]})
        return rows
    built = []
    for it in items:
        ids, keys, kind = prompts.build(it["state"], it["question"])
        built.append((ids, keys, kind, it))
    step = 32
    for i in range(0, len(built), step):
        chunk = built[i:i + step]
        ps = runner.probs([(b[0], len(b[1])) for b in chunk])
        for (ids, keys, kind, it), p in zip(chunk, ps):
            rows.append({"row": it["row"], "workflow": it["workflow"], "qid": it["qid"], "type": kind,
                         "dist": _dist(keys, kind, p), "gold": it["gold"], "tokens": len(ids)})
        if log and every and (i // step) % max(1, every // step) == 0:
            log(f"      {min(i + step, len(built))}/{len(built)} decisions")
    return rows


# ------------------------------------------------------------------------------------------------- the board

def board(rows, ref_rows, label=""):
    """pollard-decision's typed board (agreement, option KL, drift, calibration + its change from the reference),
    plus the split by question type and the readout facts."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_decision as D
    b = D.score_typed(rows, ref_rows)
    b["set"] = label or b.get("set")
    if ref_rows:
        am = lambda d: max(d, key=d.get) if d else None
        by = {}
        for r, s in zip(rows, ref_rows):
            t = by.setdefault(r.get("type") or "?", [0, 0, 0.0])
            t[0] += 1
            t[1] += am(r["dist"]) == am(s["dist"])
            t[2] += D._dist_kl(s["dist"], r["dist"])
        b["by_type"] = {k: {"n": v[0], "agreement": v[1] / v[0], "option_kl": v[2] / v[0]} for k, v in sorted(by.items())}
        kls = sorted(D._dist_kl(s["dist"], r["dist"]) for r, s in zip(rows, ref_rows))
        b["option_kl_p95"] = kls[min(len(kls) - 1, int(0.95 * len(kls)))] if kls else 0.0
        b["option_kl_max"] = kls[-1] if kls else 0.0
    return b


def _ref_cache_path(src, qset, limit):
    home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
    man = _json(os.path.join(package_dir(src), MANIFEST)) or {}
    ident = (man.get("identity") or {}).get("model_sha256") or \
        hashlib.sha256(os.path.abspath(package_dir(src)).encode()).hexdigest()
    return os.path.join(home, "cache", "d3-ref", f"{ident[:16]}-{qset}-{limit or 'all'}.json")


def reference_rows(src, items, qset, limit, prompts, device="auto", cache=None, log=print):
    """The bf16 reference through the package's runtime, computed once per (checkpoint, set) and cached."""
    cache = cache or _ref_cache_path(src, qset, limit)
    if os.path.isfile(cache):
        got = _json(cache) or {}
        if got.get("n") == len(items):
            log(f"   reference: cached bf16 readout ({cache})")
            return got["rows"]
    runner = TorchRunner(src, device=device, log=log)
    log(f"   reference: bf16 through the d3 runtime ({getattr(runner, 'device', device)}) -- {len(items)} decisions")
    rows = run_items(runner, prompts, items, log=log)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"n": len(items), "set": qset, "limit": limit, "rows": rows}, open(cache, "w", encoding="utf-8"))
    return rows


def gate(rung, src, runtime="auto", qset="typed-decisions", limit=0, ngl=0, ctx=8192, device="auto",
         systemone=None, ref_cache=None, log=print):
    """Board + verdict for one rung against the bf16 reference, in pollard-decision's result shape."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_decision as D
    if decision_config(src) is None:
        raise SystemExit(f"{src}: not a {FORMAT_ID} package -- for first-position letter models use pollard-decision")
    items, label = question_items(qset, limit)
    prompts = Prompts(src)
    ref_rows = reference_rows(src, items, qset, limit, prompts, device=device, cache=ref_cache, log=log)
    lane = "systemone" if systemone else (lane_of(rung) if runtime == "auto" else runtime)
    log(f"   rung: {rung}  [{LANE_RUNTIME.get(lane, lane)}]")
    runner = open_runner(rung, src, runtime=runtime, ngl=ngl, ctx=ctx, device=device, systemone=systemone, log=log)
    try:
        rows = run_items(runner, prompts, items, log=log)
    finally:
        runner.close()
    b = board(rows, ref_rows, label)
    used = getattr(runner, "kind", lane)
    b.update({"format": FORMAT_ID, "runtime": LANE_RUNTIME.get(used, used), "reference": LANE_RUNTIME["reference"],
              "readout": "readout.safetensors (FP32, byte-exact)"})
    if getattr(runner, "kind", None) == "hf":
        b["tensors_decoded"] = runner.replaced
    v, why = D.verdict(b)
    return {"verdict": v, "config": "decision", "reason": why, "board": b,
            "rows": D._typed_rows_out(rows, ref_rows), "sampling": ["readout", f"T={prompts.T:g}"]}


#: the d3 gate's bar -- pollard-decision's PASS rule on the readout distribution
GATE_AGREEMENT, GATE_KL = 0.9, 0.05


def gate_record(res, rung, lane):
    """The gate verdict in pollard-verify's shape ({pass, checks, gates, ...}) for the workspace MANIFEST."""
    b = res.get("board") or {}
    checks = [{"check": "d3 agreement with bf16", "metric": "agreement", "value": b.get("agreement"), "op": ">=",
               "threshold": GATE_AGREEMENT, "ok": (b.get("agreement") or 0) >= GATE_AGREEMENT},
              {"check": "d3 option KL vs bf16", "metric": "option_kl", "value": b.get("option_kl"), "op": "<=",
               "threshold": GATE_KL, "ok": b.get("option_kl") is not None and b["option_kl"] <= GATE_KL}]
    return {"pass": res.get("verdict") == "PASS", "checks": checks, "gates": "pollard-d3", "rung": rung,
            "lane": lane, "verdict": res.get("verdict"), "reason": res.get("reason")}


def board_line(b):
    line = (f"agreement {b['agreement']:.1%}  option KL {b['option_kl']:.5f} (p95 {b.get('option_kl_p95', 0):.5f}, "
            f"max {b.get('option_kl_max', 0):.4f})  drift {b['mean_abs_drift']:.4f}  | gold acc {b['accuracy']:.1%} "
            f"(ref {b['ref_accuracy']:.1%})")
    if "ece" in b:
        line += f"\n  calibration: ECE {b['ece']:.4f} (ref {b.get('ref_ece', 0):.4f}, dECE {b.get('delta_ece', 0):+.4f})" \
                f"  Brier {b['brier']:.4f}  NLL {b['nll']:.4f}  flips {b.get('flips', 0)}/{b['n']}"
    for t, v in (b.get("by_type") or {}).items():
        line += f"\n  {t:7s} n={v['n']:<5d} agreement {v['agreement']:.1%}  option KL {v['option_kl']:.5f}"
    return line


# ------------------------------------------------------------------------------------------------- /v1/systemone

def systemone_response(model_name, answers, tokens):
    """vLLM's /v1/systemone response shape."""
    return {"object": "structured_decision", "model": model_name, "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0}}


def answer_obj(rt, question, probs):
    """The d3 runtime's own answer object (choice / noul / score, confidence) for one question's probabilities."""
    q = rt.normalize_question(question)
    return rt.product_answer(q, probs)


def serve(rung, src, port=8000, host="127.0.0.1", runtime="auto", ngl=0, ctx=8192, device="auto", log=print):
    """A /v1/systemone server for any rung (GGUF / MLX / HF), so one client -- and the decision-index http engine --
    reaches every lane the same way. Text only; one question per forward pass, as the d3 runtime decides them."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    prompts = Prompts(src)
    runner = open_runner(rung, src, runtime=runtime, ngl=ngl, ctx=ctx, device=device, log=log)
    name = os.path.basename(str(rung).rstrip("/"))
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self._send(200, {"status": "ok"})
            if self.path.startswith("/v1/models"):
                return self._send(200, {"object": "list", "data": [{"id": name, "object": "model", "architecture":
                                                                    {"output_modalities": ["decisions"]}}]})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/systemone"):
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._send(400, {"error": "invalid JSON"})
            if "instructions" in req:
                return self._send(400, {"error": "top-level instructions are not part of a System One request"})
            if req.get("images") or req.get("videos"):
                return self._send(400, {"error": "this server answers text requests; images/videos need the d3 "
                                                 "runtime or vLLM"})
            qs = req.get("questions")
            if not isinstance(qs, dict) or not qs:
                return self._send(400, {"error": "questions must be a nonempty mapping"})
            answers, tokens, batch, keys = {}, 0, [], []
            for qid, q in qs.items():
                try:
                    ids, opts, _ = prompts.build(req.get("state"), q)
                except ValueError as e:
                    answers[qid] = {"type": (q or {}).get("type") if isinstance(q, dict) else None,
                                    "error": "invalid_question", "message": str(e)}
                    continue
                batch.append((ids, len(opts)))
                keys.append(qid)
                tokens += len(ids)
            with lock:
                ps = runner.probs(batch) if batch else []
            for qid, p in zip(keys, ps):
                answers[qid] = answer_obj(prompts.rt, qs[qid], p)
            self._send(200, systemone_response(req.get("model") or name, {k: answers[k] for k in qs if k in answers},
                                               tokens))

    srv = ThreadingHTTPServer((host, port), H)
    log(f"   /v1/systemone for {name} [{LANE_RUNTIME.get(lane_of(rung), '?')}] on http://{host}:{port}")
    try:
        srv.serve_forever()
    finally:
        runner.close()


# ------------------------------------------------------------------------------------------------- Jev Index (opt-in)

def jev_index_cmd(kit_python, base_url, model_name, out, limit=0, edition=None):
    """apolinario/decision-index (MIT) against a /v1/systemone server: the Decision Index public part (0.3)."""
    cmd = [kit_python, "-m", "decision_index", "pipeline", "--engine", "http", "--option", f"base_url={base_url}",
           "--option", f"model={model_name}", "--out", out]
    if limit:
        cmd += ["--limit", str(limit)]
    if edition:
        cmd += ["--edition", str(edition)]
    return cmd


def read_index(out):
    """The public index from a decision-index run directory (scores.json, else index.json)."""
    for f in ("scores.json", "index.json"):
        d = _json(os.path.join(out, f))
        if isinstance(d, dict):
            v = d.get("index")
            if isinstance(v, dict):
                v = v.get("index")
            if isinstance(v, (int, float)):
                return float(v)
    return None


def jev_index(rung, src, kit_python, out, port=0, limit=0, runtime="auto", ngl=0, ctx=8192, device="auto",
              results=None, rung_name=None, log=print):
    """Serve the rung on /v1/systemone, run the kit's pipeline through its http engine, read the index back. Hours on
    the full suite: an overnight step, never part of a publish."""
    import socket, subprocess, time, urllib.request
    if not port:
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    me = [sys.executable, os.path.abspath(__file__), "serve", "--rung", rung, "--source", src, "--port", str(port),
          "--runtime", runtime, "--ngl", str(ngl), "--ctx", str(ctx), "--device", device]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_bench as PB
    if PB.SERVER_BIN:
        me += ["--llama-server", PB.SERVER_BIN]
    srv = subprocess.Popen(me)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(900):
            if srv.poll() is not None:
                raise SystemExit("the /v1/systemone server exited before it was ready")
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:                                # noqa: BLE001
                time.sleep(1)
        name = rung_name or os.path.basename(str(rung).rstrip("/"))
        cmd = jev_index_cmd(kit_python, base, name, out, limit)
        log("   " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        idx = read_index(out)
        if results and idx is not None:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import pollard_workspace as ws
            ws.merge_results(results, name, {"decision": {"jev_index": idx, "jev_index_limit": limit or None,
                                                          "jev_index_run": os.path.abspath(out)}})
        return rc, idx
    finally:
        srv.terminate()


# ------------------------------------------------------------------------------------------------- card text

def serve_lines(info_, repo=None, lanes=()):
    """How a d3 rung is served, per lane, for the card."""
    src = (info_ or {}).get("repo_id") or "vllm-sr/d3"
    repo = repo or "<this repo>"
    out = ["### Serving (d3 decision model: `/v1/systemone`)", "",
           f"d3 answers typed questions with a probability per option through its {FORMAT_ID} readout "
           "(non-causal body, last-token pooling, FP32 `readout.safetensors` over 255 answer codes). It never "
           "generates text. Every rung here ships `readout.safetensors` and `decision_config.json` byte-exact; only "
           "the body is quantized.", ""]
    if not lanes or any(l in lanes for l in ("gptq", "mx")):
        out += ["vLLM (nightly; vllm-project/vllm#61565 serves a bare `Qwen3_5Model` + `decision_config.json` as "
                "`D3ForSequenceClassification`, the readout as its FP32 score head, on `/v1/systemone`):", "", "```sh",
                f"vllm serve {src}                       # the bf16 source",
                f"hf download {repo} --include \"GPTQ/<rung>/*\" --local-dir .",
                "vllm serve ./GPTQ/<rung> --max-model-len 32768   # a rung folder (MX/<rung> the same way)",
                "curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{",
                '  "state": "I was charged twice for my order.",',
                '  "questions": {"team": {"type": "choice", "instructions": "Which team?",',
                '                         "criteria": {"billing": null, "shipping": null, "technical": null}}}}\'',
                "```", ""]
    if not lanes or "gguf" in lanes:
        out += ["llama.cpp (any GGUF rung): llama-server returns the last token's hidden state; the readout is one "
                "matrix-vector product:", "", "```sh",
                "llama-server -m <rung>.gguf --embeddings --pooling last --embd-normalize -1 -b 8192 -ub 8192",
                "pollard-d3 serve --rung <rung>.gguf --source . --port 8000   # the same /v1/systemone over it",
                "```", ""]
    if not lanes or "mlx" in lanes:
        out += ["MLX (Apple silicon):", "", "```sh", "pollard-d3 serve --rung ./MLX/<rung> --source . --port 8000",
                "```", ""]
    out += ["transformers (the package's own runtime, any HF-format rung):", "", "```python",
            "from transformers import AutoModel",
            f"m = AutoModel.from_pretrained(\"{src}\", trust_remote_code=True)",
            "m.system_one(state=\"...\", questions={\"q\": {\"type\": \"noul\", \"instructions\": \"Is it urgent?\"}})",
            "```", ""]
    return out


# ------------------------------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard-d3", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="is this a d3 package; its readout facts")
    p.add_argument("--model", required=True)
    p = sub.add_parser("convert", help="the d3 body -> GGUF (non-causal attention + last pooling in its metadata)")
    p.add_argument("--hf", required=True)
    p.add_argument("--outfile", required=True)
    p.add_argument("--outtype", default="f16")
    p.add_argument("--converter", help="convert_hf_to_gguf.py (default: pollard-convert's pick)")
    p.add_argument("--mmproj", help="also write the vision tower here (mmproj GGUF)")
    p.add_argument("--print", action="store_true", help="print the command, do not run it")
    p = sub.add_parser("ship", help="decision files byte-exact + the rung's own MODEL_MANIFEST.json")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lane", required=True, choices=["gguf", "mlx", "gptq", "mx", "exl3", "exlr8", "hf"])
    p.add_argument("--rung")
    p.add_argument("--repo-id")
    for name in ("gate", "serve", "jev-index"):
        p = sub.add_parser(name)
        p.add_argument("--rung", help="a GGUF, an MLX folder or an HF-format rung folder (the source = the reference)")
        p.add_argument("--source", required=True, help="the d3 package (bf16 checkpoint with its runtime)")
        p.add_argument("--runtime", default="auto", choices=["auto", "gguf", "mlx", "hf", "vllm"],
                       help="auto: GGUF -> llama-server, MLX -> mlx-lm, an HF-format rung -> vllm serve where vLLM and "
                            "a CUDA device are here, else the d3 runtime on its decoded weights")
        p.add_argument("--ngl", default="0", help="GPU layers for llama-server (a number or auto)")
        p.add_argument("--ctx", type=int, default=8192)
        p.add_argument("--device", default="auto", help="torch device for the bf16 reference / HF rungs: auto (CUDA, "
                       "MPS on Apple silicon, else CPU) | cpu | cuda | mps -- the bf16 readout agrees across them "
                       "(MPS vs CPU on d3-edge: option KL 4e-5)")
        p.add_argument("--llama-server", help="the llama-server binary for a GGUF rung (default: pollard-bench's pick)")
        if name == "gate":
            p.add_argument("--systemone", help="measure a SERVED rung at this base URL (vLLM /v1/systemone)")
            p.add_argument("--set", dest="qset", default="typed-decisions", choices=["typed-decisions", "builtin"])
            p.add_argument("--limit", type=int, default=0, help="typed-decisions: N of the 400 states, evenly spaced "
                           "across the workflows (a smoke run; the board is comparable at the full set)")
            p.add_argument("--ref-cache", help="where the bf16 reference rows are cached")
            p.add_argument("--results", help="merge the board into this results.json (pollard-card --results)")
            p.add_argument("--rung-name", help="the rung's name in --results")
            p.add_argument("--json", action="store_true")
        if name == "serve":
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--host", default="127.0.0.1")
        if name == "jev-index":
            p.add_argument("--kit-python", required=True, help="the python that has apolinario/decision-index installed")
            p.add_argument("--out", required=True)
            p.add_argument("--limit", type=int, default=0, help="first N requests only (a smoke run, not an index)")
            p.add_argument("--results")
            p.add_argument("--rung-name")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        i = info(a.model)
        if i is None:
            print(f"{a.model}: not a {FORMAT_ID} package")
            return 1
        print(json.dumps(i, indent=1))
        return 0
    if a.cmd == "convert":
        conv = a.converter
        if not conv:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from pollard_convert import find_converter
            conv, note = find_converter(convert_view(a.hf, os.path.join(os.path.dirname(os.path.abspath(a.outfile)),
                                                                         ".pollard-d3-view")))
            if conv is None:
                raise SystemExit(note)
        cmds = [convert_cmd(a.hf, a.outfile, conv, outtype=a.outtype)]
        if a.mmproj:
            cmds.append(mmproj_cmd(a.hf, a.mmproj, conv, outtype=a.outtype))
        import subprocess
        for cmd in cmds:
            print("   " + " ".join(cmd))
            if not a.print and subprocess.run(cmd).returncode != 0:
                return 1
        if not a.print:
            ship(a.hf, os.path.dirname(os.path.abspath(a.outfile)), "gguf")
        return 0
    if a.cmd == "ship":
        ship(a.source, a.out, a.lane, rung=a.rung, repo_id=a.repo_id)
        return 0
    rung = a.rung or a.source
    if a.llama_server:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pollard_bench as PB
        PB.SERVER_BIN = a.llama_server
    if a.cmd == "serve":
        serve(rung, a.source, port=a.port, host=a.host, runtime=a.runtime, ngl=a.ngl, ctx=a.ctx, device=a.device)
        return 0
    if a.cmd == "jev-index":
        rc, idx = jev_index(rung, a.source, a.kit_python, a.out, limit=a.limit, runtime=a.runtime, ngl=a.ngl,
                            ctx=a.ctx, device=a.device, results=a.results, rung_name=a.rung_name)
        print(f"   Decision Index (public): {idx if idx is not None else 'not produced'}")
        return rc
    res = gate(rung, a.source, runtime=a.runtime, qset=a.qset, limit=a.limit, ngl=a.ngl, ctx=a.ctx, device=a.device,
               systemone=a.systemone, ref_cache=a.ref_cache)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pollard_workspace as ws
    name = a.rung_name or os.path.basename(str(rung).rstrip("/"))
    g = gate_record(res, name, lane_of(rung) if a.runtime == "auto" else a.runtime)
    if a.results and res.get("board"):
        b = {k: v for k, v in res["board"].items() if not isinstance(v, list)}
        b["verdict"] = res["verdict"]
        ws.merge_results(a.results, name, {"decision": b, "verified": {"pass": g["pass"], "gates": g["gates"],
                                                                      "failed": [c["check"] for c in g["checks"]
                                                                                 if not c["ok"]]}},
                         meta={"_decision_ref": "bf16"})
    if a.rung and not a.systemone:
        try:                                   # the MANIFEST verdict pollard-ls / release resume / the card read
            ws.record_gate(rung, g)
        except Exception:                      # noqa: BLE001 -- a rung built outside the workspace
            pass
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res["rows"]:
            print(f"  [{'ok  ' if not r['loop'] else 'BAD '}] {r['prompt']:<48} {r['reason']:<14} {r['sample']}")
        print("  " + board_line(res["board"]))
        print(f"VERDICT: {res['verdict']} -- {res.get('reason', '')}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
