#!/usr/bin/env python3
"""pollard-stream -- the layer-streaming engine every transformers lane builds on (smooth / GPTQ / MX / EXLR8).

    pollard-stream plan --model <hf dir> [--trust-remote-code on] [--nsamples 128] [--max-ram-gb 24]

A lane that needs activations (SmoothQuant's per-channel maxima, GPTQ's Hessians, NVFP4's input scales, EXLR8's
LDLQ) used to load the WHOLE model through transformers. That wall stopped three kinds of model:

  * bigger than RAM: LFM2.5-24B-A2B is 48 GB of bf16 on a 32 GB box;
  * custom code (config.json auto_map): NVIDIA's Nemotron-3.5-Bolt ships its own modelling files;
  * omni / multimodal with no causal-LM class: Qwen3-Omni-MoE (clef-omni) is a thinker + talker + towers.

What this does instead, for any of them:

  1. RESOLVE THE TEXT BACKBONE from config.json, never by running code to decide: a native transformers class for
     the checkpoint (causal LM, VLM, omni -- built whole, ON META, so it costs no memory), else a native class for
     its nested language config (`llm_config` / `text_config`), with `*_puzzle` per-layer block configs mapped onto
     the native blocks, else the checkpoint's own code -- only with consent (see CONSENT). The decoder is the
     biggest stack of repeated blocks; vision / audio towers, talkers and projectors are never touched and are
     copied through at source precision by every lane.
  2. READ TENSORS ONE AT A TIME from the safetensors files (no full load, no mmap of the model), dequantizing on
     read when the source is already quantized: modelopt NVFP4 (W4A16 / W4A4), modelopt FP8, compressed-tensors
     NVFP4, ggml-packed blocks inside safetensors (NVIDIA's IQ2_XXS / Q8_0 experts, via gguf-py), block FP8.
     The encoded bytes stay available for a byte-exact passthrough.
  3. RECORD each decoder block's call (position embeddings, the per-layer-type masks, ...) with the blocks
     replaced by recorders, then run the calibration set through ONE block at a time: its weights loaded from
     disk, the lane's statistics captured, the lane's transform applied, the block's outputs kept as the next
     block's inputs, the block freed. Peak memory = one block + its statistics + the activations, preflighted.
  4. MEMORY PREFLIGHT before anything starts (plan_memory): the peak is computed from the shapes on disk and the
     calibration size and checked against free RAM (and --max-ram-gb). Too big -> fewer calibration samples, then
     MoE Hessians in expert chunks, then a refusal with the numbers. Free RAM is re-checked before every block
     load; the run stops cleanly instead of swapping the machine to death. The measured peak RSS is reported.

CONSENT for custom code: a checkpoint whose config names an auto_map runs Python files from the checkpoint. That
never happens silently: `--trust-remote-code on` (or `trust_remote_code: true` in a release plan) runs them and
names every file; `auto` asks y/N on a terminal and refuses (naming the files and the flag) everywhere else;
`off` never runs them. A model with a native mapping (Bolt through transformers' own NemotronH) runs no
checkpoint code at all, consent or not.
"""
from __future__ import annotations

import contextlib, json, os, re, struct, sys, time

# ----------------------------------------------------------------------------------------------- config / consent

AUTO_MAP_KEYS = ("auto_map",)
NESTED_CONFIGS = ("llm_config", "text_config", "language_config", "thinker_config", "talker_config",
                  "vision_config", "audio_config", "sound_config")


class RemoteCodeRefused(SystemExit):
    """The checkpoint needs its own Python code and nobody consented to running it."""


class Unresolvable(SystemExit):
    """No way to build this checkpoint's text backbone here -- the message is the exact reason."""


def config_json(model_dir):
    p = os.path.join(model_dir, "config.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _auto_map_entries(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in AUTO_MAP_KEYS and isinstance(v, dict):
                for ref in v.values():
                    for r in (ref if isinstance(ref, (list, tuple)) else [ref]):
                        if isinstance(r, str):
                            out.add(r)
            elif isinstance(v, (dict, list)):
                _auto_map_entries(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _auto_map_entries(v, out)
    return out


def auto_map_refs(model_dir):
    """Every `module.Class` an auto_map names (config.json, nested configs, tokenizer / processor configs)."""
    refs = set()
    for fn in ("config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
        p = os.path.join(model_dir, fn)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    _auto_map_entries(json.load(f), refs)
            except (OSError, ValueError):
                pass
    return sorted(refs)


def needs_remote_code(model_dir):
    return bool(auto_map_refs(model_dir))


def remote_code_files(model_dir):
    """The .py files transformers would import for this checkpoint: every module an auto_map names, plus every
    .py beside them (a modelling file imports its siblings -- Bolt's omni file pulls in five)."""
    mods = {r.split("--")[-1].rsplit(".", 1)[0] + ".py" for r in auto_map_refs(model_dir)}
    present = sorted(f for f in os.listdir(model_dir) if f.endswith(".py")) if os.path.isdir(model_dir) else []
    return sorted(set(present) | {m for m in mods if m in present}), sorted(m for m in mods if m not in present)


def consent(model_dir, mode="auto", log=print, ask=None):
    """True when the checkpoint's own code may run. Never silent: names every file it would run.

    mode: 'on' (consent given: --trust-remote-code on / plan trust_remote_code: true), 'off' (never), 'auto'
    (ask on a terminal; elsewhere refuse with the files and the flag that consents). POLLARD_TRUST_REMOTE_CODE=1
    counts as consent given."""
    mode = {True: "on", False: "off", None: "auto"}.get(mode, mode) if not isinstance(mode, str) else mode
    mode = str(mode).lower()
    if mode in ("1", "yes", "true"):
        mode = "on"
    if mode in ("0", "no", "false"):
        mode = "off"
    refs = auto_map_refs(model_dir) if os.path.isdir(model_dir) else []
    if not refs:
        return False
    files, missing = remote_code_files(model_dir)
    listing = ", ".join(files) or "(none on disk)"
    if missing:
        listing += f"  [named but missing: {', '.join(missing)}]"
    if mode == "off":
        log(f"   [remote code] {model_dir} ships its own code ({listing}) -- NOT running it (--trust-remote-code off)")
        return False
    if mode != "on" and os.environ.get("POLLARD_TRUST_REMOTE_CODE", "").lower() in ("1", "on", "yes", "true"):
        mode = "on"
    if mode == "on":
        log(f"   [remote code] running the checkpoint's own code (consent given): {listing}")
        return True
    question = (f"   {model_dir} needs its own Python code to load: {listing}\n"
                f"   run it? [y/N] ")
    if ask is None and sys.stdin is not None and sys.stdin.isatty():
        ask = input
    if ask is not None:
        try:
            ans = ask(question)
        except EOFError:
            ans = ""
        if str(ans).strip().lower() in ("y", "yes"):
            log(f"   [remote code] consent given at the prompt: {listing}")
            return True
        raise RemoteCodeRefused(f"not running the checkpoint's code ({listing}): consent declined")
    raise RemoteCodeRefused(
        f"{model_dir} needs its own Python code to load ({listing}). Pollard never runs it silently: pass "
        "--trust-remote-code on (release plan: trust_remote_code: true) to run those files, or --trust-remote-code "
        "off to skip the lanes that need it.")


# ----------------------------------------------------------------------------------------------- safetensors source

ST_NP = {"F32": "float32", "F16": "float16", "BF16": "uint16", "U8": "uint8", "I8": "int8", "F8_E4M3": "uint8",
         "F8_E5M2": "uint8", "I32": "int32", "I64": "int64", "F64": "float64", "BOOL": "bool", "U16": "uint16",
         "I16": "int16", "U32": "uint32", "U64": "uint64"}
FLOAT_DT = ("F32", "F16", "BF16", "F64")
#: auxiliary tensors of an encoded linear (hidden behind the module's logical `.weight`)
AUX_SUFFIXES = ("weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale", "weight_global_scale",
                "input_global_scale", "weight_packed", "weight_zero_point", "weight_shape", "k_scale", "v_scale")
NVFP4_BLOCK = 16
E4M3_MAX = 448.0


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def _e4m3_to_f32(b):
    import numpy as np
    b = np.asarray(b, np.uint8)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    v = np.where((e == 15) & ((b & 7) == 7), np.nan, v)
    return (s * v).astype(np.float32)


def _to_f32(dt, arr):
    import numpy as np
    if dt == "BF16":
        return (arr.astype(np.uint32) << 16).view(np.float32)
    if dt in ("F32", "F16", "F64"):
        return arr.astype(np.float32)
    if dt == "F8_E4M3":
        return _e4m3_to_f32(arr)
    raise ValueError(f"not a float tensor: {dt}")


def _mx_scales(entries, keys, k_in):
    """True when the module's `weight_scale` is an E8M0 MX scale: uint8, one byte per 32 of the `k_in` inputs."""
    sk = keys.get("weight_scale")
    if not sk or sk not in entries or not k_in:
        return False
    _, dt, shp, _, _ = entries[sk]
    return dt == "U8" and len(shp) == 2 and k_in % 32 == 0 and shp[-1] == k_in // 32


def weight_files(model_dir):
    """The checkpoint's weight shards: the index's files when there is one (so a side file such as clef-omni's
    joint_head.safetensors is NOT a model shard), else every *.safetensors."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f).get("weight_map") or {}
        return sorted({os.path.join(model_dir, v) for v in wm.values()})
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    local = {os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")}
    # a rung whose shards were uploaded + freed as they were written: they are still its shards (read from the Hub)
    local |= {os.path.join(model_dir, fn) for fn in read_streamed(model_dir)}
    return sorted(local)


class Source:
    """Tensor-at-a-time reader over an HF safetensors checkpoint, dequantizing encoded linears on read.

    `entries`  raw tensor name -> (path, dtype, shape, data offset, nbytes)
    `modules`  encoded linear module -> {"algo": ..., "keys": {suffix: raw name}, ...}
    `logical`  the float view: every raw tensor that is not an encoded linear's part, plus `<module>.weight` for
               each encoded one (shape = the dequantized [out, in])."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.base_dir = model_dir
        self.overlay = {}
        ov = read_overlay(model_dir)
        if ov:
            self.base_dir = ov["base"]
            self.overlay = ov["ops"]
        self.files = weight_files(self.base_dir)
        if not self.files:
            raise SystemExit(f"no safetensors weights in {self.base_dir}")
        # a sparse remote mirror (pollard_mlx.RemoteMirror): its headers are real, its tensor bytes are holes until
        # the MLX lane fetches them -- every other reader would read zeros
        self.mirror = os.path.isfile(os.path.join(self.base_dir, ".pollard-remote"))
        self.entries, self._hdr = {}, {}
        streamed = read_streamed(self.base_dir)
        self._remote = {}                                 # path -> streamed entry: shards that live on the Hub now
        for p in self.files:
            ent = streamed.get(os.path.basename(p)) if not os.path.exists(p) else None
            if ent is not None:
                self._remote[p] = ent
                hdr, base = ent["header"], ent["data_start"]
            else:
                hdr, base = st_header(p)
            self._hdr[p] = (hdr, base)
            for name, info in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = info["data_offsets"]
                self.entries[name] = (p, info["dtype"], tuple(info["shape"]), base + a, b - a)
        try:
            self.config = config_json(model_dir)
        except (OSError, ValueError):
            self.config = {}
        self.qcfg = self._quant_config()
        self.modules = self._encoded_modules()
        self.logical = self._logical()

    # -- quantization metadata
    def _quant_config(self):
        q = dict(self.config.get("quantization_config") or {})
        hq = os.path.join(self.dir, "hf_quant_config.json")
        if not q and os.path.isfile(hq):
            try:
                with open(hq, encoding="utf-8") as f:
                    d = json.load(f)
                q = dict(d.get("quantization") or d)
                q.setdefault("quant_method", "modelopt")
            except (OSError, ValueError):
                pass
        return q

    def _encoded_modules(self):
        mods = {}
        layers = self.qcfg.get("quantized_layers") or {}
        algo_all = str(self.qcfg.get("quant_algo") or "").upper()
        for name, (p, dt, shape, off, n) in self.entries.items():
            if not name.endswith(".weight"):
                continue
            mod = name[:-len(".weight")]
            spec = layers.get(mod) or {}
            algo = str(spec.get("quant_algo") or "").upper()
            keys = {s: f"{mod}.{s}" for s in AUX_SUFFIXES if f"{mod}.{s}" in self.entries}
            if spec.get("packing") == "ggml" or (algo and algo not in ("NVFP4", "W4A16_NVFP4", "FP8", "W4A8_NVFP4_FP8")
                                                 and dt == "U8" and len(shape) == 3):
                mods[mod] = {"algo": "ggml:" + algo, "keys": {"weight": name}, "spec": spec}
            elif dt == "U8" and "weight_scale" in keys and ("weight_scale_2" in keys or algo.endswith("NVFP4")):
                mods[mod] = {"algo": "nvfp4", "flavour": "modelopt", "keys": {"weight": name, **keys},
                             "a4": "input_scale" in keys, "spec": spec}
            elif dt == "F8_E4M3" and _mx_scales(self.entries, keys, shape[-1] if shape else 0):
                # compressed-tensors MXFP8: E4M3 elements, one E8M0 byte per 32 along the input dim
                mods[mod] = {"algo": "mxfp8", "flavour": "compressed-tensors", "keys": {"weight": name, **keys},
                             "spec": spec}
            elif dt == "F8_E4M3" and "weight_scale_inv" in keys:
                mods[mod] = {"algo": "fp8-block", "keys": {"weight": name, **keys}, "spec": spec}
            elif dt == "F8_E4M3":
                mods[mod] = {"algo": "fp8", "keys": {"weight": name, **keys}, "spec": spec}
            elif algo_all and dt == "U8" and len(shape) == 2 and "weight_scale" in keys:
                mods[mod] = {"algo": "nvfp4", "flavour": "modelopt", "keys": {"weight": name, **keys}, "spec": spec}
        for name, (p, dt, shape, off, n) in self.entries.items():         # compressed-tensors NVFP4
            if name.endswith(".weight_packed") and f"{name[:-7]}_global_scale" in self.entries:
                mod = name[:-len(".weight_packed")]
                keys = {s: f"{mod}.{s}" for s in AUX_SUFFIXES if f"{mod}.{s}" in self.entries}
                mods[mod] = {"algo": "nvfp4", "flavour": "compressed-tensors", "keys": keys,
                             "a4": "input_global_scale" in keys}
            elif name.endswith(".weight_packed") and name[:-len(".weight_packed")] not in mods:
                # compressed-tensors MXFP4 (format mxfp4-pack-quantized, llm-compressor's MXFP4 / MXFP4A16): the same
                # two-per-byte E2M1 packing as NVFP4, but one E8M0 scale byte per 32 inputs and NO global scale --
                # w = fp4 * 2^(weight_scale - 127)
                mod = name[:-len(".weight_packed")]
                keys = {s: f"{mod}.{s}" for s in AUX_SUFFIXES if f"{mod}.{s}" in self.entries}
                if dt == "U8" and len(shape) == 2 and _mx_scales(self.entries, keys, shape[1] * 2):
                    mods[mod] = {"algo": "mxfp4", "flavour": "compressed-tensors", "keys": keys,
                                 "a4": "input_global_scale" in keys}
        return mods

    def _logical(self):
        hidden = set()
        out = {}
        for mod, m in self.modules.items():
            hidden.update(m["keys"].values())
            out[mod + ".weight"] = ("encoded", mod)
        for name in self.entries:
            if name not in hidden:
                out.setdefault(name, ("raw", name))
        return out

    # -- shapes / dtypes
    def shape(self, key):
        kind, ref = self.logical[key]
        if kind == "raw":
            return self.entries[ref][2]
        m = self.modules[ref]
        if m["algo"] in ("nvfp4", "mxfp4"):
            p, dt, shp, off, n = self.entries[m["keys"].get("weight", m["keys"].get("weight_packed"))]
            return (shp[0], shp[1] * 2)
        if m["algo"].startswith("ggml:"):
            from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as Q
            bs, bb = GGML_QUANT_SIZES[Q[m["algo"][5:]]]
            shp = self.entries[m["keys"]["weight"]][2]
            rows = int(_prod(shp[:-2])) if len(shp) > 2 else shp[0]
            nbytes = shp[-1] * (shp[-2] if len(shp) > 2 else 1)
            return (rows, nbytes // bb * bs)
        return self.entries[m["keys"]["weight"]][2]

    def dtype(self, key):
        """The source storage dtype of a logical tensor ('BF16', ..., or the encoding: 'nvfp4' / 'ggml:IQ2_XXS')."""
        kind, ref = self.logical[key]
        return self.entries[ref][1] if kind == "raw" else self.modules[ref]["algo"]

    def nbytes(self, key):
        kind, ref = self.logical[key]
        if kind == "raw":
            return self.entries[ref][4]
        return sum(self.entries[k][4] for k in self.modules[ref]["keys"].values())

    def encoding(self, key):
        """The encoding record of a logical `<module>.weight`, or None for a plain tensor."""
        kind, ref = self.logical.get(key, ("raw", None))
        return self.modules[ref] if kind == "encoded" else None

    # -- reads
    def read_raw(self, name):
        """(safetensors dtype, numpy array in its storage dtype) of a RAW tensor -- the bytes as stored."""
        import numpy as np
        if getattr(self, "mirror", False):
            raise SystemExit(f"{self.base_dir} is a remote mirror (sparse: headers only until the MLX lane fetches a "
                             "tensor) -- this lane needs the checkpoint itself on disk")
        p, dt, shape, off, n = self.entries[name]
        npdt = np.dtype(ST_NP[dt])
        if p in getattr(self, "_remote", {}):
            ent = self._remote[p]
            rd = (ent["repo"], ent["path_in_repo"], off, n)
            data = hub_range(*rd, revision=ent["revision"]) if ent.get("revision") else hub_range(*rd)
            arr = np.frombuffer(data, dtype=npdt)
            return dt, arr.reshape(shape)
        arr = np.fromfile(p, dtype=npdt, count=n // npdt.itemsize if npdt.itemsize else 0, offset=off)
        return dt, arr.reshape(shape)

    def modified(self, key):
        """True when an overlay (pollard-hf-smooth --overlay) changes this tensor: never byte-copy it."""
        return key in self.overlay

    def _apply_overlay(self, key, arr):
        import numpy as np
        op, sc = self.overlay[key]
        a = np.asarray(arr, np.float32)
        if op == "mul_in":
            return a * sc.reshape(1, -1)
        if op == "div":
            return a / sc
        if op == "div_offset":
            return (1.0 + a) / sc - 1.0
        raise SystemExit(f"unknown overlay op {op!r} for {key}")

    def read(self, key):
        """float32 numpy of a logical tensor (dequantized when it is encoded; smoothed when an overlay says so)."""
        out = self._read(key)
        return self._apply_overlay(key, out) if key in self.overlay else out

    def _read(self, key):
        kind, ref = self.logical[key]
        if kind == "raw":
            dt, arr = self.read_raw(ref)
            if dt in FLOAT_DT or dt.startswith("F8"):
                return _to_f32(dt, arr)
            return arr
        return self._decode(self.modules[ref])

    def read_torch(self, key, dtype=None):
        import numpy as np, torch
        if key in self.overlay:
            t = torch.from_numpy(np.ascontiguousarray(self.read(key)))
            return t.to(dtype) if dtype is not None else t
        kind, ref = self.logical[key]
        if kind == "raw":
            dt, arr = self.read_raw(ref)
            if dt == "BF16":
                t = torch.from_numpy(arr.astype(np.int16, copy=False)).view(torch.bfloat16)
            elif dt in ("F32", "F16", "F64"):
                t = torch.from_numpy(arr)
            elif dt.startswith("F8"):
                t = torch.from_numpy(_to_f32(dt, arr))
            else:
                t = torch.from_numpy(np.ascontiguousarray(arr))
                return t
            return t.to(dtype) if dtype is not None else t
        t = torch.from_numpy(self._decode(self.modules[ref]))
        return t.to(dtype) if dtype is not None else t

    def _raw_f32(self, name):
        dt, arr = self.read_raw(name)
        return _to_f32(dt, arr) if (dt in FLOAT_DT or dt.startswith("F8")) else arr.astype("float32")

    def _decode(self, m):
        import numpy as np
        algo, k = m["algo"], m["keys"]
        if algo == "nvfp4":
            import pollard_mx as X
            packed = self.read_raw(k.get("weight_packed", k.get("weight")))[1]
            sc = self._raw_f32(k["weight_scale"])
            if "weight_global_scale" in k:                   # compressed-tensors: w = fp4 * s / g
                g = float(np.asarray(self._raw_f32(k["weight_global_scale"])).reshape(-1)[0])
            else:                                            # modelopt: w = fp4 * s * s2
                s2 = float(np.asarray(self._raw_f32(k["weight_scale_2"])).reshape(-1)[0]) if "weight_scale_2" in k else 1.0
                g = 1.0 / s2 if s2 else 1.0
            return X.nvfp4_decode(packed, sc, g).astype(np.float32)
        if algo in ("mxfp4", "mxfp8"):
            import pollard_mxfp6 as M6
            parts = {"weight_scale": np.asarray(self.read_raw(k["weight_scale"])[1], np.uint8)}
            parts["weight_packed" if algo == "mxfp4" else "weight"] = self.read_raw(
                k.get("weight_packed", k.get("weight")))[1]
            return M6.decode_atom(parts, algo)
        if algo.startswith("ggml:"):
            from gguf import quants, GGMLQuantizationType as Q
            dt, raw = self.read_raw(k["weight"])
            rows = int(_prod(raw.shape[:-2])) if raw.ndim > 2 else raw.shape[0]
            flat = raw.reshape(rows, -1)
            return quants.dequantize(flat, Q[algo[5:]]).astype(np.float32)
        if algo == "fp8":
            w = _to_f32("F8_E4M3", self.read_raw(k["weight"])[1])
            if "weight_scale" in k:
                s = np.asarray(self._raw_f32(k["weight_scale"]), np.float32)
                w = w * (s.reshape(-1, 1) if s.size == w.shape[0] else s.reshape(-1)[0])
            return w
        if algo == "fp8-block":
            w = _to_f32("F8_E4M3", self.read_raw(k["weight"])[1])
            s = np.asarray(self._raw_f32(k["weight_scale_inv"]), np.float32)
            bo, bi = -(-w.shape[0] // s.shape[0]), -(-w.shape[1] // s.shape[1])
            return w * np.repeat(np.repeat(s, bo, 0), bi, 1)[:w.shape[0], :w.shape[1]]
        raise SystemExit(f"{self.dir}: no decoder for the {algo} encoding")

    def raw_parts(self, key):
        """{suffix: (dtype, numpy)} of an encoded linear's stored tensors, byte-exact (for passthrough)."""
        m = self.encoding(key)
        return {s: self.read_raw(n) for s, n in m["keys"].items()}


OVERLAY_JSON = "pollard-smooth.json"
OVERLAY_ST = "pollard-smooth-scales.safetensors"


def read_overlay(model_dir):
    """{"base": dir, "ops": {key: (op, float32 scale)}} of a smoothing overlay folder, else None."""
    p = os.path.join(model_dir, OVERLAY_JSON)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format") != "pollard-smooth-overlay/1":
        return None
    base = meta["base"]
    if not os.path.isabs(base):
        base = os.path.normpath(os.path.join(model_dir, base))
    hdr, off = st_header(os.path.join(model_dir, OVERLAY_ST))
    import numpy as np
    ops = {}
    for k, op in meta["ops"].items():
        info = hdr[k]
        a, b = info["data_offsets"]
        ops[k] = (op, np.fromfile(os.path.join(model_dir, OVERLAY_ST), dtype=np.float32, count=(b - a) // 4,
                                  offset=off + a))
    return {"base": base, "ops": ops, "meta": meta}


def is_overlay(model_dir):
    return os.path.isfile(os.path.join(model_dir, OVERLAY_JSON))


class MemSource:
    """A Source over tensors already in memory ({checkpoint name: numpy}) -- a decoded rung (EXLR8, GPTQ, MX) loaded
    through the same backbone machinery as a checkpoint on disk."""

    def __init__(self, model_dir, tensors):
        self.dir = self.base_dir = model_dir
        self.t = tensors
        self.overlay, self.modules = {}, {}
        self.logical = {k: ("raw", k) for k in tensors}
        self.entries = {}
        self.files = []
        try:
            self.config = config_json(model_dir)
        except (OSError, ValueError):
            self.config = {}

    def shape(self, key):
        return tuple(self.t[key].shape)

    def dtype(self, key):
        return "F32"

    def encoding(self, key):
        return None

    def modified(self, key):
        return False

    def read(self, key):
        import numpy as np
        return np.asarray(self.t[key], np.float32) if np.asarray(self.t[key]).dtype.kind == "f" else self.t[key]

    def read_torch(self, key, dtype=None):
        import numpy as np, torch
        t = torch.from_numpy(np.ascontiguousarray(self.read(key)))
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t


def _prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return out


# ----------------------------------------------------------------------------------------------- backbone

_LAYER_RE = re.compile(r"^(.*?)(?:^|(?<=\.))(layers|h|blocks|block)\.(\d+)\.")


def layer_prefixes(keys, sizes=None):
    """{checkpoint prefix ending in 'layers.': (n layers, bytes)} over a key set."""
    out = {}
    for k in keys:
        m = re.match(r"^(.*?\b(?:layers|h))\.(\d+)\.", k)
        if not m:
            continue
        p = m.group(1) + "."
        n, b, idx = out.get(p, (0, 0, set()))
        idx.add(int(m.group(2)))
        out[p] = (n, b + (sizes.get(k, 0) if sizes else 0), idx)
    return {p: (len(idx), b) for p, (n, b, idx) in out.items()}


@contextlib.contextmanager
def _empty_params():
    """Build modules with parameters on meta (no memory, no init) while buffers -- rotary inv_freq and friends,
    computed in __init__ -- stay real."""
    import torch
    old = torch.nn.Module.register_parameter

    def reg(self, name, param):
        old(self, name, param)
        if param is not None and not param.is_meta:
            p = self._parameters[name]
            self._parameters[name] = torch.nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)
    torch.nn.Module.register_parameter = reg
    try:
        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            try:
                from transformers.modeling_utils import no_init_weights
            except ImportError:
                no_init_weights = contextlib.nullcontext
        with no_init_weights():
            yield
    finally:
        torch.nn.Module.register_parameter = old


def _quiet_transformers():
    try:
        from transformers.utils import logging as hl
        hl.disable_progress_bar()
        hl.set_verbosity_error()
    except Exception:                                                     # noqa: BLE001
        pass


def _decode_json_floats(obj):
    """transformers writes inf / nan into config.json as {"__float__": "Infinity"}; read them back."""
    if isinstance(obj, dict):
        if set(obj) == {"__float__"}:
            return float(obj["__float__"])
        return {k: _decode_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_json_floats(v) for v in obj]
    return obj


def _config_from_dict(Conf, d):
    d = _decode_json_floats({k: v for k, v in d.items() if k not in ("auto_map",)})
    d.pop("model_type", None)
    try:
        return Conf.from_dict(d)
    except Exception:                                                     # noqa: BLE001
        return Conf(**d)


def _native_class(cfg_dict):
    """(transformers class, config object) for a config dict with a native model_type, or (None, why)."""
    import transformers as T
    from transformers import AutoConfig
    mt = cfg_dict.get("model_type")
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if mt not in CONFIG_MAPPING:
            return None, f"transformers {T.__version__} has no config class for model_type {mt!r}"
        conf = _config_from_dict(CONFIG_MAPPING[mt], cfg_dict)
    except Exception as e:                                                # noqa: BLE001
        return None, f"transformers could not read the {mt!r} config ({str(e)[:160]})"
    for arch in cfg_dict.get("architectures") or []:
        cls = getattr(T, arch, None)
        if cls is not None and getattr(cls, "config_class", None) is type(conf):
            return (cls, conf), ""
    from transformers.models.auto import modeling_auto as MA
    for mapping in ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
                    "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        name = (getattr(MA, mapping, {}) or {}).get(mt)
        if name and getattr(T, name, None) is not None:
            return (getattr(T, name), conf), ""
    for arch in cfg_dict.get("architectures") or []:                       # e.g. Qwen3OmniMoeForConditionalGeneration
        cls = getattr(T, arch, None)
        if cls is not None:
            return (cls, conf), ""
    name = (getattr(MA, "MODEL_MAPPING_NAMES", {}) or {}).get(mt)
    if name and getattr(T, name, None) is not None:
        return (getattr(T, name), conf), ""
    return None, f"transformers has no model class for model_type {mt!r}"


def _puzzle_build(cfg_dict):
    """A `<base>_puzzle` config (NVIDIA Puzzle / Star Elastic: one block config per layer) built from transformers'
    NATIVE <base> blocks, each layer from its own config -- what the checkpoint's own puzzle module does, without
    importing its code (Bolt's needs mamba_ssm, which does not build on Windows)."""
    import copy, importlib
    import transformers as T
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    mt = cfg_dict["model_type"]
    base = mt[:-len("_puzzle")]
    if base not in CONFIG_MAPPING:
        return None, f"no native transformers model for the puzzle base {base!r}"
    blocks = list(cfg_dict.get("block_configs") or [])
    if not blocks:
        return None, "a *_puzzle config without block_configs"
    g = {k: v for k, v in _decode_json_floats(cfg_dict).items()
         if k not in ("auto_map", "model_type", "block_configs", "architectures", "blockwise_members",
                      "mtp_block_configs")}
    g["layers_block_type"] = [b.get("block_type") for b in blocks]
    g["num_hidden_layers"] = len(blocks)
    g.pop("hybrid_override_pattern", None)
    Conf = CONFIG_MAPPING[base]
    conf = _config_from_dict(Conf, g)
    per = []
    aliases = dict(getattr(Conf, "attribute_map", {}) or {})            # e.g. num_local_experts -> n_routed_experts
    for b in blocks:
        d = dict(g)
        own = {k: v for k, v in b.items() if k != "block_type" and v is not None}
        for alias, target in aliases.items():
            if target in own or alias in own:            # the block's own value wins over the global alias
                d.pop(alias, None)
                d.pop(target, None)
        if "n_routed_experts" in own:
            d.pop("num_experts", None)
        d.update(own)
        per.append(_config_from_dict(Conf, d))
    mod_name = Conf.__module__.replace("configuration_", "modeling_")
    modeling = importlib.import_module(mod_name)
    block_cls = [getattr(modeling, n) for n in dir(modeling)
                 if n.endswith(("Block", "DecoderLayer")) and isinstance(getattr(modeling, n), type)
                 and getattr(getattr(modeling, n), "__module__", "") == mod_name]
    if not block_cls:
        return None, f"no per-layer block class in {mod_name}"
    cls = None
    for arch in ("ForCausalLM",):
        cand = [getattr(T, n) for n in dir(T) if n.endswith(arch) and getattr(getattr(T, n), "config_class", None) is Conf]
        if cand:
            cls = cand[0]
    if cls is None:
        return None, f"no native causal LM for {base!r}"
    return (cls, conf, per, modeling, block_cls), ""


def static_backbone(model_dir):
    """How Backbone() will build this checkpoint's text backbone, decided from config.json alone (nothing is built,
    no checkpoint code runs): {"route": "native" | "puzzle" | "remote" | None, "how": ..., "why": ...}. None = it
    cannot be built here, `why` says exactly why. Raises nothing; transformers missing -> route "unknown"."""
    try:
        cfg = config_json(model_dir)
    except (OSError, ValueError) as e:
        return {"route": None, "how": "", "why": f"config.json unreadable in {model_dir} ({e})"}
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        import transformers as T
    except Exception:                                                     # noqa: BLE001
        if needs_remote_code(model_dir):                 # no transformers to look for a native class: the code
            files, _ = remote_code_files(model_dir)      # is all this machine could run -- consent decides
            return {"route": "remote", "how": f"the checkpoint's own code ({', '.join(files)})",
                    "why": "transformers is not installed here to look for a native class", "files": files}
        return {"route": "unknown", "how": "", "why": "transformers is not installed here"}
    cands = [("", cfg)]
    for k in NESTED_CONFIGS[:4]:
        v = cfg.get(k)
        if isinstance(v, dict) and v.get("model_type"):
            cands.append((k, v))
            tv = v.get("text_config")
            if isinstance(tv, dict) and tv.get("model_type"):
                cands.append((k + ".text_config", tv))
    why = []
    for path, cd in cands:
        mt = cd.get("model_type") or ""
        if path and mt.endswith("_puzzle") and cd.get("block_configs"):
            if mt[:-len("_puzzle")] in CONFIG_MAPPING:
                return {"route": "puzzle", "how": f"transformers' native {mt[:-len('_puzzle')]} blocks, one config "
                                                  f"per layer from {path}.block_configs", "why": ""}
            why.append(f"{path}: no native base for {mt}")
            continue
        if mt in CONFIG_MAPPING and not (cd.get("auto_map") and path == "" and not _same_native(cd)):
            return {"route": "native", "how": f"transformers {mt} (native" + (f", from {path}" if path else "") + ")",
                    "why": ""}
        if mt:
            why.append(f"{path or 'config'}: transformers {T.__version__} has no {mt!r}")
    if needs_remote_code(model_dir):
        files, _ = remote_code_files(model_dir)
        return {"route": "remote", "how": f"the checkpoint's own code ({', '.join(files)})",
                "why": "; ".join(why), "files": files}
    return {"route": None, "how": "", "why": "; ".join(why) or "no model_type in config.json"}


class Backbone:
    """The text backbone of a checkpoint, built on meta, plus the map between its checkpoint keys and modules."""

    def __init__(self, model_dir, trust_remote_code="auto", log=print, ask=None, dtype=None, source=None):
        import torch
        _quiet_transformers()
        self.dir = model_dir
        self.log = log
        self.dtype = dtype or torch.float32
        self.src = source if source is not None else Source(model_dir)
        self.config = self.src.config
        self.remote = False
        self.how = ""
        self._build(trust_remote_code, ask)
        self._locate()

    # -- building
    def _sub_configs(self):
        c = self.config
        out = [("", c)]
        for k in NESTED_CONFIGS[:4]:
            v = c.get(k)
            if isinstance(v, dict) and v.get("model_type"):
                out.append((k, v))
                tv = v.get("text_config")
                if isinstance(tv, dict) and tv.get("model_type"):
                    out.append((k + ".text_config", tv))
        return out

    def _build(self, trc, ask):
        import torch
        why = []
        has_code = needs_remote_code(self.dir)
        for path, cd in self._sub_configs():
            if path and cd.get("model_type", "").endswith("_puzzle") and cd.get("block_configs"):
                got, w = _puzzle_build(cd)
                if got:
                    cls, conf, per, modeling, blocks = got
                    self._make(cls, conf, puzzle=(per, modeling, blocks))
                    self.how = (f"transformers {cls.__name__} (native), one block config per layer from "
                                f"{path or 'config'}.block_configs -- no checkpoint code runs")
                    return
                why.append(f"{path}: {w}")
                continue
            if cd.get("auto_map") and not path:
                why.append("config.json: custom code (auto_map) -- tried the native classes first")
            got, w = _native_class(cd)
            if got and not (cd.get("auto_map") and path == "" and not _same_native(cd)):
                cls, conf = got
                try:
                    self._make(cls, conf)
                except Exception as e:                                    # noqa: BLE001
                    why.append(f"{path or 'config'}: {cls.__name__} did not build ({str(e)[:160]})")
                    continue
                self.how = (f"transformers {cls.__name__} (native"
                            + (f", from {path}" if path else "") + ") -- no checkpoint code runs")
                return
            if w:
                why.append(f"{path or 'config'}: {w}")
        if has_code:
            if not consent(self.dir, trc, log=self.log, ask=ask):
                raise RemoteCodeRefused("the checkpoint needs its own code and --trust-remote-code is off")
            self._make_remote()
            return
        raise Unresolvable("no way to build the text backbone here: " + "; ".join(why))

    def _make(self, cls, conf, puzzle=None):
        import torch
        for c in [conf] + [getattr(conf, n, None) for n in ("text_config", "thinker_config", "llm_config")]:
            if c is not None and hasattr(c, "_experts_implementation"):
                try:
                    c._experts_implementation = "eager"
                except Exception:                                         # noqa: BLE001
                    pass
        try:
            conf._attn_implementation = "sdpa"
        except Exception:                                                 # noqa: BLE001
            pass
        conf.use_cache = False
        with _empty_params():
            if puzzle:
                per, modeling, blocks = puzzle
                originals = {b.__name__: b for b in blocks}
                patched = {}
                for b in blocks:
                    def make(base):
                        class Per(base):
                            def __init__(self, config, layer_idx=None, *a, **k):
                                c = per[layer_idx] if layer_idx is not None and layer_idx < len(per) else config
                                for at in ("_attn_implementation", "_experts_implementation"):
                                    try:
                                        setattr(c, at, getattr(config, at))
                                    except Exception:             # noqa: BLE001
                                        pass
                                super().__init__(c, layer_idx, *a, **k)
                        Per.__name__ = base.__name__
                        return Per
                    patched[b.__name__] = make(b)
                try:
                    for n, p in patched.items():
                        setattr(modeling, n, p)
                    self.root = cls(conf)
                finally:
                    for n, b in originals.items():
                        setattr(modeling, n, b)
            else:
                self.root = cls(conf)
        self.root.eval()
        self.root_config = conf

    def _make_remote(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
        conf = AutoConfig.from_pretrained(self.dir, trust_remote_code=True)
        conf.use_cache = False
        last = None
        with _empty_params():
            for Auto in (AutoModelForCausalLM, AutoModel):
                try:
                    self.root = Auto.from_config(conf, trust_remote_code=True)
                    break
                except Exception as e:                                    # noqa: BLE001
                    last = e
            else:
                raise Unresolvable(f"the checkpoint's own code did not build ({str(last)[:200]})")
        self.root.eval()
        self.root_config = conf
        self.remote = True
        self.how = f"the checkpoint's own code ({type(self.root).__name__}, consent given)"

    # -- locating the decoder + the key map
    def _locate(self):
        import torch
        best = None
        for name, mod in self.root.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 1:
                n = sum(p.numel() for p in mod.parameters())
                kinds = {type(m).__name__ for m in mod}
                if any(t in name.lower() for t in ("vision", "visual", "audio", "talker", "code2wav", "radio",
                                                    "image", "speech", "sound", "encoder.")):
                    continue
                if len(kinds) > 3 or n == 0:
                    continue
                if best is None or n > best[0]:
                    best = (n, name, mod)
        if best is None:
            raise Unresolvable(f"no decoder layer stack in {type(self.root).__name__}")
        _, self.layers_path, self.layers = best
        self.text_root_path = self.layers_path.rsplit(".", 1)[0] if "." in self.layers_path else ""
        self.text_root = self.root.get_submodule(self.text_root_path) if self.text_root_path else self.root
        self.n_layers = len(self.layers)
        self._key_map()

    def _mapping(self):
        try:
            from transformers.conversion_mapping import get_model_conversion_mapping
            return get_model_conversion_mapping(self.root) or []
        except Exception:                                                 # noqa: BLE001
            return []

    def _key_map(self):
        """checkpoint key -> (module param name, converter pattern or None) for every logical tensor the root owns;
        the checkpoint-side prefix Q the root's keys sit under (e.g. 'language_model.' for Bolt's LLM)."""
        from transformers.core_model_loading import WeightConverter, WeightRenaming
        try:
            from transformers.core_model_loading import rename_source_key
        except ImportError:                                               # pragma: no cover
            rename_source_key = None
        self.weight_mapping = self._mapping()
        ren = [w for w in self.weight_mapping if isinstance(w, WeightRenaming)]
        conv = [w for w in self.weight_mapping if isinstance(w, WeightConverter)]
        meta_sd = self.root.state_dict()
        self._meta_keys = set(meta_sd)
        bmp = getattr(self.root, "base_model_prefix", None)
        sizes = {k: _prod(self.src.shape(k)) for k in self.src.logical}
        lp = layer_prefixes(self.src.logical, sizes)
        # candidate checkpoint prefixes Q: '' and every dotted prefix of the layer stacks' prefixes
        cands = {""}
        for p in lp:
            parts = p.split(".")
            for i in range(1, len(parts)):
                cands.add(".".join(parts[:i]) + ".")

        def rename(k):
            if rename_source_key is None:
                return (k if k in meta_sd else None), None
            try:
                t, pat = rename_source_key(k, ren, conv, bmp, meta_sd)
            except Exception:                                             # noqa: BLE001
                return None, None
            return (t if t in meta_sd else None), pat
        best = None
        for q in sorted(cands, key=len):
            hits = {}
            for k in self.src.logical:
                if not k.startswith(q):
                    continue
                t, pat = rename(k[len(q):])
                if t is not None:
                    hits[k] = (t, pat)
            layer_hits = sum(1 for t, _ in hits.values() if t.startswith(self.layers_path + "."))
            score = (layer_hits, len(hits))
            if best is None or score > best[0]:
                best = (score, q, hits)
        (layer_hits, _), self.Q, self.kmap = best
        if not layer_hits:
            raise Unresolvable(f"none of the checkpoint's tensors map onto {type(self.root).__name__}.{self.layers_path}"
                               " -- the config and the weights do not describe the same model")
        lpfx = self.layers_path + "."
        self.layer_keys = [[] for _ in range(self.n_layers)]
        self.outer_keys, self.other_keys = [], []
        for k in self.src.logical:
            t = self.kmap.get(k)
            if t is None:
                self.other_keys.append(k)
                continue
            mod = t[0]
            if mod.startswith(lpfx):
                i = int(mod[len(lpfx):].split(".", 1)[0])
                self.layer_keys[i].append(k)
            elif self.text_root_path == "" or mod.startswith(self.text_root_path + "."):
                self.outer_keys.append(k)
            else:
                self.other_keys.append(k)           # lm_head, towers, talker ... copied through by every lane
        ck = [p for p in lp if any(k.startswith(p) for k in self.layer_keys[0])]
        self.layer_ckpt_prefix = ck[0] if ck else self.Q

    # -- descriptions
    def describe(self):
        hid = getattr(self.text_config(), "hidden_size", None)
        return {"how": self.how, "root": type(self.root).__name__, "text_root": type(self.text_root).__name__,
                "layers": self.n_layers, "hidden": hid, "checkpoint_prefix": self.Q,
                "layer_prefix": self.layer_ckpt_prefix, "remote_code": self.remote,
                "copied_through": len(self.other_keys),
                "encoded_source": sorted({m["algo"] for m in self.src.modules.values()})}

    def text_config(self):
        return getattr(self.text_root, "config", None) or self.root_config

    def hidden_size(self):
        c = self.text_config()
        return int(getattr(c, "hidden_size", 0) or getattr(c, "d_model", 0) or 0)

    # -- loading
    def load(self, keys, device="cpu", dtype=None, tensors=None):
        """Load checkpoint tensors (by logical key) into the meta root -- through transformers' own conversion
        (expert fusing, renames), so the module holds exactly what from_pretrained would give it."""
        import torch
        dtype = dtype or self.dtype
        sd = {}
        for k in keys:
            t = tensors[k] if tensors is not None and k in tensors else self.src.read_torch(k)
            if t.is_floating_point():
                t = t.to(dtype)
            sd[k[len(self.Q):]] = t
        _load_into(self.root, sd, self.weight_mapping, device, dtype)

    def free_layer(self, i):
        import torch
        layer = self.layers[i]
        for mod in layer.modules():
            for n, p in list(mod._parameters.items()):
                if p is not None and not p.is_meta:
                    mod._parameters[n] = torch.nn.Parameter(torch.empty_like(p, device="meta"),
                                                            requires_grad=False)

    def check_loaded(self, i):
        left = [n for n, p in self.layers[i].named_parameters() if p.is_meta]
        if left:
            raise SystemExit(f"layer {i}: {len(left)} parameters were not in the checkpoint "
                             f"({', '.join(left[:4])}) -- the backbone map is wrong for this model")


def _same_native(cd):
    """A config with an auto_map whose model_type transformers ALSO knows natively -- the native class is used
    (it is the reference implementation; the checkpoint's code is never imported)."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        return cd.get("model_type") in CONFIG_MAPPING
    except Exception:                                                     # noqa: BLE001
        return False


def _load_into(root, sd, mapping, device, dtype):
    import inspect, torch
    try:
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig
    except ImportError:
        convert_and_load_state_dict_in_model = None
    if convert_and_load_state_dict_in_model is not None:
        lc = LoadStateDictConfig(weight_mapping=mapping, dtype=dtype, device_map={"": device})
        params = inspect.signature(convert_and_load_state_dict_in_model).parameters
        if "tp_plan" in params:                              # transformers 5.15
            convert_and_load_state_dict_in_model(root, sd, lc, None)
        else:
            convert_and_load_state_dict_in_model(root, sd, lc)
        return
    missing, unexpected = root.load_state_dict(sd, strict=False, assign=True)   # pragma: no cover
    root.to(device)


# ----------------------------------------------------------------------------------------------- recording + run


class _Stop(Exception):
    pass


def _to(v, device, dtype=None, cache=None):
    import torch
    if torch.is_tensor(v):
        key = (id(v), str(device))
        if cache is not None and key in cache:
            return cache[key]
        out = v.to(device)
        if dtype is not None and out.is_floating_point() and out.dtype != dtype and out.dim() > 0 and out.numel() > 1:
            out = out.to(dtype)
        if cache is not None:
            cache[key] = out
        return out
    if isinstance(v, tuple):
        return tuple(_to(x, device, dtype, cache) for x in v)
    if isinstance(v, list):
        return [_to(x, device, dtype, cache) for x in v]
    if isinstance(v, dict):
        return {k: _to(x, device, dtype, cache) for k, x in v.items()}
    return v


class Engine:
    """Calibration activations through the backbone, one block at a time."""

    def __init__(self, bb: Backbone, device="cpu", dtype=None, act_dtype=None, log=print, cast_blocks=False):
        import torch
        self.bb, self.device, self.log = bb, device, log
        self.dtype = dtype or bb.dtype
        # transformers' loader casts every tensor to the dtype of the (meta) parameter it lands in -- the backbone is
        # built in float32, so a bf16 plan would still hold float32 blocks. cast_blocks: each block's meta parameters
        # take the plan's dtype first (free: they are meta), so the block is loaded AS bf16
        self.cast_blocks = cast_blocks
        self.act_dtype = act_dtype or torch.bfloat16
        self.hidden, self.kwargs, self.ret_tuple = [], [], False
        self.peak_rss = 0

    def record(self, rows, outer_tensors=None):
        """Run the calibration rows through the text root with every block replaced by a recorder: the block
        inputs at layer 0 (kept) and each block's call arguments (kept per layer, per row). `outer_tensors` loads the
        embeddings from elsewhere (a rung's own) instead of the source."""
        import torch
        bb = self.bb
        bb.load(bb.outer_keys, self.device, self.dtype, tensors=outer_tensors)
        left = [n for n, p in bb.text_root.named_parameters() if p.is_meta and not n.startswith(
            bb.layers_path[len(bb.text_root_path) + 1:] if bb.text_root_path else bb.layers_path)]
        if left:
            raise SystemExit(f"the text root still has unloaded weights outside the blocks: {left[:4]}")
        orig = list(bb.layers)
        for attempt in (False, True):
            self.ret_tuple = attempt
            self.hidden, self.kwargs = [], []
            ok = True
            shapes = {}

            class Rec(torch.nn.Module):
                def __init__(s, idx, real):
                    super().__init__()
                    s.idx = idx
                    object.__setattr__(s, "_real", real)

                def __getattr__(s, name):
                    try:
                        return torch.nn.Module.__getattr__(s, name)
                    except AttributeError:
                        return getattr(object.__getattribute__(s, "_real"), name)

                def forward(s, *args, **kw):
                    hs = args[0] if args else kw.pop("hidden_states")
                    rest = args[1:]
                    sink(s.idx, hs, rest, kw)
                    return (hs,) if self.ret_tuple else hs

            rowcache = {}

            def sink(idx, hs, rest, kw):
                shapes.setdefault(idx, tuple(hs.shape))
                if idx == 0:
                    self.hidden.append(hs.detach().to("cpu", self.act_dtype))
                kw = {k: v for k, v in kw.items() if k not in ("past_key_values", "past_key_value", "cache_params",
                                                             "use_cache", "output_attentions", "output_router_logits")}
                # the tensors every block of a row shares (rotary tables, masks) are kept ONCE per row
                self.kwargs[-1].append((_to(tuple(rest), "cpu", None, rowcache), _to(kw, "cpu", None, rowcache)))

            for i in range(len(orig)):
                bb.layers[i] = Rec(i, orig[i])
            try:
                with torch.no_grad():
                    for r in rows:
                        self.kwargs.append([])
                        rowcache.clear()
                        ids = torch.tensor([r], device=self.device)
                        try:
                            bb.text_root(input_ids=ids, use_cache=False)
                        except TypeError:
                            bb.text_root(ids)
            finally:
                for i in range(len(orig)):
                    bb.layers[i] = orig[i]
            if len(set(shapes.values())) <= 1:
                break
            ok = False
        if not ok:
            raise SystemExit(f"could not record the blocks' calls (shapes {sorted(set(shapes.values()))})")
        # the tensors the recorded calls share (rotary tables, masks) are kept once
        return self

    def act_bytes(self):
        n = sum(h.numel() * h.element_size() for h in self.hidden)
        seen = set()
        for per in self.kwargs:
            for args, kw in per:
                for t in _tensors((args, kw)):
                    if id(t) not in seen:
                        seen.add(id(t))
                        n += t.numel() * t.element_size()
        return n

    def load_layer(self, i, tensors=None, guard_gb=None):
        bb = self.bb
        need = sum(_prod(bb.src.shape(k)) for k in bb.layer_keys[i]) * _elsize(self.dtype) / 1e9
        if guard_gb is not False:
            free_ram_guard(need + (guard_gb or 0.0), what=f"layer {i}")
        if self.cast_blocks:
            import torch
            for mod in bb.layers[i].modules():
                for n, p in list(mod._parameters.items()):
                    if p is not None and p.is_meta and p.is_floating_point() and p.dtype != self.dtype:
                        mod._parameters[n] = torch.nn.Parameter(torch.empty_like(p, device="meta", dtype=self.dtype),
                                                                requires_grad=False)
        bb.load(bb.layer_keys[i], self.device, self.dtype, tensors=tensors)
        bb.check_loaded(i)
        self.note_rss()

    def run_layer(self, i, keep_outputs=True):
        """Forward every calibration row through block i; returns the outputs (CPU, act dtype) when asked."""
        import torch
        layer = self.bb.layers[i]
        outs = []
        with torch.no_grad():
            for r, h in enumerate(self.hidden):
                args, kw = self.kwargs[r][i]
                cache = {}                                   # one row's shared call tensors, moved once
                x = h.to(self.device, self.dtype)
                o = layer(x, *_to(args, self.device, self.dtype, cache), **_to(kw, self.device, self.dtype, cache))
                o = o[0] if isinstance(o, (tuple, list)) else o
                if keep_outputs:
                    outs.append(o.detach().to("cpu", self.act_dtype))
                del o, x, cache
        self.note_rss()
        return outs

    def note_rss(self):
        self.peak_rss = max(self.peak_rss, peak_rss_bytes())


def _tensors(obj):
    import torch
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _tensors(x)


def _elsize(dtype):
    import torch
    return torch.empty(0, dtype=dtype).element_size()


# ----------------------------------------------------------------------------------------------- statistics


class InputStats:
    """Per-checkpoint-weight input statistics of one block: X^T X (GPTQ / LDLQ) and per-channel |X| max
    (SmoothQuant / FP8 / NVFP4 input scales), for plain nn.Linear modules AND for transformers' fused expert
    tensors (captured per expert from the module's own F.linear calls, so any gating layout is right)."""

    def __init__(self, engine, i, hessian=True, amax=True, expert_range=None, only=None):
        self.e, self.i = engine, i
        self.hessian, self.amax = hessian, amax
        self.expert_range = expert_range
        self.only = only
        self.H, self.A, self.ntok = {}, {}, {}
        self.experts_in = {}
        self._hooks = []

    def keys_for(self):
        """{module path (within the root): [checkpoint keys]} for nn.Linear weights, and
        {(fused param path, expert): [checkpoint keys]} for expert tensors."""
        bb = self.e.bb
        lin, ex = {}, {}
        for k in bb.layer_keys[self.i]:
            mod, pat = bb.kmap[k]
            if pat is None and mod.endswith(".weight"):
                lin.setdefault(mod[:-len(".weight")], []).append(k)
            elif pat is not None:
                m = re.search(r"\.(?:experts|local_experts)\.(\d+)\.", k)
                if m:
                    ex.setdefault((mod, int(m.group(1))), []).append(k)
        return lin, ex

    def __enter__(self):
        import torch
        bb = self.e.bb
        lin, ex = self.keys_for()
        self.lin_keys, self.ex_keys = lin, ex
        mods = dict(bb.root.named_modules())
        shared = {}
        self.alias = {}
        self._pass = 0

        def add(key, x):
            x2 = x.reshape(-1, x.shape[-1]).float()
            if self.hessian:
                h = x2.T @ x2
                self.H[key] = self.H[key] + h if key in self.H else h
            if self.amax:
                a = x2.abs().amax(0)
                self.A[key] = torch.maximum(self.A[key], a) if key in self.A else a
            self.ntok[key] = self.ntok.get(key, 0) + x2.shape[0]

        for path in lin:
            m = mods.get(path)
            if m is None or not isinstance(m, torch.nn.Linear):
                continue
            if self.only is not None and path not in self.only:
                continue

            def pre(mod, inp, _path=path):
                x = inp[0]
                ptr = (x.data_ptr(), tuple(x.shape), x.dtype)
                if ptr in shared:                                # q/k/v / gate/up see one input: count it once
                    first = shared[ptr][1]
                    if first != _path:
                        self.alias[_path] = first
                        return
                shared[ptr] = (x, _path)                         # held for the block call: no address reuse
                add(_path, x.detach())
            self._hooks.append(m.register_forward_pre_hook(pre))

        def reset(mod, inp):
            shared.clear()
        self._hooks.append(bb.layers[self.i].register_forward_pre_hook(reset))
        # fused experts: patch F.linear while the experts module runs
        fused = {}
        for (param, e) in ex:
            mpath, pname = param.rsplit(".", 1)
            fused.setdefault(mpath, set()).add(pname)
        for mpath, pnames in fused.items():
            m = mods.get(mpath)
            if m is None:
                continue
            self._hooks.append(_ExpertTap(m, mpath, pnames, self, add))
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        return False

    def next_pass(self):
        self._pass += 1

    def _expert_key(self, key):
        for (param, e), ks in self.ex_keys.items():
            if key in ks:
                return param, e
        return None

    def hessian_for(self, key):
        """(H [in, in] float32 averaged per token, n tokens) for a checkpoint weight key, or (None, 0). A cold
        expert (no routed token) gets the experts block's input over ALL tokens, reported with 0 tokens."""
        for path, ks in self.lin_keys.items():
            if key in ks:
                src = self.alias.get(path, path)
                if src in self.H:
                    return self.H[src] / max(self.ntok[src], 1), self.ntok[src]
                return None, 0
        pe = self._expert_key(key)
        if pe is None:
            return None, 0
        param, e = pe
        want = self.e.bb.src.shape(key)[1]
        for c, h in self.H.items():
            if isinstance(c, tuple) and c[0] == "x" and c[1] == param and c[2] == e and h.shape[0] == want:
                return h / max(self.ntok[c], 1), self.ntok[c]
        fb = self.experts_in.get(param.rsplit(".", 1)[0])
        if fb is not None and fb[0].shape[0] == want:
            return fb[0] / max(fb[1], 1), 0
        return None, 0

    def amax_for(self, key):
        """Per-input-channel |X| max for a checkpoint weight key (None when this block never fed it)."""
        for path, ks in self.lin_keys.items():
            if key in ks:
                return self.A.get(self.alias.get(path, path))
        pe = self._expert_key(key)
        if pe is None:
            return None
        param, e = pe
        want = self.e.bb.src.shape(key)[1]
        for c, a in self.A.items():
            if isinstance(c, tuple) and c[0] == "x" and c[1] == param and c[2] == e and a.shape[0] == want:
                return a
        return None


class _ExpertTap:
    """Captures, per expert, the inputs of the F.linear calls a fused-experts module makes on slices of its 3-D
    weights -- the routed tokens of each expert, through whatever gating the module implements."""

    def __init__(self, module, mpath, pnames, stats, add):
        import torch
        self.m, self.stats, self.add = module, stats, add
        self.params = {}
        for pn in pnames:
            p = getattr(module, pn, None)
            if isinstance(p, torch.nn.Parameter) and p.dim() == 3:
                self.params[pn] = p
        self.mpath = mpath
        self.orig_forward = module.forward
        tap = self

        def forward(*a, **k):
            import torch.nn.functional as F
            hs = a[0] if a else k.get("hidden_states")
            if hs is not None and stats.hessian:
                x2 = hs.reshape(-1, hs.shape[-1]).float()
                h = x2.T @ x2
                cur = stats.experts_in.get(mpath)
                stats.experts_in[mpath] = (h, x2.shape[0]) if cur is None else (cur[0] + h, cur[1] + x2.shape[0])
            real = F.linear

            def lin(x, w, b=None):
                pe = tap._which(w)
                if pe is not None:
                    pn, e = pe
                    rng = stats.expert_range
                    if rng is None or rng[0] <= e < rng[1]:
                        add(("x", f"{mpath}.{pn}", e), x.detach())
                return real(x, w, b)
            F.linear = lin
            try:
                return tap.orig_forward(*a, **k)
            finally:
                F.linear = real
        module.forward = forward

    def _which(self, w):
        for pn, p in self.params.items():
            if w.dim() != 2 or p.device != w.device:
                continue
            base = p.data_ptr()
            step = p.stride(0) * p.element_size()
            off = w.data_ptr() - base
            if 0 <= off < p.numel() * p.element_size() and off % step == 0:
                return pn, off // step
        return None

    def remove(self):
        self.m.forward = self.orig_forward


# ----------------------------------------------------------------------------------------------- memory


def peak_rss_bytes():
    """This process's peak resident set, in bytes (0 when the platform will not say)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return int(c.PeakWorkingSetSize)
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)
    except Exception:                                                     # noqa: BLE001
        return 0


def free_ram_gb():
    try:
        from pollard_calc import detect_available_ram_gb
        return detect_available_ram_gb()
    except Exception:                                                     # noqa: BLE001
        return None


class OutOfMemoryBudget(SystemExit):
    pass


def free_ram_guard(need_gb, what="this step", margin_gb=None):
    """Stop cleanly (never swap the machine to death) when free RAM cannot hold the next allocation."""
    margin = float(os.environ.get("POLLARD_RAM_MARGIN_GB", 2.0)) if margin_gb is None else margin_gb
    free = free_ram_gb()
    if free is None:
        return
    if need_gb + margin > free:
        raise OutOfMemoryBudget(f"stopping before {what}: it needs ~{need_gb:.1f} GB (+{margin:g} GB margin) and "
                                f"{free:.1f} GB of RAM is free. Nothing was half-written; close something or lower "
                                "--nsamples / --max-ram-gb and run again.")


def layer_shapes(bb):
    """[(n params, [(key, shape)])] per decoder layer, from the shapes on disk."""
    out = []
    for i in range(bb.n_layers):
        ks = [(k, bb.src.shape(k)) for k in bb.layer_keys[i]]
        out.append((sum(_prod(s) for _, s in ks), ks))
    return out


def plan_memory(bb, nsamples, seqlen, compute_bytes=4, act_bytes=2, hessian=True, writer_gb=None,
                expert_chunks=1):
    """The run's peak RAM (GB) by component, from the checkpoint's shapes: the biggest block at compute precision
    + its statistics (X^T X per distinct input; experts per expert, / expert_chunks) + the calibration activations
    (inputs and outputs of one block) + the embeddings / final norm + the writer's buffer + torch's overhead."""
    shapes = layer_shapes(bb)
    hid = bb.hidden_size() or 1
    worst = (0.0, -1, 0.0)
    for i, (n, ks) in enumerate(shapes):
        w = n * compute_bytes / 1e9
        h = 0.0
        if hessian:
            seen = set()
            ex = 0.0
            for k, s in ks:
                if len(s) != 2:
                    continue
                if re.search(r"\.(?:experts|local_experts)\.\d+\.", k):
                    ex += s[1] * s[1] * 4 / 1e9 * (0.5 if re.search(r"(gate|w1|w3|up)", k.rsplit(".", 2)[-2]) else 1.0)
                    continue
                key = (s[1], k.rsplit(".", 2)[-3] if k.count(".") >= 2 else k)
                if key in seen:
                    continue
                seen.add(key)
                h += s[1] * s[1] * 4 / 1e9
            h += ex / max(1, expert_chunks)
        if w + h > worst[0] + worst[2]:
            worst = (w, i, h)
    acts = nsamples * seqlen * hid * act_bytes * 2 / 1e9
    # recorded per-row call arguments: rotary tables (2 x seq x head_dim, fp32) and, when materialised, masks
    head = int(getattr(bb.text_config(), "head_dim", 0) or hid // max(1, int(getattr(bb.text_config(), "num_attention_heads", 1) or 1)))
    kw = nsamples * (2 * seqlen * head * 4) / 1e9
    outer = sum(_prod(bb.src.shape(k)) for k in bb.outer_keys) * compute_bytes / 1e9
    if writer_gb is None:                                 # ShardWriter flushes at 2 GB; a small model never gets there
        total = sum(_prod(bb.src.shape(k)) for k in bb.src.logical) * 2 / 1e9
        writer_gb = round(min(2.0, total), 2)
    overhead = 0.6                                        # python + torch + transformers, measured ~0.45 GB
    total = worst[0] + worst[2] + acts + kw + outer + writer_gb + overhead
    return {"layer": worst[1], "layer_weights_gb": round(worst[0], 2), "statistics_gb": round(worst[2], 2),
            "activations_gb": round(acts + kw, 2), "outer_gb": round(outer, 2), "writer_gb": writer_gb,
            "overhead_gb": overhead, "peak_gb": round(total, 2), "nsamples": nsamples, "seqlen": seqlen,
            "expert_chunks": expert_chunks}


def fit_memory(bb, nsamples, seqlen, budget_gb, log=print, min_samples=8, **kw):
    """Shrink the plan until its peak fits `budget_gb`: the block computed in bf16 instead of fp32, then fewer
    calibration rows, then MoE statistics in expert chunks. Returns the plan (its `compute_bytes` says which
    precision the run must use); raises OutOfMemoryBudget with the numbers when nothing fits."""
    cb = kw.pop("compute_bytes", 4)
    n, chunks = nsamples, kw.pop("expert_chunks", 1)
    plan = plan_memory(bb, n, seqlen, compute_bytes=cb, expert_chunks=chunks, **kw)
    while plan["peak_gb"] > budget_gb:
        if cb == 4:
            cb = 2
        elif n > min_samples and plan["activations_gb"] >= plan["statistics_gb"] / 2:
            n = max(min_samples, n // 2)
        elif chunks < 64 and plan["statistics_gb"] > 0.5:
            chunks *= 2
        elif n > min_samples:
            n = max(min_samples, n // 2)
        else:
            raise OutOfMemoryBudget(
                f"this build needs ~{plan['peak_gb']:.1f} GB of RAM at its smallest ({n} calibration rows, bf16 "
                f"blocks, MoE statistics in {chunks} expert chunks: block {plan['layer']} = {plan['layer_weights_gb']} "
                f"GB of weights + {plan['statistics_gb']} GB of statistics, {plan['activations_gb']} GB of "
                f"activations, {plan['outer_gb']} GB embeddings) and the budget is {budget_gb:.1f} GB")
        plan = plan_memory(bb, n, seqlen, compute_bytes=cb, expert_chunks=chunks, **kw)
    plan["compute_bytes"] = cb
    if n != nsamples or chunks != 1 or cb != 4:
        log(f"   memory: {nsamples} -> {n} calibration rows, {'bf16' if cb == 2 else 'fp32'} blocks, expert "
            f"statistics in {chunks} chunk(s) to fit {budget_gb:.1f} GB (peak ~{plan['peak_gb']:.1f} GB)")
    return plan


def plan_dtype(plan):
    import torch
    return torch.bfloat16 if plan.get("compute_bytes") == 2 else torch.float32


def ram_budget(max_ram_gb=None):
    free = free_ram_gb()
    caps = [x for x in (max_ram_gb, (free * 0.85) if free else None) if x]
    return min(caps) if caps else 16.0


def stream_device(want="auto"):
    """Where a streamed lane runs (pollard_device decides what works here): CUDA when asked or `auto` finds a usable
    card, else the CPU. An explicit `cpu` hides CUDA from torch BEFORE it is imported -- the shared box's GPU is
    someone else's -- and MPS is not used (fp64 Cholesky / index ops the GPTQ and EXLR8 solvers need)."""
    import pollard_device as D
    w = (want or "auto").strip().lower()
    if w == "cpu":
        if "torch" not in sys.modules:
            D.hide_cuda()
        return "cpu"
    d = D.pick(w)
    return "cpu" if d.startswith("mps") else d


# ----------------------------------------------------------------------------------------------- calibration


def calib_rows(model_dir, calib, nsamples=128, seqlen=2048, trust=False):
    """Token rows from the shared corpus (pollard-calib's whole samples, agent traces intact)."""
    from transformers import AutoTokenizer
    import pollard_workspace as ws
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=bool(trust))
    samples = ws.calib_samples(calib) if calib and os.path.exists(calib) else \
        ["The quick brown fox jumps over the lazy dog. " * 64]
    rows, buf = [], []
    for s in samples:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seqlen:
            rows.append(buf[:seqlen])
            buf = buf[seqlen:]
            if len(rows) >= nsamples:
                return rows
    if buf and len(rows) < nsamples and len(buf) >= 16:
        rows.append(buf)
    return rows


def collect_input_amax(model, calib=None, nsamples=128, seqlen=2048, device="cpu", trust_remote_code="auto",
                       max_ram_gb=None, rows=None, log=print):
    """{checkpoint weight key: max |x| of its input} for every block linear (experts per expert), from one streamed
    pass of the calibration set through the ORIGINAL weights -- the activation scales NVFP4 (input_global_scale)
    and static FP8 (input_scale) need. Returns (amax, summary)."""
    t0 = time.time()
    bb = Backbone(model, trust_remote_code, log=log)
    plan = fit_memory(bb, nsamples, seqlen, ram_budget(max_ram_gb), log=log, hessian=False)
    if rows is None:
        rows = calib_rows(bb.src.base_dir if bb.src.overlay else model, calib, plan["nsamples"], seqlen,
                          trust=bb.remote)
    bb.dtype = plan_dtype(plan)
    eng = Engine(bb, device=device, log=log)
    eng.record(rows)
    amax = {}
    for i in range(bb.n_layers):
        eng.load_layer(i)
        with InputStats(eng, i, hessian=False, amax=True) as st:
            outs = eng.run_layer(i)
        for k in bb.layer_keys[i]:
            if k.endswith(".weight") and len(bb.src.shape(k)) == 2:
                a = st.amax_for(k)
                if a is not None:
                    amax[k] = float(a.max())
        bb.free_layer(i)
        eng.hidden = outs
    summary = {"backbone": bb.describe(), "rows": len(rows), "tokens": sum(len(r) for r in rows),
               "memory_plan": plan, "peak_rss_gb": round(eng.peak_rss / 1e9, 2),
               "minutes": round((time.time() - t0) / 60, 2), "linears": len(amax)}
    return amax, summary


# ----------------------------------------------------------------------------------------------- writing


#: pollard release sets this for ONE build of a rung bigger than the free disk: json {"out", "repo", "path_in_repo",
#: "revision", "private"}. Every safetensors shard written into exactly `out` is uploaded to <repo>/<path_in_repo>/ on the branch
#: `revision` (the rung's STAGING branch, pollard-staging-<rung>: never the repo's main branch, which may already be
#: public), confirmed there (size + sha256), recorded in pollard-streamed.json (with its header, so the rung can still
#: be READ -- verify streams it back from that branch with range requests) and deleted before the next shard is
#: written. Only a rung that passes its gate is copied onto main (promote_staging: a server-side copy, no bytes
#: re-sent) and its branch deleted; a rung that fails has its branch deleted and main never saw it. Any other directory
#: (the smoothed source, a text copy) is written normally.
SHARD_UPLOAD_ENV = "POLLARD_SHARD_UPLOAD"
STREAMED = "pollard-streamed.json"
STAGING_PREFIX = "pollard-staging-"


def staging_branch(path_in_repo):
    """The staging branch of the rung streamed to `path_in_repo` (GPTQ/int8 -> pollard-staging-GPTQ-int8)."""
    tail = re.sub(r"[^A-Za-z0-9._-]+", "-", str(path_in_repo or "")).strip("-.") or "rung"
    return STAGING_PREFIX + tail


def shard_upload_spec(out):
    raw = os.environ.get(SHARD_UPLOAD_ENV)
    if not raw:
        return None
    try:
        spec = json.loads(raw)
    except ValueError:
        return None
    norm = lambda p: os.path.normcase(os.path.abspath(str(p).rstrip("/\\")))
    return spec if spec.get("out") and spec.get("repo") and norm(spec["out"]) == norm(out) else None


def read_streamed(model_dir):
    """{shard file name: entry} of the shards this rung uploaded + freed while it was written ({} for a normal one)."""
    p = os.path.join(model_dir, STREAMED)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("files") or {}
    except (OSError, ValueError):
        return {}


def _record_streamed(out, fn, entry):
    p = os.path.join(out, STREAMED)
    d = {"files": {}}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    d.setdefault("files", {})[fn] = entry
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, p)


def _hf_api():
    from pollard_privacy import SafeApi
    return SafeApi()


def stream_shard(out, path, spec, api=None, log=print):
    """Upload ONE finished shard, confirm the Hub holds it byte for byte (size + sha256 read back), record it, delete
    the local copy. Any failure raises with the shard still on disk -- the build stops there, nothing is lost."""
    import pollard_hubcheck as H
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    fn = os.path.basename(path)
    rp = f"{spec['path_in_repo'].strip('/')}/{fn}" if spec.get("path_in_repo") else fn
    size, sha = os.path.getsize(path), H.sha256_file(path)
    hdr, base = st_header(path)
    api = api or _hf_api()
    rev = spec.get("revision")
    # public from the start (pollard release: `private: true` in the plan keeps it private); the shard itself goes to
    # the rung's staging branch, so main never holds an unverified byte
    api.create_repo(spec["repo"], repo_type="model", exist_ok=True, private=spec.get("private") is True)
    if rev:                                     # the rung's staging branch: main (maybe public) never sees it unverified
        api.create_branch(spec["repo"], branch=rev, exist_ok=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=rp, repo_id=spec["repo"], revision=rev)
    else:
        api.upload_file(path_or_fileobj=path, path_in_repo=rp, repo_id=spec["repo"])
    bad = H.mismatches(api, spec["repo"], [(rp, path)], revision=rev)
    if bad:
        raise RuntimeError(f"shard {fn} is not intact on {spec['repo']} ({bad[0]}) -- kept at {path}, build stopped")
    ent = {"repo": spec["repo"], "path_in_repo": rp, "size": size, "sha256": sha,
           "git_sha1": H.git_blob_sha1(path), "header": hdr, "data_start": base}
    if rev:
        ent["revision"] = rev
    _record_streamed(out, fn, ent)
    os.remove(path)
    log(f"   shard {fn} ({size / 1e9:.2f} GB) -> {spec['repo']}/{rp}" + (f" @ {rev}" if rev else "")
        + ": confirmed on the Hub, freed locally")


def staged_revisions(entries):
    """{(repo, staging branch)} the streamed entries ({file: entry}) still sit on."""
    return sorted({(e["repo"], e["revision"]) for e in entries.values() if e.get("revision")})


def promote_staging(api, entries, log=print):
    """A verified streamed rung onto main: every shard copied from its staging branch onto the default branch in ONE
    commit (CommitOperationCopy -- the Hub copies an LFS / Xet file by its content id server side, so nothing is
    re-uploaded and the shards' local copies are not needed), confirmed on main (size + sha256 recorded at upload),
    then the staging branch deleted. Raises with main unchanged (the commit is atomic) when the copy cannot be made,
    and with the branch kept when main does not hold every shard intact afterwards."""
    import pollard_hubcheck as H
    from huggingface_hub import CommitOperationCopy
    staged = [e for e in entries.values() if e.get("revision")]
    for repo, rev in staged_revisions(entries):
        mine = [e for e in staged if e["repo"] == repo and e["revision"] == rev]
        ops = [CommitOperationCopy(src_path_in_repo=e["path_in_repo"], path_in_repo=e["path_in_repo"], src_revision=rev)
               for e in mine]
        api.create_commit(repo_id=repo, operations=ops, commit_message=f"{len(ops)} verified shard(s) from {rev}")
        bad = H.mismatches(api, repo, [(e["path_in_repo"], {"size": e["size"], "sha256": e["sha256"],
                                                             "git_sha1": e.get("git_sha1")}) for e in mine])
        if bad:
            raise RuntimeError(f"{repo}: the shards copied from {rev} are not intact on main ({bad[0]}) -- the branch is "
                               "kept")
        discard_staging(api, repo, rev, log=log)
        log(f"   {len(ops)} shard(s) {repo}@{rev} -> main (copied on the Hub, confirmed), branch deleted")


def discard_staging(api, repo, rev, log=print):
    """Delete a staging branch (a rung that failed its gate, or one already copied to main). Never main itself."""
    if not rev or not str(rev).startswith(STAGING_PREFIX):
        raise ValueError(f"{rev!r} is not a Pollard staging branch -- refusing to delete it")
    try:
        api.delete_branch(repo, branch=rev)
    except Exception as e:                                        # noqa: BLE001 -- already gone is fine
        if "404" not in str(e) and "not found" not in str(e).lower():
            raise
    log(f"   staging branch {repo}@{rev} deleted")


def hub_range(repo, path_in_repo, start, n, retries=3, revision=None):
    """`n` bytes of a file on the Hub from `start` (an HTTP range request; the token comes from the usual place), on
    the branch `revision` (a streamed rung's staging branch) or the default one."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session
    url = hf_hub_url(repo, path_in_repo, revision=revision) if revision else hf_hub_url(repo, path_in_repo)
    last = None
    for _ in range(retries):
        try:
            h = build_hf_headers()
            h["Range"] = f"bytes={start}-{start + n - 1}"
            r = get_session().get(url, headers=h, timeout=600, allow_redirects=True)
            r.raise_for_status()
            data = r.content if r.status_code == 206 else r.content[start:start + n]
            if len(data) == n:
                return data
            last = f"{len(data)} of {n} bytes"
        except Exception as e:                                    # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(2)
    raise RuntimeError(f"could not read {n} bytes of {repo}/{path_in_repo} at {start}: {last}")


class ShardWriter:
    """Writes a checkpoint shard by shard (bounded buffer) and the index at the end. Raw tensors keep their bytes."""

    def __init__(self, out, max_shard_bytes=2 << 30, prefix="model", log=print):
        os.makedirs(out, exist_ok=True)
        self.out, self.max = out, max_shard_bytes
        self.buf, self.size, self.n = {}, 0, 0
        self.weight_map, self.files = {}, []
        self.prefix = prefix
        self.total = 0
        self.log = log
        # pollard release, for a rung bigger than the free disk: every shard of THIS directory is uploaded, confirmed
        # on the Hub and freed before the next one is written (see stream_shard). Names are final from the start
        # (model-00001.safetensors ...): the index maps them, so nothing is renamed after it left.
        self.spec = shard_upload_spec(out)

    def add(self, name, dt, arr):
        import numpy as np
        a = np.ascontiguousarray(arr)
        self.buf[name] = (dt, a)
        self.size += a.nbytes
        self.total += a.nbytes
        if self.size >= self.max:
            self.flush()

    def add_torch(self, name, t, dt=None):
        import numpy as np, torch
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            self.add(name, "BF16", t.view(torch.int16).numpy().view(np.uint16))
        elif t.dtype == torch.float16:
            self.add(name, "F16", t.numpy())
        elif t.dtype == torch.float32:
            self.add(name, "F32", t.numpy())
        elif t.dtype == torch.int32:
            self.add(name, "I32", t.numpy())
        elif t.dtype == torch.uint8:
            self.add(name, dt or "U8", t.numpy())
        elif str(t.dtype) == "torch.float8_e4m3fn":
            self.add(name, "F8_E4M3", t.view(torch.uint8).numpy())
        else:
            self.add(name, dt or "F32", t.float().numpy())

    def flush(self):
        if not self.buf:
            return
        self.n += 1
        fn = f"{self.prefix}-{self.n:05d}.safetensors"
        _st_write(os.path.join(self.out, fn), self.buf, {"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fn
        self.files.append(fn)
        self.buf, self.size = {}, 0
        if self.spec:
            stream_shard(self.out, os.path.join(self.out, fn), self.spec, log=self.log)

    def close(self):
        self.flush()
        if self.spec:                                    # streamed: no renames, and always an index (uploaded LAST)
            with open(os.path.join(self.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                json.dump({"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))},
                          f, indent=2)
            return self.weight_map
        n = len(self.files)
        final = {}
        for i, fn in enumerate(self.files, 1):
            new = f"{self.prefix}-{i:05d}-of-{n:05d}.safetensors" if n > 1 else f"{self.prefix}.safetensors"
            os.replace(os.path.join(self.out, fn), os.path.join(self.out, new))
            final[fn] = new
        self.weight_map = {k: final[v] for k, v in self.weight_map.items()}
        if n > 1:
            with open(os.path.join(self.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                json.dump({"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))},
                          f, indent=2)
        elif os.path.exists(os.path.join(self.out, "model.safetensors.index.json")):
            os.remove(os.path.join(self.out, "model.safetensors.index.json"))
        return self.weight_map


def _st_write(path, tensors, metadata=None):
    import numpy as np
    hdr, off = {}, 0
    for name, (dt, a) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for name, (dt, a) in tensors.items():
            f.write(np.ascontiguousarray(a).tobytes())
    os.replace(tmp, path)


#: non-weight files every derived checkpoint carries over (configs, tokenizer, chat template, custom code, extras)
def copy_side_files(src_dir, out, skip_weights=True):
    import shutil
    shards = {os.path.basename(p) for p in weight_files(src_dir)} if skip_weights else set()
    # a d3 package's MODEL_MANIFEST.json lists the SOURCE's shard hashes: on a rung the d3 runtime would refuse to load
    # it. The decision files (readout.safetensors, decision_config.json) are copied as stored like every side file;
    # the rung's own manifest is written by pollard-d3 ship / pollard-complete --fill once its shards exist.
    d3 = os.path.isfile(os.path.join(src_dir, "decision_config.json"))
    for f in os.listdir(src_dir):
        p = os.path.join(src_dir, f)
        if not os.path.isfile(p) or f in shards or f == "model.safetensors.index.json" or f.endswith(".tmp"):
            continue
        if f.startswith(".") or f in ("README.md",) or (d3 and f == "MODEL_MANIFEST.json"):
            continue
        shutil.copy2(p, os.path.join(out, f))


def copy_raw(src, writer, key, float_dtype="BF16"):
    """Write a logical tensor exactly as stored (its encoded parts too) -- the copy-through for every tensor a lane
    does not transform: towers, talkers, projectors, norms, routers. A tensor a smoothing overlay changed is written
    from its smoothed values in the source's float dtype instead (its bytes are not the source's any more)."""
    if src.modified(key):
        write_float(src, writer, key, float_dtype)
        return
    enc = src.encoding(key)
    if enc is None:
        kind, ref = src.logical[key]
        dt, arr = src.read_raw(ref)
        writer.add(ref, dt, arr)
        return
    for s, name in enc["keys"].items():
        dt, arr = src.read_raw(name)
        writer.add(name, dt, arr)


def write_float(src, writer, key, float_dtype=None, arr=None):
    """Write a logical tensor as a float tensor in the source's own float dtype (BF16 for an encoded source)."""
    import numpy as np
    a = src.read(key) if arr is None else np.asarray(arr, np.float32)
    dt = src.dtype(key)
    dt = dt if dt in FLOAT_DT else (float_dtype or "BF16")
    writer.add(key, dt, to_storage(a, dt))


def to_storage(a, dt):
    """float32 numpy -> the safetensors storage array for dtype dt (BF16 round-to-nearest-even)."""
    import numpy as np
    a = np.ascontiguousarray(a, np.float32)
    if dt == "BF16":
        u = a.view(np.uint32)
        r = ((u >> 16) & 1) + 0x7FFF
        return ((u + r) >> 16).astype(np.uint16)
    if dt == "F16":
        return a.astype(np.float16)
    if dt == "F64":
        return a.astype(np.float64)
    return a


def copy_side_files_of(src, out):
    """Side files (config, tokenizer, code, extras) of a Source: from the base checkpoint, then any the overlay
    folder adds on top."""
    copy_side_files(src.base_dir, out)
    if src.dir != src.base_dir:
        for f in os.listdir(src.dir):
            p = os.path.join(src.dir, f)
            if os.path.isfile(p) and f not in (OVERLAY_JSON, OVERLAY_ST) and not f.endswith(".safetensors"):
                import shutil
                shutil.copy2(p, os.path.join(out, f))


# ----------------------------------------------------------------------------------------------- rung readers + streamed fidelity


class RungReader:
    """The float weights a quantized rung stores, by the SOURCE checkpoint's logical names, read one tensor at a time:
    a GPTQ v1 module decoded from qweight / qzeros / scales, a Pollard MX module decoded from its sidecar's atom, a
    compressed-tensors NVFP4 / FP8 module decoded by the Source reader, an EXLR8 rung decoded layer by layer from its
    trellis records. Anything the rung stores plainly is read as stored."""

    def __init__(self, rung_dir):
        self.dir = rung_dir
        self.kind = None
        self._layer_cache = (None, {})
        if os.path.isfile(os.path.join(rung_dir, "pollard-exlr8.json")):
            self.kind = "exlr8"
            self._vocab = None
            return
        self.src = Source(rung_dir)
        q = self.src.qcfg or {}
        self.gptq = q if q.get("quant_method") == "gptq" else None
        self.mx = None
        side = os.path.join(rung_dir, "pollard_mx.json")
        if os.path.isfile(side):
            with open(side, encoding="utf-8") as f:
                self.mx = json.load(f)
        self.kind = "gptq" if self.gptq else ("pollard-mx" if self.mx else "checkpoint")

    def _gptq_bits(self, mod):
        import pollard_gptqpack as G
        return G.module_bits(self.gptq, mod)

    def read_torch(self, key, dtype=None):
        import numpy as np, torch
        if self.kind != "exlr8" and key in self.src.logical and not self._quantized(key):
            return self.src.read_torch(key, dtype)          # stored plainly: as stored, no float32 detour
        a = self.read(key)
        t = torch.from_numpy(np.ascontiguousarray(a))
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t

    def _quantized(self, key):
        mod = key[:-len(".weight")] if key.endswith(".weight") else None
        return bool(mod and ((self.gptq and f"{mod}.qweight" in self.src.entries)
                             or (self.mx and mod in (self.mx.get("modules") or {}))))

    def read(self, key):
        import numpy as np
        if self.kind == "exlr8":
            return self._exlr8(key)
        mod = key[:-len(".weight")] if key.endswith(".weight") else None
        if self.gptq and mod and f"{mod}.qweight" in self.src.entries:
            import pollard_gptqpack as G
            t = {s: self.src.read_raw(f"{mod}.{s}")[1] for s in ("qweight", "qzeros", "scales", "g_idx")}
            return G.dequant_module(t, self._gptq_bits(mod), t["g_idx"].shape[0]).astype(np.float32)
        if self.mx and mod and mod in (self.mx.get("modules") or {}):
            import pollard_mxfp6 as M6
            atom = self.mx["modules"][mod]
            parts = {}
            for name in self.src.entries:
                if name.startswith(mod + ".") and "." not in name[len(mod) + 1:]:
                    parts[name[len(mod) + 1:]] = self.src.read_raw(name)[1]
            if atom in M6.SCHEME_WEIGHTS:
                return M6.decode_scheme_parts(parts, atom)
            return M6.decode_atom(parts, atom)
        if key in self.src.logical:
            return self.src.read(key)
        raise KeyError(f"{self.dir}: no tensor for {key}")

    def _exlr8(self, key):
        import pollard_exlr8 as X
        dense = os.path.join(self.dir, "dense")
        m = re.search(r"\.(\d+)\.", key)
        meta = json.load(open(os.path.join(self.dir, "pollard-exlr8.json"), encoding="utf-8"))
        prefix = meta.get("tensor_prefix") or ""
        if not prefix:
            for fn in os.listdir(dense):
                if fn.startswith("exlr8-dense-"):
                    prefix = X.SafeFile(os.path.join(dense, fn)).meta.get("tensor_prefix", "model.layers.")
                    break
        if key.startswith(prefix) and m:
            L = int(key[len(prefix):].split(".", 1)[0])
            if self._layer_cache[0] != L:
                f = X.SafeFile(os.path.join(dense, f"exlr8-dense-{L:03d}.safetensors"))
                p = prefix + f"{L}."
                self._layer_cache = (L, dict(X.load_matrices(f, lambda n, p=p: p + n).items()))
            if key in self._layer_cache[1]:
                return self._layer_cache[1][key]
        if self._vocab is None:
            self._vocab = {}
            for fn in ("exlr8-vocab.safetensors", "exlr8-extra-model.safetensors"):
                fp = os.path.join(dense, fn)
                if not os.path.exists(fp):
                    continue
                f = X.SafeFile(fp)
                names = json.loads(f.meta.get("hf_names") or "{}") if fn.startswith("exlr8-vocab") else {}
                for n, v in X.load_matrices(f, lambda n: n).items():
                    stem = n[:-len(".weight")] if n.endswith(".weight") and n[:-len(".weight")] in names else None
                    self._vocab[(names[stem] + ".weight") if stem else names.get(n, n)] = v
        if key in self._vocab:
            return self._vocab[key]
        raise KeyError(f"{self.dir}: no tensor for {key}")


def decoded_bytes(rung_dir, bytes_per_param=2):
    """Bytes the rung takes once every quantized module is decoded to a plain float model (bf16 by default), from the
    safetensors headers alone: a GPTQ module is in_features x out_features, an encoded one its logical shape."""
    rd = RungReader(rung_dir)
    if rd.kind == "exlr8":
        raise ValueError("an EXLR8 rung: pollard_exlr8.decoded_bytes")
    src, n = rd.src, 0
    hidden = set()
    for k, ent in src.entries.items():
        if k.endswith(".qweight") and rd.gptq:
            mod = k[:-len(".qweight")]
            gi = src.entries.get(mod + ".g_idx")
            n += int(_prod(gi[2]) if gi else 0) * int(ent[2][-1])
            hidden.update(f"{mod}.{x}" for x in ("qweight", "qzeros", "scales", "g_idx"))
    for k in src.logical:
        if k in hidden:
            continue
        try:
            n += int(_prod(src.shape(k)))
        except Exception:                                                # noqa: BLE001 -- a part we cannot size
            continue
    return n * bytes_per_param


def decoded_hf_model(rung_dir, device="cpu", trust_remote_code=False, dtype=None, log=None):
    """A plain transformers model of this rung with NO quantization loader: the architecture from the rung's own
    config, every module read through RungReader -- a GPTQ module dequantized at the bits ITS `dynamic` rule gives it,
    a compressed-tensors NVFP4 / FP8 or Pollard-MX module decoded from its bytes, everything else as stored.

    Why it exists: transformers' GPTQ loader (optimum's GPTQQuantizer.convert_model) builds every linear at the BASE
    bits and never reads `dynamic`, so a Pollard 4/8 mix loaded with from_pretrained is noise (KLD 14, top-1 0% on
    Qwen2.5-0.5B, 2026-10-17) while gptqmodel and vLLM honour the map. Weight-only formats, so this is the math those
    runtimes do with the same bytes, on any device, with no GPTQ kernel installed."""
    import numpy as np, torch
    from transformers import AutoConfig, AutoModelForCausalLM
    rd = RungReader(rung_dir)
    cfg = AutoConfig.from_pretrained(rung_dir, trust_remote_code=trust_remote_code)
    for sub in (cfg, getattr(cfg, "text_config", None)):
        if sub is not None and hasattr(sub, "quantization_config"):
            delattr(sub, "quantization_config")
    if dtype is None:
        dtype = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None) or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, trust_remote_code=trust_remote_code)
    except TypeError:                                                    # transformers 4.x spelling
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, trust_remote_code=trust_remote_code)
    names = set(rd.src.logical) if rd.kind != "exlr8" else set()
    if rd.gptq:
        names |= {k[:-len(".qweight")] + ".weight" for k in rd.src.entries if k.endswith(".qweight")}
    by_tail = {}
    for n in names:                                                      # a wrapper prefix the model names apart
        by_tail.setdefault(n.split(".", 1)[-1], n)
    sd, missing = model.state_dict(), []
    with torch.no_grad():
        for key, t in sd.items():
            name = key if key in names else by_tail.get(key.split(".", 1)[-1])
            if name is None:
                if not (key.endswith("lm_head.weight") or "rotary" in key):
                    missing.append(key)
                continue
            a = np.asarray(rd.read(name))
            if tuple(a.shape) != tuple(t.shape):
                raise SystemExit(f"{rung_dir}: {name} decodes to {tuple(a.shape)}, the model wants {tuple(t.shape)}")
            t.copy_(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(t.dtype))
    if missing:
        raise SystemExit(f"{rung_dir}: no tensor for {missing[:3]}{'...' if len(missing) > 3 else ''}")
    head_stored = any(k.endswith("lm_head.weight") for k in names)
    if getattr(cfg, "tie_word_embeddings", False) or not head_stored:
        if not getattr(cfg, "tie_word_embeddings", False) and any(k.endswith("lm_head.weight") for k in sd):
            raise SystemExit(f"{rung_dir}: no lm_head tensor and the config does not tie it")
        model.tie_weights()
    if log:
        log(f"   decoded {os.path.basename(str(rung_dir).rstrip('/'))} ({rd.kind}) into a plain {dtype} model")
    return model.to(device).eval()


def _final_head(bb):
    """(final norm module or None, output projection module) of the backbone."""
    tr = bb.text_root
    norm = None
    for name in ("norm", "norm_f", "final_layernorm", "ln_f", "final_norm", "embedding_norm"):
        m = getattr(tr, name, None)
        if m is not None and hasattr(m, "weight"):
            norm = m
            break
    head = bb.root.get_output_embeddings() if hasattr(bb.root, "get_output_embeddings") else None
    return norm, head


class _LazyRung:
    """The rung's tensors of one block, read (decoded) one at a time AS the loader asks for them and converted to the
    block's dtype straight away: never the whole block decoded to float32 beside its loaded copy."""

    def __init__(self, rd, keys, dtype):
        self.rd, self.keys, self.dtype = rd, set(keys), dtype

    def __contains__(self, k):
        return k in self.keys

    def __getitem__(self, k):
        return self.rd.read_torch(k, self.dtype)


def verify_memory(bb, nrows, seqlen, budget_gb, device="cpu", block=256, log=print):
    """The streamed verify's peak RAM (GB) by component, for BOTH passes -- they run one after the other (the source
    to its final hidden states, cached on disk; then the rung), so the peak is ONE pass: the biggest block while it
    loads (the checkpoint tensors + the module's copy -- an MoE's experts are fused on load), the block inputs AND
    outputs of every eval row (float32), the per-row call tensors (rotary tables, a materialised mask), the
    embeddings, the source's output projection kept for the rung pass, one block of logits, torch itself. fp32 blocks
    when that fits `budget_gb`, else bf16; nothing fits -> OutOfMemoryBudget with the numbers (the eval rows are the
    protocol's and are never cut). On CUDA the block lives in VRAM: RAM holds only its staging copy."""
    shapes = layer_shapes(bb)
    i_max = max(range(len(shapes)), key=lambda i: shapes[i][0]) if shapes else -1
    biggest = shapes[i_max][0] if shapes else 0
    tc = bb.text_config()
    hid = bb.hidden_size() or 1
    vocab = int(getattr(tc, "vocab_size", 0) or 0)
    if not vocab:
        emb = [k for k in bb.outer_keys if bb.kmap.get(k) and bb.kmap[k][0].endswith(("embed_tokens.weight",
                                                                                   "embeddings.weight"))]
        vocab = int(bb.src.shape(emb[0])[0]) if emb else 0
    heads = max(1, int(getattr(tc, "num_attention_heads", 1) or 1))
    head_dim = int(getattr(tc, "head_dim", 0) or hid // heads)
    acts = nrows * seqlen * hid * 4 * 2 / 1e9                     # block inputs + outputs, float32
    kw = nrows * (2 * seqlen * head_dim * 4 + seqlen * seqlen * 4) / 1e9
    outer_n = sum(_prod(bb.src.shape(k)) for k in bb.outer_keys)
    logits = min(block, seqlen) * vocab * 8 * 4 / 1e9             # 2 sides x (logits + log-softmax), float64
    overhead = 0.8                                                # python + torch + transformers
    cuda = str(device).startswith("cuda")
    plan = None
    for cb in (4, 2):
        layer = biggest * cb / 1e9
        load = layer * (1.0 if cuda else 2.0)
        outer = outer_n * cb / 1e9
        head = vocab * hid * cb / 1e9                             # the source's projection, kept for the rung pass
        total = load + acts + kw + outer + head + logits + overhead
        plan = {"layer": i_max, "layer_weights_gb": round(layer, 2), "load_gb": round(load, 2),
                "activations_gb": round(acts + kw, 2), "outer_gb": round(outer + head, 2),
                "logits_gb": round(logits, 2), "overhead_gb": overhead, "peak_gb": round(total, 2),
                "vram_gb": round(layer * 2 + seqlen * hid * cb * 8 / 1e9, 2) if cuda else 0.0,
                "compute_bytes": cb, "rows": nrows, "seqlen": seqlen, "budget_gb": round(budget_gb, 2),
                "device": str(device)}
        if total <= budget_gb:
            if cb == 2:
                log(f"   memory: bf16 blocks to fit {budget_gb:.1f} GB (peak ~{total:.1f} GB: block {i_max} "
                    f"{layer:.2f} GB, activations {acts + kw:.2f} GB)")
            return plan
    raise OutOfMemoryBudget(
        f"the streamed verify needs ~{plan['peak_gb']:.1f} GB of RAM at its smallest (bf16 blocks: block "
        f"{plan['layer']} = {plan['layer_weights_gb']} GB loading as {plan['load_gb']} GB, {plan['activations_gb']} GB "
        f"of activations for {nrows} x {seqlen} eval tokens, {plan['outer_gb']} GB embeddings + output projection) "
        f"and the budget is {budget_gb:.1f} GB -- raise --max-ram-gb or run it on CUDA (--device cuda)")


def disk_usage(path):
    """shutil.disk_usage of `path`: the one place the streamed lanes read the disk (pollard_mlx's build preflight, remote
    mirror and reserve, pollard_verify's reference logits and this module's hidden-state memmap), so a test sets the
    free space with one monkeypatch instead of depending on the machine it runs on."""
    import shutil
    return shutil.disk_usage(path)


def _state_store(nrows_scored, hid):
    """A float32 memmap for the source's final hidden states (scored positions only) in the workspace cache, and its
    path: the source pass writes it, the rung pass reads it -- the two passes never sit in memory together."""
    import numpy as np
    import tempfile
    try:
        import pollard_workspace as ws
        d = ws.cache_dir(create=True)
    except Exception:                                                     # noqa: BLE001
        d = tempfile.gettempdir()
    need = int(nrows_scored) * int(hid) * 4
    free = disk_usage(d).free
    if free < need + 1e9:
        raise OutOfMemoryBudget(f"disk: the source's final hidden states need {need / 1e9:.2f} GB at {d} (+1 GB "
                                f"margin) and {free / 1e9:.2f} GB is free")
    fd, path = tempfile.mkstemp(prefix="verify-state-", suffix=".f32", dir=d)
    os.close(fd)
    return np.memmap(path, dtype=np.float32, mode="w+", shape=(max(1, int(nrows_scored)), int(hid))), path


def stream_fidelity(source, rung, rows, device="cpu", trust_remote_code="auto", log=print, max_ram_gb=None,
                    starts=None, block=256):
    """{kld, top1, kld_p99, n, ppl, ref_ppl}: the rung against its source over token `rows`, each streamed block by
    block, final norm + output projection applied per row -- the same KL(source || rung) and top-1 agreement every
    lane is gated on, for a model of any size the machine can stream.

    Two passes, ONE in memory at a time (verify_memory plans the peak against `max_ram_gb` / free RAM): the SOURCE,
    block by block (each block's weights freed as soon as every row went through it), to its final hidden states --
    written to a disk memmap, scored positions only -- then the RUNG the same way, its tensors decoded one at a time
    into the block's dtype. The logits are formed per row from the cached source states with the source's own norm +
    projection: the math of the side-by-side pass, without two engines' activations and two decoded blocks resident.

    `starts[i]`: the first scored position of row i (pollard_verify.PROTOCOL scores the second half of each
    BOS-led chunk, as llama-perplexity does); None scores every position. The output projection runs `block`
    positions at a time -- a 2048-token row's float64 logits at a 150k vocabulary are 2.5 GB whole."""
    import gc
    import numpy as np, torch
    bb = Backbone(source, trust_remote_code, log=log)
    rd = RungReader(rung)
    plan = verify_memory(bb, len(rows), max(len(r) for r in rows), ram_budget(max_ram_gb), device=device,
                         block=block, log=log)
    bb.dtype = plan_dtype(plan)
    head_keys = [k for k in bb.other_keys + bb.outer_keys
                 if bb.kmap.get(k) and bb.kmap[k][0].endswith(("lm_head.weight", "output.weight"))]
    tied = not head_keys
    st_of = [int(starts[r]) if starts is not None else 0 for r in range(len(rows))]
    offs = np.cumsum([0] + [max(0, len(r) - st) for r, st in zip(rows, st_of)])
    hid = bb.hidden_size()
    peak = 0

    def blocks(eng, tensors_for=None):
        h = eng.hidden
        for i in range(bb.n_layers):
            eng.hidden = h
            eng.load_layer(i, tensors=tensors_for(i) if tensors_for else None)
            h = eng.run_layer(i)
            eng.hidden = None                          # this block's inputs go now, its weights next
            bb.free_layer(i)
        return h

    # -- pass 1: the source, to its final hidden states on disk
    mm, mm_path = None, None
    try:
        es = Engine(bb, device=device, act_dtype=torch.float32, log=log, cast_blocks=True).record(rows)
        hs_final = blocks(es)
        del es.kwargs
        peak = max(peak, es.peak_rss)
        mm, mm_path = _state_store(offs[-1], hid or hs_final[0].shape[-1])
        for r, h in enumerate(hs_final):
            mm[offs[r]:offs[r + 1]] = h[0, st_of[r]:].numpy()
        mm.flush()
        del hs_final, es
        gc.collect()
        norm, head = _final_head(bb)
        bb.load(bb.outer_keys + head_keys, device)
        if tied and hasattr(bb.root, "tie_weights"):
            bb.root.tie_weights()
        names = {n: p for n, p in bb.root.named_parameters()}
        head_name = next((n for n, p in names.items() if p is head.weight), None)
        norm_name = next((n for n, p in names.items() if norm is not None and p is norm.weight), None)
        rev = {t: k for k, (t, _) in bb.kmap.items()}
        w_s = head.weight.detach().clone()
        n_s = norm.weight.detach().clone() if norm is not None else None
        hk = rev.get(head_name) or next(k for k in bb.outer_keys if bb.kmap[k][0].endswith(("embed_tokens.weight",
                                                                                              "embeddings.weight")))
        # -- pass 2: the rung (its own embeddings / norms / every block), the source states read back per row
        eq = Engine(bb, device=device, act_dtype=torch.float32, log=log, cast_blocks=True).record(
            rows, outer_tensors=_LazyRung(rd, bb.outer_keys, bb.dtype))
        hidden_q = blocks(eq, lambda i: _LazyRung(rd, bb.layer_keys[i], bb.dtype))
        del eq.kwargs
        peak = max(peak, eq.peak_rss)
        w_q = rd.read_torch(hk, w_s.dtype).to(device)
        n_q = rd.read_torch(rev[norm_name], n_s.dtype).to(device) if norm is not None and norm_name in rev else n_s
        softcap = getattr(bb.text_config(), "final_logit_softcapping", None)
        kls, top, nll_s, nll_q = [], [], [], []
        with torch.no_grad():
            for r, hq in enumerate(hidden_q):
                ids = torch.as_tensor(list(rows[r]), dtype=torch.long)
                st = st_of[r]
                hs = torch.from_numpy(np.array(mm[offs[r]:offs[r + 1]]))[None]     # positions st.. of row r
                n = hq.shape[1]
                for a0 in range(st, n, max(1, int(block))):
                    a1 = min(n, a0 + int(block))
                    lg = []
                    for x, w, nw in ((hs[:, a0 - st:a1 - st], w_s, n_s), (hq[:, a0:a1], w_q, n_q)):
                        x = x.to(device, bb.dtype)
                        if norm is not None:
                            norm.weight.data = nw
                            x = norm(x)
                        logits = torch.nn.functional.linear(x.to(w.dtype), w)[0]
                        if softcap is not None:                 # Gemma2/3 apply this in their native LM forward
                            logits = torch.tanh(logits / softcap) * softcap
                        lg.append(logits.double())
                    a = torch.log_softmax(lg[0], -1)
                    b = torch.log_softmax(lg[1], -1)
                    kls.append((a.exp() * (a - b)).sum(-1).cpu().numpy())
                    top.append((a.argmax(-1) == b.argmax(-1)).cpu().numpy())
                    nxt = ids[a0 + 1:a1 + 1].to(a.device)          # position t predicts ids[t + 1]
                    m = len(nxt)
                    if m:
                        nll_s.append((-a[torch.arange(m), nxt]).cpu().numpy())
                        nll_q.append((-b[torch.arange(m), nxt]).cpu().numpy())
                    del lg, a, b
                hidden_q[r] = None                               # this row is scored: its states go
        if norm is not None:
            norm.weight.data = n_s
    finally:
        if mm is not None:
            del mm
        if mm_path:
            try:
                os.remove(mm_path)
            except OSError:
                pass
    kl = np.concatenate(kls)
    ns = np.concatenate(nll_s) if nll_s else np.zeros(0)
    nq = np.concatenate(nll_q) if nll_q else np.zeros(0)
    return {"kld": float(kl.mean()), "top1": float(np.concatenate(top).mean() * 100.0),
            "kld_p99": float(np.percentile(kl, 99)), "n": int(kl.size), "how": "streamed",
            "ppl": float(np.exp(nq.mean())) if nq.size else None,
            "ref_ppl": float(np.exp(ns.mean())) if ns.size else None,
            "peak_rss_gb": round(max(peak, peak_rss_bytes()) / 1e9, 2),
            "memory_plan": plan}


# ----------------------------------------------------------------------------------------------- text extraction

#: a wrapper's language model that IS a standard causal LM on text, verified token for token on a tiny random
#: checkpoint (tests/test_stream_lanes.py): {wrapper model_type: how to lift it out}. Used where a consumer reads
#: files itself and knows only the plain class (exllamav3).
TEXT_EQUIV = {
    "qwen3_omni_moe": {"arch": "Qwen3MoeForCausalLM", "model_type": "qwen3_moe",
                       "config": ("thinker_config", "text_config"),
                       "renames": [("thinker.model.", "model."), ("thinker.lm_head.", "lm_head.")],
                       "rope_drop": ("mrope_section", "mrope_interleaved"),
                       "why": "Qwen3-Omni's thinker language model is Qwen3-MoE; its M-RoPE on text-only positions "
                              "(equal on all three axes) is plain RoPE"},
}


def text_equivalent(model_dir):
    cfg = config_json(model_dir)
    return TEXT_EQUIV.get(cfg.get("model_type"))


def extract_text(model_dir, out, log=print):
    """Lift a wrapper's language model out as the plain causal LM TEXT_EQUIV names: its config, its tensors renamed
    (streamed, byte for byte), the tokenizer files. Towers / talkers are left behind -- the result is text-only."""
    spec = text_equivalent(model_dir)
    if not spec:
        raise SystemExit(f"{model_dir}: no plain causal-LM equivalent recorded for its language model")
    src = Source(model_dir)
    cfg = config_json(model_dir)
    tc = cfg
    for k in spec["config"]:
        tc = tc[k]
    tc = dict(_decode_json_floats(tc))
    tc["model_type"] = spec["model_type"]
    tc["architectures"] = [spec["arch"]]
    for rk in ("rope_parameters", "rope_scaling"):
        if isinstance(tc.get(rk), dict):
            tc[rk] = {k: v for k, v in tc[rk].items() if k not in spec["rope_drop"]}
    for k in ("bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings"):
        if tc.get(k) is None and cfg.get(k) is not None:
            tc[k] = cfg[k]
    w = ShardWriter(out)
    n = 0
    for key in src.logical:
        new = None
        for a, b in spec["renames"]:
            if key.startswith(a):
                new = b + key[len(a):]
                break
        if new is None:
            continue
        if src.encoding(key) is not None or src.modified(key):
            dt = "BF16"
            w.add(new, dt, to_storage(src.read(key), dt))
        else:
            dt, arr = src.read_raw(src.logical[key][1])
            w.add(new, dt, arr)
        n += 1
    w.close()
    copy_side_files_of(src, out)
    json.dump(tc, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), indent=2)
    for f in ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    with open(os.path.join(out, ".pollard-text-backbone.json"), "w", encoding="utf-8") as f:
        json.dump({"from": os.path.abspath(model_dir), "as": spec["arch"], "why": spec["why"], "tensors": n}, f)
    log(f"   text backbone of {model_dir} -> {out} as {spec['arch']} ({n} tensors; {spec['why']})")
    return out


# ----------------------------------------------------------------------------------------------- CLI


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="pollard-stream", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="resolve the text backbone and preflight the memory of a streamed build")
    p.add_argument("--model", required=True)
    p.add_argument("--trust-remote-code", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-ram-gb", type=float)
    p.add_argument("--budget-gb", type=float, help="plan against this RAM budget (another machine's) instead of "
                   "what is free here")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    bb = Backbone(a.model, a.trust_remote_code)
    d = bb.describe()
    plan = fit_memory(bb, a.nsamples, a.seqlen, a.budget_gb or ram_budget(a.max_ram_gb))
    if a.json:
        print(json.dumps({"backbone": d, "memory": plan}, indent=1))
        return 0
    print(f"== pollard-stream plan :: {a.model}")
    for k, v in d.items():
        print(f"   {k:18s} {v}")
    print(f"   memory: peak ~{plan['peak_gb']} GB (block {plan['layer']}: {plan['layer_weights_gb']} GB weights + "
          f"{plan['statistics_gb']} GB statistics; activations {plan['activations_gb']} GB; embeddings "
          f"{plan['outer_gb']} GB) with {plan['nsamples']} x {plan['seqlen']} calibration tokens, "
          f"{'bf16' if plan['compute_bytes'] == 2 else 'fp32'} blocks, {plan['expert_chunks']} expert chunk(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
