#!/usr/bin/env python3
"""pollard-complete -- does a lane's output carry the WHOLE model?

Every lane ships the whole checkpoint: the language model at the lane's format, and every other component the source
has -- vision tower, audio tower, modality projector, speech output (talker / code2wav), side weights such as
clef-omni's joint head -- plus the processor / preprocessor configs a runtime needs to feed them. A lane that drops
any of it fails here with the missing parts listed. Nothing is ever silently text-only.

  pollard-complete --source <hf dir> --out <lane output dir> --lane gptq|mx|exlr8|exl3|mlx|gguf
  pollard-complete --source <hf dir> --inventory          # what the source has besides its language model
  pollard-complete --source <hf dir> --out <dir> --lane gguf --fill
      write what a lane cannot hold in its own format as source-precision sidecars next to it
      (components-<name>.safetensors + the processor files), then check again

What counts as "carried", per lane:
  * gptq / mx / exlr8 / exl3: every source tensor of every component is in the output under its own name (as stored,
    or as that module's quantized parts: qweight / trellis / weight_packed / ...), plus every processor file and side
    weight file of the source
  * mlx: the output holds each component (mlx-vlm renames and re-lays tensors, so a component is checked by presence,
    not name for name), plus the processor files
  * gguf: vision / audio towers and the projector are in an mmproj-*.gguf that declares that encoder
    (clip.has_vision_encoder / clip.has_audio_encoder); anything a GGUF cannot hold (speech output, side weights)
    sits next to it as a source-precision safetensors sidecar. The processor files ride inside the mmproj.

Exit 0 = complete, 1 = incomplete (missing parts printed), 2 = usage error.
"""
import argparse, json, os, re, struct, sys

#: non-text components, first match wins (a merger INSIDE the vision tower belongs to the tower)
COMPONENTS = (
    ("vision", re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder|vision_embed_tokens|image_encoder"
                          r"|image_tower|vit|radio_model|siglip|patch_embed)\.", re.I)),
    ("audio", re.compile(r"(^|\.)(audio_tower|audio_encoder|audio_model|speech_encoder|sound_encoder"
                         r"|whisper_encoder|audio_embed)\.", re.I)),
    ("projector", re.compile(r"(^|\.)(multi_modal_projector|mm_projector|mm_proj|mlp1|sound_projection"
                             r"|audio_projection|vision_projection|modality_projector|connector)\.", re.I)),
    ("speech_out", re.compile(r"^(talker|code2wav|token2wav|speech_decoder|vocoder|mimi)\.", re.I)),
)

#: files a runtime needs to feed the non-text inputs (and the chat template that places them)
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
                   "feature_extractor_config.json", "image_processor_config.json", "audio_processor_config.json",
                   "chat_template.json", "chat_template.jinja", "generation_config.json")
PROCESSOR_CODE_RE = re.compile(r"^(processing|image_processing|video_processing|feature_extraction|audio_processing)"
                               r"_.*\.py$")

#: a quantized module's stored parts, by format (a source tensor `<m>.weight` is carried if `<m>.<part>` is there)
QUANT_PARTS = ("weight", "qweight", "qzeros", "scales", "g_idx", "trellis", "suh", "svh", "su", "sv", "mcg", "mul1",
               "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_global_scale",
               "weight_shape", "exlr8")

GGUF_LANES = ("gguf",)
PRESENCE_LANES = ("mlx",)
LANES = ("gptq", "mx", "exlr8", "exl3", "mlx", "gguf")


def component_of(key):
    for name, rx in COMPONENTS:
        if rx.search(key):
            return name
    return None


def _st_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _weight_shards(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f).get("weight_map") or {}
        return sorted({os.path.join(model_dir, v) for v in wm.values()})
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    return sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))


def _base_dir(model_dir):
    """A Pollard smoothing overlay points at its base checkpoint (the towers are the base's)."""
    p = os.path.join(model_dir, "pollard-smooth.json")
    if os.path.isfile(p):
        try:
            base = json.load(open(p, encoding="utf-8")).get("base")
            if base and os.path.isdir(base):
                return base
        except (OSError, ValueError):
            pass
    return model_dir


def inventory(src):
    """What the source has besides its language model:
    {"components": {name: {key: shape}}, "processor_files": [...], "side_weights": {file: [keys]}}"""
    base = _base_dir(src)
    shards = _weight_shards(base)
    comps = {}
    for p in shards:
        for k, info in _st_keys(p).items():
            c = component_of(k)
            if c:
                comps.setdefault(c, {})[k] = info.get("shape")
    files = sorted({f for d in (base, src) for f in os.listdir(d)
                    if os.path.isfile(os.path.join(d, f)) and (f in PROCESSOR_FILES or PROCESSOR_CODE_RE.match(f))})
    shard_names = {os.path.basename(p) for p in shards}
    side = {}
    for f in sorted(os.listdir(base)):
        if f.endswith(".safetensors") and f not in shard_names and not f.startswith("pollard-smooth"):
            side[f] = sorted(_st_keys(os.path.join(base, f)))
    return {"components": comps, "processor_files": files, "side_weights": side}


def _out_tensors(out):
    """{key: file} over every safetensors under the output (a lane may nest: EXLR8's store/base, dense/)."""
    keys = {}
    for root, _, fs in os.walk(out):
        for f in fs:
            if f.endswith(".safetensors"):
                p = os.path.join(root, f)
                try:
                    for k in _st_keys(p):
                        keys.setdefault(k, p)
                except (OSError, ValueError, struct.error):
                    continue
    return keys


def _out_files(out):
    return {f for _, _, fs in os.walk(out) for f in fs}


def _carried(key, out_keys):
    if key in out_keys:
        return True
    mod = key.rsplit(".", 1)[0] if "." in key else key
    return any(f"{mod}.{p}" in out_keys for p in QUANT_PARTS)


def _gguf_encoders(path):
    """(has vision, has audio) declared by an mmproj GGUF: its clip.has_*_encoder keys, else its tensor prefixes."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    rd = GGUFReader(path)
    flags = {}
    for name in ("clip.has_vision_encoder", "clip.has_audio_encoder"):
        f = rd.fields.get(name)
        if f is not None:
            flags[name] = bool(f.parts[f.data[0]][0])
    names = [t.name for t in rd.tensors]
    vision = flags.get("clip.has_vision_encoder", any(n.startswith("v.") for n in names))
    audio = flags.get("clip.has_audio_encoder", any(n.startswith("a.") for n in names))
    return vision, audio


def check(src, out, lane):
    """{"ok": bool, "missing": [str], "carried": [str], "lane": lane} -- `missing` names every component, processor
    file and side weight the output lacks, with counts."""
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r} (one of {', '.join(LANES)})")
    inv = inventory(src)
    out_keys = _out_tensors(out)
    out_files = _out_files(out)
    missing, carried = [], []
    comps = inv["components"]
    if lane in GGUF_LANES:
        mmproj = [os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs
                  if f.startswith("mmproj") and f.endswith(".gguf")]
        enc = [e for e in (_gguf_encoders(p) for p in mmproj) if e is not None]
        has_v = any(v for v, _ in enc)
        has_a = any(a for _, a in enc)
        if mmproj and not enc:                      # gguf-py missing: trust the file is there, say so
            has_v = has_a = True
            carried.append("mmproj present (encoders not read: gguf-py not installed)")
        for name, ok in (("vision", has_v), ("audio", has_a), ("projector", has_v or has_a)):
            if name in comps:
                (carried if ok else missing).append(
                    f"{name}: {len(comps[name])} tensors -> " + ("mmproj" if ok else
                                                                 "no mmproj-*.gguf declaring this encoder"))
        if "speech_out" in comps:
            keys = comps["speech_out"]
            lost = [k for k in keys if k not in out_keys]
            (missing if lost else carried).append(
                f"speech_out: {len(keys) - len(lost)}/{len(keys)} tensors in a sidecar"
                + (f" (missing e.g. {lost[0]}; GGUF cannot hold a talker/code2wav: run --fill)" if lost else ""))
    else:
        for name, keys in comps.items():
            if lane in PRESENCE_LANES:
                rx = dict(COMPONENTS)[name]
                n = sum(1 for k in out_keys if rx.search(k))
                (carried if n else missing).append(f"{name}: source {len(keys)} tensors, output {n}")
                continue
            lost = [k for k in keys if not _carried(k, out_keys)]
            (missing if lost else carried).append(
                f"{name}: {len(keys) - len(lost)}/{len(keys)} tensors" + (f" (missing e.g. {lost[0]})" if lost else ""))
        for f in inv["processor_files"]:
            (carried if f in out_files else missing).append(f"file {f}")
    for f, keys in inv["side_weights"].items():
        ok = f in out_files or all(k in out_keys for k in keys)
        (carried if ok else missing).append(f"side weights {f} ({len(keys)} tensors)")
    return {"ok": not missing, "missing": missing, "carried": carried, "lane": lane}


def fill(src, out, lane):
    """Write what the output lacks as source-precision sidecars: the missing component tensors into
    components-<name>.safetensors (names and bytes as the source stores them), the missing processor files and side
    weight files copied. Returns the list of what was written. Never touches the lane's own files."""
    import shutil
    base = _base_dir(src)
    res = check(src, out, lane)
    if res["ok"]:
        return []
    inv = inventory(src)
    out_keys = _out_tensors(out)
    out_files = _out_files(out)
    wrote = []
    for name, keys in inv["components"].items():
        if lane in GGUF_LANES and name in ("vision", "audio", "projector"):
            continue                                 # the mmproj's job: a sidecar would not make llama.cpp read it
        if lane in PRESENCE_LANES:
            rx = dict(COMPONENTS)[name]
            if any(rx.search(k) for k in out_keys):
                continue
            todo = sorted(keys)
        else:
            todo = sorted(k for k in keys if not _carried(k, out_keys))
        if todo:
            path = os.path.join(out, f"components-{name}.safetensors")
            _copy_tensors(base, todo, path)
            wrote.append(os.path.basename(path))
    if lane not in GGUF_LANES:
        for f in inv["processor_files"]:
            if f not in out_files:
                srcf = os.path.join(src, f) if os.path.isfile(os.path.join(src, f)) else os.path.join(base, f)
                shutil.copy2(srcf, os.path.join(out, f))
                wrote.append(f)
    for f in inv["side_weights"]:
        if f not in out_files:
            shutil.copy2(os.path.join(base, f), os.path.join(out, f))
            wrote.append(f)
    return wrote


def _copy_tensors(model_dir, keys, path):
    """Byte-exact copy of `keys` (as the source stores them) into one safetensors file."""
    where = {}
    for p in _weight_shards(model_dir):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, info in hdr.items():
            if k != "__metadata__":
                where[k] = (p, 8 + n, info)
    header, blobs, off = {}, [], 0
    for k in keys:
        p, base, info = where[k]
        a, b = info["data_offsets"]
        with open(p, "rb") as f:
            f.seek(base + a)
            blob = f.read(b - a)
        header[k] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [off, off + len(blob)]}
        blobs.append(blob)
        off += len(blob)
    header["__metadata__"] = {"format": "pt", "pollard": "source-precision component sidecar"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for blob in blobs:
            f.write(blob)
    os.replace(tmp, path)


def report(res, log=print):
    head = "COMPLETE" if res["ok"] else "INCOMPLETE"
    log(f"== pollard-complete [{res['lane']}]: {head}")
    for c in res["carried"]:
        log(f"   ok       {c}")
    for m in res["missing"]:
        log(f"   MISSING  {m}")
    if not res["missing"] and not res["carried"]:
        log("   (text-only source: nothing besides the language model)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--source", required=True, help="the source HF checkpoint folder (or a smoothing overlay)")
    ap.add_argument("--out", help="the lane's output folder")
    ap.add_argument("--lane", choices=LANES)
    ap.add_argument("--inventory", action="store_true", help="list the source's non-text components and exit")
    ap.add_argument("--fill", action="store_true",
                    help="write what is missing as source-precision sidecars, then check again")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    if a.inventory:
        inv = inventory(a.source)
        if a.json:
            print(json.dumps({"components": {k: len(v) for k, v in inv["components"].items()},
                              "processor_files": inv["processor_files"],
                              "side_weights": {k: len(v) for k, v in inv["side_weights"].items()}}, indent=1))
        else:
            for k, v in inv["components"].items():
                print(f"   {k:11s} {len(v)} tensors")
            for f in inv["processor_files"]:
                print(f"   file       {f}")
            for f, ks in inv["side_weights"].items():
                print(f"   side       {f} ({len(ks)} tensors)")
        return 0
    if not (a.out and a.lane):
        ap.error("--out and --lane are required (or --inventory)")
    if a.fill:
        wrote = fill(a.source, a.out, a.lane)
        for w in wrote:
            print(f"   wrote    {w}")
    res = check(a.source, a.out, a.lane)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        report(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
