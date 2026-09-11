#!/usr/bin/env python3
"""dsv41_exl3_splice.py — build the hybrid DeepSeek-V4.1-Flash checkpoint: routed experts in EXL3, everything else byte-for-byte the
ORIGINAL fp8/fp4 checkpoint (EXL3_PORT_PLAN.md (c) and (b.9)). Pure Python (json/struct/os) — no torch, so it runs on the store host.

Inputs
  --src   original checkpoint dir (the fp4 release: fp8+ue8m0 dense, MXFP4 experts, fp8 Engram tables, bf16 misc, mtp.*)
  --exl3  dir with exl3_L{LL}.safetensors (+ exl3_L{LL}.json ledgers) from dsv41_exl3_experts.py
  --recipe (optional) the per-(layer, gu|down) K map used for the cook — cross-checked against the trellis shapes
Output (--out, must be on the SAME filesystem as --src for hardlinks; else --link copy)
  model-000NN-of-000NN.safetensors  same count/numbering as the source (48): shards 1,2,43-48 (vision, embed, norm/head, mtp, Engram)
                                    are hardlinked unchanged; each body shard (one layer per shard, 3..42) is rewritten with its
                                    non-expert tensors copied byte-for-byte + the layer's EXL3 expert tensors appended. So Tony's
                                    preflight `model-00048-of-00048.safetensors` still holds and layers.{1,14}.engram.embed.* stay mapped.
  model.safetensors.index.json      rebuilt (weight_map + metadata.total_size)
  config.json                       copied; quantization_config replaced by the exl3 block (original nested as original_quantization_config)
  quantization_config.json          the same block + tensor_storage: {"layers.N.ffn.experts.E.w1": {"stored_tensors": {"<name>.trellis":
                                    {"shape","n_bytes","dtype"}, ".suh", ".svh", ".mul1"}, "quant_format":"exl3", "bits_per_weight":K,
                                    "mul1_multiplier": 2212286765}} — exactly what cuda-exl3 Exl3Config._parse_module reads (config.py:182-207:
                                    trellis shape -> bits = packed//16, in = k_tiles*16, out = n_tiles*16; presence of .mul1 -> codebook)
  everything else in --src (tokenizer*, chat_template, generation_config, *.py, README) copied

Layers without an exl3 file: error, or with --allow-partial the original body shard is hardlinked (that layer keeps MXFP4 experts;
cuda-exl3 resolves per layer, so a partial artifact is servable once patch P2 delegates non-EXL3 experts to the native path).

usage:
  dsv41_exl3_splice.py --src /share/.../DeepSeek-V4.1-Flash --exl3 /share/.../dsv41exl3 --out /share/.../DeepSeek-V4.1-Flash-EXL3 [--recipe r.json]
  dsv41_exl3_splice.py --verify-only --out DIR          # re-run the consistency checks on an existing artifact
  dsv41_exl3_splice.py --selftest                       # tiny synthetic checkpoint in a tempdir, end to end (no torch needed)
"""
import argparse, glob, json, os, re, shutil, struct, sys, tempfile, time
from collections import Counter, OrderedDict

MUL1_MULT = 0x83DCD12D
LEAVES = ("trellis", "suh", "svh", "mul1")
MATS = ("w1", "w3", "w2")
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(\w+)$")
ST_TO_TORCH = {"I16": "torch.int16", "F16": "torch.float16", "I32": "torch.int32", "BF16": "torch.bfloat16", "F32": "torch.float32",
               "U8": "torch.uint8", "I8": "torch.int8", "F8_E4M3": "torch.float8_e4m3fn", "F8_E5M2": "torch.float8_e5m2", "I64": "torch.int64",
               "BOOL": "torch.bool", "F64": "torch.float64", "U16": "torch.uint16", "U32": "torch.uint32"}
ST_SIZE = {"I16": 2, "F16": 2, "I32": 4, "BF16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "BOOL": 1, "F64": 8, "U16": 2, "U32": 4}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default=""); ap.add_argument("--exl3", default=""); ap.add_argument("--out", default="")
ap.add_argument("--recipe", default="", help="JSON/YAML (layer, gu|down)->K used for the cook; cross-checked against the trellis shapes")
ap.add_argument("--bits", type=int, default=0, help="integer `bits` for the config (default: most common K over all expert tensors)")
ap.add_argument("--layers", default="", help="layers to splice (default: every layer that has an exl3_L##.safetensors)")
ap.add_argument("--allow-partial", action="store_true", help="layers without an exl3 file keep their original (MXFP4) shard")
ap.add_argument("--link", default="auto", choices=["auto", "hard", "copy"], help="how unchanged shards get into --out (auto: hardlink, fall back to copy)")
ap.add_argument("--version", default="dsv41-routeB-1", help="quantization_config.version string")
ap.add_argument("--mirror-base-keys", action="store_true", help="also copy weight_block_size/scale_fmt/expert_dtype/activation_scheme to the top level of the exl3 block")
ap.add_argument("--verify-only", action="store_true"); ap.add_argument("--selftest", action="store_true")
ap.add_argument("--force", action="store_true", help="overwrite existing files in --out")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_range(s):
    out = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part: continue
        a, _, b = part.partition("-"); out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def load_recipe(path, default_bits):
    table, dflt = {}, {"gu": default_bits, "down": default_bits}
    if not path: return lambda L, kind: dflt[kind]
    txt = open(path).read()
    try: raw = json.loads(txt)
    except json.JSONDecodeError:
        import yaml; raw = yaml.safe_load(txt)
    if "default" in raw: dflt.update({k: int(v) for k, v in raw["default"].items() if k in dflt})
    for key, val in (raw.get("layers") or {}).items(): table[int(key)] = {k: int(v) for k, v in dict(val).items()}
    for key, val in raw.items():
        if key in ("default", "layers"): continue
        k = str(key)[len("layers."):] if str(key).startswith("layers.") else str(key); L, _, kind = k.partition(".")
        if not L.isdigit(): continue          # allocator metadata (target_bpw, cook_gu, rows, ...) is not a layer entry
        table.setdefault(int(L), {})[kind] = int(val)
    return lambda L, kind: table.get(int(L), {}).get(kind, dflt[kind])


# ---------------------------------------------------------------------------------------------------------------------
# safetensors header I/O (pure python)
# ---------------------------------------------------------------------------------------------------------------------
def read_header(path):
    """-> (OrderedDict name -> {dtype, shape, data_offsets}, metadata dict, data_start)"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n).decode("utf-8"), object_pairs_hook=OrderedDict)
    meta = h.pop("__metadata__", None) or {}
    return h, meta, 8 + n


def header_bytes(entries, metadata):
    """entries: list of (name, dtype, shape, nbytes) in write order -> (header bytes incl. length prefix, data_start)."""
    h = OrderedDict(); h["__metadata__"] = {str(k): str(v) for k, v in (metadata or {}).items()}
    off = 0
    for name, dtype, shape, nbytes in entries:
        h[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}; off += nbytes
    js = json.dumps(h, separators=(",", ":")).encode("utf-8"); js += b" " * (-len(js) % 8)
    return struct.pack("<Q", len(js)) + js, 8 + len(js), off


def copy_range(fo, src_path, start, nbytes, handles):
    if src_path not in handles: handles[src_path] = open(src_path, "rb")
    fi = handles[src_path]; done = 0
    fo.flush()
    try:
        while done < nbytes:
            n = os.sendfile(fo.fileno(), fi.fileno(), start + done, min(nbytes - done, 1 << 30))
            if n == 0: raise IOError("sendfile returned 0")
            done += n
    except (OSError, AttributeError):
        fi.seek(start + done)
        while done < nbytes:
            b = fi.read(min(nbytes - done, 64 << 20))
            if not b: raise IOError(f"short read from {src_path}")
            fo.write(b); done += len(b)


def write_shard(out_path, tensors, metadata):
    """tensors: list of (name, dtype, shape, src_path, src_abs_start, nbytes) -> writes header + streamed bytes; returns data bytes."""
    hb, data_start, total = header_bytes([(n, d, s, nb) for n, d, s, _, _, nb in tensors], metadata)
    tmp = out_path + ".tmp"; handles = {}
    with open(tmp, "wb") as fo:
        fo.write(hb)
        for name, dtype, shape, src, start, nb in tensors: copy_range(fo, src, start, nb, handles)
    for f in handles.values(): f.close()
    if os.path.getsize(tmp) != data_start + total: raise IOError(f"{out_path}: size {os.path.getsize(tmp)} != {data_start + total}")
    os.replace(tmp, out_path); return total


def link_or_copy(src, dst, mode, force=False):
    if os.path.exists(dst) or os.path.islink(dst):
        if not force and os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src): return "kept"
        os.remove(dst)
    if mode in ("auto", "hard"):
        try: os.link(src, dst); return "hardlink"
        except OSError as e:
            if mode == "hard": raise SystemExit(f"hardlink {src} -> {dst} failed ({e}); use --link copy or put --out on the same filesystem")
    shutil.copyfile(src, dst); return "copy"


# ---------------------------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------------------------
def model_dims(cfg):
    tc = cfg.get("text_config", cfg)
    return int(tc.get("hidden_size", 5120)), int(tc.get("moe_intermediate_size", 2304)), int(tc.get("n_routed_experts", 384)), int(tc.get("num_hidden_layers", 40))


def inspect_exl3_file(path, layer, D, I, E, k_for=None):
    """Validate the per-layer EXL3 file: 12 tensors per expert, shapes as pack_trellis emits, one K for gu and one for down.
    -> (header, meta, data_start, {"gu": K, "down": K})"""
    h, meta, ds = read_header(path); Ks = {"gu": set(), "down": set()}; seen = set()
    for name, info in h.items():
        m = EXPERT_RE.match(name)
        if not m or int(m[1]) != layer: raise SystemExit(f"{path}: unexpected tensor {name}")
        e, mat, leaf = int(m[2]), m[3], m[4]
        if leaf not in LEAVES: raise SystemExit(f"{path}: unexpected leaf {name}")
        seen.add((e, mat, leaf)); k_in, n_out = (D, I) if mat != "w2" else (I, D)
        if leaf == "trellis":
            kt, nt, packed = info["shape"]
            if info["dtype"] != "I16" or kt * 16 != k_in or nt * 16 != n_out or packed % 16 or not 1 <= packed // 16 <= 8:
                raise SystemExit(f"{path}: {name} shape/dtype {info['shape']} {info['dtype']} != ({k_in // 16}, {n_out // 16}, 16K) I16")
            Ks["down" if mat == "w2" else "gu"].add(packed // 16)
        elif leaf == "suh" and (info["dtype"] != "F16" or info["shape"] != [k_in]): raise SystemExit(f"{path}: {name} must be F16 [{k_in}], got {info}")
        elif leaf == "svh" and (info["dtype"] != "F16" or info["shape"] != [n_out]): raise SystemExit(f"{path}: {name} must be F16 [{n_out}], got {info}")
        elif leaf == "mul1" and (info["dtype"] != "I32" or info["shape"] not in ([], [1])): raise SystemExit(f"{path}: {name} must be I32 scalar, got {info}")
    missing = [(e, m, l) for e in range(E) for m in MATS for l in LEAVES if (e, m, l) not in seen]
    if missing: raise SystemExit(f"{path}: {len(missing)} expert tensors missing, e.g. {missing[:4]}")
    for kind in Ks:
        if len(Ks[kind]) != 1: raise SystemExit(f"{path}: {kind} tensors have mixed K {sorted(Ks[kind])} — cuda-exl3 needs one K per (layer, gate+up) and (layer, down)")
    K = {k: v.pop() for k, v in Ks.items()}
    if k_for is not None:
        for kind in K:
            if k_for(layer, kind) != K[kind]: raise SystemExit(f"{path}: {kind} K={K[kind]} but recipe says {k_for(layer, kind)}")
    return h, meta, ds, K


def body_layer_of(keys):
    """layer index if this shard holds a layer's routed experts, else None"""
    ls = {int(m[1]) for k in keys for m in [EXPERT_RE.match(k)] if m}
    if len(ls) > 1: raise SystemExit(f"shard holds experts of several layers {sorted(ls)} — the one-layer-per-shard assumption is broken")
    return ls.pop() if ls else None


def build(a):
    src, out, ex = a.src.rstrip("/"), a.out.rstrip("/"), a.exl3.rstrip("/")
    if not (src and out and ex): raise SystemExit("--src, --exl3 and --out are required")
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json"))); D, I, E, L = model_dims(cfg)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json"))); wm = idx["weight_map"]
    by_shard = OrderedDict()
    for k, s in wm.items(): by_shard.setdefault(s, []).append(k)
    shards = sorted(by_shard)
    shard_layer = {s: body_layer_of(by_shard[s]) for s in shards}
    avail = {int(re.search(r"exl3_L(\d+)\.safetensors$", p)[1]): p for p in glob.glob(os.path.join(ex, "exl3_L*.safetensors"))}
    want = parse_range(a.layers) if a.layers else sorted(avail)
    missing = [l for l in range(L) if l not in want or l not in avail]
    if missing and not a.allow_partial:
        raise SystemExit(f"{len(missing)} layers have no EXL3 file (e.g. {missing[:8]}); cook them or pass --allow-partial")
    for l in want:
        if l not in avail: raise SystemExit(f"--layers names {l} but {ex}/exl3_L{l:02d}.safetensors is missing")
    k_for = load_recipe(a.recipe, 0) if a.recipe else None
    log(f"src {src}: {len(wm)} tensors in {len(shards)} shards; body layers {sorted(l for l in shard_layer.values() if l is not None)[:3]}..; "
        f"EXL3 layers to splice: {len(want)} of {L}" + (f" (partial: {len(missing)} keep MXFP4)" if missing else ""))
    new_wm, total_size, storage, layer_K, per_layer_bytes = {}, 0, OrderedDict(), {}, {}
    for s in shards:
        dst = os.path.join(out, s); layer = shard_layer[s]
        if layer is None or layer not in want:
            how = link_or_copy(os.path.join(src, s), dst, a.link, a.force)
            h, _, _ = read_header(os.path.join(src, s))
            for k in by_shard[s]: new_wm[k] = s; total_size += h[k]["data_offsets"][1] - h[k]["data_offsets"][0]
            log(f"  {s}: unchanged ({how}), {len(by_shard[s])} tensors" + (f" [layer {layer} keeps MXFP4 experts]" if layer is not None else ""))
            continue
        exh, exmeta, exds, K = inspect_exl3_file(avail[layer], layer, D, I, E, k_for); layer_K[layer] = K
        sh, smeta, sds = read_header(os.path.join(src, s)); tensors = []; dropped = 0
        for name, info in sh.items():
            if EXPERT_RE.match(name): dropped += 1; continue          # layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}
            b, e_ = info["data_offsets"]; tensors.append((name, info["dtype"], info["shape"], os.path.join(src, s), sds + b, e_ - b))
        if dropped != E * 3 * 2: log(f"  WARNING {s}: dropped {dropped} expert tensors, expected {E * 3 * 2}")
        for name in sorted(exh, key=lambda n: (int(EXPERT_RE.match(n)[2]), MATS.index(EXPERT_RE.match(n)[3]), LEAVES.index(EXPERT_RE.match(n)[4]))):
            info = exh[name]; b, e_ = info["data_offsets"]; tensors.append((name, info["dtype"], info["shape"], avail[layer], exds + b, e_ - b))
            mod = name.rsplit(".", 1)[0]; ent = storage.setdefault(mod, {"stored_tensors": OrderedDict(), "quant_format": "exl3"})
            ent["stored_tensors"][name] = {"shape": list(info["shape"]), "n_bytes": e_ - b, "dtype": ST_TO_TORCH.get(info["dtype"], info["dtype"])}
            if name.endswith(".trellis"): ent["bits_per_weight"] = info["shape"][-1] // 16; ent["mul1_multiplier"] = MUL1_MULT
        kept = False
        if os.path.exists(dst) and not a.force:
            h2, _, _ = read_header(dst); kept = list(h2) == [t[0] for t in tensors] and os.path.getsize(dst) == read_header(dst)[2] + sum(t[5] for t in tensors)
        nb = sum(t[5] for t in tensors) if kept else write_shard(dst, tensors, {"format": "pt", "layer": str(layer), "exl3_gu_bits": str(K["gu"]), "exl3_down_bits": str(K["down"])})
        for t in tensors: new_wm[t[0]] = s
        total_size += nb; per_layer_bytes[layer] = nb
        log(f"  {s}: layer {layer} {'already spliced (same tensor list, kept; --force rewrites)' if kept else 'rewritten'}: {len(tensors) - len(exh)} native tensors + {len(exh)} EXL3 (gu K={K['gu']}, down K={K['down']}), {nb / 1e9:.2f} GB")
    # index
    json.dump({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(new_wm.items()))}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    # bits: most common K over expert tensors (integer — cuda-exl3 / GLM lesson), average over bytes for information
    allK = Counter(); numel = 0; ebytes = 0
    for l, K in layer_K.items(): allK[K["gu"]] += 2 * E; allK[K["down"]] += E; numel += 3 * E * D * I; ebytes += per_layer_bytes.get(l, 0) - 0
    bits = a.bits or (allK.most_common(1)[0][0] if allK else 4)
    orig_q = cfg.get("quantization_config") or {}
    qcfg = OrderedDict([("quant_method", "exl3"), ("version", a.version), ("bits", int(bits)), ("head_bits", 16), ("codebook", "mul1"), ("out_scales", "always"),
                        ("expert_bits", {str(l): layer_K[l] for l in sorted(layer_K)}),
                        ("quantized_modules", "layers.N.ffn.experts.E.{w1,w3,w2} only; every other tensor is the original checkpoint's"),
                        ("original_quantization_config", orig_q)])
    if a.mirror_base_keys:
        for k in ("weight_block_size", "scale_fmt", "expert_dtype", "activation_scheme"):
            if k in orig_q: qcfg[k] = orig_q[k]
    led = {}
    for l in layer_K:
        jp = os.path.join(ex, f"exl3_L{l:02d}.json")
        if os.path.exists(jp):
            try: led[l] = json.load(open(jp))
            except Exception: pass
    if led:
        bp = [v["bpw_layer"] for v in led.values() if "bpw_layer" in v]
        if bp: qcfg["average_bits_per_weight"] = round(sum(bp) / len(bp), 4)
        rows = {v.get("dumps") and sum(m.get("tokens", 0) for m in v["dumps"]) for v in led.values()}
        rows.discard(None)
        if rows: qcfg["calibration"] = {"tokens_per_layer": sorted(rows)[-1], "source": "dsv41_capture.py ffn_in dumps (routed-only Hessians)"}
    cfg2 = json.loads(json.dumps(cfg)); cfg2["quantization_config"] = qcfg
    if isinstance(cfg2.get("text_config"), dict) and "quantization_config" in cfg2["text_config"]: cfg2["text_config"]["quantization_config"] = qcfg
    json.dump(cfg2, open(os.path.join(out, "config.json"), "w"), indent=2)
    full = OrderedDict(qcfg); full["tensor_storage"] = storage
    json.dump(full, open(os.path.join(out, "quantization_config.json"), "w"), indent=1)
    # the rest of the source dir
    for fn in sorted(os.listdir(src)):
        sp = os.path.join(src, fn)
        if fn in ("config.json", "quantization_config.json", "model.safetensors.index.json") or fn.startswith("model-") and fn.endswith(".safetensors"): continue
        if os.path.isfile(sp) and (a.force or not os.path.exists(os.path.join(out, fn))): shutil.copyfile(sp, os.path.join(out, fn))
        elif os.path.isdir(sp) and not os.path.exists(os.path.join(out, fn)): shutil.copytree(sp, os.path.join(out, fn))
    log(f"index: {len(new_wm)} tensors, total_size {total_size / 1e9:.1f} GB; tensor_storage {len(storage)} modules; bits={bits} (K histogram {dict(allK)})")
    return verify(out, want, E)


# ---------------------------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------------------------
def verify(out, spliced_layers=None, E=None):
    ok = True
    def check(cond, msg):
        nonlocal ok
        if not cond: ok = False
        log(("  ok   " if cond else "  FAIL ") + msg)
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json"))); wm = idx["weight_map"]
    shards = sorted(set(wm.values())); m = re.match(r"model-(\d+)-of-(\d+)\.safetensors", shards[-1])
    check(m is not None and os.path.exists(os.path.join(out, f"model-{m[2]}-of-{m[2]}.safetensors")), f"last shard model-{m[2] if m else '?'}-of-{m[2] if m else '?'} exists (launcher preflight)")
    check(all(os.path.exists(os.path.join(out, s)) for s in shards), "every shard in the index exists")
    check(int(m[2]) == len(shards) if m else False, f"shard count {len(shards)} == numbering")
    seen, tot = set(), 0
    for s in shards:
        h, meta, ds = read_header(os.path.join(out, s)); offs = sorted(v["data_offsets"] for v in h.values()); end = 0; contiguous = True
        for b, e in offs: contiguous &= (b == end); end = e
        for k, v in h.items():
            b, e = v["data_offsets"]; n = 1
            for d in v["shape"]: n *= d
            contiguous &= (e - b == n * ST_SIZE.get(v["dtype"], 0)) if v["dtype"] in ST_SIZE else True
        size_ok = os.path.getsize(os.path.join(out, s)) == ds + end
        if not (contiguous and size_ok): check(False, f"{s}: contiguous offsets {contiguous}, file size matches {size_ok}")
        keys = set(h); seen |= keys; tot += end
        extra = [k for k in keys if wm.get(k) != s]
        if extra: check(False, f"{s}: {len(extra)} tensors not mapped to this shard in the index, e.g. {extra[:3]}")
    check(seen == set(wm), f"index keys == shard header keys ({len(wm)} tensors)")
    check(tot == idx.get("metadata", {}).get("total_size"), f"metadata.total_size {idx.get('metadata', {}).get('total_size')} == sum of tensor bytes {tot}")
    check(all(k in wm for k in ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale", "layers.14.engram.embed.weight", "layers.14.engram.embed.scale")) or E is not None and E < 8,
          "Engram tables layers.{1,14}.engram.embed.{weight,scale} mapped (vLLM preflight)")
    q = json.load(open(os.path.join(out, "quantization_config.json"))); c = json.load(open(os.path.join(out, "config.json")))
    check(q.get("quant_method") == "exl3" and isinstance(q.get("bits"), int) and q.get("head_bits") == 16 and q.get("codebook") == "mul1" and "tensor_storage" in q, "quantization_config.json: quant_method/bits(int)/head_bits/codebook/tensor_storage")
    check(c.get("quantization_config", {}).get("quant_method") == "exl3" and "tensor_storage" not in c["quantization_config"] and c["quantization_config"].get("original_quantization_config", {}).get("quant_method") in ("fp8", None),
          "config.json quantization_config: exl3 summary with original_quantization_config nested")
    ts = q["tensor_storage"]; bad = 0; layers_in_ts = set()
    for name, ent in ts.items():
        st = ent.get("stored_tensors", {}); tr = st.get(f"{name}.trellis")
        if ent.get("quant_format") != "exl3" or tr is None or len(tr["shape"]) != 3 or tr["shape"][2] // 16 != ent.get("bits_per_weight") or f"{name}.mul1" not in st or ent.get("mul1_multiplier") != MUL1_MULT: bad += 1
        if any(f"{name}.{l}" not in wm for l in LEAVES): bad += 1
        mm = EXPERT_RE.match(name + ".trellis")
        if mm: layers_in_ts.add(int(mm[1]))
    check(bad == 0, f"tensor_storage entries parse the cuda-exl3 way ({len(ts)} modules, {bad} bad)")
    exp_keys = [k for k in wm if EXPERT_RE.match(k)]
    for l in sorted(layers_in_ts):
        ks = [k for k in exp_keys if k.startswith(f"layers.{l}.ffn.experts.")]
        n_native = sum(1 for k in ks if k.endswith((".weight", ".scale"))); n_exl3 = sum(1 for k in ks if k.split(".")[-1] in LEAVES)
        if n_native or (E and n_exl3 != E * 12): check(False, f"layer {l}: {n_native} native expert tensors left, {n_exl3} EXL3 tensors")
    check(True, f"{len(layers_in_ts)} spliced layers, no native expert tensors left in them") if ok else None
    if spliced_layers is not None: check(set(spliced_layers) == layers_in_ts, f"spliced layers {sorted(layers_in_ts)[:5]}.. == requested")
    log("SPLICE-VERIFY-OK" if ok else "SPLICE-VERIFY-FAIL"); return ok


# ---------------------------------------------------------------------------------------------------------------------
# selftest: synthetic tiny checkpoint (D=64, I=32, E=2, L=2) with the real shard topology, spliced end to end
# ---------------------------------------------------------------------------------------------------------------------
def write_raw_safetensors(path, tensors, metadata=None):
    """tensors: list of (name, dtype, shape, bytes)"""
    hb, ds, total = header_bytes([(n, d, s, len(b)) for n, d, s, b in tensors], metadata or {"format": "pt"})
    with open(path, "wb") as f:
        f.write(hb)
        for _, _, _, b in tensors: f.write(b)


def selftest():
    import random
    random.seed(0); D, I, E, L, K = 64, 32, 2, 2, 3; ok = True
    def rb(n): return bytes(random.getrandbits(8) for _ in range(n))
    with tempfile.TemporaryDirectory() as td:
        src, ex, out = [os.path.join(td, d) for d in ("src", "exl3", "out")]; os.makedirs(src); os.makedirs(ex)
        N = L + 3; shard = lambda i: f"model-{i:05d}-of-{N:05d}.safetensors"; wm = {}; tot = 0
        def put(i, tensors):
            nonlocal tot
            write_raw_safetensors(os.path.join(src, shard(i)), tensors)
            for n, _, _, b in tensors: wm[n] = shard(i); tot += len(b)
        put(1, [("embed.weight", "BF16", [8, D], rb(8 * D * 2))])
        for l in range(L):
            ts = [(f"layers.{l}.attn.wq_a.weight", "F8_E4M3", [16, D], rb(16 * D)), (f"layers.{l}.attn.wq_a.scale", "F32", [1, 2], rb(8)),
                  (f"layers.{l}.ffn.gate.weight", "BF16", [E, D], rb(E * D * 2)), (f"layers.{l}.ffn.shared_experts.w1.weight", "F8_E4M3", [I, D], rb(I * D))]
            for e in range(E):
                for m in MATS:
                    o, i_ = (I, D) if m != "w2" else (D, I)
                    ts += [(f"layers.{l}.ffn.experts.{e}.{m}.weight", "U8", [o, i_ // 2], rb(o * i_ // 2)), (f"layers.{l}.ffn.experts.{e}.{m}.scale", "U8", [o, i_ // 32], rb(o * i_ // 32))]
            ts.append((f"layers.{l}.ffn_norm.weight", "BF16", [D], rb(D * 2)))
            put(2 + l, ts)
        put(L + 2, [("norm.weight", "BF16", [D], rb(D * 2)), ("head.weight", "BF16", [8, D], rb(8 * D * 2))])
        put(L + 3, [("layers.1.engram.embed.weight", "F8_E4M3", [4, 256], rb(1024)), ("layers.1.engram.embed.scale", "U8", [4, 8], rb(32)),
                    ("layers.14.engram.embed.weight", "F8_E4M3", [4, 256], rb(1024)), ("layers.14.engram.embed.scale", "U8", [4, 8], rb(32))])
        json.dump({"metadata": {"total_size": tot}, "weight_map": wm}, open(os.path.join(src, "model.safetensors.index.json"), "w"))
        json.dump({"architectures": ["DeepseekV41ForCausalLM"], "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
                   "text_config": {"hidden_size": D, "moe_intermediate_size": I, "n_routed_experts": E, "num_hidden_layers": L}}, open(os.path.join(src, "config.json"), "w"))
        open(os.path.join(src, "tokenizer.json"), "w").write("{}")
        # EXL3 files (layer 0 only at first -> partial), shapes exactly as pack_trellis
        def exl3_file(l):
            ts = []
            for e in range(E):
                for m in MATS:
                    k_in, n_out = (D, I) if m != "w2" else (I, D); nm = f"layers.{l}.ffn.experts.{e}.{m}"
                    ts += [(f"{nm}.trellis", "I16", [k_in // 16, n_out // 16, 16 * K], rb(k_in // 16 * n_out // 16 * 16 * K * 2)), (f"{nm}.suh", "F16", [k_in], rb(k_in * 2)),
                           (f"{nm}.svh", "F16", [n_out], rb(n_out * 2)), (f"{nm}.mul1", "I32", [], struct.pack("<I", MUL1_MULT))]
            write_raw_safetensors(os.path.join(ex, f"exl3_L{l:02d}.safetensors"), ts, {"layer": str(l)})
            json.dump({"layer": l, "bpw_layer": 3.2, "dumps": [{"tokens": 100}]}, open(os.path.join(ex, f"exl3_L{l:02d}.json"), "w"))
        exl3_file(0)
        class A: pass
        a = A(); a.src, a.exl3, a.out, a.recipe, a.bits, a.layers, a.allow_partial, a.link, a.version, a.mirror_base_keys, a.force = src, ex, out, "", 0, "", False, "auto", "t", False, False
        try: build(a); print("  FAIL missing layer without --allow-partial should raise"); ok = False
        except SystemExit as e_: print(f"  ok   refuses partial without flag: {str(e_)[:60]}")
        a.allow_partial = True; ok &= build(a)
        h1, _, _ = read_header(os.path.join(out, shard(3))); ok &= "layers.1.ffn.experts.0.w1.weight" in h1; print(f"  {'ok  ' if 'layers.1.ffn.experts.0.w1.weight' in h1 else 'FAIL'} partial: layer 1 shard kept native")
        exl3_file(1); a.allow_partial = False
        rp = os.path.join(td, "r.json"); json.dump({"default": {"gu": K, "down": K}}, open(rp, "w")); a.recipe = rp
        ok &= build(a)
        # byte identity of a copied dense tensor and of an EXL3 tensor
        def get_bytes(dirp, name):
            wm_ = json.load(open(os.path.join(dirp, "model.safetensors.index.json")))["weight_map"]; h, _, ds = read_header(os.path.join(dirp, wm_[name])); b, e_ = h[name]["data_offsets"]
            with open(os.path.join(dirp, wm_[name]), "rb") as f: f.seek(ds + b); return f.read(e_ - b)
        same = get_bytes(src, "layers.1.attn.wq_a.weight") == get_bytes(out, "layers.1.attn.wq_a.weight") and get_bytes(src, "embed.weight") == get_bytes(out, "embed.weight")
        print(f"  {'ok  ' if same else 'FAIL'} dense tensors byte-identical to the source"); ok &= same
        hx, _, dx = read_header(os.path.join(ex, "exl3_L01.safetensors")); b, e_ = hx["layers.1.ffn.experts.1.w2.trellis"]["data_offsets"]
        with open(os.path.join(ex, "exl3_L01.safetensors"), "rb") as f: f.seek(dx + b); tb = f.read(e_ - b)
        same = tb == get_bytes(out, "layers.1.ffn.experts.1.w2.trellis"); print(f"  {'ok  ' if same else 'FAIL'} EXL3 trellis bytes identical to the cook output"); ok &= same
        q = json.load(open(os.path.join(out, "quantization_config.json"))); ent = q["tensor_storage"]["layers.0.ffn.experts.0.w1"]
        good = ent["bits_per_weight"] == K and ent["stored_tensors"]["layers.0.ffn.experts.0.w1.trellis"]["shape"] == [D // 16, I // 16, 16 * K] and ent["stored_tensors"]["layers.0.ffn.experts.0.w1.trellis"]["dtype"] == "torch.int16" \
            and ent["mul1_multiplier"] == MUL1_MULT and q["bits"] == K and q["original_quantization_config"]["expert_dtype"] == "fp4" and len(q["tensor_storage"]) == L * E * 3
        print(f"  {'ok  ' if good else 'FAIL'} tensor_storage entry shape for cuda-exl3 _parse_module"); ok &= good
        st = os.stat(os.path.join(out, shard(1))); print(f"  {'ok  ' if st.st_nlink == 2 else 'FAIL'} unchanged shard is a hardlink (nlink={st.st_nlink})"); ok &= st.st_nlink == 2
        ok &= os.path.exists(os.path.join(out, "tokenizer.json"))
        # a wrong recipe must be refused
        json.dump({"default": {"gu": K + 1, "down": K}}, open(rp, "w")); a.force = True
        try: build(a); print("  FAIL wrong recipe accepted"); ok = False
        except SystemExit as e_: print(f"  ok   recipe mismatch refused: {str(e_)[:70]}")
    print("SELFTEST-OK" if ok else "SELFTEST-FAIL"); return 0 if ok else 1


def main():
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    if a.verify_only:
        if not a.out: raise SystemExit("--out required"); sys.exit(0 if verify(a.out) else 1)
    sys.exit(0 if build(a) else 1)


if __name__ == "__main__":
    main()
