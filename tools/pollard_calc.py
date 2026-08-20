#!/usr/bin/env python3
"""pollard-calc — know what your hardware can do BEFORE you download 300GB.

Reads a Hugging Face config.json (local path or hub id) plus a hardware
profile, and computes the memory-movement economics of running that model:

  * total / active parameters (MoE-aware: routed + shared experts)
  * bytes read per token at your quantization, cold and cache-assisted
  * throughput floors: flash-streaming, RAM-bandwidth ceiling, resident
  * the verdict: FITS RESIDENT / STREAMING-VIABLE / NEEDS BIGGER TIER

Zero GPU required. Zero downloads beyond the ~10KB config. Estimates, not
gospel: every output states its assumptions — check them against your hardware.

Usage:
  pollard_calc.py --config path/to/config.json
  pollard_calc.py --model moonshotai/Kimi-K3
  pollard_calc.py --model mistralai/Mixtral-8x7B-v0.1 --ram 16 --quant q4
  pollard_calc.py --model Qwen/Qwen3-30B-A3B --ram 16 --flash 3.5 --rambw 120
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import urllib.request


def detect_available_ram_gb():
    """Measure what's ACTUALLY free right now — nameplate RAM lies once the OS,
    the desktop, and everything else take their cut. stdlib only."""
    try:
        if sys.platform == "darwin":
            page = 16384
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            stats = {}
            for line in out.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip().rstrip(".")
                    if v.isdigit():
                        stats[k.strip()] = int(v)
            free = (stats.get("Pages free", 0) + stats.get("Pages inactive", 0)
                    + stats.get("Pages speculative", 0) + stats.get("Pages purgeable", 0))
            return free * page / 1e9
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return None

QUANTS = {  # effective bits per weight, format overheads included
    "f16": 16.0, "q8": 8.5, "q6": 6.6, "q5": 5.5, "q4": 4.6,
    "q3": 3.5, "q2": 2.6,
}


def find_llama_bin(name):
    """Resolve a llama.cpp binary (llama-perplexity, llama-cli, llama-quantize…):
    honor an explicit path, else PATH, else the runtime build install.sh created,
    else common spots. Returns the path or None — so tools AUTO-DETECT what's there
    instead of dead-ending on 'not found' when the binary is right where we put it."""
    import shutil
    if not name:
        return None
    if os.path.sep in name or name.startswith("~"):        # explicit path given
        p = os.path.expanduser(name)
        return p if os.path.exists(p) else None
    hit = shutil.which(name)                                # on PATH
    if hit:
        return hit
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in (os.path.join(repo, "runtime", "llama.cpp", "build", "bin", name),
              os.path.expanduser(f"~/llama.cpp/build/bin/{name}"),
              f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if os.path.exists(c):
            return c
    return None


def _shard_paths(path):
    """A model may be split across shards `<prefix>-00001-of-000NN.gguf`. Given
    ANY shard, return every shard in order (shard 1 first — it holds the full KV
    metadata; later shards hold only their tensor slice). Non-split → [path].
    Reading a single shard against the whole param count is what produced Frank's
    0.00 bpw / 1.6M tok/s garbage — every big model (DeepSeek, Kimi…) is sharded."""
    m = re.search(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(path))
    if not m:
        return [path]
    prefix, total = m.group(1), int(m.group(3))
    d = os.path.dirname(path) or "."
    shards = [os.path.join(d, f"{prefix}-{i:05d}-of-{total:05d}.gguf")
              for i in range(1, total + 1)]
    present = [s for s in shards if os.path.exists(s)]
    return present or [path]


def _read_one_gguf(path):
    """Read ONE GGUF file: (metadata KV dict, tensor_param_sum). stdlib only."""
    _SIMPLE = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
               5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8),
               11: ("q", 8), 12: ("d", 8)}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"ERROR: {path} is not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        def rd_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rd_val(t):
            if t in _SIMPLE:
                fmt, sz = _SIMPLE[t]
                return struct.unpack("<" + fmt, f.read(sz))[0]
            if t == 8:
                return rd_str()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                return [rd_val(et) for _ in range(n)]
            sys.exit(f"ERROR: unknown GGUF value type {t}")

        meta = {}
        for _ in range(n_kv):
            k = rd_str()
            t, = struct.unpack("<I", f.read(4))
            v = rd_val(t)
            if not (isinstance(v, list) and len(v) > 64):  # skip tokenizer blobs
                meta[k] = v
        # tensor table follows the KV section (cursor is already there):
        # exact per-tensor shapes. Summing gives a ground-truth param count
        # for ANY architecture (LLM, DiT, VAE...) with no key conventions.
        total_params = 0
        dcounts = {}
        try:
            for _ in range(n_tensors):
                rd_str()  # tensor name
                nd, = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                dt, = struct.unpack("<I", f.read(4)); f.read(8)  # ggml dtype + offset
                dcounts[dt] = dcounts.get(dt, 0) + 1
                n = 1
                for d in dims:
                    n *= d
                total_params += n
        except Exception:
            total_params = None  # malformed tail: key-based analysis only
    return meta, total_params, dcounts


def read_gguf_meta(path):
    """Shard-aware GGUF metadata reader. Reads the arch KV from shard 1 and sums
    the tensor param count AND file bytes across ALL shards, so multi-shard models
    report their true size instead of one shard's slice."""
    shards = _shard_paths(path)
    meta, param_sum, ok, dcounts = None, 0, True, {}
    for i, sp in enumerate(shards):
        kv, psum, dc = _read_one_gguf(sp)
        if i == 0:
            meta = kv  # full arch KV lives in the first shard
        if psum is None:
            ok = False
        else:
            param_sum += psum
        for t, c in dc.items():
            dcounts[t] = dcounts.get(t, 0) + c
    meta["_tensor_param_sum"] = param_sum if ok else None
    meta["_total_file_bytes"] = sum(os.path.getsize(s) for s in shards)
    meta["_shard_count"] = len(shards)
    # the most common tensor type IS the model's quant, read from the file itself —
    # ground truth that doesn't depend on the (drifting) general.file_type enum.
    meta["_dominant_ggml_type"] = max(dcounts, key=dcounts.get) if dcounts else None
    return meta


def read_gguf_tensor_names(path):
    """Every tensor name across all shards. Used by pollard-fit to catch tensors
    that would fall through to an aggressive base preset (the exotic ones — e.g.
    DeepSeek's indexer_compressor / output_hc_fn — that llama-quantize hard-fails
    on when they get an imatrix-required type without coverage)."""
    _SIMPLE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    names = []
    for sp in _shard_paths(path):
        try:
            with open(sp, "rb") as f:
                if f.read(4) != b"GGUF":
                    continue
                f.read(4)                                   # version
                nt, = struct.unpack("<Q", f.read(8))
                nkv, = struct.unpack("<Q", f.read(8))

                def rd_str():
                    n, = struct.unpack("<Q", f.read(8))
                    return f.read(n)

                def skip(t):
                    if t == 8:
                        n, = struct.unpack("<Q", f.read(8)); f.read(n)
                    elif t == 9:
                        et, = struct.unpack("<I", f.read(4))
                        n, = struct.unpack("<Q", f.read(8))
                        for _ in range(n):
                            skip(et)
                    else:
                        f.read(_SIMPLE[t])

                for _ in range(nkv):
                    rd_str()
                    t, = struct.unpack("<I", f.read(4)); skip(t)
                for _ in range(nt):
                    nm = rd_str().decode("utf-8", "replace")
                    nd, = struct.unpack("<I", f.read(4))
                    f.read(8 * nd + 4 + 8)                   # dims + dtype + offset
                    names.append(nm)
        except Exception:
            continue
    return names


def gguf_to_config(meta, path):
    """Map GGUF metadata keys onto the HF-config vocabulary analyse() speaks."""
    arch = meta.get("general.architecture", "unknown")

    def g(suffix, default=None):
        return meta.get(f"{arch}.{suffix}", default)

    cfg = {
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": g("block_count"),
        "intermediate_size": g("feed_forward_length"),
        "num_attention_heads": g("attention.head_count"),
        "num_key_value_heads": g("attention.head_count_kv"),
        "head_dim": g("attention.key_length"),        # explicit head_dim (often != h/heads)
        "kv_lora_rank": g("attention.kv_lora_rank"),  # MLA (DeepSeek/GLM): compressed KV
        "qk_rope_head_dim": g("rope.dimension_count"),
        "vocab_size": g("vocab_size", meta.get("general.vocab_size", 0)) or 0,
        "num_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "n_shared_experts": g("expert_shared_count", 0),
        "moe_intermediate_size": g("expert_feed_forward_length",
                                   g("feed_forward_length")),
        "_gguf_arch": arch,
        "_gguf_file_bytes": meta.get("_total_file_bytes", os.path.getsize(path)),
        "_tensor_param_sum": meta.get("_tensor_param_sum"),
        "_shard_count": meta.get("_shard_count", 1),
    }
    if isinstance(cfg["num_attention_heads"], list):  # per-layer lists
        cfg["num_attention_heads"] = max(cfg["num_attention_heads"])
    if isinstance(cfg["num_key_value_heads"], list):
        cfg["num_key_value_heads"] = max(cfg["num_key_value_heads"])
    return cfg


def fetch_config(model_id):
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def first(cfg, *keys, default=None):
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def analyse(cfg):
    """Return a dict of architecture facts from a HF config."""
    # multimodal wrappers nest the LM under a sub-config
    multimodal = None
    for mm in ("vision_config", "video_config", "audio_config"):
        if isinstance(cfg.get(mm), dict):
            multimodal = "vision+video" if cfg.get("video_token_id") is not None else "vision"
            break
    for sub in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(sub), dict) and cfg[sub].get("hidden_size"):
            cfg = {**cfg[sub], "vocab_size": first(cfg[sub], "vocab_size",
                                                   default=cfg.get("vocab_size", 0))}
            break
    h = first(cfg, "hidden_size", "n_embd", "d_model")
    layers = first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    if (h is None or layers is None) and cfg.get("_tensor_param_sum"):
        # non-LLM GGUF (DiT / VAE / etc.): no standard keys, but the tensor
        # table gave us the exact param count. Treat as opaque dense.
        total = cfg["_tensor_param_sum"]
        return {"kind": "dense (opaque tensor-sum)", "hidden": 0, "layers": 0,
                "total": total, "active": total, "n_experts": 0, "top_k": 0,
                "shared": 0, "expert_params": 0, "dense_layers": 0}
    vocab = first(cfg, "vocab_size", default=0)
    heads = first(cfg, "num_attention_heads", "n_head", default=0)
    kv_heads = first(cfg, "num_key_value_heads", default=heads) or heads
    head_dim = first(cfg, "head_dim", default=(h // heads if heads else 0))
    ffn = first(cfg, "intermediate_size", "n_inner", "ffn_dim", default=4 * h)
    # Multi-head Latent Attention (DeepSeek / GLM glm-dsa): KV is a COMPRESSED latent
    # (kv_lora_rank + rope part) per token per layer, ~50x smaller than GQA's
    # kv_heads*head_dim — the difference between "needs 256 GB" and "fits a Spark".
    kv_lora_rank = first(cfg, "kv_lora_rank", default=0) or 0
    qk_rope_head_dim = first(cfg, "qk_rope_head_dim", default=0) or 0

    n_experts = first(cfg, "num_experts", "n_routed_experts",
                      "num_local_experts", "moe_num_experts")
    # NOTE: bare "top_k" is deliberately excluded — it's a SAMPLING parameter
    # in many configs (top_k=50) and misreads as expert top-k.
    top_k = first(cfg, "num_experts_per_tok", "num_experts_per_token",
                  "experts_per_token", "moe_top_k", "moe_k",
                  "num_selected_experts", default=0)
    if n_experts and not top_k:
        sys.exit("ERROR: MoE config but no experts-per-token field found — "
                 "add the key to pollard_calc or pass a patched config.")
    shared = first(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0
    moe_ffn = first(cfg, "moe_intermediate_size", default=ffn)
    # some MoEs run experts in a reduced latent space (e.g. Kimi-K3:
    # routed_expert_hidden_size 3584 vs hidden 7168) — ignoring this key
    # doubles every expert-derived number. Found by community review.
    expert_h = first(cfg, "routed_expert_hidden_size", "expert_hidden_size",
                     "moe_hidden_size", default=h)
    dense_layers = first(cfg, "first_k_dense_replace", default=0) or 0

    # per-layer attention params (q,k,v,o)
    attn = h * (heads * head_dim) + 2 * h * (kv_heads * head_dim) + (heads * head_dim) * h
    gated = 3  # SwiGLU-family gate/up/down
    dense_ffn_params = gated * h * ffn

    if n_experts:
        expert_params = gated * expert_h * moe_ffn
        moe_layers = layers - dense_layers
        router = h * n_experts
        total = (layers * attn + dense_layers * dense_ffn_params
                 + moe_layers * (n_experts + shared) * expert_params
                 + moe_layers * router + 2 * vocab * h)
        active = (layers * attn + dense_layers * dense_ffn_params
                  + moe_layers * (top_k + shared) * expert_params
                  + moe_layers * router + 2 * vocab * h)
        kind = "moe"
    else:
        expert_params = 0
        total = layers * (attn + dense_ffn_params) + 2 * vocab * h
        active = total
        kind = "dense"

    # A GGUF source carries the EXACT param count in its tensor table — trust it
    # over the dim estimate, which misses tied/extra embeddings when vocab_size
    # isn't in the metadata (undersized total -> builds that bust the RAM budget).
    if cfg.get("_tensor_param_sum"):
        total = cfg["_tensor_param_sum"]
        if kind == "dense":
            active = total

    # Hybrid linear-attention / SSM layers (Qwen3.5 / Mamba / DeltaNet-style): not
    # every layer is standard attention, so the attn params above are approximate,
    # AND per-token behaviour differs — linear layers barely grow the KV cache.
    lt = cfg.get("layer_types")
    n_linear = n_full = 0
    if isinstance(lt, list):
        n_linear = sum(1 for t in lt if any(s in str(t) for s in ("linear", "mamba", "ssm")))
        n_full = len(lt) - n_linear
    hybrid = bool(n_linear) or any(k in cfg for k in
                  ("mamba_ssm_dtype", "linear_conv_kernel_dim", "linear_num_key_heads"))
    # Multi-token prediction: emits >1 token per weight-read pass -> the RAM-bandwidth
    # "ceiling" is really a floor for these models.
    mtp = first(cfg, "mtp_num_hidden_layers", "num_nextn_predict_layers",
                "num_mtp_layers", "nextn_predict_layers", default=0) or 0

    return {
        "kind": kind, "hidden": h, "layers": layers, "total": total,
        "active": active, "n_experts": n_experts or 0, "top_k": top_k,
        "shared": shared, "expert_params": expert_params,
        "dense_ffn_params": dense_ffn_params,     # per-layer FFN bulk (for dense allocation)
        "attn_params": attn,                      # per-layer attention (q,k,v,o) — its own group
        "dense_layers": dense_layers,
        "multimodal": multimodal, "hybrid": hybrid,
        "n_linear": n_linear, "n_full": n_full, "mtp": mtp,
        "kv_heads": kv_heads, "head_dim": head_dim,
        "mla": kv_lora_rank > 0, "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
    }


def kv_cache_bytes(a, ctx, kv_bytes=2.0):
    """KV-cache bytes at `ctx` tokens. Arch-aware: MLA (DeepSeek/GLM) stores one
    compressed latent per token/layer (tiny); hybrid models only grow KV on their
    full-attention layers. kv_bytes: 2 = f16 cache (default), 1 = q8_0, ~0.56 = q4."""
    n_attn = a.get("n_full") or a["layers"]                 # linear layers barely grow KV
    if a.get("mla") and a.get("kv_lora_rank"):
        per_tok_layer = a["kv_lora_rank"] + (a.get("qk_rope_head_dim") or 0)
        return n_attn * ctx * per_tok_layer * kv_bytes      # MLA: single latent, no K/V split
    kvh, hd = a.get("kv_heads") or 0, a.get("head_dim") or 0
    return 2 * n_attn * ctx * kvh * hd * kv_bytes           # GQA/MHA: K and V


# per-card VRAM (GB) for the --gpu convenience; anything not listed, pass GB directly
_GPU_VRAM = {"3050": 8, "3060": 12, "3060ti": 8, "3070": 8, "3080": 10, "3090": 24,
             "4050": 6, "4060": 8, "4060ti": 16, "4070": 12, "4080": 16, "4090": 24,
             "5060": 8, "5070": 12, "5080": 16, "5090": 32, "a4000": 16, "a5000": 24,
             "a6000": 48, "rtx6000": 48, "rtx6000pro": 96,
             "6000pro": 96, "a40": 48, "l40": 48, "l40s": 48, "v100": 32, "a100": 80,
             "h100": 80, "h200": 141, "b100": 192, "b200": 192, "mi300x": 192,
             "spark": 128, "gb10": 128, "m4max": 128, "m3ultra": 512}


def parse_gpu(spec):
    """'5090x4' / '24x8' / 'rtx6000pro x2' / '96' -> total VRAM GB (left = per-card
    name or GB, right = count). Plain number or a bare card name = that much. None if
    unparseable — so any stack of any card works, not a fixed menu."""
    s = spec.lower().replace(" ", "")
    if s in _GPU_VRAM:                          # bare card name (may contain 'x': rtx…)
        return _GPU_VRAM[s]
    try:
        return float(s)                         # plain GB total
    except ValueError:
        pass
    if "x" in s:                                # CARDxCOUNT / GBxCOUNT — split on LAST x
        card, _, cnt = s.rpartition("x")
        per = _GPU_VRAM.get(card)
        if per is None:
            try:
                per = float(card)
            except ValueError:
                return None
        try:
            return per * int(cnt)
        except ValueError:
            return None
    return None


# fraction of a device's RAM actually usable by the model. A dedicated GPU gives
# almost all its VRAM; a phone hands an app only ~half (iOS/Android reserve the rest
# and OOM-kill past it — an 8 GB phone ≈ 4-5 GB usable, ~3B practical cap); Apple/APU
# unified memory wires ~75%. Sources: iOS jetsam limits, community mobile-LLM guides.
_DEVICE_USABLE = {"gpu": 0.94, "unified": 0.75, "mac": 0.75, "apu": 0.75,
                  "phone": 0.55, "mobile": 0.55}


def fit_report(a, weights_gb, ctx, kv_bytes, kv_label, rig_gb=None, device="gpu"):
    """The pre-flight: weights + KV cache at a chosen context + overhead = what it
    takes to RUN, and which devices that fits — so you decide BEFORE the hours-long
    build/download whether it's worth it (requested by a user running 300B MoEs)."""
    kv_gb = kv_cache_bytes(a, ctx, kv_bytes) / 1e9
    overhead_gb = 1.0 + ctx / 262144 * 2.0                  # compute/activation buffers (approx)
    total_gb = weights_gb + kv_gb + overhead_gb
    tag = ("  (MLA — compressed latent, not GQA)" if a.get("mla")
           else f"  (hybrid: {a['n_full']}/{a['layers']} layers grow KV)"
           if a.get("hybrid") and a.get("n_full") else "")
    print(f"\n== will it fit? @ {ctx:,} tokens of context ==")
    print(f"{'weights':<20}: {weights_gb:7.1f} GB")
    print(f"{'KV cache (' + kv_label + ')':<20}: {kv_gb:7.1f} GB{tag}")
    print(f"{'compute overhead':<20}: {overhead_gb:7.1f} GB  (approx)")
    print(f"{'TOTAL to run':<20}: {total_gb:7.1f} GB")
    if rig_gb:                                              # verdict for THEIR actual rig
        frac = _DEVICE_USABLE.get(device, 0.94)
        usable = rig_gb * frac
        spare = usable - total_gb
        verdict = (f"FITS, {spare:.0f} GB to spare" if spare >= usable * 0.10
                   else f"TIGHT, {spare:.0f} GB headroom" if spare >= 0
                   else f"SHORT by {-spare:.0f} GB — smaller quant/model, more cards, or --rpc")
        extra = (f"  [~{frac:.0%} of {rig_gb:.0f} GB usable = {usable:.0f} GB]"
                 if frac < 0.9 else "")
        print(f"{'>> YOUR RIG (' + f'{rig_gb:.0f} GB {device})':<20}: {verdict}{extra}")
        if device in ("phone", "mobile"):
            print("   note: phones OOM-kill past ~half their RAM; iOS is practical only to "
                  "~3B, flagship Android to ~7B. Target a 1-3B at q4 for a smooth phone run.")
    print("reference tiers — dedicated VRAM, ~6% for the OS (phones give an app ~half):")
    for name, cap in [("8 GB   (4060 / 8GB card)", 8), ("12 GB  (3060 / 4070)", 12),
                      ("16 GB  (4080 / 5080)", 16), ("24 GB  (4090 / 3090)", 24),
                      ("32 GB  (5090)", 32), ("48 GB  (2x24 / A6000)", 48),
                      ("96 GB  (RTX 6000 Pro / 4x24)", 96), ("128 GB (DGX Spark)", 128),
                      ("192 GB (B200 / 6x32)", 192), ("256 GB (2x Spark)", 256)]:
        print(f"  [{'YES' if total_gb <= cap * 0.94 else 'no ':<3}] {name}")
    if total_gb > 256 * 0.94:
        print("  -> over 256 GB: stack more cards, pool nodes (--rpc), or a smaller quant")
    if kv_bytes >= 2:
        kv_q8 = kv_cache_bytes(a, ctx, 1.0) / 1e9
        if kv_gb - kv_q8 > 0.5:
            print(f"  tip: --kv-quant q8 halves KV to {kv_q8:.1f} GB "
                  f"(total {weights_gb + kv_q8 + overhead_gb:.1f} GB)")


def gb(nbytes):
    return nbytes / 1e9


def report(a, ram_gb, flash_gbps, rambw_gbps, qbits, cache_gb):
    wb = qbits / 8.0
    total_b, active_b = a["total"] * wb, a["active"] * wb
    print(f"architecture        : {a['kind'].upper()}"
          + (f"  ({a['n_experts']} experts/layer, top-{a['top_k']}"
             + (f" + {a['shared']} shared" if a['shared'] else "") + ")"
             if a["kind"] == "moe" else ""))
    print(f"total params        : {a['total']/1e9:,.1f}B  -> {gb(total_b):,.1f} GB @ {qbits:.2f}bpw")
    print(f"active per token    : {a['active']/1e9:,.1f}B  -> {gb(active_b):,.1f} GB reads/cold-token")
    if a["kind"] == "moe":
        eb = a["expert_params"] * wb
        pool = a["n_experts"] * (a["layers"] - a["dense_layers"])
        print(f"expert size         : {a['expert_params']/1e6:.1f}M params ({eb/1e6:.0f} MB)")
        print(f"expert pool         : {pool:,} instances")
        if eb > 0:
            hold = int(cache_gb * 1e9 / eb)
            print(f"cache {cache_gb:.0f} GB holds   : {hold:,} experts "
                  f"({min(100.0, 100.0*hold/max(pool,1)):.1f}% of pool, uniform-routing worst case)")
    print()
    resident = total_b <= ram_gb * 1e9 * 0.85
    t_flash = flash_gbps * 1e9 / active_b if active_b else 0
    t_ram = rambw_gbps * 1e9 / active_b if active_b else 0
    print(f"flash-stream floor  : {t_flash:6.2f} tok/s  (@ {flash_gbps} GB/s sequential)")
    print(f"RAM-bandwidth ceil  : {t_ram:6.2f} tok/s  (@ {rambw_gbps} GB/s)")
    print()
    if resident:
        print(f"VERDICT: FITS RESIDENT in {ram_gb} GB — compute-bound, "
              f"expect up to ~{t_ram:.1f} tok/s ceiling.")
    elif a["kind"] == "moe" and active_b <= ram_gb * 1e9 * 0.85:
        cov = cache_gb * 1e9 / total_b * 100
        print(f"VERDICT: STREAMING-VIABLE — active set fits RAM; throughput is "
              f"governed by routing reuse (cache covers {cov:.1f}% of weights "
              f"uniform-case). CAVEAT: reuse is UNPROVEN in general — our one "
              f"measured testbed showed near-uniform routing (notes/e5), which "
              f"makes caching ineffective there. Measure your workload with "
              f"experiments/e2 before betting on this verdict.")
        need = gb(total_b) / 0.85
        print(f"         full residency would need the ~{need:,.0f} GB tier.")
    else:
        need = gb(total_b) / 0.85
        print(f"VERDICT: NEEDS BIGGER TIER — ~{need:,.0f} GB RAM for residency; "
              f"streaming floor here is {t_flash:.2f} tok/s.")

    # Architecture notes — where a new arch makes the numbers above approximate or
    # conservative. Better to say so than to emit a dense number and stay silent.
    notes = []
    if a.get("multimodal"):
        notes.append(f"multimodal ({a['multimodal']}): the sizes above are the TEXT "
                     "model — the vision projector (mmproj) ships separately (~1 GB in GGUF).")
    if a.get("hybrid"):
        mix = (f"{a['n_linear']}/{a['n_linear'] + a['n_full']} layers are linear-attention"
               if a.get("n_linear") else "linear-attention / SSM layers present")
        notes.append(f"hybrid attention ({mix}): the attention params are APPROXIMATE "
                     "(standard-attention formula applied to non-standard layers), and the "
                     "linear layers barely grow the KV cache — throughput holds at long "
                     "context in a way this single-token ceiling doesn't capture.")
    if a.get("mtp"):
        notes.append(f"MTP present ({a['mtp']} predictor layer(s)): decode emits >1 token "
                     "per weight-read pass, so measured tok/s can EXCEED the RAM-bandwidth "
                     "figure above — treat it as a FLOOR for this model, not a ceiling.")
    if notes:
        print("\narch notes:")
        for n in notes:
            print("  - " + n)


# GGUF general.file_type (LLAMA_FTYPE) -> friendly quant name. Best-effort: the
# enum has drifted across llama.cpp versions, so it's sanity-checked against the
# measured bpw (the bpw is the ground truth; the name is the convenience).
_FTYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
          9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
          14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
          19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
          24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
          29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}


# GGML per-tensor type enum -> name. These are the model's ACTUAL on-disk tensor
# types (ground truth), used when general.file_type is a value we don't recognize —
# so a brand-new or fork quant still gets named from what's really in the file.
_GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
              14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0"}
_FULL_PRECISION = {"F32", "F16", "BF16"}


def describe_source(meta, bpw):
    """What quant is this GGUF, and is it a good pollard-fit source? Named from the
    model's declared file_type (friendly preset name) OR — if that enum value is
    unrecognized — the ACTUAL dominant tensor type read from the file. Only if BOTH
    are unknown do we say so honestly (with the raw ids), never a fake bpw label.
    Returns (label, is_full_precision, advice)."""
    ft = meta.get("general.file_type")
    dom = meta.get("_dominant_ggml_type")
    name = (_FTYPE.get(ft) if isinstance(ft, int) else None) \
        or (_GGML_TYPE.get(dom) if isinstance(dom, int) else None)
    if name is None:                                    # truly unrecognized — be honest
        ids = [s for s in (f"file_type {ft}" if isinstance(ft, int) else None,
                           f"ggml type {dom}" if isinstance(dom, int) else None) if s]
        name = "unrecognized quant" + (f" ({', '.join(ids)})" if ids else "")
    full = name in _FULL_PRECISION or (name.startswith("unrecognized") and bpw >= 15.0)
    if full:
        advice = "ideal source — pollard-fit builds straight from this"
    else:
        advice = ("already quantized — pollard-fit CAN requantize it, but for best "
                  "quality grab the f16/bf16 source (usually the base repo, not a -GGUF one)")
    return name, full, advice


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to a local config.json")
    src.add_argument("--model", help="Hugging Face model id (fetches config.json)")
    src.add_argument("--gguf", help="path to a local .gguf file (reads its header; "
                                    "uses REAL on-disk bytes, no bpw estimate)")
    p.add_argument("--ram", default="16",
                   help="system RAM GB, or 'auto' to measure what is actually "
                        "available right now (default 16)")
    p.add_argument("--flash", type=float, default=3.5,
                   help="sequential flash read GB/s (default 3.5, measured M4 4MB-block)")
    p.add_argument("--rambw", type=float, default=120,
                   help="memory bandwidth GB/s (default 120, M4)")
    p.add_argument("--quant", default="q4", choices=sorted(QUANTS),
                   help="quantization (default q4)")
    p.add_argument("--cache", type=float, default=None,
                   help="hot-cache budget GB (default: RAM minus 4)")
    p.add_argument("--ctx", type=int, default=0,
                   help="context length for a 'will it fit?' pre-flight: adds KV-cache "
                        "size + total RAM-to-run + device fit (e.g. --ctx 262144 for 256k)")
    p.add_argument("--kv-quant", default="f16", choices=["f16", "q8", "q4"],
                   help="KV cache precision for the --ctx estimate (default f16)")
    p.add_argument("--gpu", help="your rig for the fit verdict: total VRAM GB, a card "
                                 "name, or CARDxCOUNT — e.g. '96', '5090x4', '3090x8', "
                                 "'rtx6000prox2'. Any stack of any card.")
    p.add_argument("--device", default="gpu", choices=["gpu", "unified", "mac", "phone"],
                   help="what --gpu's number is: dedicated 'gpu' VRAM (~94%% usable, "
                        "default), 'unified'/'mac' RAM (~75%%), or 'phone' (~55%% — the OS "
                        "OOM-kills past ~half)")
    a = p.parse_args()

    meta = None
    if a.gguf:
        meta = read_gguf_meta(a.gguf)
        cfg = gguf_to_config(meta, a.gguf)
        name = f"{a.gguf} [{cfg['_gguf_arch']}]"
    elif a.config:
        cfg, name = json.load(open(a.config)), a.config
    else:
        cfg, name = fetch_config(a.model), a.model
    arch = analyse(cfg)
    if str(a.ram).lower() == "auto":
        avail = detect_available_ram_gb()
        if avail is None:
            sys.exit("ERROR: could not measure available RAM on this platform — "
                     "pass --ram <GB> explicitly.")
        a.ram = avail
        print(f"[--ram auto] measured available memory: {avail:.1f} GB "
              f"(nameplate lies; this is what you can actually use right now)")
    else:
        a.ram = float(a.ram)
    cache = a.cache if a.cache is not None else max(a.ram - 4, 1)
    qbits = QUANTS[a.quant]
    if cfg.get("_gguf_file_bytes"):
        # ground truth beats estimate: derive effective bpw from the file itself
        qbits = cfg["_gguf_file_bytes"] * 8.0 / arch["total"]
    print(f"== pollard-calc :: {name} ==")
    print(f"hardware            : {a.ram:.0f} GB RAM, {a.flash} GB/s flash, "
          f"{a.rambw} GB/s membw, "
          + (f"measured {qbits:.2f} bpw from file" if a.gguf else f"quant {a.quant}")
          + "\n")
    if meta is not None:                                # a "what have I got" model check
        label, full, advice = describe_source(meta, qbits)
        shards = cfg.get("_shard_count", 1)
        shard_note = f", {shards} shards" if shards > 1 else ""
        print(f"source quant        : {label} (~{qbits:.2f} bpw{shard_note})  "
              f"{'✅' if full else '⚠️'} {advice}\n")
    report(arch, a.ram, a.flash, a.rambw, qbits, cache)
    if a.ctx:
        kv_bytes = {"f16": 2.0, "q8": 1.0, "q4": 0.5625}[a.kv_quant]
        rig_gb = None
        if a.gpu:
            rig_gb = parse_gpu(a.gpu)
            if rig_gb is None:
                print(f"(could not parse --gpu '{a.gpu}'; skipping the rig verdict)")
        fit_report(arch, arch["total"] * qbits / 8 / 1e9, a.ctx, kv_bytes, a.kv_quant,
                   rig_gb, a.device)


if __name__ == "__main__":
    main()
