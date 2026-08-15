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


def read_gguf_meta(path):
    """Minimal GGUF v2/v3 header reader — metadata KV only, stdlib only."""
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
        try:
            for _ in range(n_tensors):
                rd_str()  # tensor name
                nd, = struct.unpack("<I", f.read(4))
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                f.read(4 + 8)  # dtype + offset
                n = 1
                for d in dims:
                    n *= d
                total_params += n
            meta["_tensor_param_sum"] = total_params
        except Exception:
            pass  # malformed tail: fall back to key-based analysis only
    return meta


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
        "vocab_size": g("vocab_size", meta.get("general.vocab_size", 0)) or 0,
        "num_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "n_shared_experts": g("expert_shared_count", 0),
        "moe_intermediate_size": g("expert_feed_forward_length",
                                   g("feed_forward_length")),
        "_gguf_arch": arch,
        "_gguf_file_bytes": os.path.getsize(path),
        "_tensor_param_sum": meta.get("_tensor_param_sum"),
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
        "dense_layers": dense_layers,
        "multimodal": multimodal, "hybrid": hybrid,
        "n_linear": n_linear, "n_full": n_full, "mtp": mtp,
    }


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
    a = p.parse_args()

    if a.gguf:
        cfg = gguf_to_config(read_gguf_meta(a.gguf), a.gguf)
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
    report(arch, a.ram, a.flash, a.rambw, qbits, cache)


if __name__ == "__main__":
    main()
