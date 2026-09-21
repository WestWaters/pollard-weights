#!/usr/bin/env python3
"""Read the non-GGUF lanes: GPTQ, MLX, EXL3, MX/NVFP4 — anything stored as safetensors.

Studio could measure a GGUF and nothing else, which left every other lane invisible. This closes
that: same contract as ggufread.py, so a build is a build regardless of which emitter made it.

Header only. A safetensors file starts with an 8-byte little-endian length, then a JSON header
giving every tensor's dtype, shape and byte range. Parsing it is fast on a 300 GB shard set and
safe to call while something else is mid-build.

Bits-per-weight is computed from REAL file bytes over REAL logical parameters, not from the
format's own arithmetic. Every lane derives bpw differently -- NVFP4 is 4 + 8/16 + 32/N, MXFP4 is
4 + 8/32, GPTQ is bits + (scale+zero)/group, MLX is bits + 32/group -- and each of those is a
claim about the format rather than a measurement of the file. Measuring sidesteps all of it and
stays correct when a lane changes its packing.

    python -m pollard_studio.saferead /path/to/model-dir
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

# bytes per element for the dtypes safetensors declares
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "U64": 8, "U32": 4, "U16": 2, "F4": 1, "U4": 1, "I4": 1}

# a packed tensor stores several logical weights per stored element
PACKED = re.compile(r"\.(qweight|weight_packed|qzeros|packed_weight)$")
# the companions a quantized linear carries alongside its packed weights
COMPANION = re.compile(r"\.(scales?|qzeros|zeros?|biases|g_idx|weight_scale|"
                       r"weight_zero_point|weight_shape|weight_g_idx|input_scale)$")


def read_header(path: Path) -> dict:
    """The JSON header of one safetensors file."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n <= 0 or n > 200_000_000:
            raise ValueError(f"{path.name}: implausible safetensors header length {n}")
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    return head


def _config(d: Path) -> dict:
    """Whatever the lane recorded about how it quantized."""
    out = {}
    for name in ("config.json", "quantize_config.json", "quantization_config.json",
                 "quant_config.json"):
        p = d / name
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def detect_lane(d: Path, cfg: dict) -> tuple[str, dict]:
    """Which emitter made this, and with what settings.

    Detection is by what is IN the files, not by directory name -- a name is a label someone
    typed, and it is wrong exactly when it matters.
    """
    blob = json.dumps(cfg).lower()
    main = cfg.get("config.json", {})
    qc = (main.get("quantization_config") or main.get("quantization") or {})
    qc = qc if isinstance(qc, dict) else {}
    merged = {**cfg.get("quantize_config.json", {}), **cfg.get("quant_config.json", {}), **qc}

    bits = merged.get("bits") or merged.get("weight_bits") or merged.get("num_bits")
    group = merged.get("group_size") or merged.get("group") or merged.get("block_size")
    method = str(merged.get("quant_method") or merged.get("method") or "").lower()

    if list(d.glob("*.exl3")) or "exl3" in blob or "exllama" in method:
        lane = "EXL3"
    elif "nvfp4" in blob or "mxfp4" in blob or "fp4" in blob:
        lane = "MX"
    elif "compressed-tensors" in blob or "compressed_tensors" in method:
        lane = "MX" if "fp4" in blob or "fp8" in blob else "GPTQ"
    elif "gptq" in blob or "awq" in blob or (d / "quantize_config.json").exists():
        lane = "GPTQ"
    elif "mlx" in blob or (main.get("quantization") and not qc.get("quant_method")):
        lane = "MLX"
    elif bits:
        lane = "GPTQ"
    else:
        lane = "SAFETENSORS"
    return lane, {"bits": bits, "group_size": group, "method": method or None}


def read(model_dir: str | Path) -> dict:
    """Every safetensors shard in a directory, rolled up."""
    d = Path(model_dir)
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors in {d}")

    cfg = _config(d)
    lane, q = detect_lane(d, cfg)
    bits = q.get("bits")

    tensors, file_bytes, missing = [], 0, []
    for shard in shards:
        file_bytes += shard.stat().st_size
        try:
            head = read_header(shard)
        except Exception as e:
            missing.append(f"{shard.name}: {e}")
            continue
        for name, spec in head.items():
            shape = spec.get("shape") or []
            dtype = spec.get("dtype", "?")
            lo, hi = (spec.get("data_offsets") or [0, 0])[:2]
            stored = 1
            for s in shape:
                stored *= s
            tensors.append({
                "name": name, "shape": shape, "dtype": dtype, "shard": shard.name,
                "stored_elements": stored, "bytes": max(0, hi - lo),
                "logical": _logical(name, shape, dtype, stored, bits),
                "companion": bool(COMPANION.search(name)),
            })

    params = sum(t["logical"] for t in tensors)
    main = cfg.get("config.json", {})
    return {
        "path": str(d), "name": d.name, "lane": lane, "shards": len(shards),
        "file_bytes": file_bytes, "tensors": tensors, "params": params,
        "architecture": (main.get("model_type")
                         or (main.get("architectures") or [None])[0] or "unknown"),
        "block_count": main.get("num_hidden_layers") or main.get("n_layer") or 0,
        "quant": q, "config_files": sorted(cfg), "warnings": missing,
    }


def _logical(name: str, shape: list, dtype: str, stored: int, bits) -> int:
    """How many real weights a stored tensor represents.

    A GPTQ qweight is int32 holding 32/bits weights per element; counting stored elements would
    undercount params by 8x at 4-bit and report a bits-per-weight that looks far too good.
    Companions -- scales, zeros, g_idx -- are overhead, not parameters, and count as zero.
    """
    if COMPANION.search(name):
        return 0
    if PACKED.search(name) and bits:
        try:
            per = max(1, int(DTYPE_BYTES.get(dtype, 4) * 8 // int(bits)))
        except (TypeError, ValueError):
            per = 1
        return stored * per
    return stored


def summarise(info: dict) -> dict:
    """The same shape ggufread.summarise returns, so the UI does not care which lane it is."""
    groups: dict[str, dict] = {}
    for t in info["tensors"]:
        g = _group(t["name"])
        e = groups.setdefault(g, {"params": 0, "bytes": 0, "tensors": 0, "types": {}})
        e["params"] += t["logical"]
        e["bytes"] += t["bytes"]
        e["tensors"] += 1
        e["types"][t["dtype"]] = e["types"].get(t["dtype"], 0) + 1
    for g in groups.values():
        g["bpw"] = g["bytes"] * 8 / g["params"] if g["params"] else 0.0

    types: dict[str, int] = {}
    for t in info["tensors"]:
        types[t["dtype"]] = types.get(t["dtype"], 0) + 1

    params = info["params"]
    return {
        "name": info["name"], "path": info["path"], "lane": info["lane"],
        "architecture": info["architecture"], "block_count": info["block_count"],
        "params": params, "file_bytes": info["file_bytes"],
        "weight_bytes": sum(t["bytes"] for t in info["tensors"]),
        "bpw": info["file_bytes"] * 8 / params if params else 0.0,
        "groups": groups, "types": types, "quant": info["quant"],
        "shards": info["shards"], "mtp_block": None,
        "inexact": len(info["warnings"]),
    }


# Vision and audio towers live INSIDE the checkpoint on every lane except GGUF, where the
# projector ships as a separate mmproj file. Matching them first matters: `visual.blocks.0.attn`
# contains "attn", so a language-first pattern order silently files two thirds of a VL model's
# vision tower under the language mixing path -- and then the allocator crushes the model's eyes
# at the body atom. The vision encoder and the projector are the parts that carry modality
# alignment, and they are the ones that should be held high.
_VISION = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder|image_encoder"
                     r"|patch_embed|pixel)", re.I)
_PROJ = re.compile(r"(multi_modal_projector|mm_projector|mm_proj|\bmerger\b|modality_project"
                   r"|resampler|perceiver|connector)", re.I)
_AUDIO = re.compile(r"(^|\.)(audio_tower|audio_encoder|audio_model|whisper|speech_encoder)", re.I)


def _group(name: str) -> str:
    if _PROJ.search(name):
        return "projector"
    if _VISION.search(name):
        return "vision"
    if _AUDIO.search(name):
        return "audio"
    if re.search(r"embed|wte|lm_head|output\.weight", name):
        return "embeddings"
    if re.search(r"attn|attention|self_attn|\.ssm_", name):
        return "mixing"
    if re.search(r"mlp|ffn|feed_forward|experts?", name):
        return "ffn"
    return "other"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    print(json.dumps(summarise(read(sys.argv[1])), indent=2)[:3000])


if __name__ == "__main__":
    main()
