#!/usr/bin/env python3
"""Read a GGUF header and report what is ACTUALLY in the file.

The point of this module is that Studio should never quote a number it guessed. Parameter counts,
byte sizes and bits-per-weight all come from the tensor table of the real file, so the projection
panel can be checked against a build instead of being an educated estimate that drifts.

Header only -- the tensor data is never read, so this is fast on a 50 GB file and safe to call
while something else is mid-build.

    python ggufread.py model.gguf
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

MAGIC = b"GGUF"

# GGUF metadata value types
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)
_FIXED = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
          U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
          U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8)}

# ggml type -> (elements per block, bytes per block). Anything absent is handled by falling back
# to the gap between tensor offsets, which is how ik_llama's trellis types stay readable here
# without this file having to track ik_llama's type numbering.
GGML: dict[int, tuple[str, int, int]] = {
    0:  ("F32", 1, 4),        1:  ("F16", 1, 2),        2:  ("Q4_0", 32, 18),
    3:  ("Q4_1", 32, 20),     6:  ("Q5_0", 32, 22),     7:  ("Q5_1", 32, 24),
    8:  ("Q8_0", 32, 34),     9:  ("Q8_1", 32, 36),     10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),   12: ("Q4_K", 256, 144),   13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),   15: ("Q8_K", 256, 292),   16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),  18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),   21: ("IQ3_S", 256, 110),  22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),         25: ("I16", 1, 2),
    26: ("I32", 1, 4),        27: ("I64", 1, 8),        28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),   30: ("BF16", 1, 2),
}


class _R:
    """Sequential reader over the header bytes."""

    def __init__(self, fh):
        self.fh = fh

    def take(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def num(self, fmt: str, n: int):
        return struct.unpack(fmt, self.take(n))[0]

    def string(self) -> str:
        return self.take(self.num("<Q", 8)).decode("utf-8", "replace")

    def value(self, t: int):
        if t in _FIXED:
            fmt, n = _FIXED[t]
            return self.num(fmt, n)
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.num("<I", 4)
            n = self.num("<Q", 8)
            if et == STRING:
                return [self.string() for _ in range(n)]
            if et in _FIXED:
                fmt, sz = _FIXED[et]
                raw = self.take(sz * n)
                return list(struct.unpack("<" + fmt[1] * n, raw))
            raise ValueError(f"unsupported array element type {et}")
        raise ValueError(f"unsupported metadata type {t}")


def read(path: str | Path) -> dict:
    """Parse the header. Returns metadata, the tensor table, and per-tensor byte sizes."""
    path = Path(path)
    with open(path, "rb") as fh:
        r = _R(fh)
        if r.take(4) != MAGIC:
            raise ValueError(f"{path.name} is not a GGUF file")
        version = r.num("<I", 4)
        n_tensors = r.num("<Q", 8)
        n_kv = r.num("<Q", 8)

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.num("<I", 4))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.num("<Q", 8) for _ in range(r.num("<I", 4))]
            ttype = r.num("<I", 4)
            offset = r.num("<Q", 8)
            elements = 1
            for d in dims:
                elements *= d
            tensors.append({"name": name, "dims": dims, "type": ttype,
                            "type_name": GGML.get(ttype, (f"TYPE_{ttype}",))[0],
                            "elements": elements, "offset": offset})

        align = int(meta.get("general.alignment", 32))
        data_start = fh.tell()
        if data_start % align:
            data_start += align - (data_start % align)

    total = path.stat().st_size
    _size_tensors(tensors, data_start, total)

    return {"path": str(path), "name": path.name, "version": version,
            "file_bytes": total, "alignment": align, "data_start": data_start,
            "metadata": meta, "tensors": tensors,
            "architecture": meta.get("general.architecture", "unknown"),
            "block_count": _block_count(meta, tensors)}


def _size_tensors(tensors, data_start: int, total: int) -> None:
    """Byte size per tensor: from the ggml type table, else from the gap to the next tensor.

    The gap is what makes unknown types (ik_llama trellis atoms among them) still measurable --
    it includes alignment padding, so it is flagged rather than passed off as exact.
    """
    order = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    for pos, i in enumerate(order):
        t = tensors[i]
        nxt = (tensors[order[pos + 1]]["offset"] if pos + 1 < len(order)
               else total - data_start)
        t["gap_bytes"] = max(0, nxt - t["offset"])
        spec = GGML.get(t["type"])
        if spec and t["elements"] % spec[1] == 0:
            t["bytes"] = t["elements"] // spec[1] * spec[2]
            t["exact"] = True
        else:
            t["bytes"] = t["gap_bytes"]
            t["exact"] = False
        t["bpw"] = (t["bytes"] * 8 / t["elements"]) if t["elements"] else 0.0


def _block_count(meta: dict, tensors: list) -> int:
    for k, v in meta.items():
        if k.endswith(".block_count"):
            return int(v)
    seen = {int(m.group(1)) for t in tensors
            if (m := re.match(r"blk\.(\d+)\.", t["name"]))}
    return max(seen) + 1 if seen else 0


# ── grouping: the four buckets the allocation panel steers ──────────────────────────────────────
# Named the way Pollard names them, so a number on screen can be traced to a tensor in the file.
GROUPS = (
    ("embeddings", (r"^token_embd", r"^output\.weight$", r"^output_norm")),
    ("mixing",     (r"\.attn_", r"\.ssm_")),           # attention + SSM: the sequence-mixing path
    ("ffn",        (r"\.ffn_",)),
    ("other",      (r".",)),                            # norms, biases, anything left
)


def group_of(name: str, mtp_block: int | None) -> str:
    if mtp_block is not None and name.startswith(f"blk.{mtp_block}."):
        return "mtp"
    for label, pats in GROUPS:
        if any(re.search(p, name) for p in pats):
            return label
    return "other"


def summarise(info: dict) -> dict:
    """Roll the tensor table up into the numbers Studio shows."""
    blocks = info["block_count"]
    names = {t["name"] for t in info["tensors"]}
    # a trailing block carrying nextn/MTP tensors is never calibrated and is pinned separately
    mtp = blocks - 1 if any(n.startswith(f"blk.{blocks - 1}.nextn") for n in names) else None

    groups: dict[str, dict] = {}
    for t in info["tensors"]:
        g = groups.setdefault(group_of(t["name"], mtp),
                              {"params": 0, "bytes": 0, "tensors": 0, "types": {}})
        g["params"] += t["elements"]
        g["bytes"] += t["bytes"]
        g["tensors"] += 1
        g["types"][t["type_name"]] = g["types"].get(t["type_name"], 0) + 1

    params = sum(g["params"] for g in groups.values())
    wbytes = sum(g["bytes"] for g in groups.values())
    for g in groups.values():
        g["bpw"] = g["bytes"] * 8 / g["params"] if g["params"] else 0.0

    types: dict[str, int] = {}
    for t in info["tensors"]:
        types[t["type_name"]] = types.get(t["type_name"], 0) + 1

    return {
        "name": info["name"], "path": info["path"],
        "architecture": info["architecture"], "block_count": blocks, "mtp_block": mtp,
        "params": params, "weight_bytes": wbytes, "file_bytes": info["file_bytes"],
        # file bytes, not weight bytes: it is what the disk and the loader actually see
        "bpw": info["file_bytes"] * 8 / params if params else 0.0,
        "groups": groups, "types": types,
        "inexact": sum(1 for t in info["tensors"] if not t["exact"]),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    print(json.dumps(summarise(read(sys.argv[1])), indent=2))


if __name__ == "__main__":
    main()
