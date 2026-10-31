"""Nothing Pollard publishes carries the publisher's local paths (2026-10-31: every PollardWeights GGUF showed
`quantize.imatrix.file = /Users/<name>/pollard/...`). GGUF header rewrite keeps the tensor data byte for byte; the legacy
.imatrix dataset record and JSON sidecars are cut to file names; SafeApi refuses what it cannot clean."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_privacy as PV  # noqa: E402

HOMEISH = "/Users/someone/pollard/models/X/calibration"


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kvs, tensors, align=32):
    """kvs: [(key, 'str'|'arr'|'u32', value)]; tensors: [(name, bytes)] stored as u8 1-D, aligned."""
    kv = b""
    for k, kind, v in kvs:
        if kind == "str":
            kv += _s(k) + struct.pack("<I", 8) + _s(v)
        elif kind == "arr":
            kv += _s(k) + struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_s(x) for x in v)
        else:
            kv += _s(k) + struct.pack("<II", 4, v)
    ti, off, blobs = b"", 0, []
    for name, data in tensors:
        ti += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", len(data)) + struct.pack("<IQ", 24, off)
        pad = (-len(data)) % align
        blobs.append(data + b"\x00" * pad)
        off += len(data) + pad
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + kv + ti
    head += b"\x00" * ((-len(head)) % align)
    open(path, "wb").write(head + b"".join(blobs))


def data_hash(path):
    h = PV._gguf_header(path)
    with open(path, "rb") as f:
        f.seek(h[5])
        return hashlib.sha256(f.read()).hexdigest()


def test_gguf_scrub_rewrites_only_the_strings(tmp_path):
    p = str(tmp_path / "m-IQ3_S.gguf")
    tensors = [("blk.0.w", os.urandom(1000)), ("blk.1.w", os.urandom(77))]
    write_gguf(p, [("general.name", "str", "M"),
                   ("quantize.imatrix.file", "str", f"{HOMEISH}/M.imatrix"),
                   ("quantize.imatrix.dataset", "str", f"{HOMEISH}/release-calib.txt"),
                   ("general.alignment", "u32", 32),
                   ("x.sources", "arr", ["C:\\Users\\someone\\data\\a.txt", "plain"])], tensors)
    before = data_hash(p)
    assert len(PV.gguf_leaks(p)) == 3
    assert PV.scrub_gguf(p)
    assert PV.gguf_leaks(p) == [] and data_hash(p) == before
    h = PV._gguf_header(p)
    vals = {k.decode(): (payload.decode() if kind == "str" else [x.decode() for x in payload] if kind == "arr" else None)
            for k, _, _, _, kind, payload in h[2]}
    assert vals["quantize.imatrix.file"] == "M.imatrix"
    assert vals["quantize.imatrix.dataset"] == "release-calib.txt"
    assert vals["x.sources"] == ["a.txt", "plain"] and vals["general.name"] == "M"
    assert not PV.scrub_gguf(p)                                  # idempotent: nothing left to change


def test_legacy_imatrix_dataset_record(tmp_path):
    p = str(tmp_path / "M.imatrix")
    body = struct.pack("<i", 1) + struct.pack("<i", 7) + b"blk.0.w" + struct.pack("<ii", 10, 2) + struct.pack("<2f", 1, 2)
    ds = f"{HOMEISH}/release-calib.txt".encode()
    open(p, "wb").write(body + struct.pack("<i", 1000) + struct.pack("<i", len(ds)) + ds)
    assert PV.imatrix_leaks(p)
    assert PV.scrub_imatrix(p)
    raw = open(p, "rb").read()
    assert raw.startswith(body + struct.pack("<i", 1000)) and raw.endswith(b"release-calib.txt")
    assert PV.imatrix_leaks(p) == []


def test_text_sidecars_and_home_folder(tmp_path):
    j = tmp_path / "release-calib.txt.sources.json"
    j.write_text(json.dumps({"rendering": "/home/someone/pollard/downloads/Org__M's own chat template",
                             "out": "/private/var/folders/x/T/calib.txt", "src": "HuggingFaceH4/ultrachat_200k"}))
    assert PV.check([str(tmp_path)])
    PV.scrub([str(tmp_path)], log=lambda *a: None)
    d = json.loads(j.read_text())
    assert d["rendering"] == "Org__M's own chat template" and d["out"] == "calib.txt"
    assert d["src"] == "HuggingFaceH4/ultrachat_200k"                  # repo ids are not paths
    assert PV.check([str(tmp_path)]) == {}


def test_safe_api_scrubs_before_it_sends(tmp_path, monkeypatch):
    p = str(tmp_path / "m.gguf")
    write_gguf(p, [("quantize.imatrix.file", "str", f"{HOMEISH}/M.imatrix")], [("t", b"\x01" * 64)])
    import huggingface_hub
    sent = []

    class Stub:
        def upload_file(self, *a, **k):
            sent.append(k["path_or_fileobj"])
    monkeypatch.setattr(huggingface_hub, "HfApi", Stub)
    PV.SafeApi().upload_file(path_or_fileobj=p, path_in_repo="m.gguf", repo_id="o/r")
    assert sent == [p] and PV.gguf_leaks(p) == []


def test_guard_refuses_what_it_cannot_clean(tmp_path, monkeypatch):
    p = tmp_path / "card.md"
    p.write_text(f"built at {HOMEISH}/x")
    monkeypatch.setattr(PV, "scrub_file", lambda path: False)        # a format it cannot rewrite
    with pytest.raises(PV.LocalPathLeak):
        PV.guard([str(tmp_path)], log=lambda *a: None)
