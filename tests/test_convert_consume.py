"""pollard-convert --consume-source: a shard is deleted once every tensor in it has been read, reads are copies (no
mapping keeps the deleted file's pages), files outside the model folder are never touched. Offline: plain files."""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_convert as PC  # noqa: E402


def _shard(path, tensors):
    """A minimal safetensors file: {name: bytes}."""
    meta, off = {}, 0
    for k, b in tensors.items():
        meta[k] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [off, off + len(b)]}
        off += len(b)
    head = json.dumps(meta).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(tensors.values()))
    return 8 + len(head)


def test_a_shard_goes_once_all_its_tensors_are_read(tmp_path):
    a = tmp_path / "model-00001-of-00002.safetensors"
    b = tmp_path / "model-00002-of-00002.safetensors"
    sa = _shard(a, {"x": b"\x01\x02\x03", "y": b"\x04\x05"})
    sb = _shard(b, {"z": b"\x06"})
    sc = PC.ShardConsumer(tmp_path, log=lambda m: None)
    assert bytes(sc.read(a, sa, 3)) == b"\x01\x02\x03" and a.exists()
    assert bytes(sc.read(a, sa + 3, 2)) == b"\x04\x05" and not a.exists()   # last tensor of a: gone
    assert b.exists()
    sc.read(b, sb, 1)
    assert not b.exists() and sc.freed > 0


def test_a_tensor_read_twice_does_not_count_twice_and_outside_files_stay(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    a = d / "model.safetensors"
    sa = _shard(a, {"x": b"\x01", "y": b"\x02"})
    other = tmp_path / "elsewhere.safetensors"
    so = _shard(other, {"q": b"\x09"})
    sc = PC.ShardConsumer(d, log=lambda m: None)
    sc.read(a, sa, 1)
    sc.read(a, sa, 1)                     # the same tensor again: y still unread
    assert a.exists()
    sc.read(other, so, 1)                 # not under the model folder: never deleted
    assert other.exists()


def test_a_short_read_raises_instead_of_writing_a_broken_gguf(tmp_path):
    import pytest
    a = tmp_path / "model.safetensors"
    sa = _shard(a, {"x": b"\x01\x02"})
    with pytest.raises(OSError):
        PC.ShardConsumer(tmp_path, log=lambda m: None).read(a, sa, 50)
    assert a.exists()
