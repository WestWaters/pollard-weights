"""The streamed MLX verify from a remote mirror on a 16 GB Mac took the disk to 0 bytes: the mirror's cache filled it
down to a FIXED 3 GB reserve that ignored swap growth and the state / result writes, results-mlx.json and
release-state.json failed with ENOSPC, and the OS ran with a full disk for minutes.

Now: the reserve is never under the plan's disk_margin_gb, and on macOS never under 10 GB (plus an optional share of the volume);
free disk is re-checked before EACH tensor lands (least recently used cache evicted first); a disk that is still short
holds the step (pollard-verify exit 3 -> release HELD, rung kept), never ENOSPC; and every JSON the step writes goes
through tmp + rename, so a full disk leaves the previous file whole. numpy-only: every CI shape."""
from __future__ import annotations

import collections
import errno
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import pollard_mlx as M  # noqa: E402
import pollard_stream as S  # noqa: E402
import pollard_workspace as ws  # noqa: E402

GB = 1e9


# ----------------------------------------------------------------------------------------------- the floor

def test_floor_is_margin_off_mac_and_10gb_on_mac(monkeypatch):
    monkeypatch.delenv("POLLARD_DISK_MARGIN_GB", raising=False)
    assert M.disk_reserve_gb(total_bytes=245 * GB, platform="linux") == 3.0
    assert M.disk_reserve_gb(total_bytes=245 * GB, platform="darwin") == pytest.approx(10.0)   # flat floor by default
    monkeypatch.setattr(M, "MAC_DISK_FLOOR_FRAC", 0.10)
    assert M.disk_reserve_gb(total_bytes=245 * GB, platform="darwin") == pytest.approx(24.5)   # opt-in share of the volume
    assert M.disk_reserve_gb(total_bytes=50 * GB, platform="darwin") == 10.0                   # never under 10 GB
    monkeypatch.setenv("POLLARD_DISK_MARGIN_GB", "40")                                        # the plan's margin
    assert M.disk_reserve_gb(total_bytes=245 * GB, platform="darwin") == 40.0
    assert M.disk_reserve_gb(total_bytes=245 * GB, platform="linux") == 40.0
    monkeypatch.setenv("POLLARD_DISK_MARGIN_GB", "1")                                         # cannot lower it
    assert M.disk_reserve_gb(total_bytes=245 * GB, platform="darwin") == pytest.approx(24.5)


def test_writer_and_verify_no_longer_hardcode_3gb():
    src = (ROOT / "tools" / "pollard_mlx.py").read_text()
    run = src[src.index("    def run(self, rows):"):src.index("    def logits(self, b, start=0):")]
    assert "reserve_gb=3.0" not in run, "the streamed verify sizes its reserve from the floor"
    i = src.index("    def _reserve_gb(self):")
    w = src[i:src.index("    def __enter__(self):", i)]
    assert "+ 3.0" not in w and "floor_gb()" in w


# ---------------------------------------------------------------------------------------- a simulated disk

def _src(tmp_path, n=6, size=1 << 16):
    """A one-shard safetensors source with `n` random tensors of `size` bytes each (adjacent: one fetch range)."""
    d = tmp_path / "remote"
    d.mkdir()
    rng = np.random.default_rng(0)
    hdr, blobs, off = {}, [], 0
    for i in range(n):
        b = rng.integers(0, 255, size, dtype=np.uint8).tobytes()
        hdr[f"model.layers.{i}.mlp.weight"] = {"dtype": "U8", "shape": [size], "data_offsets": [off, off + size]}
        blobs.append(b)
        off += size
    h = json.dumps(hdr).encode()
    with open(d / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return d


class Disk:
    """Free bytes = start - what the mirror wrote + what it punched - what 'swap' took since. `swap(i)` is called on
    every free-space read, so a test can grow it between tensor fetches."""

    def __init__(self, mirror, start, swap=lambda i: 0):
        self.m, self.start, self.swap, self.written, self.reads, self.low = mirror, start, swap, 0, 0, None

    def free(self):
        self.reads += 1
        return int(self.start - self.written + self.m.evicted - self.swap(self.reads))

    def note_write(self, n):
        self.written += n
        f = self.start - self.written + self.m.evicted - self.swap(self.reads)
        self.low = f if self.low is None else min(self.low, f)


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(M.RemoteMirror, "_cmd", lambda self: [sys.executable, "-"])
    monkeypatch.delenv("POLLARD_DISK_MARGIN_GB", raising=False)
    src = _src(tmp_path)
    m = M.RemoteMirror.create(f"ssh://u@h{src}?python=py", str(tmp_path / "mirror"), log=lambda *a: None)
    m.total_bytes = lambda: 0                     # the floor = 10 GB on macOS, the 3 GB margin elsewhere
    return m


def _attach(m, disk, monkeypatch):
    m.free_bytes = disk.free
    real = os.pwrite

    def pwrite(fd, b, pos):
        disk.note_write(len(b))
        return real(fd, b, pos)
    monkeypatch.setattr(M.os, "pwrite", pwrite)


def _keys(m):
    return sorted(m.entries, key=lambda k: m.entries[k][1])


def test_reserve_holds_through_a_fetch_while_swap_grows(mirror, monkeypatch):
    """Room for the whole block at the start; swap then eats a tensor's worth per read. Each tensor's check sees it,
    evicts LRU cache, and the free disk never drops under the floor at any write."""
    m, ks = mirror, _keys(mirror)
    n = m.entries[ks[0]][2]
    R = m.floor_gb() * GB
    m.ensure(ks[:3])                               # cache: 3 old tensors (LRU first), fetched on a roomy disk
    m.done(ks[:3])
    disk = Disk(m, R + 3.2 * n, swap=lambda i: 0)
    disk.written = 0
    _attach(m, disk, monkeypatch)
    grow = {"on": False, "base": 0}

    def swap(i):
        if not grow["on"]:
            grow["on"], grow["base"] = True, i
        return (i - grow["base"]) * n              # a tensor's worth of swap per free-space read
    disk.swap = swap
    m.ensure(ks[3:5])
    assert all(k in m.present for k in ks[3:5])
    assert disk.low >= R, f"free disk fell to {disk.low - R:+.0f} bytes over the reserve"
    assert m.evicted >= n and ks[0] not in m.present, "the least recently used went first"


def test_lru_is_evicted_before_fetching_never_the_recent_or_pinned(mirror, monkeypatch):
    m, ks = mirror, _keys(mirror)
    n = m.entries[ks[0]][2]
    R = m.floor_gb() * GB
    m.ensure([ks[0]])
    m.ensure([ks[1]])
    m.ensure([ks[2]])                              # pinned, never evicted
    m.done([ks[0]])
    m.done([ks[1]])
    m.ensure([ks[0]])                              # touched again: now ks[1] is the least recently used
    m.done([ks[0]])
    disk = Disk(m, R + n // 2)                     # one tensor more only fits if one is evicted
    _attach(m, disk, monkeypatch)
    m.ensure([ks[3]])
    assert ks[1] not in m.present and ks[0] in m.present and ks[2] in m.present and ks[3] in m.present
    assert disk.low >= R


def test_still_short_holds_cleanly_and_leaves_no_half_tensor(mirror, monkeypatch):
    """Swap outruns what the cache can give back mid-fetch: OutOfMemoryBudget (the step is held), never ENOSPC; the
    tensors that landed whole are recorded, the half-written one is punched and never listed as present."""
    m, ks = mirror, _keys(mirror)
    n = m.entries[ks[0]][2]
    R = m.floor_gb() * GB
    disk = Disk(m, R + 6.5 * n)
    _attach(m, disk, monkeypatch)
    disk.swap = lambda i: 0 if i <= 2 else 4 * n   # fine for the first tensor, then swap takes 4 tensors' worth
    punched = []
    real_punch = M.punch
    monkeypatch.setattr(M, "punch", lambda p, off, k: punched.append((off, k)) or real_punch(p, off, k))
    with pytest.raises(S.OutOfMemoryBudget, match="disk"):
        m.ensure(ks)
    assert disk.low >= R
    st = json.loads((Path(m.dir) / M.MIRROR_STATE).read_text())
    assert set(st) == {k for k in m.present}
    raw = (Path(m.dir).parent / "remote" / "model.safetensors").read_bytes()
    loc = (Path(m.dir) / "model.safetensors").read_bytes()
    for k in st:                                   # what the state lists is whole on disk
        f, off, size = m.entries[k]
        assert loc[off:off + size] == raw[off:off + size]
    for k in set(ks) - set(st):
        f, off, size = m.entries[k]
        assert loc[off:off + size] == bytes(size), f"{k}: a half-fetched tensor is punched out"


def test_enospc_from_the_write_itself_is_a_hold_not_a_crash(mirror, monkeypatch):
    m, ks = mirror, _keys(mirror)

    def full(fd, b, pos):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(M.os, "pwrite", full)
    with pytest.raises(S.OutOfMemoryBudget, match="full while fetching"):
        m.ensure(ks[:2])
    assert not m.present


def test_full_disk_state_write_still_evicts(mirror, monkeypatch):
    """A disk too full to write even the cache's state file: the state goes (refetch later, never a zero read) and the
    punch still frees the bytes."""
    m, ks = mirror, _keys(mirror)
    m.ensure(ks[:2])
    m.done(ks[:2])
    monkeypatch.setattr(ws, "write_json_atomic", lambda *a, **k: (_ for _ in ()).throw(
        OSError(errno.ENOSPC, "No space left on device")))
    freed = m.make_room(1e18)
    assert freed == sum(m.entries[k][2] for k in ks[:2]) and not m.present
    assert not (Path(m.dir) / M.MIRROR_STATE).exists()
    assert not M.RemoteMirror(m.dir, log=lambda *a: None).present


# --------------------------------------------------------------------------- the verify's own writes

def test_reference_logits_get_room_from_the_mirror_then_results_are_written(mirror, tmp_path, monkeypatch):
    import shutil
    import pollard_verify as V
    m, ks = mirror, _keys(mirror)
    m.ensure(ks)
    m.done(ks)
    cache = sum(m.entries[k][2] for k in ks)
    R = M.disk_reserve_gb(total_bytes=0)
    du = collections.namedtuple("du", "total used free")
    base = {"free": R * GB + 1000}
    monkeypatch.setattr(shutil, "disk_usage", lambda p: du(0, 0, int(base["free"] + m.evicted)))
    m.free_bytes = lambda: int(base["free"] + m.evicted)
    ref = tmp_path / "home" / "cache" / "verify-ref-x.npy"
    ref.parent.mkdir(parents=True)
    V.ref_disk_room(str(ref), cache // 2, m, log=lambda *a: None)          # fits once the cache gives room
    assert m.evicted >= cache // 2 - 1000
    res = tmp_path / "results-mlx.json"
    ws.merge_results(str(res), "M-MLX-4bit", {"kld": 0.1})
    assert json.loads(res.read_text())["M-MLX-4bit"]["kld"] == 0.1
    with pytest.raises(S.OutOfMemoryBudget, match="reference logits"):     # more than disk + cache: held
        V.ref_disk_room(str(ref), cache * 4, m, log=lambda *a: None)


def test_verify_holds_with_exit_3_and_keeps_the_old_results(tmp_path, monkeypatch, capsys):
    import pollard_verify as V
    res = tmp_path / "results-mlx.json"
    res.write_text(json.dumps({"M-MLX-3bit": {"kld": 0.2}}))
    before = res.read_text()

    def short(*a, **k):
        raise S.OutOfMemoryBudget("disk: the mirror needs 1.00 GB more and only 0.10 GB can be freed")
    monkeypatch.setattr(V, "measure_mlx", short)
    (tmp_path / "rung").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "e.txt").write_text("x")
    monkeypatch.setattr(sys, "argv", ["pollard-verify", "--lane", "mlx", "--model", str(tmp_path / "rung"),
                                      "--source", str(tmp_path / "src"), "--rung", "M-MLX-4bit",
                                      "--results", str(res), "--eval", str(tmp_path / "e.txt")])
    with pytest.raises(SystemExit) as e:
        V.main()
    assert e.value.code == V.EXIT_HELD == M.EXIT_HELD
    assert "HELD: disk" in capsys.readouterr().out
    assert res.read_text() == before


def test_release_holds_the_mlx_verify_on_exit_3_and_passes_the_plan_margin(tmp_path, monkeypatch):
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    seen = tmp_path / "env.txt"
    code = f"import os, sys; open({str(seen)!r}, 'w').write(os.environ.get('POLLARD_DISK_MARGIN_GB', '')); sys.exit(3)"
    steps = [R.Step(id="mlx/4bit/verify", lane="mlx", rung="4bit", kind="verify", requires="any",
                    cmd=[sys.executable, "-c", code], out=str(tmp_path / "r4"), held_codes=(R.EXIT_HELD,)),
             R.Step(id="mlx/4bit/publish", lane="mlx", rung="4bit", kind="publish", requires="any",
                    fn="publish_rung", out=str(tmp_path / "r4"))]
    c = {"state": str(tmp_path / "release-state.json"), "mdir": str(tmp_path)}
    st = R.execute(steps, c, {"disk_margin_gb": 12}, run=True, have={"any": True}, out=lambda *a: None)
    assert st["mlx/4bit/verify"].startswith("held:")
    assert seen.read_text() == "12.0"
    assert "mlx/4bit/verify" not in R.load_state(c["state"]), "held: re-run on the next release run"
    assert str(st["mlx/4bit/publish"]).startswith(("kept", "skipped")), st["mlx/4bit/publish"]


def test_mlx_verify_step_carries_the_held_code(tmp_path, monkeypatch):
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    src = tmp_path / "M"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    cfg = R.load_plan(None, "text")
    cfg.update(calib_file=str(tmp_path / "c.txt"), heldout_file=str(tmp_path / "h.txt"),
               sensitivity=str(tmp_path / "s.json"))
    _, _, steps = R.plan_steps(cfg, str(src), {"mlx"})
    ver = [s for s in steps if s.lane == "mlx" and s.kind == "verify"]
    assert ver and all(R.EXIT_HELD in (s.held_codes or ()) for s in ver)


# --------------------------------------------------------------------------------- atomic JSON on ENOSPC

def _enospc_dump(monkeypatch, module):
    real = json.dump

    def dump(obj, f, **kw):
        f.write('{"half": ')
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(module.json, "dump", dump)
    return real


def test_results_write_on_enospc_keeps_the_old_file(tmp_path, monkeypatch):
    res = tmp_path / "results-mlx.json"
    ws.merge_results(str(res), "M-MLX-3bit", {"kld": 0.2})
    before = res.read_text()
    _enospc_dump(monkeypatch, ws)
    with pytest.raises(OSError) as e:
        ws.merge_results(str(res), "M-MLX-4bit", {"kld": 0.1})
    assert e.value.errno == errno.ENOSPC and "previous file is kept" in str(e.value)
    assert res.read_text() == before
    assert not (tmp_path / "results-mlx.json.tmp").exists()


def test_release_state_write_on_enospc_keeps_the_old_file(tmp_path, monkeypatch, capsys):
    import pollard_release as R
    p = tmp_path / "release-state.json"
    assert R.save_entry(str(p), "mlx/3bit/build", {"status": "done"})
    before = p.read_text()
    _enospc_dump(monkeypatch, ws)
    with pytest.raises(OSError):
        R.save_state(str(p), {"x": 1})
    assert R.save_entry(str(p), "mlx/3bit/verify", {"status": "done"}) is False      # warned, not crashed
    assert "runs again" in capsys.readouterr().err
    assert p.read_text() == before and json.loads(before) == {"mlx/3bit/build": {"status": "done"}}
    assert not (tmp_path / "release-state.json.tmp").exists()
