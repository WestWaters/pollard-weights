"""Manifest writers in separate tool processes must not lose each other's updates."""
import errno
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import pollard_workspace as ws
import pollard_release as R
from test_release_free import hub, inputs, model, plan  # noqa: F401 -- shared fixtures


# Pause after a real manifest read. The second writer announces either its lock attempt (fixed
# code) or its stale read (old code), so the parent controls the overlap without scheduling sleeps.
WORKER = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
import pollard_workspace as ws
role, operation, key, path = sys.argv[2:]
announced = False
def ready():
    global announced
    if not announced:
        print("ready", flush=True)
        announced = True
if role == "second":
    if os.name == "nt":
        import msvcrt
        real_lock = msvcrt.locking
        def locking(fd, mode, size):
            if mode != msvcrt.LK_UNLCK:
                ready()
            return real_lock(fd, mode, size)
        msvcrt.locking = locking
    else:
        import fcntl
        real_lock = fcntl.flock
        def flock(fd, mode):
            if mode & fcntl.LOCK_EX:
                ready()
            return real_lock(fd, mode)
        fcntl.flock = flock
real_read = ws.read_manifest
def read(model):
    man = real_read(model)
    ready()
    assert sys.stdin.readline().strip() == "continue"
    return man
ws.read_manifest = read
if operation == "atomic":
    real_dump = ws.json.dump
    def dump(obj, file, **kwargs):
        real_dump(obj, file, **kwargs)
        ready()
        assert sys.stdin.readline().strip() == "continue"
    ws.json.dump = dump
    ws.write_json_atomic(path, {"writer": role})
elif operation == "build":
    ws.record_build(key, "gguf", path, tag="Q4")
elif operation == "gate":
    assert ws.record_gate(path, {"pass": True, "checks": []})
elif operation == "verified":
    assert ws.mark_verified(path, True, ppl=3.5)
elif operation == "calibration":
    ws.record_calibration(key, path, {"dataset": "local"})
'''


def _worker(role, operation, key, path):
    return subprocess.Popen(
        [sys.executable, "-I", "-c", WORKER, str(Path(ws.__file__).parent),
         role, operation, key, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _ready(process):
    executor = ThreadPoolExecutor(1)
    try:
        assert executor.submit(process.stdout.readline).result(timeout=20).strip() == "ready"
    finally:
        executor.shutdown(wait=False)


@pytest.mark.parametrize("operation", ["gate", "verified", "calibration"])
def test_build_and_metadata_updates_from_separate_processes(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    key = "Org/Model"
    old, new = tmp_path / "old.gguf", tmp_path / "new.gguf"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    ws.record_build(key, "gguf", str(old), extra={"uploaded": "Org/Release"})
    first = _worker("first", "build", key, new)
    second = None
    try:
        _ready(first)
        second = _worker("second", operation, key, old)
        _ready(second)
        _, errors = first.communicate("continue\n", timeout=20)
        assert first.returncode == 0, errors
        _, errors = second.communicate("continue\n", timeout=20)
        assert second.returncode == 0, errors
    finally:
        for process in (first, second):
            if process and process.poll() is None:
                process.kill()
                process.communicate()
    man = ws.read_manifest(key)
    builds = {b["path"]: b for b in man["builds"]}
    assert set(builds) == {str(old), str(new)}
    assert builds[str(old)]["uploaded"] == "Org/Release"
    if operation == "calibration":
        assert man["calibration"][0]["path"] == str(old)
    else:
        assert builds[str(old)]["verified"] is True
        if operation == "gate":
            assert builds[str(old)]["gate"]["pass"] is True
        else:
            assert builds[str(old)]["ppl"] == 3.5


def test_failed_manifest_write_keeps_previous_file_and_releases_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    key = "Org/Model"
    ws.record_build(key, "gguf", str(tmp_path / "old"))
    manifest = Path(ws._manifest_path(key))
    before = manifest.read_bytes()
    real_dump = json.dump

    def fail(obj, file, **kwargs):
        file.write('{"half":')
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(json, "dump", fail)
    with pytest.raises(OSError) as error:
        ws.record_build(key, "gguf", str(tmp_path / "bad"))
    assert error.value.errno == errno.ENOSPC
    assert manifest.read_bytes() == before
    assert not list(manifest.parent.glob("*.tmp"))
    monkeypatch.setattr(json, "dump", real_dump)
    ws.record_build(key, "gguf", str(tmp_path / "good"))
    assert len(ws.read_manifest(key)["builds"]) == 2


def test_atomic_writers_use_independent_temporary_files(tmp_path):
    path = tmp_path / "state.json"
    first = _worker("first", "atomic", "unused", path)
    second = None
    try:
        _ready(first)
        second = _worker("second", "atomic", "unused", path)
        _ready(second)
        _, errors = first.communicate("continue\n", timeout=20)
        assert first.returncode == 0, errors
        _, errors = second.communicate("continue\n", timeout=20)
        assert second.returncode == 0, errors
    finally:
        for process in (first, second):
            if process and process.poll() is None:
                process.kill()
                process.communicate()
    assert json.loads(path.read_text()) == {"writer": "second"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("confirmed", [True, False])
def test_publish_preserves_build_and_gate_updates_during_upload(
        model, inputs, hub, monkeypatch, confirmed):
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    key = c["f16"]
    old = Path(ws.model_dir(key, create=True)) / "Tiny-1B-Pollard-Q6_K.gguf"
    new = old.with_name("Tiny-1B-Pollard-Q4_K.gguf")
    old.write_bytes(b"GGUF" * 10)
    new.write_bytes(b"GGUF" * 5)
    ws.record_build(key, "gguf", str(old), tag="Q6_K")
    ws.record_gate(str(old), {"pass": True})

    def confirm(api, repo, pairs):
        process = _worker("first", "build", key, new)
        try:
            _ready(process)
            _, errors = process.communicate("continue\n", timeout=20)
            assert process.returncode == 0, errors
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        ws.record_gate(str(old), {"pass": True, "checks": [{"check": "new measurement", "ok": True}]})
        return [] if confirmed else ["missing remote file"]

    monkeypatch.setattr(R, "remote_mismatches", confirm)
    step = R._publish_rung_step(c, "gguf", "Q6_K", str(old))
    if confirmed:
        assert R.publish_rung(step, c, cfg, True, out=lambda *_: None) == "done"
    else:
        with pytest.raises(RuntimeError, match="not intact"):
            R.publish_rung(step, c, cfg, True, out=lambda *_: None)
    builds = {b["path"]: b for b in ws.read_manifest(key)["builds"]}
    assert set(builds) == {str(old), str(new)}
    assert builds[str(old)]["gate"]["checks"] == [{"check": "new measurement", "ok": True}]
    if confirmed:
        assert builds[str(old)]["uploaded"] == step.repo
        assert builds[str(old)]["freed"] == "uploaded + confirmed on the Hub"
        assert not old.exists()
    else:
        assert "upload not confirmed" in builds[str(old)]["kept"]
        assert old.exists()


def test_recording_a_rebuilt_path_still_resets_its_previous_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    path = str(tmp_path / "same.gguf")
    ws.record_build("Org/Model", "gguf", path, verified=True, extra={"uploaded": "Org/Release"})
    ws.record_build("Org/Model", "gguf", path)
    builds = ws.read_manifest("Org/Model")["builds"]
    assert len(builds) == 1
    assert builds[0]["verified"] is None
    assert "uploaded" not in builds[0]


def test_mark_verified_uses_discovered_manifest_when_local_source_was_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    source = tmp_path / "local-source"
    source.mkdir()
    path = str(tmp_path / "build.gguf")
    ws.record_build(str(source), "gguf", path)
    manifest = Path(ws._manifest_path(str(source)))
    source.rmdir()
    assert ws.mark_verified(path, True, ppl=4.5)
    build = json.loads(manifest.read_text())["builds"][0]
    assert build["verified"] is True
    assert build["ppl"] == 4.5
    assert ws.list_models() == ["local-source"]
