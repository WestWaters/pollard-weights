"""The machine locks cover a release's COMPUTE steps only -- never its publish / upload / relay steps.

An upload-only release (rungs already built, hours of relaying to the Hub) held the shared Mac lock the whole time
and every test run on the machine waited behind it. Now each build / measure / verify step takes the machine lock
($POLLARD_HOME/.pollard.lock, waiting instead of refusing) and the shared POLLARD_HEAVY_LOCK directory for its own
length; publish / upload / card / finalize steps run outside them.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as A  # noqa: E402
import pollard_release as R  # noqa: E402

S = R.Step


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


@pytest.mark.parametrize("st,heavy", [
    (S(id="gguf/ladder", lane="gguf", kind="build"), True),
    (S(id="prep/imatrix", lane="prep", kind="prep", fn="imatrix"), True),
    (S(id="prep/calib", lane="prep", kind="prep", cmd=["pollard-calib"]), True),
    (S(id="gguf/Q4/measure", lane="gguf", kind="measure"), True),
    (S(id="gguf/Q4/taskeval", lane="gguf", kind="measure", fn="taskeval_rung"), True),
    (S(id="gguf/Q4/verify", lane="gguf", kind="verify"), True),
    (S(id="gguf/complete", lane="gguf", kind="gate"), True),
    (S(id="gguf/Q4/publish", lane="gguf", kind="publish", fn="publish_rung"), False),
    (S(id="gguf/upload", lane="gguf", kind="upload", fn="upload"), False),
    (S(id="gguf/card-upload", lane="gguf", kind="upload", fn="card_upload"), False),
    (S(id="gguf/card", lane="gguf", kind="card", cmd=["pollard-card"]), False),
    (S(id="release/public", lane="release", kind="finalize", fn="go_public"), False),
    (S(id="prep/fetch", lane="prep", kind="prep", fn="fetch"), False),
    (S(id="prep/mirror", lane="prep", kind="prep", cmd=["pollard-mlx", "--mirror-only"]), False),
])
def test_only_compute_steps_are_heavy(st, heavy):
    assert R.is_heavy(st) is heavy


def _run(tmp_path, monkeypatch, steps, seen):
    lock = str(tmp_path / "mac-heavy.lock")
    monkeypatch.setenv(A.HEAVY_LOCK_ENV, lock)
    monkeypatch.setattr(R, "engine_arch", lambda c: None)

    def held():
        return {"dir": os.path.isdir(lock), "machine": os.path.exists(os.environ["POLLARD_MACHINE_LOCK"]),
                "marker": os.environ.get(A.HEAVY_HELD_ENV)}

    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,), env=None: seen.append(("cmd", held(), env)))
    monkeypatch.setattr(R, "publish_rung", lambda st, c, cfg, up, out=print: seen.append(("publish", held(), None))
                        or "done")
    monkeypatch.setattr(R, "upload", lambda st, c, cfg, up: seen.append(("upload", held(), None)) or "uploaded")
    c = {"state": str(tmp_path / "release-state.json"), "src": "M", "mdir": str(tmp_path)}
    st = R.execute(steps, c, {}, run=True, run_upload=True, have={"any": True}, out=lambda *a: None)
    return lock, st


def test_a_measure_holds_the_locks_and_a_publish_does_not(tmp_path, monkeypatch):
    seen = []
    lock, st = _run(tmp_path, monkeypatch, [
        S(id="gguf/Q4/measure", lane="gguf", rung="Q4", kind="measure", requires="any", cmd=["pollard-bench"]),
        S(id="gguf/Q4/publish", lane="gguf", rung="Q4", kind="publish", requires="any", fn="publish_rung"),
        S(id="gguf/upload", lane="gguf", kind="upload", requires="any", fn="upload")], seen)
    assert [s[0] for s in seen] == ["cmd", "publish", "upload"]
    assert seen[0][1] == {"dir": True, "machine": True, "marker": lock}
    for _, h, _ in seen[1:]:
        assert h == {"dir": False, "machine": False, "marker": None}
    assert all(v == "done" for v in st.values()), st
    assert not os.path.exists(lock) and A.HEAVY_HELD_ENV not in os.environ


def test_an_upload_only_release_never_waits_for_a_busy_machine(tmp_path, monkeypatch):
    """Someone else's job holds the shared lock: the relay / upload steps carry on regardless."""
    lock = str(tmp_path / "mac-heavy.lock")
    os.mkdir(lock)
    with open(os.path.join(lock, "owner"), "w") as f:
        f.write(f"{os.getppid()} pytest-elsewhere\n")              # a live pid that is not us
    seen = []
    t0 = time.time()
    _, st = _run(tmp_path, monkeypatch, [
        S(id="gguf/Q4/publish", lane="gguf", rung="Q4", kind="publish", requires="any", fn="publish_rung"),
        S(id="gguf/upload", lane="gguf", kind="upload", requires="any", fn="upload")], seen)
    assert time.time() - t0 < 5 and [s[0] for s in seen] == ["publish", "upload"]
    assert os.path.isdir(lock) and open(os.path.join(lock, "owner")).read().startswith(str(os.getppid()))


def test_a_child_env_built_before_the_hold_gets_the_held_marker(tmp_path, monkeypatch):
    """A streamed / EXL3 step passes its own env dict: the marker still reaches the child, so a child that honours
    POLLARD_HEAVY_LOCK (pollard-draft train) does not wait on the lock held for it."""
    monkeypatch.setattr(R, "EXL3_ENV_LANES", ("exl3",))
    monkeypatch.setattr(R, "exl3_env", lambda e: dict(e or os.environ, EXL3="1"))
    seen = []
    lock, _ = _run(tmp_path, monkeypatch, [
        S(id="exl3/4.0/build", lane="exl3", rung="4.0", kind="build", requires="any", cmd=["pollard-exl3"])], seen)
    env = seen[0][2]
    assert env[A.HEAVY_HELD_ENV] == lock and env["EXL3"] == "1"


def test_a_heavy_step_waits_for_another_holder(tmp_path, monkeypatch):
    lock = str(tmp_path / "mac-heavy.lock")
    monkeypatch.setenv(A.HEAVY_LOCK_ENV, lock)
    os.mkdir(lock)
    with open(os.path.join(lock, "owner"), "w") as f:
        f.write(f"{os.getppid()} ci-run\n")
    threading.Timer(0.3, lambda: __import__("shutil").rmtree(lock)).start()
    t0 = time.time()
    with A.heavy_step("test build", poll=0.05, log=None):
        assert time.time() - t0 >= 0.25 and os.path.isdir(lock)
        assert open(os.path.join(lock, "owner")).read().split()[0] == str(os.getpid())
    assert not os.path.exists(lock)


def test_a_dead_holder_is_taken_over(tmp_path):
    lock = str(tmp_path / "mac-heavy.lock")
    os.mkdir(lock)
    with open(os.path.join(lock, "owner"), "w") as f:
        f.write(f"{_dead_pid()} mlx-stream Thu Oct 23\n")
    t0 = time.time()
    with A.heavy_dir_lock(lock, poll=5, log=None):
        assert time.time() - t0 < 2
    assert not os.path.exists(lock)


def test_under_a_holder_the_dir_lock_is_a_no_op(tmp_path, monkeypatch):
    lock = str(tmp_path / "mac-heavy.lock")
    os.mkdir(lock)                                                   # held by our parent release
    monkeypatch.setenv(A.HEAVY_HELD_ENV, lock)
    with A.heavy_dir_lock(lock, poll=5, log=None):
        pass
    assert os.path.isdir(lock), "a child must not release its parent's lock"


def test_parallel_heavy_steps_share_one_hold(tmp_path, monkeypatch):
    """The parallel scheduler runs several heavy steps of one release at once: the first takes the locks, the last
    releases them."""
    lock = str(tmp_path / "mac-heavy.lock")
    monkeypatch.setenv(A.HEAVY_LOCK_ENV, lock)
    with A.heavy_step("a", log=None):
        with A.heavy_step("b", log=None):
            assert os.path.isdir(lock)
        assert os.path.isdir(lock), "released while another step still ran"
    assert not os.path.exists(lock) and not os.path.exists(os.environ["POLLARD_MACHINE_LOCK"])


def test_force_takes_no_lock(tmp_path, monkeypatch):
    lock = str(tmp_path / "mac-heavy.lock")
    monkeypatch.setenv(A.HEAVY_LOCK_ENV, lock)
    with A.heavy_step("x", force=True, log=None):
        assert not os.path.exists(lock)


def test_machine_lock_waits_or_refuses(tmp_path):
    import json
    path = os.environ["POLLARD_MACHINE_LOCK"]
    with open(path, "w") as f:
        json.dump({"pid": os.getppid(), "what": "another pollard"}, f)
    with pytest.raises(SystemExit):
        with A.MachineLock("x"):
            pass
    threading.Timer(0.3, lambda: os.remove(path)).start()
    with A.MachineLock("x", wait=True, poll=0.05):
        assert json.load(open(path))["pid"] == os.getpid()
    assert not os.path.exists(path)


def test_release_main_holds_no_whole_run_lock():
    src = (ROOT / "tools" / "pollard_release.py").read_text(encoding="utf-8")
    body = src.split("\ndef main(", 1)[1]
    assert "MachineLock(" not in body, "the release holds the machine lock across its uploads again"
