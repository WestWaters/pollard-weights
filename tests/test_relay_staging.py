"""Relay upload staging lives in the relay machine's Pollard workspace, resumes, and is reclaimed.

The relay (--upload-via ssh://...) copies a built rung to another machine (the one with the HF login) and uploads it
from there. These tests stand in for ssh / scp with a local runner: the "remote" python runs here, against a fake
huggingface_hub whose repos are folders, and scp is a file copy (optionally dying part way, like a box rebooted by
Windows Update mid-transfer)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import types

import pytest

FAKEHUB = '''
import os, hashlib, shutil
ROOT = os.environ["FAKEHUB"]
class _F:
    def __init__(self, path, size, sha):
        self.path, self.size = path, size
        self.lfs = type("L", (), {"sha256": sha})() if sha else None
class HfApi:
    def create_repo(self, repo, repo_type="model", exist_ok=True, private=True):
        os.makedirs(os.path.join(ROOT, repo), exist_ok=True)
        return repo
    def _files(self, repo):
        base = os.path.join(ROOT, repo)
        for r, _, fs in os.walk(base):
            for f in fs:
                yield os.path.relpath(os.path.join(r, f), base).replace(os.sep, "/"), os.path.join(r, f)
    def list_repo_files(self, repo):
        return sorted(p for p, _ in self._files(repo))
    def list_repo_tree(self, repo, recursive=True):
        if os.environ.get("FAKEHUB_DOWN"):
            raise OSError("hub unreachable")
        for p, full in self._files(repo):
            sha = hashlib.sha256(open(full, "rb").read()).hexdigest()
            yield _F(p, os.path.getsize(full), sha if p.endswith(".safetensors") else None)
    def upload_folder(self, folder_path, path_in_repo, repo_id, ignore_patterns=None):
        if os.environ.get("FAKEHUB_FAIL_UPLOAD"):
            raise RuntimeError("upload failed")
        shutil.copytree(folder_path, os.path.join(ROOT, repo_id, path_in_repo), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*(ignore_patterns or [])))
        return "ok"
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        dst = os.path.join(ROOT, repo_id, path_in_repo)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(path_or_fileobj, dst)
        return "ok"
'''


class FakeRemote:
    """ssh / scp stand-in. `die_after` = how many bytes scp may write in total before it 'loses the connection'
    (the last file is left truncated, as a reboot mid-scp leaves it)."""

    def __init__(self, tmp, hub, free_gb=500.0, die_after=None):
        self.tmp, self.hub, self.free, self.die_after, self.cmds, self.sent = tmp, hub, free_gb, die_after, [], []
        (tmp / "fakehub.py").write_text(FAKEHUB)

    def __call__(self, cmd, input=None, capture_output=False, text=False, **kw):
        self.cmds.append(cmd)
        if cmd[0] == "scp":
            dst = cmd[-1].split(":", 1)[1]
            for src in cmd[cmd.index("-q") + 1:-1]:
                data = open(src, "rb").read()
                if self.die_after is not None and len(data) > self.die_after:
                    open(os.path.join(dst, os.path.basename(src)), "wb").write(data[:self.die_after])
                    return types.SimpleNamespace(returncode=1, stdout="", stderr="connection lost")
                if self.die_after is not None:
                    self.die_after -= len(data)
                shutil.copy(src, dst)
                self.sent.append(os.path.basename(src))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        code = input.replace("from huggingface_hub import HfApi", "from fakehub import HfApi")
        code = code.replace("shutil.disk_usage(", f"(lambda p: type('U', (), {{'free': {self.free * 1e9}}}))(")
        env = dict(os.environ, PYTHONPATH=str(self.tmp), FAKEHUB=str(self.hub))
        return subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True, env=env)

    @property
    def ran_on_remote(self):
        return [c for c in self.cmds if c[0] == "ssh"]


@pytest.fixture
def R(tmp_path, monkeypatch):
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "machome"))
    for k in ("FAKEHUB_DOWN", "FAKEHUB_FAIL_UPLOAD"):
        monkeypatch.delenv(k, raising=False)
    return R


def _rung(tmp_path, name="M-MLX-4bit", big=b"\x01" * 6000):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text('{"a": 1}')
    (d / "model.safetensors").write_bytes(big)
    return d


def _relay(R, tmp_path, fake, home=None, extra=""):
    home = home or (tmp_path / "boxhome")
    logs = []
    api = R.RelayApi(f"ssh://jwate@box:3333?python=py&min_free_gb=1&home={home}{extra}", runner=fake,
                     out=logs.append)
    return api, home, logs


def _mark(stage):
    return json.loads((stage / ".pollard-staging.json").read_text())


# ------------------------------------------------------------------------------------------- where staging lives

def test_staging_path_is_inside_the_relays_workspace_and_fixed_per_rung(R, tmp_path):
    fake = FakeRemote(tmp_path, tmp_path / "hub")
    api, home, _ = _relay(R, tmp_path, fake)
    stage = api.stage_dir("Org/M-Pollard", "MLX/4bit")
    assert stage == f"{home}/staging/Org__M-Pollard/MLX__4bit"
    assert stage == api.stage_dir("Org/M-Pollard", "MLX/4bit"), "deterministic: a rerun lands in the same folder"
    assert api.stage_dir("Org/M-Pollard", "MLX/4bit/README.md") != stage, "a file never nests inside a rung's copy"
    import pollard_workspace as ws
    assert ws.staging_dir("Org/M-Pollard", "MLX/4bit", home=str(home)) == stage


def test_home_is_discovered_from_the_relay_when_not_given(R, tmp_path, monkeypatch):
    """No home= in the URL: the relay's python is asked (its pollard_workspace, else its POLLARD_HOME)."""
    fake = FakeRemote(tmp_path, tmp_path / "hub")
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "remotehome"))      # what the 'remote' python sees
    logs = []
    api = R.RelayApi(f"ssh://jwate@box/C:/pollard/mlx-staging?python=py", runner=fake, out=logs.append)
    assert api.staging == f"{tmp_path / 'remotehome'}/staging".replace(os.sep, "/")
    assert any("no longer used" in m for m in logs), "the old URL staging path is named as unused"


def test_upload_stages_in_workspace_then_cleans_up_on_success(R, tmp_path):
    hub = tmp_path / "hub"
    fake = FakeRemote(tmp_path, hub)
    api, home, _ = _relay(R, tmp_path, fake)
    rung = _rung(tmp_path)
    api.create_repo("Org/M-Pollard")
    api.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    assert sorted(api.list_repo_files("Org/M-Pollard")) == ["MLX/4bit/config.json", "MLX/4bit/model.safetensors"], \
        "the marker file is never uploaded"
    scp_dest = [c[-1].split(":", 1)[1] for c in fake.cmds if c[0] == "scp"]
    assert scp_dest and all(d.startswith(f"{home}/staging/") for d in scp_dest), "nothing staged outside the workspace"
    assert os.path.isdir(home / "staging") and not any((home / "staging").iterdir()), "staging deleted after the check"
    assert rung.is_dir(), "nothing local is touched"
    api.upload_file(str(rung / "config.json"), "MLX/4bit/extra.json", "Org/M-Pollard")
    assert "MLX/4bit/extra.json" in api.list_repo_files("Org/M-Pollard")
    assert not any((home / "staging").iterdir())


def test_upload_failure_keeps_the_staging_copy(R, tmp_path, monkeypatch):
    fake = FakeRemote(tmp_path, tmp_path / "hub")
    api, home, logs = _relay(R, tmp_path, fake)
    rung = _rung(tmp_path)
    monkeypatch.setenv("FAKEHUB_FAIL_UPLOAD", "1")
    with pytest.raises(RuntimeError, match="upload failed"):
        api.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    stage = home / "staging" / "Org__M-Pollard" / "MLX__4bit"
    assert (stage / "model.safetensors").stat().st_size == 6000, "kept for the next run to resume"
    m = _mark(stage)
    assert m["repo_id"] == "Org/M-Pollard" and m["path_in_repo"] == "MLX/4bit"
    assert m["files"]["MLX/4bit/model.safetensors"] == [6000, hashlib.sha256(b"\x01" * 6000).hexdigest()]
    assert any("is kept" in x for x in logs)


def test_rerun_resumes_and_skips_files_already_there(R, tmp_path):
    """The box reboots mid-scp: the partial copy stays; the rerun sends only what is missing or truncated."""
    hub = tmp_path / "hub"
    rung = _rung(tmp_path)
    (rung / "tokenizer.json").write_text("t" * 100)
    dying = FakeRemote(tmp_path, hub, die_after=200)           # config.json lands, model.safetensors is cut short
    api, home, _ = _relay(R, tmp_path, dying)
    with pytest.raises(RuntimeError, match="scp exited 1.*kept"):
        api.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    stage = home / "staging" / "Org__M-Pollard" / "MLX__4bit"
    assert (stage / "model.safetensors").stat().st_size < 6000, "a truncated file is left behind"
    assert dying.sent == ["config.json"]
    # a stray file from another build of the rung must not ride along in the upload
    (stage / "stale.bin").write_bytes(b"old")

    fake = FakeRemote(tmp_path, hub)
    api2, _, logs = _relay(R, tmp_path, fake)
    api2.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    assert fake.sent == ["model.safetensors", "tokenizer.json"], \
        "a file already there at the same size + sha256 is not sent again; the truncated one is"
    assert any("resuming: 1 of 3" in m for m in logs)
    assert "MLX/4bit/stale.bin" not in api2.list_repo_files("Org/M-Pollard")
    assert not stage.exists()

    # same size, different bytes (a corrupt partial): the sha256 catches it and the file is re-sent
    s5 = home / "staging" / "Org__M-Pollard" / "MLX__5bit"
    _stage_dir(s5.parent, s5.name, files={"config.json": (rung / "config.json").read_bytes(),
                                          "tokenizer.json": b"t" * 100, "model.safetensors": b"\x09" * 6000})
    again = FakeRemote(tmp_path, hub)
    api4, _, _ = _relay(R, tmp_path, again)
    api4.upload_folder(str(rung), "MLX/5bit", "Org/M-Pollard")
    assert again.sent == ["model.safetensors"]


def test_disk_check_counts_only_the_bytes_still_to_send(R, tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    rung = _rung(tmp_path, big=b"\x03" * 3_000_000)                       # 3 MB
    monkeypatch.setenv("FAKEHUB_FAIL_UPLOAD", "1")
    api, home, _ = _relay(R, tmp_path, FakeRemote(tmp_path, hub))
    with pytest.raises(RuntimeError, match="upload failed"):
        api.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")        # fully staged, upload failed
    monkeypatch.delenv("FAKEHUB_FAIL_UPLOAD")
    # 1.002 GB free and min_free_gb=1: a fresh 3 MB copy does not fit, a resumed one (0 bytes to send) does
    tight = FakeRemote(tmp_path, hub, free_gb=1.002)
    api2, _, logs = _relay(R, tmp_path, tight)
    with pytest.raises(RuntimeError, match="still to stage.*must keep 1 GB free"):
        api2.upload_folder(str(rung), "MLX/8bit", "Org/M-Pollard")
    api2.upload_folder(str(rung), "MLX/4bit", "Org/M-Pollard")
    assert tight.sent == [] and any("0.00 GB to send" in m for m in logs)
    assert "MLX/4bit/model.safetensors" in api2.list_repo_files("Org/M-Pollard")


# ---------------------------------------------------------------------------------------------------- the sweep

def _stage_dir(root, *parts, files=None, mark=None, age_days=0):
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in (files or {}).items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    if mark is not None:
        (d / ".pollard-staging.json").write_text(json.dumps(mark))
    if age_days:
        t = time.time() - age_days * 86400
        for r, _, fs in os.walk(d):
            for f in fs:
                os.utime(os.path.join(r, f), (t, t))
        os.utime(d, (t, t))
    return d


def _publish(hub, repo, path_in_repo, files):
    for name, data in files.items():
        p = hub / repo / path_in_repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _marked(path_in_repo, files, repo="Org/M-Pollard"):
    return {"repo_id": repo, "path_in_repo": path_in_repo,
            "files": {f"{path_in_repo}/{n}": [len(d), hashlib.sha256(d).hexdigest()] for n, d in files.items()}}


def test_startup_sweep_removes_empty_and_uploaded_keeps_and_reports_the_rest(R, tmp_path):
    hub = tmp_path / "hub"
    fake = FakeRemote(tmp_path, hub)
    api, home, logs = _relay(R, tmp_path, fake, extra="&report_days=7")
    st = home / "staging"
    done = {"model.safetensors": b"\x05" * 500, "config.json": b"{}"}
    _publish(hub, "Org/M-Pollard", "MLX/4bit", done)
    up = _stage_dir(st, "Org__M-Pollard", "MLX__4bit", files=done, mark=_marked("MLX/4bit", done))
    empty = _stage_dir(st, "Org__M-Pollard", "MLX__6bit")
    partial = {"model.safetensors": b"\x06" * 300}
    recent = _stage_dir(st, "Org__M-Pollard", "MLX__3bit", files=partial, mark=_marked("MLX/3bit", partial))
    old = _stage_dir(st, "Org__Other", "MLX__2bit", files={"x.bin": b"z" * 50}, age_days=30)
    freed, rep = api.sweep(["Org/M-Pollard"])
    assert not up.exists(), "a copy already confirmed on the Hub goes"
    assert not empty.exists(), "an empty folder goes"
    assert recent.exists(), "a recent partial copy stays for resume"
    assert old.exists(), "an unknown old folder is reported, not deleted"
    acts = {os.path.basename(p): a for p, a, _ in rep}
    assert acts == {"MLX__4bit": "remove", "MLX__6bit": "remove", "MLX__3bit": "keep", "MLX__2bit": "report"}
    assert freed == 502
    assert any("KEPT" in m and "MLX__2bit" in m for m in logs)


def test_sweep_keeps_everything_when_the_hub_cannot_be_read(R, tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    api, home, _ = _relay(R, tmp_path, FakeRemote(tmp_path, hub))
    done = {"model.safetensors": b"\x05" * 500}
    _publish(hub, "Org/M-Pollard", "MLX/4bit", done)
    up = _stage_dir(home / "staging", "Org__M-Pollard", "MLX__4bit", files=done, mark=_marked("MLX/4bit", done))
    monkeypatch.setenv("FAKEHUB_DOWN", "1")
    api.sweep(["Org/M-Pollard"])
    assert up.exists(), "not proven on the Hub -> not deleted"


def test_sweep_does_not_remove_a_copy_whose_hub_file_differs(R, tmp_path):
    hub = tmp_path / "hub"
    api, home, _ = _relay(R, tmp_path, FakeRemote(tmp_path, hub))
    files = {"model.safetensors": b"\x05" * 500}
    _publish(hub, "Org/M-Pollard", "MLX/4bit", {"model.safetensors": b"\x07" * 500})   # same size, other bytes
    up = _stage_dir(home / "staging", "Org__M-Pollard", "MLX__4bit", files=files, mark=_marked("MLX/4bit", files))
    api.sweep(["Org/M-Pollard"])
    assert up.exists()


def test_legacy_mlx_staging_is_reclaimed_only_for_uploaded_or_empty(R, tmp_path):
    hub = tmp_path / "hub"
    api, home, logs = _relay(R, tmp_path, FakeRemote(tmp_path, hub))
    legacy = tmp_path / "legacy-mlx-staging"
    done = {"model.safetensors": b"\x05" * 500, "config.json": b"{}"}
    _publish(hub, "Org/M-Pollard", "MLX/4bit", done)
    gone = _stage_dir(legacy, "3fa9c2e1b0aa", "M-MLX-4bit", files=done)            # orphan of an uploaded rung
    empty = _stage_dir(legacy, "77aa00bb11cc")
    unknown = _stage_dir(legacy, "deadbeef0001", "M-MLX-2bit", files={"model.safetensors": b"\x08" * 400})
    _, rep = api.sweep(["Org/M-Pollard"], legacy_root=str(legacy))
    assert not gone.parent.exists() and not empty.exists()
    assert unknown.exists(), "a legacy copy not on the Hub is kept"
    assert any("removed" in m and "3fa9c2e1b0aa" in m for m in logs), "what was removed is logged"
    # once the last orphan is uploaded, the legacy folder itself goes
    _publish(hub, "Org/M-Pollard", "MLX/2bit", {"model.safetensors": b"\x08" * 400})
    api.sweep(["Org/M-Pollard"], legacy_root=str(legacy))
    assert not legacy.exists()
    assert any("empty legacy folder" in m for m in logs)


def test_sweep_never_deletes_outside_its_roots(tmp_path):
    import pollard_workspace as ws
    root = tmp_path / "staging"
    (root / "a" / "b").mkdir(parents=True)
    outside = tmp_path / "keep"
    outside.mkdir()
    with pytest.raises(ValueError):
        ws.remove_within(str(root), str(outside))
    with pytest.raises(ValueError):
        ws.remove_within(str(root), str(root))
    with pytest.raises(ValueError):
        ws.remove_within(str(root), str(root / ".." / "keep"))
    assert outside.exists()
    ws.remove_within(str(root), str(root / "a" / "b"))
    assert not (root / "a").exists() and root.exists()


def test_force_does_not_reach_the_legacy_folder(tmp_path, monkeypatch):
    """--force clears the workspace staging area, but the legacy folder only ever loses empty / uploaded copies."""
    import pollard_workspace as ws
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    legacy = tmp_path / "legacy"
    mine = _stage_dir(tmp_path / "home" / "staging", "Org__M", "MLX__4bit", files={"a.bin": b"1"})
    theirs = _stage_dir(legacy, "abc", "R", files={"a.bin": b"1"})
    freed, _ = ws.sweep_staging(None, [], force=True, legacy_root=str(legacy), log=lambda *a: None)
    assert not mine.exists() and theirs.exists() and freed == 1


# ------------------------------------------------------------------------------------- pollard-ls + the release

def test_pollard_ls_shows_and_cleans_staging(tmp_path, monkeypatch, capsys):
    import pollard_ls as L
    import pollard_workspace as ws
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ws, "LEGACY_STAGING", str(tmp_path / "no-legacy"))
    files = {"model.safetensors": b"\x01" * 2048}
    _stage_dir(tmp_path / "home" / "staging", "Org__M-Pollard", "MLX__4bit", files=files,
               mark=_marked("MLX/4bit", files))
    empty = _stage_dir(tmp_path / "home" / "staging", "Org__M-Pollard", "MLX__8bit")
    monkeypatch.setattr(sys, "argv", ["pollard-ls"])
    L.main()
    out = capsys.readouterr().out
    assert "staging/" in out and "Org/M-Pollard:MLX/4bit" in out and "2.0KB" in out
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: None))
    monkeypatch.setattr(sys, "argv", ["pollard-ls", "--clean-staging"])
    L.main()
    out = capsys.readouterr().out
    assert not empty.exists() and (tmp_path / "home" / "staging" / "Org__M-Pollard" / "MLX__4bit").exists()
    assert "kept 1 copy" in out
    monkeypatch.setattr(sys, "argv", ["pollard-ls", "--clean-staging", "--force"])
    L.main()
    assert not (tmp_path / "home" / "staging" / "Org__M-Pollard").exists()


def test_release_sweeps_relay_staging_before_an_uploading_run(R, tmp_path):
    hub = tmp_path / "hub"
    fake = FakeRemote(tmp_path, hub)
    api, home, logs = _relay(R, tmp_path, fake)
    files = {"model.safetensors": b"\x05" * 500}
    _publish(hub, "PollardWeights/M-Pollard", "MLX/4bit", files)
    up = _stage_dir(home / "staging", "PollardWeights__M-Pollard", "MLX__4bit", files=files,
                    mark=_marked("MLX/4bit", files, repo="PollardWeights/M-Pollard"))
    cfg = {"upload_via": "ssh://unused", "org": "PollardWeights"}
    R.sweep_staging(cfg, {"src": "Org/M", "org": "PollardWeights"}, out=logs.append, relay=api)
    assert not up.exists()
    assert any("relay staging: removed" in m for m in logs)


def test_release_sweep_never_fails_the_run(R, tmp_path):
    class Broken:
        host = "box"

        def sweep(self, repos):
            raise RuntimeError("ssh: connection refused")
    logs = []
    R.sweep_staging({"upload_via": "ssh://x"}, {"src": "Org/M", "org": "O"}, out=logs.append, relay=Broken())
    assert any("sweep skipped" in m for m in logs)


def test_model_sized_work_dirs_live_in_the_workspace(tmp_path, monkeypatch):
    """Probe GGUFs / base logits (moefit, lowbit, sensitivity, eval, kvbench) go to $POLLARD_HOME/cache, not /tmp."""
    import pollard_moefit as MF
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    assert MF._work_root() == str(tmp_path / "home" / "cache") and os.path.isdir(MF._work_root())
