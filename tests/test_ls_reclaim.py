"""pollard-ls --clean-cache / --clean-mirrors / --clean-all: reclaim everything that is safe to reclaim, inside
$POLLARD_HOME only. A fake workspace and a fake Hub (file list + size + sha256 per file) -- no network, no model."""
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_workspace as ws  # noqa: E402

REPO = "Org/M-Pollard"
OLD = time.time() - 3 * 86400


class FakeHub:
    """{repo: {path in repo: bytes}}; `down` = the Hub cannot be reached."""

    def __init__(self):
        self.repos, self.down = {}, False

    def put(self, repo, path, data):
        self.repos.setdefault(repo, {})[path] = data

    def _check(self):
        if self.down:
            raise ConnectionError("hub unreachable")

    def list_repo_files(self, repo):
        self._check()
        return sorted(self.repos.get(repo, {}))

    def get_paths_info(self, repo, paths):
        self._check()
        files = self.repos.get(repo, {})
        return [types.SimpleNamespace(path=p, size=len(files[p]), lfs={"sha256": hashlib.sha256(files[p]).hexdigest()},
                                      blob_id=None) for p in paths if p in files]


def _age(path, t=OLD):
    for r, ds, fs in os.walk(path):
        for n in ds + fs:
            os.utime(os.path.join(r, n), (t, t))
    os.utime(path, (t, t))


def _write(path, data=b"x" * 100, old=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if old:
        os.utime(path, (OLD, OLD))
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("POLLARD_HOME", str(h))
    monkeypatch.delenv("POLLARD_MACHINE_LOCK", raising=False)
    monkeypatch.setattr(ws, "held_open", lambda root: set())        # nothing open unless a test says so
    monkeypatch.setattr(ws, "LEGACY_STAGING", str(tmp_path / "no-legacy"))
    return h


def _mirror(home, name="M", files=None):
    d = home / "mirror" / name
    for rel, data in (files or {"config.json": b"{}", "tokenizer.json": b"t" * 300}).items():
        _write(d / rel, data)
    _age(d)
    return d


def _rung(home, slug, name, files):
    d = home / "models" / slug / name
    for rel, data in files.items():
        _write(d / rel, data)
    return d


def _manifest(home, slug, model, builds):
    p = home / "models" / slug / "MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model": model, "builds": builds}))


def _published_model(home, hub, publish_6bit=True):
    """mirror/M with two MLX rungs built from it: 4bit still on disk, 6bit freed after its upload (recorded
    hub_files) -- plus a GGUF rung recorded under M-f16.gguf, freed with no record (an older release)."""
    mirror = _mirror(home)
    four = {"config.json": b'{"q": 4}', "model.safetensors": b"\x04" * 4000}
    r4 = _rung(home, "M", "M-Pollard-MLX-4bit", four)
    six = b"\x06" * 6000
    builds = [
        {"name": "M-Pollard-MLX-4bit", "lane": "mlx", "tag": "MLX-4bit", "path": str(r4), "bytes": 4008,
         "uploaded": REPO, "uploaded_as": "MLX/4bit"},
        {"name": "M-Pollard-MLX-6bit", "lane": "mlx", "tag": "MLX-6bit", "path": str(home / "models/M/gone6"),
         "uploaded": REPO, "uploaded_as": "MLX/6bit", "freed": "uploaded + confirmed on the Hub",
         "hub_files": {"MLX/6bit/model.safetensors": {"size": len(six), "sha256": hashlib.sha256(six).hexdigest(),
                                                      "git_sha1": None}}},
    ]
    _manifest(home, "M", str(mirror), builds)
    _manifest(home, "M-f16.gguf", str(home / "models/M/gguf/M-f16.gguf"), [
        {"name": "M-Pollard-Q6_K.gguf", "lane": "gguf", "tag": "Q6_K", "path": str(home / "models/x/gone.gguf"),
         "uploaded": REPO, "uploaded_as": "M-Pollard-Q6_K.gguf", "freed": "uploaded + confirmed on the Hub"}])
    for rel, data in four.items():
        hub.put(REPO, f"MLX/4bit/{rel}", data)
    if publish_6bit:
        hub.put(REPO, "MLX/6bit/model.safetensors", six)
    hub.put(REPO, "M-Pollard-Q6_K.gguf", b"g" * 50)
    return mirror, r4


# ------------------------------------------------------------------------------------------------------- cache

def test_clean_cache_removes_old_entries_and_keeps_recent_or_open_ones(home, monkeypatch):
    c = home / "cache"
    ref = _write(c / "verify-ref-abc.npy", b"r" * 1000)
    kl = _write(c / "kl-base-Q6.bin", b"k" * 500)
    work = c / "pollard_sens_123"
    _write(work / "probe.gguf", b"p" * 2000)
    _age(work)
    old_tc = _write(c / "toolcall" / "old.json", b"t" * 10)
    new_tc = _write(c / "toolcall" / "new.json", b"n" * 10, old=False)             # written just now
    busy = c / "pollard_eval_9"
    held = _write(busy / "logits.bin", b"l" * 700)
    _age(busy)
    monkeypatch.setattr(ws, "held_open", lambda root: {os.path.realpath(held)})
    logs = []
    freed, plan = ws.clean_cache(log=logs.append)
    assert not ref.exists() and not kl.exists() and not work.exists() and not old_tc.exists()
    assert new_tc.exists(), "an entry written in the last 30 min may belong to a running job"
    assert held.exists(), "an entry with a file open in some process is kept"
    assert freed == 1000 + 500 + 2000 + 10
    assert any("KEPT" in m and "open" in m for m in logs) and any("KEPT" in m and "min ago" in m for m in logs)
    assert any("removed" in m and "verify-ref-abc.npy" in m for m in logs)


def test_clean_cache_without_open_file_info_says_it_falls_back_to_age(home, monkeypatch):
    monkeypatch.setattr(ws, "held_open", lambda root: None)
    _write(home / "cache" / "a.bin")
    logs = []
    freed, _ = ws.clean_cache(min_age_min=45, log=logs.append)
    assert freed == 100 and any("cannot list open files" in m and "45 min" in m for m in logs)


def test_clean_cache_names_a_running_pollard_job(home):
    (home / ".pollard.lock").write_text(json.dumps({"pid": os.getppid(), "what": "pollard-mlx M"}))
    _write(home / "cache" / "a.bin")
    logs = []
    ws.clean_cache(log=logs.append)
    assert any("a Pollard job is running" in m and "pollard-mlx M" in m for m in logs)


def test_held_open_sees_a_file_this_process_has_open(tmp_path):
    f = tmp_path / "d" / "open.bin"
    _write(f)
    with open(f, "rb"):
        got = ws.held_open(str(tmp_path / "d"))
    if got is None:
        pytest.skip("no /proc and no lsof here")
    assert os.path.realpath(f) in got


# ------------------------------------------------------------------------------------------------------ mirrors

def test_mirror_goes_only_when_every_build_is_confirmed_on_the_hub(home):
    hub = FakeHub()
    mirror, r4 = _published_model(home, hub)
    logs = []
    freed, plan = ws.clean_mirrors(hub, log=logs.append)
    assert not mirror.exists() and freed == 302
    assert r4.exists() and (home / "models" / "M" / "MANIFEST.json").exists(), "models/ is never touched"
    assert plan[0][1] == "remove" and "every build (3)" in plan[0][2]


def test_mirror_kept_when_one_rung_is_missing_from_the_hub_and_force_removes_it(home):
    hub = FakeHub()
    mirror, _ = _published_model(home, hub, publish_6bit=False)
    logs = []
    freed, plan = ws.clean_mirrors(hub, log=logs.append)
    assert mirror.exists() and freed == 0
    assert "M-Pollard-MLX-6bit" in plan[0][2] and "not on the Hub" in plan[0][2]
    freed, plan = ws.clean_mirrors(hub, force=True, log=logs.append)
    assert not mirror.exists() and freed == 302
    assert any("--force" in m and "NOT confirmed" in m for m in logs), "a forced removal is said out loud"


def test_mirror_kept_when_a_rung_differs_on_the_hub(home):
    hub = FakeHub()
    mirror, _ = _published_model(home, hub)
    hub.put(REPO, "MLX/4bit/model.safetensors", b"\x09" * 4000)              # same size, other bytes
    ws.clean_mirrors(hub, log=lambda *a: None)
    assert mirror.exists()


def test_mirror_kept_when_the_hub_cannot_be_reached(home):
    hub = FakeHub()
    mirror, _ = _published_model(home, hub)
    hub.down = True
    _, plan = ws.clean_mirrors(hub, log=lambda *a: None)
    assert mirror.exists() and "could not" in plan[0][2]
    _, plan = ws.clean_mirrors(None, log=lambda *a: None)                    # huggingface_hub not installed
    assert mirror.exists()


def test_mirror_with_no_recorded_build_is_kept(home):
    mirror = _mirror(home, "Building")
    _, plan = ws.clean_mirrors(FakeHub(), log=lambda *a: None)
    assert mirror.exists() and "no build recorded" in plan[0][2]


def test_mirror_in_use_is_kept_even_with_force(home, monkeypatch):
    hub = FakeHub()
    mirror, _ = _published_model(home, hub)
    _write(mirror / "model-00001.safetensors", b"s" * 10, old=False)          # the mirror fetched a tensor just now
    ws.clean_mirrors(hub, force=True, log=lambda *a: None)
    assert mirror.exists()
    _age(mirror)
    monkeypatch.setattr(ws, "held_open", lambda root: {os.path.realpath(mirror / "config.json")})
    ws.clean_mirrors(hub, force=True, log=lambda *a: None)
    assert mirror.exists()


def test_repo_names_the_hub_repo_when_the_manifest_has_none(home):
    hub = FakeHub()
    mirror = _mirror(home)
    four = {"config.json": b"{}", "model.safetensors": b"\x04" * 400}
    r4 = _rung(home, "M", "M-Pollard-MLX-4bit", four)
    _manifest(home, "M", str(mirror), [{"name": "M-Pollard-MLX-4bit", "lane": "mlx", "tag": "MLX-4bit",
                                        "path": str(r4)}])
    for rel, data in four.items():
        hub.put(REPO, f"MLX/4bit/{rel}", data)
    _, plan = ws.clean_mirrors(hub, log=lambda *a: None)
    assert mirror.exists() and "--repo" in plan[0][2]
    ws.clean_mirrors(hub, repos=[REPO], log=lambda *a: None)
    assert not mirror.exists() and r4.exists()


def test_a_symlinked_mirror_is_unlinked_never_followed(home, tmp_path):
    outside = tmp_path / "elsewhere"
    _write(outside / "weights.bin", b"w" * 999)
    (home / "mirror").mkdir()
    link = home / "mirror" / "Linked"
    os.symlink(outside, link)
    ws.clean_mirrors(FakeHub(), force=True, min_age_min=0, log=lambda *a: None)
    assert not os.path.lexists(link) and (outside / "weights.bin").exists(), "only the link goes"


def test_remove_within_refuses_anything_outside_its_root(tmp_path):
    root = tmp_path / "cache"
    (root / "sub").mkdir(parents=True)
    f = _write(root / "sub" / "f.bin")
    outside = _write(tmp_path / "keep.bin")
    for bad in (root, root / ".." / "keep.bin", outside, tmp_path):
        with pytest.raises(ValueError):
            ws.remove_within(str(root), str(bad))
    os.symlink(outside, root / "out-link")
    ws.remove_within(str(root), str(root / "out-link"))
    assert outside.exists() and not os.path.lexists(root / "out-link")
    ws.remove_within(str(root), str(f))
    assert not (root / "sub").exists() and root.exists(), "a file goes, then the folders it leaves empty"


def test_downloads_confirmed_mode_keeps_unpublished_sources(home):
    hub = FakeHub()
    dl = home / "downloads"
    pub = dl / "Org__Pub"
    _write(pub / "model.safetensors", b"m" * 800)
    _age(pub)
    unpub = dl / "Org__Unpub"
    _write(unpub / "model.safetensors", b"u" * 800)
    _age(unpub)
    _write(dl / ".hf-cache" / "x", b"c" * 5)
    _age(dl / ".hf-cache")
    four = {"config.json": b"{}"}
    r = _rung(home, "Org__Pub", "Pub-Pollard-MLX-4bit", four)
    _manifest(home, "Org__Pub", "Org/Pub", [{"name": "Pub-Pollard-MLX-4bit", "lane": "mlx", "path": str(r),
                                             "uploaded": "Org/Pub-Pollard", "uploaded_as": "MLX/4bit"}])
    hub.put("Org/Pub-Pollard", "MLX/4bit/config.json", b"{}")
    freed, _ = ws.clean_downloads_confirmed(hub, log=lambda *a: None)
    assert not pub.exists() and unpub.exists() and (dl / ".hf-cache").exists() and freed == 800
    freed, _ = ws.clean_downloads_confirmed(hub, force=True, log=lambda *a: None)
    assert not unpub.exists() and not (dl / ".hf-cache").exists()


# ------------------------------------------------------------------------------------------------- the CLI

def _run(monkeypatch, capsys, *argv, hub=None):
    import pollard_ls as L
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda: hub))
    monkeypatch.setattr(sys, "argv", ["pollard-ls", *argv])
    L.main()
    return capsys.readouterr().out


def test_pollard_ls_shows_mirrors_and_how_to_reclaim_them(home, monkeypatch, capsys):
    hub = FakeHub()
    _published_model(home, hub)
    _mirror(home, "Building")
    _write(home / "cache" / "a.bin")
    out = _run(monkeypatch, capsys, hub=hub)
    assert "mirror/" in out
    mirrors = out.split("mirror/ (", 1)[1].splitlines()
    line = next(ln for ln in mirrors if ln.strip().startswith("M "))
    assert "reclaimable" in line and REPO in line
    assert "kept: no build recorded" in next(ln for ln in mirrors if "Building" in ln)
    assert "pollard-ls --clean-mirrors" in out and "pollard-ls --clean-cache" in out
    assert (home / "mirror" / "M").exists(), "a plain listing removes nothing"


def test_clean_flags_and_clean_all_summary(home, monkeypatch, capsys):
    hub = FakeHub()
    mirror, r4 = _published_model(home, hub)
    _write(home / "cache" / "verify-ref-1.npy", b"r" * 2048)
    out = _run(monkeypatch, capsys, "--clean-cache", hub=hub)
    assert "cleaned cache/ -- freed 2.0KB" in out and not (home / "cache" / "verify-ref-1.npy").exists()
    _write(home / "cache" / "verify-ref-2.npy", b"r" * 2048)
    keep = _mirror(home, "Unpub")
    _write(home / "downloads" / "Org__Unpub" / "w.bin", b"w" * 10)
    _age(home / "downloads")
    out = _run(monkeypatch, capsys, "--clean-all", hub=hub)
    assert "reclaimed in" in out
    rows = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip().split()[:1] and
            ln.strip().split()[0] in ("downloads/", "staging/", "cache/", "mirror/", "total")}
    assert set(rows) == {"downloads/", "staging/", "cache/", "mirror/", "total"}
    assert "2.0KB" in rows["cache/"] and "302B" in rows["mirror/"]
    assert not mirror.exists() and keep.exists() and (home / "downloads" / "Org__Unpub").exists()
    assert r4.exists(), "models/ is never touched"
    _run(monkeypatch, capsys, "--clean-all", "--force", hub=hub)
    assert not keep.exists() and not (home / "downloads" / "Org__Unpub").exists() and r4.exists()


def test_force_alone_is_refused(home, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run(monkeypatch, capsys, "--force")


# ------------------------------------------------------------------------------------------ the release hook

def test_release_reclaims_its_mirror_once_public(home):
    import pollard_release as R
    hub = FakeHub()
    mirror, _ = _published_model(home, hub)
    c = {"hf_dir": str(mirror), "smooth_dir": str(mirror) + "-smoothed"}
    logs = []
    assert R.reclaim_after_public(c, {"reclaim_mirror": False}, REPO, hub, logs.append) == 0 and mirror.exists()
    hub.down = True
    assert R.reclaim_after_public(c, {}, REPO, hub, logs.append) == 0 and mirror.exists()
    hub.down = False
    _write(mirror / "fresh.json", b"{}", old=False)          # its own steps just finished: no age rule here
    assert R.reclaim_after_public(c, {}, REPO, hub, logs.append) > 0 and not mirror.exists()
    assert any("mirror: freed" in m for m in logs)


def test_release_never_reclaims_a_source_outside_the_mirror_root(home, tmp_path):
    import pollard_release as R
    hub = FakeHub()
    _published_model(home, hub)
    src = tmp_path / "my-model"
    _write(src / "config.json")
    assert R.reclaim_after_public({"hf_dir": str(src)}, {}, REPO, hub, print) == 0 and src.exists()
