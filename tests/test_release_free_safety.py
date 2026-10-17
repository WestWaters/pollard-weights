"""--free-after-upload deletes a local rung ONLY after the Hub holds that exact file (size + sha256 read back).

The LFM2.5-24B-A2B release (2026-10-16) built Q6_K / IQ4_XS / IQ3_S for five hours, pollard-taskeval failed on each
(its llama-server died), verify was skipped -- and the publish step "freed" all three, unverified and never uploaded.
These tests pin the opposite: a failed / skipped measure keeps the file, an upload error keeps the file, a file the
Hub does not hold intact keeps the file, and only a verified rung confirmed on the Hub is deleted. Fakes only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402
from test_release_free import ALL, hub, inputs, model, plan  # noqa: E402,F401 -- shared fixtures

REPO = "PollardWeights/Tiny-1B-Pollard"


def _ladder(c, monkeypatch, fail=None):
    """run_cmd for a full free-mode GGUF run: builds each rung, `fail` = {rung: step} makes that step fail."""
    fail = fail or {}
    rdir = ws.model_dir(c["f16"], create=True)
    built = {}

    def fake(cmd, ok=(0,)):
        tool = cmd[0]
        if tool == "pollard":
            label = cmd[cmd.index("--rung") + 1]
            if label in ("IQ2", "flagship"):
                return
            p = os.path.join(rdir, f"Tiny-1B-f16.gguf-Pollard-GGUF-{label}.gguf")
            open(p, "wb").write(b"GGUF" + label.encode() * 50)
            ws.record_build(c["f16"], "gguf", p, tag=label)
            built[label] = p
            return
        rung = cmd[cmd.index("--rung") + 1] if "--rung" in cmd else None
        if tool == "pollard-taskeval" and fail.get(rung) == "taskeval":
            raise RuntimeError("exit 1")                  # the LFM2.5 failure: the measure never produced a number
        if tool == "pollard-verify":
            m = cmd[cmd.index("--model") + 1]
            ok_gate = fail.get(rung) != "verify"
            ws.record_gate(m, {"pass": ok_gate})
            if not ok_gate:
                raise RuntimeError("exit 1")
        if tool == "pollard-card":
            out = cmd[cmd.index("--out") + 1]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "w").write("card")
    monkeypatch.setattr(R, "run_cmd", fake)
    monkeypatch.setattr(R, "disk_free_gb", lambda p: 1000.0)
    return built


def _paths(c):
    return {b["tag"]: b["path"] for b in R.gguf_builds(c)}


def test_a_failed_measure_keeps_the_rung_on_disk_and_off_the_hub(model, inputs, hub, monkeypatch):
    cfg = plan(inputs, measure={"taskeval": {"suite": "quick", "limit": 5}, "toolcall": False}, _taskeval=True)
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    _ladder(c, monkeypatch, fail={"Q6_K": "taskeval", "IQ4_XS": "taskeval", "IQ3_S": "taskeval"})
    said = []
    st = R.execute(steps, c, cfg, run=True, run_upload=True, have=ALL, out=said.append)
    for tag in ("Q6_K", "IQ4_XS", "IQ3_S"):
        assert st[f"gguf/{tag}/taskeval"] == "failed"
        assert st[f"gguf/{tag}/verify"] == "skipped: upstream failed"
        assert st[f"gguf/{tag}/publish"] == "kept: upstream failed"
        assert os.path.exists(_paths(c)[tag]), f"{tag}: unverified and never uploaded -- must NOT be deleted"
        assert f"Tiny-1B-Pollard-{tag}.gguf" not in hub.get(REPO, set())
    assert any("KEPT on disk, NOT published" in x for x in said), "the kept rungs are reported at the end"


def test_a_rung_that_is_not_verified_is_kept_by_publish_rung_itself(model, inputs, hub):
    """Even called directly (a resumed run, a stale state file), publish_rung never deletes an unverified rung."""
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    p = os.path.join(ws.model_dir(c["f16"], create=True), "Tiny-1B-Pollard-Q6_K.gguf")
    Path(p).write_bytes(b"GGUF" * 10)
    ws.record_build(c["f16"], "gguf", p, tag="Q6_K")            # verified: None -- what the LFM2.5 MANIFEST held
    res = R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", p), c, cfg, True, out=lambda *_: None)
    assert res.startswith("kept") and os.path.exists(p)
    b = R.gguf_builds(c)[0]
    assert "not verified" in b["kept"] and not b.get("freed") and not b.get("uploaded")
    assert REPO not in hub or "Tiny-1B-Pollard-Q6_K.gguf" not in hub[REPO]


def test_an_upload_error_keeps_the_rung(model, inputs, hub, monkeypatch):
    cfg = plan(inputs)
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    _ladder(c, monkeypatch)
    hub["__fail_upload__"] = True
    st = R.execute(steps, c, cfg, run=True, run_upload=True, have=ALL, out=lambda *_: None)
    for tag in ("Q6_K", "IQ4_XS", "IQ3_S"):
        assert st[f"gguf/{tag}/verify"] == "done" and st[f"gguf/{tag}/publish"] == "failed"
        assert os.path.exists(_paths(c)[tag]), f"{tag}: the upload raised -- the local copy is all there is"
        assert not R.gguf_builds(c)[0].get("uploaded")


def test_a_file_the_hub_does_not_hold_intact_is_kept(model, inputs, hub, monkeypatch):
    """The upload 'succeeded' but the Hub's copy is short / different: size + sha256 say so, nothing is deleted."""
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    p = os.path.join(ws.model_dir(c["f16"], create=True), "Tiny-1B-Pollard-Q6_K.gguf")
    Path(p).write_bytes(b"GGUF" * 100)
    ws.record_build(c["f16"], "gguf", p, tag="Q6_K")
    ws.record_gate(p, {"pass": True})
    import test_release_free as T
    orig = T.FakeApi._put

    def truncated(self, repo, path_in_repo, local):
        orig(self, repo, path_in_repo, local)
        size, sha = self.meta[(repo, path_in_repo)]
        self.meta[(repo, path_in_repo)] = (size - 7, sha)         # an interrupted transfer
    monkeypatch.setattr(T.FakeApi, "_put", truncated)
    with pytest.raises(RuntimeError, match="keeping the local copy"):
        R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", p), c, cfg, True, out=lambda *_: None)
    assert os.path.exists(p)
    b = R.gguf_builds(c)[0]
    assert not b.get("uploaded") and "not confirmed" in b["kept"]

    monkeypatch.setattr(T.FakeApi, "_put", orig)                    # same bytes, wrong hash
    def wrong_hash(self, repo, path_in_repo, local):
        orig(self, repo, path_in_repo, local)
        size, _ = self.meta[(repo, path_in_repo)]
        self.meta[(repo, path_in_repo)] = (size, "0" * 64)
    monkeypatch.setattr(T.FakeApi, "_put", wrong_hash)
    with pytest.raises(RuntimeError):
        R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", p), c, cfg, True, out=lambda *_: None)
    assert os.path.exists(p)


def test_a_hub_that_cannot_be_asked_frees_nothing(model, inputs, hub, monkeypatch):
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    p = os.path.join(ws.model_dir(c["f16"], create=True), "Tiny-1B-Pollard-Q6_K.gguf")
    Path(p).write_bytes(b"GGUF" * 10)
    ws.record_build(c["f16"], "gguf", p, tag="Q6_K")
    ws.record_gate(p, {"pass": True})
    import test_release_free as T
    monkeypatch.delattr(T.FakeApi, "get_paths_info")
    with pytest.raises(RuntimeError):
        R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", p), c, cfg, True, out=lambda *_: None)
    assert os.path.exists(p)


def test_a_verified_rung_confirmed_on_the_hub_is_deleted(model, inputs, hub, monkeypatch):
    cfg = plan(inputs)
    c, _, steps = R.plan_steps(cfg, model, {"gguf"})
    built = _ladder(c, monkeypatch, fail={"IQ3_S": "verify"})
    st = R.execute(steps, c, cfg, run=True, run_upload=True, have=ALL, out=lambda *_: None)
    for tag in ("Q6_K", "IQ4_XS"):
        assert st[f"gguf/{tag}/publish"] == "done"
        assert not os.path.exists(built[tag]) and not os.path.exists(_paths(c)[tag]), f"{tag}: on the Hub, freed"
        assert f"Tiny-1B-Pollard-{tag}.gguf" in hub[REPO]
    b = {x["tag"]: x for x in R.gguf_builds(c)}
    assert b["Q6_K"]["uploaded"] == REPO and "confirmed" in b["Q6_K"]["freed"]
    assert os.path.exists(_paths(c)["IQ3_S"]), "the rung that failed its gate stays"


def test_remote_mismatches_checks_size_and_hash(tmp_path):
    import types
    f = tmp_path / "a.gguf"
    f.write_bytes(b"x" * 33)
    sha = R._sha256(str(f))

    class Api:
        def __init__(self, rows):
            self.rows = rows

        def get_paths_info(self, repo, paths):
            return [types.SimpleNamespace(**r) for r in self.rows if r["path"] in paths]
    ok = Api([{"path": "a.gguf", "size": 33, "lfs": {"sha256": sha}, "blob_id": None}])
    assert R.remote_mismatches(ok, "r", [("a.gguf", str(f))]) == []
    assert R.remote_mismatches(Api([]), "r", [("a.gguf", str(f))])[0].endswith("not on the Hub")
    short = Api([{"path": "a.gguf", "size": 30, "lfs": {"sha256": sha}, "blob_id": None}])
    assert "bytes on the Hub" in R.remote_mismatches(short, "r", [("a.gguf", str(f))])[0]
    small = Api([{"path": "a.gguf", "size": 33, "lfs": None, "blob_id": R._git_blob_sha1(str(f))}])
    assert R.remote_mismatches(small, "r", [("a.gguf", str(f))]) == [], "a non-LFS file is checked by its git blob id"
    nohash = Api([{"path": "a.gguf", "size": 33, "lfs": None, "blob_id": None}])
    assert R.remote_mismatches(nohash, "r", [("a.gguf", str(f))]), "no hash to compare = not confirmed"


def test_git_blob_id_matches_git(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello\n")
    assert R._git_blob_sha1(str(f)) == "ce013625030ba8dba906f756967f9e9ca394464a"   # `git hash-object` of "hello\n"


def test_a_rerun_rung_replaces_the_stale_entry_at_its_published_name(model, inputs, hub, monkeypatch):
    """The LFM2.5 MANIFEST after the bad night: Q6_K 'freed' at its published name. A rebuilt Q6_K renamed onto that
    name must be the entry publish_rung finds -- not the stale, unverified one."""
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    rdir = ws.model_dir(c["f16"], create=True)
    canon = os.path.join(rdir, "Tiny-1B-Pollard-Q6_K.gguf")
    ws.record_build(c["f16"], "gguf", canon, tag="Q6_K", extra={"freed": "did not pass the gate -- not published"})
    fresh = os.path.join(rdir, "Tiny-1B-f16.gguf-Pollard-GGUF-Q6_K.gguf")
    Path(fresh).write_bytes(b"GGUF" * 9)
    ws.record_build(c["f16"], "gguf", fresh, tag="Q6_K")
    R.canonical_gguf(c)
    builds = R.gguf_builds(c)
    assert len(builds) == 1 and builds[0]["path"] == os.path.abspath(canon) and not builds[0].get("freed")
    ws.record_gate(canon, {"pass": True})
    assert R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", canon), c, cfg, True, out=lambda *_: None) == "done"
    assert "Tiny-1B-Pollard-Q6_K.gguf" in hub[REPO] and not os.path.exists(canon)


def test_measure_ref_from_the_plan_is_the_served_reference_and_verify_stays_on_the_f16(model, inputs, tmp_path):
    q8 = tmp_path / "Tiny-1B-Q8_0.gguf"
    q8.write_bytes(b"GGUF")
    cfg = plan(inputs, measure={"taskeval": {"suite": "quick", "limit": 5}, "toolcall": True},
               measure_ref=str(q8), _taskeval=True)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    steps = {s.id.split("/")[-1]: s for s in R.gguf_rung_steps(cfg, c, [{"tag": "Q6_K", "path": "r.gguf"}])}
    ref = lambda s: s.cmd[s.cmd.index("--ref") + 1]
    assert ref(steps["taskeval"]) == str(q8) and ref(steps["toolcall"]) == str(q8)
    assert ref(steps["verify"]) == c["f16"], "the KLD gate is always against the f16"



def test_taskeval_is_opt_in_and_the_release_path_is_the_fast_measure_set(model, inputs):
    """Owner (2026-10-16): Pollard publishes per rung on KLD + PPL vs the f16, the tool-call gate and the complete-model
    gate -- no lm-eval per rung (it took ~1 h a rung). --taskeval opts in; --taskeval-only measures published rungs later."""
    cfg = plan(inputs, measure={"taskeval": {"suite": "quick", "limit": 200}, "toolcall": True})
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    ids = [s.id.split("/")[-1] for s in R.gguf_rung_steps(cfg, c, [{"tag": "Q6_K", "path": "r.gguf"}])]
    assert "taskeval" not in ids and {"toolcall", "verify", "publish"} <= set(ids)
    cfg["_taskeval"] = True
    ids = [s.id.split("/")[-1] for s in R.gguf_rung_steps(cfg, c, [{"tag": "Q6_K", "path": "r.gguf"}])]
    assert "taskeval" in ids


def test_taskeval_only_measures_published_rungs_from_a_temporary_download(model, inputs, hub, monkeypatch, tmp_path):
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    p = os.path.join(ws.model_dir(c["f16"], create=True), "Tiny-1B-Pollard-Q6_K.gguf")
    ws.record_build(c["f16"], "gguf", p, tag="Q6_K", extra={"uploaded": REPO, "uploaded_as": "Tiny-1B-Pollard-Q6_K.gguf"})
    ws.record_gate(p, {"pass": True})
    steps = R.taskeval_only_steps(cfg, c)
    assert [s.id for s in steps] == ["gguf/Q6_K/taskeval", "gguf/card", "gguf/card-upload"]
    got = {}
    dl = tmp_path / "dl.gguf"
    import types
    fake = sys.modules["huggingface_hub"]
    def hf_hub_download(repo, name, local_dir=None):
        dl.write_bytes(b"GGUF")
        got["fetched"] = (repo, name)
        return str(dl)
    monkeypatch.setattr(fake, "hf_hub_download", hf_hub_download, raising=False)
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,): got.setdefault("model", cmd[cmd.index("--model") + 1]))
    assert R.taskeval_rung(steps[0], c, True, out=lambda *_: None) == "done"
    assert got["fetched"] == (REPO, "Tiny-1B-Pollard-Q6_K.gguf") and got["model"] == str(dl)
    assert not dl.exists(), "the temporary download is deleted after the measurement"
