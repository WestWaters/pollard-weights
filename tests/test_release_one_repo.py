"""ONE Hugging Face repo per model (owner, 2026-10-16): PollardWeights/<M>-Pollard holds every lane -- GGUF rungs and
the mmproj at the root, MLX/ GPTQ/ MX/ EXL3/ EXLR8/ in their own folders with a subfolder per rung, the drafter in
DRAFT/ -- under ONE card. The repo is created PUBLIC (owner, 2026-10-25; `private: true` keeps it private), the card
goes up before the rungs, and the last step checks every shipped lane's file list. Fakes only: no model, no Hub."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402

ALL = {"any": True, "apple": True, "cuda": True}
REPO = "PollardWeights/Tiny-1B-Pollard"


def _meta(path):
    import hashlib
    return os.path.getsize(path), hashlib.sha256(open(path, "rb").read()).hexdigest()


class Hub:
    def __init__(self):
        self.files, self.private, self.calls, self.meta = {}, {}, [], {}

    def api(self):
        hub = self

        class Api:
            def create_repo(self, repo, repo_type="model", exist_ok=True, private=False):
                hub.files.setdefault(repo, set())
                hub.private.setdefault(repo, private)

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
                hub.calls.append(("file", path_in_repo))
                hub.files.setdefault(repo_id, set()).add(path_in_repo)
                hub.meta[(repo_id, path_in_repo)] = _meta(path_or_fileobj)

            def upload_folder(self, folder_path, repo_id, path_in_repo=""):
                hub.calls.append(("folder", path_in_repo))
                for f in os.listdir(folder_path):
                    rp = f"{path_in_repo}/{f}" if path_in_repo else f
                    hub.files.setdefault(repo_id, set()).add(rp)
                    hub.meta[(repo_id, rp)] = _meta(os.path.join(folder_path, f))

            def get_paths_info(self, repo, paths):
                return [types.SimpleNamespace(path=p, size=hub.meta[(repo, p)][0], lfs={"sha256": hub.meta[(repo, p)][1]},
                                              blob_id=None) for p in paths if (repo, p) in hub.meta]

            def list_repo_files(self, repo):
                return sorted(hub.files.get(repo, set()))

            def update_repo_settings(self, repo_id, private=None):
                hub.private[repo_id] = private
        return Api()


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = h.api
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return h


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


def test_every_lane_goes_to_the_one_repo_and_shares_one_card(model):
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"]["enabled"] = True
    c, lanes, steps = R.plan_steps(cfg, model)
    repos = {s.repo for s in steps if s.repo}
    assert repos == {REPO}, "no per-lane -MLX / -GPTQ / -DRAFT repos"
    cards = {tuple(s.cmd) for s in steps if s.kind == "card"}
    assert len(cards) == 1, "every lane regenerates the SAME combined card"
    card = list(next(iter(cards)))
    assert card[card.index("--lane") + 1] == "gguf" and card[card.index("--out") + 1].endswith("README.md")
    also = [card[i + 1] for i, x in enumerate(card) if x == "--also"]
    assert also == ["mlx", "gptq", "mx", "exl3", "exlr8", "draft"]
    assert steps[-1].id == "release/public" and steps[-1].lanes == lanes, "the file-list check after every lane"


def _dir_build(c, lane, tag, sub):
    out = ws.resolve_out(c["src"], lane, tag, create_parent=True)
    os.makedirs(out, exist_ok=True)
    Path(out, "config.json").write_text("{}")
    ws.record_build(R.lane_key(c, lane), lane, out, tag=f"{lane.upper()}-{tag}")
    ws.record_gate(out, {"pass": True})
    return out


def test_directory_lanes_upload_into_their_folder_never_the_root(model, hub):
    cfg = R.load_plan(None)
    c = R._ctx(cfg, model)
    out = _dir_build(c, "mlx", "4bit", "4bit")
    os.makedirs(c["release_dir"], exist_ok=True)
    Path(c["release_dir"], "README.md").write_text("card")
    st = R.Step(id="mlx/upload", lane="mlx", kind="upload", repo=REPO, out=c["release_dir"])
    assert R.upload(st, c, cfg, True) == "uploaded"
    sub = C.rung_subfolder(ws.find_build(out)[1], "mlx")
    assert ("folder", f"MLX/{sub}") in hub.calls and ("folder", "") not in hub.calls, "the root is the GGUF lane's"
    assert f"MLX/{sub}/config.json" in hub.files[REPO] and "README.md" in hub.files[REPO]
    assert hub.private[REPO] is False, "created public"
    assert hub.calls[0] == ("file", "README.md"), "the card goes up before the rungs"


def test_free_mode_publishes_a_directory_rung_into_its_folder(model, hub):
    cfg = R.load_plan(None)
    cfg.update({"upload": True, "_free_after_upload": True})
    c = R._ctx(cfg, model)
    out = _dir_build(c, "mx", "NVFP4", "nvfp4")
    st = R._publish_rung_step(c, "mx", "NVFP4", out)
    assert st.repo == REPO
    assert R.publish_rung(st, c, cfg, True, out=lambda *_: None) == "done"
    b = ws.find_build(out)[1]
    assert b["uploaded_as"].startswith("MX/") and f"{b['uploaded_as']}/config.json" in hub.files[REPO]
    assert not os.path.exists(out), "freed once it landed"


def test_free_mode_gguf_rung_records_its_runtime_before_it_is_freed(model, hub, monkeypatch):
    cfg = R.load_plan(None)
    cfg.update({"upload": True, "_free_after_upload": True})
    c = R._ctx(cfg, model)
    rdir = ws.model_dir(c["f16"], create=True)
    p = os.path.join(rdir, "Tiny-1B-Pollard-Q6_K.gguf")
    Path(p).write_bytes(b"GGUF")
    ws.record_build(c["f16"], "gguf", p, tag="Q6_K")
    ws.record_gate(p, {"pass": True})
    import pollard_ggufcompat as gc
    monkeypatch.setattr(gc, "runtime_of", lambda path: ("ik_llama", {}))
    st = R._publish_rung_step(c, "gguf", "Q6_K", p)
    assert R.publish_rung(st, c, cfg, True, out=lambda *_: None) == "done"
    b = R.gguf_builds(c)[0]
    assert b["runtime"] == "ik_llama" and b["uploaded"] == REPO and not os.path.exists(p)
    # the card reads the recorded runtime for a rung whose file is gone
    assert C.build_runtimes([b]) == {b["path"]: "ik_llama"}


def _public(c, cfg, lanes, statuses, hub):
    st = R.public_step(cfg, c, lanes)
    return R.go_public(st, c, cfg, True, statuses, out=lambda *_: None)


def test_last_step_checks_the_file_list_and_never_touches_visibility(model, hub):
    cfg = R.load_plan(None)
    c = R._ctx(cfg, model)
    out = _dir_build(c, "mlx", "4bit", "4bit")
    sub = C.rung_subfolder(ws.find_build(out)[1], "mlx")
    hub.files[REPO] = {"README.md"}
    hub.private[REPO] = False                                  # public since its first upload
    with pytest.raises(RuntimeError, match="is missing"):
        _public(c, cfg, ["mlx"], {"mlx/upload": "done"}, hub)
    hub.files[REPO].add(f"MLX/{sub}/config.json")
    with pytest.raises(RuntimeError, match="gptq upload failed"):
        _public(c, cfg, ["mlx", "gptq"], {"mlx/upload": "done", "gptq/upload": "failed"}, hub)
    # a lane that built nothing (held, failed build, another machine's lane) ships nothing and does not block
    assert _public(c, cfg, ["mlx", "gptq", "mx"], {"mlx/upload": "done", "gptq/upload": "skipped: upstream failed",
                                                    "mx/upload": "skipped: needs cuda"}, hub) == "done"
    assert hub.private[REPO] is False
    # a no-op on visibility: a repo kept private (private: true) is still private after it
    hub.private[REPO] = True
    assert _public(c, dict(cfg, private=True), ["mlx"], {"mlx/upload": "done"}, hub) == "done"
    assert hub.private[REPO] is True


def test_nothing_shipped_or_uploads_off_leaves_the_repo_alone(model, hub):
    cfg = R.load_plan(None)
    c = R._ctx(cfg, model)
    st = R.public_step(cfg, c, ["gguf"])
    assert R.go_public(st, c, cfg, False, {"gguf/upload": "done"}, out=lambda *_: None) == "held"
    assert R.go_public(st, c, cfg, True, {"gguf/upload": "skipped: upstream failed"}, out=lambda *_: None) == "held"
    assert hub.private == {}


def test_combined_card_has_a_folder_section_per_lane_with_download_and_serve(tmp_path, monkeypatch):
    home = tmp_path / "home"
    src = tmp_path / "Org__M"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024,
                                                 "num_hidden_layers": 4, "vocab_size": 1000}))
    (src / "README.md").write_text("---\nlicense: mit\n---\n")
    monkeypatch.setenv("POLLARD_HOME", str(home))
    d = home / "models" / "Org__M"
    d.mkdir(parents=True)
    gg = [{"name": "M-Pollard-Q6_K.gguf", "lane": "gguf", "tag": "Q6_K", "path": str(d / "M-Pollard-Q6_K.gguf"),
           "bytes": 9e8, "gate": {"pass": True}}]
    (d / "MANIFEST.json").write_text(json.dumps({"model": "Org/M", "builds": gg}))
    mlx_key = tmp_path / "hfdir"
    mlx_key.mkdir()
    md = home / "models" / "hfdir"
    md.mkdir(parents=True)
    mb = [{"name": "hfdir-Pollard-MLX-4bit", "lane": "mlx", "tag": "MLX-4bit", "bytes": 5e8, "bpw": 4.5,
           "path": str(md / "hfdir-Pollard-MLX-4bit"), "gate": {"pass": True}},
          {"name": "hfdir-Pollard-MLX-3bit", "lane": "mlx", "tag": "MLX-3bit", "bytes": 4e8,
           "path": str(md / "hfdir-Pollard-MLX-3bit"), "gate": {"pass": False}}]
    (md / "MANIFEST.json").write_text(json.dumps({"model": str(mlx_key), "builds": mb}))
    monkeypatch.setattr(C, "load_builds", lambda key, lane=None: [b for b in (gg if "Org" in str(key) else mb)
                                                                   if not lane or b["lane"] == lane])
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"M-Pollard-Q6_K.gguf": {"ppl": 10.1}}))
    mres = tmp_path / "results-mlx.json"
    mres.write_text(json.dumps({"hfdir-Pollard-MLX-4bit": {"ppl": 10.4, "kld": 0.021,
                                                           "toolcall": {"pass": True, "rate": 0.97}}}))
    out = tmp_path / "README.md"
    monkeypatch.setattr(sys, "argv", ["pollard-card", "--model", str(src), "--builds-from", "Org/M", "--lane", "gguf",
                                      "--repo", "PollardWeights/M-Pollard", "--results", str(res), "--out", str(out),
                                      "--verified-only", "--also", "mlx", str(mlx_key), str(mres)])
    C.main()
    text = out.read_text()
    assert "## MLX (Apple Silicon) -- `MLX/`" in text and "## Every format in this repo" in text
    assert "| `MLX/4bit/` |" in text and "MLX/3bit" not in text, "verified-only applies to every lane"
    assert "10.4" in text and "yes (97%)" in text, "the lane's measured numbers"
    assert 'hf download PollardWeights/M-Pollard --include "MLX/4bit/*" --local-dir ./M-Pollard' in text
    assert "mlx_lm.server --model ./M-Pollard/MLX/4bit" in text
    assert "- mlx" in text.split("---")[1] and "library_name: gguf" in text, "tags of every lane, GGUF library"
    assert "M-Pollard-Q6_K.gguf" in text


def test_hub_reads_carry_the_token_only_to_huggingface(monkeypatch):
    import pollard_ggufcompat as gc
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert gc.hf_headers("https://huggingface.co/P/M/resolve/main/a.gguf")["Authorization"] == "Bearer hf_x"
    assert "Authorization" not in gc.hf_headers("https://example.com/a.gguf")


# ---------------------------------------------------------------- public from the start, the card first (2026-10-25)

def _fake_card(monkeypatch, c, seen=None):
    """pollard-card stands in: writes the README the card step would."""
    def run(cmd, ok_codes=(0,), env=None):
        assert cmd[0] == "pollard-card" and "--verified-only" in cmd
        os.makedirs(c["release_dir"], exist_ok=True)
        Path(c["release_dir"], "README.md").write_text("card")
        if seen is not None:
            seen.append(cmd)
    monkeypatch.setattr(R, "run_cmd", run)


def _gguf_rung(c, tag):
    rdir = ws.model_dir(c["f16"], create=True)
    p = os.path.join(rdir, f"Tiny-1B-Pollard-{tag}.gguf")
    Path(p).write_bytes(b"GGUF" + tag.encode())
    ws.record_build(c["f16"], "gguf", p, tag=tag)
    ws.record_gate(p, {"pass": True})
    return p


def test_the_repo_is_public_from_the_first_rung_and_the_card_goes_up_before_it(model, hub, monkeypatch):
    cfg = R.load_plan(None)
    cfg.update({"upload": True, "_free_after_upload": True})
    c = R._ctx(cfg, model)
    import pollard_ggufcompat as gc
    monkeypatch.setattr(gc, "runtime_of", lambda path: ("stock", {}))
    cards = []
    _fake_card(monkeypatch, c, cards)
    first = _gguf_rung(c, "Q6_K")
    assert R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", first), c, cfg, True, out=lambda *_: None) == "done"
    assert hub.private[REPO] is False, "public on the first rung, not at the end"
    files = [x for x in hub.calls if not str(x[1]).startswith("pollard/")]   # the results sidecars ride after each rung
    assert files[:2] == [("file", "README.md"), ("file", "Tiny-1B-Pollard-Q6_K.gguf")], "card first, then the rung"
    assert any(str(x[1]).startswith("pollard/results-gguf.") for x in hub.calls), "the rung's numbers as a sidecar"
    second = _gguf_rung(c, "IQ4_XS")
    assert R.publish_rung(R._publish_rung_step(c, "gguf", "IQ4_XS", second), c, cfg, True, out=lambda *_: None) == "done"
    files = [x for x in hub.calls if not str(x[1]).startswith("pollard/")]
    assert files[2:4] == [("file", "README.md"), ("file", "Tiny-1B-Pollard-IQ4_XS.gguf")], "card refreshed first"
    assert len(cards) == 2, "the card is rebuilt from the rungs passed so far, once per rung"
    # the last step: nothing to flip, only the file-list check
    hub.private[REPO] = "untouched"
    assert _public(c, cfg, ["gguf"], {"gguf/upload": "done"}, hub) == "done"
    assert hub.private[REPO] == "untouched", "go_public no longer changes visibility"


def test_private_true_creates_it_private_and_keeps_it_private(model, hub, monkeypatch):
    cfg = R.load_plan(None)
    cfg.update({"upload": True, "_free_after_upload": True, "private": True})
    c = R._ctx(cfg, model)
    import pollard_ggufcompat as gc
    monkeypatch.setattr(gc, "runtime_of", lambda path: ("stock", {}))
    _fake_card(monkeypatch, c)
    p = _gguf_rung(c, "Q6_K")
    assert R.publish_rung(R._publish_rung_step(c, "gguf", "Q6_K", p), c, cfg, True, out=lambda *_: None) == "done"
    assert hub.private[REPO] is True
    assert _public(c, cfg, ["gguf"], {"gguf/upload": "done"}, hub) == "done"
    assert hub.private[REPO] is True, "never flipped"


def test_a_repo_an_older_pollard_left_private_goes_public_at_its_first_upload(model, hub):
    cfg = R.load_plan(None)
    hub.files[REPO], hub.private[REPO] = set(), True
    R.open_repo(hub.api(), REPO, cfg, out=lambda *_: None)
    assert hub.private[REPO] is False
    hub.private[REPO] = True
    R.open_repo(hub.api(), REPO, dict(cfg, private=True), out=lambda *_: None)
    assert hub.private[REPO] is True


def test_without_free_mode_each_rung_uploads_its_card_right_after_its_gate(model, hub, monkeypatch):
    cfg = R.load_plan(None)
    cfg["upload"] = True
    c, lanes, steps = R.plan_steps(cfg, model, only=["mlx"])
    ids = [s.id for s in steps]
    rung_cards = [s for s in steps if s.fn == "rung_card"]
    assert rung_cards, "a card step per rung"
    for st in rung_cards:
        assert ids[ids.index(st.id) - 1] == f"mlx/{st.rung}/verify", "right after the rung's gate"
        assert not R.is_heavy(st)
    assert ids.index("mlx/upload") > max(ids.index(s.id) for s in rung_cards), "the rungs follow with the lane upload"
    _fake_card(monkeypatch, c)
    st = rung_cards[0]
    assert R.rung_card(st, c, cfg, False, out=lambda *_: None) == "held", "uploads off: nothing goes up"
    assert R.rung_card(st, c, cfg, True, out=lambda *_: None) == "done"
    assert hub.calls == [("file", "README.md")] and hub.private[REPO] is False
    # a plan that does not upload gets no card steps
    cfg2 = R.load_plan(None)
    assert not [s for s in R.plan_steps(cfg2, model, only=["mlx"])[2] if s.fn == "rung_card"]


def test_strict_gates_in_the_plan_passes_strict_to_every_verify(model):
    cfg = R.load_plan(None)
    verifies = [s for s in R.plan_steps(cfg, model, only=["mlx"])[2] if s.kind == "verify"]
    assert verifies and not any("--strict" in s.cmd for s in verifies), "report-only by default"
    cfg["strict_gates"] = True
    verifies = [s for s in R.plan_steps(cfg, model, only=["mlx"])[2] if s.kind == "verify"]
    assert all("--strict" in s.cmd for s in verifies)


def test_a_card_that_cannot_be_built_never_holds_the_rung_back(model, hub, monkeypatch):
    cfg = R.load_plan(None)
    cfg["upload"] = True
    c, _, steps = R.plan_steps(cfg, model, only=["mlx"])

    def boom(cmd, ok_codes=(0,), env=None):
        raise RuntimeError("exit 2")
    monkeypatch.setattr(R, "run_cmd", boom)
    st = next(s for s in steps if s.fn == "rung_card")
    assert R.rung_card(st, c, cfg, True, out=lambda *_: None) == "card not uploaded"
