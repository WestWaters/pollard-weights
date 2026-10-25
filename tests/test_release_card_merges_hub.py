"""The card UPDATES the repo's card, it never replaces it: every machine's numbers meet in the repo as
pollard/results-<lane>.<host>.json sidecars, and any machine that pushes the card builds it from the union of its own
rungs, every sidecar, and the rungs actually on the Hub -- so nothing another machine published is dropped."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402


class Hub:
    """A fake repo: files by path; upload_file writes, hf_hub_download reads."""

    def __init__(self, tmp):
        self.files, self.tmp = {}, Path(tmp)

    def list_repo_files(self, repo):
        return list(self.files)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **k):
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def hf_hub_download(self, repo_id, filename, **k):
        p = self.tmp / "dl" / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.files[filename])
        return str(p)


def _machine(home, monkeypatch, lane, rung, kld):
    """A machine's workspace: one verified rung of `lane` in its MANIFEST and its results."""
    monkeypatch.setenv("POLLARD_HOME", str(home))
    hf = home / "src" / "M"
    hf.mkdir(parents=True, exist_ok=True)
    (hf / "config.json").write_text("{}")
    cfg = R.load_plan(None)
    for l in cfg["lanes"]:
        cfg["lanes"][l]["enabled"] = l == lane
    c = R._ctx(cfg, str(hf))
    key = R.lane_key(c, lane)
    name = f"M-Pollard-{rung}.gguf" if lane == "gguf" else f"M-Pollard-{rung}"
    ws.record_build(key, lane, str(home / name), tag=rung)
    ws.update_build(key, str(home / name), {"verified": True})
    os.makedirs(os.path.dirname(R.results_path(c, lane)), exist_ok=True)
    json.dump({rung: {"mean_kld": kld}}, open(R.results_path(c, lane), "w"))
    return cfg, c, name


def _fake_card(seen):
    """pollard-card stand-in: writes a card naming every verified build + result it can see."""
    def run(cmd, **k):
        out = cmd[cmd.index("--out") + 1]
        rows = []
        for flag_i, tok in enumerate(cmd):
            if tok in ("--builds-from",) or (tok == "--also"):
                key = cmd[flag_i + 1] if tok == "--builds-from" else cmd[flag_i + 2]
                for b in ws.read_manifest(key).get("builds", []):
                    rows.append(b["name"])
        res = {}
        for tok_i, tok in enumerate(cmd):
            if tok == "--results" or (tok == "--also"):
                p = cmd[tok_i + 1] if tok == "--results" else cmd[tok_i + 3]
                if os.path.exists(p):
                    res.update(json.load(open(p)))
        seen.append((sorted(set(rows)), res))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write("# card\n" + "\n".join(f"| `{r}` |" for r in sorted(set(rows))) + "\n## Credits\n")
    return run


def test_two_machines_each_push_keeps_the_others_rows(tmp_path, monkeypatch):
    hub, seen = Hub(tmp_path), []
    monkeypatch.setattr(R, "run_cmd", _fake_card(seen))
    # machine A: a GGUF rung, published (file + sidecar) and carded
    cfg_a, c_a, n_a = _machine(tmp_path / "a", monkeypatch, "gguf", "Q6_K", 0.013)
    cfg_a["host"] = "mac"
    hub.files[n_a] = b"gguf"
    R.push_card(hub, "O/M-Pollard", c_a, cfg_a, out=lambda *a: None)
    assert R.push_sidecar(hub, "O/M-Pollard", c_a, cfg_a, "gguf", out=lambda *a: None)
    assert "pollard/results-gguf.mac.json" in hub.files
    # machine B: an MLX rung -- its card must keep A's GGUF row
    cfg_b, c_b, n_b = _machine(tmp_path / "b", monkeypatch, "mlx", "MLX-4bit", 0.05)
    cfg_b["host"] = "box"
    hub.files["MLX/4bit/config.json"] = b"{}"
    monkeypatch.setattr(R, "_on_hub", lambda b, lane, files: True)
    R.push_card(hub, "O/M-Pollard", c_b, cfg_b, out=lambda *a: None)
    rows, res = seen[-1]
    assert n_a in rows and n_b in rows, "B's card keeps A's rung"
    assert res.get("Q6_K") == {"mean_kld": 0.013} and res.get("MLX-4bit") == {"mean_kld": 0.05}
    card = hub.files["README.md"].decode()
    assert n_a in card and n_b in card
    # B's own workspace is left as it was: the overlay is for the card build only
    assert not any(b["name"] == n_a for b in ws.read_manifest(R.lane_key(c_b, "gguf")).get("builds", []))


def test_a_rung_with_no_numbers_anywhere_is_listed_pending(tmp_path, monkeypatch):
    hub, seen = Hub(tmp_path), []
    monkeypatch.setattr(R, "run_cmd", _fake_card(seen))
    cfg, c, n = _machine(tmp_path / "a", monkeypatch, "gguf", "Q6_K", 0.013)
    hub.files[n] = b"gguf"
    hub.files["M-Pollard-IQ1_KT.gguf"] = b"gguf"            # published by another machine, no sidecar yet
    hub.files["GPTQ/int4/config.json"] = b"{}"
    R.push_card(hub, "O/M-Pollard", c, cfg, out=lambda *a: None)
    card = hub.files["README.md"].decode()
    assert "numbers pending" in card and "`M-Pollard-IQ1_KT.gguf`" in card and "`GPTQ/int4/`" in card
    assert card.index("numbers pending") < card.index("## Credits")


def test_card_and_sidecar_failures_never_block(tmp_path, monkeypatch):
    class Broken(Hub):
        def upload_file(self, **k):
            raise RuntimeError("hub down")
    cfg, c, n = _machine(tmp_path / "a", monkeypatch, "gguf", "Q6_K", 0.013)
    said = []
    assert R.push_sidecar(Broken(tmp_path), "O/M-Pollard", c, cfg, "gguf", out=said.append) is False
    assert "not uploaded" in said[-1]
    src = (Path(__file__).parents[1] / "tools" / "pollard_release.py").read_text()
    assert "never holds the rung back" in src and "push_sidecar(api, repo, c, cfg, lane, out)" in src


def test_card_args_reach_pollard_card(tmp_path, monkeypatch):
    cfg, c, n = _machine(tmp_path / "a", monkeypatch, "gguf", "Q6_K", 0.013)
    cfg["card_args"] = ["--params", "7.996B"]
    cmd = R.card_cmd(cfg, c, ["gguf"])
    assert cmd[-2:] == ["--params", "7.996B"]
