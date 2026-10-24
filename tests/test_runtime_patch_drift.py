"""pollard-runtime --check-patches: the daily CI drift check runs --update's own re-apply, and names a patch that drifted.

The qwen3omni-deepstack patch drifted against upstream and blocked `pollard-runtime --update` on the night a release
needed the update. A canary patch that conflicts with "upstream" must be reported -- plain and 3-way -- with the
upstream commit, and the exit code must fail the job; a patch that applies (or that upstream absorbed) must not.
"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_runtime_update as U  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True, text=True)


def _diff(repo, path, new):
    """A patch (as `git diff` writes it) changing `path` to `new` against the repo's HEAD; the tree is left at HEAD."""
    open(os.path.join(repo, path), "w").write(new)
    d = subprocess.run(["git", "-C", repo, "diff"], capture_output=True, text=True, check=True).stdout
    _git(repo, "checkout", "--", path)
    return d


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """A tiny 'upstream' with history, and three captured patches written against its first commit:
    clean (still applies), canary (upstream rewrote the same line since), absorbed (upstream merged it)."""
    for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"), ("GIT_COMMITTER_NAME", "t"),
                 ("GIT_COMMITTER_EMAIL", "t@t")):
        monkeypatch.setenv(k, v)
    repo = str(tmp_path / "upstream")
    os.makedirs(os.path.join(repo, "src"))
    _git(tmp_path, "init", "-q", repo)
    body = "".join(f"line {i}\n" for i in range(40))
    open(os.path.join(repo, "src", "a.cpp"), "w").write(body)
    open(os.path.join(repo, "src", "b.cpp"), "w").write(body)
    open(os.path.join(repo, "src", "c.cpp"), "w").write(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    pdir = tmp_path / "patches"
    pdir.mkdir()
    patches = {
        "eng-clean": _diff(repo, "src/a.cpp", body.replace("line 5\n", "line 5 // pollard clean change\n")),
        "eng-canary": _diff(repo, "src/b.cpp", body.replace("line 20\n", "line 20 // pollard canary change\n")),
        "eng-absorbed": _diff(repo, "src/c.cpp", body.replace("line 30\n", "line 30 absorbed upstream verbatim here\n")),
    }
    metas = []
    for name, text in patches.items():
        p = pdir / f"{name}.patch"
        p.write_text(text)
        metas.append({"_patch": str(p), "files": [f"src/{ {'eng-clean': 'a', 'eng-canary': 'b', 'eng-absorbed': 'c'}[name]}.cpp"],
                      "base_date": "2026-01-01"})
    # upstream moves on: rewrites the canary's line, merges the absorbed change itself
    open(os.path.join(repo, "src", "b.cpp"), "w").write(body.replace("line 20\n", "line 20 rewritten upstream\n"))
    open(os.path.join(repo, "src", "c.cpp"), "w").write(body.replace("line 30\n", "line 30 absorbed upstream verbatim here\n"))
    _git(repo, "commit", "-q", "-am", "upstream moves on")
    eng_dir = str(tmp_path / "runtime" / "eng")
    monkeypatch.setitem(U.ENGINES, "eng", {"url": repo, "dir": eng_dir})
    monkeypatch.setattr(U, "_patches_for", lambda tree: metas if os.path.abspath(tree) == os.path.abspath(eng_dir) else [])
    monkeypatch.setattr(U, "REPO", str(tmp_path))         # no runtime-patches/scripts here
    return repo


def test_canary_conflict_is_reported_plain_and_3way(upstream, tmp_path):
    staging = str(tmp_path / "work" / "eng")
    ok, head = U.stage_upstream("eng", staging)
    assert ok and head
    rows, _notes = U.check_patches("eng", staging)
    by = {r["patch"]: r for r in rows}
    assert [r["patch"] for r in rows] == ["eng-clean", "eng-canary", "eng-absorbed"]        # the updater's order
    assert by["eng-clean"]["plain"] and by["eng-clean"]["threeway"] and by["eng-clean"]["verdict"] == "applied"
    assert not by["eng-canary"]["plain"] and not by["eng-canary"]["threeway"]
    assert by["eng-canary"]["verdict"] == "CONFLICT"
    assert by["eng-absorbed"]["verdict"] in ("absorbed", "applied")
    md, bad = U.drift_report([("eng", head, rows, None)])
    assert len(bad) == 1 and bad[0].startswith(f"eng-canary no longer applies to eng @ {head}")
    assert "| `eng-canary` | FAILS | FAILS | **CONFLICT -- needs a rebase** |" in md
    assert "| `eng-clean` | ok | ok | applied |" in md


def test_cli_fails_the_job_and_writes_the_summary(upstream, tmp_path, monkeypatch):
    import pollard_runtime as R
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", ["pollard-runtime", "--check-patches", "eng", "--work", str(tmp_path / "w"),
                                      "--summary", str(summary)])
    assert R.main() == 1
    text = summary.read_text()
    assert "eng-canary" in text and "Needs a rebase" in text


def test_every_patch_applying_passes(upstream, tmp_path, monkeypatch):
    metas = [m for m in U._patches_for(U.ENGINES["eng"]["dir"]) if "canary" not in m["_patch"]]
    monkeypatch.setattr(U, "_patches_for", lambda tree: metas)
    assert U.drift_main(["eng"], work=str(tmp_path / "w")) == 0


def test_given_tree_is_used_as_is(upstream, tmp_path):
    staging = str(tmp_path / "pre")
    subprocess.run(["git", "clone", "-q", upstream, staging], check=True)
    assert U.drift_main(["eng"], tree=staging) == 1


def test_an_unreachable_upstream_fails_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setitem(U.ENGINES, "gone", {"url": str(tmp_path / "nope"), "dir": str(tmp_path / "rt" / "gone")})
    monkeypatch.setattr(U, "_patches_for", lambda tree: [{"_patch": "x.patch", "base_date": "2026-01-01"}])
    summary = tmp_path / "s.md"
    assert U.drift_main(["gone"], work=str(tmp_path / "w"), summary=str(summary)) == 1
    assert "could not stage upstream" in summary.read_text()


def test_reapply_without_record_is_unchanged(upstream, tmp_path):
    """--update's own call (no record) still blocks on the same patch."""
    staging = str(tmp_path / "work" / "eng")
    assert U.stage_upstream("eng", staging)[0]
    block, _ = U.reapply(staging, U.ENGINES["eng"]["dir"], fresh=True)
    assert len(block) == 1 and block[0].startswith("eng-canary.patch is live")


def test_workflow_is_hardened():
    p = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "runtime-patch-drift.yml")
    if not os.path.exists(p):
        pytest.skip("workflow not shipped in this install")
    text = open(p, encoding="utf-8").read()
    import re
    assert re.search(r"^permissions:\n  contents: read$", text, re.M)          # top level read-only
    assert text.count("issues: write") == 1                                     # only the issue job
    uses = re.findall(r"uses: (\S+)", text)
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses)
    assert "--check-patches all" in text and "cmake" not in text               # apply-check only, no build
    assert "workflow_dispatch" in text and "cron:" in text
