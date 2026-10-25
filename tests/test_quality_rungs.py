"""Quality-targeted GGUF rungs (pollard_qtarget): the smallest measured-allocation build whose MEASURED KLD / top-1
meet a class, found by building + measuring candidates -- fakes only, no model."""
import fnmatch
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_qtarget as QT  # noqa: E402

# (ram, preset, projected GB): a 15 GB f16's grid, 2.0 .. 7.0 GB
POINTS = [(round(1.0 + 0.1 * i, 2), "IQ4_XS", round(2.0 + 0.1 * i, 2)) for i in range(51)]


def _fake(threshold_gb):
    """A model whose KLD falls with size and crosses 0.08 at `threshold_gb` (top-1 crosses 90 there too)."""
    built, gone = [], []

    def build(ram):
        gb = next(g for r, _, g in POINTS if r == ram)
        p = f"/tmp/cand-{gb:.2f}.gguf"
        built.append(gb)
        return p

    def measure(path):
        gb = float(path.rsplit("-", 1)[1][:-5])
        ok = gb >= threshold_gb - 1e-9
        return {"kld": 0.05 if ok else 0.5, "top1": 92.0 if ok else 70.0, "ppl": 6.7}

    return build, measure, built, gone


def test_finds_the_smallest_passing_budget_within_tolerance():
    build, measure, built, gone = _fake(4.6)
    res = QT.search(POINTS, "Q90", build, measure, discard=gone.append, log=lambda *a: None, max_steps=10)
    assert res["path"] and 4.6 <= res["gb"] <= 4.6 + QT.TOL_GB + 1e-9
    assert built[0] == 7.0, "the top of the range is measured first"
    assert all(p != res["path"] for p in gone), "the winner is never discarded"
    assert len(gone) == len(built) - 1, "every other candidate is deleted"


def test_unreachable_class_returns_nothing_and_cleans_up():
    build, measure, built, gone = _fake(99.0)
    res = QT.search(POINTS, "Q90", build, measure, discard=gone.append, log=lambda *a: None)
    assert res["path"] is None and built == [7.0] and gone == ["/tmp/cand-7.00.gguf"]


def test_top_known_to_pass_is_not_rebuilt():
    build, measure, built, gone = _fake(3.0)
    res = QT.search(POINTS, "Q85", build, measure, hi=4.6, discard=gone.append, log=lambda *a: None,
                    top_passes=True, max_steps=10)
    assert 4.6 not in built and res["path"] and 3.0 <= res["gb"] <= 3.0 + QT.TOL_GB + 1e-9


def test_top_known_and_nothing_smaller_passes_gives_no_rung():
    build, measure, built, gone = _fake(4.6)
    res = QT.search(POINTS, "Q85", build, measure, hi=4.6, discard=gone.append, log=lambda *a: None,
                    top_passes=True)
    assert res["path"] is None


def test_class_gate_and_names():
    assert QT.passes({"kld": 0.08, "top1": 90.0}, *QT.QUALITY_CLASSES["Q90"])
    assert not QT.passes({"kld": 0.081, "top1": 95.0}, *QT.QUALITY_CLASSES["Q90"])
    assert not QT.passes({"kld": 0.01, "top1": None}, *QT.QUALITY_CLASSES["Q90"])
    assert QT.rung_tag("Q90", 4.6312) == "Q90-4.6GB" and QT.manifest_tag_ok("Q85-3.9GB")
    assert QT.parse_class("q85") == "Q85" and QT.parse_class("IQ4_XS") is None


def test_default_gates_hold_a_quality_rung_to_its_class():
    import pollard_verify as V
    import yaml
    g = yaml.safe_load(V.DEFAULT_GATES) if hasattr(V, "DEFAULT_GATES") else None
    if g is None:
        pytest.skip("default gates text not exposed")
    rows = {tuple(r["match"]): r for r in g.get("rungs") or []}
    q90 = next(r for m, r in rows.items() if any(fnmatch.fnmatch("Q90-4.6GB", p) for p in m))
    assert q90["kld_max"] == 0.08 and q90["top1_min"] == 90
    assert not any(fnmatch.fnmatch("Q90-4.6GB", p) for m in rows for p in m if p.startswith(("Q6", "Q5", "Q4", "Q3")))


def test_release_labels_quality_rungs_and_filters_the_ladder(tmp_path, monkeypatch):
    import pollard_release as R
    f16 = tmp_path / "m-f16.gguf"
    f16.write_bytes(b"\0" * 1000)
    c = {"f16": str(f16), "imatrix": None,
         "gguf_lane": {"ladder": ["Q6_K"], "quality": ["Q90", "q85"]}}
    monkeypatch.setattr(R, "_size_gb", lambda p: 15.0)
    monkeypatch.setattr(R, "_is_moe_gguf", lambda p: False)
    labels = [lb for lb, _ in R.gguf_ladder_labels(c)]
    assert labels == ["Q6_K", "Q90", "Q85"]
    c["gguf_lane"]["quality"] = ["Q70"]
    with pytest.raises(SystemExit):
        R.gguf_ladder_labels(c)


def test_fit_tag_and_no_record_flags_exist():
    src = (Path(__file__).parents[1] / "tools" / "pollard_fit.py").read_text()
    assert '"--tag"' in src and '"--no-record"' in src and "tag=a.tag or base_preset" in src


def test_one_candidate_on_disk_at_a_time_and_the_winner_is_rebuilt():
    live, peak, builds = set(), [0], []

    def build(ram):
        gb = next(g for r, _, g in POINTS if r == ram)
        p = f"/tmp/cand-{gb:.2f}.gguf"
        live.add(p)
        builds.append(gb)
        peak[0] = max(peak[0], len(live))
        return p

    def measure(path):
        gb = float(path.rsplit("-", 1)[1][:-5])
        return {"kld": 0.05, "top1": 92.0} if gb >= 4.6 - 1e-9 else {"kld": 0.5, "top1": 70.0}

    res = QT.search(POINTS, "Q90", build, measure, discard=live.discard, log=lambda *a: None, max_steps=10)
    assert peak[0] == 1, "never two candidates on the disk"
    assert res["path"] in live and len(live) == 1, "only the winner remains, rebuilt at its budget"
    assert builds[-1] == res["gb"], "the winner is the last file built (rebuilt when a later candidate failed)"


def test_quality_build_checks_disk_and_a_complete_gguf():
    src = (Path(__file__).parents[1] / "tools" / "pollard_auto.py").read_text()
    assert "a full disk truncates the GGUF without an error" in src and "GGUFReader(out)" in src
