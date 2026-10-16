"""Retention vs f16 on the card: pollard-taskeval's per-task retention, its mt suite (chrF++ per language),
pollard-toolcall and pollard-decision all land in one results.json and one card table. No harness, no server,
no sacrebleu: the scorers are injected."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_taskeval as T  # noqa: E402
import pollard_workspace as ws  # noqa: E402

BUILDS = [{"name": "M-Pollard-Q6_K.gguf", "tag": "Q6_K", "bytes": 6_000_000_000},
          {"name": "M-Pollard-IQ2_M.gguf", "tag": "IQ2_M", "bytes": 2_000_000_000}]


def test_retention_table_categories_and_overall():
    got = {"gsm8k": 0.72, "ifeval": 0.80, "humaneval_plus": 0.5}
    ref = {"gsm8k": 0.80, "ifeval": 0.80, "humaneval_plus": 0.0}
    t = T.retention_table(got, ref, ["gsm8k", "ifeval", "humaneval_plus", "missing"])
    assert t["tasks"]["gsm8k"] == {"score": 72.0, "ref": 80.0, "retention": 90.0}
    assert "retention" not in t["tasks"]["humaneval_plus"], "a 0 reference has no retention"
    assert "missing" not in t["tasks"]
    assert t["categories"] == {"Instruction Following": 100.0, "Math": 90.0}
    assert t["overall"] == 95.0


def test_mt_rows_one_row_per_language_with_retention():
    data = {"en-de_DE": [("Hello", "Hallo"), ("Bye", "Tschuess")], "en-ja_JP": [("Hi", "x")]}
    hyps = {"en-de_DE": ["Hallo", "Ciao"], "en-ja_JP": ["x"]}
    ref_hyps = {"en-de_DE": ["Hallo", "Tschuess"], "en-ja_JP": ["x"]}
    fake_chrf = lambda h, r: 100.0 * sum(a == b for a, b in zip(h, r)) / len(r)
    rows = T.mt_rows(data, hyps, ref_hyps, chrf=fake_chrf, comet=lambda s, h, r: 0.5)
    assert rows["en-de_DE"] == {"n": 2, "chrf": 50.0, "ref_chrf": 100.0, "retention": 50.0,
                                "comet": 0.5, "ref_comet": 0.5}
    assert rows["en-ja_JP"]["retention"] == 100.0
    assert T.pair_langs("en-zh_CN") == ("en", "zh")
    msg = T.mt_prompt("Hello", "en", "de")[0]["content"]
    assert "English" in msg and "German" in msg and msg.endswith("Hello")


def test_mt_loader_drops_bad_sources_and_zips_flores_by_id():
    def wmt(name, cfg, split, revision, streaming):
        assert name == "google/wmt24pp" and cfg == "en-de_DE" and revision == T.MT_SETS["wmt24pp"]["revision"]
        return [{"source": "a", "target": "A", "is_bad_source": False},
                {"source": "b", "target": "B", "is_bad_source": True},
                {"source": "c", "target": "C", "is_bad_source": False}]
    assert T.load_mt_pairs("wmt24pp", "en-de_DE", 0, wmt) == [("a", "A"), ("c", "C")]
    assert T.load_mt_pairs("wmt24pp", "en-de_DE", 1, wmt) == [("a", "A")]

    def flores(name, cfg, split, revision, streaming):
        assert split == "devtest", "FLORES dev is calibration data; only devtest may be scored"
        return {"eng_Latn": [{"id": 1, "text": "one"}, {"id": 2, "text": "two"}],
                "deu_Latn": [{"id": 2, "text": "zwei"}, {"id": 1, "text": "eins"}]}[cfg]
    assert T.load_mt_pairs("flores", "en-de_DE", 0, flores) == [("two", "zwei"), ("one", "eins")]


def test_comet_refuses_non_commercial_checkpoints_by_default():
    import pytest
    with pytest.raises(SystemExit, match="NC"):
        T.comet_scorer("Unbabel/wmt22-cometkiwi-da")
    assert T.COMET_OPEN == {"Unbabel/wmt22-comet-da": "apache-2.0"}


def test_results_merge_keeps_every_tools_part(tmp_path):
    p = str(tmp_path / "results.json")
    ws.merge_results(p, "Q6_K", {"ppl": 9.1, "toolcall": {"rate": 1.0}})
    ws.merge_results(p, "Q6_K", {"retention": {"overall": 99.0}, "toolcall": {"ref_rate": 1.0}},
                     {"_retention_suite": {"suite": "quick"}})
    r = json.load(open(p))
    assert r["Q6_K"] == {"ppl": 9.1, "toolcall": {"rate": 1.0, "ref_rate": 1.0}, "retention": {"overall": 99.0}}
    assert r["_retention_suite"] == {"suite": "quick"}


def test_card_retention_table_merges_tasks_mt_toolcall_and_decisions():
    results = {
        "Q6_K": {"retention": T.retention_table({"gsm8k": 0.79, "ifeval": 0.8}, {"gsm8k": 0.8, "ifeval": 0.8}),
                 "mt": {"en-de_DE": {"n": 100, "chrf": 60.0, "ref_chrf": 60.5, "retention": 99.17}},
                 "toolcall": {"pass": True, "rate": 1.0, "ref_rate": 1.0},
                 "decision": {"agreement": 0.98}},
        "IQ2_M": {"retention": T.retention_table({"gsm8k": 0.6, "ifeval": 0.7}, {"gsm8k": 0.8, "ifeval": 0.8}),
                  "toolcall": {"pass": False, "rate": 0.5}},
        "_retention_suite": {"suite": "quick", "tasks": ["gsm8k", "ifeval"], "limit": 200},
        "_mt_suite": {"set": "google/wmt24pp", "split": "train", "limit": 100},
    }
    md = "\n".join(C.retention_section(BUILDS, results))
    assert "## Retention vs f16" in md
    head = next(l for l in md.splitlines() if l.startswith("| file"))
    assert "Math" in head and "Instruction Following" in head and "Translation (chrF++)" in head
    assert "Tool calls" in head and "Decisions (agree)" in head and "Overall" in head
    q6 = next(l for l in md.splitlines() if "Q6_K.gguf" in l and l.startswith("| `"))
    iq2 = next(l for l in md.splitlines() if "IQ2_M.gguf" in l and l.startswith("| `"))
    assert "98.8%" in q6 and "100.0%" in q6 and "98%" in q6
    assert "50% valid" in iq2, "no f16 tool-call run recorded: say valid, not retention"
    assert md.index("Q6_K.gguf") < md.index("IQ2_M.gguf"), "largest first, like every other table"
    assert "### Translation, per language" in md and "| en-de_DE | 60.5 | 60.0 (99%) |" in md and "IQ2_M.gguf` |" not in md.split("### Translation")[1]
    assert "`gsm8k`" in md and "limit 200" in md


def test_card_has_no_retention_section_when_nothing_measured():
    assert C.retention_section(BUILDS, {"Q6_K": {"ppl": 9.0}}) == []


def test_toolcall_writes_rate_and_reference_into_results(monkeypatch, tmp_path):
    import pytest
    import pollard_toolcall as TC
    good = {"ok": True, "failure": None, "tool": "t", "detail": "valid call", "sample": ""}
    bad = {"ok": False, "failure": "no-call", "tool": "t", "detail": "prose", "sample": "hi"}
    monkeypatch.setattr(TC, "run", lambda cli, model, ngl, threads=None:
                        [good, bad] if "IQ2" in model else [good, good])
    res = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["pollard-toolcall", "--gguf", "M-IQ2_M.gguf", "--ref", "M-f16.gguf",
                                      "--results", str(res), "--rung", "M-Pollard-IQ2_M.gguf", "--quick"])
    with pytest.raises(SystemExit):
        TC.main()
    tc = json.load(open(res))["M-Pollard-IQ2_M.gguf"]["toolcall"]
    assert tc["rate"] == 0.5 and tc["ref_rate"] == 1.0 and tc["drop"] == 0.5 and tc["pass"] is False
    md = "\n".join(C.retention_section(BUILDS, json.load(open(res))))
    assert "| `M-Pollard-IQ2_M.gguf` | 50.0% |" in md


def test_full_toolcall_writes_retention_with_ci_into_results(monkeypatch, tmp_path):
    """The served gate: per-probe rows for the rung and the f16 -> retention + paired CI in results.json, which the
    card's retention section and pollard-verify's toolcall_retention_min read."""
    import pytest
    import pollard_toolcall as TC

    def rows(n_ok):
        return [{"id": p["id"], "category": p["category"], "expected": p["expect"] is not None,
                 "called": True, "parsed": True, "name_ok": True, "args_valid": True, "exact": i < n_ok,
                 "ok": i < n_ok, "failure": None if i < n_ok else "wrong-value", "detail": "", "sample": ""}
                for i, p in enumerate(TC.SUITE)]
    monkeypatch.setattr(TC, "run_model", lambda *a, **k: (rows(45), {"family": "hermes"}))
    monkeypatch.setattr(TC, "run_ref_cached", lambda *a, **k: (rows(50), {"family": "hermes"}))
    res = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", ["pollard-toolcall", "--gguf", "M-IQ2_M.gguf", "--ref", "M-f16.gguf",
                                      "--results", str(res), "--rung", "M-Pollard-IQ2_M.gguf"])
    with pytest.raises(SystemExit):
        TC.main()
    tc = json.load(open(res))["M-Pollard-IQ2_M.gguf"]["toolcall"]
    assert tc["mode"] == "full" and tc["retention"] == 90.0 and tc["pass"] is True     # 45/50, gate at 90
    lo, hi = tc["retention_ci"]
    assert lo < 90.0 <= hi <= 100.0
    assert tc["family"] == "hermes" and tc["n"] == 60 and tc["rate_ci"][0] < tc["rate"] < tc["rate_ci"][1]
    md = "\n".join(C.retention_section(BUILDS, json.load(open(res))))
    assert "| `M-Pollard-IQ2_M.gguf` | 90.0% |" in md
    import pollard_verify as V
    m = V.metrics_from_results(json.load(open(res)), "M-Pollard-IQ2_M.gguf")
    assert m["toolcall_retention"] == 90.0
