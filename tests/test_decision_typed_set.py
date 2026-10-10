"""pollard-decision --set typed-decisions: the 2000-decision benchmark as the decision gate.

On the 20 built-in questions every clef rung (and the reference) scored 100%, so agreement could not
separate rungs; typed-decisions (Apache-2.0, 400 states x 5 typed questions, soft gold, each row already a
/v1/systemone request) can.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_decision as pd  # noqa: E402

ROW = {"id": "x_000", "workflow": "invoice_processing", "state": json.dumps({"amount": 12.5}),
       "questions": {"route": {"type": "choice", "instructions": "Who?", "criteria": {"ap": "a", "fraud": "f"}},
                     "dup": {"type": "noul", "instructions": "Duplicate?", "criteria": {"true": "t", "false": "f"}},
                     "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "mid", "high"]}},
       "gold": {"route": {"type": "choice", "label": "ap", "probabilities": {"ap": 0.8, "fraud": 0.2}},
                "dup": {"type": "noul", "label": "false", "probabilities": {"true": 0.1, "false": 0.9}},
                "risk": {"type": "score", "label": "0", "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}}}}


def _post_factory(route_p=0.8, noul_p=0.1, score=(0.7, 0.2, 0.1)):
    def post(base, path, payload, timeout=3600):
        assert path == "/v1/systemone" and set(payload) == {"state", "questions"}
        return {"answers": {"route": {"type": "choice", "probabilities": {"ap": route_p, "fraud": 1 - route_p}},
                            "dup": {"type": "noul", "noul": noul_p},
                            "risk": {"type": "score", "probabilities": {"0": score[0], "1": score[1], "2": score[2]}}}}
    return post


def test_load_typed_parses_string_fields_and_caches(tmp_path):
    page = {"num_rows_total": 1, "rows": [{"row": dict(ROW, questions=json.dumps(ROW["questions"]),
                                                       gold=json.dumps(ROW["gold"]))}]}
    calls = []

    def fetch(url):
        calls.append(url)
        return {"sha": "6a7927fa3c8a"} if "api/datasets" in url else page
    cache = tmp_path / "td.json"
    d = pd.load_typed(str(cache), fetch)
    assert d["revision"].startswith("6a7927fa") and d["rows"][0]["questions"]["dup"]["type"] == "noul"
    pd.load_typed(str(cache), lambda u: (_ for _ in ()).throw(AssertionError("refetched")))   # cached


def test_answers_map_to_gold_keys():
    rows = pd.run_typed("http://x", _post_factory(), [ROW])
    by = {r["qid"]: r["dist"] for r in rows}
    assert by["dup"] == {"true": 0.1, "false": 0.9} and set(by["risk"]) == {"0", "1", "2"}


def test_board_identical_rung_is_a_pass_and_matches_gold():
    ref = pd.run_typed("http://x", _post_factory(), [ROW])
    rung = pd.run_typed("http://x", _post_factory(), [ROW])
    b = pd.score_typed(rung, ref)
    assert b["n"] == 3 and b["accuracy"] == 1.0 and b["agreement"] == 1.0 and b["option_kl"] < 1e-12
    assert b["kl_from_gold"] < 1e-9 and pd.verdict(b)[0] == "PASS"


def test_board_drifted_rung_is_seen():
    ref = pd.run_typed("http://x", _post_factory(), [ROW])
    rung = pd.run_typed("http://x", _post_factory(route_p=0.3, noul_p=0.7, score=(0.1, 0.2, 0.7)), [ROW])
    b = pd.score_typed(rung, ref)
    assert b["agreement"] == 0.0 and b["option_kl"] > 0.3 and pd.verdict(b)[0] == "FAIL"


def test_card_names_the_benchmark():
    builds = [{"name": "m-Q6_K.gguf", "tag": "Q6_K", "bytes": 1}]
    res = {"m-Q6_K.gguf": {"decision": {"n": 2000, "agreement": 1.0, "option_kl": 1e-4, "mean_abs_drift": 0.001,
                                        "accuracy": 0.7, "set": "LocalLLaMA/typed-decisions@6a7927fa"}}}
    md = "\n".join(C.decision_section(builds, res))
    assert "typed-decisions" in md and "2000 decisions" in md and "6a7927fa" in md
