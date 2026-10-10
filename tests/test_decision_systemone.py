"""A native decision model (a decision head: Cloudflare's clef) is gated through /v1/systemone.

A clef server answers only /v1/systemone -- it cannot generate text, so the letter-logprob readout the
gate uses for OpenJev-style models has nothing to read. The gate asks /v1/models whether the model is a
native decision model and, if so, takes the option probabilities the endpoint returns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_bench  # noqa: E402
import pollard_decision as pd  # noqa: E402
import pollard_modelkind as mk  # noqa: E402

MODELS_HEAD = {"data": [{"id": "clef-flash", "architecture": {"input_modalities": ["text"],
                                                              "output_modalities": ["decisions"]}}]}
MODELS_TEXT = {"data": [{"id": "qwen", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}]}


def _systemone(answer_of):
    """A fake server: /v1/systemone puts 0.9 on the option answer_of(state) names, the rest spread."""
    calls = []

    def post(base, path, payload, timeout=3600):
        calls.append(path)
        assert path == "/v1/systemone", path
        opts = list(payload["questions"]["q"]["criteria"])
        pick = answer_of(payload["state"], opts)
        rest = 0.1 / (len(opts) - 1)
        return {"answers": {"q": {"type": "choice", "choice": pick,
                                  "probabilities": {o: (0.9 if o == pick else rest) for o in opts}}}}
    return post, calls


def test_native_reads_output_modalities(monkeypatch):
    monkeypatch.setattr(pollard_bench, "_get", lambda base, path, timeout=10: MODELS_HEAD)
    assert pd.native("http://x")
    monkeypatch.setattr(pollard_bench, "_get", lambda base, path, timeout=10: MODELS_TEXT)
    assert not pd.native("http://x")


def test_head_rows_come_from_systemone(monkeypatch):
    monkeypatch.setattr(pollard_bench, "_get", lambda base, path, timeout=10: MODELS_HEAD)
    truth = {s: o[a] for s, q, o, a in pd.DECISION_SET}
    post, calls = _systemone(lambda state, opts: truth[state])
    rows = pd.run("http://x", post)
    assert calls and set(calls) == {"/v1/systemone"}          # never the chat endpoint
    assert all(r["mass"] == 1.0 for r in rows)
    for r in rows:
        assert abs(sum(r["probs"]) - 1) < 1e-9
        assert max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["answer"]
    board = pd.score_rows(rows, rows)
    assert board["accuracy"] == 1.0 and board["agreement"] == 1.0 and board["option_kl"] < 1e-9
    assert pd.verdict(board)[0] == "PASS"


def test_head_build_that_flips_answers_fails(monkeypatch):
    monkeypatch.setattr(pollard_bench, "_get", lambda base, path, timeout=10: MODELS_HEAD)
    truth = {s: o[a] for s, q, o, a in pd.DECISION_SET}
    ref = pd.run("http://x", _systemone(lambda s, o: truth[s])[0])
    bad = pd.run("http://x", _systemone(lambda s, o: o[-1] if truth[s] != o[-1] else o[0])[0])
    v, why = pd.verdict(pd.score_rows(bad, ref))
    assert v == "FAIL", why


def test_clef_checkout_is_a_decision_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                      "model_type": "qwen3_5"}))
    assert not mk._is_decision(str(tmp_path), {})
    (tmp_path / "joint_head_config.json").write_text("{}")
    assert mk._is_decision(str(tmp_path), {})


def test_clef_gguf_is_a_decision_model():
    assert mk._is_decision("x.gguf", {"general.architecture": "clef", "clef.decision.head_count": 8})
    assert not mk._is_decision("x.gguf", {"general.architecture": "qwen35", "qwen35.block_count": 32})


def test_native_ignores_non_dict_model_entries(monkeypatch):
    """Joey's review: a non-dict entry in /v1/models data raised AttributeError outside the try."""
    monkeypatch.setattr(pollard_bench, "_get", lambda base, path, timeout=10: {"data": ["junk", None, MODELS_HEAD["data"][0]]})
    assert pd.native("http://x")
    monkeypatch.setattr(pollard_bench, "_get", lambda base, path, timeout=10: {"data": ["junk"]})
    assert not pd.native("http://x")
