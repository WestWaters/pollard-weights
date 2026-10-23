"""pollard-decision: calibration on the board (ECE / Brier / NLL / over95, flips + agreement CI) and pre-gate checks.

What it protects: a decision model is sold on calibrated probabilities, and agreement + option KL can both
look fine while a rung pushes 0.7s to 0.97s -- an app that thresholds on confidence then acts on answers it
should have escalated. So the board carries calibration against the gold labels, and against a reference the
ECE change, the count of flipped decisions and a paired-bootstrap CI on agreement (20 questions is not 2000).

And two converter/runtime gaps that make the board measure the wrong model, read off the GGUF before it runs:
a clef-type head whose SOURCE calibrated temperatures (Unsloth's FastDecisionModel writes them into
joint_head_config.json) but whose GGUF lacks `<arch>.decision.temperature.*` answers uncalibrated -- Cloudflare's own
clef checkpoints declare none, so T=1 is right for them; a pplx-decider whose source is noncausal but whose GGUF
lacks `<arch>.attention.causal = false` runs causally upstream (~3/25 decisions flip). Warnings, not failures.
"""
from __future__ import annotations

import json
import os
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_decision as pd  # noqa: E402


def test_calibration_numbers_on_a_hand_checked_set():
    items = [({"a": 0.9, "b": 0.1}, "a"),       # right, conf .9
             ({"a": 0.6, "b": 0.4}, "b"),       # wrong, conf .6
             ({"a": 0.97, "b": 0.03}, "b"),     # wrong, conf .97 -> over95
             ({"a": 0.2, "b": 0.8}, "b")]       # right, conf .8
    c = pd.calibration(items)
    # bins: .9 -> [9] (1 right), .6 -> [6] (0), .97 -> [9] (0), .8 -> [8] (1)
    # bin 9: n=2 conf 1.87 ok 1 -> |1-1.87|=.87; bin 6: .6; bin 8: |1-.8|=.2  => (.87+.6+.2)/4
    assert c["ece"] == pytest.approx((0.87 + 0.6 + 0.2) / 4)
    assert c["brier"] == pytest.approx((2 * 0.01 + 2 * 0.36 + 2 * 0.97 ** 2 + 2 * 0.04) / 4)
    assert c["nll"] == pytest.approx(-(math.log(0.9) + math.log(0.4) + math.log(0.03) + math.log(0.8)) / 4)
    assert c["wrong"] == 2 and c["over95_n"] == 1 and c["over95"] == 0.5


def test_perfectly_calibrated_and_overconfident():
    # 10 answers at confidence 0.8, 8 right: calibrated -> ECE 0
    items = [({"y": 0.8, "n": 0.2}, "y")] * 8 + [({"y": 0.8, "n": 0.2}, "n")] * 2
    assert pd.calibration(items)["ece"] == pytest.approx(0.0)
    hot = [({"y": 0.99, "n": 0.01}, "y")] * 8 + [({"y": 0.99, "n": 0.01}, "n")] * 2
    c = pd.calibration(hot)
    assert c["ece"] == pytest.approx(0.19) and c["over95"] == 1.0


def test_bootstrap_ci_brackets_agreement_and_is_reproducible():
    flags = [1] * 17 + [0] * 3
    lo, hi = pd.bootstrap_ci(flags)
    assert lo <= 0.85 <= hi and hi - lo > 0.1          # 20 questions: the interval is wide, and says so
    assert pd.bootstrap_ci(flags) == (lo, hi)
    assert pd.bootstrap_ci([1] * 50) == (1.0, 1.0)


def _row(p, ans):
    return {"probs": p, "mass": 1.0, "answer": ans}


def test_score_rows_carries_calibration_and_flips():
    ref = [_row([0.9, 0.1], 0), _row([0.3, 0.7], 1), _row([0.6, 0.4], 0)]
    rung = [_row([0.99, 0.01], 0), _row([0.6, 0.4], 1), _row([0.55, 0.45], 0)]
    b = pd.score_rows(rung, ref)
    assert {"ece", "brier", "nll", "over95"} <= set(b)
    assert b["flips"] == 1 and b["agreement"] == pytest.approx(2 / 3)
    assert b["delta_ece"] == pytest.approx(b["ece"] - b["ref_ece"])
    lo, hi = b["agreement_ci"]
    assert lo <= b["agreement"] <= hi
    assert "flips 1/3" in pd.board_calibration_line(b)
    assert "flips" not in pd.score_rows(rung)                           # no reference, no flips


def test_score_typed_carries_calibration_against_the_gold_label():
    def r(dist, label):
        return {"dist": dist, "gold": {"label": label, "probabilities": {}}, "workflow": "w", "type": "choice"}
    rows = [r({"ap": 0.97, "fraud": 0.03}, "fraud"), r({"ap": 0.8, "fraud": 0.2}, "ap")]
    ref = [r({"ap": 0.4, "fraud": 0.6}, "fraud"), r({"ap": 0.8, "fraud": 0.2}, "ap")]
    b = pd.score_typed(rows, ref)
    assert b["over95"] == 1.0 and b["flips"] == 1 and b["ref_ece"] < b["ece"]
    assert pd.verdict(b)[0] in ("PASS", "WEAK", "FAIL")                    # verdict still reads the board


def test_card_decision_table_shows_ece_and_flips_when_present():
    builds = [{"name": "c-Q6_K.gguf", "bytes": 2}, {"name": "c-IQ2.gguf", "bytes": 1}]
    res = {"c-Q6_K.gguf": {"decision": {"n": 20, "agreement": 1.0, "option_kl": 1e-4, "mean_abs_drift": 0.001,
                                        "accuracy": 1.0, "ece": 0.0312, "flips": 0}},
           "c-IQ2.gguf": {"decision": {"n": 20, "agreement": 0.9, "option_kl": 0.04, "mean_abs_drift": 0.03,
                                       "accuracy": 0.9, "ece": 0.0871, "flips": 2}}}
    md = "\n".join(C.decision_section(builds, res))
    assert "| ECE | flips vs f16 |" in md and "| 0.0312 | 0 |" in md and "| 0.0871 | 2 |" in md
    old = {k: {"decision": {x: y for x, y in v["decision"].items() if x not in ("ece", "flips")}} for k, v in res.items()}
    assert "ECE" not in "\n".join(C.decision_section(builds, old))


# ── pre-gate checks ──────────────────────────────────────────────────────────────────────────────────

def test_clef_temperature_check_follows_the_source_head_config():
    meta = {"general.architecture": "clef", "general.name": "clef-flash", "clef.decision.n_types": 3}
    # no source head config to read: it cannot tell -- says so and asks for --source, never claims a drop
    w = pd.pregate_checks(meta)
    assert len(w) == 1 and "--source" in w[0] and "dropped" not in w[0]
    # Cloudflare's checkpoints declare no temperatures (joint_schema_model.py: plain softmax) -> T=1 IS the reference
    cloudflare = {"hidden_size": 2048, "width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}
    assert pd.pregate_checks(meta, None, cloudflare, True) == []
    # a source that calibrated them (Unsloth FastDecisionModel) and a GGUF that lost them: warned, with the numbers
    tuned = dict(cloudflare, temperatures={"choice": 1.3, "noul": 0.8})
    w = pd.pregate_checks(meta, None, tuned, True)
    assert len(w) == 1 and "1.3" in w[0] and "T=1" in w[0]
    # the GGUF carries them: llama-server applies them itself, nothing to warn about
    meta["clef.decision.temperature.choice"] = 1.3
    assert pd.pregate_checks(meta) == [] and pd.pregate_checks(meta, None, tuned, True) == []
    assert pd.pregate_checks({"general.architecture": "qwen3", "general.name": "Qwen3-8B"}) == []


def test_noncausal_pplx_decider_without_the_causal_flag_warns():
    meta = {"general.architecture": "qwen3", "general.name": "pplx-decider-v1.1"}
    w = pd.pregate_checks(meta, {"attention_mode": "noncausal"})
    assert len(w) == 1 and "qwen3.attention.causal = false" in w[0]
    assert pd.pregate_checks(dict(meta, **{"qwen3.attention.causal": False}), {"attention_mode": "noncausal"}) == []
    assert pd.pregate_checks(meta, {"attention_mode": "causal"}) == []
    assert "--source" in pd.pregate_checks(meta, None)[0]               # cannot tell without the source config


def test_source_config_is_read_from_source_or_beside_the_gguf(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"attention_mode": "noncausal"}))
    assert pd._source_config(str(tmp_path / "m.gguf"), None)["attention_mode"] == "noncausal"
    other = tmp_path / "src"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"attention_mode": "causal"}))
    assert pd._source_config(str(tmp_path / "m.gguf"), str(other))["attention_mode"] == "causal"


def test_gate_attaches_warnings_and_still_runs(monkeypatch):
    monkeypatch.setattr(pd, "pregate", lambda model, source=None: ["dropped temperatures"])
    monkeypatch.setattr(pd, "_gate", lambda *a: {"verdict": "PASS", "rows": [], "reason": "ok"})
    res = pd.gate("m.gguf", 0)
    assert res["verdict"] == "PASS" and res["warnings"] == ["dropped temperatures"]
    monkeypatch.setattr(pd, "pregate", lambda model, source=None: [])
    assert "warnings" not in pd.gate("m.gguf", 0)


# ── decision temperatures: read like llama-server, applied like with like ───────────────────────────────

def test_head_temperatures_shapes():
    assert pd.head_temperatures(None) == {} and pd.head_temperatures({"layers": 4}) == {}
    assert pd.head_temperatures({"temperature": 1.5}) == {"all": 1.5}
    assert pd.head_temperatures({"temperatures": {"choice": 1.2, "noul.2": 0.9, "bad": "x", "zero": 0}}) == {
        "choice": 1.2, "noul.2": 0.9}
    meta = {"general.architecture": "clef-omni", "clef-omni.decision.temperature.choice.3_5": 1.1,
            "clef-omni.decision.type": "clef"}
    assert pd.gguf_temperatures(meta) == {"choice.3_5": 1.1}


def test_temperature_lookup_matches_llama_server_buckets():
    t = {"choice.3_5": 1.4, "choice": 1.2, "noul": 0.7}
    assert pd.temperature_for(t, "choice", 4) == 1.4            # <type>.<bucket> first
    assert pd.temperature_for(t, "choice", 2) == 1.2            # then <type>
    assert pd.temperature_for(t, "noul", 2) == 0.7
    assert pd.temperature_for(t, "score", 5) == 1.0             # nothing -> T=1, as the server
    assert pd.temperature_for({"all": 2.0}, "score", 11) == 2.0
    assert [pd._bucket(n) for n in (2, 3, 5, 6, 10, 11)] == ["2", "3_5", "3_5", "6_10", "6_10", "11"]


def test_apply_temperature_is_softmax_of_logits_over_t():
    import math
    z = [2.0, 0.5, -1.0]
    sm = lambda v: [math.exp(x - max(v)) / sum(math.exp(y - max(v)) for y in v) for x in v]
    got = pd.apply_temperature(sm(z), 1.7)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, sm([x / 1.7 for x in z])))
    d = pd.apply_temperature({"true": 0.8, "false": 0.2}, 0.5)
    assert abs(d["true"] - 0.64 / 0.68) < 1e-12
    assert pd.apply_temperature([0.3, 0.7], 1.0) == [0.3, 0.7]


def test_client_temperatures_never_double_the_servers():
    tuned = {"temperatures": {"choice": 1.3}}
    bare = {"general.architecture": "clef"}
    carried = dict(bare, **{"clef.decision.temperature.choice": 1.3})
    assert pd.client_temperatures(bare, tuned) == {"choice": 1.3}     # GGUF lost them: the board applies them
    assert pd.client_temperatures(carried, tuned) == {}               # server applies them: never twice
    assert pd.client_temperatures(bare, {"layers": 4}) == {}          # Cloudflare: T=1 both sides


def test_board_applies_source_temperatures_to_build_and_reference(monkeypatch, tmp_path):
    import contextlib
    import pollard_bench as B
    (tmp_path / "joint_head_config.json").write_text(json.dumps({"layers": 4, "temperatures": {"choice": 2.0}}))
    seen = []

    @contextlib.contextmanager
    def fake_served(model, ngl, ctx=4096, port=0, extra=()):
        yield "http://" + os.path.basename(model)

    def fake_run(base, post, items=None, temps=None):
        seen.append((base, temps))
        return [{"question": q, "options": o, "answer": a, "first": "", "probs": [0.7] + [0.3 / (len(o) - 1)] * (len(o) - 1),
                 "mass": 1.0} for _, q, o, a in pd.DECISION_SET]

    monkeypatch.setattr(B, "_served", fake_served)
    monkeypatch.setattr(pd, "run", fake_run)
    monkeypatch.setattr(pd, "_meta", lambda m: {"general.architecture": "clef-omni", "clef-omni.decision.type": "clef"})
    res = pd.gate("rung.gguf", "0", ref="ref.gguf", source=str(tmp_path))
    assert seen == [("http://rung.gguf", {"choice": 2.0}), ("http://ref.gguf", {"choice": 2.0})]
    assert res["temperatures"]["build"] == {"choice": 2.0} and "warnings" in res
    assert any("2.0" in w for w in res["warnings"])
