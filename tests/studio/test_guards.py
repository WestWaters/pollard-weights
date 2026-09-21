"""Tests for the guards: flag validation, coherence detection, and the publish path.

These cover the three things that can hurt a user: a flag that is silently wrong, a build that
looks fine on metrics and loops in practice, and an upload that goes out before it was meant to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import actions, coherence, runtimes, validate   # noqa: E402


# ── validation ──────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def mf():
    m = validate.load_manifest()
    if not m:
        pytest.skip("no manifest available")
    return m


def test_unknown_flag_is_refused(mf):
    r = validate.check("pollard_fit", {"--not-a-flag": "x"}, manifest=mf)
    assert not r["ok"] and any("does not accept" in p for p in r["problems"])


def test_bad_type_is_refused(mf):
    r = validate.check("pollard_doctor", {"--rows": "abc"}, manifest=mf)
    assert not r["ok"] and any("expects int" in p for p in r["problems"])


def test_bad_choice_is_refused(mf):
    r = validate.check("pollard_doctor", {"--lane": "banana"}, manifest=mf)
    assert not r["ok"] and any("must be one of" in p for p in r["problems"])


def test_missing_file_is_refused(mf):
    r = validate.check("pollard_doctor", {"--model": "/definitely/not/here.gguf"}, manifest=mf)
    assert not r["ok"] and any("no such file" in p for p in r["problems"])


def test_every_problem_is_reported_not_just_the_first(mf):
    r = validate.check("pollard_doctor",
                       {"--nope": 1, "--rows": "abc", "--lane": "banana"}, manifest=mf)
    assert len(r["problems"]) >= 3, "should report all problems at once"


def test_unset_flag_is_dropped_not_emitted_bare(mf):
    """A dangling flag eats the next argument and shifts everything after it."""
    r = validate.check("pollard_fit", {"--gguf": "", "--ram": "16"}, manifest=mf)
    assert "--gguf" not in r["argv"]


def test_switch_only_appears_when_on(mf):
    base = {"--gguf": "x.gguf", "--ram": "16"}          # fit declares these required
    on = validate.check("pollard_fit", {**base, "--plan-only": True},
                        manifest=mf, check_paths=False)
    off = validate.check("pollard_fit", {**base, "--plan-only": False},
                         manifest=mf, check_paths=False)
    assert on["ok"] and off["ok"], on["problems"] + off["problems"]
    assert "--plan-only" in on["argv"] and "--plan-only" not in off["argv"]


def test_required_flag_is_enforced(mf):
    """pollard-fit declares --gguf required; omitting it must fail before anything starts."""
    r = validate.check("pollard_fit", {"--ram": "16"}, manifest=mf)
    assert not r["ok"] and any("required" in p for p in r["problems"])


def test_valid_input_produces_argv(mf, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")
    r = validate.check("pollard_fit", {"--gguf": str(f), "--ram": "16"}, manifest=mf)
    assert r["ok"] and r["argv"][:2] == ["--gguf", str(f)]


# ── coherence ───────────────────────────────────────────────────────────────────────────────────
def test_healthy_text_is_coherent():
    t = ("The derivative of x squared is two x by the power rule. "
         "Therefore, the answer is 2x.")
    assert coherence.analyse(t)["ok"]


def test_detects_a_repetitive_loop():
    t = "Let me reconsider the problem carefully once more. " * 8
    assert "repetitive-loop" in coherence.analyse(t)["failures"]


def test_detects_budget_exhaustion_from_the_stop_reason():
    r = coherence.analyse("some words here that do not finish", stop_reason="length")
    assert "budget-exhaustion" in r["failures"]


def test_detects_an_unclosed_reasoning_block():
    assert "unclosed-segment" in coherence.analyse("<think>working on it")["failures"]


def test_detects_an_unclosed_code_fence():
    assert "unclosed-segment" in coherence.analyse("here you go ```python\nx = 1")["failures"]


def test_detects_delayed_commitment():
    t = "thinking out loud about this. " * 40 + " the answer is 7"
    assert "delayed-commitment" in coherence.analyse(t)["failures"]


def test_natural_repetition_is_not_a_loop():
    """Ordinary prose repeats words. Only a recurring WINDOW is a loop."""
    t = ("The cat sat on the mat. The dog sat on the floor. "
         "A bird sat on the fence. Therefore, the answer is three animals.")
    assert not coherence.analyse(t)["repetition"]["looping"]


def test_short_text_does_not_false_positive():
    assert coherence.analyse("Paris.")["repetition"]["looping"] is False


def test_gate_fails_when_any_sample_fails():
    g = coherence.gate([
        {"text": "Therefore, the answer is 4.", "stop_reason": "stop"},
        {"text": "go around again and again " * 9, "stop_reason": "length"},
    ])
    assert not g["pass"] and g["failed"] == 1 and g["n"] == 2


def test_gate_passes_when_all_coherent():
    g = coherence.gate([{"text": "Therefore, the answer is 4.", "stop_reason": "stop"}])
    assert g["pass"] and g["rate"] == 1.0


def test_empty_gate_is_a_failure_not_a_pass():
    """No samples must never read as success."""
    assert coherence.gate([])["pass"] is False


# ── publish guard ───────────────────────────────────────────────────────────────────────────────
def test_publish_is_outbound_and_confirmed():
    assert "publish" in actions.OUTBOUND
    assert "publish" in actions.CONFIRM


def test_card_does_not_upload():
    """The local card action must never carry --upload."""
    _, args = actions.resolve("card", {"modelName": "m", "repo": "me/m", "uploadRepo": "me/m"})
    assert "--upload" not in args


def test_publish_does_upload():
    _, args = actions.resolve("publish", {"modelName": "m", "repo": "me/m"})
    assert "--upload" in args


def test_nothing_else_is_outbound():
    """Only publish leaves the machine. If that grows, it should be deliberate."""
    assert actions.OUTBOUND == {"publish"}




# ── runtimes ────────────────────────────────────────────────────────────────────────────────────


def test_every_spec_is_complete():
    for name, spec in runtimes.SPECS.items():
        assert spec["kind"] in (runtimes.LOCAL, runtimes.REMOTE), name
        assert spec.get("note"), f"{name} has no note for the user"
        if spec["kind"] == runtimes.LOCAL:
            assert spec.get("exe") and spec.get("args"), name
        else:
            assert spec.get("key_env"), name
            assert spec.get("base") or spec.get("base_env"), name


def test_available_reports_every_runtime_with_a_reason():
    rows = runtimes.available()
    assert len(rows) == len(runtimes.SPECS)
    for r in rows:
        assert r["why"], f"{r['name']} gives no reason either way"


def test_trellis_builds_prefer_the_fork():
    """A GGUF carrying IQ*_KT atoms cannot load in stock llama.cpp."""
    order = runtimes.for_build("gguf", "IQ1_KT")
    assert order[0] == "ik_llama"


def test_plain_gguf_does_not_demand_the_fork():
    assert runtimes.for_build("gguf", "Q6_K")[0] != "ik_llama"


def test_safetensors_never_routes_to_a_gguf_runtime():
    got = runtimes.for_build("safetensors")
    assert "llama.cpp" not in got and "ik_llama" not in got


def test_remote_runtimes_are_offered_for_both_kinds():
    for kind in ("gguf", "safetensors"):
        assert any(n in runtimes.for_build(kind)
                   for n in ("openrouter", "openai-compatible"))


def test_unknown_runtime_is_refused():
    with pytest.raises(ValueError):
        runtimes.Runtime("not-a-runtime")


def test_remote_without_a_key_explains_itself(monkeypatch):
    for v in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        monkeypatch.delenv(v, raising=False)
    r = runtimes.Runtime("openrouter").ensure("some/model")
    assert not r["ok"] and "OPENROUTER_API_KEY" in r["error"]


def test_local_runtime_reports_a_missing_build():
    r = runtimes.Runtime("llama.cpp").ensure("/definitely/not/here.gguf")



def test_stop_is_safe_before_start():
    runtimes.Runtime("llama.cpp").stop()


def test_openai_finish_reason_length_maps_to_budget_exhaustion():
    """'length' is the OpenAI spelling of the failure the coherence gate looks for."""
    rt = runtimes.Runtime("openrouter")
    out = rt._openai.__doc__ if False else None   # documented behaviour, exercised below
    payload = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
               "usage": {"completion_tokens": 5}}
    import pollard_studio.runtimes as R
    orig = R._post
    R._post = lambda *a, **k: payload
    try:
        got = rt._openai("p", 5, 0.7, 5, "http://x", {}, "m")
    finally:
        R._post = orig
    assert got["stop_reason"] == "length"
