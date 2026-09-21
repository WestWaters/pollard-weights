"""Tool-call formatting: the capability that breaks first and is cheapest to protect.

Standard benchmarks call 4-bit near-lossless, and that rests on single-turn scoring. The skill
that actually goes is narrow -- emitting a well-formed call with the right name and argument types
-- and it lives in a small band of layers, so a drop here is usually a few tensors rather than the
whole model.

Full agentic evaluation needs a live environment (pollard-taskeval --suite agentic says so). This
grades the part that does not, and these tests grade the grader.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("ptc", ROOT / "tools/pollard_toolcall.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ptc"] = m
    spec.loader.exec_module(m)
    return m


TC = _load()
TOOL = TC.PROBES[0]["tool"]
EXP = TC.PROBES[0]["expect"]


def grade(text):
    return TC.check_call(text, TOOL, EXP)


VALID = '{"name":"get_weather","arguments":{"location":"Paris"}}'


def test_a_plain_call_passes():
    assert grade(VALID)[0] is True


@pytest.mark.parametrize("wrapped", [
    '```json\n' + VALID + '\n```',
    '<tool_call>' + VALID + '</tool_call>',
    'Sure, here you go:\n' + VALID,
    VALID + '\n\nLet me know if you need anything else.',
])
def test_the_wrapper_is_not_the_capability(wrapped):
    """Fences, tags and surrounding prose are style. The object is what is under test."""
    assert grade(wrapped)[0] is True


def test_double_encoded_arguments_are_accepted():
    """Several models emit arguments as a JSON string. Ugly, not broken."""
    assert grade('{"name":"get_weather","arguments":"{\\"location\\": \\"Paris\\"}"}')[0] is True


def test_prose_instead_of_a_call_is_no_call_not_not_json():
    """These want different fixes, so they must not collapse into one failure."""
    assert grade("The weather in Paris is sunny.")[1] == "no-call"


def test_call_shaped_but_unparseable_is_not_json():
    assert grade('{"name": "get_weather", "arguments": {location: Paris')[1] == "not-json"


def test_the_wrong_function_is_caught():
    assert grade('{"name":"lookup","arguments":{"location":"Paris"}}')[1] == "wrong-tool"


def test_a_missing_required_argument_is_caught():
    assert grade('{"name":"get_weather","arguments":{}}')[1] == "missing-arg"


def test_an_optional_argument_may_be_absent():
    assert grade('{"name":"get_weather","arguments":{"location":"Paris"}}')[0] is True


def test_an_argument_outside_the_schema_is_caught():
    assert grade('{"name":"get_weather","arguments":{"location":"Paris","zoom":3}}')[1] == "hallucinated"


def test_a_wrong_type_is_caught():
    assert grade('{"name":"get_weather","arguments":{"location":42}}')[1] == "bad-type"


def test_a_number_sent_as_a_string_is_its_own_failure():
    """It breaks a strict caller but is a formatting slip, not a wrong answer."""
    tool = TC.PROBES[1]["tool"]
    ok, failure, detail = TC.check_call('{"name":"calculate","arguments":{"a":"17","b":25}}', tool)
    assert failure == "bad-type" and "string" in detail


def test_the_wrong_value_is_distinguished_from_the_wrong_shape():
    ok, failure, _ = grade('{"name":"get_weather","arguments":{"location":"Berlin"}}')
    assert failure == "wrong-value", "a well-formed call with the wrong content is not a format failure"


def test_scoring_reports_which_failure_dominates():
    rows = [{"ok": True, "failure": None}, {"ok": False, "failure": "no-call"},
            {"ok": False, "failure": "no-call"}, {"ok": False, "failure": "bad-type"}]
    s = TC.score(rows)
    assert s["rate"] == 0.25 and s["modes"]["no-call"] == 2


def test_every_probe_has_a_required_argument():
    """A schema with nothing required cannot detect a missing-argument failure."""
    for p in TC.PROBES:
        assert any(m.get("required") for m in p["tool"]["parameters"].values()), p["tool"]["name"]


def test_generation_is_deterministic():
    """Format validity must not move between runs because sampling moved."""
    src = (ROOT / "tools/pollard_toolcall.py").read_text()
    assert '"--temp", "0"' in src


def test_it_points_at_the_narrow_band_rather_than_raising_bits_everywhere():
    src = (ROOT / "tools/pollard_toolcall.py").read_text()
    assert "pollard-sensitivity" in src and "narrow band" in src


def test_the_tool_stays_ascii():
    assert all(ord(c) < 128 for c in (ROOT / "tools/pollard_toolcall.py").read_text())
