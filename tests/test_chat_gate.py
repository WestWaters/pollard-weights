"""The coherence gate must run a model the way it is actually used.

An instruct model's stop token lives inside a chat turn -- Qwen's EOS is <|im_end|>, which the
template emits and a raw prompt never can. Scored untemplated, such a model has no reachable way
to stop: it runs to the token budget and repeats, at ANY bit width. The old gate did exactly that
and reported BELOW FLOOR, condemning builds that were fine and sending people off to spend size
bumping tiers to escape a harness bug.

Measured on Qwen3.8-27B-Pollard-FLAGSHIP (1.93 bpw): untemplated it looped; templated it answered
correctly, reasoned in a proper <think> block and closed it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import pollard_bench as b  # noqa: E402


# ── the repetition measure ──────────────────────────────────────────────────────────────────────

def test_a_real_loop_scores_high():
    assert b.tail_repeat("the sky is blue. " * 20) > 0.8


def test_prose_scores_zero():
    assert b.tail_repeat(
        "Jupiter is the largest planet in our solar system, about eleven times the diameter of "
        "Earth and composed mostly of hydrogen and helium with a faint ring system.") < 0.1


def test_correct_code_is_not_a_loop():
    """THE false positive that failed the flagship: a correct fib() scored 0.24 on distinct-word
    ratio and was called a phrase loop. Code reuses tokens by construction."""
    code = ("def fib(n):\n    if n == 0:\n        return 0\n    if n == 1:\n        return 1\n"
            "    return fib(n-1) + fib(n-2)")
    assert b.tail_repeat(code) < 0.25


def test_a_table_is_not_a_loop():
    tbl = "\n".join(f"| row {i} | value {i} | note {i} |" for i in range(12))
    assert b.tail_repeat(tbl) < 0.35


def test_cjk_is_scored_on_tokens_not_whitespace():
    """Splitting on spaces makes one Chinese sentence a single "word", so the distinct-word ratio
    collapsed to noise and flagged fluent output as a loop."""
    cjk = "太阳系中最大的行星是木星。它的直径约为地球的十一倍，主要由氢和氦组成。"
    assert b.tail_repeat(cjk) < 0.4


def test_repetition_is_measured_on_the_tail():
    """A loop is a tail phenomenon; a long correct answer must not hide one at the end."""
    body = "Jupiter is the largest planet in our solar system. " * 2
    assert b.tail_repeat(body + " and a and a and a and a and a and a and a and a and a") > 0.4


def test_short_output_is_not_judged():
    assert b.tail_repeat("Jupiter.") == 0.0


# ── the branch ──────────────────────────────────────────────────────────────────────────────────

def test_a_missing_model_reports_no_template_rather_than_raising():
    assert b.has_chat_template("/does/not/exist.gguf") == ""


def test_the_gate_branches_on_the_template(monkeypatch):
    """A model WITH a template is served and scored on its own turn; one without is raw
    completion. Using one path for both is the bug."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_bench.py"), encoding="utf-8").read()
    fn = src[src.index("def coherence_gate("):src.index("def _gate_chat(")]
    assert "has_chat_template" in fn, "the gate does not consult the template"
    assert "_gate_raw" in fn and "_served" in fn, "both paths must exist"


def test_the_served_gate_asks_for_jinja_and_separates_reasoning():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_bench.py"), encoding="utf-8").read()
    fn = src[src.index("def _served("):src.index("def _post(")]
    assert '"--jinja"' in fn, "without --jinja the server may not apply the model's own template"
    assert '"--reasoning-format", "deepseek"' in fn, \
        "thoughts must land in reasoning_content, or a <think> block reads as an incoherent answer"


def test_sampling_translates_to_the_server_payload():
    got = b._sampling_json(["--temp", "0.7", "--repeat-penalty", "1.15", "--top-k", "40",
                            "--repeat-last-n", "256", "--top-p", "0.9"])
    assert got == {"temperature": 0.7, "repeat_penalty": 1.15, "top_k": 40,
                   "repeat_last_n": 256, "top_p": 0.9}


def test_unknown_sampling_flags_are_dropped_not_passed_through():
    """A flag the server does not take would be rejected for the whole request."""
    assert b._sampling_json(["--mirostat", "2", "--temp", "0.5"]) == {"temperature": 0.5}


# -- taskeval: the suite that answers a competing release ----------------------------------------

def _taskeval_src():
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "tools", "pollard_taskeval.py"), encoding="utf-8").read()


def test_an_instruct_model_is_scored_through_a_chat_turn():
    """lm-eval's `gguf` backend posts to /v1/completions -- the RAW endpoint. Scored that way an
    instruct model never reaches its end-of-turn token, runs past the answer, and the harness
    scrapes a stop string out of the overrun. Every generative task in the suite loses points to
    that, while the release being answered quoted numbers measured WITH their template."""
    src = _taskeval_src()
    assert "local-chat-completions" in src, "still scoring through the raw completions endpoint"
    assert "/v1/chat/completions" in src
    assert "--apply_chat_template" in src


def test_a_base_model_still_uses_the_raw_endpoint():
    """A base model has no template; forcing one on it would be the same bug in reverse."""
    src = _taskeval_src()
    seg = src[src.index("if serve and harness"):src.index("    if r.returncode != 0:")]
    assert "chat_model(path)" in seg, "the backend is not chosen from the model"
    assert '_invoke("gguf"' in seg, "the raw path must remain for base models"


def test_the_served_model_applies_its_own_template():
    src = _taskeval_src()
    seg = src[src.index("def served("):src.index("def run(")]
    assert '"--jinja"' in seg, "llama-server will not apply the model's template without --jinja"
    assert '"--reasoning-format", "deepseek"' in seg, \
        "a <think> block would otherwise be parsed as the answer"


def test_the_bonsai_suite_is_unchanged():
    """The comparison only means something if the task list still matches theirs, category for
    category. The fix is HOW it is scored, not WHAT is scored."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "tools"))
    import pollard_taskeval as t
    assert t.SUITES["bonsai"] == [
        "mmlu_redux_generative", "leaderboard_musr",
        "gsm8k", "minerva_math500", "aime25", "aime26",
        "humaneval_plus", "mbpp_plus",
        "ifeval"]
