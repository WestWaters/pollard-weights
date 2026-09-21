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
    assert got == {"seed": 0,                       # pinned so a re-gate is comparable
                   "temperature": 0.7, "repeat_penalty": 1.15, "top_k": 40,
                   "repeat_last_n": 256, "top_p": 0.9}


def test_unknown_sampling_flags_are_dropped_not_passed_through():
    """A flag the server does not take would be rejected for the whole request."""
    assert b._sampling_json(["--mirostat", "2", "--temp", "0.5"]) == {"seed": 0,
                                                                      "temperature": 0.5}


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


def test_the_served_helper_has_every_module_it_uses():
    """A NameError here only surfaces on a real box mid-gate, after the model has loaded. Call it
    with a binary that cannot exist: it must fail by yielding None, not by raising."""
    got = []
    with b._served("/no/such/model.gguf", 0, ctx=256) as base:
        got.append(base)
    assert got == [None], "the server context manager should yield None when it cannot start"


def test_an_explicit_gate_budget_is_the_one_reported():
    """The printed budget used to be read before --gate-tokens was applied, so the log announced
    a number the run did not use."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_bench.py"), encoding="utf-8").read()
    fn = src[src.index("def coherence_gate("):src.index("def _gate_chat(")]
    i_override = fn.index("budget = gate_tokens")
    i_print = fn.index("gate budget {budget}")
    assert i_override < i_print, "the budget is printed before the override is applied"


def test_the_servers_own_output_is_kept_not_discarded():
    """When the server dies mid-gate its output is the only account of why. Sent to DEVNULL, the
    whole diagnosis was "connection forcibly closed by the remote host"."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_bench.py"), encoding="utf-8").read()
    fn = src[src.index("def _served("):src.index("def _post(")]
    assert "subprocess.DEVNULL" not in fn, "the server's output is still being thrown away"
    assert "_SERVER_LOG" in fn


def test_a_dead_server_is_reported_as_such():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_bench.py"), encoding="utf-8").read()
    fn = src[src.index("def _chat("):src.index("#: the CLI sampling lists")]
    assert "proc.poll() is not None" in fn, "a socket error cannot tell a crash from a hiccup"
    assert "exited" in fn


def test_stopping_without_answering_is_not_called_a_loop():
    """finish_reason=stop with empty content fell through to a generic branch and was scored as a
    failure. With reasoning present it is a budget problem; without, the build produced nothing."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_bench.py"), encoding="utf-8").read()
    fn = src[src.index("def _gate_chat("):src.index("def _gate_raw(")]
    assert 'fin == "stop" and not body' in fn, "an empty answer is still handled generically"
    assert "INCONCLUSIVE" in fn and "EMPTY" in fn


# -- the server is launched so it cannot take the gate down ---------------------------------------

def _bench_src():
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "tools", "pollard_bench.py"), encoding="utf-8").read()


def test_the_server_runs_on_one_slot():
    """Upstream's default is auto, which makes FOUR slots sharing one KV pool while telling each
    it owns the whole context. They exhaust it, llama_decode returns 1, and the throw is uncaught
    on the main thread -- the process aborts and the client sees a reset socket."""
    fn = _bench_src()
    fn = fn[fn.index("def _served("):fn.index("def _server_supports") if "def _server_supports" in fn
            else fn.index("def _post(")]
    src = _bench_src()
    blk = src[src.index("cmd = [binary,"):src.index("proc, log = None, None")]
    assert '"-np", "1"' in blk, "the gate can be killed by slot KV exhaustion"
    assert '"--no-context-shift"' in blk, "ik defaults context-shift ON and truncates the prompt"
    assert '"--no-cont-batching"' in blk


def test_an_unknown_flag_is_probed_not_assumed():
    """Passing a flag a build does not know is a hard startup failure, and the forks disagree."""
    src = _bench_src()
    assert "def _server_supports" in src
    assert "--skip-chat-parsing" in src


def test_requests_disable_the_prompt_cache():
    """Documented as a source of nondeterministic results, and the path by which finished prompts
    accumulate until the KV pool is exhausted."""
    src = _bench_src()
    fn = src[src.index("def _chat("):src.index("def _read_chat(")]
    assert '"cache_prompt": False' in fn
    assert '"reasoning_format": "none"' in fn


def test_the_seed_is_pinned():
    import pollard_bench as B
    assert B._sampling_json(["--temp", "0.7"])["seed"] == 0


def test_a_socket_error_is_retried_only_while_the_server_lives():
    """An idle connection closed by the server looks identical to a crash from the client side."""
    src = _bench_src()
    fn = src[src.index("def _chat("):src.index("def _read_chat(")]
    assert "proc.poll() is None" in fn, "a dead server would be retried pointlessly"
    assert "proc.poll() is not None" in fn, "a crash is not distinguished from a hiccup"


def test_connections_are_not_pooled():
    """The server closes an idle connection after 5 seconds; a pooled client races that close."""
    src = _bench_src()
    assert '"Connection": "close"' in src


def test_the_server_is_checked_before_a_run_is_spent_on_it():
    src = _bench_src()
    assert "def _check_server" in src
    assert "total_slots" in src and "n_ctx" in src


def test_reasoning_is_split_out_even_when_the_server_does_not():
    """With a pure content parser the thinking stays inline; an answer that exists must not read
    as an empty one."""
    import pollard_bench as B
    got = B._read_chat({"choices": [{"message": {
        "content": "<think>weighing it up</think>Jupiter."}, "finish_reason": "stop"}]})
    assert got[0] == "Jupiter."
    assert "weighing it up" in got[1]


def test_a_server_that_files_reasoning_separately_still_works():
    import pollard_bench as B
    got = B._read_chat({"choices": [{"message": {
        "content": "Jupiter.", "reasoning_content": "weighing it up"}, "finish_reason": "stop"}]})
    assert got == ("Jupiter.", "weighing it up", "stop")


# -- a verdict from the wrong method is worse than no verdict -------------------------------------

def test_an_instruct_model_without_a_server_is_ungated_not_failed(monkeypatch, tmp_path):
    """This is the bug that cost a whole rebuild. llama-server could not be found, the gate
    quietly fell back to raw completion, and an instruct model -- whose stop token only exists
    inside a chat turn -- was reported BELOW FLOOR. The build was fine."""
    monkeypatch.setattr(b, "has_chat_template", lambda m: "{{ messages }}")
    monkeypatch.setattr(b, "_served", lambda *a, **k: __import__("contextlib").nullcontext(None))
    res = b.coherence_gate("llama-cli", "m.gguf", 0)
    assert res["verdict"] == "NO_SERVER", f"got {res['verdict']} -- a verdict it could not earn"
    assert "llama-server" in res["rows"][0]["reason"]


def test_a_base_model_without_a_server_still_uses_raw_completion(monkeypatch):
    """Raw completion is CORRECT for a base model -- that is what it is."""
    monkeypatch.setattr(b, "has_chat_template", lambda m: "")
    called = {}
    def _raw(*a, **k):
        called["raw"] = True
        return {"verdict": "PASS", "config": "x", "rows": []}
    monkeypatch.setattr(b, "_gate_raw", _raw)
    res = b.coherence_gate("llama-cli", "m.gguf", 0)
    assert called.get("raw") and res["verdict"] == "PASS"


def test_the_ungated_verdict_is_not_reported_as_a_pass():
    res = {"verdict": "NO_SERVER", "config": None,
           "rows": [{"prompt": "x", "loop": None, "reason": "no server", "sample": ""}]}
    assert b.print_gate(res) is False


def test_the_server_is_searched_for_not_guessed():
    """find_llama_bin covers a llama.cpp checkout and PATH; the trellis atoms need ik_llama,
    which people build wherever they like."""
    src = _bench_src()
    fn = src[src.index("def _server_bin("):src.index("@contextlib.contextmanager")]
    assert "ik_llama.cpp/build/bin" in fn, "ik builds are not looked for"
    assert "SERVER_BIN" in fn, "the caller cannot say where it is"


def test_automap_tells_the_gate_where_the_server_is():
    """automap already resolved the binaries; not passing the server is what made the gate fall
    back to the wrong scoring method."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "pollard_automap.py"), encoding="utf-8").read()
    seg = src[src.index('steps.append(("coherence gate"'):]
    seg = seg[:seg.index("]))") + 3]
    assert "--llama-server" in seg and "--llama-cli" in seg
