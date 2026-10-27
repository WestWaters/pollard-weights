"""pollard-toolcall's full gate: the real serving path (llama-server --jinja, OpenAI `tools`, the server's own parser).

These grade the grader and the plumbing without a model: the probe suite's shape, the schema validator, the argument
matchers, the reply grader on the replies llama-server actually returns (parsed `tool_calls`, prose, unparsed call
text, think blocks), Wilson / paired-bootstrap statistics, the verdict, and -- against a fake OpenAI endpoint -- the
request each probe sends and the results.json the gate writes.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location("ptc_served", ROOT / "tools/pollard_toolcall.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ptc_served"] = m
    spec.loader.exec_module(m)
    return m


TC = _load()
PROBE = {p["id"]: p for p in TC.SUITE}


def reply(calls=None, content="", finish="stop"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"type": "function", "id": "x", "function": {"name": n, "arguments": json.dumps(a)
                                                                           if not isinstance(a, str) else a}}
                             for n, a in calls]
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else finish}]}


# ------------------------------------------------------------------ the suite

def test_sixty_probes_fifteen_per_category():
    assert len(TC.SUITE) == 60
    assert Counter(p["category"] for p in TC.SUITE) == {c: 15 for c in TC.CATEGORIES}
    assert len({p["id"] for p in TC.SUITE}) == 60


def test_every_expectation_is_answerable_from_its_schema():
    """An expected tool must be on offer, and every expected argument must be one its schema defines."""
    for p in TC.SUITE:
        assert all(t in TC.TOOLBOX for t in p["tools"]), p["id"]
        if p["expect"] is None:
            continue
        assert p["expect"]["name"] in p["tools"], p["id"]
        props = TC.TOOLBOX[p["expect"]["name"]]["function"]["parameters"]["properties"]
        assert set(p["expect"]["args"]) <= set(props), p["id"]


def test_choose_probes_offer_many_tools_and_single_offers_one():
    for p in TC.SUITE:
        if p["category"] == "choose":
            assert len(p["tools"]) >= 6
        if p["category"] == "single":
            assert len(p["tools"]) == 1


def test_nested_probes_exercise_objects_and_arrays():
    kinds = set()
    for p in TC.SUITE:
        if p["category"] != "nested":
            continue
        for v in p["expect"]["args"].values():
            kinds.add("object" if isinstance(v, dict) and not ({"set", "has"} & set(v)) else
                      "array" if isinstance(v, dict) else "scalar")
    assert {"object", "array"} <= kinds


def test_balanced_limit_keeps_every_category():
    sub = TC._balanced(2)
    assert len(sub) == 8 and Counter(p["category"] for p in sub) == {c: 2 for c in TC.CATEGORIES}


# ------------------------------------------------------------------ schema + matchers

USER = TC.TOOLBOX["create_user"]["function"]["parameters"]


@pytest.mark.parametrize("args,failure", [
    ({"user": {"name": "a", "email": "b"}}, None),
    ({}, "missing-arg"),
    ({"user": {"name": "a"}}, "missing-arg"),                             # nested required
    ({"user": {"name": "a", "email": "b"}, "extra": 1}, "hallucinated"),
    ({"user": {"name": "a", "email": "b", "age": "36"}}, "bad-type"),      # integer as a string
    ({"user": {"name": "a", "email": "b", "age": True}}, "bad-type"),      # a bool is not an integer
    ({"user": {"name": "a", "email": "b"}, "roles": ["root"]}, "bad-type"),  # array item outside its enum
    ({"user": "a"}, "bad-type"),
])
def test_validate_args(args, failure):
    ok, fail, _ = TC.validate_args(args, USER)
    assert fail == failure and ok is (failure is None)


def test_enum_is_enforced():
    p = TC.TOOLBOX["create_ticket"]["function"]["parameters"]
    assert TC.validate_args({"title": "x", "priority": "critical"}, p)[1] == "bad-enum"


@pytest.mark.parametrize("got,want,ok", [
    (25, 25, True), (25.0, 25, True), ("25", 25, False), (True, 1, False),
    (True, True, True), (1, True, False),
    ("Tokyo, Japan", "~tokyo", True), ("es", "~spanish|es", True), ("French", "~spanish|es", False),
    ("Celsius", "=celsius", True), ("celsius degrees", "=celsius", False),
    (["Admin", "editor"], {"set": ["admin", "editor"]}, True), (["admin"], {"set": ["admin", "editor"]}, False),
    (["a@x", "b@x"], {"has": ["a@x"]}, True),
    ({"name": "Ada L", "email": "ada@example.com", "age": 36},
     {"name": "~ada", "email": "=ada@example.com", "age": 36}, True),
])
def test_match(got, want, ok):
    assert TC._match(got, want) is ok


# ------------------------------------------------------------------ grading the server's reply

def test_a_parsed_exact_call_passes():
    r = TC.grade_reply(reply([("get_weather", {"city": "Tokyo", "unit": "celsius"})]), PROBE["s01"])
    assert r["ok"] and r["parsed"] and r["name_ok"] and r["args_valid"] and r["exact"]


def test_wrong_value_is_not_a_format_failure():
    r = TC.grade_reply(reply([("get_weather", {"city": "Osaka", "unit": "celsius"})]), PROBE["s01"])
    assert r["failure"] == "wrong-value" and r["args_valid"] and not r["exact"]


def test_wrong_tool_vs_hallucinated_tool():
    assert TC.grade_reply(reply([("search_web", {"query": "x"})]), PROBE["c01"])["failure"] == "wrong-tool"
    assert TC.grade_reply(reply([("teleport", {})]), PROBE["c01"])["failure"] == "hallucinated-tool"


def test_unparsed_call_text_is_its_own_failure():
    """The model wrote a call but the server's parser did not accept it: a FORMAT failure, not prose."""
    txt = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"\n</tool_call>'
    assert TC.grade_reply(reply(content=txt), PROBE["s01"])["failure"] == "unparsed"
    assert TC.grade_reply(reply(content="It is sunny in Tokyo."), PROBE["s01"])["failure"] == "no-call"


def test_arguments_that_are_not_json():
    assert TC.grade_reply(reply([("get_weather", "{city: Tokyo")]), PROBE["s01"])["failure"] == "bad-args"


def test_abstention():
    assert TC.grade_reply(reply(content="Paris."), PROBE["a01"])["ok"]
    r = TC.grade_reply(reply([("lookup_contact", {"name": "France"})]), PROBE["a01"])
    assert r["failure"] == "false-call" and not r["ok"]


def test_thinking_is_kept_out_of_the_grade():
    """A JSON sketch inside a think block is not a call, and a block that never closed is a budget failure."""
    sketch = '<think>maybe {"name": "get_weather", "arguments": {}}</think>The weather is unknown.'
    assert TC.grade_reply(reply(content=sketch), PROBE["s01"])["failure"] == "no-call"
    assert TC.grade_reply(reply(content="<think>let me consider", finish="length"),
                          PROBE["s01"])["failure"] == "unclosed-think"


def test_server_error_is_recorded():
    assert TC.grade_reply({"error": "parse failure"}, PROBE["s01"])["failure"] == "server-error"


# ------------------------------------------------------------------ statistics

def test_wilson_interval():
    lo, hi = TC.wilson(30, 60)
    assert lo < 0.5 < hi and 0.37 < lo < 0.38 and 0.62 < hi < 0.63
    assert TC.wilson(0, 0) == (0.0, 0.0)
    assert TC.wilson(60, 60)[1] > 0.9999 and TC.wilson(60, 60)[0] > 0.9


def _rows(flags):
    return [{"id": f"p{i}", "ok": f} for i, f in enumerate(flags)]


def test_retention_is_paired():
    ref = _rows([True] * 40 + [False] * 20)
    same = TC.retention_ci(_rows([True] * 40 + [False] * 20), ref)
    assert same[0] == 1.0 and same[1] == 1.0 and same[2] == 1.0       # identical outcomes: no uncertainty
    lost = TC.retention_ci(_rows([True] * 30 + [False] * 30), ref)
    assert lost[0] == 0.75 and lost[1] < 0.75 < lost[2]
    assert TC.retention_ci(_rows([True] * 5), _rows([False] * 5)) == (None, None, None)


def test_score_suite_components():
    rows = [TC.grade_reply(reply([("get_weather", {"city": "Tokyo", "unit": "celsius"})]), PROBE["s01"]),
            TC.grade_reply(reply([("get_weather", {"city": "Osaka"})]), PROBE["s01"]),
            TC.grade_reply(reply(content="no"), PROBE["s01"]),
            TC.grade_reply(reply(content="Paris."), PROBE["a01"])]
    s = TC.score_suite(rows)
    assert s["n"] == 4 and s["valid"] == 2 and s["rate"] == 0.5
    assert s["parse_rate"] == round(2 / 3, 4) and s["exact"] == round(1 / 3, 4) and s["abstention"] == 1.0
    assert s["modes"] == {"wrong-value": 1, "no-call": 1}
    assert s["categories"] == {"single": round(1 / 3, 4), "abstain": 1.0}


def test_verdict_is_retention_with_a_reference():
    s = {"rate": 0.4, "valid": 24}
    rs = {"rate": 0.42, "valid": 25}
    assert TC.verdict_for(s, rs, 0.96, min_retention=90) is True        # a weak model is not failed for being weak
    assert TC.verdict_for(s, rs, 0.80, min_retention=90) is False
    assert TC.verdict_for({"rate": 0.8, "valid": 48}) is True
    assert TC.verdict_for({"rate": 0.5, "valid": 30}) is False
    assert TC.verdict_for(s, {"rate": 0.0, "valid": 0}, None) is True   # nothing to retain


# ------------------------------------------------------------------ the request and the plumbing

def test_request_uses_openai_tools_and_greedy_decoding():
    body = TC.request_for(PROBE["c01"], 384)
    assert [t["function"]["name"] for t in body["tools"]] == TC.MENU_A
    assert body["temperature"] == 0 and body["tool_choice"] == "auto" and body["max_tokens"] == 384
    assert body["messages"][-1] == {"role": "user", "content": PROBE["c01"]["prompt"]}
    assert "chat_template_kwargs" not in body
    assert TC.request_for(PROBE["c01"], 9, {"enable_thinking": False})["chat_template_kwargs"] == \
        {"enable_thinking": False}


def test_the_server_runs_the_models_own_template():
    src = (ROOT / "tools/pollard_toolcall.py").read_text()
    assert '"--jinja"' in src and "--skip-chat-parsing" not in src.split("def served")[1].split("def _post")[0]


class _Fake(http.server.BaseHTTPRequestHandler):
    """An OpenAI endpoint that answers like a perfect Hermes model behind llama.cpp's parser."""
    seen = []

    def do_POST(self):                                                    # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append(body)
        p = next(p for p in TC.SUITE if p["prompt"] == body["messages"][-1]["content"])
        if p["expect"] is None:
            out = reply(content="Sure.")
        else:
            args = {k: _concrete(v) for k, v in p["expect"]["args"].items()}
            tool = TC.TOOLBOX[p["expect"]["name"]]["function"]["parameters"]
            for k in tool.get("required") or []:
                args.setdefault(k, 1 if tool["properties"][k].get("type") in ("integer", "number") else "x")
            out = reply([(p["expect"]["name"], args)])
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _concrete(v):
    if isinstance(v, str):
        return v.lstrip("=~").split("|")[0]
    if isinstance(v, dict) and ("set" in v or "has" in v):
        return list(v.get("set", v.get("has")))
    if isinstance(v, dict):
        return {k: _concrete(x) for k, x in v.items()}
    return v


def test_run_suite_against_an_openai_endpoint():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _Fake.seen.clear()
        rows = TC.run_suite(f"http://127.0.0.1:{srv.server_port}", max_tokens=64)
    finally:
        srv.shutdown()
    assert len(_Fake.seen) == 60 and all(b["tools"] for b in _Fake.seen)
    s = TC.score_suite(rows)
    assert s["rate"] == 1.0, [r for r in rows if not r["ok"]]
    assert s["parse_rate"] == s["function_acc"] == s["exact"] == s["abstention"] == 1.0


def test_a_call_the_server_parser_dropped_is_unparsed_not_prose():
    """mlx_lm.server drops call text its parser rejects but still finishes with tool_calls: that is a format failure."""
    row = TC.grade_reply(reply(content="", finish="tool_calls"), PROBE["s01"])
    assert row["failure"] == "unparsed"
    assert TC.grade_reply(reply(content="", finish="stop"), PROBE["s01"])["failure"] == "no-call"


@pytest.mark.parametrize("candidate,reference", [
    ("error", "error"), ("error", "zero"), ("partial", "good"), ("partial", None),
])
def test_server_errors_fail_the_gate_without_recording_a_measurement(
        monkeypatch, tmp_path, capsys, candidate, reference):
    """Exercise the actual CLI, URL transport, grading and persisted gate result."""
    import io
    import urllib.error
    import pollard_verify as verify

    counts = Counter()

    def urlopen(req, **kwargs):
        side = "reference" if req.full_url.startswith("http://reference/") else "candidate"
        counts[side] += 1
        behavior = reference if side == "reference" else candidate
        # "partial": every probe but the last gets no reply -- a majority outage, still not a measurement
        if behavior == "error" or (behavior == "partial" and counts[side] < len(TC._balanced(1))):
            raise urllib.error.URLError("server unavailable")
        if behavior == "zero":
            out = reply([("unknown_tool", {})])
        else:
            body = json.loads(req.data)
            p = next(p for p in TC.SUITE if p["prompt"] == body["messages"][-1]["content"])
            out = reply(content="Sure.") if p["expect"] is None else reply([
                (p["expect"]["name"], {k: _concrete(v) for k, v in p["expect"]["args"].items()})])
        return io.BytesIO(json.dumps(out).encode())

    monkeypatch.setattr(TC.urllib.request, "urlopen", urlopen)
    res, out = tmp_path / "results.json", tmp_path / "rows.json"
    # A failed rerun must also clear a previous passing measurement's numbers.
    res.write_text(json.dumps({"served": {"toolcall": {
        "pass": True, "status": "measured", "retention": 100, "rate": 1, "ref_rate": 1}}}))
    args = ["--backend", "url", "--url", "http://candidate", "--limit", "1",
            "--results", str(res), "--out", str(out)]
    if reference is not None:
        args += ["--ref-backend", "url", "--ref-url", "http://reference"]
    with pytest.raises(SystemExit) as exc:
        TC.main(args)
    assert exc.value.code == 1
    assert "\nPASS\n" not in capsys.readouterr().out
    results = json.loads(res.read_text())
    tc = results["served"]["toolcall"]
    assert tc["pass"] is False and tc["status"] == "error"
    assert "server-error" in tc["reason"]
    assert tc.get("rate") is tc.get("ref_rate") is tc.get("retention") is None
    metrics = verify.metrics_from_results(results, "served")
    assert not verify.evaluate_gate(metrics, {"toolcall_pass": True})["pass"]
    assert not verify.evaluate_gate(metrics, {"toolcall_retention_min": 90})["pass"]
    assert any(r["failure"] == "server-error" for r in json.loads(out.read_text())["rows"] +
               (json.loads(out.read_text())["ref_rows"] or []))


@pytest.mark.parametrize("reference", ["error", "partial"])
def test_a_reference_that_gets_no_reply_leaves_the_rungs_own_numbers(monkeypatch, tmp_path, reference):
    """Nanbeige4.3-33B (2026-10-27): the Q8_0 reference paged on the box until 60/60 probes timed out, and the
    flagship's own 54/60 was recorded as status=error with every number cleared (#193 treated a reference
    outage like a rung outage). The rung was measured; only the comparison was not."""
    import io
    import urllib.error
    counts = Counter()

    def urlopen(req, **kwargs):
        side = "reference" if req.full_url.startswith("http://reference/") else "candidate"
        counts[side] += 1
        if side == "reference" and (reference == "error" or counts[side] < len(TC._balanced(1))):
            raise urllib.error.URLError("timed out")
        body = json.loads(req.data)
        p = next(p for p in TC.SUITE if p["prompt"] == body["messages"][-1]["content"])
        out = reply(content="Sure.") if p["expect"] is None else reply([
            (p["expect"]["name"], {k: _concrete(v) for k, v in p["expect"]["args"].items()})])
        return io.BytesIO(json.dumps(out).encode())

    monkeypatch.setattr(TC.urllib.request, "urlopen", urlopen)
    res = tmp_path / "results.json"
    with pytest.raises(SystemExit) as exc:
        TC.main(["--backend", "url", "--url", "http://candidate", "--ref-backend", "url",
                 "--ref-url", "http://reference", "--limit", "1", "--results", str(res)])
    assert exc.value.code == 0                                   # gated on the rung's absolute rate
    tc = json.loads(res.read_text())["served"]["toolcall"]
    assert tc["status"] == "measured" and tc["pass"] is True
    assert tc["rate"] == 1.0 and tc["exact"] == 1.0              # the rung's numbers, kept
    assert tc.get("ref_rate") is None and tc.get("retention") is None
    assert "reference not measured" in tc["ref_note"]


def test_an_unservable_reference_is_not_retried_on_every_rung(monkeypatch, tmp_path):
    """Each rung of a ladder used to re-serve the same unservable reference (hours of timeouts per rung):
    once it got no reply on this machine it is not measured again until --refresh-ref."""
    cache = tmp_path / "reference.json"
    monkeypatch.setattr(TC, "_ref_cache_path", lambda *a: str(cache))
    error = TC.grade_reply({"error": "server-error: TimeoutError('timed out')"}, PROBE["s01"])
    good = TC.grade_reply(reply([("get_weather", {"city": "Tokyo", "unit": "celsius"})]), PROBE["s01"])
    assert TC.is_transport_failure(error) and "timed out" in str(error.get("detail", "")).lower()
    runs = []
    monkeypatch.setattr(TC, "run_model", lambda *a, **k: (runs.append(1), ([error], {"backend": "gguf"}))[1])
    assert TC.run_ref_cached("reference.gguf", "0")[0] == [error]
    with pytest.raises(TC.NotMeasured, match="not servable on this machine"):
        TC.run_ref_cached("reference.gguf", "0")
    assert len(runs) == 1                                        # the second rung did not serve it again
    monkeypatch.setattr(TC, "run_model", lambda *a, **k: (runs.append(1), ([good], {"backend": "gguf"}))[1])
    assert TC.run_ref_cached("reference.gguf", "0", refresh=True)[0] == [good]
    assert len(runs) == 2


def test_real_zero_capability_completions_still_pass_with_a_reference(monkeypatch, tmp_path):
    import io
    monkeypatch.setattr(TC.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(
        json.dumps(reply([("unknown_tool", {})])).encode()))
    res = tmp_path / "results.json"
    with pytest.raises(SystemExit) as exc:
        TC.main(["--backend", "url", "--url", "http://candidate", "--ref-backend", "url",
                 "--ref-url", "http://reference", "--limit", "1", "--results", str(res)])
    assert exc.value.code == 0
    tc = json.loads(res.read_text())["served"]["toolcall"]
    assert tc["pass"] is True and tc["status"] == "measured"
    assert tc["rate"] == tc["ref_rate"] == 0


@pytest.mark.parametrize("cached_error", [False, True])
def test_reference_cache_recovers_after_server_errors(monkeypatch, tmp_path, cached_error):
    cache = tmp_path / "reference.json"
    monkeypatch.setattr(TC, "_ref_cache_path", lambda *a: str(cache))
    error = TC.grade_reply({"error": "temporary outage"}, PROBE["s01"])
    good = TC.grade_reply(reply([("get_weather", {"city": "Tokyo", "unit": "celsius"})]), PROBE["s01"])
    runs = []
    if cached_error:
        cache.write_text(json.dumps({"rows": [error], "meta": {"backend": "gguf"}}))

    def run(*args, **kwargs):
        runs.append(1)
        return ([good] if cached_error or len(runs) > 1 else [error]), {"backend": "gguf"}

    monkeypatch.setattr(TC, "run_model", run)
    first, _ = TC.run_ref_cached("reference.gguf", "0")
    if cached_error:
        assert first == [good] and len(runs) == 1
    else:
        assert first == [error] and not cache.exists()
        assert TC.run_ref_cached("reference.gguf", "0")[0] == [good]
    count = len(runs)
    assert TC.run_ref_cached("reference.gguf", "0")[0] == [good]
    assert len(runs) == count


def _served(monkeypatch, fail, prose=()):
    """A URL backend answering every probe correctly, except where fail(side, n) raises; a side in `prose` only
    ever answers in prose (fails every call probe)."""
    import io
    seen = Counter()

    def urlopen(req, **kwargs):
        side = "reference" if req.full_url.startswith("http://reference/") else "candidate"
        seen[side] += 1
        fail(side, seen[side])
        body = json.loads(req.data)
        p = next(p for p in TC.SUITE if p["prompt"] == body["messages"][-1]["content"])
        out = reply(content="Sure.") if p["expect"] is None or side in prose else reply([
            (p["expect"]["name"], {k: _concrete(v) for k, v in p["expect"]["args"].items()})])
        return io.BytesIO(json.dumps(out).encode())
    monkeypatch.setattr(TC.urllib.request, "urlopen", urlopen)


def _gate(tmp_path, *extra):
    res = tmp_path / "results.json"
    with pytest.raises(SystemExit) as exc:
        TC.main(["--backend", "url", "--url", "http://candidate", "--ref-backend", "url",
                 "--ref-url", "http://reference", "--results", str(res), *extra])
    return exc.value.code, json.loads(res.read_text())["served"]["toolcall"]


def _parse_500(url):
    import io
    import urllib.error
    return urllib.error.HTTPError(url, 500, "Internal Server Error", {}, io.BytesIO(json.dumps(
        {"error": {"code": 500, "message": "Failed to parse input at pos 12"}}).encode()))


@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_one_parser_500_is_a_failed_probe_not_a_run_error(monkeypatch, tmp_path, side):
    """llama-server answers an unparseable generation with HTTP 500 + body: the model failed that probe."""
    def fail(s, n):
        if s == side and n == 3:
            raise _parse_500(f"http://{s}/v1/chat/completions")
    _served(monkeypatch, fail)
    code, tc = _gate(tmp_path)
    assert code == 0 and tc["status"] == "measured" and tc["pass"] is True
    assert tc["retention"] == pytest.approx(100 * 59 / 60 if side == "candidate" else 100 * 60 / 59, abs=0.01)
    assert (tc["modes"] if side == "candidate" else {}).get("server-error", 0) == (side == "candidate")


def test_parser_500s_on_every_probe_are_a_measured_fail(monkeypatch, tmp_path):
    """A broken rung whose every generation fails the parser is a real 0% -- FAIL, not 'error'."""
    def fail(s, n):
        if s == "candidate":
            raise _parse_500("http://candidate/v1/chat/completions")
    _served(monkeypatch, fail)
    code, tc = _gate(tmp_path, "--limit", "1")
    assert code == 1 and tc["status"] == "measured" and tc["pass"] is False and tc["rate"] == 0


def test_a_reference_that_only_errors_never_passes_the_rung_vacuously(monkeypatch, tmp_path):
    """Every reference probe an HTTP error: no evidence there is nothing to retain -- the rung's own rate decides."""
    def fail(s, n):
        if s == "reference":
            raise _parse_500("http://reference/v1/chat/completions")
    _served(monkeypatch, fail)
    code, tc = _gate(tmp_path, "--limit", "1")
    assert code == 0 and tc["pass"] is True and tc["rate"] == 1      # the rung itself calls tools fine
    _served(monkeypatch, fail, prose=("candidate",))
    code, tc = _gate(tmp_path, "--limit", "1")
    assert code == 1 and tc["pass"] is False                          # and a rung that cannot is failed


@pytest.mark.parametrize("share,status", [(0.5, "measured"), (0.0, "error")])
def test_a_minority_of_transport_failures_stays_in_the_score(monkeypatch, tmp_path, share, status):
    import urllib.error

    def fail(s, n):
        if s == "candidate" and n == 2:
            raise urllib.error.URLError("timed out")
    _served(monkeypatch, fail)
    code, tc = _gate(tmp_path, "--max-transport-share", str(share))
    assert tc["status"] == status
    if status == "measured":
        assert code == 0 and tc["retention"] == pytest.approx(100 * 59 / 60, abs=0.01)
    else:
        assert code == 1 and tc["pass"] is False and "1/60 probes got no reply" in tc["reason"]


def test_reference_cache_keeps_a_parser_500_but_not_an_outage(monkeypatch, tmp_path):
    cache = tmp_path / "reference.json"
    monkeypatch.setattr(TC, "_ref_cache_path", lambda *a: str(cache))
    parse = TC.grade_reply({"error": {"code": 500, "message": "Failed to parse input"}, "http_status": 500},
                           PROBE["s01"])
    assert parse["failure"] == "server-error" and parse["transport"] is False
    monkeypatch.setattr(TC, "run_model", lambda *a, **k: ([parse], {"backend": "gguf"}))
    TC.run_ref_cached("reference.gguf", "0")
    assert cache.exists()
