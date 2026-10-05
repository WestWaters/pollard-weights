"""Served comparisons must not call an incomplete token sum full KL."""
import math
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_serve_eval as E


def logs(**probs):
    return {k: math.log(v) for k, v in probs.items()}


def test_full_support_matches_known_kl():
    expected = .75 * math.log(.75 / .5) + .25 * math.log(.25 / .5)
    assert E.coarsened_kl(logs(a=.75, b=.25), logs(a=.5, b=.5)) == pytest.approx(expected)


def test_shared_token_negative_term_needs_the_other_bucket():
    # The old intersection-only score is .1*log(.1/.9), which is negative.
    expected = .1 * math.log(.1 / .9) + .9 * math.log(.9 / .1)
    assert E.coarsened_kl(logs(a=.1, b=.8), logs(a=.9, c=.05)) == pytest.approx(expected)


def test_identical_truncated_distributions_are_zero():
    assert E.coarsened_kl(logs(a=.3, b=.2), logs(a=.3, b=.2)) == 0


def test_no_shared_support_is_not_an_informative_zero():
    assert E.coarsened_kl(logs(a=.9), logs(b=.9)) is None


@pytest.mark.parametrize("bad", [{"a": 0.1}, {"a": float("nan")}, {"a": float("inf")},
                               logs(a=.8, b=.8)])
def test_invalid_distributions_fail(bad):
    with pytest.raises(ValueError):
        E.coarsened_kl(bad, logs(a=.5))


def test_position_average_does_not_divide_by_reported_tokens(monkeypatch):
    base, cand = logs(a=.75, b=.25), logs(a=.5, b=.5)
    monkeypatch.setattr(E, "_greedy_next", lambda endpoint, *_args, **_kwargs: ("a", base if endpoint == "base" else cand))
    agreement, positions, divergence = E.ab_agreement("base", "model", "cand", "model", ["one two three four five six seven"], stride=1)
    assert agreement == 1 and positions == 3
    assert divergence == pytest.approx(E.coarsened_kl(base, cand))


def test_zero_stride_fails_before_any_network_call():
    with pytest.raises(ValueError, match="positive"):
        E.ab_agreement("base", "model", "cand", "model", ["text"], stride=0)


def test_coarsening_is_a_lower_bound_on_full_kl():
    p, q = logs(a=.4, b=.3, c=.2, d=.1), logs(a=.2, b=.3, c=.4, d=.1)
    coarse = E.coarsened_kl({k: p[k] for k in ("a", "b")}, {k: q[k] for k in ("a", "c")})
    assert 0 < coarse < E.coarsened_kl(p, q)


def test_random_truncations_remain_nonnegative_lower_bounds():
    import random
    rng = random.Random(42)
    for _ in range(200):
        p, q = [rng.uniform(.01, 1) for _ in range(12)], [rng.uniform(.01, 1) for _ in range(12)]
        p, q = {str(i): math.log(x / sum(p)) for i, x in enumerate(p)}, {str(i): math.log(x / sum(q)) for i, x in enumerate(q)}
        left = dict(sorted(p.items(), key=lambda item: item[1], reverse=True)[:7])
        right = dict(sorted(q.items(), key=lambda item: item[1], reverse=True)[:7])
        coarse = E.coarsened_kl(left, right)
        assert coarse is not None and 0 <= coarse <= E.coarsened_kl(p, q) + 1e-12


def test_http_completion_path_uses_per_position_divergence():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            top = logs(a=.75, b=.25) if payload["model"] == "base" else logs(a=.5, b=.5)
            response = json.dumps({"choices": [{"text": "a", "logprobs": {
                "tokens": ["a"], "top_logprobs": [top]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        agree, positions, divergence = E.ab_agreement(url, "base", url, "cand", ["one two three four five six"])
        assert agree == 1 and positions == 1
        assert divergence == pytest.approx(.75 * math.log(1.5) + .25 * math.log(.5))
        assert all(r["max_tokens"] == 1 and r["logprobs"] == 20 for r in requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
