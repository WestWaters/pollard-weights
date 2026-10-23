"""pollard-node binds loopback by default, sends no wildcard CORS, and refuses foreign Hosts / Origins.

Security audit: it bound 0.0.0.0 and answered `Access-Control-Allow-Origin: *`, so any web page could read the
box's specs and model paths -- and through DNS rebinding, from any LAN box running it.
"""
from __future__ import annotations

import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import cluster, node  # noqa: E402


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(node, "report", lambda home=None: {"agent": "pollard-node", "ram_gb": 1.0})
    srvs = []

    def start(**kw):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), node.make_handler(**kw))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        srvs.append(srv)
        return srv.server_address[1]
    yield start
    for s in srvs:
        s.shutdown()
        s.server_close()


def _get(port, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/info", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)


def test_default_bind_is_loopback_and_lan_is_explicit():
    assert node.bind_host(None, False) == "127.0.0.1"
    assert node.bind_host(None, True) == "0.0.0.0"
    assert node.bind_host("192.0.2.10", False) == "192.0.2.10"


def test_main_defaults_to_loopback(monkeypatch):
    got = {}
    monkeypatch.setattr(node, "serve", lambda host, port, *a: got.update(host=host, port=port))
    monkeypatch.setattr(sys, "argv", ["pollard-node"])
    node.main()
    assert got == {"host": "127.0.0.1", "port": node.DEFAULT_PORT}
    monkeypatch.setattr(sys, "argv", ["pollard-node", "--lan"])
    node.main()
    assert got["host"] == "0.0.0.0"


def test_no_cors_header_by_default_and_studio_still_reads_it(served):
    port = served()
    status, h = _get(port)
    assert status == 200 and "Access-Control-Allow-Origin" not in h
    assert cluster.probe_agent("127.0.0.1", port, timeout=3)["agent"] == "pollard-node"


def test_a_foreign_origin_is_refused(served):
    port = served()
    status, h = _get(port, {"Origin": "https://evil.example"})
    assert status == 403 and "Access-Control-Allow-Origin" not in h


def test_an_allowed_origin_is_echoed_never_a_wildcard(served):
    port = served(allow_origins=["http://studio.lan:8000"])
    status, h = _get(port, {"Origin": "http://studio.lan:8000"})
    assert status == 200 and h["Access-Control-Allow-Origin"] == "http://studio.lan:8000"
    assert _get(port, {"Origin": "http://other.lan"})[0] == 403


def test_a_rebinding_host_is_refused(served):
    port = served()
    assert _get(port, {"Host": f"attacker.example:{port}"})[0] == 403
    assert _get(port, {"Host": f"localhost:{port}"})[0] == 200
    assert _get(port, {"Host": f"192.0.2.10:{port}"})[0] == 200          # an IP literal: what Studio dials
    port2 = served(allow_hosts=["pcbox.lan"])
    assert _get(port2, {"Host": f"pcbox.lan:{port2}"})[0] == 200


def test_host_header_parsing():
    assert node.host_allowed("[::1]:50053") and node.host_allowed("127.0.0.1")
    assert not node.host_allowed("") and not node.host_allowed(None)
    assert not node.host_allowed("evil.example")
