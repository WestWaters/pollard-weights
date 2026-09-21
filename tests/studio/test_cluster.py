"""The pool: this machine plus every linked box.

The probe speaks ggml's own RPC protocol by hand, so these tests stand up a REAL server and talk
to it where one is available -- a hand-rolled wire format that is only ever tested against a mock
of itself proves nothing.
"""
from __future__ import annotations

import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

# the package lives at <repo>/tools/pollard_studio, so ROOT is the tools dir:
# every path below stays written as "pollard_studio/..." and the import works too
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(ROOT))

from pollard_studio import cluster  # noqa: E402

# the runtime build this repo keeps current, wherever the repo happens to live
RPC_BIN = REPO_ROOT / "runtime/llama.cpp/build/bin/ggml-rpc-server"


# ── endpoint parsing ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,host,port", [
    ("10.0.0.5:50052", "10.0.0.5", 50052),
    ("  box.local:9999  ", "box.local", 9999),
    ("box.local", "box.local", cluster.DEFAULT_RPC_PORT),
    ("[::1]:50052", "::1", 50052),
    ("[fe80::1]", "fe80::1", cluster.DEFAULT_RPC_PORT),
])
def test_endpoints_parse(raw, host, port):
    assert cluster.parse_endpoint(raw) == (host, port)


def test_ipv6_does_not_split_on_the_wrong_colon():
    """A naive rsplit(':') turns '::1' into host ':' port '1'."""
    assert cluster.parse_endpoint("[::1]:50052")[0] == "::1"


# ── this machine ────────────────────────────────────────────────────────────────────────────────

def test_local_reads_the_real_box():
    got = cluster.local()
    assert got["reachable"] is True
    assert got["ram_gb"] and got["ram_gb"] > 0
    assert got["cpus"] and got["cpus"] > 0
    assert got["platform"] == sys.platform


def test_unified_memory_is_not_double_counted():
    """On Apple silicon the GPU shares system RAM. Reporting both would count the same bytes
    twice and offer a build a budget that does not exist."""
    got = cluster.local()
    if got["unified"]:
        assert got["vram_gb"] is None
        assert got["usable_gb"] == got["ram_gb"]
        assert "unified" in got["note"]
    else:
        assert got["usable_gb"] == (got["vram_gb"] or got["ram_gb"])


# ── a peer that is not there ────────────────────────────────────────────────────────────────────

def test_an_unreachable_peer_is_reported_not_raised():
    got = cluster.probe("127.0.0.1:1", timeout=0.4)
    assert got["reachable"] is False
    assert got["total_gb"] is None
    assert got["note"]


def test_a_socket_that_is_not_an_rpc_server_degrades_honestly():
    """Something listening on the port is not the same as a peer a build can use."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    # accept and say nothing, the way an unrelated service would
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    try:
        got = cluster.probe(f"127.0.0.1:{port}", timeout=0.5)
        assert got["reachable"] is True          # the socket IS open
        assert got["total_gb"] is None           # but there is no memory to report
        assert got["note"]
    finally:
        srv.close()


# ── against a real ggml-rpc-server ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def live_rpc():
    """A real ggml-rpc-server, so the hand-written protocol is checked against the thing it
    claims to speak."""
    if not RPC_BIN.exists():
        pytest.skip("no ggml-rpc-server built")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = subprocess.Popen([str(RPC_BIN), "-p", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):                          # it compiles Metal shaders on first start
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.25)
    else:
        proc.kill()
        pytest.skip("ggml-rpc-server did not come up")
    yield f"127.0.0.1:{port}"
    proc.kill()
    proc.wait(timeout=10)


def test_the_handshake_matches_the_real_server(live_rpc):
    got = cluster.probe(live_rpc, timeout=5)
    assert got["reachable"] is True
    assert got["protocol"], "no version came back — the HELLO framing is wrong"
    major = int(got["protocol"].split(".")[0])
    assert 1 <= major <= 99


def test_real_device_memory_comes_back(live_rpc):
    got = cluster.probe(live_rpc, timeout=5)
    assert got["devices"], f"no devices reported: {got['note']}"
    assert got["total_gb"] and got["total_gb"] > 0
    for d in got["devices"]:
        assert d["total_gb"] > 0, "a device with no memory should be dropped, not listed"


def test_a_version_mismatch_still_reports_memory(live_rpc):
    """A peer one protocol behind usually answers the memory query identically. Saying 'offline'
    about a box that is plainly up and serving is worse than a caveat."""
    got = cluster.probe(live_rpc, timeout=5)
    if got["protocol"] and int(got["protocol"].split(".")[0]) != cluster.PROTO_MAJOR:
        assert got["total_gb"], "memory was dropped purely because of a version difference"
        assert "rpc" in got["note"].lower()


def test_the_survey_combines_this_box_and_the_peer(live_rpc):
    got = cluster.survey(live_rpc, timeout=5)
    assert got["pooled"] is True
    assert got["online"] == 1
    assert len(got["nodes"]) == 2
    here = cluster.local()
    peer = cluster.probe(live_rpc, timeout=5)
    assert got["combined_gb"] == pytest.approx(
        round(here["usable_gb"] + peer["total_gb"], 1), abs=0.2)
    assert got["largest_single_gb"] == max(here["usable_gb"], peer["total_gb"])


# ── the survey ──────────────────────────────────────────────────────────────────────────────────

def test_a_survey_with_no_peers_is_just_this_machine():
    got = cluster.survey("")
    assert got["pooled"] is False
    assert got["online"] == 0
    assert len(got["nodes"]) == 1
    assert got["combined_gb"] == cluster.local()["usable_gb"]


def test_the_environment_is_consulted_when_nothing_is_typed(monkeypatch):
    """A pool configured outside Studio should still show up."""
    monkeypatch.setenv("POLLARD_RPC", "127.0.0.1:1")
    got = cluster.survey(None, timeout=0.3)
    assert got["configured"] == 1
    assert got["offline"] == ["127.0.0.1:1"]


def test_a_dead_peer_is_listed_as_offline_not_silently_dropped():
    """A build will not see it, so the panel must not imply it is there."""
    got = cluster.survey("127.0.0.1:1", timeout=0.4)
    assert got["online"] == 0
    assert got["offline"] == ["127.0.0.1:1"]
    assert len(got["nodes"]) == 2, "the offline box still has to appear"
    assert got["combined_gb"] == cluster.local()["usable_gb"], \
        "an unreachable box must not contribute memory"


def test_several_peers_are_probed_at_once():
    """Ten dead peers at a 1s timeout must not take ten seconds."""
    eps = ",".join(f"127.0.0.1:{p}" for p in range(9001, 9011))
    t0 = time.monotonic()
    got = cluster.survey(eps, timeout=1.0)
    assert time.monotonic() - t0 < 5.0
    assert got["configured"] == 10


# ── discovery is opt-in ─────────────────────────────────────────────────────────────────────────

def test_the_subnet_helper_sends_nothing():
    """It connects a UDP socket to a reserved address purely to read the local interface."""
    got = cluster._local_subnet()
    assert got is None or got.count(".") == 2


def test_discover_is_not_called_anywhere_automatically():
    """Sweeping a network is something a user asks for. On a shared or corporate network it is
    exactly the traffic that gets noticed."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "cluster_discover" in js, "discovery is not reachable at all"
    # it may only appear behind an explicit click
    for line in js.splitlines():
        if "cluster_discover" in line:
            assert "await bridge().cluster_discover" in line
    calls = [ln for ln in js.splitlines() if "discoverPeers()" in ln]
    assert any("onclick" in c for c in calls), "discovery must be behind a button"
    assert not any("onclick" not in c and "function" not in c for c in calls), \
        "discovery is being called on its own somewhere"


def test_discovery_only_touches_the_one_rpc_port():
    src = (ROOT / "pollard_studio/cluster.py").read_text()
    fn = src[src.index("def discover("):src.index("def _local_subnet")]
    assert "create_connection((h, port)" in fn, "discovery should open one port per host"
    assert "range(1, 255)" in fn, "discovery should stay inside a single /24"


# ── the cluster counted as one machine ──────────────────────────────────────────────────────────

def test_totals_are_broken_out_by_resource(live_rpc):
    """"Combined" as a single number does not tell you whether you can hold the weights. RAM,
    accelerator memory and storage are different questions."""
    got = cluster.survey(live_rpc, timeout=5)
    for k in ("total_ram_gb", "total_vram_gb", "total_disk_gb", "boxes", "devices"):
        assert k in got, f"survey does not report {k}"
    assert got["boxes"] == 2
    assert got["devices"] >= 2


def test_accelerator_memory_sums_across_the_cluster(live_rpc):
    """This is the number that bounds a build: what every box can actually hold weights in."""
    here = cluster.local()
    peer = cluster.probe(live_rpc, timeout=5)
    got = cluster.survey(live_rpc, timeout=5)
    expect = (here["ram_gb"] if here["unified"] else (here["vram_gb"] or 0)) + peer["total_gb"]
    assert got["total_vram_gb"] == pytest.approx(round(expect, 1), abs=0.2)


def test_unified_memory_is_counted_once_not_in_both_totals():
    """On a unified box the same bytes serve both roles. They belong in each total once, and the
    cluster figure must not add them twice."""
    got = cluster.survey("")
    here = cluster.local()
    if here["unified"]:
        assert got["total_ram_gb"] == here["ram_gb"]
        assert got["total_vram_gb"] == here["ram_gb"]
        assert got["combined_gb"] == here["ram_gb"]


def test_peer_ram_and_storage_are_not_invented(live_rpc, monkeypatch):
    """RPC exposes device memory, never the host's RAM or disk. A peer with no pollard-node must
    contribute nothing to those totals rather than a number nobody measured."""
    monkeypatch.setattr(cluster, "probe_agent", lambda *a, **k: None)   # no agent anywhere
    got = cluster.survey(live_rpc, timeout=5)
    here = cluster.local()
    assert got["ram_partial"] is True
    assert got["disk_partial"] is True
    assert got["no_agent"] == [cluster.probe(live_rpc, timeout=5)["endpoint"]]
    assert got["total_ram_gb"] == here["ram_gb"]
    assert got["total_disk_gb"] == pytest.approx(here["disk_free_gb"], abs=1.0)
    # the accelerator total is still a real cluster figure -- that part RPC does answer
    assert got["total_vram_gb"] > (here["ram_gb"] if here["unified"] else (here["vram_gb"] or 0))


def test_an_agent_completes_a_peers_ram_and_storage(live_rpc, monkeypatch):
    """With pollard-node on the box, its RAM, cards and disk join the cluster totals -- which is
    the whole reason the agent exists."""
    fake = {"agent": "pollard-node", "host": "pcbox", "platform": "win32", "cpus": 32,
            "ram_gb": 64.0, "disk_free_gb": 900.0, "unified": False,
            "cards": [{"name": "RTX 5070 Ti", "gb": 16.0}], "usable_gb": 16.0}
    monkeypatch.setattr(cluster, "probe_agent", lambda *a, **k: fake)
    got = cluster.survey(live_rpc, timeout=5)
    here = cluster.local()
    assert got["agents"] == 1
    assert got["no_agent"] == []
    assert got["ram_partial"] is False and got["disk_partial"] is False
    assert got["total_ram_gb"] == pytest.approx(here["ram_gb"] + 64.0, abs=0.2)
    assert got["total_disk_gb"] == pytest.approx(here["disk_free_gb"] + 900.0, abs=1.0)
    peer = [n for n in got["nodes"] if n.get("agent")][0]
    assert peer["host"] == "pcbox"
    assert peer["cards"][0]["name"] == "RTX 5070 Ti"


def test_the_agent_never_overrides_a_measured_device_figure(live_rpc, monkeypatch):
    """RPC read the devices directly; the agent's own view must not replace that."""
    rpc_only = cluster.probe(live_rpc, timeout=5)
    fake = {"agent": "pollard-node", "host": "pcbox", "ram_gb": 64.0,
            "usable_gb": 999.0, "unified": False, "cards": [], "disk_free_gb": 10.0}
    monkeypatch.setattr(cluster, "probe_agent", lambda *a, **k: fake)
    got = cluster.survey(live_rpc, timeout=5)
    peer = [n for n in got["nodes"] if n.get("agent")][0]
    assert peer["total_gb"] == rpc_only["total_gb"], "the agent overwrote what RPC measured"


def test_an_offline_peer_contributes_no_memory():
    got = cluster.survey("127.0.0.1:1", timeout=0.4)
    here = cluster.local()
    assert got["total_vram_gb"] == (here["ram_gb"] if here["unified"] else here["vram_gb"])
    assert got["boxes"] == 1


def test_the_readout_lives_with_the_knob_and_faders():
    """It is what the faders are spending, so it belongs in the memory-target panel -- not in a
    separate card further down the screen."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    bay = js[js.index("LANE · MEMORY TARGET"):js.index("ACROSS MACHINES")]
    assert 'id="cl-pool"' in bay, "the cluster readout is not in the memory-target panel"
    assert "knobbay" in bay and "faders" in bay
    assert js.count('id="cl-pool"') == 1, "the readout is rendered in two places"


# ── the agent itself ────────────────────────────────────────────────────────────────────────────

def test_the_agent_reports_the_same_shape_studio_reads():
    from pollard_studio import node
    got = node.report()
    assert got["agent"] == "pollard-node"
    assert got["schema"] == node.SCHEMA
    for k in ("host", "platform", "cpus", "ram_gb", "disk_free_gb", "unified", "usable_gb"):
        assert k in got, f"the agent does not report {k}"


def test_the_agent_serves_and_studio_reads_it_back():
    """End to end over a real socket, because that is the only thing that proves the two halves
    agree."""
    import threading
    from http.server import ThreadingHTTPServer
    from pollard_studio import node
    srv = ThreadingHTTPServer(("127.0.0.1", 0), node._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        got = cluster.probe_agent("127.0.0.1", srv.server_address[1], timeout=3)
        assert got is not None, "Studio could not read what the agent served"
        assert got["ram_gb"] == node.report()["ram_gb"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_agent_answers_nothing_but_info():
    """It is read-only on purpose: no endpoint runs, writes, or takes a parameter."""
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from pollard_studio import node
    srv = ThreadingHTTPServer(("127.0.0.1", 0), node._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        for path in ("/run", "/../etc/passwd", "/info/x", "/shutdown"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3)
            assert e.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_agent_exposes_no_way_to_execute_anything():
    src = (ROOT / "pollard_studio/node.py").read_text()
    for danger in ("subprocess", "os.system", "eval(", "exec(", "do_POST", "shell=True"):
        assert danger not in src, f"pollard-node must not contain {danger}"


def test_a_non_pollard_service_on_the_port_is_not_mistaken_for_an_agent():
    """Something else answering JSON on that port must not be read as a box report."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Impostor(BaseHTTPRequestHandler):
        def do_GET(self):                                  # noqa: N802
            body = _json.dumps({"ram_gb": 999, "host": "nope"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Impostor)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert cluster.probe_agent("127.0.0.1", srv.server_address[1], timeout=3) is None
    finally:
        srv.shutdown()
        srv.server_close()


def test_pollard_node_is_installable_as_a_command():
    """Every box in the cluster runs it, so it has to be an entry point, not a file to copy."""
    pp = (REPO_ROOT / "pyproject.toml").read_text()
    assert "pollard-node" in pp and "pollard_studio.node:main" in pp


# ── a box that reports but is not serving ───────────────────────────────────────────────────────

AGENT = {"agent": "pollard-node", "host": "pcbox", "platform": "win32", "cpus": 32,
         "ram_gb": 34.2, "disk_free_gb": 44.6, "unified": False, "usable_gb": 15.9,
         "cards": [{"name": "RTX 5070 Ti", "gb": 15.9}]}


def test_a_box_with_only_the_agent_still_joins_the_cluster(monkeypatch):
    """Requiring ggml-rpc-server to be up as well meant a box that was plainly online showed as
    offline and contributed nothing to any total."""
    monkeypatch.setattr(cluster, "probe_agent", lambda *a, **k: AGENT)
    got = cluster.survey("127.0.0.1:1", timeout=0.4)      # nothing listening for RPC
    here = cluster.local()
    assert got["boxes"] == 2, "the reporting box was dropped"
    assert got["offline"] == []
    assert got["total_ram_gb"] == pytest.approx(here["ram_gb"] + 34.2, abs=0.2)
    assert got["total_vram_gb"] == pytest.approx(
        (here["ram_gb"] if here["unified"] else (here["vram_gb"] or 0)) + 15.9, abs=0.2)
    assert got["total_disk_gb"] == pytest.approx(here["disk_free_gb"] + 44.6, abs=1.0)


def test_reporting_is_not_the_same_as_serving(monkeypatch):
    """A build can only place weights where ggml-rpc-server answers. The panel must not let a
    box that merely reports its specs look like capacity a build can use."""
    monkeypatch.setattr(cluster, "probe_agent", lambda *a, **k: AGENT)
    got = cluster.survey("127.0.0.1:1", timeout=0.4)
    assert got["serving"] == 1, "a box with no rpc-server was counted as serving"
    peer = [n for n in got["nodes"] if n.get("agent")][0]
    assert peer["serving"] is False
    assert "ggml-rpc-server" in peer["note"]


def test_an_open_port_alone_is_not_serving(monkeypatch):
    """The agent's own HTTP port accepts a TCP connection; that must not read as an RPC peer."""
    import threading
    from http.server import ThreadingHTTPServer
    from pollard_studio import node
    srv = ThreadingHTTPServer(("127.0.0.1", 0), node._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        got = cluster.survey(f"127.0.0.1:{port}", timeout=2)
        assert got["serving"] == 1
        peer = got["nodes"][1]
        assert peer["reachable"] is True and peer["serving"] is False
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_agent_is_found_on_a_forwarded_port(monkeypatch):
    """Behind an SSH tunnel the agent answers on whatever local port was forwarded, so insisting
    on 50053 would never find a box linked that way."""
    import threading
    from http.server import ThreadingHTTPServer
    from pollard_studio import node
    srv = ThreadingHTTPServer(("127.0.0.1", 0), node._Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    assert port != cluster.NODE_PORT
    try:
        got = cluster.survey(f"127.0.0.1:{port}", timeout=2)
        assert got["agents"] == 1, "the agent on a forwarded port was not found"
    finally:
        srv.shutdown()
        srv.server_close()


# ── keeping ggml-rpc-server out of the way ──────────────────────────────────────────────────────

def test_the_binary_is_found_without_being_told_where_it_is():
    got = cluster.rpc_binary()
    if got is None:
        pytest.skip("no ggml-rpc-server on this machine")
    assert got.is_file()


def test_an_explicit_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "ggml-rpc-server"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("POLLARD_RPC_BIN", str(fake))
    assert cluster.rpc_binary() == fake


def test_the_version_compared_against_is_MEASURED_not_read_from_a_header():
    """A binary older than its own sources is normal. Comparing a peer against the HEADER while
    the two machines' binaries agree perfectly would send people off to rebuild for nothing."""
    b = cluster.rpc_binary()
    if b is None:
        pytest.skip("no ggml-rpc-server on this machine")
    measured, header = cluster.measured_proto_major(b), cluster.header_proto_major(b)
    assert measured is not None, "could not measure what the local binary speaks"
    src = (ROOT / "pollard_studio/cluster.py").read_text()
    fn = src[src.index("def rpc_status("):src.index("def _reachable_off_box")]
    assert "measured_proto_major" in fn, "rpc_status compares against the header"
    if measured != header:
        # exactly this machine's situation, and it must NOT be reported as a peer mismatch
        d = cluster.probe("127.0.0.1:1", timeout=0.2, expect=measured)
        assert d["mismatch"] is False


def test_a_peer_matching_the_local_binary_raises_no_mismatch(live_rpc):
    b = cluster.rpc_binary()
    if b is None:
        pytest.skip("no ggml-rpc-server on this machine")
    got = cluster.probe(live_rpc, timeout=5, expect=cluster.measured_proto_major(b))
    assert got["mismatch"] is False, f"false mismatch: {got['note']}"
    assert got["note"] == ""


def test_a_genuine_mismatch_is_still_reported(live_rpc):
    got = cluster.probe(live_rpc, timeout=5, expect=99)
    assert got["mismatch"] is True
    assert "rebuild" in got["note"]
    assert got["total_gb"], "memory should still be read despite the mismatch"


def test_the_start_command_binds_where_other_boxes_can_reach_it():
    """The default is 127.0.0.1, which is the single most common reason a box never joins."""
    cmd = cluster.serve_command("/x/ggml-rpc-server", 50052)
    assert cmd[1:] == ["-H", "0.0.0.0", "-p", "50052"]


def test_staleness_is_detected_from_the_sources_beside_it(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "build/bin").mkdir(parents=True)
    (root / "ggml/src/ggml-rpc").mkdir(parents=True)
    (root / "ggml/include").mkdir(parents=True)
    binary = root / "build/bin/ggml-rpc-server"
    binary.write_text("x")
    binary.chmod(0o755)
    src = root / "ggml/src/ggml-rpc/ggml-rpc.cpp"
    src.write_text("x")
    import os as _os
    _os.utime(binary, (1_000, 1_000))
    _os.utime(src, (2_000, 2_000))            # sources newer than the build
    assert cluster.rpc_staleness(binary)["stale"] is True
    _os.utime(binary, (3_000, 3_000))
    assert cluster.rpc_staleness(binary)["stale"] is False


def test_studio_can_start_and_stop_a_server():
    """The whole point: a user should not have to go and run this by hand."""
    if cluster.rpc_binary() is None:
        pytest.skip("no ggml-rpc-server on this machine")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    got = cluster.serve_local(port, timeout=40)
    try:
        assert got["ok"], got.get("error")
        assert cluster.probe(f"127.0.0.1:{port}", timeout=3)["protocol"]
    finally:
        cluster.stop_local()
    assert cluster.probe(f"127.0.0.1:{port}", timeout=1)["protocol"] is None


def test_stopping_only_touches_what_studio_started():
    """A user's own rpc-server, started outside Studio, must survive."""
    src = (ROOT / "pollard_studio/cluster.py").read_text()
    fn = src[src.index("def stop_local("):]
    fn = fn[:fn.index("\n\n\n")] if "\n\n\n" in fn else fn
    assert "_SERVER" in fn
    for danger in ("pkill", "killall", "taskkill"):
        assert danger not in fn, f"stop_local must not {danger}"


# ── per-device placement ────────────────────────────────────────────────────────────────────────

def _surv(peers, local_cards=None, unified=False, ram=16.0):
    nodes = [{"role": "this machine", "host": "here", "unified": unified, "ram_gb": ram,
              "cards": local_cards or [], "reachable": True}]
    nodes += peers
    return {"nodes": nodes}


def test_rpc_devices_come_before_local_ones():
    """llama.cpp inserts RPC servers at the FRONT of its device list ("to minimize network
    transfers"). Ordering -ts the other way sends each box the wrong share, silently."""
    peers = [{"role": "rpc peer", "endpoint": "boxA:50052", "host": "boxA", "serving": True,
              "reachable": True, "devices": [{"device": 0, "total_gb": 16.0, "free_gb": 15.0}]}]
    order = cluster.device_order(_surv(peers, local_cards=[{"name": "RTX 4090", "gb": 24.0}]))
    assert [d["label"] for d in order] == ["boxA", "RTX 4090"]
    assert order[0]["remote"] is True and order[1]["remote"] is False


def test_peers_keep_the_order_they_were_listed_in():
    """-ts positions follow the order given to --rpc, so the survey's order has to be preserved."""
    peers = [{"role": "rpc peer", "endpoint": f"box{c}:50052", "host": f"box{c}", "serving": True,
              "reachable": True, "devices": [{"device": 0, "total_gb": 8.0, "free_gb": 8.0}]}
             for c in "ABC"]
    order = cluster.device_order(_surv(peers))
    assert [d["host"] for d in order] == ["boxA", "boxB", "boxC"]


def test_a_peer_that_is_not_serving_is_not_a_placement_target():
    """It cannot hold weights, so giving it a fader would promise something that fails."""
    peers = [{"role": "rpc peer", "endpoint": "boxA:50052", "host": "boxA", "serving": False,
              "reachable": True, "devices": []}]
    assert cluster.device_order(_surv(peers)) == []


def test_every_device_on_a_multi_gpu_peer_gets_its_own_position():
    peers = [{"role": "rpc peer", "endpoint": "boxA:50052", "host": "boxA", "serving": True,
              "reachable": True,
              "devices": [{"device": 0, "total_gb": 24.0, "free_gb": 24.0},
                          {"device": 1, "total_gb": 24.0, "free_gb": 24.0}]}]
    order = cluster.device_order(_surv(peers))
    assert len(order) == 2
    assert [d["detail"] for d in order] == ["rpc dev 0", "rpc dev 1"]


def test_unified_memory_is_offered_as_one_device():
    order = cluster.device_order(_surv([], unified=True, ram=17.2))
    assert len(order) == 1 and order[0]["label"] == "unified memory"
    assert order[0]["gb"] == 17.2


def test_shares_become_a_tensor_split_in_that_order():
    peers = [{"role": "rpc peer", "endpoint": "boxA:50052", "host": "boxA", "serving": True,
              "reachable": True, "devices": [{"device": 0, "total_gb": 16.0, "free_gb": 16.0}]}]
    order = cluster.device_order(_surv(peers, local_cards=[{"name": "RTX 4090", "gb": 24.0}]))
    ts = cluster.tensor_split({order[0]["key"]: 40, order[1]["key"]: 60}, order)
    assert ts == "40,60"


def test_a_device_at_zero_is_kept_in_position_not_dropped():
    """Dropping it would shift every later device onto the wrong box."""
    peers = [{"role": "rpc peer", "endpoint": "boxA:50052", "host": "boxA", "serving": True,
              "reachable": True, "devices": [{"device": 0, "total_gb": 16.0, "free_gb": 16.0}]}]
    order = cluster.device_order(_surv(peers, local_cards=[{"name": "RTX 4090", "gb": 24.0}]))
    assert cluster.tensor_split({order[0]["key"]: 0, order[1]["key"]: 100}, order) == "0,100"


def test_everything_at_zero_means_no_split_at_all():
    """An all-zero -ts is rejected by llama.cpp; the honest result is to omit it."""
    order = cluster.device_order(_surv([], unified=True))
    assert cluster.tensor_split({order[0]["key"]: 0}, order) is None
    assert cluster.placement_args({order[0]["key"]: 0}, order, ngl=0) == ["-ngl", "0"]


def test_placement_args_pair_ngl_with_the_split():
    order = cluster.device_order(_surv([], unified=True))
    got = cluster.placement_args({order[0]["key"]: 100}, order, ngl=32)
    assert got == ["-ngl", "32", "-ts", "100"]


def test_the_placement_reaches_llama_cpp_through_pollard_run():
    """pollard-run hardcodes '-ngl 999' and appends --extra verbatim, so these must come after
    it -- llama.cpp takes the last value."""
    from pollard_studio import actions
    tool, args = actions.resolve("placement", {"gguf": "/m.gguf", "rpc": "a:1",
                                               "placement": ["-ngl", "32", "-ts", "70,30"]})
    assert tool == "pollard_run"
    assert args[args.index("--extra") + 1] == "-ngl 32 -ts 70,30"


def test_a_users_own_extra_args_are_kept_alongside_the_placement():
    from pollard_studio import actions
    _, args = actions.resolve("placement", {"gguf": "/m.gguf", "placement": ["-ngl", "8"],
                                            "extraArgs": "--flash-attn"})
    assert args[args.index("--extra") + 1] == "-ngl 8 --flash-attn"


def test_the_device_order_is_documented_against_llama_cpp():
    """This ordering is not guessable and breaks silently, so the reason has to sit next to it."""
    src = (ROOT / "pollard_studio/cluster.py").read_text()
    blk = src[src.index("# ── per-device placement"):src.index("def device_order")]
    assert "front" in blk and "--rpc" in blk


def test_the_placement_faders_live_with_the_knob_and_faders():
    """Deciding which device holds which layers IS the memory-target decision, so it belongs in
    that panel -- not in the card that only configures connections."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    bay = js[js.index("LANE \u00b7 MEMORY TARGET"):js.index("ACROSS MACHINES")]
    across = js[js.index("ACROSS MACHINES"):js.index("THE COMMAND THIS IS")]
    assert 'id="f-bank"' in bay, "the fader bank is not in the memory-target panel"
    assert 'id="dev-place"' in bay, "the placement readout is not in the memory-target panel"
    assert 'id="dev-place"' not in across, "they are still in the across-machines card"
    assert 'id="cl-pool"' in bay
    assert js.count('id="dev-place"') == 1, "rendered in two places"
    # the connection controls stay where they were
    for ident in ("cl-rpc", "cl-tp", "rpc-state"):
        assert ident in across, f"{ident} should remain in ACROSS MACHINES"


def test_each_device_gets_a_gold_fader_in_the_same_bank_as_ram():
    """They control the same thing the RAM and RESERVE faders do -- how memory gets spent -- so
    they are the same hardware, not sliders stacked underneath."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("async function renderDevices"):js.index("function shortLabel")]
    assert '$("#f-bank")' in fn, "device faders are not put in the RAM/RESERVE bank"
    assert 'class="fader dev"' in fn, "they are not built as faders"
    for part in ('class="fslot"', 'class="rail"', 'class="ticks"', 'class="fv"', 'class="fl"'):
        assert part in fn, f"a device fader is missing {part} — it will not match RAM/RESERVE"
    assert "flat blue" not in fn, "still rendering horizontal sliders"


def test_rebuilding_the_bank_does_not_stack_duplicate_faders():
    """renderDevices runs again on every pool rescan."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("async function renderDevices"):js.index("function shortLabel")]
    assert 'querySelectorAll(".fader.dev").forEach(el => el.remove())' in fn


def test_a_long_hostname_is_cut_to_fit_a_fader_legend():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("function shortLabel"):]
    fn = fn[:fn.index("\n}\n") + 3]
    assert "slice" in fn and "split" in fn


def test_the_command_readout_class_is_actually_styled():
    """.cmd was used on every "this is what will run" box and had no rule at all."""
    css = (ROOT / "pollard_studio/ui/app.css").read_text()
    assert "\n.cmd {" in css, ".cmd has no style rule"
    assert ".cmd .fl {" in css, "flags in a command are unstyled"


# ── models across the cluster, every lane ───────────────────────────────────────────────────────

def _mk(d, files, cfg=None):
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_bytes(b"x" * 2048)
    if cfg is not None:
        import json as _j
        (d / "config.json").write_text(_j.dumps(cfg))
    return d


def test_every_lane_is_found_not_just_gguf(tmp_path):
    """A cluster that only lists GGUFs hides most of what Pollard builds."""
    from pollard_studio import node
    (tmp_path / "rungs").mkdir()
    (tmp_path / "rungs" / "m-IQ4_XS.gguf").write_bytes(b"x" * 4096)
    _mk(tmp_path / "models" / "mlx-build", ["model.safetensors"],
        cfg={"quantization": {"group_size": 64, "bits": 4}})
    _mk(tmp_path / "models" / "mx-build", ["model.safetensors"],
        cfg={"quantization_config": {"quant_method": "compressed-tensors"}})
    _mk(tmp_path / "models" / "gptq-build", ["model.safetensors", "quantize_config.json"])
    _mk(tmp_path / "models" / "exl3-build", ["out_tensor.safetensors"])
    _mk(tmp_path / "models" / "hf-source", ["model.safetensors"], cfg={"architectures": ["X"]})

    got = {m["name"]: m["lane"] for m in node.models(tmp_path)}
    assert got["m-IQ4_XS.gguf"] == "GGUF"
    assert got["mlx-build"] == "MLX"
    assert got["mx-build"] == "MX"
    assert got["gptq-build"] == "GPTQ"
    assert got["exl3-build"] == "EXL3"
    assert got["hf-source"] == "HF"


def test_a_directory_build_is_one_model_not_its_shards(tmp_path):
    """"model.safetensors" is not a model, and there is no such thing as picking half an MLX
    export."""
    from pollard_studio import node
    _mk(tmp_path / "models" / "mlx-build",
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        cfg={"quantization": {"bits": 4}})
    got = node.models(tmp_path)
    assert [m["name"] for m in got] == ["mlx-build"]
    assert got[0]["bytes"] == 4096, "the build's size should be the sum of its shards"


def test_the_lane_is_read_from_the_files_not_the_folder_name(tmp_path):
    from pollard_studio import node
    _mk(tmp_path / "models" / "totally-a-gguf-folder", ["model.safetensors"],
        cfg={"quantization": {"bits": 4}})
    assert node.models(tmp_path)[0]["lane"] == "MLX"


def test_a_workspace_with_nothing_in_it_is_not_an_error(tmp_path):
    from pollard_studio import node
    assert node.models(tmp_path) == []


def test_the_agent_reports_its_models_and_its_workspace():
    from pollard_studio import node
    r = node.report()
    assert "models" in r and isinstance(r["models"], list)
    assert r.get("workspace"), "a path with no workspace named is not selectable"


def test_remote_models_carry_the_host_that_holds_them(monkeypatch):
    """A path only means something on its own machine. A list that hides which box a file is on
    invites a run that cannot find it."""
    from pollard_studio import app, cluster as C
    fake = {"agent": "pollard-node", "host": "pcbox", "ram_gb": 34.2, "unified": False,
            "cards": [], "disk_free_gb": 44.6, "usable_gb": 15.9, "workspace": "C:/pollard",
            "models": [{"name": "penjing-IQ4_XS.gguf", "path": "C:/pollard/rungs/p.gguf",
                        "dir": "C:/pollard/rungs", "lane": "GGUF",
                        "bytes": 4e9, "mtime": 1.0}]}
    monkeypatch.setattr(C, "probe_agent", lambda *a, **k: fake)
    api = app.Api.__new__(app.Api)
    api._ws = None
    got = api.cluster_models("127.0.0.1:1")
    assert got["models"], "no remote models came back"
    m = got["models"][0]
    assert m["host"] == "pcbox" and m["remote"] is True
    assert m["lane"] == "GGUF" and m["workspace"] == "C:/pollard"
    assert got["hosts"] == ["pcbox"]


def test_this_machines_models_are_not_duplicated_into_the_remote_list(monkeypatch):
    """Studio already lists the local workspace; repeating it under a host name would show every
    local build twice."""
    from pollard_studio import app
    api = app.Api.__new__(app.Api)
    api._ws = None
    got = api.cluster_models("")
    assert got["models"] == [], "local builds leaked into the cluster list"


def test_a_remote_path_is_flagged_before_the_run_not_at_read_time(monkeypatch):
    """Pollard is not capped on model size and a cluster is the point, but a GGUF still has to be
    where the process reading it is. A confusing read error twenty minutes in is the worst way to
    learn that."""
    from pollard_studio import app, cluster as C
    fake = {"agent": "pollard-node", "host": "pcbox", "ram_gb": 34.2, "unified": False,
            "cards": [], "disk_free_gb": 44.6, "usable_gb": 15.9, "workspace": "C:/pollard",
            "models": [{"name": "p.gguf", "path": "C:/pollard/rungs/p.gguf", "dir": "C:/pollard",
                        "lane": "GGUF", "bytes": 4e9, "mtime": 1.0}]}
    monkeypatch.setattr(C, "probe_agent", lambda *a, **k: fake)
    api = app.Api.__new__(app.Api)
    api._ws, api._remote, api._flagspec = None, {}, {}
    api.cluster_models("127.0.0.1:1")
    got = api._remote_problems({"--gguf": "C:/pollard/rungs/p.gguf"})
    assert len(got) == 1
    assert "pcbox" in got[0] and "not readable from this machine" in got[0]


def test_a_local_path_is_not_flagged():
    from pollard_studio import app
    api = app.Api.__new__(app.Api)
    api._ws, api._remote = None, {}
    assert api._remote_problems({"--gguf": "/tmp/whatever.gguf"}) == []


def test_shared_storage_is_not_flagged_as_a_problem(tmp_path, monkeypatch):
    """Most clusters put the same path on every node. Warning there would be noise, and wrong."""
    from pollard_studio import app
    shared = tmp_path / "onnfs.gguf"
    shared.write_bytes(b"x")
    api = app.Api.__new__(app.Api)
    api._ws = None
    api._remote = {str(shared): {"host": "othernode"}}
    assert api._remote_problems({"--gguf": str(shared)}) == []


def test_the_host_name_comes_from_the_box_not_from_a_constant():
    """Whoever runs the agent gets their own hostname -- Sparks, workstations, anything."""
    from pollard_studio import cluster
    import socket as _s
    assert cluster.local()["host"] == _s.gethostname()
    src = (ROOT / "pollard_studio/cluster.py").read_text()
    assert "gethostname()" in src


def test_the_cluster_builds_reach_every_build_dropdown():
    """They arrive after the first render, so the selects have to be refilled or the boxes'
    builds never appear in any list."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "function refillBuildSelects" in js
    assert "refillBuildSelects();" in js[js.index("async function loadRemoteModels"):
                                         js.index("async function renderDevices")]
    ids = set(re.findall(r'sel\("([a-z0-9-]+)", (?:\[\["", "none"\], \.\.\.)?buildOpts\(\)', js))
    listed = set(re.findall(r'"([a-z0-9-]+)"', js[js.index("const BUILD_SELECTS"):
                                                  js.index("function refillBuildSelects")]))
    assert ids <= listed, f"these build selects are never refilled: {sorted(ids - listed)}"


def test_the_cluster_is_loaded_at_boot_not_only_on_one_screen():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    boot = js[js.index("async function boot()"):]
    assert "loadRemoteModels()" in boot[:boot.index("\n}\n") + 3]


# ── the top model picker ────────────────────────────────────────────────────────────────────────

def test_the_model_picker_groups_by_box():
    """The picker is "which model am I working on". On a cluster that question has answers on
    every box, so each gets its own group -- not a flat list where you cannot tell whose is
    whose."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("function renderModelPicker"):]
    fn = fn[:fn.index("\n}\n") + 3]
    assert "optgroup" in fn, "the picker does not group at all"
    assert "this machine" in fn, "local models are not labelled as this box's"
    assert "remoteGroups()" in fn, "the cluster's models never reach the picker"


def test_picking_a_remote_model_does_not_ask_the_local_scanner_for_it():
    """A box's builds come from its agent; a local workspace scan cannot see them."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    # scope to the MODEL picker: there is another el.onchange earlier, on the runtime picker
    fn = js[js.index("function renderModelPicker"):]
    fn = fn[fn.index("el.onchange"):]
    fn = fn[:fn.index("\n  };") + 4]
    assert 'v.startsWith("remote:")' in fn, "a remote pick falls through to bridge().select()"
    i_remote, i_select = fn.index('startsWith("remote:")'), fn.index("bridge().select(")
    assert i_remote < i_select, "the remote branch must come first"


def test_the_rungs_of_one_model_group_under_it():
    """Pollard names builds <model>-Pollard-<TAG>.gguf. Left ungrouped every rung looks like a
    model of its own and the picker becomes unusable on a real workspace."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    assert "function modelNameOf" in js and "function tagOf" in js
    assert "QUANT_TAG" in js


def test_the_picker_is_refreshed_when_the_cluster_answers():
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("async function loadRemoteModels"):js.index("async function renderDevices")]
    assert "renderModelPicker()" in fn, "the picker is built before the cluster replies and never redrawn"


def test_the_picker_names_the_lane_for_every_lane_not_just_gguf():
    """A cluster holds MLX, MX, EXL3, GPTQ and brains as well as GGUFs. "Qwen3-4B (1)" does not
    say which of them you are about to select."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("function renderModelPicker"):]
    fn = fn[:fn.index("\n}\n") + 3]
    assert "b.lane" in fn, "the picker never looks at the lane"
    assert "lanes.join" in fn, "a multi-lane model does not say so"


def test_every_lane_survives_the_grouping(tmp_path):
    """End to end through the agent: a build in each lane must come back as its own model."""
    from pollard_studio import node
    (tmp_path / "rungs").mkdir()
    (tmp_path / "rungs" / "M-Pollard-IQ4_XS.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "rungs" / "brain-v3.pt").write_bytes(b"x" * 1024)
    _mk(tmp_path / "models" / "M-mlx", ["model.safetensors"], cfg={"quantization": {"bits": 4}})
    _mk(tmp_path / "models" / "M-nvfp4", ["model.safetensors"],
        cfg={"quantization_config": {"quant_method": "nvfp4"}})
    _mk(tmp_path / "models" / "M-gptq", ["model.safetensors", "quantize_config.json"])
    _mk(tmp_path / "models" / "M-exl3", ["out_tensor.safetensors"])
    lanes = {m["lane"] for m in node.models(tmp_path)}
    assert lanes == {"GGUF", "BRAIN", "MLX", "MX", "GPTQ", "EXL3"}, f"missing lanes: {lanes}"


def test_a_brain_is_found_whether_its_connectome_was_fly_or_human(tmp_path):
    """pollard-connectome builds either graph and pollard-flybrain trains on whichever it is
    given, so the trained artifact is a .pt either way."""
    from pollard_studio import node
    (tmp_path / "rungs").mkdir()
    (tmp_path / "rungs" / "FlyBrain-Pollard-CNSv1.pt").write_bytes(b"x" * 512)
    (tmp_path / "rungs" / "H01-human-cortex-brain.pt").write_bytes(b"x" * 512)
    got = {m["name"]: m["lane"] for m in node.models(tmp_path)}
    assert got["FlyBrain-Pollard-CNSv1.pt"] == "BRAIN"
    assert got["H01-human-cortex-brain.pt"] == "BRAIN"


def test_a_connectome_is_listed_but_named_for_what_it_is(tmp_path):
    """The H01 human graph is expensive to build and worth finding on a cluster -- but it is the
    thing a brain trains ON, not a model you run."""
    from pollard_studio import node
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "h01_human.feather").write_bytes(b"x" * 512)
    got = node.models(tmp_path)
    assert got[0]["lane"] == "CONNECTOME"


def test_a_connectome_is_not_offered_as_something_to_run():
    """Pointing --gguf at a connectome is a command that cannot work."""
    js = (ROOT / "pollard_studio/ui/app.js").read_text()
    fn = js[js.index("const buildOpts = ()"):js.index("const remoteOf =")]
    assert 'lane !== "CONNECTOME"' in fn, "a connectome can be selected as a build"
