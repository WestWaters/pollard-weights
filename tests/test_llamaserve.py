"""pollard_llamaserve: how every measurement serves a GGUF (pollard-taskeval / -toolcall / -verify), and the ladder
fixes from the same night. Fakes only: no llama-server, no GPU, no model.

The LFM2.5-24B-A2B release (2026-10-16): `-ngl auto` + four slots put a 48 GB f16 on a 16 GB GPU / 32 GB box at
0.75 tok/s until it died silently; `--rung IQ2` built nothing; the flagship mix was built and never recorded; and
`--rung flagship` also ran an unasked fit-your-RAM IQ3_S."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_llamaserve as LS  # noqa: E402

GB = 1 << 30


def _blocks(n=10, each=GB, other=0):
    return {i: each for i in range(n)}, n * each + other


def test_auto_ngl_comes_from_free_vram_never_the_bare_word(monkeypatch):
    monkeypatch.delenv("POLLARD_GPU_UNTIL", raising=False)
    monkeypatch.setattr(LS, "_block_bytes", lambda g: _blocks(40, 1185 << 20))     # the LFM2.5 f16: ~46 GB
    monkeypatch.setattr(LS.shutil, "which", lambda b: "/usr/bin/nvidia-smi")
    assert LS.resolve_ngl("f16.gguf", "auto", free_mib=15300) == "10", "the last blocks that fit, like ngl_calc.py"
    monkeypatch.setattr(LS, "_block_bytes", lambda g: _blocks(40, 260 << 20))      # a 10 GB rung
    assert LS.resolve_ngl("iq3.gguf", "auto", free_mib=15300) == "99", "the whole file fits: all of it"
    assert LS.resolve_ngl("iq3.gguf", "7", free_mib=15300) == "7", "a number is passed as given"
    assert LS.resolve_ngl("iq3.gguf", "auto", free_mib=0) == "0"


def test_no_nvidia_smi_leaves_auto_to_llama_cpp(monkeypatch):
    monkeypatch.delenv("POLLARD_GPU_UNTIL", raising=False)
    monkeypatch.setattr(LS.shutil, "which", lambda b: None)
    assert LS.resolve_ngl("m.gguf", "auto") == "auto", "Apple unified memory: llama.cpp's own fit"


def test_the_gpu_is_handed_back_by_the_clock(monkeypatch):
    at = lambda h, m: time.mktime((2026, 10, 16, h, m, 0, 0, 0, -1))
    monkeypatch.setenv("POLLARD_GPU_UNTIL", "22:30")
    assert LS.gpu_lent(at(20, 0)) and not LS.gpu_lent(at(22, 30)) and not LS.gpu_lent(at(3, 0))
    monkeypatch.setenv("POLLARD_GPU_UNTIL", "01:30")                  # past midnight
    assert LS.gpu_lent(at(23, 0)) and LS.gpu_lent(at(1, 0)) and not LS.gpu_lent(at(2, 0))
    monkeypatch.setattr(LS.shutil, "which", lambda b: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(LS, "gpu_lent", lambda now=None: False)
    assert LS.resolve_ngl("m.gguf", "auto", free_mib=16000) == "0", "after the deadline: the CPU"
    assert LS.resolve_ngl("m.gguf", "12", free_mib=16000) == "12", "an explicit number is the caller's call"


def test_one_slot_and_a_wait_that_scales_with_the_file(tmp_path):
    cmd = LS.server_cmd("llama-server", "m.gguf", 9000, "10", 4096, ["--jinja"])
    assert cmd[cmd.index("-np") + 1] == "1", "LFM2 serves wrongly with >1 sequence (llama.cpp #29002)"
    assert cmd[cmd.index("-ngl") + 1] == "10" and "--jinja" in cmd
    small, big = tmp_path / "s.gguf", tmp_path / "b.gguf"
    small.write_bytes(b"0")
    with open(big, "wb") as f:
        f.truncate(20 * 10 ** 9)                                        # sparse: 20 GB on paper
    assert LS.load_timeout_s(str(big)) >= LS.load_timeout_s(str(small)) + 400, "a 20 GB load gets minutes more"


def test_a_failed_gpu_start_is_retried_on_the_cpu_with_the_servers_words(tmp_path, monkeypatch):
    monkeypatch.delenv("POLLARD_GPU_UNTIL", raising=False)
    monkeypatch.setattr(LS, "resolve_ngl", lambda g, n, free_mib=None: "10")
    tried = []

    class P:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    def start(cmd, log_path, port, timeout_s, what):
        n = cmd[cmd.index("-ngl") + 1]
        tried.append(n)
        if n != "0":
            Path(log_path).write_text("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000 MiB failed\n")
            raise LS.ServerFailed(f"{what} exited (1) before serving\n{LS.log_tail(log_path)}")
        return P(), f"http://127.0.0.1:{port}", open(log_path, "w")
    said = []
    with LS.serve("m.gguf", "auto", binary="llama-server", log_path=str(tmp_path / "s.log"), timeout_s=5,
                  out=said.append, _start_fn=start) as srv:
        assert srv.ngl == "0" and srv.alive()
    assert tried == ["10", "0"]
    assert any("allocating 12000 MiB failed" in x for x in said), "the server's own reason is shown"
    assert any("retrying on the CPU" in x for x in said)


def test_neither_start_comes_up_raises_with_both_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr(LS, "resolve_ngl", lambda g, n, free_mib=None: "5")

    def start(cmd, log_path, port, timeout_s, what):
        raise LS.ServerFailed(f"{what} not ready after {timeout_s}s")
    with pytest.raises(LS.ServerFailed) as e:
        with LS.serve("m.gguf", "auto", binary="x", log_path=str(tmp_path / "s.log"), timeout_s=3,
                      out=lambda *_: None, _start_fn=start):
            pass
    assert "-ngl 5" in str(e.value) and "-ngl 0" in str(e.value)


def test_start_reports_an_early_exit_with_the_log_tail(tmp_path):
    class Dead:
        returncode = 3

        def __init__(self, *a, stdout=None, **k):
            stdout.write("llama_model_load: error loading model: invalid ggml type 153\n")
            stdout.flush()

        def poll(self):
            return 3
    with pytest.raises(LS.ServerFailed) as e:
        LS._start(["llama-server"], str(tmp_path / "l.log"), 9, 10, "llama-server on m", popen=Dead,
                  sleep=lambda s: None, probe=lambda b: False)
    assert "exited (3)" in str(e.value) and "invalid ggml type 153" in str(e.value)


def test_start_times_out_instead_of_waiting_forever(tmp_path):
    class Slow:
        returncode = None

        def __init__(self, *a, **k):
            self.stopped = False

        def poll(self):
            return None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout=None):
            return 0
    clock = iter(range(0, 10 ** 6, 50))
    import pollard_llamaserve as M
    real = M.time.time
    M.time.time = lambda: next(clock)
    try:
        with pytest.raises(LS.ServerFailed, match="not ready after 100s"):
            LS._start(["llama-server"], str(tmp_path / "l.log"), 9, 100, "llama-server on m", popen=Slow,
                      sleep=lambda s: None, probe=lambda b: False)
    finally:
        M.time.time = real


def test_free_port_steps_off_a_busy_one():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        assert LS.free_port(busy) != busy


def test_the_tools_serve_through_the_helper():
    """taskeval, toolcall and verify all take -ngl and the launch from pollard_llamaserve."""
    src = lambda n: (ROOT / "tools" / n).read_text(encoding="utf-8")
    te = src("pollard_taskeval.py")
    seg = te[te.index("def served("):te.index("def run(")]
    assert "LS.serve(" in seg and "taskeval-server-{stem}.log" in seg, "one log per served model"
    tc = src("pollard_toolcall.py")
    assert "LS.serve(" in tc[tc.index("def served(model"):tc.index("def preflight(")]
    vf = src("pollard_verify.py")
    assert "LS.resolve_ngl(" in vf[vf.index("def measure_gguf("):vf.index("def measure_mlx(")]


def test_taskeval_reference_scores_are_measured_once_per_ladder(tmp_path):
    import pollard_taskeval as T
    ref = tmp_path / "ref.gguf"
    ref.write_bytes(b"GGUF")
    a = T.ref_cache_path(str(ref), ["gsm8k", "ifeval"], 200, str(tmp_path / "reports" / "taskeval" / "Q6_K"))
    b = T.ref_cache_path(str(ref), ["gsm8k", "ifeval"], 200, str(tmp_path / "reports" / "taskeval" / "IQ4_XS"))
    assert a == b, "every rung of the ladder reads the same reference scores"
    assert a != T.ref_cache_path(str(ref), ["gsm8k"], 200, str(tmp_path / "reports" / "taskeval" / "Q6_K"))
    assert a != T.ref_cache_path(str(ref), ["gsm8k", "ifeval"], 100, str(tmp_path / "reports" / "taskeval" / "Q6_K"))


# ------------------------------------------------------------------------------- the ladder (item 3)

def test_the_flagship_is_found_where_automap_writes_it(tmp_path):
    import pollard_auto as A
    import pollard_automap as AM
    assert A.flagship_mix_path(str(tmp_path / "LFM2.5-24B-A2B-f16.gguf")) == str(tmp_path / "LFM2.5-24B-A2B-mix.gguf")
    assert A.flagship_mix_path(str(tmp_path / "M.f16.gguf")) == str(tmp_path / "M-mix.gguf")
    assert A.flagship_mix_path(str(tmp_path / "M-bf16.gguf")) == str(tmp_path / "M-bf16-mix.gguf")
    assert 'stem = re.sub(r"[-.]f16\\.gguf$|\\.gguf$", ""' in Path(AM.__file__).read_text(encoding="utf-8"), \
        "automap's naming rule changed: flagship_mix_path must follow it"


def test_the_flagship_is_recorded_under_the_f16_without_its_suffix(tmp_path, monkeypatch):
    import pollard_auto as A
    import pollard_workspace as ws
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    f16 = tmp_path / "LFM2.5-24B-A2B-f16.gguf"
    f16.write_bytes(b"GGUF")
    (tmp_path / "LFM2.5-24B-A2B-mix.gguf").write_bytes(b"GGUF" * 4)
    monkeypatch.setattr(A, "read_gguf_meta", lambda g: {})
    monkeypatch.setattr(A, "gguf_to_config", lambda m, g: {})
    monkeypatch.setattr(A, "analyse", lambda cfg: {"total": 100})
    e = A._record_flagship(argparse.Namespace(gguf=str(f16)), str(tmp_path))
    assert e and e["flagship"] and e["tag"] == "IQ1_KT"
    assert [b["name"] for b in ws.read_manifest(str(f16))["builds"]] == ["LFM2.5-24B-A2B-mix.gguf"]


def test_rung_flagship_builds_only_the_flagship(monkeypatch):
    """--rung flagship fell through to a bare `pollard-fit --ram 16` and built an unasked IQ3_S first."""
    import pollard_auto as A
    src = Path(A.__file__).read_text(encoding="utf-8")
    seg = src[src.index("want = _wanted_rungs(a)\n    if a.ladder and"):src.index("if a.imatrix and ik_known is False")]
    i_only, i_else = seg.index('elif a.ladder and want == {"flagship"}'), seg.index("else:\n        cmd = [\"pollard-fit\"")
    assert i_only < i_else, "the flagship-only case must be decided before the bare fit build"


def test_rung_iq2_by_name_builds_the_stock_2bit_on_an_ik_box(monkeypatch, capsys):
    import pollard_auto as A
    pts = [(5.0, "IQ2_M", 4.0), (8.0, "IQ3_S", 7.0), (10.0, "IQ4_XS", 9.0), (14.0, "Q6_K", 13.0)]
    monkeypatch.setattr(A, "ladder_points", lambda *a: pts)
    monkeypatch.setattr(A.os.path, "getsize", lambda p: 30 * 10 ** 9)
    ran = []
    monkeypatch.setattr(A, "_run", lambda cmd, run, **k: ran.append(cmd))
    a = argparse.Namespace(gguf="m.gguf", imatrix="m.imatrix", sensitivity=None, bin=None, run=True, rung=["IQ2"])
    A._build_ladder(a, ik_flagship=True)
    assert len(ran) == 1 and ran[0][ran[0].index("--ram") + 1] == "5.0", "the stock IQ2 rung, asked for by name"
    ran.clear()
    a.rung = None
    A._build_ladder(a, ik_flagship=True)
    assert all(c[c.index("--ram") + 1] != "5.0" for c in ran), "the full ladder still leaves 2-bit to the flagship"


def test_threads_follow_pollard_threads(monkeypatch):
    monkeypatch.setenv("POLLARD_THREADS", "19")
    cmd = LS.server_cmd("llama-server", "m.gguf", 9000, "0", 4096, ["--jinja"])
    assert cmd[cmd.index("-t") + 1] == "19" and cmd[cmd.index("-tb") + 1] == "19"
    cmd = LS.server_cmd("llama-server", "m.gguf", 9000, "0", 4096, ["-t", "8"])
    assert cmd.count("-t") == 1, "an explicit -t wins"
    monkeypatch.delenv("POLLARD_THREADS")
    assert "-t" not in LS.server_cmd("llama-server", "m.gguf", 9000, "0", 4096)
