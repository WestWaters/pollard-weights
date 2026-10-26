"""prep/imatrix keeps its progress: llama-imatrix saves a partial every IMATRIX_SAVE_EVERY chunks, and a run handed
back from the GPU (the CPU retry), or a step killed and re-run, resumes from that partial (--in-file + --chunk) instead
of chunk 0 -- 2026-10-25: a 17:30 GPU hand-back threw away 508 of Nanbeige4.3-33B's 1000 chunks. A partial from another
run (other host / corpus / shape / budget) is never mixed in. Fakes only: no model, no runtime."""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_release as R  # noqa: E402


def dat(path, chunks):
    """A legacy .dat imatrix (llama.cpp save_imatrix_legacy layout) that holds `chunks` chunks."""
    name = b"blk.0.ffn_up.weight"
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 1) + struct.pack("<i", len(name)) + name + struct.pack("<ii", chunks, 2))
        f.write(struct.pack("<ff", 1.0, 2.0) + struct.pack("<i", chunks))
        ds = b"calib.txt"
        f.write(struct.pack("<i", len(ds)) + ds)


def test_chunks_done_reads_the_trailing_chunk_count(tmp_path):
    dat(tmp_path / "a.dat", 508)
    assert R.imatrix_chunks_done(str(tmp_path / "a.dat")) == 508
    (tmp_path / "b.dat").write_bytes(b"\x01\x00")                 # cut short
    assert R.imatrix_chunks_done(str(tmp_path / "b.dat")) == 0
    assert R.imatrix_chunks_done(str(tmp_path / "missing.dat")) == 0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    host = tmp_path / "m-Q8_0-host.gguf"
    host.write_bytes(b"GGUF")
    calib = tmp_path / "calib.txt"
    calib.write_text("x" * 100)
    c = {"imatrix": str(tmp_path / "cal" / "m.imatrix"), "calib": str(calib), "f16": str(host), "ngl": "auto",
         "gguf_dir": str(tmp_path)}
    cfg = {"imatrix_host": str(host), "imatrix_chunks": 1000}
    import pollard_calc
    import pollard_auto
    monkeypatch.setattr(pollard_calc, "find_llama_bin", lambda name, arch=None: f"/bin/{name}")
    monkeypatch.setattr(pollard_auto, "_imatrix_shape", lambda g: ["-c", "512", "-b", "512", "-ub", "512"])
    monkeypatch.setattr(R, "imatrix_ngl", lambda cfg, c, host: "17")
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": ""})())
    return c, cfg


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_the_cpu_retry_resumes_where_the_gpu_run_was_handed_back(setup, monkeypatch):
    c, cfg = setup
    ran = []

    def fake(cmd, ok=(0,)):
        ran.append(cmd)
        out = _arg(cmd, "-o")
        if _arg(cmd, "-ngl") == "17":
            dat(out, 500)                                         # saved at chunk 500, then the GPU was taken back
            raise RuntimeError("exit 1")
        assert R.imatrix_chunks_done(_arg(cmd, "--in-file")) == 500, "loads what the GPU run computed"
        dat(out, 1000)                                            # cumulative: the loaded 500 + 500 more
    monkeypatch.setattr(R, "run_cmd", fake)
    assert R.build_imatrix(c, cfg, out=lambda *_: None) == c["imatrix"]
    gpu, cpu = ran
    assert _arg(gpu, "--chunks") == "1000" and "--in-file" not in gpu and _arg(gpu, "--output-frequency") == "10"
    assert _arg(cpu, "-ngl") == "0" and _arg(cpu, "--chunk") == "500" and _arg(cpu, "--chunks") == "500"
    assert R.imatrix_chunks_done(c["imatrix"]) == 1000
    left = sorted(os.listdir(os.path.dirname(c["imatrix"])))
    assert left == ["m.imatrix"], "the kept partial and run records are cleaned up once the imatrix is in place"


def test_a_killed_step_rerun_resumes_and_a_foreign_partial_is_dropped(setup, monkeypatch):
    c, cfg = setup
    os.makedirs(os.path.dirname(c["imatrix"]), exist_ok=True)
    monkeypatch.setattr(R, "imatrix_ngl", lambda cfg, c, host: "0")
    calls = []

    def killed(cmd, ok=(0,)):
        calls.append(cmd)
        dat(_arg(cmd, "-o"), 230)
        raise RuntimeError("exit -1")                             # the process was killed at chunk ~235
    monkeypatch.setattr(R, "run_cmd", killed)
    with pytest.raises(RuntimeError):
        R.build_imatrix(c, cfg, out=lambda *_: None)

    def finish(cmd, ok=(0,)):
        calls.append(cmd)
        dat(_arg(cmd, "-o"), 1000)
    monkeypatch.setattr(R, "run_cmd", finish)
    R.build_imatrix(c, cfg, out=lambda *_: None)                  # the step re-run
    assert _arg(calls[-1], "--chunk") == "230" and _arg(calls[-1], "--chunks") == "770"

    # another corpus: the old partial is not this computation -- dropped, started at chunk 0
    os.remove(c["imatrix"])
    dat(c["imatrix"] + ".partial", 400)
    R._write_json(c["imatrix"] + ".partial.run.json", {"other": "run"})
    R.build_imatrix(c, cfg, out=lambda *_: None)
    assert "--in-file" not in calls[-1] and _arg(calls[-1], "--chunks") == "1000"
