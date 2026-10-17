"""pollard release --parallel: shared prep once, a CPU build slot (GGUF quantize) and a GPU build slot (streamed HF
lanes / EXL3 / drafter) side by side, ONE serialized measure queue, admission control on RAM / VRAM / disk (a job that
does not fit waits, never OOMs), lane cards / uploads / the public flip last. Fakes only: no model, no GPU."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_release as R  # noqa: E402

S = R.Step


def _plan():
    st = [S(id="prep/smoke", lane="prep", kind="prep"), S(id="prep/smooth", lane="prep", kind="prep")]
    for lab in ("Q6_K", "IQ4_XS", "IQ3_S"):
        st += [S(id=f"gguf/ladder-{lab}", lane="gguf", rung=lab, kind="build", disk_gb=1),
               S(id=f"gguf/rungs-{lab}", lane="gguf", kind="expand")]
    for r in ("int4", "int8"):
        st += [S(id=f"gptq/{r}/build", lane="gptq", rung=r, kind="build"),
               S(id=f"gptq/{r}/verify", lane="gptq", rung=r, kind="verify"),
               S(id=f"gptq/{r}/publish", lane="gptq", rung=r, kind="publish")]
    st += [S(id="exlr8/encode", lane="exlr8", kind="build"),
           S(id="exlr8/3.0bpw/build", lane="exlr8", rung="3.0bpw", kind="build"),
           S(id="exlr8/3.0bpw/verify", lane="exlr8", rung="3.0bpw", kind="verify")]
    st += [S(id="gguf/card", lane="gguf", kind="card"), S(id="gguf/upload", lane="gguf", kind="upload"),
           S(id="gptq/card", lane="gptq", kind="card"), S(id="release/public", lane="release", kind="finalize")]
    return st


class Open:
    def __init__(self):
        self.calls = []

    def acquire(self, sid, peak, slot, poll_s=30):
        pass

    def release(self, sid):
        pass


def _run(tmp_path, fail=(), dur=0.05, admission=None):
    events, lock = [], threading.Lock()

    def runner(ss):
        t0 = time.time()
        with lock:
            events.append(("start", [s.id for s in ss], t0, threading.current_thread().name))
        if any(s.kind in ("build", "expand", "verify") for s in ss):
            time.sleep(dur)
        res = {s.id: ("failed" if s.id in fail else "done") for s in ss}
        with lock:
            events.append(("end", [s.id for s in ss], time.time(), threading.current_thread().name))
        return res
    c = {"mdir": str(tmp_path), "ngl": "auto"}
    st = R.execute_parallel(_plan(), c, {}, True, out=lambda *_: None, runner=runner,
                            admission=admission or Open(), timeline=str(tmp_path / "tl.log"))
    return st, events


def _span(events, sid):
    s = next(e[2] for e in events if e[0] == "start" and sid in e[1])
    e = next(e[2] for e in events if e[0] == "end" and sid in e[1])
    return s, e


def test_prep_once_first_slots_overlap_measures_serial_final_last(tmp_path):
    st, ev = _run(tmp_path)
    assert ev[0][1] == ["prep/smoke", "prep/smooth"], "shared prep runs ONCE, before anything else"
    assert ev[-1][1] == ["gguf/card", "gguf/upload", "gptq/card", "release/public"], "cards / uploads / public last"
    cpu = _span(ev, "gguf/ladder-Q6_K")
    gpu = _span(ev, "gptq/int4/build")
    assert cpu[0] < gpu[1] and gpu[0] < cpu[1], "the CPU and GPU build slots run at the same time"
    meas = sorted((_span(ev, sid) for sid in ("gguf/rungs-Q6_K", "gguf/rungs-IQ4_XS", "gguf/rungs-IQ3_S",
                                              "gptq/int4/verify", "gptq/int8/verify", "exlr8/3.0bpw/verify")))
    assert all(a[1] <= b[0] + 1e-3 for a, b in zip(meas, meas[1:])), "ONE measure at a time"
    for build, meas_id in (("gguf/ladder-Q6_K", "gguf/rungs-Q6_K"), ("gptq/int8/build", "gptq/int8/verify")):
        assert _span(ev, build)[1] <= _span(ev, meas_id)[0], "a rung is measured after it is built"
    assert _span(ev, "exlr8/encode")[1] <= _span(ev, "exlr8/3.0bpw/build")[0], "the encode before its rungs"
    assert st["gptq/int8/publish"] == "done" and st["release/public"] == "done"
    tl = (tmp_path / "tl.log").read_text()
    assert "cpu" in tl and "gpu" in tl and "measure" in tl and "start" in tl and "end" in tl


def test_a_failed_build_skips_its_measures_and_keeps_nothing_published(tmp_path):
    st, ev = _run(tmp_path, fail={"gptq/int4/build"})
    assert st["gptq/int4/verify"] == "skipped: upstream failed" and st["gptq/int4/publish"] == "kept: upstream failed"
    assert not any("gptq/int4/verify" in e[1] for e in ev), "never run"
    assert st["gptq/int8/publish"] == "done", "the other rungs carry on"


def test_admission_waits_for_room_instead_of_overcommitting(tmp_path):
    cfg = {"disk_margin_gb": 3}
    c = {"mdir": str(tmp_path)}
    adm = R.Admission(cfg, c, free_ram=lambda: 30.0, free_vram=lambda: 15.0, free_disk=lambda: 100.0,
                      out=lambda *_: None)
    adm.acquire("a", (24.0, 0.0, 0.0), "gpu")
    assert not adm.fits((10.0, 0.0, 0.0)), "24 + 10 GB > the 30 GB RAM budget"
    assert adm.fits((4.0, 0.0, 5.0)), "a GGUF quantize fits beside it"
    got = []
    t = threading.Thread(target=lambda: (adm.acquire("b", (10.0, 0.0, 0.0), "measure", poll_s=0.01), got.append(1)))
    t.start()
    time.sleep(0.1)
    assert not got, "b waits while a holds the RAM"
    adm.release("a")
    t.join(2)
    assert got == [1], "b starts once a is done"


def test_admission_checks_free_vram_and_disk_with_what_is_reserved(tmp_path):
    c = {"mdir": str(tmp_path)}
    adm = R.Admission({"disk_margin_gb": 3}, c, free_ram=lambda: 30.0, free_vram=lambda: 4.0,
                      free_disk=lambda: 30.0, out=lambda *_: None)
    adm.acquire("q", (4.0, 0.0, 20.0), "cpu")
    assert not adm.fits((2.0, 0.0, 10.0)), "20 GB reserved + 10 + margin > 30 GB free"
    assert not adm.fits((2.0, 12.0, 0.0)), "12 GB of VRAM wanted, 4 free"
    assert adm.fits((2.0, 0.0, 5.0))
    adm.release("q")
    assert adm.fits((29.0, 15.0, 0.0)), "alone on the machine a job always starts (its own guards decide)"


def test_job_peaks_follow_the_lane():
    c = {"ngl": "0"}
    assert R.job_peak(S(id="gguf/ladder-Q6_K", lane="gguf", rung="Q6_K", kind="build", disk_gb=20), c, {}) == (4.0, 0.0, 20.0)
    assert R.job_peak(S(id="gptq/int8/build", lane="gptq", rung="int8", kind="build"), c, {"max_ram_gb": 24})[0] == 24
    assert R.job_peak(S(id="exl3/3.0bpw/build", lane="exl3", rung="3.0bpw", kind="build"), c, {})[1] > 0


def test_state_entries_from_concurrent_executes_are_never_lost(tmp_path):
    path = str(tmp_path / "state.json")
    ts = [threading.Thread(target=R.save_entry, args=(path, f"s{i}", {"status": "done"})) for i in range(40)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(R.load_state(path)) == 40


def test_rung_chains_group_build_and_expand_and_keep_lane_steps():
    ch = {(l, r): [s.id for s in ss] for l, r, ss in R.rung_chains(_plan())}
    assert ch[("gguf", "Q6_K")] == ["gguf/ladder-Q6_K", "gguf/rungs-Q6_K"]
    assert ch[("exlr8", "_lane")] == ["exlr8/encode"]
    assert ch[("gptq", "int4")] == ["gptq/int4/build", "gptq/int4/verify", "gptq/int4/publish"]
