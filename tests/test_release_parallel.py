"""pollard release --parallel: shared prep once, a CPU build slot (GGUF quantize) and a GPU build slot (streamed HF
lanes / EXL3 / drafter) side by side, ONE serialized measure queue, admission control on RAM / VRAM / disk (a job that
does not fit waits, never OOMs), lane cards / uploads / the public flip last. Fakes only: no model, no GPU."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

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


def test_an_unmeasurable_mlx_serve_step_merges_into_state_instead_of_overwriting_it(tmp_path, monkeypatch):
    """An mlx-serve step that cannot be measured here used to write release-state.json back from the snapshot
    taken when execute() started -- while every other step writes through save_entry, which re-reads the file
    and merges, precisely because the parallel scheduler's build slots and measure queue share it. A stale
    wholesale write drops a 'done' entry another slot recorded in between, and that step re-runs next release."""
    path = str(tmp_path / "release-state.json")

    def short(st, c, cfg):
        # between this execute's load_state and its own step write, a parallel slot records its step
        R.save_entry(path, "other-slot/step", {"status": "done"})
        return "mlx_lm.server with the whole rung needs ~400.0 GB of RAM and 1.0 GB is usable here"

    monkeypatch.setattr(R, "memory_short", short)
    monkeypatch.setattr(R, "gpu_returned", lambda: False)
    st = S(id="mlx/4bit/toolcall", lane="mlx", rung="4bit", kind="measure", requires="any",
           cmd=["pollard-toolcall", "--gguf", "x"], mem="mlx-serve")
    c = {"mdir": str(tmp_path), "state": path}
    statuses = R.execute([st], c, {}, run=True, have={"any": True}, out=lambda *_: None)
    assert statuses["mlx/4bit/toolcall"] == "done"
    on_disk = R.load_state(path)
    assert on_disk["mlx/4bit/toolcall"]["status"] == "done"
    assert on_disk["other-slot/step"]["status"] == "done", "the parallel slot's record was overwritten"


def test_rung_chains_group_build_and_expand_and_keep_lane_steps():
    ch = {(l, r): [s.id for s in ss] for l, r, ss in R.rung_chains(_plan())}
    assert ch[("gguf", "Q6_K")] == ["gguf/ladder-Q6_K", "gguf/rungs-Q6_K"]
    assert ch[("exlr8", "_lane")] == ["exlr8/encode"]
    assert ch[("gptq", "int4")] == ["gptq/int4/build", "gptq/int4/verify", "gptq/int4/publish"]


def test_after_the_gpu_deadline_hf_lanes_go_to_the_cpu_and_gpu_only_steps_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_GPU_UNTIL", "22:45")
    monkeypatch.setattr(R, "gpu_returned", lambda: True)
    ran = []
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,), **k: ran.append(list(cmd)))
    c = {"mdir": str(tmp_path), "state": str(tmp_path / "st.json"), "ngl": "auto"}
    steps = [S(id="gptq/int4/build", lane="gptq", rung="int4", kind="build", requires="any", cmd=["pollard-export", "--bits", "4"]),
             S(id="exl3/3.0bpw/build", lane="exl3", rung="3.0bpw", kind="build", requires="any", cmd=["pollard-exl3", "--bpw", "3"]),
             S(id="exl3/3.0bpw/verify", lane="exl3", rung="3.0bpw", kind="verify", requires="any", cmd=["pollard-verify"])]
    st = R.execute(steps, c, {}, run=True, have={"any": True, "cuda": True}, out=lambda *_: None)
    assert ran[0][-2:] == ["--device", "cpu"], "the streamed GPTQ build moves to the CPU"
    assert st["exl3/3.0bpw/build"].startswith("held: GPU-only") and len(ran) == 1
    assert st["exl3/3.0bpw/verify"] == "skipped: upstream failed"


def _execute_plan(tmp_path, monkeypatch, steps, fail=(), have=None, free=False):
    """Exercise the real execute runner across the scheduler's separate jobs."""
    ran = []

    def command(cmd, ok=(0,), **kwargs):
        ran.append(cmd[0])
        if cmd[0] in fail:
            raise RuntimeError("exit 1")

    monkeypatch.setattr(R, "run_cmd", command)
    monkeypatch.setattr(R, "upload", lambda st, *args: ran.append(st.id) or "uploaded")
    c = {"mdir": str(tmp_path), "state": str(tmp_path / "state.json"), "ngl": "0", "free": free}
    statuses = R.execute_parallel(steps, c, {}, run_upload=True, have=have or {"any": True, "cuda": True},
                                  out=lambda *_: None, admission=Open())
    return statuses, ran


def _lane_final(lane):
    return [S(id=f"{lane}/card", lane=lane, kind="card", requires="any", cmd=[f"{lane}-card"]),
            S(id=f"{lane}/upload", lane=lane, kind="upload", requires="any", fn="upload"),
            S(id="release/public", lane="release", kind="finalize", requires="any", fn="go_public",
              lanes=[lane], repo="test/model")]


@pytest.mark.parametrize("held", [False, True])
def test_parallel_prep_blocks_dependent_lane_but_independent_lane_continues(tmp_path, monkeypatch, held):
    prep = S(id="prep/smooth", lane="prep", kind="hold" if held else "prep", requires="any",
             cmd=["smooth"], blocks=["gptq"], desc="smoothing unavailable")
    steps = [prep,
             S(id="gptq/int4/build", lane="gptq", rung="int4", kind="build", requires="any", cmd=["gptq-build"]),
             S(id="gptq/int4/verify", lane="gptq", rung="int4", kind="verify", requires="any", cmd=["gptq-verify"]),
             S(id="gguf/build", lane="gguf", kind="build", requires="any", cmd=["gguf-build"]),
             S(id="gguf/verify", lane="gguf", kind="verify", requires="any", cmd=["gguf-verify"])] + _lane_final("gptq")
    st, ran = _execute_plan(tmp_path, monkeypatch, steps, fail={"smooth"})
    assert "gguf-build" in ran and "gguf-verify" in ran
    assert not any(x.startswith("gptq") for x in ran)
    assert st["gptq/int4/build"] == st["gptq/int4/verify"] == "skipped: upstream failed"
    assert st["gptq/card"] == st["gptq/upload"] == "skipped: upstream failed"
    assert st["release/public"] == "held"


def test_parallel_failed_lane_complete_gate_blocks_card_upload_and_public(tmp_path, monkeypatch):
    steps = [S(id="gguf/build", lane="gguf", kind="build", requires="any", cmd=["gguf-build"]),
             S(id="gguf/complete", lane="gguf", kind="gate", requires="any", blocks=["gguf"], cmd=["complete"]),
             S(id="gptq/int4/build", lane="gptq", rung="int4", kind="build", requires="any", cmd=["gptq-build"]),
             S(id="gptq/int4/verify", lane="gptq", rung="int4", kind="verify", requires="any", cmd=["gptq-verify"])
             ] + _lane_final("gguf")
    st, ran = _execute_plan(tmp_path, monkeypatch, steps, fail={"complete"})
    assert st["gguf/complete"] == "failed"
    assert "gptq-build" in ran and "gptq-verify" in ran
    assert "gguf-card" not in ran and "gguf/upload" not in ran
    assert st["gguf/card"] == st["gguf/upload"] == "skipped: upstream failed"
    assert st["release/public"] == "held"


def test_parallel_unbuildable_lane_does_not_run_card_or_upload(tmp_path, monkeypatch):
    steps = [S(id="exl3/3bpw/build", lane="exl3", rung="3bpw", kind="build", requires=R.REQUIRES["exl3"],
               cmd=["exl3-build"])] + _lane_final("exl3")
    st, ran = _execute_plan(tmp_path, monkeypatch, steps, have={"any": True, R.REQUIRES["exl3"]: False})
    assert ran == []
    assert st["exl3/card"].startswith("skipped: needs") and st["exl3/upload"].startswith("skipped: needs")
    assert st["release/public"] == "held"


@pytest.mark.parametrize("gate_fails", [False, True])
def test_parallel_free_completion_gate_runs_before_expanded_rungs_or_later_builds(tmp_path, monkeypatch, gate_fails):
    rung = tmp_path / "rung.gguf"
    rung.write_bytes(b"GGUF")
    monkeypatch.setattr(R, "canonical_gguf", lambda c: None)
    monkeypatch.setattr(R, "measure_ref", lambda cfg, c: "reference")
    monkeypatch.setattr(R, "gguf_builds", lambda c: [{"tag": "Q6_K", "path": str(rung)}])
    monkeypatch.setattr(R, "gguf_rung_steps", lambda cfg, c, builds: [
        S(id="gguf/Q6_K/verify", lane="gguf", rung="Q6_K", kind="verify", requires="any", cmd=["gguf-verify"]),
        S(id="gguf/Q6_K/publish", lane="gguf", rung="Q6_K", kind="publish", requires="any", cmd=["gguf-publish"])])
    steps = [S(id="gguf/ladder-Q6_K", lane="gguf", rung="Q6_K", kind="build", requires="any", cmd=["first-build"]),
             S(id="gguf/complete", lane="gguf", kind="gate", requires="any", blocks=["gguf"], cmd=["complete"]),
             S(id="gguf/rungs-Q6_K", lane="gguf", kind="expand", requires="any"),
             S(id="gguf/ladder-IQ4_XS", lane="gguf", rung="IQ4_XS", kind="build", requires="any", cmd=["later-build"]),
             S(id="gptq/int4/build", lane="gptq", rung="int4", kind="build", requires="any", cmd=["gptq-build"]),
             S(id="gptq/int4/verify", lane="gptq", rung="int4", kind="verify", requires="any", cmd=["gptq-verify"])]
    st, ran = _execute_plan(tmp_path, monkeypatch, steps, fail={"complete"} if gate_fails else (), free=True)
    assert "first-build" in ran
    assert "gptq-build" in ran and "gptq-verify" in ran
    if gate_fails:
        assert st["gguf/complete"] == "failed"
        assert "gguf-verify" not in ran and "gguf-publish" not in ran and "later-build" not in ran
    else:
        assert st["gguf/complete"] == "done"
        assert ran.index("complete") < ran.index("gguf-verify") < ran.index("gguf-publish")
        assert ran.index("complete") < ran.index("later-build")


def test_a_measure_worker_crash_wakes_the_waiting_slot_and_stops_the_release(tmp_path):
    """A non-SystemExit error in the measure worker must not leave a slot blocked on its lane's gate forever."""
    steps = [S(id="gguf/ladder-Q6_K", lane="gguf", rung="Q6_K", kind="build"),
             S(id="gguf/complete", lane="gguf", kind="gate", blocks=["gguf"]),
             S(id="gguf/rungs-Q6_K", lane="gguf", kind="expand"),
             S(id="gguf/ladder-IQ4_XS", lane="gguf", rung="IQ4_XS", kind="build"),
             S(id="gguf/rungs-IQ4_XS", lane="gguf", kind="expand"),
             S(id="gguf/card", lane="gguf", kind="card"), S(id="release/public", lane="release", kind="finalize")]
    ran = []

    def runner(ss):
        ran.extend(s.id for s in ss)
        if any(s.kind == "gate" for s in ss):
            raise RuntimeError("bug in the measure path")
        return {s.id: "done" for s in ss}

    box = {}

    def go():
        try:
            R.execute_parallel(steps, {"mdir": str(tmp_path), "ngl": "0"}, {}, True, out=lambda *_: None,
                               runner=runner, admission=Open(), timeline=str(tmp_path / "tl.log"))
        except BaseException as e:                  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive(), "the slot waiting on the gate hung"
    assert isinstance(box.get("err"), RuntimeError) and "measure worker stopped" in str(box["err"])
    assert "gguf/ladder-IQ4_XS" not in ran, "no later build of the lane without its gate"
    assert "gguf/card" not in ran and "release/public" not in ran, "nothing final on top of ungated rungs"


def test_a_gate_queued_after_the_measure_worker_died_does_not_hang(tmp_path):
    """The worker dies on ANOTHER lane's job first; a slot that queues its gate afterwards must not wait forever."""
    steps = [S(id="gptq/int4/build", lane="gptq", rung="int4", kind="build"),
             S(id="gptq/int4/verify", lane="gptq", rung="int4", kind="verify"),
             S(id="gguf/ladder-Q6_K", lane="gguf", rung="Q6_K", kind="build"),
             S(id="gguf/complete", lane="gguf", kind="gate", blocks=["gguf"]),
             S(id="gguf/rungs-Q6_K", lane="gguf", kind="expand"),
             S(id="gguf/ladder-IQ4_XS", lane="gguf", rung="IQ4_XS", kind="build"),
             S(id="gguf/card", lane="gguf", kind="card")]
    ran = []

    def runner(ss):
        ran.extend(s.id for s in ss)
        if any(s.id == "gguf/ladder-Q6_K" for s in ss):
            time.sleep(0.3)                          # the GPU lane's measure job crashes first
        if any(s.kind == "verify" for s in ss):
            raise RuntimeError("bug in the measure path")
        return {s.id: "done" for s in ss}

    box = {}

    def go():
        try:
            box["st"] = R.execute_parallel(steps, {"mdir": str(tmp_path), "ngl": "0"}, {}, True, out=lambda *_: None,
                                           runner=runner, admission=Open(), timeline=str(tmp_path / "tl.log"))
        except BaseException as e:                  # noqa: BLE001
            box["err"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive(), "the slot waiting on the gate hung"
    assert isinstance(box.get("err"), RuntimeError)
    assert "gguf/complete" not in ran and "gguf/ladder-IQ4_XS" not in ran and "gguf/card" not in ran
