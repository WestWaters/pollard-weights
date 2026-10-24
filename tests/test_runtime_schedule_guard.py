"""pollard-runtime --schedule: the scheduled engine update never collides with a build, and a refused update says why.

The Nemotron-3.5-Bolt release failed four times over two nights on engine plumbing. One of them was a captured patch
(qwen3omni-deepstack) that had drifted and blocked `--update`, which nobody saw until a release needed the update.
These pin: the guard (machine lock, heavy-lock dirs, release / llama-* processes, the box's SYSTEM release tasks),
skip-and-retry, the re-check before the swap, the recorded status `--check` prints, no half-built .next left on disk,
and the Windows task line (past start date, SYSTEM, low priority, half the cores).
"""
import os
import plistlib
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_runtime_update as U  # noqa: E402

# a real row of `schtasks /query /fo csv /v /nh` on the build box while the Bolt release ran (2026-10-24)
BOX_TASK_ROWS = (
    '"MYWORKSTATION","\\BoltLaneA","N/A","Running","Interactive/Background","10/24/2026 1:52:46 AM","267009",'
    '"Myworkstation\\jwate","C:\\pollard\\release\\boltday.bat ","N/A","N/A","Enabled","Disabled",'
    '"Stop On Battery Mode, No Start On Batteries","SYSTEM","Disabled","72:00:00","Scheduling data is not available '
    'in this format.","One Time Only","12:00:00 AM","1/1/2026","N/A","N/A","N/A","Disabled","Disabled","Disabled",'
    '"Disabled"\n'
    '"MYWORKSTATION","\\build7b","N/A","Ready","Interactive/Background","N/A","0","Myworkstation\\jwate",'
    '"C:\\pollard\\build7b.bat","N/A","N/A","Enabled","Disabled","","SYSTEM","Disabled","72:00:00","","One Time Only",'
    '"12:00:00 AM","1/1/2026","N/A","N/A","N/A","Disabled","Disabled","Disabled","Disabled"\n'
    '"MYWORKSTATION","\\PollardRuntimeUpdate","N/A","Running","Interactive/Background","N/A","0","x",'
    '"C:\\pollard\\pw\\runtime\\runtime-update.bat","N/A","N/A","Enabled","Disabled","","SYSTEM","Disabled",'
    '"06:00:00","","Daily","4:30:00 AM","1/1/2026","N/A","N/A","N/A","Disabled","Disabled","Disabled","Disabled"\n'
    '"MYWORKSTATION","\\Google Play Games Notifier","N/A","Running","Interactive/Background","N/A","0","x",'
    '"C:\\Program Files\\Google\\notifier.exe","N/A","N/A","Enabled","Disabled","","jwate","Disabled","72:00:00","",'
    '"Daily","4:30:00 AM","1/1/2026","N/A","N/A","N/A","Disabled","Disabled","Disabled","Disabled"\n')


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "STATUS_FILE", str(tmp_path / "update-status.json"))
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POLLARD_MACHINE_LOCK", str(tmp_path / "home" / ".pollard.lock"))
    monkeypatch.delenv("POLLARD_HEAVY_LOCK", raising=False)


def _live_pid():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    return p


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ---- what counts as a heavy job -------------------------------------------------------------------------------

@pytest.mark.parametrize("cmd,heavy", [
    ("/usr/bin/python3 -m pollard_release --hf X --run", True),
    ("/Users/m/.venv/bin/python /Users/m/.venv/bin/pollard --hf Qwen/Q --run --ladder", True),
    ("C:\\pollard\\venv312\\Scripts\\python.exe -X utf8 C:\\pollard\\release\\clefnight_run.py -- --hf C:\\src\\M "
     "--plan p.yaml --run --upload", True),
    ("pollard-toolcall --gguf C:\\a.gguf --rung PT75", True),
    ("/Users/m/pw/runtime/llama.cpp/build/bin/llama-imatrix -m x.gguf", True),
    ("C:\\pollard\\pw\\runtime\\llama.cpp\\build\\bin\\Release\\llama-quantize.exe a b Q4_K", True),
    ("python3 runtime/llama.cpp/convert_hf_to_gguf.py nvidia/Bolt --outtype bf16", True),
    ("sh -c .venv/bin/python runtime/llama.cpp/convert_hf_to_gguf.py nvidia/Bolt", True),
    ("python3 /pw/tools/pollard_runtime.py --update llama.cpp", True),
    ("python3 /pw/tools/pollard_runtime.py --update all --check", False),
    ("/bin/zsh -c cd ~/pollard && ls", False),
    ("python3 /pw/tools/pollard_ls.py", False),
    ("/Applications/Pollard Studio.app/Contents/MacOS/pollard-studio", False),
    ("/usr/sbin/cfprefsd agent", False),
])
def test_heavy_process(cmd, heavy):
    assert bool(U.heavy_process(cmd.split())) is heavy


def test_running_release_tasks_reads_the_box_csv():
    # BoltLaneA (SYSTEM, running, a pollard path) counts; a Ready task, our own update task and a user's
    # running non-Pollard task do not
    assert U.running_release_tasks(BOX_TASK_ROWS) == ["\\BoltLaneA"]


# ---- the guard ----------------------------------------------------------------------------------------------

def test_a_free_machine_is_not_busy():
    assert U.busy_reasons(procs=[(1, 0, ["/sbin/launchd"])], tasks_text="") == []


def test_machine_lock_held_by_a_live_job_is_busy(tmp_path):
    p = _live_pid()
    try:
        lock = tmp_path / "home" / ".pollard.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text('{"pid": %d, "what": "pollard-release Bolt-E4B gguf/ladder"}' % p.pid)
        why = U.busy_reasons(procs=[], tasks_text="")
        assert len(why) == 1 and "machine lock" in why[0] and "Bolt-E4B" in why[0]
    finally:
        p.kill()


def test_stale_machine_lock_is_not_busy(tmp_path):
    lock = tmp_path / "home" / ".pollard.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text('{"pid": %d, "what": "old"}' % _dead_pid())
    assert U.busy_reasons(procs=[], tasks_text="") == []


def test_heavy_lock_dirs(tmp_path, monkeypatch):
    held, stale, free = tmp_path / "mac-heavy.lock", tmp_path / "stale.lock", tmp_path / "free.lock"
    p = _live_pid()
    try:
        held.mkdir()
        (held / "owner").write_text(f"{p.pid} E4B release 2026-10-24\n")
        stale.mkdir()
        (stale / "owner").write_text(f"{_dead_pid()} gone\n")
        monkeypatch.setenv("POLLARD_HEAVY_LOCK", os.pathsep.join(map(str, (held, stale, free))))
        why = U.busy_reasons(procs=[], tasks_text="")
        assert len(why) == 1 and str(held) in why[0] and "E4B release" in why[0]
    finally:
        p.kill()


def test_heavy_lock_dir_without_owner_yet_is_busy(tmp_path, monkeypatch):
    d = tmp_path / "just-taken.lock"
    d.mkdir()
    monkeypatch.setenv("POLLARD_HEAVY_LOCK", str(d))
    assert U.busy_reasons(procs=[], tasks_text="")


def test_release_processes_are_busy_but_not_our_own_ancestors():
    me, parent = os.getpid(), os.getppid()
    procs = [(parent, 1, ["python3", "-m", "pollard_release", "--hf", "X"]),    # whoever started us: excluded
             (me, parent, ["python3", "pollard_runtime.py", "--update", "all"]),
             (4242, 1, ["/rt/llama.cpp/build/bin/llama-perplexity", "-m", "x"])]
    why = U.busy_reasons(procs=procs, tasks_text="")
    assert why == ["llama-perplexity is running (pid 4242)"]


def test_box_release_task_is_busy():
    why = U.busy_reasons(procs=[], tasks_text=BOX_TASK_ROWS)
    assert why == ["the release task \\BoltLaneA is running"]


# ---- skip, retry, status -------------------------------------------------------------------------------------

def test_scheduled_update_skips_while_busy_and_says_why(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(U, "update_engine", lambda *a, **k: called.append(a) or True)
    ok = U.scheduled_update(["llama.cpp", "ik_llama.cpp"], "daily", busy=lambda: ["the release task \\BoltLaneA is running"])
    assert ok and called == []
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "BoltLaneA" in out and "retries" in out
    st = U.read_status()
    assert st["llama.cpp"]["state"] == "skipped" and "BoltLaneA" in st["llama.cpp"]["reason"]
    assert "_scheduled" not in st                       # a skip is not a run: the next hourly attempt is still due
    assert "skipped" in U.status_line("llama.cpp")


def test_scheduled_update_runs_once_per_period(monkeypatch):
    import datetime as dt
    called = []

    def fake(n, **k):
        called.append((n, k.get("jobs"), k.get("guard") is not None))
        U.write_status(n, "current")
        return True
    monkeypatch.setattr(U, "update_engine", fake)
    t0 = dt.datetime(2026, 10, 25, 4, 30)
    assert U.scheduled_update(["llama.cpp"], "daily", jobs=16, busy=lambda: [], now=t0)
    assert called == [("llama.cpp", 16, True)]                    # the guard rides along to the swap
    assert U.scheduled_update(["llama.cpp"], "daily", busy=lambda: [], now=t0 + dt.timedelta(hours=1))
    assert len(called) == 1                                        # 05:30 retry: already ran today
    assert U.scheduled_update(["llama.cpp"], "daily", busy=lambda: [], now=t0 + dt.timedelta(days=1))
    assert len(called) == 2


def test_a_deferred_swap_stays_due(monkeypatch):
    def fake(n, **k):
        U.write_status(n, "deferred", "built, not swapped: pollard-release is running (pid 1)")
        return False
    monkeypatch.setattr(U, "update_engine", fake)
    assert not U.scheduled_update(["llama.cpp"], "daily", busy=lambda: [])
    assert "_scheduled" not in U.read_status()


def _fake_engine(tmp_path, monkeypatch, block=(), guard_says=()):
    tree = str(tmp_path / "rt" / "llama.cpp")
    os.makedirs(tree)
    monkeypatch.setitem(U.ENGINES, "llama.cpp", {"url": "https://github.com/ggml-org/llama.cpp", "dir": tree})
    monkeypatch.setattr(U, "trust_trees", lambda *a: None)
    monkeypatch.setattr(U, "upstream_head", lambda url, ref=None: ("8e5a58d00", "2026-10-24"))
    monkeypatch.setattr(U, "live_state", lambda t: {"present": True, "git": True, "commit": "ccdbe2900",
                                                    "date": "2026-10-12", "dirty": False, "bin": None})
    monkeypatch.setattr(U, "cache_flags", lambda *a, **k: {"GGML_METAL": "ON"})
    monkeypatch.setattr(U, "_patches_for", lambda t: [])

    def run(cmd, **k):
        if cmd[:2] == ["git", "clone"]:
            os.makedirs(os.path.join(cmd[-1], "build"))       # the staged clone, gigabytes in real life
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U, "_run", run)
    monkeypatch.setattr(U, "reapply", lambda s, t, fresh=False: (list(block), ["re-applied x"]))
    monkeypatch.setattr(U, "build", lambda s, f, j: (True, "built all targets"))
    monkeypatch.setattr(U, "smoke", lambda s: (True, "ok"))
    monkeypatch.setattr(U, "lost_archs", lambda a, b: [])
    monkeypatch.setattr(U, "_git", lambda *a: "8e5a58d00")
    monkeypatch.setattr(U, "REPO", str(tmp_path))                 # the lock.json a swap writes lands here
    os.makedirs(os.path.join(str(tmp_path), "runtime"), exist_ok=True)
    return tree


DRIFT = ("llama.cpp-qwen3omni-deepstack.patch is live in this engine and does not apply to upstream -- rebase it "
         "first (it changes conversion/qwen3vl.py, tools/mtmd/mtmd.cpp; git: error: patch failed)")


def test_a_patch_that_needs_a_rebase_blocks_and_check_says_so(tmp_path, monkeypatch, capsys):
    tree = _fake_engine(tmp_path, monkeypatch, block=[DRIFT])
    assert not U.update_engine("llama.cpp")
    assert not os.path.exists(tree + ".next")                     # no half-built tree left taking disk
    assert os.path.isdir(tree)                                    # the live build is untouched
    st = U.read_status()["llama.cpp"]
    assert st["state"] == "blocked" and st["patches"] == ["llama.cpp-qwen3omni-deepstack"]
    line = U.status_line("llama.cpp")
    assert line.startswith("update blocked: llama.cpp-qwen3omni-deepstack needs a rebase")
    capsys.readouterr()
    assert U.update_engine("llama.cpp", check=True)
    assert "update blocked: llama.cpp-qwen3omni-deepstack needs a rebase" in capsys.readouterr().out


def test_cli_check_alone_reports_every_engine(tmp_path, monkeypatch):
    import pollard_runtime as R
    seen = []
    monkeypatch.setattr(U, "update_engine", lambda n, **k: seen.append((n, k["check"])) or True)
    monkeypatch.setattr(sys, "argv", ["pollard-runtime", "--check"])
    assert R.main() == 0
    assert seen == [(n, True) for n in U.ENGINES]


def test_a_failed_build_leaves_no_staging(tmp_path, monkeypatch):
    tree = _fake_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(U, "build", lambda s, f, j: (False, "build step failed: cmake --build\nerror"))
    assert not U.update_engine("llama.cpp")
    assert not os.path.exists(tree + ".next")
    assert U.read_status()["llama.cpp"]["state"] == "failed"


def test_guard_is_asked_again_before_the_swap(tmp_path, monkeypatch):
    tree = _fake_engine(tmp_path, monkeypatch)
    open(os.path.join(tree, "LIVE"), "w").close()
    assert not U.update_engine("llama.cpp", guard=lambda: ["pollard-release is running (pid 77)"])
    assert os.path.exists(os.path.join(tree, "LIVE"))              # never swapped under the release
    assert not os.path.exists(tree + ".next") and not os.path.exists(tree + ".prev")
    assert U.read_status()["llama.cpp"]["state"] == "deferred"


def test_a_clean_update_swaps_and_keeps_prev(tmp_path, monkeypatch):
    tree = _fake_engine(tmp_path, monkeypatch)
    open(os.path.join(tree, "LIVE"), "w").close()
    assert U.update_engine("llama.cpp", guard=lambda: [])
    assert os.path.isfile(os.path.join(str(tmp_path), "runtime", "llama.cpp.lock.json"))
    assert os.path.exists(os.path.join(tree + ".prev", "LIVE")) and os.path.isdir(os.path.join(tree, "build"))
    assert not os.path.exists(os.environ["POLLARD_MACHINE_LOCK"])  # the swap's hold is released
    assert U.read_status()["llama.cpp"]["state"] == "live"


# ---- the task itself ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("when,sc", [("daily", "/sc daily"), ("weekly", "/sc weekly /d SUN")])
def test_windows_task_line(when, sc):
    cmd = U.windows_task_command(when, r"C:\pollard\pw\runtime\runtime-update.bat")
    assert cmd.startswith(f"schtasks /create /tn {U.TASK_NAME} ")
    assert f" {sc} " in cmd
    assert "/sd 01/01/2026" in cmd                    # past start date: it can never fire on its own at creation
    assert "/sc once" not in cmd
    assert "/ru SYSTEM /rl HIGHEST" in cmd            # survives logoff and the console's Ctrl+C
    assert "/ri 60 /du 08:00" in cmd                  # hourly retries while a release holds the machine
    assert cmd.endswith(" /f")


def test_windows_task_line_quotes_a_path_with_spaces():
    cmd = U.windows_task_command("daily", r"C:\Program Files\pw\runtime\runtime-update.bat")
    assert '/tr "\\"C:\\Program Files\\pw\\runtime\\runtime-update.bat\\""' in cmd


def test_windows_bat_runs_low_priority_on_half_the_cores(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    jobs = U.scheduled_jobs()
    assert jobs == 16
    bat = U.windows_task_bat("weekly", r"C:\pollard\venv312\Scripts\python.exe", r"C:\pollard\pw\tools\pollard_runtime.py",
                             jobs, r"C:\pollard\pw\runtime\update.log", {"POLLARD_HOME": r"C:\Users\jwate\pollard"})
    assert 'set "POLLARD_HOME=C:\\Users\\jwate\\pollard"' in bat      # SYSTEM's ~ is not the user's
    assert ('start "" /low /b /wait "C:\\pollard\\venv312\\Scripts\\python.exe" "C:\\pollard\\pw\\tools\\pollard_runtime.py" '
            "--update all --scheduled weekly --jobs 16 >>") in bat
    assert bat.endswith("\r\n")


def test_launchd_plist(tmp_path):
    xml = U.launchd_plist("weekly", "/venv/bin/python", "/pw/tools/pollard_runtime.py", 7, "/pw/runtime/update.log",
                          {"POLLARD_HOME": "/Users/m/pollard", "PATH": "/opt/homebrew/bin:/usr/bin"})
    d = plistlib.loads(xml.encode())
    assert d["ProgramArguments"] == ["/venv/bin/python", "/pw/tools/pollard_runtime.py", "--update", "all",
                                     "--scheduled", "weekly", "--jobs", "7"]
    assert d["EnvironmentVariables"]["POLLARD_HOME"] == "/Users/m/pollard"
    cal = d["StartCalendarInterval"]
    assert len(cal) == U.RETRY_HOURS and cal[0] == {"Weekday": 0, "Hour": 4, "Minute": 30}
    assert d["Nice"] == 15 and d["LowPriorityIO"] is True


def test_schedule_on_windows_writes_the_bat_and_runs_the_task_line(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(U, "WIN", True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(U, "REPO", str(tmp_path))
    monkeypatch.setattr(U, "_run", lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    U.schedule("daily")
    bat = os.path.join(str(tmp_path), "runtime", "runtime-update.bat")
    assert os.path.isfile(bat) and "--scheduled daily" in open(bat).read()
    assert ran[0] == U.windows_task_command("daily", bat)
    assert "Priority=7" in ran[1]
