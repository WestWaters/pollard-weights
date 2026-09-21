"""--threads: a build should not have to take the whole machine.

Every heavy step here shells out to a binary or hands work to torch, and both default to every
core. On a shared box -- someone else's GPU, a desktop that has to stay usable -- that is the
difference between a build you can start and one you have to wait for the night for.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


@pytest.mark.parametrize("tool", ["pollard_fit.py", "pollard_bench.py", "pollard_gptq.py"])
def test_the_flag_exists(tool):
    assert '"--threads"' in (TOOLS / tool).read_text()


@pytest.mark.parametrize("tool", ["pollard_fit.py", "pollard_bench.py"])
def test_the_env_var_configures_a_shared_box_once(tool):
    src = (TOOLS / tool).read_text()
    assert "POLLARD_THREADS" in src
    assert "def _env_threads" in src


def test_fit_appends_nthreads_as_a_trailing_positional():
    """llama-quantize takes nthreads last, after the type -- not as a flag."""
    src = (TOOLS / "pollard_fit.py").read_text()
    i = src.index("a.gguf, out, base_preset]")
    seg = src[i:i + 320]
    assert "if a.threads:" in seg and 'cmd += [str(a.threads)]' in seg


def test_bench_passes_dash_t_to_every_binary_it_spawns():
    """Three call sites -- perplexity, the gate generator, the speed run. One helper, so they
    cannot drift apart."""
    src = (TOOLS / "pollard_bench.py").read_text()
    assert "def _t():" in src
    assert src.count("+ _t()") >= 3, "a subprocess builder is missing the thread flag"


def test_bench_sets_the_global_from_the_flag():
    src = (TOOLS / "pollard_bench.py").read_text()
    assert re.search(r"global THREADS\s*\n\s*THREADS = a\.threads", src)


def test_gptq_limits_torch_rather_than_a_subprocess():
    """gptq does the work in-process, so the flag has to reach torch."""
    src = (TOOLS / "pollard_gptq.py").read_text()
    assert "torch.set_num_threads(a.threads)" in src


def test_unset_means_the_tool_decides():
    """Absent the flag, nothing is imposed -- a default thread count is a policy, and guessing one
    for someone else's machine is worse than leaving the tool's own."""
    src = (TOOLS / "pollard_bench.py").read_text()
    seg = src[src.index("def _t():"):]
    seg = seg[:seg.index("\n\n")]
    assert "if THREADS else []" in seg
    fit = (TOOLS / "pollard_fit.py").read_text()
    assert "default=_env_threads()" in fit


def test_env_threads_survives_a_bad_value():
    src = (TOOLS / "pollard_fit.py").read_text()
    seg = src[src.index("def _env_threads"):]
    seg = seg[:seg.index("\n\ndef ")]
    ns = {"os": __import__("os")}
    exec(seg, ns)
    import os
    for bad, expect in (("", None), ("0", None), ("-4", None), ("nonsense", None), ("19", 19)):
        os.environ["POLLARD_THREADS"] = bad
        assert ns["_env_threads"]() == expect, bad
    os.environ.pop("POLLARD_THREADS", None)
