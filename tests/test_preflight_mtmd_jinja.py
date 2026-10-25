"""The engine preflight's mtmd check serves the model with its own chat template (--jinja), as every Pollard
llama.cpp step does -- without it a Gemma 4 template aborts mtmd-cli at start-up on a model that runs fine."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_preflight as PF  # noqa: E402


def test_mtmd_check_passes_jinja(monkeypatch):
    seen = []

    def fake_run(cmd, **k):
        seen.append(cmd)
        return 0, "main: loading model\n chat mode \n"
    monkeypatch.setattr(PF, "_run", fake_run)
    ok, detail = PF._mtmd("llama-mtmd-cli", "m.gguf", "mmproj.gguf")
    assert ok and "--jinja" in seen[0] and seen[0][seen[0].index("--mmproj") + 1] == "mmproj.gguf"


def test_mtmd_check_reports_the_abort(monkeypatch):
    monkeypatch.setattr(PF, "_run", lambda cmd, **k: (134, "libc++abi: terminating due to uncaught exception"))
    ok, detail = PF._mtmd("llama-mtmd-cli", "m.gguf", "mmproj.gguf")
    assert not ok and detail
