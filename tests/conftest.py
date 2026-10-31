"""Shared test isolation."""
import os
import tempfile

# No test reaches the real Hugging Face Hub, however a mock is written. Set before anything imports huggingface_hub
# (it reads these once, at import): offline mode, and an empty HF home so no stored login token is found either.
# 2026-10-31: a wrapper that bypassed the tests' HfApi stub let the release tests create a real public repo
# (PollardWeights/Tiny-1B-Pollard) with the developer's own token.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="pollard-tests-hf-")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_model_arch(monkeypatch):
    """POLLARD_ARCH steers every binary lookup (pollard_calc.set_model_arch). A tool's main() sets it in os.environ,
    and without this one test's model would steer the next test's lookups."""
    monkeypatch.delenv("POLLARD_ARCH", raising=False)


@pytest.fixture(autouse=True)
def _private_machine_locks(monkeypatch, tmp_path):
    """The release takes the machine lock around every compute step (pollard_auto.heavy_step). A test must neither
    wait behind a real job on this machine nor leave the real ~/pollard/.pollard.lock or a shared POLLARD_HEAVY_LOCK
    behind: each test gets its own lock file and no shared lock directory."""
    monkeypatch.setenv("POLLARD_MACHINE_LOCK", str(tmp_path / ".pollard.lock"))
    monkeypatch.delenv("POLLARD_HEAVY_LOCK", raising=False)
    monkeypatch.delenv("POLLARD_HEAVY_LOCK_HELD", raising=False)


@pytest.fixture(autouse=True)
def _no_engine_preflight(monkeypatch):
    """The engine preflight (pollard_preflight, prep/engines in a release, before the imatrix in `pollard --run`) loads
    the real model with the real binaries. Tests that fake the pipeline have neither, so it is off unless a test turns
    it back on (tests/test_engine_preflight.py)."""
    monkeypatch.setenv("POLLARD_ENGINE_PREFLIGHT", "0")
