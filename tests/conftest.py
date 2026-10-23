"""Shared test isolation."""
import pytest


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
