"""Shared test isolation."""
import pytest


@pytest.fixture(autouse=True)
def _no_model_arch(monkeypatch):
    """POLLARD_ARCH steers every binary lookup (pollard_calc.set_model_arch). A tool's main() sets it in os.environ,
    and without this one test's model would steer the next test's lookups."""
    monkeypatch.delenv("POLLARD_ARCH", raising=False)
