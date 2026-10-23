"""A .pt brain / training state is loaded with torch's restricted unpickler, never plain pickle.

Security audit: torch.load(..., weights_only=False) on a shared .pt runs whatever its pickle says. The brain,
its .flystate, --continue-from and pollard-forge's resume state now go through pollard_safeload: tensors,
containers, numbers -- plus, for a brain, numpy arrays (its connectome) allowed by name.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

TRIPPED = []


def _tripwire(*a):
    TRIPPED.append(a)
    return {}


class _Evil:
    def __reduce__(self):                    # what a hostile .pt does: a callable run at load time
        return (_tripwire, ("ran",))


def _brain_blob(np, torch):
    return {"src": np.arange(6, dtype=np.int64), "dst": np.arange(6, dtype=np.int64)[::-1].copy(),
            "w": np.linspace(0, 1, 6, dtype=np.float32), "sign": np.ones(6, dtype=np.float32),
            "addr": {"weight": torch.randn(3, 4), "bias": torch.zeros(3)}, "voice": torch.full((1,), 0.1),
            "meta": {"neurons": 6, "width": 328, "hidden": 896, "bits": 18, "codec": "tokens", "lr": 1e-3}}


def test_no_tool_loads_a_pickle_unrestricted():
    """the regression gate: no torch.load(..., weights_only=False) anywhere in tools/"""
    hits = []
    for p in sorted((ROOT / "tools").rglob("*.py")):
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = ln.split("#", 1)[0]
            if "torch.load(" in code and re.search(r"weights_only\s*=\s*False", code):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, hits


def test_a_brain_shaped_payload_round_trips_identically(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from pollard_safeload import safe_torch_load
    blob = _brain_blob(np, torch)
    p = tmp_path / "brain.pt"
    torch.save(blob, p)
    got = safe_torch_load(str(p), numpy=True)
    for k in ("src", "dst", "w", "sign"):
        assert isinstance(got[k], np.ndarray) and got[k].dtype == blob[k].dtype and np.array_equal(got[k], blob[k])
    assert torch.equal(got["addr"]["weight"], blob["addr"]["weight"]) and got["meta"] == blob["meta"]


def test_numpy_is_refused_unless_asked_for(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from pollard_safeload import safe_torch_load
    p = tmp_path / "state.pt"
    torch.save({"opt": {"state": {}, "param_groups": [{"lr": 1e-3, "betas": (0.9, 0.95)}]}, "step": 7}, p)
    assert safe_torch_load(str(p))["step"] == 7                 # forge's resume state: no numpy needed
    torch.save({"x": np.zeros(2)}, p)
    with pytest.raises(Exception):
        safe_torch_load(str(p))


@pytest.mark.parametrize("numpy", [False, True])
def test_a_hostile_pickle_is_refused_and_never_runs(tmp_path, numpy):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from pollard_safeload import safe_torch_load
    p = tmp_path / "evil.pt"
    torch.save({"meta": {"neurons": 1}, "payload": _Evil()}, p)
    TRIPPED.clear()
    with pytest.raises(Exception):
        safe_torch_load(str(p), numpy=numpy)
    assert TRIPPED == []


def test_brainattach_reads_meta_safely_and_still_ships_a_hostile_file(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    import pollard_brainattach as BA
    good = tmp_path / "Good.pt"
    torch.save(_brain_blob(np, torch), good)
    out = tmp_path / "build"
    out.mkdir()
    BA.attach_brain(str(good), str(out), "gguf")
    note = (out / "BRAIN.md").read_text()
    assert "6 x 328" in note and "hidden size 896" in note
    evil = tmp_path / "Evil.pt"
    torch.save({"meta": {"neurons": 1}, "payload": _Evil()}, evil)
    TRIPPED.clear()
    BA.attach_brain(str(evil), str(out), "gguf")
    assert TRIPPED == [] and (out / "Evil.pt").exists()
    assert "could not read brain metadata" in (out / "BRAIN.md").read_text()


def test_flystate_round_trips_through_the_safe_loader(tmp_path):
    """FlyBrain.save_state / load_state: magic + a torch.save of tensors and ints -- weights_only=True is enough."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import pollard_flybrain as FB
    from pollard_safeload import safe_torch_load
    p = tmp_path / "s.flystate"
    with open(p, "wb") as f:
        f.write(FB._STATE_MAGIC)
        torch.save({"mem": torch.randn(4, 8), "z": torch.zeros(4), "carry": None, "neurons": 4, "width": 8}, f)
    with open(p, "rb") as f:
        assert f.read(4) == FB._STATE_MAGIC
        s = safe_torch_load(f)
    assert s["neurons"] == 4 and s["carry"] is None and s["mem"].shape == (4, 8)


def test_the_shipped_brains_load_bit_identically_if_present():
    """On a box holding the real brains: the safe load is the same data the old unrestricted load returned."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from pollard_safeload import safe_torch_load
    d = Path(os.environ.get("POLLARD_BRAINS_DIR") or Path.home() / "Desktop" / "Qstate" / "data" / "brains")
    brains = sorted(d.glob("*.pt")) if d.is_dir() else []
    if not brains:
        pytest.skip("no brains on this machine")
    p = brains[0]
    s, u = safe_torch_load(str(p), numpy=True), torch.load(str(p), map_location="cpu", weights_only=False)
    assert list(s) == list(u)
    for k in ("src", "dst", "w", "sign"):
        assert np.array_equal(s[k], u[k]) and s[k].dtype == u[k].dtype
    assert s["meta"] == u["meta"]
