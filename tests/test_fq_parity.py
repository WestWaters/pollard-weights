"""pollard-fq's HARD GATE: fq(W, imatrix) must equal dequant(quantize(W, imatrix)) from the real ggml,
bit for bit, for every type in pollard_fq.SUPPORTED -- random and structured tensors, three imatrix shapes.

Three layers, so CI still guards something:
  * live parity against the runtime's libggml (runtime/llama.cpp/build/bin, or POLLARD_LLAMA_BIN_DIR) and,
    for STQ1_0, a reference compiled from Pollard's patched llama-stq tree -- skipped when absent;
  * golden hashes (tests/fixtures/fq_golden.json) of ggml's own output on deterministic inputs, which the
    port must reproduce with no ggml present (IQ2 types also need the runtime's ggml sources for the grids);
  * the training-side contract: STE gradient, FakeQuantLinear, refusals.
Regenerate the goldens (needs the runtime build):  python tests/test_fq_parity.py --write-golden
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pollard_fq as fq  # noqa: E402
import ggml_ref  # noqa: E402

GOLDEN = ROOT / "tests" / "fixtures" / "fq_golden.json"
IQ2 = ("iq2_xxs", "iq2_xs", "iq2_s")


def _lcg(n, seed):
    """Deterministic uint32 stream (numpy-version independent) -> float32 in [-0.5, 0.5)."""
    x = (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) + np.uint64(seed * 40503 + 12345)) % np.uint64(2 ** 32)
    x = (x * np.uint64(1103515245) + np.uint64(12345)) % np.uint64(2 ** 32)
    return ((x >> np.uint64(8)).astype(np.float32) * np.float32(2.0 ** -24)) - np.float32(0.5)


def inputs(rows=8, n=512):
    """Named deterministic weight matrices exercising the encoders' edge cases."""
    u = _lcg(rows * n, 1).reshape(rows, n)
    v = _lcg(rows * n, 2).reshape(rows, n)
    out = {
        "uniform": u * np.float32(0.04),
        "heavy": (u * u * u) * np.float32(0.6) + v * np.float32(0.002),        # cubic: long tails
        "ties": np.round(u * np.float32(16)) / np.float32(128),                  # exact ties everywhere
        "sparse": np.where(v > 0.2, u * np.float32(0.04), np.float32(0)),
        "outlier": np.where(v > 0.49, np.float32(1), u * np.float32(0.02)),
        "tiny": u * np.float32(1e-6),
        "zeroblk": np.concatenate([np.zeros((rows, 256), np.float32), u[:, 256:] * np.float32(0.04)], 1),
        "const": np.full((rows, n), np.float32(0.01)) * np.where(v > 0, np.float32(1), np.float32(-1)),
    }
    return {k: np.ascontiguousarray(v_, dtype=np.float32) for k, v_ in out.items()}


def imatrices(n=512):
    u = _lcg(n, 3) + np.float32(0.5)
    return {"none": None,
            "uniform": np.float32(0.05) + u * np.float32(5),
            "spiky": np.float32(0.01) + (u * u * u * u * u * u * u * u) * np.float32(200),
            "deadch": np.where(u > 0.2, np.float32(0.1) + u, np.float32(0))}


def combos():
    for t in fq.SUPPORTED:
        for iname, im in imatrices().items():
            if im is None and t in fq.NEEDS_IMATRIX:
                continue
            if t == "tq2_0" and iname not in ("none", "uniform"):
                continue                                       # tq2_0 ignores the imatrix
            yield t, iname, im


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _port(t, W, im):
    return fq.quant_dequant(torch.from_numpy(W), t, None if im is None else torch.from_numpy(im)).numpy()


def _grids_present():
    try:
        fq._iq2_tables("iq2_xxs")
        return True
    except Exception:
        return False


def _reference(t, W, im, g, stq):
    if t == "stq1_0":
        return stq.dequant(stq.quantize(W, im), *W.shape)
    return g.dequant(t, g.quantize(t, W, im), *W.shape)


@pytest.fixture(scope="module")
def ggml():
    g = ggml_ref.load()
    if g is None:
        pytest.skip("no runtime libggml-base (runtime/llama.cpp/build/bin or POLLARD_LLAMA_BIN_DIR)")
    return g


@pytest.fixture(scope="module")
def stq():
    r = ggml_ref.stq_reference()
    if r is None:
        pytest.skip("no STQ1_0 reference (Pollard's llama-stq tree + a C compiler, or POLLARD_STQ_GGML_LIB)")
    return r


# ------------------------------------------------------------------ live gate -----------------------
@pytest.mark.parametrize("t", [t for t in fq.SUPPORTED if t != "stq1_0"])
def test_bit_exact_vs_runtime_ggml(t, ggml):
    bad = []
    for name, W in inputs().items():
        for iname, im in imatrices().items():
            if im is None and t in fq.NEEDS_IMATRIX:
                continue
            ref = ggml.dequant(t, ggml.quantize(t, W, im), *W.shape)
            got = _port(t, W, im)
            if not np.array_equal(ref.view(np.uint32), got.view(np.uint32)):
                blocks = (ref.view(np.uint32) != got.view(np.uint32)).reshape(-1, 256).any(1).sum()
                bad.append(f"{name}/{iname}: {blocks} block(s)")
    assert not bad, f"{t} is not bit-exact: " + "; ".join(bad)


def test_bit_exact_random_stress(ggml):
    """Fresh random tensors (not the fixed goldens): a wider net for rare rounding paths."""
    rng = np.random.default_rng(int(os.environ.get("POLLARD_FQ_SEED", "0")))
    W = np.concatenate([rng.standard_normal((48, 512)) * 0.02, rng.standard_t(2, (48, 512)) * 0.01]).astype(np.float32)
    im = rng.lognormal(0, 1.5, 512).astype(np.float32)
    for t in fq.SUPPORTED:
        if t == "stq1_0":
            continue
        ref = ggml.dequant(t, ggml.quantize(t, W, im), *W.shape)
        assert np.array_equal(ref.view(np.uint32), _port(t, W, im).view(np.uint32)), t


def test_stq1_0_bit_exact_vs_patched_encoder(stq):
    for name, W in inputs().items():
        for iname, im in imatrices().items():
            ref = stq.dequant(stq.quantize(W, im), *W.shape)
            assert np.array_equal(ref.view(np.uint32), _port("stq1_0", W, im).view(np.uint32)), (name, iname)


def test_goldens_are_ggml_output(ggml, stq):
    """Keeps the fixture honest: every stored hash is what ggml itself produces today."""
    gold = json.loads(GOLDEN.read_text())
    ins = inputs()
    for key, h in gold["hashes"].items():
        t, name, iname = key.split("/")
        assert _sha(_reference(t, ins[name], imatrices()[iname], ggml, stq)) == h, key


# ------------------------------------------------------------------ CI gate (no ggml) ----------------
def test_port_reproduces_ggml_goldens():
    gold = json.loads(GOLDEN.read_text())
    ins, ims = inputs(), imatrices()
    have_grids = _grids_present()
    checked = 0
    for key, h in gold["hashes"].items():
        t, name, iname = key.split("/")
        if t in IQ2 and not have_grids:
            continue
        assert _sha(_port(t, ins[name], ims[iname])) == h, key
        checked += 1
    assert checked >= sum(1 for k in gold["hashes"] if k.split("/")[0] not in IQ2)


def test_golden_covers_every_supported_type():
    gold = json.loads(GOLDEN.read_text())
    assert {k.split("/")[0] for k in gold["hashes"]} == set(fq.SUPPORTED)


# ------------------------------------------------------------------ training contract ----------------
def test_ste_gradient_flows_and_forward_is_quantized():
    W = torch.from_numpy(inputs()["uniform"]).clone().requires_grad_(True)
    im = torch.from_numpy(imatrices()["uniform"])
    out = fq.fake_quant(W, "q3_K", im)
    assert torch.equal(out.detach(), fq.quant_dequant(W, "q3_k", im))
    (out * 3).sum().backward()
    assert torch.allclose(W.grad, torch.full_like(W, 3.0))


def test_fake_quant_linear():
    lin = torch.nn.Linear(512, 8, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(inputs()["uniform"]))
    im = torch.from_numpy(imatrices()["uniform"])
    fql = fq.FakeQuantLinear(lin, "stq1_0", im, refresh_every=2)
    x = torch.randn(3, 512)
    y = fql(x)
    want = torch.nn.functional.linear(x, fq.quant_dequant(lin.weight, "stq1_0", im), lin.bias)
    assert torch.allclose(y, want, atol=1e-6)
    y.sum().backward()
    assert fql.weight.grad is not None and torch.isfinite(fql.weight.grad).all()


def test_refusals():
    W = torch.from_numpy(inputs()["uniform"])
    with pytest.raises(ValueError, match="imatrix"):
        fq.quant_dequant(W, "q2_K")
    with pytest.raises(ValueError, match="imatrix"):
        fq.quant_dequant(W, "iq2_xxs")
    with pytest.raises(ValueError, match="no port"):
        fq.quant_dequant(W, "iq1_s")
    with pytest.raises(ValueError, match="multiple of 256"):
        fq.quant_dequant(torch.zeros(2, 100), "tq2_0")


def _write_golden():
    g, stq = ggml_ref.load(), ggml_ref.stq_reference()
    assert g is not None and stq is not None, "needs the runtime libggml and the STQ1_0 reference"
    ins = inputs()
    hashes = {}
    for t, iname, im in combos():
        for name, W in ins.items():
            ref = _reference(t, W, im, g, stq)
            assert np.array_equal(ref.view(np.uint32), _port(t, W, im).view(np.uint32)), (t, name, iname)
            hashes[f"{t}/{name}/{iname}"] = _sha(ref)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"about": "sha256 of ggml dequant(quantize(W, imatrix)) float32 bytes on "
                                           "tests/test_fq_parity.py inputs(); reference = runtime libggml "
                                           "(STQ1_0: Pollard's patched encoder)", "hashes": hashes}, indent=1) + "\n")
    print(f"wrote {len(hashes)} golden hashes to {GOLDEN}")


if __name__ == "__main__" and "--write-golden" in sys.argv:
    _write_golden()
