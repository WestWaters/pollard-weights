"""pollard-mx --fp4-scale 4over6: Four Over Six (MIT HAN Lab, arXiv 2512.02010, MIT licence) block scales for
NVFP4, picked per block by Pollard's imatrix-weighted error. The output must stay STANDARD NVFP4 (packed E2M1,
E4M3 block scales, FP32 tensor scale; compressed-tensors' w = fp4 * scale / global_scale) and lower the
weighted error. numpy-only except the checkpoint round trip (torch + safetensors)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_mx as X  # noqa: E402


def _w(n=64, k=256, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    w[:, rng.integers(0, k, 4)] *= 6
    imp = rng.gamma(0.5, 1.0, k).astype(np.float32)
    imp[rng.integers(0, k, 8)] *= 30
    return w, imp


def _e4m3_grid():
    vals = {0.0}
    for e in range(-6, 9):
        for m in range(8):
            v = (1 + m / 8) * 2.0 ** e
            if v <= 448:
                vals.add(v)
    for m in range(1, 8):
        vals.add(m / 8 * 2.0 ** -6)                       # subnormals
    return np.array(sorted(vals), np.float32)


def test_round_e4m3_lands_on_the_e4m3_grid_and_saturates():
    grid = _e4m3_grid()
    x = np.abs(np.random.default_rng(1).standard_normal(5000).astype(np.float32)) * 30
    r = X.round_e4m3(x)
    assert np.all(np.isin(r, grid))
    nearest = grid[np.abs(x[:, None] - grid[None, :]).argmin(1)]
    assert np.allclose(np.abs(r - x), np.abs(nearest - x))
    assert X.round_e4m3(np.float32(1e6)) == 448.0
    torch = pytest.importorskip("torch")
    assert np.array_equal(r, torch.from_numpy(x).to(torch.float8_e4m3fn).float().numpy())


def test_fp4_codes_round_to_nearest_e2m1():
    y = np.array([0.2, 0.26, 1.24, 2.6, 4.9, 5.1, 7.0, -3.4, -0.1], np.float32)
    v = X.fp4_values(X.fp4_codes(y))
    assert v.tolist() == [0.0, 0.5, 1.0, 3.0, 4.0, 6.0, 6.0, -3.0, 0.0]
    c = np.arange(16, dtype=np.uint8).reshape(2, 8)
    assert np.array_equal(X.unpack_fp4(X.pack_fp4(c)), c)
    assert X.pack_fp4(np.array([[1, 2]], np.uint8))[0, 0] == 0x21, "element 0 in the low nibble"


@pytest.mark.parametrize("rule", X.FP4_SCALE_RULES)
def test_output_is_standard_nvfp4(rule):
    w, imp = _w()
    packed, sc, g, four = X.nvfp4_encode(w, rule, imp)
    assert packed.dtype == np.uint8 and packed.shape == (64, 128)
    assert sc.shape == (64, 256 // 16) and np.all(np.isin(sc, _e4m3_grid())) and sc.max() <= 448
    assert isinstance(g, np.float32) and g > 0
    if rule == "6":
        assert not four.any()
        assert np.isclose(g, 448 * 6 / np.abs(w).max())
    else:
        assert np.isclose(g, 256 * 6 / np.abs(w).max()) and four.any() and not four.all()
    deq = X.nvfp4_decode(packed, sc, g)
    assert np.abs(deq - w).max() < np.abs(w).max() / 4


def test_four_over_six_lowers_weighted_error_and_never_loses_a_block():
    w, imp = _w()
    p6, s6, g6, _ = X.nvfp4_encode(w, "6", imp)
    p46, s46, g46, four = X.nvfp4_encode(w, "4over6", imp)
    e6 = X.nvfp4_weighted_error(w, p6, s6, g6, imp)
    e46 = X.nvfp4_weighted_error(w, p46, s46, g46, imp)
    assert e46 < 0.95 * e6, (e46, e6)
    # with the SAME tensor scale, 4over6 is never worse than 6 on any block (it picks the better)
    p6b, s6b, _, _ = X.nvfp4_encode(w, "6", imp, global_scale=g46)
    d6 = ((w - X.nvfp4_decode(p6b, s6b, g46)) ** 2 * imp).reshape(64, 16, 16).sum(-1)
    d46 = ((w - X.nvfp4_decode(p46, s46, g46)) ** 2 * imp).reshape(64, 16, 16).sum(-1)
    assert np.all(d46 <= d6 + 1e-12)


def test_weighting_changes_the_choice():
    w, imp = _w(seed=4)
    _, _, _, f_mse = X.nvfp4_encode(w, "4over6")
    _, _, _, f_imp = X.nvfp4_encode(w, "4over6", imp)
    assert (f_mse != f_imp).any(), "imatrix weighting should flip some blocks"


def test_rejects_bad_shapes_and_rules():
    with pytest.raises(ValueError):
        X.nvfp4_encode(np.ones((2, 20), np.float32))
    with pytest.raises(ValueError):
        X.nvfp4_encode(np.ones((2, 32), np.float32), rule="8")


def test_checkpoint_post_pass_keeps_layout_and_fused_scale_sharing(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    w, imp = _w()
    w2, _ = _w(seed=9)
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir(); out.mkdir()
    st.save_file({"model.layers.0.self_attn.q_proj.weight": torch.from_numpy(w).to(torch.bfloat16),
                  "model.layers.0.self_attn.k_proj.weight": torch.from_numpy(w2).to(torch.bfloat16)},
                 str(src / "model.safetensors"))
    # what llm-compressor writes for NVFP4: one shared global scale for the fused q/k/v group
    wb, w2b = (torch.from_numpy(x).to(torch.bfloat16).float().numpy() for x in (w, w2))
    g_shared = np.float32(448 * 6 / max(np.abs(wb).max(), np.abs(w2b).max()))
    sd = {}
    for name, ww in (("q_proj", wb), ("k_proj", w2b)):
        p, sc, g, _ = X.nvfp4_encode(ww, "6", global_scale=g_shared)
        base = f"model.layers.0.self_attn.{name}"
        sd[base + ".weight_packed"] = torch.from_numpy(p)
        sd[base + ".weight_scale"] = torch.from_numpy(sc).to(torch.float8_e4m3fn)
        sd[base + ".weight_global_scale"] = torch.tensor([g_shared], dtype=torch.float32)
    sd["model.norm.weight"] = torch.ones(256)
    st.save_file(sd, str(out / "model.safetensors"))
    res = X.apply_four_over_six(str(out), str(src), {"model.layers.0.self_attn.q_proj": imp}, log=lambda *_: None)
    assert res["tensors"] == 2 and res["weighted"] == 1 and res["err_after"] < res["err_before"]
    got = st.load_file(str(out / "model.safetensors"))
    for name in ("q_proj", "k_proj"):
        base = f"model.layers.0.self_attn.{name}"
        assert got[base + ".weight_packed"].dtype == torch.uint8 and tuple(got[base + ".weight_packed"].shape) == (64, 128)
        assert got[base + ".weight_scale"].dtype == torch.float8_e4m3fn
        assert got[base + ".weight_global_scale"].dtype == torch.float32
    gq = float(got["model.layers.0.self_attn.q_proj.weight_global_scale"][0])
    gk = float(got["model.layers.0.self_attn.k_proj.weight_global_scale"][0])
    assert gq == gk and np.isclose(gq, g_shared * 256 / 448), "fused q/k/v must keep ONE global scale"
    assert torch.equal(got["model.norm.weight"], torch.ones(256))
