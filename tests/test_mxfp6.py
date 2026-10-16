"""pollard-mx MXFP6 (E3M2 / E2M3) rungs and measured MX mixes (tools/pollard_mxfp6.py). numpy only: the encoder,
the safetensors writer and the read-back are pure numpy, so this runs in every CI shape.

The element values are checked against the OCP MX v1.0 field definitions, compressed-tensors' own MXFP6 test vector
and Graft's generated lookup tables; the blocks against GRAFT's decoder (tests/fixtures/mxfp6_graft_fixture.py,
generated from Graft's graft_qld.dequantize on Pollard's encode)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
import pollard_mxfp6 as M  # noqa: E402
from mxfp6_graft_fixture import FIXTURE, GRAFT_LUTS  # noqa: E402


def _f32(h, shape=None):
    a = np.frombuffer(bytes.fromhex(h), np.float32)
    return a.reshape(shape) if shape else a


# ------------------------------------------------------------------------------------------------ elements

def test_fp6_element_values_are_the_ocp_spec():
    e3 = M.elem_table("e3m2")
    e2 = M.elem_table("e2m3")
    assert len(e3) == len(e2) == 64 and not np.isnan(e3).any() and not np.isnan(e2).any()   # no Inf/NaN codes
    # E3M2: bias 3, max 1.75 * 2^4 = 28, min normal 2^-2, min subnormal 0.25 * 2^-2
    assert e3.max() == 28.0 and e3[0b000100] == 0.25 and e3[0b000001] == 0.0625 and e3[0b011111] == 28.0
    # E2M3: bias 1, max 1.875 * 2^2 = 7.5, min normal 1, min subnormal 0.125
    assert e2.max() == 7.5 and e2[0b001000] == 1.0 and e2[0b000001] == 0.125 and e2[0b011111] == 7.5
    assert e3[0b100000] == 0.0 and np.signbit(e3[0b100000]) and e2[0b111111] == -7.5      # sign is bit 5
    assert list(M.elem_table("e2m1")[:8]) == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    assert np.isnan(M.elem_table("e4m3")[0x7F]) and M.elem_table("e4m3")[0x7E] == 448


def test_fp6_tables_equal_grafts_generated_luts():
    for e in ("e3m2", "e2m3"):
        lut = np.asarray(GRAFT_LUTS[e], np.uint32).view(np.float32)
        assert np.array_equal(M.elem_table(e).astype(np.float32), lut), e


def test_compressed_tensors_mxfp6_test_vector():
    # compressed-tensors' (and Graft's) vector: bytes 48 F8 01 = E2M3 codes [1.0, -0.125, 7.5, 0.0]
    p = np.zeros((1, 24), np.uint8)
    p[0, :3] = [0x48, 0xF8, 0x01]
    v = M.mx_decode(M.unpack6(p), np.array([[127]], np.uint8), "e2m3")
    assert list(v[0, :4]) == [1.0, -0.125, 7.5, 0.0]
    codes = M.quantize_elem(np.array([1.0, -0.125, 7.5, 0.0] + [0.0] * 28), "e2m3")[None]
    assert list(M.pack6(codes)[0, :3]) == [0x48, 0xF8, 0x01]


@pytest.mark.parametrize("elem", ["e3m2", "e2m3", "e2m1", "e4m3"])
def test_quantize_is_exact_on_grid_ties_to_even_and_saturates(elem):
    t = M.elem_table(elem)
    finite = np.isfinite(t)
    c = M.quantize_elem(t[finite], elem)
    assert np.array_equal(M.decode_elem(c, elem), t[finite])          # every value is its own code
    pos, mids = M._positive(elem)
    got = M.decode_elem(M.quantize_elem(mids, elem), elem)
    lo, hi = M.quantize_elem(pos[:-1], elem), M.quantize_elem(pos[1:], elem)
    want = np.where(lo % 2 == 0, pos[:-1], pos[1:])                    # the even (mantissa 0) neighbour
    assert np.array_equal(got, want)
    assert np.array_equal(hi, np.arange(1, len(pos)))
    assert M.decode_elem(M.quantize_elem(np.array([1e9, -1e9]), elem), elem).tolist() == [pos[-1], -pos[-1]]


def test_pack6_is_the_4_in_3_lsb_first_layout_and_round_trips_all_codes():
    codes = np.arange(64, dtype=np.uint8).reshape(2, 32)
    p = M.pack6(codes)
    assert p.shape == (2, 24)
    w = int(p[0, 0]) | int(p[0, 1]) << 8 | int(p[0, 2]) << 16
    assert [(w >> s) & 63 for s in (0, 6, 12, 18)] == [0, 1, 2, 3]
    assert np.array_equal(M.unpack6(p), codes)
    assert np.array_equal(M.unpack4(M.pack4(codes & 15)), codes & 15)


def test_ocp_shared_exponent_and_e8m0():
    w = np.zeros((1, 64), np.float32)
    w[0, 3] = 0.9                          # floor(log2 0.9) = -1
    w[0, 40] = 20.0                        # floor(log2 20) = 4
    for elem, emax in (("e2m3", 2), ("e3m2", 4)):
        _, s = M.mx_encode(w, elem, "ocp")
        assert list(s[0]) == [127 - 1 - emax, 127 + 4 - emax]
    _, s = M.mx_encode(np.zeros((1, 32)), "e2m3", "ocp")
    assert s[0, 0] == 0                    # an all-zero block: smallest scale, zero codes
    assert M.e8m0_value(np.array([127, 0, 254])).tolist() == [1.0, 2.0 ** -127, 2.0 ** 127]


@pytest.mark.parametrize("elem", ["e3m2", "e2m3"])
def test_scale_search_never_loses_to_the_spec_scale(elem):
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((16, 256)) * 0.02).astype(np.float32)
    w[:, rng.integers(0, 256, 6)] *= 9
    imp = rng.gamma(0.5, 1.0, 256)
    errs = {}
    for rule in ("ocp", "search"):
        c, s = M.mx_encode(w, elem, rule, imp)
        errs[rule] = float((((w - M.mx_decode(c, s, elem)) ** 2) * imp).sum())
        assert c.max() < 64 and s.dtype == np.uint8
    assert errs["search"] <= errs["ocp"]


# ------------------------------------------------------------------------------------------------ Graft

@pytest.mark.parametrize("case", sorted(FIXTURE["cases"]))
def test_pollard_bytes_and_decode_match_grafts_decoder(case):
    elem, rule = case.split("/")
    fx = FIXTURE["cases"][case]
    w = _f32(FIXTURE["weight_f32"], FIXTURE["shape"])
    codes, scales = M.mx_encode(w, elem, rule)
    packed = M.pack6(codes)
    assert packed.tobytes().hex() == fx["packed"] and scales.tobytes().hex() == fx["scales"]
    nat = M.graft_blocks(packed, scales, elem)
    assert nat.tobytes().hex() == fx["graft_blocks"]                 # {u8 e8m0; u8 qs[24]} per 32 weights
    assert nat.shape[1] == FIXTURE["shape"][1] // 32 * 25
    graft = _f32(fx["graft_decode_f32"], FIXTURE["shape"])
    mine = M.mx_decode(M.unpack6(packed), scales, elem).astype(np.float32)
    assert mine.tobytes() == graft.tobytes(), "bit-exact with Graft's MXFP6 decode"


# ------------------------------------------------------------------------------------------------ atoms

@pytest.mark.parametrize("atom", ["mxfp4", "nvfp4", "mxfp6_e3m2", "mxfp6_e2m3", "mxfp8"])
def test_every_atom_round_trips_through_its_stored_tensors(atom):
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((8, 128)) * 0.05).astype(np.float32)
    parts, dec = M.encode_atom(w, atom)
    stored = {k: a for k, (dt, a) in parts.items()}
    assert np.array_equal(M.decode_atom(stored, atom), dec)
    rel = float(((w - dec) ** 2).sum() / (w ** 2).sum())
    assert rel < {"mxfp4": 0.03, "nvfp4": 0.03, "mxfp6_e3m2": 0.006, "mxfp6_e2m3": 0.003, "mxfp8": 0.001}[atom], rel
    shapes = {k: a.shape for k, a in stored.items()}
    if atom.startswith("mxfp6"):
        assert shapes == {"weight": (8, 96), "weight_scale": (8, 4)}
    if atom == "mxfp8":
        assert parts["weight"][0] == "F8_E4M3" and shapes["weight"] == (8, 128)


# ------------------------------------------------------------------------------------------------ allocation

def _groups(rel4=(0.02, 0.02, 0.02, 0.02), n=1000):
    g = {}
    for i, r in enumerate(rel4):
        g[f"model.layers.{i}.mlp.down_proj"] = {"n": n, "layer": i, "members": [f"model.layers.{i}.mlp.down_proj"],
                                                "w2": 1.0, "err": {"mxfp4": r, "mxfp6_e2m3": r / 16,
                                                                   "mxfp6_e3m2": r / 8, "mxfp8": r / 256}}
    return g


def test_mix_respects_the_budget_and_follows_sensitivity():
    atoms = ["mxfp4", "mxfp6_e2m3", "mxfp6_e3m2", "mxfp8"]
    g = _groups()
    sens = {"0": 1.0, "1": 1.0, "2": 9.0, "3": 1.0}
    for target in (4.25, 4.75, 5.25, 6.25, 7.25, 8.25, 9.0):
        a = M.allocate_mix(g, target, atoms, sens)
        assert M.avg_bpw(g, a) <= target + 1e-9
        assert "mxfp6_e3m2" not in a.values(), "auto keeps the better FP6 element only"
    a = M.allocate_mix(g, 4.75, atoms, sens)
    assert a["model.layers.2.mlp.down_proj"] != "mxfp4" and list(a.values()).count("mxfp4") == 3
    assert set(M.allocate_mix(g, 9.0, atoms, sens).values()) == {"mxfp8"}
    with pytest.raises(SystemExit):
        M.allocate_mix(g, 4.0, atoms, sens)


def test_mix_spends_on_the_noisier_group_without_a_profile_and_honours_focus():
    atoms = ["mxfp4", "mxfp6_e2m3", "mxfp8"]
    g = _groups(rel4=(0.01, 0.08, 0.01, 0.01))
    a = M.allocate_mix(g, 4.75, atoms)
    assert a["model.layers.1.mlp.down_proj"] != "mxfp4"
    a = M.allocate_mix(g, 6.25, atoms, focus={3})
    assert a["model.layers.3.mlp.down_proj"] == "mxfp8"


def test_fp6_auto_picks_the_better_element_per_group():
    g = _groups()
    k = "model.layers.0.mlp.down_proj"
    g[k]["err"]["mxfp6_e3m2"], g[k]["err"]["mxfp6_e2m3"] = 1e-6, 1e-3   # an outlier-heavy group: E3M2 wins
    a = M.allocate_mix(g, 6.25, ["mxfp4", "mxfp6_e2m3", "mxfp6_e3m2", "mxfp8"])
    assert a[k] == "mxfp6_e3m2" and a["model.layers.1.mlp.down_proj"] == "mxfp6_e2m3"


def test_fused_groups():
    assert M.fused_key("model.layers.3.self_attn.k_proj") == "model.layers.3.self_attn.qkv_proj"
    assert M.fused_key("model.layers.3.mlp.up_proj") == "model.layers.3.mlp.gate_up_proj"
    assert M.fused_key("model.layers.3.mlp.experts.17.down_proj") == "model.layers.3.mlp.experts"
    assert M.fused_key("model.layers.3.self_attn.o_proj") == "model.layers.3.self_attn.o_proj"


# ------------------------------------------------------------------------------------------------ checkpoints

def _bf16(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def tiny_ckpt(d, layers=4, H=64, I=128, V=96, router=True):
    rng = np.random.default_rng(7)
    t = {"model.embed_tokens.weight": ("BF16", _bf16(rng.standard_normal((V, H)) * 0.02))}
    for i in range(layers):
        p = f"model.layers.{i}."
        for n, shp in (("self_attn.q_proj", (H, H)), ("self_attn.k_proj", (32, H)), ("self_attn.v_proj", (32, H)),
                       ("self_attn.o_proj", (H, H)), ("mlp.gate_proj", (I, H)), ("mlp.up_proj", (I, H)),
                       ("mlp.down_proj", (H, I))):
            w = rng.standard_normal(shp) * 0.02
            if i == 2:
                w[:, :3] *= 40                                       # an outlier-heavy layer
            t[p + n + ".weight"] = ("BF16", _bf16(w))
        t[p + "input_layernorm.weight"] = ("BF16", _bf16(np.ones(H)))
        t[p + "self_attn.q_proj.bias"] = ("BF16", _bf16(np.zeros(H)))
        if router and i == 1:
            t[p + "mlp.gate.weight"] = ("BF16", _bf16(rng.standard_normal((8, H))))   # a router: never quantized
        t[p + "self_attn.odd_proj.weight"] = ("BF16", _bf16(rng.standard_normal((H, 48))))  # K % 32 != 0
    t["lm_head.weight"] = ("BF16", _bf16(rng.standard_normal((V, H)) * 0.02))
    os.makedirs(d, exist_ok=True)
    M.st_write(os.path.join(d, "model.safetensors"), t, {"format": "pt"})
    json.dump({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "num_hidden_layers": layers,
               "hidden_size": H}, open(os.path.join(d, "config.json"), "w"))
    open(os.path.join(d, "tokenizer.json"), "w").write("{}")
    return t


def _load(path):
    hdr, _ = M.st_header(path)
    return hdr, {n: M.st_read(path, n)[1] for n in hdr if n != "__metadata__"}


def test_uniform_e2m3_rung_is_compressed_tensors_and_graft_shaped(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "e2m3")
    t = tiny_ckpt(src)
    meta = M.build(src, out, fmt="mxfp6-e2m3", log=lambda *a: None)
    assert meta["layout"] == "compressed-tensors" and meta["bpw"] == 6.25 and meta["atoms"] == {"mxfp6_e2m3": 1.0}
    qc = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    assert qc["quant_method"] == "compressed-tensors" and qc["format"] == "mxfp6-pack-quantized"
    w = qc["config_groups"]["group_0"]["weights"]
    assert (w["num_bits"], w["type"], w["group_size"], w["scale_dtype"]) == (6, "float", 32, "torch.uint8")
    assert os.path.exists(os.path.join(out, "tokenizer.json"))
    hdr, ten = _load(os.path.join(out, "model.safetensors"))
    q = "model.layers.0.self_attn.q_proj"
    assert hdr[q + ".weight_packed"]["dtype"] == "U8" and hdr[q + ".weight_packed"]["shape"] == [64, 48]
    assert hdr[q + ".weight_scale"]["dtype"] == "U8" and hdr[q + ".weight_scale"]["shape"] == [64, 2]
    assert q + ".weight" not in hdr
    for keep in ("lm_head.weight", "model.embed_tokens.weight", "model.layers.1.mlp.gate.weight",
                 "model.layers.0.self_attn.odd_proj.weight", "model.layers.0.input_layernorm.weight",
                 "model.layers.0.self_attn.q_proj.bias"):
        assert np.array_equal(ten[keep], t[keep][1]), keep          # untouched, same bytes
    assert "model.layers.1.mlp.gate" not in meta["modules"]
    assert "model.layers.0.self_attn.odd_proj" in meta["skipped"]
    assert any(r[0] == "Graft" and r[1] == "yes" for r in meta["runtimes"])
    dec = M.dequantized_weights(out)
    src_w = M.to_f32("BF16", t[q + ".weight"][1])
    c, s = M.mx_encode(src_w, "e2m3")
    assert np.array_equal(dec[q + ".weight"], M.mx_decode(c, s, "e2m3").astype(np.float32))


def test_uniform_e3m2_rung_is_quark_and_ct_refuses_it(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "e3m2")
    tiny_ckpt(src)
    meta = M.build(src, out, fmt="mxfp6-e3m2", log=lambda *a: None)
    assert meta["layout"] == "quark"
    qc = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    assert qc["quant_method"] == "quark" and qc["global_quant_config"]["weight"]["dtype"] == "fp6_e3m2"
    assert qc["global_quant_config"]["weight"]["scale_format"] == "e8m0" and qc["layer_quant_config"] == {}
    assert "lm_head" in qc["exclude"] and any("odd_proj" in x for x in qc["exclude"])
    hdr, _ = _load(os.path.join(out, "model.safetensors"))
    assert hdr["model.layers.0.mlp.down_proj.weight"]["shape"] == [64, 96]
    assert any(r[0] == "vLLM" and r[1] == "yes" for r in meta["runtimes"])
    with pytest.raises(SystemExit):
        M.build(src, str(tmp_path / "x"), fmt="mxfp6-e3m2", layout="compressed-tensors", log=lambda *a: None)


def test_measured_mix_rung(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "mix")
    tiny_ckpt(src)
    meta = M.build(src, out, mix=5.5, sens={"0": 1, "1": 1, "2": 1, "3": 1}, log=lambda *a: None)
    assert meta["layout"] == "quark" and meta["bpw"] <= 5.5 and len(meta["atoms"]) > 1
    mods = meta["modules"]
    for i in range(4):                                     # fused groups share one atom
        p = f"model.layers.{i}."
        assert mods[p + "self_attn.q_proj"] == mods[p + "self_attn.k_proj"] == mods[p + "self_attn.v_proj"]
        assert mods[p + "mlp.gate_proj"] == mods[p + "mlp.up_proj"]
    qc = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    for pat, spec in qc["layer_quant_config"].items():
        assert pat.startswith("*.layers.") and spec["weight"]["dtype"] in ("fp4", "fp6_e2m3", "fp6_e3m2", "fp8_e4m3")
    dec = M.dequantized_weights(out)
    assert set(dec) == {m + ".weight" for m in mods}
    hdr, _ = _load(os.path.join(out, "model.safetensors"))
    for m, a in mods.items():
        K = 128 if m.endswith("down_proj") else 64
        want = {"mxfp4": K // 2, "mxfp6_e2m3": K * 3 // 4, "mxfp6_e3m2": K * 3 // 4, "mxfp8": K}[a]
        assert hdr[m + ".weight"]["shape"][1] == want, (m, a)
    # the outlier-heavy layer 2 is where the budget goes first
    l2 = [M.ATOMS[mods[f"model.layers.2.{n}"]]["bpw"] for n in ("self_attn.q_proj", "mlp.down_proj")]
    l0 = [M.ATOMS[mods[f"model.layers.0.{n}"]]["bpw"] for n in ("self_attn.q_proj", "mlp.down_proj")]
    assert sum(l2) >= sum(l0)


def test_ct_mix_uses_nvfp4_and_mixed_precision(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "ctmix")
    tiny_ckpt(src)
    meta = M.build(src, out, mix=5.0, layout="compressed-tensors", log=lambda *a: None)
    qc = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    assert qc["format"] == "mixed-precision" and "nvfp4" in meta["atoms"]
    fmts = {g["format"] for g in qc["config_groups"].values()}
    assert "nvfp4-pack-quantized" in fmts and fmts <= {"nvfp4-pack-quantized", "mxfp6-pack-quantized", "mxfp8-quantized"}
    hdr, ten = _load(os.path.join(out, "model.safetensors"))
    nv = [m for m, a in meta["modules"].items() if a == "nvfp4"]
    assert hdr[nv[0] + ".weight_scale"]["dtype"] == "F8_E4M3" and hdr[nv[0] + ".weight_global_scale"]["dtype"] == "F32"
    by_layer = {}
    for m in nv:                                            # one global scale per fused NVFP4 layer
        by_layer.setdefault(M.fused_key(m), set()).add(float(ten[m + ".weight_global_scale"][0]))
    assert all(len(v) == 1 for v in by_layer.values())
    assert any(r[0] == "Graft" and r[1] == "no" for r in meta["runtimes"])
    dec = M.dequantized_weights(out)
    assert all(np.isfinite(v).all() for v in dec.values())
    with pytest.raises(SystemExit):
        M.build(src, str(tmp_path / "y"), mix=5.0, layout="quark", fp4="nvfp4", log=lambda *a: None)


def test_vllm_load_problem_names_the_missing_compressed_tensors(tmp_path, monkeypatch):
    src, out = str(tmp_path / "src"), str(tmp_path / "e2m3")
    tiny_ckpt(src)
    M.build(src, out, fmt="mxfp6-e2m3", log=lambda *a: None)
    monkeypatch.setitem(sys.modules, "compressed_tensors", None)
    monkeypatch.setitem(sys.modules, "compressed_tensors.config", None)
    assert "mxfp6-pack-quantized" in M.vllm_load_problem(out)
    q = str(tmp_path / "q")
    M.build(src, q, fmt="mxfp6-e3m2", log=lambda *a: None)
    assert M.vllm_load_problem(q) is None and M.vllm_load_problem(src) is None


# ------------------------------------------------------------------------------------------------ wiring

def test_release_rungs_tags_and_commands():
    import pollard_release as R
    assert R.rung_tag("mx", "mxfp6_e2m3") == "MXFP6-E2M3" and R.rung_tag("mx", "mix5.5") == "MIX-5.5bpw"
    assert R.rung_tag("mx", "mix5") == "MIX-5bpw" and R.rung_tag("mx", "nvfp4") == "NVFP4"
    cfg = R.load_plan(None)
    c, lanes, steps = R.plan_steps(cfg, "/nonexistent/Qwen2.5-0.5B", {"mx"})
    by = {s.id: s for s in steps}
    mix = by["mx/MIX-4.5bpw/build"].cmd
    assert mix[mix.index("--mix") + 1] == "4.5" and "--calib" not in mix and "--sensitivity" in mix
    e3 = by["mx/MXFP6-E3M2/build"].cmd
    assert e3[e3.index("--format") + 1] == "mxfp6-e3m2"
    assert "mx/MXFP6-E3M2/verify" in by


def test_gate_thresholds_for_the_new_tags():
    import pollard_verify as V
    g = V.load_gates(None)
    assert V.thresholds_for(g, "Qwen-Pollard-MX-MXFP6-E3M2", "mx")["kld_max"] == 0.02
    assert V.thresholds_for(g, "MIX-5.5bpw", "mx")["kld_max"] == 0.04
    assert V.thresholds_for(g, "MIX-4.5bpw", "mx")["kld_max"] == 0.08


def test_card_mx_rung_table(tmp_path):
    import pollard_card as C
    src = str(tmp_path / "src")
    tiny_ckpt(src)
    a, b = str(tmp_path / "M-Pollard-MX-MXFP6-E2M3"), str(tmp_path / "M-Pollard-MX-MIX-5.5bpw")
    M.build(src, a, fmt="mxfp6-e2m3", log=lambda *x: None)
    M.build(src, b, mix=5.5, log=lambda *x: None)
    md = "\n".join(C.rungs_section([{"path": a, "name": os.path.basename(a), "bytes": 2, "bpw": 6.25},
                                    {"path": b, "name": os.path.basename(b), "bytes": 1, "bpw": 5.5},
                                    {"path": "/x/M-Pollard-MX-NVFP4", "name": "M-Pollard-MX-NVFP4", "tag": "NVFP4",
                                     "bytes": 3, "bpw": 4.5}], "mx", "O/M-Pollard-MX"))
    assert "| folder | format | layout |" in md and "| `mxfp6-e2m3/` | MXFP6-E2M3 | compressed-tensors |" in md
    assert "`mix-5.5bpw/` | mix " in md and "Graft" in md and "vllm serve ./mix-5.5bpw" in md
    assert "| `nvfp4/` | NVFP4 | compressed-tensors |" in md


def test_pollard_mx_cli_format_plan_only(tmp_path, monkeypatch, capsys):
    import pollard_mx as X
    src = str(tmp_path / "src")
    tiny_ckpt(src)
    monkeypatch.setattr(sys, "argv", ["pollard-mx", "--model", src, "--format", "mxfp6-e3m2", "--plan-only"])
    X.main()
    o = capsys.readouterr().out
    assert "mxfp6-e3m2 -> quark: 6.250 bpw" in o and not os.path.exists(str(tmp_path / "out"))


def test_verify_measures_a_rung_through_its_decoded_bytes(tmp_path, monkeypatch):
    """pollard-verify --lane mx on an MXFP6 rung: the source architecture with the rung's linears decoded exactly."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import pollard_verify as V
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    cfg = transformers.Qwen2Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                   tie_word_embeddings=False)
    torch.manual_seed(0)
    src = str(tmp_path / "src")
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(src, safe_serialization=True)
    out = str(tmp_path / "rung")
    M.build(src, out, fmt="mxfp6-e3m2", log=lambda *a: None)

    class Tok:
        def __call__(self, text):
            return {"input_ids": [ord(c) % 96 for c in text]}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: Tok()))
    ev = tmp_path / "eval.txt"
    ev.write_text("the quick brown fox jumps over the lazy dog " * 8)
    same = V.measure_hf(src, src, str(ev), device="cpu", length=32, chunks=2)
    got = V.measure_hf(out, src, str(ev), device="cpu", length=32, chunks=2)
    assert same["kld"] < 1e-6
    assert 0 < got["kld"] < 0.05, got


def test_hf_load_of_a_rung_without_any_quant_loader(tmp_path):
    """The tool gate's in-process backend (and anything else) can run a rung with no MXFP6 kernel / Quark / CT
    package: the architecture from config, unquantized tensors as stored, linears decoded from the rung's bytes."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                   tie_word_embeddings=True)
    torch.manual_seed(0)
    src = str(tmp_path / "src")
    ref = transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    ref.save_pretrained(src, safe_serialization=True)
    out = str(tmp_path / "rung")
    M.build(src, out, mix=5.0, log=lambda *a: None)
    m = M.load_hf_model(out)
    dec = M.dequantized_weights(out)
    sd = m.state_dict()
    for k, w in dec.items():
        assert torch.equal(sd[k].float(), torch.from_numpy(w).to(torch.bfloat16).float())
    assert torch.equal(sd["model.embed_tokens.weight"], ref.state_dict()["model.embed_tokens.weight"])
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.isfinite(m(x).logits).all()
    import pollard_toolcall as T
    assert T.preflight_inprocess("hf", out)["device"] in ("cpu", "mps", "cuda")   # no quark / CT loader asked for
