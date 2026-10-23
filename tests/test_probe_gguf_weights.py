"""Exercise stored BF16 bytes and expert-specific importance using real GGUF files."""
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_probe as P


def write_tensor(path, name, weights, qtype, experts=None):
    """experts: (n_used, n_expert) -> the MoE metadata a real merged-expert GGUF carries."""
    from gguf.quants import quantize
    writer = gguf.GGUFWriter(path, "qwen3moe")
    if experts:
        writer.add_expert_used_count(experts[0])
        writer.add_expert_count(experts[1])
    writer.add_tensor(name, quantize(weights, qtype), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return gguf.GGUFReader(path).tensors[0]


def write_imatrix(path, name, importance):
    name_bytes = name.encode()
    values = np.asarray(importance, dtype=np.float32).ravel() * 2
    path.write_bytes(struct.pack("<ii", 1, len(name_bytes)) + name_bytes +
                     struct.pack("<ii", 2, len(values)) + values.tobytes())


def write_pair(path, second_name, second_in_source=True):
    weights = np.array([[1, .5, .25, -1], [.5, -.25, 1, 2]], dtype=np.float32)
    writer = gguf.GGUFWriter(path, "qwen3moe")
    writer.add_tensor("blk.0.attn_q.weight", weights)
    if second_in_source:
        writer.add_tensor(second_name, weights.copy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    records = bytearray(struct.pack("<i", 2))
    for name in ("blk.0.attn_q.weight", second_name):
        encoded = name.encode()
        records.extend(struct.pack("<i", len(encoded)) + encoded + struct.pack("<ii", 1, 4))
        records.extend(np.ones(4, dtype=np.float32).tobytes())
    matrix = path.with_suffix(".dat")
    matrix.write_bytes(records)
    return matrix


def test_known_router_is_accounted_for_without_scoring(tmp_path, monkeypatch, capsys):
    name = "blk.0.ffn_gate_inp.weight"
    model = tmp_path / "model.gguf"
    matrix = write_pair(model, name)
    decode = P._dequant

    def only_scored_weights(tensor):
        assert tensor.name != name, "router must not be treated as an expert FFN matrix"
        return decode(tensor)

    monkeypatch.setattr(P, "_dequant", only_scored_weights)
    costs, noise, layers = P._imatrix_sensitivity(str(model), str(matrix), ["attn"], 2, [("iq2_s", 2)])
    assert layers == 1 and costs["attn"]["0"] == noise["iq2_s"]
    assert costs["excluded_tensors"][name] == {
        "source_type": "F32", "reason": "MoE router; retain source precision"}
    assert "Quantizing routers separately requires a separate measurement" in capsys.readouterr().out


@pytest.mark.parametrize("name,in_source", [
    ("blk.0.unknown_mixer.weight", True),
    ("blk.0.ffn_gate_inp_s.weight", True),
    ("blk.0.ffn_gate_inp.weight", False),
])
def test_router_exception_does_not_hide_unknown_or_absent_tensors(tmp_path, name, in_source):
    model = tmp_path / "model.gguf"
    matrix = write_pair(model, name, in_source)
    with pytest.raises(SystemExit, match="unscored"):
        P._imatrix_sensitivity(str(model), str(matrix), ["attn"], 2, [])


@pytest.mark.parametrize("qtype", [gguf.GGMLQuantizationType.BF16,
                                  gguf.GGMLQuantizationType.F16,
                                  gguf.GGMLQuantizationType.F32])
def test_float_gguf_decoding_preserves_values_and_shape(tmp_path, qtype):
    weights = np.array([[0, .5, -1, 2], [4, -8, .125, -.25]], dtype=np.float32)
    tensor = write_tensor(tmp_path / "weights.gguf", "blk.0.attn_q.weight", weights, qtype)
    decoded = P._dequant(tensor)
    assert decoded.shape == weights.shape
    np.testing.assert_array_equal(decoded, weights)


@pytest.mark.parametrize("qtype", [gguf.GGMLQuantizationType.BF16,
                                  gguf.GGMLQuantizationType.F16,
                                  gguf.GGMLQuantizationType.F32])
def test_merged_expert_probe_matches_separate_expert_costs(tmp_path, qtype):
    name = "blk.0.ffn_up_exps.weight"
    weights = np.array([[[1, .5, .25, -1], [.5, -.25, 1, 2]],
                        [[.5, 1, -.5, .25], [2, 1, -1, .5]]], dtype=np.float32)
    importance = np.array([[1, 2, 3, 4], [8, 7, 6, 5]], dtype=np.float32)
    model, matrix = tmp_path / "model.gguf", tmp_path / "imatrix.dat"
    write_tensor(model, name, weights, qtype, experts=(1, 2))
    write_imatrix(matrix, name, importance)
    costs, noise, layers = P._imatrix_sensitivity(str(model), str(matrix), ["ffn"], 2, [("iq2_s", 2)])
    # each expert's conditional cost, weighted by how often it is routed to: 1 of 2 experts per token
    expected = 0.5 * sum(float(((torch.tensor(expert) - P._rtn(torch.tensor(expert), 2)) ** 2 *
                                torch.tensor(h)).sum()) for expert, h in zip(weights, importance))
    assert costs["ffn"]["0"] == pytest.approx(expected)
    assert noise["iq2_s"] == pytest.approx(expected)
    assert layers == 1 and math.isfinite(expected) and expected > 0


def test_dense_probe_uses_decoded_bf16_weights(tmp_path):
    name = "blk.0.attn_q.weight"
    weights = np.array([[1, .5, .25, -1], [.5, -.25, 1, 2]], dtype=np.float32)
    importance = np.array([1, 2, 3, 4], dtype=np.float32)
    model, matrix = tmp_path / "model.gguf", tmp_path / "imatrix.dat"
    write_tensor(model, name, weights, gguf.GGMLQuantizationType.BF16)
    write_imatrix(matrix, name, importance)
    costs, _, _ = P._imatrix_sensitivity(str(model), str(matrix), ["attn"], 2, [])
    expected = float(((torch.tensor(weights) - P._rtn(torch.tensor(weights), 2)) ** 2 *
                      torch.tensor(importance)).sum())
    assert costs["attn"]["0"] == pytest.approx(expected)


@pytest.mark.parametrize("importance", [[1, 2], [float("nan")] * 8, [-1] * 8])
def test_bad_expert_importance_refuses_partial_profile(tmp_path, importance):
    name = "blk.0.ffn_up_exps.weight"
    model, matrix = tmp_path / "model.gguf", tmp_path / "imatrix.dat"
    write_tensor(model, name, np.ones((2, 2, 4), dtype=np.float32), gguf.GGMLQuantizationType.F32, experts=(1, 2))
    write_imatrix(matrix, name, importance)
    with pytest.raises(SystemExit, match="partial"):
        P._imatrix_sensitivity(str(model), str(matrix), ["ffn"], 2, [])


@pytest.mark.parametrize("shape,importance,merged", [
    ((2, 2, 4), [1] * 4, True), ((2, 2, 4), [1] * 8, False),
    ((1, 2, 2, 4), [1] * 4, True), ((2, 0), [], False),
    ((2, 4), [[1] * 4], False),
])
def test_rejects_ambiguous_or_unsupported_importance_layout(shape, importance, merged):
    with pytest.raises(ValueError):
        P._imatrix_weights(torch.ones(shape), importance, merged)


def test_quantized_expert_tensor_retains_logical_dimensions(tmp_path):
    from gguf.quants import dequantize, quantize
    qtype = gguf.GGMLQuantizationType.Q8_0
    weights = np.linspace(-2, 2, 128, dtype=np.float32).reshape(2, 2, 32)
    tensor = write_tensor(tmp_path / "model.gguf", "blk.0.ffn_up_exps.weight", weights, qtype)
    decoded = P._dequant(tensor)
    assert decoded.shape == weights.shape
    np.testing.assert_array_equal(decoded, dequantize(quantize(weights, qtype), qtype))


def test_nonfinite_weights_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        P._imatrix_weights(torch.tensor([[float("inf")]]), [1])


def test_cost_overflow_does_not_emit_infinite_profile(tmp_path):
    name = "blk.0.attn_q.weight"
    model, matrix = tmp_path / "model.gguf", tmp_path / "imatrix.dat"
    weights = np.array([[1e20, 0.4e20, -1e20, 0.1e20]], dtype=np.float32)
    write_tensor(model, name, weights, gguf.GGMLQuantizationType.F32)
    write_imatrix(matrix, name, [1] * 4)
    with pytest.raises(SystemExit, match="non-finite probe cost"):
        P._imatrix_sensitivity(str(model), str(matrix), ["attn"], 2, [])


def write_model(path, tensors, arch="lfm2moe"):
    writer = gguf.GGUFWriter(path, arch)
    for name, w in tensors.items():
        writer.add_tensor(name, np.asarray(w, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_imatrix_many(path, names, n=4):
    records = bytearray(struct.pack("<i", len(names)))
    for name in names:
        encoded = name.encode()
        records.extend(struct.pack("<i", len(encoded)) + encoded + struct.pack("<ii", 1, n))
        records.extend(np.ones(n, dtype=np.float32).tobytes())
    path.write_bytes(records)


W = [[1, .5, .25, -1], [.5, -.25, 1, 2]]


def test_lfm2_short_convolution_is_a_sequence_mixer(tmp_path):
    # LFM2.5-24B-A2B, 2026-10-16: the profile refused with "unscored: shortconv.in_proj.weight,
    # shortconv.out_proj.weight". Its conv kernel (shortconv.conv.weight) is not a matmul: never in the
    # imatrix, never quantized -- it must neither block the profile nor be pinned.
    model, matrix = tmp_path / "lfm.gguf", tmp_path / "lfm.dat"
    write_model(model, {"blk.0.shortconv.in_proj.weight": W, "blk.0.shortconv.out_proj.weight": W,
                        "blk.0.shortconv.conv.weight": [[1, 2, 3]], "blk.0.ffn_up.weight": W})
    write_imatrix_many(matrix, ["blk.0.shortconv.in_proj.weight", "blk.0.shortconv.out_proj.weight",
                                "blk.0.ffn_up.weight"])
    costs, _, layers = P._imatrix_sensitivity(str(model), str(matrix), ["ffn", "attn"], 2, [])
    one = float(((torch.tensor(W) - P._rtn(torch.tensor(W), 2)) ** 2).sum())
    assert costs["attn"]["0"] == pytest.approx(2 * one) and costs["ffn"]["0"] == pytest.approx(one)
    assert layers == 1


def test_nemotron_latent_moe_projections_are_channel_mixers(tmp_path):
    model, matrix = tmp_path / "nh.gguf", tmp_path / "nh.dat"
    write_model(model, {"blk.0.ffn_latent_down.weight": W, "blk.0.ffn_latent_up.weight": W,
                        "blk.0.ssm_in.weight": W, "blk.0.ssm_out.weight": W}, arch="nemotron_h_moe")
    write_imatrix_many(matrix, ["blk.0.ffn_latent_down.weight", "blk.0.ffn_latent_up.weight",
                                "blk.0.ssm_in.weight", "blk.0.ssm_out.weight"])
    costs, _, _ = P._imatrix_sensitivity(str(model), str(matrix), ["ffn", "attn"], 2, [])
    assert costs["ffn"]["0"] == pytest.approx(costs["attn"]["0"]) and costs["ffn"]["0"] > 0


def test_per_tensor_output_scale_is_folded_into_the_weight(tmp_path):
    # NVIDIA's QAT Bolt GGUF: ssm_in / ssm_out are NVFP4 with a separate F32 `<name>.scale`; the raw FP4 decode
    # is ~1e4x the real weight, so an unscaled cost would swamp every other tensor in the profile.
    s = 0.01
    big = (np.asarray(W) / s).tolist()
    model, ref, matrix = tmp_path / "scaled.gguf", tmp_path / "plain.gguf", tmp_path / "m.dat"
    write_model(model, {"blk.0.ssm_in.weight": big, "blk.0.ssm_in.scale": [s]}, arch="nemotron_h_moe")
    write_model(ref, {"blk.0.ssm_in.weight": W}, arch="nemotron_h_moe")
    write_imatrix_many(matrix, ["blk.0.ssm_in.weight"])
    scaled, _, _ = P._imatrix_sensitivity(str(model), str(matrix), ["attn"], 2, [])
    plain, _, _ = P._imatrix_sensitivity(str(ref), str(matrix), ["attn"], 2, [])
    assert scaled["attn"]["0"] == pytest.approx(plain["attn"]["0"], rel=1e-5)


def test_per_row_scale_and_a_misfit_scale(tmp_path):
    w = torch.ones(2, 4)
    assert torch.equal(P._apply_scale(w, [2.0, 3.0]), torch.tensor([[2.0] * 4, [3.0] * 4]))
    with pytest.raises(ValueError):
        P._apply_scale(w, [1.0, 2.0, 3.0])


def test_from_imatrix_cli_creates_the_output_folder(tmp_path, monkeypatch):
    # LFM2.5 release, 2026-10-16: the profile scored all 220 tensors, then died writing into a calibration/ folder a
    # fresh workspace did not have yet (the plan named its corpus, so pollard-calib never made the folder).
    model, matrix = tmp_path / "m.gguf", tmp_path / "m.dat"
    write_model(model, {"blk.0.ffn_up.weight": W})
    write_imatrix_many(matrix, ["blk.0.ffn_up.weight"])
    out = tmp_path / "models" / "M" / "calibration" / "M.sensitivity.json"
    monkeypatch.setattr(sys, "argv", ["pollard-probe", "--from-imatrix", str(matrix), "--gguf", str(model),
                                      "--out", str(out), "--groups", "ffn"])
    P.main()
    import json
    assert json.load(open(out))["method"] == "imatrix-hessian"
