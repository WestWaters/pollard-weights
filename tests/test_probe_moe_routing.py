"""pollard-probe --from-imatrix on an MoE: experts are costed by how often they are routed to.

What it protects: an imatrix holds E[x_j^2] CONDITIONAL on an expert being routed to. Summed over every expert as-is,
each one is costed as if every token passed through it -- n_expert / n_used too heavy against attention, which does
see every token. clef-omni (128 experts, top-8), 2026-10-22: the IQ4_XS rung put 23 layers of attention at iq2_xxs
while the experts kept iq4_xs-q6_K, because the profile read attention as ~1000-10000x cheaper than the experts.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_probe as P
import pollard_fit as F

E, ROWS, CH = 4, 2, 4                                   # experts, rows per expert, input channels
W_EXP = np.linspace(-1, 1, E * ROWS * CH, dtype=np.float32).reshape(E, ROWS, CH)
W_ATT = W_EXP[0].copy()                                 # one expert's matrix, as a dense attention tensor


def write_model(path, n_used=1, n_expert=E, meta=True, router=False):
    w = gguf.GGUFWriter(path, "qwen3moe")
    if router:
        w.add_tensor("blk.0.ffn_gate_inp.weight", np.ones((E, CH), np.float32))
    if meta:
        w.add_expert_used_count(n_used)
        w.add_expert_count(n_expert)
    w.add_tensor("blk.0.attn_q.weight", W_ATT)
    w.add_tensor("blk.0.ffn_up_exps.weight", W_EXP)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_legacy(path, ncall=3):
    """legacy .dat: every value stored as (sum/count)*ncall, i.e. the conditional mean times ncall."""
    rec = bytearray(struct.pack("<i", 2))
    for name, n in (("blk.0.attn_q.weight", CH), ("blk.0.ffn_up_exps.weight", E * CH)):
        b = name.encode()
        rec += struct.pack("<i", len(b)) + b + struct.pack("<ii", ncall, n) + (np.ones(n, np.float32) * ncall).tobytes()
    path.write_bytes(bytes(rec))


def write_gguf_imatrix(path, expert_counts, tokens, h=1.0):
    """llama.cpp's GGUF imatrix: raw sums + per-matrix counts (one per expert), router count = tokens."""
    w = gguf.GGUFWriter(path, "imatrix")
    c = np.asarray(expert_counts, np.float32)
    w.add_tensor("blk.0.ffn_up_exps.weight.in_sum2", (np.ones((E, CH), np.float32) * h * c[:, None]))
    w.add_tensor("blk.0.ffn_up_exps.weight.counts", c.reshape(E, 1))
    w.add_tensor("blk.0.attn_q.weight.in_sum2", np.ones((1, CH), np.float32) * h * tokens)
    w.add_tensor("blk.0.attn_q.weight.counts", np.asarray([[tokens]], np.float32))
    w.add_tensor("blk.0.ffn_gate_inp.weight.in_sum2", np.ones((1, CH), np.float32) * h * tokens)
    w.add_tensor("blk.0.ffn_gate_inp.weight.counts", np.asarray([[tokens]], np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def one(wm):
    t = torch.tensor(wm)
    return float(((t - P._rtn(t, 2)) ** 2).sum())


def test_legacy_imatrix_weights_experts_by_mean_routing(tmp_path, capsys):
    model, mat = tmp_path / "m.gguf", tmp_path / "m.dat"
    write_model(model, n_used=1)
    write_legacy(mat)
    costs, noise, _ = P._imatrix_sensitivity(str(model), str(mat), ["ffn", "attn"], 2, [("iq2_s", 2)])
    per_expert = sum(one(W_EXP[e]) for e in range(E))
    assert costs["ffn"]["0"] == pytest.approx(per_expert / E)        # 1 of 4 experts per token
    assert costs["attn"]["0"] == pytest.approx(one(W_ATT))
    assert noise["iq2_s"] == pytest.approx(per_expert / E + one(W_ATT))
    assert "routing frequency: mean routing 1/4" in capsys.readouterr().out


def test_gguf_imatrix_uses_exact_per_expert_counts(tmp_path, capsys):
    model, mat = tmp_path / "m.gguf", tmp_path / "m.imatrix.gguf"
    write_model(model, n_used=2, router=True)
    counts, tokens = [10, 0, 6, 4], 10                 # 2 experts per token: counts sum to 2 x tokens
    write_gguf_imatrix(mat, counts, tokens)
    costs, _, _ = P._imatrix_sensitivity(str(model), str(mat), ["ffn", "attn"], 2, [])
    want = sum(one(W_EXP[e]) * counts[e] / tokens for e in range(E))
    assert costs["ffn"]["0"] == pytest.approx(want)
    assert "per-expert counts / router tokens" in capsys.readouterr().out


def test_gguf_imatrix_per_expert_mean_uses_each_experts_count(tmp_path):
    # the GGUF reader used to divide EVERY expert's sums by expert 0's count
    mat = tmp_path / "m.imatrix.gguf"
    write_gguf_imatrix(mat, [10, 5, 2, 1], 10, h=3.0)
    H, counts = P._imatrix_stats(str(mat))
    assert H["blk.0.ffn_up_exps.weight"] == pytest.approx([3.0] * (E * CH))
    assert counts["blk.0.ffn_up_exps.weight"] == [10, 5, 2, 1]


def test_no_routing_information_refuses_rather_than_overweighting(tmp_path):
    model, mat = tmp_path / "m.gguf", tmp_path / "m.dat"
    write_model(model, meta=False)
    write_legacy(mat)
    with pytest.raises(SystemExit, match="partial"):
        P._imatrix_sensitivity(str(model), str(mat), ["ffn", "attn"], 2, [])


def test_profile_is_marked_and_fit_flags_an_old_one(tmp_path, monkeypatch):
    model, mat, out = tmp_path / "m.gguf", tmp_path / "m.dat", tmp_path / "p.json"
    write_model(model)
    write_legacy(mat)
    monkeypatch.setattr(sys, "argv", ["pollard-probe", "--from-imatrix", str(mat), "--gguf", str(model),
                                      "--out", str(out)])
    P.main()
    prof = json.load(open(out))
    assert prof["experts"] == P.EXPERT_WEIGHTING == "routing-frequency"
    moe = {"kind": "moe"}
    assert F.stale_moe_profile(prof, moe) is None
    old = {k: v for k, v in prof.items() if k != "experts"}
    assert "routing frequency" in F.stale_moe_profile(old, moe)
    assert F.stale_moe_profile(old, {"kind": "dense"}) is None        # dense: the fix changes nothing
    assert F.stale_moe_profile({"method": "pollard-probe (torch RTN proxy)"}, moe) is None


def test_release_reprobes_an_old_imatrix_profile(tmp_path):
    import pollard_release as R
    prof = tmp_path / "p.json"
    step = R.Step(id="prep/probe", kind="prep", cmd=["pollard-probe", "--from-imatrix", "x", "--gguf", "y",
                                                       "--out", str(prof)], out=str(prof))
    done = {"prep/probe": {"status": "done"}}
    prof.write_text(json.dumps({"method": "imatrix-hessian", "ffn": {}}))
    assert R.resume_reason(step, done) is None                 # pre-fix profile: re-probe
    prof.write_text(json.dumps({"method": "imatrix-hessian", "ffn": {}, "experts": "routing-frequency"}))
    assert R.resume_reason(step, done) == "done"


def test_clef_omni_profile_shape_no_longer_crushes_attention():
    # the measured clef-omni profile, 2026-10-22, every 4th layer (ffn = summed conditional expert cost, attn):
    # with the experts at their routing weight (8/128) the IQ4_XS budget keeps attention off iq2_xxs.
    pts = {0: (1414, .1384), 4: (9815, 1.767), 8: (1.11e4, 4.379), 12: (1.331e4, 4.712), 16: (1.841e4, 8.27),
           20: (2.323e4, 8.423), 24: (2.75e4, 18.48), 28: (3.523e4, 29.08), 32: (4.185e4, 22.74),
           36: (5.286e4, 61.71), 40: (9.954e4, 239.0), 44: (1.76e5, 556.0), 47: (5.761e5, 1983.0)}
    keys = sorted(pts)
    def at(i, k):                                        # nearest measured layer
        return pts[min(keys, key=lambda x: abs(x - i))][k]
    arch = {"layers": 48, "total": 30.6e9, "kind": "moe", "expert_params": 3 * 2048 * 768, "n_experts": 128,
            "attn_params": 2 * 2048 * 4096 + 2 * 2048 * 512}
    def attn_types(scale):
        prof = {"ffn": {str(i): at(i, 0) * scale for i in range(48)}, "attn": {str(i): at(i, 1) for i in range(48)}}
        ov = F.plan_allocation(arch, 22.4, 0, prof)[0]
        return [t for p, t in ov if "attn_" in p]
    before, after = attn_types(1.0), attn_types(8 / 128)
    assert before.count("iq2_xxs") >= 20                  # the shipped IQ4_XS rung: 23 layers of attention at 2 bits
    assert after.count("iq2_xxs") <= 4
