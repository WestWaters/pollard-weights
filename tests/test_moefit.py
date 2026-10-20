"""pollard-moefit / pollard-reap / pollard-prune --score reap: per-expert MoE allocation with 1-bit experts.

The allocation, the knapsack, the error measurement, the imatrix routing reader, the EXL3 K1 recipe and the
EXLR8 per-expert weighting run on numpy + gguf-py alone (core and nix CI shapes). The REAP hooks and the K1
trellis encode need torch and skip without it."""
from __future__ import annotations

import itertools
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_moefit as M  # noqa: E402


# ------------------------------------------------------------------------------------------------ roles, sizes

def test_roles_keep_router_head_and_shared_apart_from_routed_experts():
    r = M.role_of
    assert r("blk.3.ffn_up_exps.weight", 3) == M.ROLE_EXPERT
    assert r("blk.3.ffn_down_exps.weight", 3) == M.ROLE_EXPERT
    assert r("blk.3.ffn_gate_shexp.weight") == M.ROLE_SHARED
    assert r("blk.3.ffn_gate_inp.weight") == M.ROLE_ROUTER
    assert r("blk.3.exp_probs_b.bias", 1) == M.ROLE_OTHER          # 1-D: never allocated
    assert r("blk.3.attn_q.weight") == M.ROLE_ATTN
    assert r("blk.3.ssm_in.weight") == M.ROLE_ATTN
    assert r("blk.3.ffn_down.weight") == M.ROLE_DENSE
    assert r("output.weight") == M.ROLE_HEAD and r("token_embd.weight") == M.ROLE_EMBD
    assert r("blk.3.attn_norm.weight", 1) == M.ROLE_OTHER


def test_type_bytes_are_the_exact_ggml_block_layouts():
    assert M.type_bytes("iq1_s", 256) == 50 and M.type_bytes("iq1_m", 512) == 112
    assert M.type_bytes("q8_0", 64) == 68 and M.type_bytes("iq2_xxs", 256) == 66
    assert abs(M.bpw_of("iq1_s") - 1.5625) < 1e-12 and abs(M.bpw_of("q6_K") - 6.5625) < 1e-12
    gguf = pytest.importorskip("gguf")
    for t, (bs, ts) in M.QUANT_SIZES.items():
        q = getattr(gguf.GGMLQuantizationType, t.upper())
        assert gguf.GGML_QUANT_SIZES[q] == (bs, ts), t


# ------------------------------------------------------------------------------------------------ error measure

def test_weighted_error_weighs_input_channels_by_activation():
    W = np.array([[1.0, 1.0], [1.0, 1.0]])
    Wq = np.array([[1.0, 0.0], [1.0, 1.0]])                        # error only in input channel 1
    e_hot, g_hot = M.weighted_rel_err(W, Wq, np.array([0.0, 1.0]))
    e_cold, g_cold = M.weighted_rel_err(W, Wq, np.array([1.0, 0.0]))
    assert e_hot[0] / g_hot[0] == pytest.approx(0.5) and e_cold[0] == 0.0


def test_tensor_error_is_count_weighted_so_cold_experts_weigh_their_traffic():
    err, energy = np.array([0.5, 0.01]), np.array([1.0, 1.0])
    hot_is_bad = M.aggregate(err, energy, np.array([100.0, 1.0]))
    cold_is_bad = M.aggregate(err, energy, np.array([1.0, 100.0]))
    assert hot_is_bad > 0.45 and cold_is_bad < 0.02


def _write_gguf(path, tensors, fields=None, arch="granitemoe"):
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), arch)
    for k, (v, t) in (fields or {}).items():
        w.add_key_value(k, v, t)
    for name, arr in tensors.items():
        if isinstance(arr, tuple):
            data, qt = arr
            w.add_tensor(name, data, raw_dtype=qt)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _imatrix(path, entries):
    """a GGUF-format imatrix: <name>.in_sum2 [n_mat, n_in] + <name>.counts [n_mat, 1], like llama-imatrix writes."""
    gguf = pytest.importorskip("gguf")
    w = gguf.GGUFWriter(str(path), "imatrix")
    w.add_string("general.type", "imatrix")
    for name, (s, c) in entries.items():
        w.add_tensor(name + ".in_sum2", np.asarray(s, dtype=np.float32))
        w.add_tensor(name + ".counts", np.asarray(c, dtype=np.float32).reshape(-1, 1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_imatrix_reader_gives_per_expert_routing_counts(tmp_path):
    p = _imatrix(tmp_path / "m.imatrix.gguf", {
        "blk.0.ffn_up_exps.weight": (np.ones((4, 256)), [30, 10, 0, 60]),
        "blk.0.ffn_gate_exps.weight": (np.ones((4, 256)), [30, 10, 0, 60]),
        "blk.0.attn_q.weight": (np.ones((1, 256)), [100]),
    })
    imx, kind = M.read_imatrix(p)
    assert kind == "gguf"
    s, c = imx["blk.0.ffn_up_exps.weight"]
    assert s.shape == (4, 256) and list(c) == [30, 10, 0, 60]
    prof = M.routing_profile(imx)
    assert list(prof) == [0] and prof[0]["share"] == pytest.approx([0.3, 0.1, 0.0, 0.6])
    row = M.routing_summary(prof)[0]
    assert row["dead"] == 1 and row["experts"] == 4 and 0 < row["entropy"] < 1


def test_measure_probe_reads_a_quantized_build_back_per_expert(tmp_path):
    gguf = pytest.importorskip("gguf")
    from gguf import quants
    rng = np.random.default_rng(0)
    E, rows, n_in = 4, 8, 256
    W = (rng.standard_normal((E, rows, n_in)) * 0.02).astype(np.float32)
    A = (rng.standard_normal((rows, n_in)) * 0.02).astype(np.float32)
    src = _write_gguf(tmp_path / "src.gguf", {"blk.0.ffn_up_exps.weight": W.astype(np.float16),
                                              "blk.0.attn_q.weight": A.astype(np.float16)})
    q8 = gguf.GGMLQuantizationType.Q8_0
    prb = _write_gguf(tmp_path / "p.gguf", {"blk.0.ffn_up_exps.weight": (quants.quantize(W, q8), q8),
                                            "blk.0.attn_q.weight": (quants.quantize(A, q8), q8)})
    imx = {"blk.0.ffn_up_exps.weight": (np.ones((E, n_in)) * 4.0, np.array([4.0, 4.0, 4.0, 4.0]))}
    sr = gguf.GGUFReader(src)
    cands = M.candidate_tensors(sr)
    assert {c[0] for c in cands} == {"blk.0.ffn_up_exps.weight", "blk.0.attn_q.weight"}
    store = {n: {"rel": {}} for n, *_ in cands}
    M.measure_probe(sr, gguf.GGUFReader(prb), cands, imx, store, "q8_0")
    ex = store["blk.0.ffn_up_exps.weight"]
    assert 0 < ex["rel"]["q8_0"] < 1e-3                        # q8_0 is near-lossless
    assert len(ex["experts"]["err"]["q8_0"]) == E and ex["experts"]["count"] == [4.0] * 4
    assert "type_mismatch" not in ex and store["blk.0.attn_q.weight"]["covered"] is False


# ------------------------------------------------------------------------------------------------ the plan

def _errmap(rel_by_tensor, roles, numel=256 * 1024, fixed=None, types=("iq1_s", "iq2_xxs", "iq3_s", "iq4_xs", "q6_K")):
    tensors = {}
    for n, rel in rel_by_tensor.items():
        tensors[n] = {"role": roles[n], "layer": M.layer_of(n), "numel": numel, "shape": [1024, 256],
                      "bytes": {q: M.type_bytes(q, numel) for q in types}, "rel": dict(rel)}
    return {"format": M.FORMAT_ERRMAP, "types": list(types), "tensors": tensors,
            "fixed": fixed or {"output.weight": {"role": M.ROLE_HEAD, "numel": 256 * 64, "src_bytes": 256 * 64 * 2}}}


def _rel(scale):
    return {"iq1_s": 0.30 * scale, "iq2_xxs": 0.12 * scale, "iq3_s": 0.04 * scale, "iq4_xs": 0.015 * scale,
            "q6_K": 0.002 * scale}


def test_plan_is_exact_against_brute_force_and_respects_the_budget():
    names = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.1.ffn_up_exps.weight"]
    em = _errmap({n: _rel(s) for n, s in zip(names, (1.0, 3.0, 0.2))}, {n: M.ROLE_EXPERT for n in names})
    fixed = M.fixed_bytes(em)
    budget = fixed + sum(M.type_bytes(q, 256 * 1024) for q in ("iq2_xxs", "iq3_s", "iq1_s"))
    p = M.plan(em, budget)
    assert p["bytes"] <= budget
    best = None
    for combo in itertools.product(em["types"], repeat=3):
        size = fixed + sum(M.type_bytes(q, 256 * 1024) for q in combo)
        if size > budget:
            continue
        cost = sum(em["tensors"][n]["rel"][q] / 3 for n, q in zip(names, combo))
        best = cost if best is None or cost < best else best
    assert p["predicted_kld"] == pytest.approx(best, rel=1e-9)
    # the most sensitive tensor got the most bits, the least sensitive the fewest
    assert M.bpw_of(p["types"][names[1]]) >= M.bpw_of(p["types"][names[0]]) >= M.bpw_of(p["types"][names[2]])


def test_floors_hold_attention_and_shared_high_and_no_1bit_stops_experts_at_iq2():
    names = {"blk.0.attn_q.weight": M.ROLE_ATTN, "blk.0.ffn_gate_shexp.weight": M.ROLE_SHARED,
             "blk.0.ffn_up_exps.weight": M.ROLE_EXPERT, "blk.0.ffn_down_exps.weight": M.ROLE_EXPERT}
    em = _errmap({n: _rel(0.01 if r != M.ROLE_EXPERT else 1.0) for n, r in names.items()}, names)
    tight = M.fixed_bytes(em) + 2 * M.type_bytes("iq4_xs", 256 * 1024) + 2 * M.type_bytes("iq1_s", 256 * 1024)
    p = M.plan(em, tight)
    assert p["types"]["blk.0.attn_q.weight"] == "iq4_xs" and p["types"]["blk.0.ffn_gate_shexp.weight"] == "iq4_xs"
    assert p["hist"][M.ROLE_EXPERT] == {"iq1_s": 2}
    with pytest.raises(ValueError):                              # experts can't reach 1 bit -> doesn't fit
        M.plan(em, tight, allow_1bit=False)
    roomy = M.plan(em, tight + 2 * M.type_bytes("iq2_xxs", 256 * 1024), allow_1bit=False)
    assert all(M.bpw_of(q) >= M.bpw_of("iq2_xxs") for n, q in roomy["types"].items() if names[n] == M.ROLE_EXPERT)
    with pytest.raises(ValueError, match="fixed"):
        M.plan(em, 10)


def test_measured_sensitivity_moves_bits_to_the_sensitive_layer():
    names = ["blk.0.ffn_up_exps.weight", "blk.1.ffn_up_exps.weight"]
    em = _errmap({n: _rel(1.0) for n in names}, {n: M.ROLE_EXPERT for n in names})
    budget = M.fixed_bytes(em) + M.type_bytes("iq1_s", 256 * 1024) + M.type_bytes("iq3_s", 256 * 1024)
    sens = {"ffn": {"0": 0.9, "1": 0.01}, "attn": {}, "probe": "iq2_xxs"}
    p = M.plan(em, budget, sens)
    assert p["types"] == {names[0]: "iq3_s", names[1]: "iq1_s"} and p["estimate"] == "measured"
    flipped = M.plan(em, budget, {"ffn": {"0": 0.01, "1": 0.9}, "attn": {}, "probe": "iq2_xxs"})
    assert flipped["types"] == {names[0]: "iq1_s", names[1]: "iq3_s"}


def test_tensor_types_file_is_anchored_and_escaped(tmp_path):
    p = {"types": {"blk.1.ffn_up_exps.weight": "iq1_s", "blk.11.attn_q.weight": "iq4_xs"}}
    path = M.write_tensor_types(p, str(tmp_path / "t.txt"))
    lines = open(path).read().splitlines()
    assert lines == [r"^blk\.1\.ffn_up_exps\.weight$=iq1_s", r"^blk\.11\.attn_q\.weight$=iq4_xs"]
    import re
    pat = re.compile(lines[0].split("=")[0])
    assert pat.search("blk.1.ffn_up_exps.weight") and not pat.search("blk.11.ffn_up_exps.weight")
    assert not pat.search("dec.blk.1.ffn_up_exps.weight")


def test_uniform_reference_plan_puts_every_expert_at_one_type():
    names = {"blk.0.attn_q.weight": M.ROLE_ATTN, "blk.0.ffn_up_exps.weight": M.ROLE_EXPERT}
    em = _errmap({n: _rel(1.0) for n in names}, names)
    u = M.uniform_plan(em, "iq2_xxs")
    assert u["types"] == {"blk.0.attn_q.weight": "iq4_xs", "blk.0.ffn_up_exps.weight": "iq2_xxs"}
    assert u["bytes"] == M.fixed_bytes(em) + M.type_bytes("iq4_xs", 256 * 1024) + M.type_bytes("iq2_xxs", 256 * 1024)


def test_cli_plan_writes_plan_and_tensor_types(tmp_path, capsys):
    names = {"blk.0.attn_q.weight": M.ROLE_ATTN, "blk.0.ffn_up_exps.weight": M.ROLE_EXPERT}
    em = _errmap({n: _rel(1.0) for n in names}, names)
    (tmp_path / "em.json").write_text(json.dumps(em))
    rc = M.main(["plan", "--errmap", str(tmp_path / "em.json"), "--bpw", "4.0", "--out", str(tmp_path / "p.json"),
                 "--tensor-types", str(tmp_path / "t.txt")])
    assert rc == 0
    p = json.loads((tmp_path / "p.json").read_text())
    assert p["format"] == M.FORMAT_PLAN and p["bpw_file"] <= 4.0 + 1e-9
    assert len((tmp_path / "t.txt").read_text().splitlines()) == 2


# ------------------------------------------------------------------------------------------------ EXL3: K1

def _exl3_budgeted(experts=4, layers=2, H=256, I=128):
    out = []
    for L in range(layers):
        p = f"model.layers.{L}"
        out += [(f"{p}.self_attn.{n}", H * H) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        for e in range(experts):
            out += [(f"{p}.mlp.experts.{e}.gate_proj", I * H), (f"{p}.mlp.experts.{e}.up_proj", I * H),
                    (f"{p}.mlp.experts.{e}.down_proj", I * H)]
    return out


def test_exl3_law_is_continuous_at_k2_and_steeper_below():
    import pollard_exl3 as X
    assert X.kld_law(1.0, 2) == pytest.approx(16.0)
    assert X.kld_law(1.0, 1) / X.kld_law(1.0, 2) == pytest.approx(2 ** 2.35)
    assert X.kld_law(1.0, 3) / X.kld_law(1.0, 2) == pytest.approx(0.25)


def test_exl3_per_expert_recipe_sends_cold_experts_to_k1_and_hot_ones_up():
    budgeted = _exl3_budgeted()
    routing = {"0": {"share": [0.85, 0.13, 0.01, 0.01]}, "1": {"share": [0.25, 0.25, 0.25, 0.25]}}
    sens = {"attn": {"0": 0.05, "1": 0.05}, "ffn": {"0": 0.5, "1": 0.5}, "probe": "rtn4"}
    r = M.exl3_recipe(budgeted, routing, sens, 2.0, granularity="expert", nonexpert_floor=3)
    t = r["tensors"]
    assert r["achieved_bpw"] <= 2.0 + 1e-9
    assert t["model.layers.0.mlp.experts.2.gate_proj"] == 1 and t["model.layers.0.mlp.experts.3.down_proj"] == 1
    assert t["model.layers.0.mlp.experts.0.down_proj"] > t["model.layers.0.mlp.experts.2.down_proj"]
    for L in range(2):                                     # gate + up of one expert share K (EXL3's tie)
        for e in range(4):
            assert t[f"model.layers.{L}.mlp.experts.{e}.gate_proj"] == t[f"model.layers.{L}.mlp.experts.{e}.up_proj"]
    assert all(t[f"model.layers.0.self_attn.{n}"] >= 3 for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert r["needs_mixed_k_moe"] is True and 1 in r["expert_k_hist"]
    assert set(t) == {k for k, _ in budgeted}


def test_exl3_layer_granularity_is_what_stock_exllamav3_loads():
    budgeted = _exl3_budgeted()
    routing = {"0": {"share": [0.85, 0.13, 0.01, 0.01]}, "1": {"share": [0.25] * 4}}
    r = M.exl3_recipe(budgeted, routing, None, 1.8, granularity="layer", nonexpert_floor=2)
    t = r["tensors"]
    for L in range(2):
        for proj in (("gate_proj", "up_proj"), ("down_proj",)):
            ks = {t[f"model.layers.{L}.mlp.experts.{e}.{p}"] for e in range(4) for p in proj}
            assert len(ks) == 1, (L, proj, ks)               # one K per (layer, projection): MultiLinear's assert
    assert r["needs_mixed_k_moe"] is False and r["achieved_bpw"] <= 1.8 + 1e-9


def test_exl3_recipe_passes_exl3s_own_checks_with_k1(tmp_path):
    import pollard_exl3 as X
    budgeted = _exl3_budgeted()
    r = M.exl3_recipe(budgeted, None, None, 1.5, granularity="expert", nonexpert_floor=1)
    assert X.check_recipe(r, [k for k, _ in budgeted]) == []
    assert min(r["tensors"].values()) == 1
    X.write_recipe(r, str(tmp_path / "r.yaml"))
    yaml = pytest.importorskip("yaml")
    back = yaml.safe_load(open(tmp_path / "r.yaml"))
    assert back["tensors"] == r["tensors"]


def test_pollard_exl3_recipe_accepts_k1_and_refuses_k0():
    import pollard_exl3 as X
    budgeted = _exl3_budgeted()
    sens = {"attn": {"0": 0.05, "1": 0.05}, "ffn": {"0": 0.5, "1": 0.01}, "probe": "rtn4"}
    r = X.pollard_recipe(budgeted, sens, 1.5, min_bits=1)
    assert min(r["tensors"].values()) == 1 and r["achieved_bpw"] <= 1.5 + 1e-9
    with pytest.raises(ValueError):
        X.pollard_recipe(budgeted, sens, 1.5, min_bits=0)


# ------------------------------------------------------------------------------------------------ EXLR8: K1 + routing

def test_exlr8_k1_packs_and_unpacks():
    import pollard_exlr8 as X
    rng = np.random.default_rng(1)
    vals = rng.integers(0, 2, size=(3, 256))
    raw = X.pack_values(vals, 1)
    assert raw.shape == (3, 32)
    assert (X.unpack_values(raw, 1) == vals).all()
    st = X.states_from_values(vals, 1)
    assert st.max() < 65536 and X.decode_tiles(raw, 1).shape == (3, 16, 16)


def _store(tokens):
    t = {}
    for e, tok in enumerate(tokens):
        t[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = {
            "layer": 0, "numel": 128 * 128, "hessian_tokens": tok, "err_rtn": 0.01,
            "err": {"1": 0.2, "2": 0.08, "3": 0.03, "4": 0.01}, "bytes": {str(K): 128 * 128 * K // 8 for K in (1, 2, 3, 4)}}
    t["model.layers.0.self_attn.q_proj.weight"] = {
        "layer": 0, "numel": 128 * 128, "err_rtn": 0.01, "err": {"1": 0.2, "2": 0.08, "3": 0.03, "4": 0.01},
        "bytes": {str(K): 128 * 128 * K // 8 for K in (1, 2, 3, 4)}}
    return {"widths": [1, 2, 3, 4], "tensors": t}


def test_exlr8_routing_weights_follow_the_per_expert_token_counts():
    import pollard_exlr8 as X
    meta = _store([900, 90, 9, 1])
    rw = X.routing_weights(meta["tensors"])
    assert rw["model.layers.0.mlp.experts.0.down_proj.weight"] == pytest.approx(4 * 0.9)
    assert rw["model.layers.0.self_attn.q_proj.weight"] == 1.0
    sens = {"ffn": {"0": 1.0}, "attn": {"0": 0.1}}
    flat = X.tensor_sensitivity(meta["tensors"], sens)
    routed = X.tensor_sensitivity(meta["tensors"], sens, routing=True)
    ffn = lambda d: sum(v for k, v in d.items() if "experts" in k)
    assert ffn(routed) == pytest.approx(ffn(flat))         # the layer keeps its measured S
    assert routed["model.layers.0.mlp.experts.3.down_proj.weight"] < flat["model.layers.0.mlp.experts.3.down_proj.weight"] / 50


def test_exlr8_plan_with_routing_puts_the_cold_expert_at_k1():
    import pollard_exlr8 as X
    meta = _store([900, 90, 9, 1])
    sens = {"ffn": {"0": 1.0}, "attn": {"0": 1.0}}
    p = X.plan_rung(meta, 2.4, sens, routing=True)
    w = p["widths"]
    assert w["model.layers.0.mlp.experts.3.down_proj.weight"] == 1
    assert w["model.layers.0.mlp.experts.0.down_proj.weight"] >= 3
    flat = X.plan_rung(meta, 2.4, sens)
    assert len({flat["widths"][f"model.layers.0.mlp.experts.{e}.down_proj.weight"] for e in range(4)}) == 1


def test_exlr8_encoder_runs_at_k1():
    torch = pytest.importorskip("torch")
    import pollard_exlr8 as X
    torch.manual_seed(0)
    Wg = torch.randn(128, 128) * 0.02
    enc = X.encode_matrix(Wg, 1, None, "k1", mem_bytes=96 << 20)
    rec = X.trellis_record(enc, 128, 128, 1, X.SIDE_OUT)
    dec = X.decode_record(rec["record"], rec)
    assert np.abs(dec - enc["W_hat"].numpy()).max() <= 1e-6 * np.abs(dec).max()
    assert rec["trellis"].shape == (1, 128 * 128 // 8)
    e1 = X.rel_err(Wg, torch.tensor(dec), None)
    assert 0 < e1 < 0.6


# ------------------------------------------------------------------------------------------------ REAP + prune

def test_reap_scores_are_router_weight_times_expert_output_norm():
    torch = pytest.importorskip("torch")
    import pollard_reap as R

    class Experts(torch.nn.Module):                      # transformers >= 5 fused-experts layout
        def __init__(self, E=3, H=8, I=4):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * I, H, generator=g))
            self.down_proj = torch.nn.Parameter(torch.randn(E, H, I, generator=g))
            self.act_fn = torch.nn.functional.silu

        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.layers[0].experts = Experts()

    m = Model()
    mods = R.find_expert_modules(m)
    assert [L for L, _ in mods] == [0]
    acc = R.ReapAccumulator({0: 3})
    x = torch.randn(5, 8)
    idx = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]])
    w = torch.tensor([[0.9, 0.1]] * 5)
    acc.add(0, m.layers[0].experts, x, idx, w)
    res = acc.result()["0"]
    ex = m.layers[0].experts
    want = []
    for j in range(3):
        tok, pos = torch.where(idx == j)
        if tok.numel() == 0:
            want.append(0.0)
            continue
        gate, up = (x[tok] @ ex.gate_up_proj[j].T).chunk(2, -1)
        f = (torch.nn.functional.silu(gate) * up) @ ex.down_proj[j].T
        want.append(float((w[tok, pos].double() * f.double().norm(dim=-1)).mean()))
    assert res["reap"] == pytest.approx(want, rel=1e-5)
    assert res["count"] == [5.0, 5.0, 0.0]


def test_prune_keeps_the_top_reap_experts_and_preserves_key_types(tmp_path):
    gguf = pytest.importorskip("gguf")
    E, H, I = 4, 32, 16
    rng = np.random.default_rng(0)
    up = rng.standard_normal((E, I, H)).astype(np.float16)
    down = rng.standard_normal((E, H, I)).astype(np.float16)
    router = rng.standard_normal((E, H)).astype(np.float32)
    for e in range(E):
        up[e] = e                                        # expert index readable after slicing
    V = gguf.GGUFValueType
    src = _write_gguf(tmp_path / "moe.gguf", {"blk.0.ffn_up_exps.weight": up, "blk.0.ffn_down_exps.weight": down,
                                              "blk.0.ffn_gate_inp.weight": router,
                                              "blk.0.attn_norm.weight": np.ones(H, dtype=np.float32)},
                      fields={"granitemoe.expert_count": (E, V.UINT32), "granitemoe.expert_used_count": (2, V.UINT32),
                              "tokenizer.ggml.add_bos_token": (True, V.BOOL),
                              "granitemoe.attention.scale": (0.25, V.FLOAT32)})
    scores = tmp_path / "s.json"
    scores.write_text(json.dumps({"format": "pollard-reap", "layers": {"0": {"reap": [0.1, 5.0, 0.2, 3.0],
                                                                             "count": [1, 1, 1, 1], "ean": [1, 1, 1, 1]}}}))
    out = tmp_path / "p.gguf"
    import pollard_prune as P
    sys.argv = ["pollard-prune", "--gguf", src, "--keep", "0.5", "--score", "reap", "--scores", str(scores),
                "--out", str(out), "--keep-list", str(tmp_path / "keep.json")]
    P.main()
    r = gguf.GGUFReader(str(out))
    f = r.fields
    assert int(f["granitemoe.expert_count"].contents()) == 2
    assert f["tokenizer.ggml.add_bos_token"].types[0] == V.BOOL and f["tokenizer.ggml.add_bos_token"].contents() is True
    assert f["granitemoe.attention.scale"].types[0] == V.FLOAT32
    t = {x.name: np.asarray(x.data) for x in r.tensors}
    assert t["blk.0.ffn_up_exps.weight"].shape == (2, I, H)
    assert sorted({float(t["blk.0.ffn_up_exps.weight"][k].flat[0]) for k in range(2)}) == [1.0, 3.0]
    assert t["blk.0.ffn_gate_inp.weight"].shape == (2, H)
    assert np.allclose(t["blk.0.ffn_gate_inp.weight"], router[[1, 3]])
    assert json.loads((tmp_path / "keep.json").read_text())["layers"]["0"] == [1, 3]


def test_prune_refuses_scores_of_another_model(tmp_path):
    pytest.importorskip("gguf")
    import pollard_prune as P
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"format": "pollard-reap", "layers": {"0": {"reap": [1, 2, 3]}}}))
    with pytest.raises(SystemExit):
        P.reap_layers(str(s), 4)


def test_reap_handles_gateless_relu2_experts_like_nemotron_h():
    torch = pytest.importorskip("torch")
    import pollard_reap as R

    class Experts(torch.nn.Module):                      # Nemotron-H: up_proj [E, I, D], down_proj [E, D, I], relu2
        def __init__(self, E=2, D=8, I=4):
            super().__init__()
            g = torch.Generator().manual_seed(1)
            self.up_proj = torch.nn.Parameter(torch.randn(E, I, D, generator=g))
            self.down_proj = torch.nn.Parameter(torch.randn(E, D, I, generator=g))
            self.act_fn = lambda x: torch.relu(x) ** 2

    ex = Experts()
    assert R.is_experts_module(ex)
    x = torch.randn(3, 8)
    f = R.expert_outputs(ex, x, 1)
    want = (torch.relu(x @ ex.up_proj[1].T) ** 2) @ ex.down_proj[1].T
    assert torch.allclose(f, want)
    holder = torch.nn.Module()
    holder.experts = ex
    assert [L for L, _ in R.find_expert_modules(holder, layer=7)] == [7]


def test_passthrough_keeps_qat_bytes_never_upgrades_and_only_moves_cold_tensors_down():
    names = {"blk.0.ffn_up_exps.weight": M.ROLE_EXPERT, "blk.1.ffn_up_exps.weight": M.ROLE_EXPERT,
             "blk.0.ssm_in.weight": M.ROLE_ATTN}
    em = _errmap({"blk.0.ffn_up_exps.weight": {"iq1_s": 0.5, "iq2_xxs": 0.0, "iq3_s": 0.001},
                  "blk.1.ffn_up_exps.weight": {"iq1_s": 0.01, "iq2_xxs": 0.0, "iq3_s": 0.001},
                  "blk.0.ssm_in.weight": {"iq4_xs": 0.02, "q6_K": 0.01}}, names)
    em["passthrough"] = True
    n = 256 * 1024
    for k, st in (("blk.0.ffn_up_exps.weight", "iq2_xxs"), ("blk.1.ffn_up_exps.weight", "iq2_xxs"),
                  ("blk.0.ssm_in.weight", "nvfp4")):
        em["tensors"][k]["src_type"] = st
        em["tensors"][k]["src_bytes"] = M.type_bytes(st, n)
    full = M.fixed_bytes(em) + 2 * M.type_bytes("iq2_xxs", n) + M.type_bytes("nvfp4", n)
    p = M.plan(em, full)
    assert p["types"] == {"blk.0.ffn_up_exps.weight": "iq2_xxs", "blk.1.ffn_up_exps.weight": "iq2_xxs",
                          "blk.0.ssm_in.weight": "nvfp4"} and p["predicted_kld"] == 0.0
    smaller = M.plan(em, full - (M.type_bytes("iq2_xxs", n) - M.type_bytes("iq1_s", n)))
    assert smaller["types"]["blk.1.ffn_up_exps.weight"] == "iq1_s"          # the cold one goes to 1 bit
    assert smaller["types"]["blk.0.ffn_up_exps.weight"] == "iq2_xxs"        # the hot one keeps its QAT bytes
    assert smaller["types"]["blk.0.ssm_in.weight"] == "nvfp4"
    assert "iq3_s" not in {q for q in smaller["types"].values()}


# ------------------------------------------------------------------------------------------------ measured 1-bit cost

def _two_expert_em():
    """weight-space error that flattens at the low end (iq1_s barely worse than iq2_xxs in e) -- the shape that made
    the extrapolated law price 1-bit cheaply on granite"""
    names = ["blk.0.ffn_up_exps.weight", "blk.1.ffn_up_exps.weight"]
    rel = {"iq1_s": 0.18, "iq2_xxs": 0.12, "iq3_s": 0.03, "iq4_xs": 0.015, "q6_K": 0.002}
    return names, _errmap({n: dict(rel) for n in names}, {n: M.ROLE_EXPERT for n in names})


SENS = {"ffn": {"0": 0.1, "1": 0.1}, "attn": {}, "probe": "iq2_xxs"}


def test_lowbit_cost_uses_the_measurement_and_scales_other_types_from_the_nearest_anchor():
    t = {"rel": _rel(1.0), "numel": 1}
    meas = {"iq1_s": 0.9, "iq2_xxs": 0.1}
    assert M.lowbit_cost(meas, t, "iq1_s") == 0.9                    # measured: as measured
    assert M.lowbit_cost(meas, t, "iq3_s") == pytest.approx(0.1 * 0.04 / 0.12)   # above: from the 2-bit anchor
    # the extrapolated law would have priced iq1_s at 0.1 * 0.30 / 0.12 = 0.25 -- the measurement says 0.9
    assert M.lowbit_cost(meas, t, "iq1_s") > 0.1 * 0.30 / 0.12


def test_layer_granularity_splits_a_class_by_measured_error():
    names = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"]
    em = _errmap({names[0]: _rel(1.0), names[1]: _rel(3.0)}, {n: M.ROLE_EXPERT for n in names})
    lb = {"format": M.FORMAT_LOWBIT, "classes": {"0": {"tensors": names, "kl": {"iq1_s": 0.8}}}}
    c = M.resolve_lowbit(lb, em["tensors"])
    assert c[names[0]]["iq1_s"] == pytest.approx(0.2) and c[names[1]]["iq1_s"] == pytest.approx(0.6)
    assert M.expert_class(names[1]) == "0.down" and M.expert_class(names[1], "layer") == "0"
    assert M.expert_class("blk.0.attn_q.weight") is None


def test_measured_1bit_cost_stops_the_knapsack_buying_1bit_that_does_not_pay():
    names, em = _two_expert_em()
    budget = M.fixed_bytes(em) + M.type_bytes("iq1_s", 256 * 1024) + M.type_bytes("iq3_s", 256 * 1024)
    # extrapolated: 1-bit on one tensor is cheap enough to buy iq3_s on the other
    p0 = M.plan(em, budget, SENS)
    assert "iq1_s" in p0["types"].values()
    # measured: iq1_s costs far more than the law said -> both at 2 bits instead
    lb = {"classes": {f"{i}.up": {"tensors": [n], "kl": {"iq1_s": 2.0, "iq2_xxs": 0.1}} for i, n in enumerate(names)}}
    p1 = M.plan(em, budget, SENS, lowbit=lb)
    assert "iq1_s" not in p1["types"].values() and p1["estimate"] == "measured+lowbit"
    # and where the measurement says 1-bit is nearly free on one tensor, it is bought there
    lb["classes"]["1.up"]["kl"]["iq1_s"] = 0.11
    p2 = M.plan(em, budget, SENS, lowbit=lb)
    assert p2["types"][names[1]] == "iq1_s" and M.bpw_of(p2["types"][names[0]]) > M.bpw_of("iq2_xxs")
    with pytest.raises(ValueError, match="sensitivity"):
        M.plan(em, budget, None, lowbit=lb)


# ------------------------------------------------------------------------------------------------ pruned candidates

def _expert_em(E=4, rows=256, n_in=256):
    numel = E * rows * n_in
    t = {"role": M.ROLE_EXPERT, "layer": 0, "numel": numel, "shape": [E, rows, n_in],
         "bytes": {q: M.type_bytes(q, numel) for q in ("iq1_s", "iq2_xxs")},
         "rel": {"iq1_s": 0.3, "iq2_xxs": 0.1},
         "experts": {"count": [10.0, 1.0, 50.0, 0.0], "energy": [1.0, 1.0, 1.0, 1.0],
                     "err": {"iq1_s": [0.1, 0.9, 0.2, 0.5], "iq2_xxs": [0.05, 0.3, 0.06, 0.2]}}}
    return {"format": M.FORMAT_ERRMAP, "types": ["iq1_s", "iq2_xxs"], "tensors": {"blk.0.ffn_up_exps.weight": t},
            "fixed": {"blk.0.ffn_gate_inp.weight": {"role": M.ROLE_ROUTER, "numel": 256 * E, "src_bytes": 4 * 256 * E}},
            "routing": {"0": {"counts": [10.0, 1.0, 50.0, 0.0], "share": [10 / 61, 1 / 61, 50 / 61, 0.0]}}}


def test_prune_errmap_keeps_the_kept_experts_bytes_and_errors():
    em = _expert_em()
    p = M.prune_errmap(em, {0: [0, 2]})
    t = p["tensors"]["blk.0.ffn_up_exps.weight"]
    assert t["shape"][0] == 2 and t["numel"] == em["tensors"]["blk.0.ffn_up_exps.weight"]["numel"] // 2
    assert t["bytes"]["iq1_s"] == M.type_bytes("iq1_s", t["numel"])
    assert t["rel"]["iq1_s"] == pytest.approx((10 * 0.1 + 50 * 0.2) / 60)
    assert p["fixed"]["blk.0.ffn_gate_inp.weight"]["numel"] == 512
    assert p["routing"]["0"]["counts"] == [10.0, 50.0]
    assert em["tensors"]["blk.0.ffn_up_exps.weight"]["shape"][0] == 4      # the full map is untouched
    lb = M.prune_lowbit({"classes": {"0.up": {"tensors": ["blk.0.ffn_up_exps.weight"], "kl": {"iq1_s": 0.6}}}}, em, p)
    assert lb["tensor_costs"]["blk.0.ffn_up_exps.weight"]["iq1_s"] == pytest.approx(0.6 * t["rel"]["iq1_s"] / 0.3)


def test_keep_sets_follow_the_scores_drop_below_top_k_and_dedupe():
    scores = {0: [5, 1, 9, 3, 7, 2, 8, 4], 1: [1, 2, 3, 4, 5, 6, 7, 8]}
    ks = M.keep_sets(8, 2, (0, 0.1, 0.25, 0.5, 0.75, 0.9), scores)
    assert sorted(ks) == [0.0, 0.1, 0.25, 0.5, 0.75]                   # 0.9 leaves 1 < top-2
    assert ks[0.5] == (4, {0: [0, 2, 4, 6], 1: [4, 5, 6, 7]})
    assert ks[0.1][0] == 7
    assert len(M.keep_sets(8, 2, (0.1, 0.12), scores)) == 1             # both round to K = 7


def test_slice_imatrix_keeps_the_kept_experts_rows(tmp_path):
    gguf = pytest.importorskip("gguf")
    p = _imatrix(tmp_path / "m.imatrix.gguf", {
        "blk.0.ffn_up_exps.weight": (np.arange(4 * 256).reshape(4, 256), [30, 10, 0, 60]),
        "blk.0.attn_q.weight": (np.ones((1, 256)), [100])})
    out = M.slice_imatrix(p, {0: [0, 3]}, str(tmp_path / "s.imatrix.gguf"))
    imx, kind = M.read_imatrix(out)
    s, c = imx["blk.0.ffn_up_exps.weight"]
    assert kind == "gguf" and s.shape == (2, 256) and list(c) == [30, 60] and s[1, 0] == 3 * 256
    assert imx["blk.0.attn_q.weight"][0].shape == (1, 256)
    r = gguf.GGUFReader(out)
    assert bytes(r.fields["general.type"].parts[r.fields["general.type"].data[0]]).decode() == "imatrix"


def test_auto_screens_every_fraction_measures_the_best_and_picks_the_lowest_measured(tmp_path, monkeypatch):
    """the whole search with the binaries faked: pruning damage + predicted mix rank the candidates, the top ones
    and a 1-bit plan's no-1-bit twin are 'built' and 'measured', a candidate whose tool calls drop is passed over."""
    E, rows, n_in = 8, 256, 256
    names = [f"blk.{L}.ffn_up_exps.weight" for L in range(2)]
    numel = E * rows * n_in
    rng = np.random.default_rng(1)
    em = {"format": M.FORMAT_ERRMAP, "types": ["iq1_s", "iq2_xxs", "iq3_s", "iq4_xs"], "tensors": {}, "fixed": {}}
    for n in names:
        errs = {q: list(rng.uniform(0.5, 1.5, E) * b) for q, b in
                (("iq1_s", 0.3), ("iq2_xxs", 0.12), ("iq3_s", 0.04), ("iq4_xs", 0.015))}
        em["tensors"][n] = {"role": M.ROLE_EXPERT, "layer": M.layer_of(n), "numel": numel, "shape": [E, rows, n_in],
                            "bytes": {q: M.type_bytes(q, numel) for q in em["types"]},
                            "rel": {q: float(np.mean(v)) for q, v in errs.items()},
                            "experts": {"count": list(rng.uniform(1, 10, E)), "energy": [1.0] * E, "err": errs}}
    sens = {"ffn": {"0": 0.3, "1": 0.3}, "attn": {}, "probe": "iq2_xxs"}
    lb = {"classes": {f"{L}.up": {"tensors": [names[L]], "kl": {"iq1_s": 0.6, "iq1_m": 0.45, "iq2_xxs": 0.3}}
                      for L in range(2)}}
    monkeypatch.setattr(M, "_gguf_experts", lambda p: (E, 2))
    monkeypatch.setattr(M, "read_imatrix", lambda p: ({n: (np.ones((E, n_in)), np.arange(1, E + 1, dtype=float))
                                                       for n in names}, "gguf"))
    monkeypatch.setattr(M, "_quantized_source", lambda p: False)
    monkeypatch.setattr(M, "_base_logits", lambda *a, **k: str(tmp_path / "base.kld"))
    monkeypatch.setattr(M, "_prune_gguf", lambda src, K, kl, sc, out, log=print: open(out, "wb").write(b"GGUF" + b"\0" * 2048))
    monkeypatch.setattr(M, "slice_imatrix", lambda imx, keep, out: open(out, "wb").write(b"GGUF" + b"\0" * 2048))
    prune_kl = {}

    def fake_kld(ppl, model, ev, base, ngl=99, chunks=None):
        b = os.path.basename(model)
        if b.endswith("-src.gguf"):                                    # the pruned model's own damage
            p = int(b[1:3]) / 100
            return {"mean_kld": prune_kl.setdefault(p, 0.4 * p * p)}
        plan = json.load(open(model + ".plan.json"))
        k = plan["predicted_kld"] + (0.05 if "-1bit" in b else 0.0)    # 1-bit measures worse than predicted
        return {"mean_kld": round(k + (0.1 if b.startswith("cand-p") and "p00" not in b else 0), 6), "top1": 80.0,
                "ppl": 5.0, "p99_kld": 1.0, "median_kld": 0.1}
    monkeypatch.setattr(M, "_kld", fake_kld)
    monkeypatch.setattr(M, "build", lambda src, im, plan, out, *a, **k: open(out, "wb").write(b"GGUF" + b"\0" * 2048))
    tools = {}
    monkeypatch.setattr(M, "_toolgate", lambda rung, ref, ngl, out, log=print: tools.setdefault(
        os.path.basename(rung), {"tool_retention": 50.0 if "p25" in rung else 100.0, "tool_rate": 0.5}))
    budget = M.fixed_bytes(em) + 2 * M.type_bytes("iq2_xxs", numel)
    out = str(tmp_path / "win.gguf")
    rep = M.auto("src.gguf", "imx.gguf", em, sens, lb, budget, prunes=(0, 0.25, 0.5), screen_eval="s.txt",
                 judge_eval="j.txt", workdir=str(tmp_path / "w"), out=out, top=2, log=lambda *a: None)
    rows = rep["candidates"]
    assert {r["prune"] for r in rows} == {0.0, 0.25, 0.5}
    measured = [r for r in rows if r.get("mean_kld") is not None]
    assert 2 <= len(measured) <= 4
    ok = [r for r in measured if (r.get("tool_retention") or 100) >= 90]
    assert rep["winner"]["mean_kld"] == min(r["mean_kld"] for r in ok)
    assert all(r["prune"] != 0.25 for r in rows if r.get("winner"))            # its tool calls dropped
    assert os.path.exists(out) and rep["why"].startswith(f"{int(round(100 * rep['chosen_prune']))}% of routed")
    # a 1-bit plan that was measured has its no-1-bit twin measured beside it
    for r in measured:
        if r["allow_1bit"] and r.get("n_1bit"):
            assert any(x["prune"] == r["prune"] and not x["allow_1bit"] and x.get("mean_kld") is not None
                       for x in measured)
    tab = M.auto_table(rep)
    assert "**" in tab and tab.count("\n") >= len([r for r in rows if r.get("feasible")])


def test_card_states_the_chosen_prune_and_the_candidates():
    import pollard_card as C
    b = {"name": "M-Pollard-MOE.gguf", "moefit": {"chosen_prune": 0.25, "keep": 24, "experts": 32, "scores": "reap",
                                                  "budget_gb": 0.36, "why": "25% of routed experts pruned ...",
                                                  "table": "| prune |\n|---|\n| 25% |"}}
    out = "\n".join(C.moe_allocation_section([b, {"name": "x.gguf"}]))
    assert "25% of the routed experts pruned" in out and "keep 24/32" in out and "| 25% |" in out
    assert "REAP" in out and C.moe_allocation_section([{"name": "x"}]) == []


def test_release_ladder_adds_the_moe_rung_for_a_moe(monkeypatch, tmp_path):
    import pollard_release as R
    f = tmp_path / "m-f16.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 4096)
    monkeypatch.setattr(R, "_is_moe_gguf", lambda p: True)
    labels = dict(R.gguf_ladder_labels({"f16": str(f)}))
    assert "MOE" in labels and labels["MOE"] == pytest.approx(labels["IQ2"])
    monkeypatch.setattr(R, "_is_moe_gguf", lambda p: False)
    assert "MOE" not in dict(R.gguf_ladder_labels({"f16": str(f)}))
