"""pollard-exlr8: the EXLR8 lane -- codec (checked against GR8's MIT fixture), exact knapsack, store -> rung assembly,
the reference decode, the encoder round-trip (torch), and the release / verify / card / tool-gate wiring.

Everything except the encoder runs on numpy alone, so the core and nix CI shapes exercise the format and the
allocation too. The GR8_FIXTURE bytes below are two matrices from kindlingai/gr8's
engine/loader-rs/tests/data/exlr8-fixture.safetensors (MIT, "gr8 engine contributors"), with sample values of their
decode by kindlingai/exlr8's loader/exlr8.py (MIT, Turboderp's exllamav3 notice) -- an independent reference for
the decoder written here from the format description."""
from __future__ import annotations

import base64
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_exlr8 as X  # noqa: E402

GR8_FIXTURE = {
    "t_in_k2": {
        "record": [1, 128, 128, 2, 2, 128, 0, 1],
        "trellis": (
            "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"
        ),
        "suh": (
            "yjSRn0as7Cnyr8I0F7MxrEQxBLTxNHypNB3CLQA1NbVtpYM13Kjdq9muJjW1NgatIqr3tW2jvrd8OP0zRTcsuF2yMJorsrAyzy7K"
            "sSK1Izqso303GDggJzSu+DX3Nbex2C68NzyxwbHbLdav0jDoLY+y6DZoMou62KcusjSvB7A7r3SvYjS2sAqo4zKaNoCspaDtsDs0"
            "3BSNr/kptbJ6N2kzyqzZlRi4tTEIsRI0FLXjNBcqkLVvN68sQjJDMF4gO6sZMJQsNrT6nqirV6ybvKyx8h4qs1Mv1i0lsu+2TbUj"
            "O3M0LabAr4ep5qePr5ottK4luAWyCSNKLfwxKDcsJw=="
        ),
        "svh": (
            "8Z3vJAihtqDxJbWZMCszLMSovKHkrDgtQCntKpylO6EsqXQPDiE1rL4leC3SIkusoSiSHQKxOpbRLdkkMSS/qCeqxC2AmmSiYSVX"
            "LPopOB+FKuYeFCz4JTYkCS0Uosmq2Ku9FYcpDK6VJxYhyqUWMPcYayg/LWahqqfcKFIhOzA+n9elOas2pVUiYihNrCMaEqseph8p"
            "eanapVeokqb4pGGWzqwmqRyirqrIGUwgkqs/rjGsypBUp+QnAZolpi2seiwMrOaaByVlKLOo/CGPFgElkq9lLeqioyQqIgqtyiaa"
            "JcQmjqkHJEqmGyyLIyKsbanrqKijSZYKKsWm+iYOqw=="
        ),
        "shape": [1, 4096],
        "idx": [361, 10974, 13896, 2875, 1461, 15294, 8351, 13345, 9918, 669, 10339, 7877, 10986, 4408, 2864, 549, 1229, 14028, 11941, 10406, 9157, 6921, 14921, 10615, 270, 13293, 15872, 463, 2035, 135, 5030, 8611, 10603, 44, 4907, 13855, 6448, 1316, 9070, 11928, 6603, 14130, 87, 4536, 8890, 8864, 12517, 8233],
        "want": [-7.402589835692197e-05, 0.0007238021353259683, -0.0001522556849522516, -0.007448775693774223, -0.007797282189130783, 0.0013113799504935741, -0.006576981861144304, -4.2687963286880404e-05, 4.489364437176846e-05, 0.0007940248469822109, -0.0005096410168334842, -0.003871298860758543, -0.00033414646168239415, 0.001118981745094061, 0.0009475992992520332, 0.0035682846792042255, 0.0011888676090165973, -0.002852730918675661, -0.0007290446665138006, -0.03815790265798569, 0.0005063485004939139, -0.008388406597077847, -0.004360601305961609, 0.007724825292825699, 0.002366804052144289, -0.00011938397074118257, 0.020704980939626694, 0.0039781127125024796, -0.0006920921732671559, 0.0012255939655005932, -0.001229615998454392, -0.005047058220952749, 0.0003844084858428687, -0.0003852287773042917, 0.0002983316080644727, 0.002470981562510133, 0.002761527895927429, -0.011678537353873253, -0.043904948979616165, -5.0805756472982466e-05, 0.00011419672227930278, 0.015022506937384605, -0.00146073242649436, 0.00018810934852808714, 0.004875761456787586, -0.0014596199616789818, -0.009858880192041397, -0.0009980713948607445],
        "fro": 2.2223011903583765,
    },
    "t_out_k3": {
        "record": [1, 128, 128, 3, 1, 128, 0, 1],
        "trellis": (
            "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"
        ),
        "suh": (
            "eTd8MW66XTRcreO5PyeZJN6tUSqDLF2xeDaFMOw2RzOwMiMsIK8gIVgsqCk3sESnkzSZsVK1YTYYsZaxjrNzNQC3uTBsssMhsTgV"
            "O9M6MDFYLPAq5DSFLPawezFXsmK35SpYq/GyZyzQsSgxird/uAq1Ni+uMAa4urLLNKK2rjNBMPGxEDMfMiUw8qg0rCisiq+RLqq3"
            "zLH3L9c1U7bjKFa0PjEgKFSy8qvNMAs3bzX8tR8xpS4PsAMyQrKdNbYsBrF6pJ+x66hFuEYwgjgbOJMt/jUuLYo2JbI4roC457FC"
            "MyY4VC/RsfE2ODGtqyAsiLJapyuvgrB+uE+xTTN2tA=="
        ),
        "svh": (
            "66w5qf4dJyj6JuwelKa6rSWrm6wUpIuhHx/xqXWvMafHI3kpXihhKMMdjimaIPepzSXjJbAjCSUyKSUqH7DHp9OqFislqd8nCaHG"
            "LEunIpogpB4i5ahCK3aduZWUIxWsGaogqcEoAiHeKyYsHiNNG24rXK+JKzuxYi36LAAl3CjaLp0bRC6PqxmpJq1kLPgiCKxUpP8c"
            "2SUXL3ahoqmfoZuhYBgqJWWk96mNqw+xMqoqrvEqMa35JeYoKyXvqVWv5qHELOmsUiworzOrWayuLhWoGKcAKu+lVyoPoKKoBy1f"
            "CgosjCM1JVasL6kpKW4orynunnUnT6oHLuIphBkPqg=="
        ),
        "shape": [1, 6144],
        "idx": [2457, 12416, 13606, 6162, 795, 14715, 6619, 5172, 12863, 4889, 14327, 10426, 1485, 15490, 15428, 14579, 9380, 3267, 10306, 10200, 851, 12940, 793, 7365, 6550, 16309, 3248, 10975, 1296, 3773, 9327, 5495, 10181, 9499, 12517, 14768, 957, 12548, 11002, 14065, 1372, 8355, 3912, 5980, 11537, 13025, 14638, 1728],
        "want": [0.019221993163228035, -0.016543284058570862, 0.00013239683175925165, -0.009327556006610394, 0.0127037912607193, -0.004813577514141798, 0.0002875094651244581, -0.003086305921897292, 0.03378145769238472, -0.001737889600917697, -0.011978203430771828, 0.00036892370553687215, 0.003424274967983365, 0.002812993247061968, 0.00037005136255174875, 0.014320505782961845, 0.013985145837068558, 0.006563304923474789, -0.006058403290808201, 0.00487891212105751, -4.846107913181186e-05, 0.03221750631928444, 0.0028831688687205315, 0.003133045043796301, 0.0009377247770316899, 0.004191538318991661, 0.0007465830422006547, 0.0028716297820210457, -0.004050452262163162, -0.009578946977853775, -0.00031750419293530285, -0.005230151116847992, 0.0005690459511242807, -0.00028627191204577684, -0.01861385442316532, 0.0013968105195090175, 0.00614761421456933, 0.015073558315634727, -0.01062280684709549, -0.008963169530034065, 0.007591589353978634, -9.547460649628192e-06, 0.0061382511630654335, -0.0026005024556070566, -0.00560326362028718, -0.00033548628562130034, 0.0011246517533436418, -5.7714070862857625e-05],
        "fro": 2.574150668116166,
    },
}


def _fixture(name):
    d = GR8_FIXTURE[name]
    raw = lambda k: base64.b64decode("".join(d[k]))
    t = {"trellis": np.frombuffer(raw("trellis"), np.uint8).reshape(d["shape"]),
         "suh": np.frombuffer(raw("suh"), np.float16), "svh": np.frombuffer(raw("svh"), np.float16)}
    return np.array(d["record"], np.int64), t, d


# ------------------------------------------------------------------------------------------------ codec


@pytest.mark.parametrize("name", sorted(GR8_FIXTURE))
def test_decoder_matches_the_reference_loader_on_gr8s_fixture(name):
    rec, t, d = _fixture(name)
    got = X.decode_record(rec, t)
    assert got.shape == (rec[1], rec[2])
    scale = float(np.abs(got).max())                  # fp32 matmul order: ulps of the largest term
    np.testing.assert_allclose(got.reshape(-1)[d["idx"]], d["want"], rtol=0, atol=2e-6 * scale)
    assert abs(float(np.linalg.norm(got.astype(np.float64))) - d["fro"]) < 1e-5 * d["fro"]


def test_codebook_is_the_mcg_rounded_to_e4m3():
    cb = X.codebook_table()
    assert cb.shape == (65536,) and cb.dtype == np.float32
    assert np.array_equal(X.e4m3_round(cb), cb), "every codebook value is an E4M3 value"
    assert len(np.unique(cb)) < 256, "E4M3 leaves few distinct values -- the K>=6 error floor"
    assert 1.1 < X.codebook_rms() < 1.4


def test_e4m3_round_is_round_half_even_and_saturates():
    assert X.e4m3_round([1.0625])[0] == 1.0          # halfway between 1.0 and 1.125 -> even mantissa
    assert X.e4m3_round([1.1875])[0] == 1.25         # halfway between 1.125 and 1.25 -> even
    assert X.e4m3_round([1e6, -1e6]).tolist() == [448.0, -448.0]
    assert X.e4m3_round([2.0 ** -9])[0] == 2.0 ** -9  # smallest subnormal
    codes = np.arange(256, dtype=np.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]             # no NaN codes
    vals = X.e4m3_decode(codes)
    back = X.e4m3_encode(vals)
    assert np.array_equal(X.e4m3_decode(back), vals)


@pytest.mark.parametrize("K", range(1, 9))
def test_pack_unpack_round_trip(K):
    rng = np.random.default_rng(K)
    vals = rng.integers(0, 1 << K, size=(5, 256))
    raw = X.pack_values(vals, K)
    assert raw.shape == (5, 32 * K) and raw.dtype == np.uint8
    assert np.array_equal(X.unpack_values(raw, K), vals)


def test_states_are_tail_biting_shift_registers():
    rng = np.random.default_rng(1)
    for K in (2, 3, 5):
        vals = rng.integers(0, 1 << K, size=(3, 256))
        s = X.states_from_values(vals, K)
        prev = np.roll(s, 1, axis=1)                  # state 255 precedes state 0
        assert np.array_equal(s, ((prev << K) | vals) & 0xFFFF)


@pytest.mark.parametrize("side", [X.SIDE_OUT, X.SIDE_IN])
def test_tile_grid_and_entry_order_are_inverse(side):
    rows, cols = 256, 384
    grid = np.arange((rows // 16) * (cols // 16) * 3).reshape(rows // 16, cols // 16, 3)
    ent = X.grid_to_entries(grid, side)
    assert ent.shape[0] == (rows if side == X.SIDE_IN else cols) // 128
    assert np.array_equal(X.entries_to_grid(ent, side, rows, cols), grid)
    T = np.random.default_rng(0).standard_normal((rows, cols)).astype(np.float32)
    assert np.array_equal(X.grid_to_matrix(X.matrix_to_grid(T)), T)


def test_tiles_to_stream_inverts_the_decoder_placement():
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 256, size=(4, 32 * 3), dtype=np.uint8)
    tiles = X.decode_tiles(raw, 3)
    st = X.states_from_values(X.unpack_values(raw, 3), 3)
    assert np.array_equal(X.tiles_to_stream(tiles), X.codebook_table()[st])


def test_xxh64_pure_python_matches_the_published_vectors():
    for data, want in ((b"", 0xEF46DB3751D8E999), (b"a", 0xD24EC4F1A98C6E5B), (b"abc", 0x44BC2CF5AD770999),
                       (b"Nobody inspects the spammish repetition", 0xFBCEA83C8A378BF1)):
        assert X._xxh64_py(data) == want
    blob = np.random.default_rng(3).integers(0, 256, 1000, dtype=np.uint8).tobytes()
    try:
        import xxhash
    except ImportError:
        return
    assert X._xxh64_py(blob) == xxhash.xxh64(blob, seed=0).intdigest()


def test_safetensors_writer_keeps_exlr8_alignment(tmp_path):
    p = str(tmp_path / "a.safetensors")
    a = np.arange(10, dtype=np.int64)
    b = np.ones((3, 4), np.float16)
    X.write_safetensors(p, {"a": a, "b": b, "c": (np.array([1, 2], np.uint16), "BF16")}, {"format": "exlr8-v2"})
    f = X.SafeFile(p)
    assert f.start == X.ALIGN and f.meta["format"] == "exlr8-v2"
    for n in ("a", "b", "c"):
        assert f.header[n]["data_offsets"][0] % X.ALIGN == 0
    assert any(k.startswith("__pad.") for k in f.header)
    assert np.array_equal(f.get("a"), a) and np.array_equal(f.get("b"), b) and f.dtype("c") == "BF16"
    try:
        from safetensors.numpy import load_file
    except ImportError:
        return
    q = str(tmp_path / "b.safetensors")                  # (stock numpy has no bfloat16: a file without one)
    X.write_safetensors(q, {"a": a, "b": b}, {"format": "exlr8-v2"})
    assert np.array_equal(load_file(q)["a"], a)          # a stock reader accepts the padded layout


def test_fp8_row_and_bf16_records_round_trip_with_checksums():
    W = np.random.default_rng(4).standard_normal((256, 64)).astype(np.float32) * 0.02
    r = X.fp8_row_record(W)
    got = X.decode_record(r["record"], r)
    assert np.abs(got - W).max() <= np.abs(W).max() * 0.07
    assert r["checksum"].shape == (2,)                    # one per 128 rows
    b = X.bf16_record(W)
    assert (np.abs(X.decode_record(b["record"], b) - W) <= np.abs(W) * 2 ** -8).all()
    flipped = dict(r, fp8=r["fp8"].copy())
    flipped["fp8"][130, 3] ^= 1
    assert X.record_checksums(r["record"], flipped)[1] != r["checksum"][1]
    assert X.record_checksums(r["record"], flipped)[0] == r["checksum"][0]


def test_trellis_checksum_covers_entry_and_both_vectors():
    rec, t, _ = _fixture("t_out_k3")
    base = X.record_checksums(rec, t)
    t2 = dict(t, svh=t["svh"].copy())
    t2["svh"][0] = np.float16(1.0)
    assert X.record_checksums(rec, t2)[0] != base[0]


# ------------------------------------------------------------------------------------------------ knapsack


def _brute(items, budget):
    best = None
    for combo in itertools.product(*[range(len(it)) for it in items]):
        size = sum(items[i][j][0] for i, j in enumerate(combo))
        cost = sum(items[i][j][1] for i, j in enumerate(combo))
        if size <= budget and (best is None or cost < best[0] - 1e-12):
            best = (cost, [items[i][j][2] for i, j in enumerate(combo)])
    return best


def test_knapsack_is_exact_against_brute_force():
    rng = np.random.default_rng(5)
    for _ in range(25):
        n = int(rng.integers(2, 6))
        items = []
        for _i in range(n):
            u = int(rng.integers(1, 5)) * 2048
            items.append([(u * K // 2, float(rng.random()) * 4.0 ** -K, K) for K in (2, 3, 4, 5)])
        budget = int(sum(it[0][0] for it in items) * rng.uniform(1.0, 2.6))
        res = X.knapsack(items, budget)
        bc, bl = _brute(items, budget)
        assert res["exact"] and res["size"] <= budget
        assert abs(res["cost"] - bc) < 1e-9, (res, bc, bl)


def test_knapsack_coarsening_never_overspends_and_says_so():
    items = [[(10_000 + i * 7, 1.0, 2), (20_000 + i * 7, 0.1, 4)] for i in range(50)]
    budget = sum(it[0][0] for it in items) + 150_000
    res = X.knapsack(items, budget, max_cells=200_000)
    assert not res["exact"] and res["granularity"] > 1 and res["size"] <= budget


def test_knapsack_refuses_an_impossible_budget():
    with pytest.raises(ValueError):
        X.knapsack([[(4096, 1.0, 2)], [(4096, 1.0, 2)]], 4095)


# ------------------------------------------------------------------------------------------------ synthetic store


def make_store(root, widths=(2, 3, 4), layers=2, sens_err=None, align=64):
    """A complete store on disk without torch: random trellis codes (any bytes decode), real records/checksums."""
    rng = np.random.default_rng(6)
    store = os.path.join(root, "store")
    for K in widths:
        os.makedirs(os.path.join(store, f"k{K}"), exist_ok=True)
    os.makedirs(os.path.join(store, "base", "dense"), exist_ok=True)
    meta = {"format": X.STORE_FORMAT, "version": 1, "source": "/src/Tiny", "widths": list(widths),
            "tensor_prefix": "model.layers.", "num_layers": layers, "probe": "rtn4", "tensors": {},
            "layers_done": list(range(layers)), "complete": True, "align": align, "architecture": "Qwen2ForCausalLM"}
    md = {"format": "exlr8-v2", "tensor_prefix": "model.layers."}
    shapes = {"self_attn.q_proj": (128, 128), "mlp.down_proj": (128, 256)}
    for L in range(layers):
        per_k = {K: {} for K in widths}
        for stored, (out_f, in_f) in shapes.items():
            side = X.side_of(stored)
            e0 = (sens_err or {}).get((L, stored), 1.0)
            entry = {"layer": L, "stored": stored, "shape": [out_f, in_f], "numel": out_f * in_f, "side": side,
                     "err_rtn": 0.01, "err": {}, "bytes": {}}
            for K in widths:
                grid = rng.integers(0, 256, size=(in_f // 16, out_f // 16, 32 * K), dtype=np.uint8)
                enc = {"grid": grid, "suh": rng.standard_normal(in_f).astype(np.float16),
                       "svh": (rng.standard_normal(out_f) * 0.01).astype(np.float16), "solve": 1}
                rec = X.trellis_record(enc, out_f, in_f, K, side)
                per_k[K].update(X._record_tensors(stored, rec))
                entry["err"][str(K)] = e0 * 0.07 * 4.0 ** -(K - 2)
                entry["bytes"][str(K)] = int(rec["trellis"].nbytes)
            meta["tensors"][f"model.layers.{L}.{stored}.weight"] = entry
        for K in widths:
            X.write_safetensors(os.path.join(store, f"k{K}", f"exlr8-dense-{L:03d}.safetensors"), per_k[K],
                                {**md, "layer": str(L), "k": str(K)}, align)
        X.write_safetensors(os.path.join(store, "base", "dense", f"exlr8-dense-{L:03d}.safetensors"),
                            {"input_layernorm.weight": np.ones(128, np.float32)}, {**md, "layer": str(L)}, align)
    vocab = {**X._record_tensors("embed_tokens", X.fp8_row_record(rng.standard_normal((64, 128)) * 0.02)),
             "norm.weight": np.ones(128, np.float32)}
    X.write_safetensors(os.path.join(store, "base", "dense", "exlr8-vocab.safetensors"), vocab,
                        {**md, "hf_names": {"embed_tokens": "model.embed_tokens", "norm.weight": "model.norm.weight"}},
                        align)
    with open(os.path.join(store, "base", "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "num_hidden_layers": layers}, f)
    X._write_store(store, meta)
    return store


def test_plan_rung_puts_bits_where_the_measured_sensitivity_is(tmp_path):
    store = make_store(str(tmp_path))
    meta = X.read_store(store)
    sens = {"probe": "rtn4", "attn": {"0": 0.001, "1": 0.001}, "ffn": {"0": 0.001, "1": 0.5}}
    p = X.plan_rung(meta, 3.0, sens)
    w = p["widths"]
    assert w["model.layers.1.mlp.down_proj.weight"] > w["model.layers.0.mlp.down_proj.weight"]
    assert p["bpw"] <= 3.0 + 1e-9 and p["exact"]
    kls = [X.plan_rung(meta, b, sens)["predicted_kld"] for b in (2.0, 2.5, 3.0, 3.5, 4.0)]
    assert kls == sorted(kls, reverse=True), "more bits never predicts more KL"
    assert set(X.plan_rung(meta, 4.0, sens)["widths"].values()) == {4}
    with pytest.raises(ValueError):
        X.plan_rung(meta, 5.0, sens)                       # above the widest stored width


def test_no_profile_minimises_summed_relative_error(tmp_path):
    store = make_store(str(tmp_path), sens_err={(0, "self_attn.q_proj"): 50.0})
    p = X.plan_rung(X.read_store(store), 2.5)
    assert p["widths"]["model.layers.0.self_attn.q_proj.weight"] >= 3


def test_tensor_sensitivity_splits_group_kl_by_size():
    tensors = {"model.layers.0.self_attn.q_proj.weight": {"numel": 100},
               "model.layers.0.self_attn.k_proj.weight": {"numel": 300},
               "model.layers.0.mlp.up_proj.weight": {"numel": 50}}
    S = X.tensor_sensitivity(tensors, {"attn": {"0": 0.4}, "ffn": {"0": 0.2}})
    assert S["model.layers.0.self_attn.q_proj.weight"] == pytest.approx(0.1)
    assert S["model.layers.0.self_attn.k_proj.weight"] == pytest.approx(0.3)
    assert S["model.layers.0.mlp.up_proj.weight"] == pytest.approx(0.2)


def test_build_rung_copies_the_chosen_widths_and_decodes(tmp_path):
    store = make_store(str(tmp_path))
    meta = X.read_store(store)
    sens = {"attn": {"0": 0.01, "1": 0.01}, "ffn": {"0": 0.01, "1": 0.9}}
    plan = X.plan_rung(meta, 3.0, sens)
    out = str(tmp_path / "Tiny-Pollard-EXLR8-3.0bpw")
    man = X.build_rung(store, plan, out, sens_path=None)
    assert X.is_rung(out) and not os.path.exists(out + ".partial")
    for fn in ("dense/exlr8-dense-000.safetensors", "dense/exlr8-dense-001.safetensors",
               "dense/exlr8-vocab.safetensors", "manifests/exlr8-manifest-3.0bpw.json", "config.json"):
        assert os.path.isfile(os.path.join(out, fn)), fn
    assert man["format"] == X.MANIFEST_FORMAT and man["bpw"] == plan["bpw"]
    for L in (0, 1):
        f = X.SafeFile(os.path.join(out, "dense", f"exlr8-dense-{L:03d}.safetensors"))
        for stored in ("self_attn.q_proj", "mlp.down_proj"):
            rec = f.get(stored + ".record")
            assert int(rec[3]) == plan["widths"][f"model.layers.{L}.{stored}.weight"]
            src = X.SafeFile(os.path.join(store, f"k{int(rec[3])}", f"exlr8-dense-{L:03d}.safetensors"))
            assert np.array_equal(f.get(stored + ".trellis"), src.get(stored + ".trellis")), "copied, not re-encoded"
            t = {s: f.get(f"{stored}.{s}") for s in X.DATA[X.TRELLIS]}
            assert np.array_equal(X.record_checksums(rec, t).astype(np.uint64), f.get(stored + ".checksum"))
        assert f.header[f.names()[0]]["data_offsets"][0] % 64 == 0
    dec = dict(X.iter_rung_tensors(out))
    assert dec["model.layers.1.mlp.down_proj.weight"].shape == (128, 256)
    assert dec["model.layers.0.self_attn.q_proj.weight"].shape == (128, 128)
    assert dec["model.embed_tokens.weight"].shape == (64, 128) and "model.norm.weight" in dec
    assert "model.layers.0.input_layernorm.weight" in dec
    assert X.decoded_bytes(out) == 2 * sum(v.size for v in dec.values())
    info = X.inspect(out)
    assert info["kind"] == "rung" and info["bpw"] == plan["bpw"]
    assert X.inspect(store)["kind"] == "store"


def test_materialize_writes_a_plain_bf16_checkpoint(tmp_path):
    store = make_store(str(tmp_path))
    out = str(tmp_path / "r")
    X.build_rung(store, X.plan_rung(X.read_store(store), 3.0), out)
    dec = X.materialize(out, str(tmp_path / "bf16"))
    f = X.SafeFile(os.path.join(dec, "model.safetensors"))
    assert f.dtype("model.layers.0.self_attn.q_proj.weight") == "BF16"
    assert os.path.isfile(os.path.join(dec, "config.json")) and not os.path.isfile(os.path.join(dec, X.MARKER))
    want = dict(X.iter_rung_tensors(out))["model.layers.1.mlp.down_proj.weight"]
    got = X.bf16_bits_to_f32(f.get("model.layers.1.mlp.down_proj.weight"))
    assert np.abs(got - want).max() <= np.abs(want).max() * 2 ** -8


def test_cli_plan_only_reads_a_complete_store(tmp_path, capsys):
    store = make_store(str(tmp_path))
    rc = X.main(["--model", "/src/Tiny", "--store", store, "--widths", "2,3,4", "--ladder", "2.5,3,4",
                 "--plan-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "complete (reused)" in out and "2.5bpw" in out and "predicted KL" in out


def test_cli_ladder_from_a_complete_store_builds_every_rung_without_encoding(tmp_path, monkeypatch, capsys):
    store = make_store(str(tmp_path))
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(X, "encode_model", lambda *a, **k: pytest.fail("the store is complete: no re-encode"))
    rc = X.main(["--model", "/src/Tiny", "--store", store, "--widths", "2,3,4", "--ladder", "2.5,3,4",
                 "--out", str(tmp_path / "rungs")])
    assert rc == 0
    names = sorted(os.listdir(tmp_path / "rungs"))
    assert names == ["Tiny-Pollard-EXLR8-2.5bpw", "Tiny-Pollard-EXLR8-3.0bpw", "Tiny-Pollard-EXLR8-4.0bpw"]


def test_bpw_tags():
    assert [X.bpw_tag(b) for b in (2.5, 3, 3.25, 4.0, 6)] == ["2.5bpw", "3.0bpw", "3.25bpw", "4.0bpw", "6.0bpw"]


def test_side_follows_exlr8s_row_parallel_convention():
    assert X.side_of("self_attn.o_proj") == X.SIDE_IN and X.side_of("mlp.down_proj") == X.SIDE_IN
    assert X.side_of("self_attn.q_proj") == X.SIDE_OUT and X.side_of("mlp.gate_proj") == X.SIDE_OUT


# ------------------------------------------------------------------------------------------------ encoder (torch)


def test_viterbi_paths_are_valid_tail_biting_trellis_paths():
    torch = pytest.importorskip("torch")
    cb = torch.tensor(X.codebook_table())
    x = torch.randn(3, 256)
    for K in (2, 4):
        st = X.viterbi_tailbiting(x, K, cb)
        s = st.numpy()
        vals = s & ((1 << K) - 1)
        assert np.array_equal(X.states_from_values(vals, K), s), "decoder rebuilds exactly these states"
        err = float(((cb[st] - x) ** 2).mean())
        assert err < {2: 0.12, 4: 0.012}[K]


def test_encode_matrix_round_trips_through_the_stored_record():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    Wg = torch.randn(128, 128) * 0.02
    Xc = torch.randn(512, 128) * torch.linspace(0.2, 3.0, 128)
    H = Xc.T @ Xc / 512
    errs = {}
    for K, side in ((2, X.SIDE_OUT), (4, X.SIDE_IN)):
        enc = X.encode_matrix(Wg, K, H, "t")
        rec = X.trellis_record(enc, 128, 128, K, side)
        dec = X.decode_record(rec["record"], rec)
        assert np.abs(dec - enc["W_hat"].numpy()).max() <= 1e-6 * np.abs(dec).max(), "encoder == decoder"
        assert int(rec["record"][7]) == 1 and rec["trellis"].shape == (1, 128 * 128 * K // 8)
        errs[K] = X.rel_err(Wg, torch.tensor(dec), H)
    assert errs[4] < errs[2] / 6 and errs[2] < 0.08


# ------------------------------------------------------------------------------------------------ wiring

import pollard_card as C  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_toolcall as T  # noqa: E402
import pollard_verify as V  # noqa: E402


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


def test_release_plans_one_encode_then_a_manifest_per_rung(model):
    cfg = R.load_plan(None)
    assert cfg["lanes"]["exlr8"]["rungs"] == [2.5, 3.0, 3.25, 4.0, 5.0, 6.0]
    c, lanes, steps = R.plan_steps(cfg, model, {"exlr8"})
    got = [s.id for s in steps]
    assert lanes == ["exlr8"]
    assert got.index("prep/smooth") < got.index("prep/probe-smoothed") < got.index("exlr8/encode")
    tags = ["2.5bpw", "3.0bpw", "3.25bpw", "4.0bpw", "5.0bpw", "6.0bpw"]
    assert got.index("exlr8/encode") < got.index("exlr8/2.5bpw/build")
    for t in tags:
        assert got.index(f"exlr8/{t}/build") < got.index(f"exlr8/{t}/verify")
    assert got.index("exlr8/6.0bpw/verify") < got.index("exlr8/card") < got.index("exlr8/upload")
    by = {s.id: s for s in steps}
    enc = by["exlr8/encode"].cmd
    assert enc[0] == "pollard-exlr8" and "--encode-only" in enc and c["smooth_dir"] in enc
    assert enc[enc.index("--calib") + 1] == c["calib"] and enc[enc.index("--widths") + 1] == "2,3,4,5,6"
    b = by["exlr8/3.25bpw/build"].cmd
    assert b[b.index("--bpw") + 1] == "3.25" and b[b.index("--store") + 1] == c["exlr8_store"]
    assert b[b.index("--sensitivity") + 1] == c["sens_smooth"], "the measured allocation"
    assert by["exlr8/3.25bpw/build"].out.endswith("Tiny-1B-Pollard-EXLR8-3.25bpw")
    v = by["exlr8/3.25bpw/verify"].cmd
    assert v[v.index("--lane") + 1] == "exlr8" and v[v.index("--source") + 1] == c["hf_dir"] and "--eval" in v
    assert R.repo_name("PollardWeights", "Qwen/Qwen3-8B", "exlr8") == "PollardWeights/Qwen3-8B-Pollard"
    assert R.lane_folder("exlr8") == "EXLR8"
    assert R.REQUIRES["exlr8"] == "cuda" and R.rung_tag("exlr8", 3.25) == "3.25bpw"


def test_release_toolcall_step_for_exlr8(model):
    cfg = R.load_plan(None)
    cfg["measure"]["toolcall"] = True
    c, _, steps = R.plan_steps(cfg, model, {"exlr8"})
    tc = {s.id: s for s in steps}["exlr8/4.0bpw/toolcall"].cmd
    assert tc[tc.index("--backend") + 1] == "hf" and tc[tc.index("--ref-backend") + 1] == "hf", "one in-process path"
    assert tc[tc.index("--ref") + 1] == c["hf_dir"] and "--ref-fallback" in tc


def test_release_presets_keep_exlr8_off_where_it_cannot_serve():
    for p in ("vlm", "decision", "moe"):
        assert R.load_plan(None, p)["lanes"]["exlr8"]["enabled"] is False, p
    assert R.load_plan(None)["lanes"]["exlr8"]["enabled"] is True


def test_release_skips_exlr8_without_cuda(model):
    cfg = R.load_plan(None)
    c, _, steps = R.plan_steps(cfg, model, {"exlr8"})
    lines = []
    st = R.execute(steps, c, cfg, run=False, have={"any": True, "apple": True, "cuda": False}, out=lines.append)
    assert st["exlr8/encode"].startswith("skipped") and st["exlr8/card"].startswith("skipped")


def test_card_names_the_runtime_honestly(tmp_path):
    builds = [{"name": "Tiny-Pollard-EXLR8-3.0bpw", "bytes": 100, "lane": "exlr8"},
              {"name": "Tiny-Pollard-EXLR8-4.0bpw", "bytes": 200, "lane": "exlr8"}]
    other = "\n".join(C.exlr8_section(builds, "O/Tiny-Pollard-EXLR8", {"architectures": ["Qwen2ForCausalLM"]}))
    assert "pollard-exlr8 --decode ./3.0bpw" in other and "GLM-5.3" in other and "gr8 glm53 up" not in other
    glm_cfg = {"architectures": ["GlmMoeDsaForCausalLM"]}
    glm = "\n".join(C.exlr8_section(builds, "O/GLM-Pollard-EXLR8", glm_cfg))
    assert "./gr8 glm53 up" not in glm and "per-unit expert" in glm, "dense-record rungs: no GR8 claim"
    for b in builds:
        b["path"] = str(tmp_path / b["name"])
        os.makedirs(os.path.join(b["path"], "experts"))
    assert "./gr8 glm53 up" in "\n".join(C.exlr8_section(builds, "O/GLM-Pollard-EXLR8", glm_cfg))
    assert C.rung_subfolder({"name": "M-Pollard-EXLR8-3.25bpw"}, "exlr8") == "3.25bpw"
    assert "pollard-exlr8 --decode" in "\n".join(C.rungs_section(builds, "exlr8", "O/M"))
    lines = C.tool_serve_lines({"vllm_parser": "hermes", "vllm_flags": "--tool-call-parser hermes"}, ["exlr8"],
                               "O/M", None)
    assert any("EXLR8" in l and "--tool-call-parser hermes" in l for l in lines)
    assert "exlr8" in C.LANE_TAGS and "hf" in C.TOOL_SERVERS


def _rung_dir(tmp_path):
    rung = tmp_path / "M-Pollard-EXLR8-3.0bpw"
    rung.mkdir()
    (rung / X.MARKER).write_text("{}")
    (rung / "config.json").write_text("{}")
    return str(rung)


def test_toolcall_runs_an_exlr8_rung_on_the_shared_inprocess_hf_path(tmp_path, monkeypatch):
    rung = _rung_dir(tmp_path)
    assert T.lane_of(rung) == "exlr8" and T.pick_backend(rung) == "hf"
    assert "exlr8" not in T.BACKENDS, "no backend of its own: the one hf engine, reader and grader"
    kind = {"tools": {"family": "hermes"}}
    monkeypatch.setattr(T, "_has_module", lambda n: True)
    monkeypatch.setattr(T, "exlr8_decoded_gb", lambda r: 1.0)
    monkeypatch.setattr(T, "gpu_free_gb", lambda text=None: 16.0)
    monkeypatch.setattr(T, "gpu_users", lambda *a, **k: [])
    assert T.preflight("hf", rung, kind=kind) == {"device": "cuda"}
    # a game on the GPU: never taken -- the rung runs on the CPU / unified memory instead
    monkeypatch.setattr(T, "gpu_users", lambda *a, **k: ["RobloxPlayerBeta.exe"])
    monkeypatch.setattr(T, "check_ram", lambda need, avail=None: None)
    assert T.preflight("hf", rung, kind=kind)["device"] in ("cpu", "mps")
    # the memory check sizes the DECODED weights, not the packed files
    seen = {}
    monkeypatch.setattr(T, "check_ram", lambda need, avail=None: seen.setdefault("need", need))
    monkeypatch.setattr(T, "exlr8_decoded_gb", lambda r: 7.5)
    T.preflight("hf", rung, kind=kind)
    assert seen["need"] == 7.5


def test_hf_engine_loads_an_exlr8_rung_through_the_reference_decode(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    rung = _rung_dir(tmp_path)

    class Tok:
        eos_token_id = 2

        def decode(self, ids, skip_special_tokens=False):
            return "</s>"

    class Model:
        generation_config = None

        def eval(self):
            return self

        def to(self, device):
            return self
    loaded = {}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("an EXLR8 rung is not an HF checkpoint"))
    monkeypatch.setattr(X, "load_model", lambda path, device, **k: loaded.update(path=path, device=device) or Model())
    monkeypatch.setattr(T, "_Renderer", lambda model, tok: None)
    eng = T.HFEngine(rung, "cpu")
    assert loaded == {"path": rung, "device": "cpu"} and isinstance(eng.model, Model)


def test_exlr8_decoded_size_comes_from_the_headers(tmp_path):
    store = make_store(str(tmp_path))
    out = str(tmp_path / "r")
    X.build_rung(store, X.plan_rung(X.read_store(store), 3.0), out)
    assert T.exlr8_decoded_gb(out) == pytest.approx(X.decoded_bytes(out) / 1e9)


def test_verify_gates_exlr8_rungs_on_the_bit_width_thresholds():
    gates = V.load_gates(None)
    th = V.thresholds_for(gates, "Tiny-Pollard-EXLR8-3.25bpw", "exlr8")
    assert th["kld_max"] == 0.2 and th["top1_min"] == 85
    assert V.thresholds_for(gates, "Tiny-Pollard-EXLR8-2.5bpw", "exlr8")["kld_max"] == 0.5
    assert "exlr8" in V.__doc__
