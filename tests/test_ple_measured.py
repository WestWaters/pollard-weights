"""Gemma 4's per-layer token table (per_layer_token_embd) as a MEASURED class: pollard-probe scores it from the
imatrix + the text's token frequencies, pollard-fit allocates it like every other tensor (its bytes in the budget,
q8_0 down only where that is the cheapest KL per byte), and without a measurement it keeps the q8_0 pin."""
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_fit as F  # noqa: E402

TABLE = "per_layer_token_embd.weight"
PAT = r"per_layer_token_embd\.weight"


def _arch():
    h, L = 2560, 42
    ffn, attn, table = 3 * h * 10240, 4 * h * 2048, 262144 * 42 * 256
    return dict(kind="dense", layers=L, hidden=h, total=L * (ffn + attn) + h * 262144 + table,
                expert_params=0, n_experts=0, dense_ffn_params=ffn, attn_params=attn), table


def _disk(table, typ="auto"):
    return {"params": table, "type": typ, "gb": table * 8.5 / 8 / 1e9, "patterns": [PAT], "names": [TABLE]}


def _profile(table, cost_scale=1.0):
    L = 42
    noise = {"q6_K": 2.0, "q5_K": 4.0, "iq4_xs": 15.0, "iq3_s": 45.0, "iq2_s": 230.0, "iq2_xxs": 450.0}
    cost = {t: cost_scale * c for t, c in
            {"q8_0": 0.01, "q6_K": 0.1, "q5_K": 0.2, "iq4_xs": 0.8, "iq3_s": 2.5, "q2_K": 12.0}.items()}
    return {"ffn": {str(i): 10.0 for i in range(L)}, "attn": {str(i): 5.0 for i in range(L)}, "noise": noise,
            "method": "imatrix-hessian", "tables": {TABLE: {"params": table, "cost": cost}}}


def _table_type(ov):
    return next(t for p, t in ov if p == PAT)


def test_unmeasured_table_keeps_the_q8_0_pin_outside_the_budget():
    arch, table = _arch()
    prof = _profile(table)
    prof.pop("tables")
    ov, _, gb, _, (summary, _) = F.plan_allocation(arch, 6, 0, prof, disk=_disk(table))
    assert _table_type(ov) == "q8_0" and "disk:" in summary
    assert gb <= 6 * 0.85 + 1e-6, "the pinned table is not charged"


def test_measured_table_is_allocated_and_charged_to_the_budget():
    arch, table = _arch()
    res = F.plan_allocation(arch, 6, 0, _profile(table), disk=_disk(table), want_pinned=True)
    ov, _, gb, _, (summary, src), explicit = res
    t = _table_type(ov)
    assert t in F.TABLE_LADDER and F.BPW[t] < F.BPW["q8_0"], "a tight budget walks the cheap table down"
    assert gb <= 6 * 0.85 + 1e-6 and gb > table * F.BPW[t] / 8 / 1e9, "its bytes are in the budget"
    assert "measured table" in summary and "measured lookup table" in src
    assert PAT in explicit, "no later pin (pin_uncovered) may promote it back"


def test_a_costly_table_is_protected_and_the_body_pays():
    arch, table = _arch()
    cheap = F.plan_allocation(arch, 6, 0, _profile(table, 1.0), disk=_disk(table))[0]
    dear = F.plan_allocation(arch, 6, 0, _profile(table, 1e4), disk=_disk(table))[0]
    assert F.BPW[_table_type(dear)] > F.BPW[_table_type(cheap)], "measured cost decides, not a rule"


def test_roomy_budget_leaves_the_table_at_q8_0():
    arch, table = _arch()
    ov = F.plan_allocation(arch, 64, 0, _profile(table), disk=_disk(table))[0]
    assert _table_type(ov) == "q8_0", "measuring never makes the table bigger than the old pin"


def test_disk_type_measured_requires_the_measurement():
    arch, table = _arch()
    prof = _profile(table)
    prof.pop("tables")
    with pytest.raises(SystemExit):
        F.plan_allocation(arch, 6, 0, prof, disk=_disk(table, "measured"))


def test_explicit_disk_type_still_pins():
    arch, table = _arch()
    ov, _, _, _, (summary, _) = F.plan_allocation(arch, 6, 0, _profile(table), disk=_disk(table, "q5_K"))
    assert _table_type(ov) == "q5_K" and "disk:" in summary


# ------------------------------------------------------------------ the probe

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")


def _write_model(path, layers=2, per=256, vocab=64, n_embd=256, seed=0):
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "gemma4")
    w.add_tensor("token_embd.weight", rng.standard_normal((vocab, n_embd)).astype(np.float32) * 0.02)
    w.add_tensor(TABLE, rng.standard_normal((vocab, layers * per)).astype(np.float32) * 0.05)
    w.add_tensor("per_layer_model_proj.weight", rng.standard_normal((layers * per, n_embd)).astype(np.float32) * 0.02)
    w.add_tensor("per_layer_proj_norm.weight", np.ones(per, dtype=np.float32))
    for i in range(layers):
        w.add_tensor(f"blk.{i}.proj.weight", rng.standard_normal((n_embd, per)).astype(np.float32) * 0.02)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _write_imatrix(path, layers=2, per=256, scale=1.0):
    rec = bytearray(struct.pack("<i", layers))
    for i in range(layers):
        n = f"blk.{i}.proj.weight".encode()
        rec += struct.pack("<i", len(n)) + n + struct.pack("<ii", 1, per)
        rec += (np.full(per, scale, dtype=np.float32)).tobytes()
    path.write_bytes(bytes(rec))


def test_table_cost_falls_with_bits_and_follows_the_tokens_read(tmp_path):
    import pollard_probe as P
    m, im = tmp_path / "m.gguf", tmp_path / "m.dat"
    _write_model(m)
    _write_imatrix(im)
    r = P._table_sensitivity(str(m), str(im), {3: 0.5, 7: 0.5})
    c = r["cost"]
    assert r["params"] == 64 * 512 and r["rows_read"] == 2 and r["layers"] == 2
    assert c["q8_0"] < c["q6_K"] < c["q5_K"] < c["iq4_xs"] < c["iq3_s"] < c["q2_K"]
    assert all(math.isfinite(v) and v > 0 for v in c.values())
    # the cost scales with how hard the consumer is driven (the imatrix) -- same units as every scored matmul
    _write_imatrix(im, scale=4.0)
    r4 = P._table_sensitivity(str(m), str(im), {3: 0.5, 7: 0.5})
    assert abs(r4["cost"]["iq3_s"] / c["iq3_s"] - 4.0) < 1e-3
    assert P._table_sensitivity(str(m), str(im), {3: 1.0})["rows_read"] == 1


def test_table_needs_its_consumer_in_the_imatrix(tmp_path):
    import pollard_probe as P
    m, im = tmp_path / "m.gguf", tmp_path / "m.dat"
    _write_model(m)
    _write_imatrix(im, layers=1)
    with pytest.raises(SystemExit, match="blk.1.proj.weight"):
        P._table_sensitivity(str(m), str(im), {3: 1.0})


def test_gemma4_per_layer_matmuls_are_scored_as_ffn():
    import pollard_probe as P
    assert P._gguf_slot("blk.5.inp_gate.weight", ["ffn", "attn"]) == ("ffn", 5)
    assert P._gguf_slot("blk.5.proj.weight", ["ffn", "attn"]) == ("ffn", 5)


def test_release_probe_gets_the_text_and_tokenizer(tmp_path, monkeypatch):
    import pollard_release as R
    hf = tmp_path / "E4B"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForConditionalGeneration"]}))
    (hf / "tokenizer.json").write_text("{}")
    f16 = tmp_path / "E4B-f16.gguf"
    f16.write_bytes(b"GGUF")
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    cfg = R.load_plan(None, "text")
    cfg["f16"] = str(f16)
    c = R._ctx(cfg, str(hf))
    c["imatrix"] = str(tmp_path / "x.imatrix")
    steps = R.prep_steps(cfg, c, ["gguf"])
    probe = next(s for s in steps if s.id == "prep/probe")
    assert probe.cmd and "--from-imatrix" in probe.cmd
    assert probe.cmd[probe.cmd.index("--text") + 1] == c["calib"]
    assert probe.cmd[probe.cmd.index("--tokenizer") + 1] == str(hf)
