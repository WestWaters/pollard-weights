"""Quantized-GGUF passthrough (pollard_passthrough + pollard-fit --passthrough): the planner only ever
copies or downgrades, the llama-quantize tensor map names every tensor exactly, and -- with a real
llama-quantize -- untouched tensors come out byte-identical while the selected ones are requantized
exactly as ggml's own quantize(dequant(source), imatrix) would. The build half skips without a
llama.cpp build (CI); the planning half needs only numpy + gguf."""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pollard_passthrough as pt  # noqa: E402
import ggml_ref  # noqa: E402

H, F, V = 256, 512, 64


def _t(name, typ, ne):
    n = int(np.prod(ne))
    return {"name": name, "type": typ, "ne": list(ne), "n_elements": n, "n_bytes": pt.nbytes(typ, n),
            "ndim": len(ne), "shard": "", "offset": 0}


def _bolt_like():
    """A Bolt-shaped tensor table: QAT IQ2_XXS experts, NVFP4 SSM, Q8_0 attention / embeddings."""
    ts = [_t("token_embd.weight", "q8_0", (H, V)), _t("output.weight", "q8_0", (H, V)),
          _t("output_norm.weight", "f32", (H,))]
    for i in range(4):
        ts += [_t(f"blk.{i}.ffn_gate_exps.weight", "iq2_xxs", (H, F, 8)),
               _t(f"blk.{i}.ffn_up_exps.weight", "iq2_xxs", (H, F, 8)),
               _t(f"blk.{i}.ffn_down_exps.weight", "iq2_xxs", (F, H, 8)),
               _t(f"blk.{i}.ffn_gate_inp.weight", "f32", (H, 8)),
               _t(f"blk.{i}.ffn_up_shexp.weight", "q8_0", (H, F)),
               _t(f"blk.{i}.ssm_in.weight", "nvfp4", (H, 2 * H)),
               _t(f"blk.{i}.attn_q.weight", "q8_0", (H, H)),
               _t(f"blk.{i}.attn_norm.weight", "f32", (H,))]
    return ts


# ------------------------------------------------------------------ planning (no binaries) ---------
def test_untouched_budget_changes_nothing():
    ts = _bolt_like()
    plan = pt.plan_passthrough(ts, sum(t["n_bytes"] for t in ts) + 1)
    assert plan["fits"] and not plan["changed"] and plan["bytes"] == plan["source_bytes"]


def test_only_bulk_moves_and_only_down():
    ts = _bolt_like()
    src = sum(t["n_bytes"] for t in ts)
    plan = pt.plan_passthrough(ts, int(src * 0.85))
    assert plan["fits"] and plan["changed"]
    for name, (a, b) in plan["changed"].items():
        assert pt.tensor_class(name) == "bulk", name                   # NVFP4 / Q8_0 / shexp untouched
        assert pt.bpw(b) < pt.bpw(a)
        assert b in ("iq1_m", "iq1_s")                                  # below IQ2_XXS there is only 1-bit


def test_sensitivity_decides_which_layers_drop():
    ts = _bolt_like()
    src = sum(t["n_bytes"] for t in ts)
    sens = {"ffn": {"0": 5.0, "1": 0.01, "2": 0.01, "3": 5.0}}
    plan = pt.plan_passthrough(ts, int(src * 0.94), sens)
    layers = {pt.layer_of(n) for n in plan["changed"]}
    assert layers and layers <= {1, 2}, layers


def test_no_imatrix_means_no_1bit_floor():
    ts = _bolt_like()
    src = sum(t["n_bytes"] for t in ts)
    plan = pt.plan_passthrough(ts, int(src * 0.85), imatrix_ok=False)
    assert not plan["fits"] and not plan["changed"]                     # nothing below iq2_xxs without one
    q8 = [_t("blk.0.ffn_up.weight", "q8_0", (H, F))]
    plan = pt.plan_passthrough(q8, q8[0]["n_bytes"] // 2, imatrix_ok=False)
    assert set(b for _, b in plan["changed"].values()) <= {"q6_K", "q5_K", "iq4_xs", "iq3_s", "q2_K"}


def test_uncovered_tensor_never_gets_an_imatrix_type():
    ts = [_t("blk.0.ffn_up.weight", "q8_0", (H, F)), _t("blk.1.ffn_up.weight", "q8_0", (H, F))]
    plan = pt.plan_passthrough(ts, 1, covered={"blk.0.ffn_up.weight"})
    assert pt.norm_type(plan["assign"]["blk.1.ffn_up.weight"]) not in pt.IMATRIX_REQUIRED
    assert plan["assign"]["blk.0.ffn_up.weight"] == "iq1_s"


def test_upgrades_are_refused():
    ts = _bolt_like()
    with pytest.raises(pt.PassthroughError, match="UPGRADE"):
        pt.plan_passthrough(ts, 10**12, pins={"bulk": "q4_K"})
    with pytest.raises(pt.PassthroughError, match="upgrade"):
        pt.check_no_upgrade(ts, {"blk.0.attn_q.weight": "f16"})
    with pytest.raises(pt.PassthroughError):
        pt.check_no_upgrade(ts, {"blk.0.ssm_in.weight": "iq4_xs"} | {"blk.0.attn_q.weight": "bf16"})
    pt.check_no_upgrade(ts, {"blk.0.ssm_in.weight": "iq4_xs"})          # 4.5 -> 4.25: a downgrade, allowed


def test_tensor_map_names_every_tensor_exactly():
    ts = _bolt_like() + [_t("blk.11.ffn_up_exps.weight", "iq2_xxs", (H, F, 8))]
    assign = {t["name"]: t["type"] for t in ts}
    assign["blk.1.ffn_up_exps.weight"] = "iq1_s"
    lines, flags, stats = pt.quantize_plan(ts, assign)
    pats = [ln.split("=")[0] for ln in lines]
    import re
    for t in ts:
        hits = [p for p in pats if re.search(p, t["name"])]
        if pt.quantizable(t):
            assert len(hits) == 1, (t["name"], hits)                    # first-match-wins can't bite
        else:
            assert not hits
    assert r"^blk\.1\.ffn_up_exps\.weight$=iq1_s" in lines
    assert r"^blk\.11\.ffn_up_exps\.weight$=iq2_xxs" in lines
    assert flags[0] == "--allow-requantize" and stats["requant"] == 1
    assert flags[flags.index("--token-embedding-type") + 1] == "q8_0"
    _, flags2, _ = pt.quantize_plan(ts, {t["name"]: t["type"] for t in ts})
    assert "--allow-requantize" not in flags2


def test_ladder_targets_span_source_to_floor():
    ts = _bolt_like()
    rungs = pt.ladder_targets(ts)
    src, flo = sum(t["n_bytes"] for t in ts), pt.floor_bytes(ts)
    assert [r[0] for r in rungs] == ["PT25", "PT50", "PT75", "PT100"]
    assert rungs[-1][1] == flo and all(flo <= b < src for _, b in rungs)
    assert pt.ladder_targets(ts, imatrix_ok=False) == []


# ------------------------------------------------------------------ a real build ------------------
def _write_imatrix(path, entries):
    """Legacy .dat imatrix (ncall 1, so llama-quantize uses the values as given)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, vals in entries.items():
            b = name.encode()
            f.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", 1, len(vals)))
            f.write(np.asarray(vals, dtype=np.float32).tobytes())
        f.write(struct.pack("<i", 1))
        ds = b"synthetic"
        f.write(struct.pack("<i", len(ds)) + ds)


@pytest.fixture(scope="module")
def ggml():
    g = ggml_ref.load()
    if g is None or ggml_ref.llama_bin("llama-quantize") is None:
        pytest.skip("no llama.cpp build (libggml-base + llama-quantize) -- set POLLARD_LLAMA_BIN_DIR")
    return g


def _synthetic(tmp, g):
    """Tiny llama GGUF with mixed types, quantized by the real ggml so every block is genuine."""
    rng = np.random.default_rng(0)
    imat = {}
    types = {"token_embd.weight": ("f16", (V, H)), "output_norm.weight": ("f32", (H,)),
             "output.weight": ("q8_0", (V, H))}
    for i in range(2):
        types |= {f"blk.{i}.attn_q.weight": ("q8_0", (H, H)), f"blk.{i}.attn_k.weight": ("q6_k", (H, H)),
                  f"blk.{i}.attn_v.weight": ("q8_0", (H, H)), f"blk.{i}.attn_output.weight": ("q8_0", (H, H)),
                  f"blk.{i}.attn_norm.weight": ("f32", (H,)), f"blk.{i}.ffn_norm.weight": ("f32", (H,)),
                  f"blk.{i}.ffn_gate.weight": ("iq2_xxs", (F, H)), f"blk.{i}.ffn_up.weight": ("iq2_xxs", (F, H)),
                  f"blk.{i}.ffn_down.weight": ("q4_k", (H, F))}
    path = tmp / "tiny-qat.gguf"
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(128); w.add_embedding_length(H); w.add_block_count(2); w.add_feed_forward_length(F)
    w.add_head_count(2); w.add_head_count_kv(2); w.add_layer_norm_rms_eps(1e-5); w.add_rope_dimension_count(128)
    w.add_file_type(16)
    T = gguf.GGMLQuantizationType
    for name, (typ, shape) in types.items():
        x = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if typ == "f32":
            w.add_tensor(name, x)
        elif typ == "f16":
            w.add_tensor(name, x.astype(np.float16))
        else:
            im = None
            if name.startswith("blk."):
                im = rng.uniform(0.1, 3.0, shape[-1]).astype(np.float32)
                imat[name] = im
            q = np.frombuffer(g.quantize(typ, x.reshape(-1, shape[-1]), im), dtype=np.uint8)
            q = q.reshape(shape[0], -1)
            w.add_tensor(name, q, raw_dtype=T[typ.upper()])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    imf = tmp / "tiny.imatrix"
    _write_imatrix(imf, imat)
    return path, imf, imat


def test_build_copies_untouched_bytes_and_requantizes_selected(ggml, tmp_path):
    src, imf, imat = _synthetic(tmp_path, ggml)
    ts = pt.read_tensors(str(src))
    assert pt.is_quantized_source(ts)
    assign = {t["name"]: t["type"] for t in ts}
    assign["blk.1.ffn_gate.weight"] = "iq1_s"                          # QAT IQ2_XXS -> 1-bit (needs imatrix)
    assign["blk.1.ffn_up.weight"] = "iq1_m"
    assign["blk.0.ffn_down.weight"] = "q2_K"                           # a K-quant downgrade too
    lines, flags, stats = pt.quantize_plan(ts, assign)
    tt = tmp_path / "tt.txt"
    tt.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out.gguf"
    cmd = pt.build_command(ggml_ref.llama_bin("llama-quantize"), str(src), str(out), str(tt), flags, str(imf))
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = pt.verify_build(ts, str(out), assign)
    assert rep["ok"], rep
    assert rep["requantized"] == 3 and rep["identical"] == len(ts) - 3
    # the requantized tensors are exactly ggml's quantize(dequant(source), imatrix)
    s_t = {t.name: t for t in gguf.GGUFReader(str(src)).tensors}
    o_t = {t.name: t for t in gguf.GGUFReader(str(out)).tensors}
    for name, typ in (("blk.1.ffn_gate.weight", "iq1_s"), ("blk.1.ffn_up.weight", "iq1_m"),
                      ("blk.0.ffn_down.weight", "q2_k")):
        st = s_t[name]
        rows, n = int(st.shape[1]), int(st.shape[0])
        deq = ggml.dequant(st.tensor_type.name.lower(), bytes(st.data), rows, n)
        want = ggml.quantize(typ, deq, imat[name])
        assert o_t[name].tensor_type.name.lower() == typ
        assert bytes(o_t[name].data) == want, name


def test_fit_cli_passthrough_end_to_end(ggml, tmp_path, monkeypatch, capsys):
    import pollard_fit
    src, imf, _ = _synthetic(tmp_path, ggml)
    ts = pt.read_tensors(str(src))
    target_gb = (sum(t["n_bytes"] for t in ts) - 20_000) / 1e9         # force a few downgrades
    out = tmp_path / "rung.gguf"
    monkeypatch.setattr(sys, "argv", ["pollard-fit", "--gguf", str(src), "--ram", repr(target_gb / 0.85),
                                      "--reserve", "0", "--imatrix", str(imf), "--out", str(out),
                                      "--llama-quantize", ggml_ref.llama_bin("llama-quantize")])
    pollard_fit.main()
    text = capsys.readouterr().out
    assert "PASSTHROUGH" in text and "byte-identical" in text, text
    got = {t["name"]: t["type"] for t in pt.read_tensors(str(out))}
    changed = [n for n, ty in got.items() if ty != next(t["type"] for t in ts if t["name"] == n)]
    assert changed and all(pt.tensor_class(n) == "bulk" for n in changed)
    assert os.path.getsize(out) < os.path.getsize(src)


def test_fit_cli_refuses_an_upgrade(ggml, tmp_path, monkeypatch):
    import pollard_fit
    src, imf, _ = _synthetic(tmp_path, ggml)
    monkeypatch.setattr(sys, "argv", ["pollard-fit", "--gguf", str(src), "--ram", "16", "--imatrix", str(imf),
                                      "--tier", "bulk=q6_K", "--plan-only"])
    with pytest.raises(SystemExit, match="UPGRADE"):
        pollard_fit.main()


# ------------------------------------------------------------------ the one-shot plans it -----------
def _gguf_py_quantized(tmp):
    """A Q8_0 / Q4_0 llama GGUF made with gguf-py alone (no ggml needed) -- enough to plan with."""
    rng = np.random.default_rng(1)
    path = tmp / "q8-src.gguf"
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(128); w.add_embedding_length(H); w.add_block_count(2); w.add_feed_forward_length(F)
    w.add_head_count(2); w.add_head_count_kv(2); w.add_layer_norm_rms_eps(1e-5); w.add_rope_dimension_count(128)
    T = gguf.GGMLQuantizationType
    from gguf.quants import quantize

    def add(name, shape, typ):
        x = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if typ == "f32":
            w.add_tensor(name, x)
        else:
            w.add_tensor(name, quantize(x, T[typ]), raw_dtype=T[typ])
    add("token_embd.weight", (V, H), "Q8_0"); add("output_norm.weight", (H,), "f32")
    for i in range(2):
        for n in ("attn_q", "attn_k", "attn_v", "attn_output"):
            add(f"blk.{i}.{n}.weight", (H, H), "Q8_0")
        add(f"blk.{i}.ffn_gate.weight", (F, H), "Q8_0"); add(f"blk.{i}.ffn_up.weight", (F, H), "Q8_0")
        add(f"blk.{i}.ffn_down.weight", (H, F), "Q4_0")
        add(f"blk.{i}.attn_norm.weight", (H,), "f32"); add(f"blk.{i}.ffn_norm.weight", (H,), "f32")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path


def test_pollard_ladder_plans_passthrough_rungs(tmp_path, monkeypatch, capsys):
    import pollard_auto
    src = _gguf_py_quantized(tmp_path)
    imf = tmp_path / "x.imatrix"
    _write_imatrix(imf, {"blk.0.ffn_up.weight": np.ones(H)})
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", ["pollard", "--gguf", str(src), "--ladder", "--imatrix", str(imf)])
    pollard_auto.main()
    text = capsys.readouterr().out
    assert "ALREADY QUANTIZED" in text and "PASSTHROUGH" in text
    assert ("pollard-sensitivity --gguf" in text or "pollard_sensitivity.py --gguf" in text) \
        and "--passthrough --out" in text                                        # measured on this file
    for label in ("PT25", "PT50", "PT75", "PT100"):
        assert f"-{label}.gguf" in text
    assert text.count("--passthrough on") == 4 and "--plan-only" in text
    assert "pollard-automap" not in text                                         # no requantize-all flagship


def test_passthrough_rungs_are_written_to_the_workspace_not_beside_the_source(tmp_path, monkeypatch):
    """The Bolt release (2026-10-24): `pollard --gguf <vendor GGUF in downloads/> --ladder --rung PT75` wrote the 38 GB
    rung next to the source, outside the workspace. A rung goes to $POLLARD_HOME/models/<gguf>/ -- the folder
    pollard-fit writes every other --gguf build to, whose MANIFEST records it."""
    import pollard_auto
    import pollard_workspace as ws
    src = _gguf_py_quantized(tmp_path)                        # a vendor GGUF outside the workspace
    imf = tmp_path / "x.imatrix"
    _write_imatrix(imf, {"blk.0.ffn_up.weight": np.ones(H)})
    home = tmp_path / "home"
    monkeypatch.setenv("POLLARD_HOME", str(home))
    ran = []
    monkeypatch.setattr(pollard_auto, "_run", lambda cmd, do_run, cwd=None: ran.append([str(x) for x in cmd]) or True)
    monkeypatch.setattr(sys, "argv", ["pollard", "--gguf", str(src), "--ladder", "--imatrix", str(imf),
                                      "--rung", "PT75"])
    pollard_auto.main()
    fit = [c for c in ran if "--passthrough" in c and c[c.index("--passthrough") + 1] == "on"]
    assert len(fit) == 1
    out = fit[0][fit[0].index("--out") + 1]
    assert out == os.path.join(ws.model_dir(str(src)), "q8-src-PT75.gguf")
    assert out.startswith(str(home) + os.sep) and os.path.dirname(out) != str(tmp_path)
    assert not os.path.isdir(ws.model_dir(str(src))), "a plan writes nothing"
    made = pollard_auto._rung_out(str(src), "PT50", create=True)  # a real build: the folder exists for llama-quantize
    assert os.path.isdir(os.path.dirname(made)) and made.endswith("q8-src-PT50.gguf")


def test_fit_plan_only_on_gguf_py_source(tmp_path, monkeypatch, capsys):
    import pollard_fit
    src = _gguf_py_quantized(tmp_path)
    ts = pt.read_tensors(str(src))
    target = sum(t["n_bytes"] for t in ts) * 0.8
    monkeypatch.setattr(sys, "argv", ["pollard-fit", "--gguf", str(src), "--ram", f"{target / 0.85e9:.9f}",
                                      "--reserve", "0", "--out", str(tmp_path / "o.gguf"), "--plan-only"])
    pollard_fit.main()
    text = capsys.readouterr().out
    assert "PASSTHROUGH" in text and "--allow-requantize" in text and "Q8_0" in text
    lines = (tmp_path / "o.gguf.tensor-types.txt")
    assert not lines.exists()                                                    # plan-only writes nothing
    monkeypatch.setattr(sys, "argv", ["pollard-fit", "--gguf", str(src), "--ram", "16", "--passthrough", "off",
                                      "--plan-only"])
    pollard_fit.main()
    assert "PASSTHROUGH" not in capsys.readouterr().out


def test_passthrough_sweep_crushes_one_layer_at_a_time():
    ts = _bolt_like()
    src = {t["name"]: t["type"] for t in ts}
    seen = []

    def build(assign):
        diff = {n: ty for n, ty in assign.items() if ty != src[n]}
        assert all(pt.tensor_class(n) == "bulk" for n in diff)               # never touches the rest
        seen.append(diff)
        return "probe.gguf"

    def kl(path):
        d = seen[-1]
        layers = {pt.layer_of(n) for n in d}
        return 0.5 if len(layers) > 1 else (0.3 if layers == {2} else 0.01)

    prof = pt.sweep(ts, 4, ["ffn"], pt.default_probe(ts), build, kl, log=lambda *_: None)
    assert prof["passthrough"] and prof["probe"] == "iq1_s"
    assert set(prof["noise"]) == {"iq1_m", "iq1_s"}
    assert prof["ffn"] == {"0": 0.01, "1": 0.01, "2": 0.3, "3": 0.01}
    for d in seen[2:]:                                                         # per-layer probes
        assert len({pt.layer_of(n) for n in d}) == 1 and set(d.values()) == {"iq1_s"}
    plan = pt.plan_passthrough(ts, int(sum(t["n_bytes"] for t in ts) * 0.94), prof)
    assert 2 not in {pt.layer_of(n) for n in plan["changed"]}                  # the sensitive layer is kept
