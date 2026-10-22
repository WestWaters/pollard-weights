"""d3 decision models (vllm-sr/d3 family, `decision_config.json` format_id `d3-code-readout-v1`): detected as their own
decision kind, every lane quantizes the body only and ships readout + decision config byte-exact (pollard-complete fails
a rung without them, or with the source's manifest), the GGUF lane converts the bare Qwen3_5Model body with non-causal
attention + last pooling in its metadata, release gates each rung with pollard-d3 (readout distribution vs bf16) and
holds EXL3 / EXLR8 with the reason, and the card carries the d3 board and serve commands.

numpy only (every CI shape): hand-written safetensors folders; the runtimes are faked."""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import pollard_d3 as D3  # noqa: E402
import pollard_complete as PC  # noqa: E402

H = 8


def _st(path, tensors):
    """safetensors from {name: ndarray} (F32 / BF16 via uint16), by hand."""
    hdr, blobs, off = {}, [], 0
    for k, a in tensors.items():
        if a.dtype == np.uint16:
            dt, raw = "BF16", a.tobytes()
        else:
            dt, raw = "F32", a.astype(np.float32).tobytes()
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def _d3(tmp_path, name="d3-tiny", index=True):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5Model"], "model_type": "qwen3_5",
        "text_config": {"hidden_size": H, "num_hidden_layers": 1, "model_type": "qwen3_5_text"},
        "vision_config": {"depth": 1, "model_type": "qwen3_5_vision"},
        "auto_map": {"AutoModel": "modeling_d3.D3Model"}}))
    codes = [chr(65 + i) for i in range(26)] + [f"A{chr(65 + i)}" for i in range(26)]
    (d / D3.CONFIG).write_text(json.dumps({"format_version": 1, "format_id": D3.FORMAT_ID, "prompt": "d3",
                                           "codes": codes, "token_ids": list(range(32, 32 + len(codes))),
                                           "temperature": 1.0, "attention_mode": "noncausal_full_attention",
                                           "pooling": "last", "readout_dtype": "float32"}))
    rng = np.random.default_rng(0)
    _st(d / D3.READOUT, {"weight": rng.standard_normal((255, H)).astype(np.float32)})
    body = {"language_model.embed_tokens.weight": np.zeros((64, H), np.uint16),
            "language_model.layers.0.mlp.down_proj.weight": np.zeros((H, H), np.uint16),
            "language_model.norm.weight": np.zeros((H,), np.uint16),
            "visual.blocks.0.attn.qkv.weight": np.zeros((3 * H, H), np.uint16)}
    _st(d / "model-00001-of-00001.safetensors", body)
    if index:
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in body}}))
    for f in ("d3_runtime.py", "d3_format.py", "modeling_d3.py"):
        (d / f).write_text("# the package's runtime\n")
    (d / "preprocessor_config.json").write_text("{}")
    (d / D3.MANIFEST).write_text(json.dumps({"schema": D3.MANIFEST_SCHEMA, "model_name": name, "repo_id": f"vllm-sr/{name}",
                                             "identity": {"model_sha256": "ab" * 32, "files_sha256": {}},
                                             "licence": {"spdx": "apache-2.0"},
                                             "files_sha256": {"model-00001-of-00001.safetensors": "00"}}))
    return d


# ------------------------------------------------------------------------------------------------- detection

def test_d3_is_detected_by_its_decision_config_and_nothing_else(tmp_path):
    d = _d3(tmp_path)
    assert D3.is_d3(d) and D3.decision_config(d)["format_id"] == D3.FORMAT_ID
    i = D3.info(d)
    assert i["hidden_size"] == H and i["codes"] == 52 and i["readout"] and i["repo_id"] == "vllm-sr/d3-tiny"
    other = tmp_path / "plain"
    other.mkdir()
    (other / D3.CONFIG).write_text(json.dumps({"format_id": "something-else"}))
    assert not D3.is_d3(other) and not D3.is_d3(tmp_path / "missing")
    # a GGUF beside the package, and a converted GGUF's own tag
    assert D3.is_d3(str(d / "x-Q4_K_M.gguf"))
    assert D3.gguf_is_d3({"general.tags": ["decision-model", D3.FORMAT_ID]})
    assert not D3.gguf_is_d3({"general.tags": ["decision-model"]})


def test_modelkind_reports_the_d3_format_and_drops_the_base_templates_tool_and_thinking_kinds(tmp_path):
    import pollard_modelkind as MK
    d = _d3(tmp_path)
    (d / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m.content }}{% endfor %}<tool_call></tool_call> tools tool_call tool_call "
        "<think></think>{% if enable_thinking %}{% endif %}")
    k = MK.classify(str(d))
    assert k["decision"] and k["decision_format"] == D3.FORMAT_ID and k["eval"] == "decision"
    assert not k["agentic"] and not k["thinking"], "a d3 model never generates: no tool-call gate, no reasoning share"
    assert any("pollard-d3" in w for w in k["why"])


def test_converter_lookup_names_the_class_a_d3_body_converts_as(tmp_path):
    import pollard_convert as PCV
    d = _d3(tmp_path)
    assert PCV.model_architectures(str(d)) == ["Qwen3_5ForConditionalGeneration"]


def test_the_probe_loads_a_d3_body_with_its_own_names_not_random_weights(tmp_path):
    d = _d3(tmp_path)
    km = D3.causal_lm_kwargs(str(d))["key_mapping"]
    import re
    (pat, rep), = km.items()
    assert re.sub(pat, rep, "language_model.layers.0.mlp.down_proj.weight") == "model.layers.0.mlp.down_proj.weight"
    assert re.sub(pat, rep, "visual.blocks.0.attn.qkv.weight") == "visual.blocks.0.attn.qkv.weight"
    assert D3.causal_lm_kwargs(str(tmp_path)) == {}


# ------------------------------------------------------------------------------------------------- shipping

def test_complete_gate_requires_the_decision_files_byte_exact_and_fill_ships_them(tmp_path):
    src = _d3(tmp_path)
    out = tmp_path / "rung"
    out.mkdir()
    _st(out / "model-00001-of-00001.safetensors",
        {"language_model.layers.0.mlp.down_proj.qweight": np.zeros((2, H), np.float32),
         "language_model.layers.0.mlp.down_proj.scales": np.zeros((1, H), np.float32),
         "language_model.embed_tokens.weight": np.zeros((64, H), np.uint16),
         "language_model.norm.weight": np.zeros((H,), np.uint16),
         "visual.blocks.0.attn.qkv.weight": np.zeros((3 * H, H), np.uint16)})
    (out / "preprocessor_config.json").write_text("{}")
    res = PC.check(str(src), str(out), "gptq")
    assert not res["ok"] and any("readout.safetensors: absent" in m for m in res["missing"])
    assert any("decision_config.json: absent" in m for m in res["missing"])
    # a readout that is not the source's bytes (re-saved, cast, quantized) fails too
    _st(out / D3.READOUT, {"weight": np.ones((255, H), np.float32)})
    (out / D3.CONFIG).write_bytes((src / D3.CONFIG).read_bytes())
    res = PC.check(str(src), str(out), "gptq")
    assert any("readout.safetensors: differs from the source" in m for m in res["missing"])
    # the source's own manifest on a rung: the d3 runtime would refuse it (its shard hashes)
    (out / D3.MANIFEST).write_bytes((src / D3.MANIFEST).read_bytes())
    res = PC.check(str(src), str(out), "gptq")
    assert any("SOURCE manifest" in m for m in res["missing"])
    wrote = PC.fill(str(src), str(out), "gptq")
    assert D3.READOUT in wrote and D3.MANIFEST in wrote
    res = PC.check(str(src), str(out), "gptq")
    assert res["ok"], res["missing"]
    assert D3.sha256_file(out / D3.READOUT) == D3.sha256_file(src / D3.READOUT)
    man = json.loads((out / D3.MANIFEST).read_text())
    assert man["schema"] == D3.MANIFEST_SCHEMA and man["model_name"] == "rung"
    assert man["files_sha256"]["model-00001-of-00001.safetensors"] == D3.sha256_file(out / "model-00001-of-00001.safetensors")
    assert man["identity"]["files_sha256"][D3.READOUT] == D3.sha256_file(src / D3.READOUT)
    assert man["identity"]["model_sha256"] is None and man["pollard"]["source_model_sha256"] == "ab" * 32
    assert man["licence"]["spdx"] == "apache-2.0"
    assert (out / "d3_runtime.py").is_file(), "an HF-format rung carries the package runtime"


def test_gguf_lane_ships_the_decision_files_beside_the_gguf_and_no_manifest(tmp_path):
    src = _d3(tmp_path)
    out = tmp_path / "gguf"
    out.mkdir()
    (out / "d3-tiny-Q4_K_M.gguf").write_bytes(b"GGUF")
    carried, missing = D3.check_files(src, out, "gguf")
    assert len(missing) == 2 and not carried
    wrote = D3.ship(src, out, "gguf", log=lambda *_: None)
    assert sorted(wrote) == sorted(D3.DECISION_FILES)
    carried, missing = D3.check_files(src, out, "gguf")
    assert not missing and len(carried) == 2
    assert not (out / D3.MANIFEST).exists() and not (out / "d3_runtime.py").exists()


def test_side_file_copy_never_puts_the_source_manifest_on_a_rung(tmp_path):
    import pollard_stream as S
    src = _d3(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    S.copy_side_files(str(src), str(out))
    assert (out / D3.READOUT).is_file() and (out / D3.CONFIG).is_file()
    assert not (out / D3.MANIFEST).exists()
    assert not (out / "model-00001-of-00001.safetensors").exists()


def test_mx_source_shards_never_treat_the_readout_as_a_body_shard(tmp_path):
    import pollard_mxfp6 as M6
    src = _d3(tmp_path)
    assert [os.path.basename(p) for p in M6.source_shards(str(src))] == ["model-00001-of-00001.safetensors"]
    bare = _d3(tmp_path, "bare", index=False)
    assert D3.READOUT not in [os.path.basename(p) for p in M6.source_shards(str(bare))]


# ------------------------------------------------------------------------------------------------- GGUF convert

def test_convert_view_and_command_carry_the_three_d3_facts(tmp_path):
    src = _d3(tmp_path)
    out = tmp_path / "g" / "d3-tiny-f16.gguf"
    out.parent.mkdir()
    cmd = D3.convert_cmd(str(src), str(out), "/x/convert_hf_to_gguf.py", python="py")
    view = out.parent / ".pollard-d3-view"
    assert json.loads((view / "config.json").read_text())["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert os.path.islink(view / "model-00001-of-00001.safetensors")
    assert json.loads((src / "config.json").read_text())["architectures"] == ["Qwen3_5Model"], "source untouched"
    assert cmd[:3] == ["py", str(out.parent / ".pollard-d3-convert.py"), "/x/convert_hf_to_gguf.py"]
    assert cmd[3] == str(view) and "--no-mtp" in cmd and cmd[cmd.index("--outfile") + 1] == str(out)
    shim = (out.parent / ".pollard-d3-convert.py").read_text()
    assert '"model." + name' in shim and "add_causal_attention(False)" in shim
    assert "PoolingType.LAST" in shim and D3.FORMAT_ID in shim
    compile(shim, "shim", "exec")
    args = D3.llama_server_args({"general.architecture": "qwen35"}, ctx=4096)
    assert "--embeddings" in args and args[args.index("--pooling") + 1] == "last"
    assert args[args.index("-ub") + 1] == "4096" and "qwen35.attention.causal=bool:false" in args
    with pytest.raises(SystemExit):
        bad = _d3(tmp_path, "vl")
        c = json.loads((bad / "config.json").read_text())
        c["model_type"] = "qwen3_vl"
        (bad / "config.json").write_text(json.dumps(c))
        D3.convert_view(str(bad), str(tmp_path / "v2"))


def test_imatrix_on_a_d3_gguf_skips_the_meaningless_perplexity(monkeypatch):
    import pollard_auto as A
    monkeypatch.setattr(A, "read_gguf_meta", lambda p: {"general.architecture": "qwen35",
                                                        "general.tags": [D3.FORMAT_ID]})
    assert "--no-ppl" in A._imatrix_shape("x.gguf")
    monkeypatch.setattr(A, "read_gguf_meta", lambda p: {"general.architecture": "qwen35"})
    assert A._imatrix_shape("x.gguf") == []


# ------------------------------------------------------------------------------------------------- readout + board

def test_readout_is_a_softmax_over_the_questions_own_codes(tmp_path):
    src = _d3(tmp_path)
    W = D3.load_readout(src)
    assert W.shape == (255, H) and W.dtype == np.float32
    h = np.linspace(-1, 1, H).astype(np.float32)
    p = D3.readout_probs(h, W, 3, T=1.0)
    z = W[:3] @ h
    want = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert np.allclose(p, want, atol=1e-6) and abs(sum(p) - 1) < 1e-9
    # temperature
    p2 = D3.readout_probs(h, W, 3, T=2.0)
    assert np.allclose(p2, np.exp(z / 2 - (z / 2).max()) / np.exp(z / 2 - (z / 2).max()).sum(), atol=1e-6)


def _rows(dists, types):
    return [{"row": i, "workflow": "w", "qid": "q", "type": t, "dist": d, "gold": {"label": max(d, key=d.get)}}
            for i, (d, t) in enumerate(zip(dists, types))]


def test_board_splits_by_question_type_and_reports_the_kl_tail():
    ref = _rows([{"a": .9, "b": .1}, {"true": .7, "false": .3}, {"0": .2, "1": .8}], ["choice", "noul", "score"])
    same = _rows([{"a": .9, "b": .1}, {"true": .7, "false": .3}, {"0": .2, "1": .8}], ["choice", "noul", "score"])
    b = D3.board(same, ref, "set")
    assert b["agreement"] == 1.0 and b["option_kl"] < 1e-9 and b["option_kl_max"] < 1e-9
    assert set(b["by_type"]) == {"choice", "noul", "score"}
    moved = _rows([{"a": .4, "b": .6}, {"true": .7, "false": .3}, {"0": .2, "1": .8}], ["choice", "noul", "score"])
    b = D3.board(moved, ref, "set")
    assert b["by_type"]["choice"]["agreement"] == 0.0 and b["by_type"]["noul"]["agreement"] == 1.0
    assert b["option_kl_max"] > 0.3 and b["flips"] == 1


def test_builtin_set_becomes_d3_choice_questions():
    items, label = D3.builtin_items()
    assert len(items) == 20 and "20" in label
    it = items[0]
    assert it["question"]["type"] == "choice" and list(it["question"]["criteria"]) == ["billing", "shipping",
                                                                                      "technical", "sales"]
    assert it["gold"]["label"] == "billing"


def test_typed_subset_is_spread_over_every_workflow(monkeypatch):
    import pollard_decision as D
    rows = [{"id": i, "workflow": f"wf{i // 100}", "state": "s", "questions": {"q": {"type": "noul"}},
             "gold": {"q": {"label": "true"}}} for i in range(400)]
    monkeypatch.setattr(D, "load_typed", lambda: {"dataset": "LocalLLaMA/typed-decisions", "revision": "abc",
                                                  "rows": rows})
    items, label = D3.typed_items(8)
    assert len(items) == 8 and {i["workflow"] for i in items} == {"wf0", "wf1", "wf2", "wf3"}
    assert "8 of 400" in label


def test_gate_runs_reference_and_rung_on_the_same_prompts_and_records_the_verdict(tmp_path, monkeypatch):
    src = _d3(tmp_path)

    class P:
        T = 1.0
        rt = None

        def __init__(self, s):
            pass

        def build(self, state, q):
            keys = list(q["criteria"])
            return [1, 2, 3], keys, "choice"

    class Ref:
        kind = "reference"
        replaced = 0

        def __init__(self, *a, **k):
            pass

        def probs(self, batch):
            return [[0.7] + [0.3 / (n - 1)] * (n - 1) for _, n in batch]

        def close(self):
            pass

    class Rung(Ref):
        kind = "gguf"

        def probs(self, batch):
            return [[0.69] + [0.31 / (n - 1)] * (n - 1) for _, n in batch]

    monkeypatch.setattr(D3, "Prompts", P)
    monkeypatch.setattr(D3, "TorchRunner", Ref)
    monkeypatch.setattr(D3, "open_runner", lambda *a, **k: Rung())
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    rung = tmp_path / "x-Q4_K_M.gguf"
    rung.write_bytes(b"GGUF")
    res = D3.gate(str(rung), str(src), qset="builtin", log=lambda *_: None)
    assert res["verdict"] == "PASS" and res["board"]["agreement"] == 1.0
    assert 0 < res["board"]["option_kl"] < 0.01 and res["board"]["format"] == D3.FORMAT_ID
    assert res["board"]["runtime"].startswith("llama.cpp")
    g = D3.gate_record(res, "Q4_K_M", "gguf")
    assert g["pass"] and g["gates"] == "pollard-d3" and all(c["ok"] for c in g["checks"])
    # the reference is cached: a second rung does not recompute it
    monkeypatch.setattr(D3, "TorchRunner", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    D3.gate(str(rung), str(src), qset="builtin", log=lambda *_: None)


def test_lane_of_reads_the_rung_format(tmp_path):
    assert D3.lane_of("x.gguf") == "gguf"
    m = tmp_path / "mlx"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"quantization": {"group_size": 64, "bits": 4}}))
    assert D3.lane_of(m) == "mlx"
    raw = tmp_path / "mlx-bf16"
    raw.mkdir()
    (raw / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    _st(raw / "model.safetensors", {"language_model.model.norm.weight": np.zeros((H,), np.float32)})
    assert D3.lane_of(raw) == "mlx", "an unquantized MLX conversion is still the MLX runtime's"
    g = tmp_path / "gptq"
    g.mkdir()
    (g / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
    assert D3.lane_of(g) == "hf"


def test_systemone_answers_have_vllms_shape():
    r = D3.systemone_response("d3-edge", {"q": {"type": "choice", "choice": "a"}}, 12)
    assert r["object"] == "structured_decision" and r["usage"] == {"input_tokens": 12, "output_tokens": 0}


def test_jev_index_is_the_kits_http_engine_and_reads_scores(tmp_path):
    cmd = D3.jev_index_cmd("py", "http://127.0.0.1:9", "Q4_K_M", str(tmp_path / "run"), limit=50)
    assert cmd[:4] == ["py", "-m", "decision_index", "pipeline"] and "http" in cmd
    assert "base_url=http://127.0.0.1:9" in cmd and cmd[cmd.index("--limit") + 1] == "50"
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "scores.json").write_text(json.dumps({"index": 20.2}))
    assert D3.read_index(str(tmp_path / "run")) == 20.2
    (tmp_path / "run" / "scores.json").write_text(json.dumps({"index": {"index": 19.5}}))
    assert D3.read_index(str(tmp_path / "run")) == 19.5


# ------------------------------------------------------------------------------------------------- release + card

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    return tmp_path


def test_release_picks_the_d3_preset_and_gates_every_rung_with_pollard_d3(home):
    import pollard_release as R
    rec = ROOT / "recipes" / "release-d3.yaml"
    if rec.exists():
        assert R.load_plan(str(rec))["preset"] == "d3"
    src = _d3(home)
    cfg = R.auto_preset(R.load_plan(None), str(src))
    assert cfg["preset"] == "d3"
    assert R.auto_preset(R.load_plan(None, "text"), str(src), preset="text")["preset"] == "text", "the CLI wins"
    c, lanes, steps = R.plan_steps(cfg, str(src))
    assert c["d3"] and "exl3" in lanes and "exlr8" in lanes
    by = {s.id: s for s in steps}
    assert by["prep/smooth"].kind == "note" and "prep/probe-smoothed" not in by, "a d3 body is not smoothed"
    assert c["hf_dir"] in by["gptq/int4/build"].cmd and c["smooth_dir"] == c["hf_dir"]
    for lane, tag in (("mlx", "4bit"), ("gptq", "int4"), ("mx", "NVFP4")):
        comp, ver = by[f"{lane}/{tag}/complete"].cmd, by[f"{lane}/{tag}/verify"].cmd
        assert "--fill" in comp, "the decision files + the rung's manifest go in before the gate"
        assert ver[:2] == ["pollard-d3", "gate"] and ver[ver.index("--source") + 1] == c["hf_dir"]
        assert f"{lane}/{tag}/toolcall" not in by
    assert by["mlx/4bit/verify"].requires == "apple" and by["gptq/int4/verify"].requires == "any"
    # EXL3 / EXLR8: held with the reason, never built
    assert by["exl3/hold"].kind == "hold" and "non-causal" in by["exl3/hold"].desc
    assert by["exlr8/hold"].kind == "hold"
    assert not any(s.id.endswith("/build") for s in steps if s.lane in ("exl3", "exlr8"))


def test_release_gguf_rungs_of_a_d3_model_are_gated_on_the_readout(home):
    import pollard_release as R
    src = _d3(home)
    cfg = R.auto_preset(R.load_plan(None), str(src))
    c, _, _ = R.plan_steps(cfg, str(src), {"gguf"})
    steps = R.gguf_rung_steps(cfg, c, [{"tag": "Q4_K_M", "path": "/m/d3-tiny-Q4_K_M.gguf"}])
    assert [s.id for s in steps] == ["gguf/Q4_K_M/verify"]
    cmd = steps[0].cmd
    assert cmd[:2] == ["pollard-d3", "gate"] and cmd[cmd.index("--rung") + 1] == "/m/d3-tiny-Q4_K_M.gguf"
    assert cmd[cmd.index("--rung-name") + 1] == "Q4_K_M" and "--ngl" in cmd
    assert not any("pollard-decision" in s.cmd for s in steps), "the first-position letter readout is wrong for d3"


def test_card_shows_the_d3_board_with_runtime_ece_and_the_index_delta():
    import pollard_card as C
    builds = [{"name": "d3-edge-Q8_0.gguf", "tag": "Q8_0", "bytes": 6e8},
              {"name": "d3-edge-IQ4_XS.gguf", "tag": "IQ4_XS", "bytes": 3e8}]
    board = lambda ag, kl, idx: {"format": D3.FORMAT_ID, "n": 2000, "agreement": ag, "option_kl": kl,
                                 "option_kl_p95": kl * 3, "ece": 0.12, "delta_ece": 0.003,
                                 "runtime": "llama.cpp (llama-server --embeddings --pooling last, non-causal)",
                                 "set": "LocalLLaMA/typed-decisions@d9eda7f3",
                                 "by_type": {"choice": {"n": 800, "agreement": ag, "option_kl": kl}},
                                 "jev_index": idx}
    results = {"Q8_0": {"decision": board(0.995, 0.0004, 20.1)}, "IQ4_XS": {"decision": board(0.97, 0.004, None)},
               "bf16": {"decision": {"jev_index": 20.2}}, "_decision_ref": "bf16"}
    text = "\n".join(C.decision_section(builds, results))
    assert "## Decision fidelity (d3 readout)" in text and "bf16" in text and "readout.safetensors" in text
    assert "99.5%" in text and "llama.cpp" in text and "20.1 (-0.1)" in text and "Decision Index" in text
    assert "typed-decisions" in text
    lines = D3.serve_lines({"repo_id": "vllm-sr/d3-edge"}, "PollardWeights/d3-edge-Pollard", ["gguf", "mlx", "gptq"])
    txt = "\n".join(lines)
    assert "vllm serve vllm-sr/d3-edge" in txt and "/v1/systemone" in txt
    assert "--embeddings --pooling last" in txt and "pollard-d3 serve" in txt


def test_mlx_compat_is_harmless_without_mlx_and_restores_what_it_patched():
    with D3.mlx_compat():
        pass
    try:
        from mlx_lm.models import qwen3_5 as Q
    except Exception:                                   # noqa: BLE001 -- not an Apple / mlx box
        return
    cfg = {"model_type": "qwen3_5", "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 128,
        "linear_num_value_heads": 2, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
        "tie_word_embeddings": True, "rope_parameters": {"rope_theta": 1e6, "partial_rotary_factor": 0.25},
        "layer_types": ["linear_attention", "full_attention", "full_attention", "full_attention"]}}
    kinds = lambda m: "".join("l" if l.is_linear else "F" for l in m.language_model.model.layers)
    with D3.mlx_compat():
        assert kinds(Q.Model(Q.ModelArgs.from_dict(cfg))) == "lFFF", "layer_types decides each layer"
    assert kinds(Q.Model(Q.ModelArgs.from_dict(cfg))) == "lllF", "patches undone on exit"
