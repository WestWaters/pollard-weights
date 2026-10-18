"""Regression tests for the defects the small-model lane qualification found on the box (Qwen2.5-0.5B-Instruct, every
lane, 2026-10-17, C:\\pollard\\release\\qualification.json) and the LFM2.5-24B-A2B GGUF publish the same day:

1. pollard-mx read `profile.get("layers", profile)`: a pollard-probe profile's "layers" is the layer COUNT -> every
   NVFP4 / MXFP4 / --mix rung died on `'int' object has no attribute 'items'`.
2. pollard-verify --lane gptq took the whole-model path and kld_top1 cast [4096, 151936] logits to float64 (4.64 GiB
   per copy) -> MemoryError on a 32 GB box, every GPTQ gate failed regardless of quality.
3. A GPTQ 4/8 mix (`dynamic`) loaded through transformers is noise (optimum builds every linear at the base bits), and
   the export's rules never matched vLLM's fused module names -> decoded loader + fused-safe rules.
4. EXL3 as a SYSTEM scheduled task: no ninja on PATH, no torch_extensions cache -> 'DLL load failed'.
5. The coherence step served IQ*_KT (ik-only) rungs with stock llama-server (NO_SERVER), and a thinking model ran out
   of the 220-token budget mid-thought (BELOW FLOOR at IQ3_S, PASS at 1024).
6. The release card wrote base_model: <local path>, imatrix: no, and ~2.5B for a 24B MoE.
7. pollard-toolcall refused GPTQ / compressed-tensors rungs whenever the GPU was someone else's.

The numpy-only checks run in every CI shape; the tiny-model ones need torch + transformers (+ tokenizers)."""
from __future__ import annotations

import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import pollard_export as E  # noqa: E402
import pollard_gptqpack as G  # noqa: E402
import pollard_mx as X  # noqa: E402
import pollard_mxfp6 as M6  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_verify as V  # noqa: E402
import sensitivity_profiles as SP  # noqa: E402


# ------------------------------------------------------------------------------------- 1. MX: the profile's shape

def test_mx_reads_a_real_probe_profile_whose_layers_key_is_a_count():
    for prof, n in ((SP.LFM2_5_24B_A2B, 40), (SP.QWEN25_05B_SMOOTHED, 24)):
        assert isinstance(prof["layers"], int) and prof["layers"] == n, "the shape that crashed pollard-mx"
        s = X.layer_sensitivity(prof)
        assert sorted(s, key=int) == [str(i) for i in range(n)]
        for k in ("0", "7", str(n - 1)):                       # a layer's damage = the sum over its groups
            assert s[k] == pytest.approx(prof["attn"][k] + prof["ffn"][k])
        hot = X.allocate(s, n, 0.25)
        assert len(hot) == round(0.25 * n) and hot <= set(s)
        top = max(s, key=s.get)
        assert top in hot


def test_mx_still_reads_the_flat_profile_shapes():
    assert X.layer_sensitivity({"layers": {"0": 0.5, "1": 2.0}}) == {"0": 0.5, "1": 2.0}
    assert X.layer_sensitivity({"0": 0.5, "1": 2.0, "probe": "x"}) == {"0": 0.5, "1": 2.0}
    assert X.layer_sensitivity({"attn": {"0": 1.0, "1": None}, "ffn": {"1": 3.0}, "layers": 2,
                                "excluded_tensors": {"blk.0.x": {"reason": "r"}}, "noise": {"q6_K": 4.0},
                                "stability": {"groups": {}}}) == {"0": 1.0, "1": 3.0}
    assert X.layer_sensitivity(None) == {} and X.layer_sensitivity([1, 2]) == {}


def test_mx_cli_plans_nvfp4_from_a_probe_profile(tmp_path, monkeypatch, capsys):
    p = tmp_path / "LFM2.5-24B-A2B.sensitivity.json"
    p.write_text(json.dumps(SP.LFM2_5_24B_A2B))
    for scheme in ("NVFP4", "MXFP4"):
        monkeypatch.setattr(sys, "argv", ["pollard-mx", "--model", "/nonexistent", "--layers", "40", "--scheme", scheme,
                                          "--sensitivity", str(p), "--plan-only"])
        X.main()                                                # crashed with AttributeError before
        out = capsys.readouterr().out
        assert scheme in out


def test_mx_mix_allocates_from_a_probe_profile():
    s = X.layer_sensitivity(SP.QWEN25_05B_SMOOTHED)
    atoms = ["mxfp4", "mxfp6_e2m3", "mxfp6_e3m2", "mxfp8"]
    groups = {f"model.layers.{L}.mlp.down_proj": {"layer": L, "n": 1000, "w2": 1.0,
                                                  "members": [f"model.layers.{L}.mlp.down_proj"],
                                                  "err": {"mxfp4": 0.02, "mxfp6_e2m3": 0.02 / 16,
                                                          "mxfp6_e3m2": 0.02 / 8, "mxfp8": 0.02 / 256}}
              for L in range(24)}
    assign = M6.allocate_mix(groups, 5.0, atoms, s)
    assert set(assign) == set(groups) and M6.avg_bpw(groups, assign) <= 5.0 + 1e-9
    # the budget goes to the layers the profile says hurt most
    hi, lo = max(s, key=s.get), min(s, key=s.get)
    bpw = {a: M6.ATOMS[a]["bpw"] for a in atoms}
    assert bpw[assign[f"model.layers.{hi}.mlp.down_proj"]] > bpw[assign[f"model.layers.{lo}.mlp.down_proj"]]


# ------------------------------------------------------------------------------- 2. verify: GPTQ memory + routing

def _kld64(r, q):
    r = np.asarray(r, np.float64); q = np.asarray(q, np.float64)
    lr = r - r.max(-1, keepdims=True); lr = lr - np.log(np.exp(lr).sum(-1, keepdims=True))
    lq = q - q.max(-1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lr) * (lr - lq)).sum(-1)


def test_kld_top1_chunked_float32_matches_the_float64_definition():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 4, (700, 512)).astype(np.float32)
    q = (r + rng.normal(0, 0.5, r.shape)).astype(np.float32)
    want = _kld64(r, q)
    for rows in (1, 64, 256, 10000):
        got = V.kld_top1(r, q, rows=rows)
        assert got["kld"] == pytest.approx(want.mean(), rel=1e-5)
        assert got["kld_p99"] == pytest.approx(np.percentile(want, 99), rel=1e-4)
        assert got["top1"] == pytest.approx((r.argmax(-1) == q.argmax(-1)).mean() * 100)
        assert got["n"] == 700
    # fp16 logits (the cached reference) and a padded vocabulary on one side
    assert V.kld_top1(r.astype(np.float16), np.c_[q, np.zeros((700, 3), np.float32)])["n"] == 700


def test_kld_top1_never_holds_a_float64_copy_of_the_logits():
    import tracemalloc
    rng = np.random.default_rng(1)
    r = rng.normal(0, 3, (2048, 8192)).astype(np.float32)        # 64 MB each; a float64 copy is 128 MB
    q = (r + rng.normal(0, 0.3, r.shape)).astype(np.float32)
    tracemalloc.start()                                         # the old math peaked at ~0.6 GB here
    V.kld_top1(r, q)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 100e6, f"peak {peak / 1e6:.0f} MB: the whole-array float64 math is back"


def test_verify_streams_any_gptq_rung(tmp_path):
    rung = tmp_path / "M-Pollard-GPTQ-int4"
    rung.mkdir()
    (rung / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq", "bits": 4}}))
    (rung / "pollard-allocation.json").write_text("{}")             # what pollard-export's gptqmodel build writes
    ok, why = V.stream_wanted(str(tmp_path / "src"), "auto", str(rung))
    assert ok and "GPTQ" in why
    assert V.stream_wanted(str(tmp_path / "src"), "off", str(rung)) == (False, "")


# ------------------------------------------------------------------------- 3. GPTQ: `dynamic` the loaders honour

def _vllm_override(dyn, name, key="bits", default=4):
    """vLLM's get_dynamic_override (vllm/model_executor/layers/quantization/utils/gptq_utils.py, 0.29): first match."""
    for pattern, d in dyn.items():
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), name):
                return False
        elif re.match(pattern.removeprefix("+:"), name):
            return d.get(key, default)
    return default


def _bits(dyn, name):
    return G.module_bits({"bits": 4, "dynamic": dyn}, name)


def test_dense_dynamic_map_lands_the_same_bits_in_vllm_and_gptqmodel():
    al = {0: {"attn": 8, "ffn": 8}, 1: {"attn": 4, "ffn": 8}, 10: {"attn": 8, "ffn": 4}, 2: {"attn": 4, "ffn": 4}}
    dyn = E.dynamic_config(al, high=8)
    assert all(k.startswith("+:") for k in dyn), "spelled the way gptqmodel / vLLM document it"
    for i, g in al.items():
        p = f"model.layers.{i}"
        for hf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            assert _bits(dyn, f"{p}.{hf}") == g["attn"]
        for hf in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert _bits(dyn, f"{p}.{hf}") == g["ffn"]
        # vLLM builds FUSED modules and matches the rules against those names
        assert _vllm_override(dyn, f"{p}.self_attn.qkv_proj") == g["attn"]
        assert _vllm_override(dyn, f"{p}.self_attn.o_proj") == g["attn"]
        assert _vllm_override(dyn, f"{p}.mlp.gate_up_proj") == g["ffn"], "gate_up_proj missed: shape mismatch in vLLM"
        assert _vllm_override(dyn, f"{p}.mlp.down_proj") == g["ffn"]


def test_moe_dynamic_map_lands_the_same_bits_in_vllm_and_gptqmodel():
    al = {0: {"attn": 4, "ffn": 8}, 1: {"attn": 8, "ffn": 4}}
    dyn = E.moe_dynamic_config(al, high=8)
    for i, g in al.items():
        p = f"model.layers.{i}.mlp"
        for e in (0, 63):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert _bits(dyn, f"{p}.experts.{e}.{proj}") == g["ffn"]
        assert _vllm_override(dyn, f"{p}.experts") == g["ffn"], "vLLM's FusedMoE prefix is the layer's experts"
        assert _vllm_override(dyn, f"{p}.shared_expert.gate_up_proj") == 8
        assert _bits(dyn, f"{p}.shared_expert.gate_proj") == 8
    mix = E.moe_dynamic_config({0: {"attn": 4, "ffn": 8}}, high=8)
    assert _bits(mix, "model.layers.0.block_sparse_moe.experts.3.w1") == 8          # Mixtral naming too


def test_unprefixed_dynamic_rules_decode_at_their_bits():
    """The first qualification rung's config carried unprefixed rules (gptqmodel accepts them); the dequantizer used
    for verify only honoured `+:` ones and would have decoded every 8-bit module as 4-bit."""
    q = {"bits": 4, "dynamic": {r".*\.layers\.0\.self_attn\.[a-z_]*proj[a-z0-9_]*": {"bits": 8},
                                r"-:.*\.mlp\.gate$": {}}}
    assert G.module_bits(q, "model.layers.0.self_attn.q_proj") == 8
    assert G.module_bits(q, "model.layers.1.self_attn.q_proj") == 4
    assert G.module_bits(q, "model.layers.0.mlp.gate") is None


def test_the_card_says_which_loaders_honour_a_mixed_gptq_rung(tmp_path):
    import pollard_card as C
    mixed, uni = tmp_path / "GPTQ-int4", tmp_path / "GPTQ-int8"
    for d, dyn in ((mixed, E.dynamic_config({0: {"attn": 8, "ffn": 4}}, high=8)), (uni, {})):
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq", "bits": 4 if dyn else 8,
                                                                           "dynamic": dyn}}))
    assert C.gptq_is_mixed(str(mixed)) and not C.gptq_is_mixed(str(uni))
    sec = "\n".join(C.lane_folder_section("gptq", [{"path": str(mixed), "tag": "GPTQ-int4", "bytes": 1}], {}, "O/M-Pollard"))
    assert "from_pretrained" in sec and "vLLM and gptqmodel" in sec
    sec = "\n".join(C.lane_folder_section("gptq", [{"path": str(uni), "tag": "GPTQ-int8", "bytes": 1}], {}, "O/M-Pollard"))
    assert "from_pretrained" not in sec


def _need_torch():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")


ROWS = [list(np.random.default_rng(i).integers(0, 250, 40)) for i in range(4)]


@pytest.fixture(scope="module")
def tiny_gptq(tmp_path_factory):
    """A tiny llama quantized by Pollard's own GPTQ engine as a 4/8 mix, then given pollard-export's (gptqmodel's)
    `+:` dynamic map -- the 4/8 mix transformers' loader ignores."""
    _need_torch()
    import tiny_models as TM
    root = tmp_path_factory.mktemp("qual-gptq")
    src = TM.llama(str(root / "llama"), layers=3)
    out = str(root / "rung")
    alloc = {0: {"attn": 8, "ffn": 8}, 1: {"attn": 4, "ffn": 8}, 2: {"attn": 8, "ffn": 4}}
    G.build_stream(src, out, None, alloc=alloc, low=4, group=32, rows=ROWS, nsamples=4, seqlen=40,
                   trust_remote_code="off", log=lambda *_: None)
    cfgp = os.path.join(out, "config.json")
    cfg = json.load(open(cfgp))
    cfg["quantization_config"]["dynamic"] = E.dynamic_config(alloc, high=8)
    json.dump(cfg, open(cfgp, "w"), indent=2)
    return {"src": src, "rung": out, "alloc": alloc}


def test_decoded_loader_is_the_rung_at_its_stored_bits(tiny_gptq):
    import torch
    import pollard_stream as S
    from transformers import AutoModelForCausalLM
    m = S.decoded_hf_model(tiny_gptq["rung"], "cpu", False, dtype=torch.float32)
    # every 8-bit module really decodes at 8 bits: its decode is far closer to the source than a 4-bit one
    src = AutoModelForCausalLM.from_pretrained(tiny_gptq["src"], dtype=torch.float32).eval()
    sd, ref = m.state_dict(), src.state_dict()
    for L, g in tiny_gptq["alloc"].items():
        for side, key in (("attn", f"model.layers.{L}.self_attn.q_proj.weight"),
                          ("ffn", f"model.layers.{L}.mlp.up_proj.weight")):
            rel = float((sd[key] - ref[key]).norm() / ref[key].norm())
            assert (rel < 0.02) if g[side] == 8 else (0.02 < rel < 0.5), (key, g[side], rel)
    # and equals the reference dequantizer's state, tensor for tensor
    want = G.dequantized_state(tiny_gptq["rung"])
    for k, w in want.items():
        if k in sd:
            assert np.allclose(sd[k].numpy(), w, atol=1e-6), k
    ids = torch.tensor([ROWS[0]])
    with torch.no_grad():
        lq, lr = m(ids).logits[0].numpy(), src(ids).logits[0].numpy()
    kl = V.kld_top1(lr, lq)
    assert kl["kld"] < 0.05 and kl["top1"] > 80


def test_streamed_verify_and_decoded_loader_agree_on_the_mix(tiny_gptq):
    import torch
    import pollard_stream as S
    from transformers import AutoModelForCausalLM
    r = S.stream_fidelity(tiny_gptq["src"], tiny_gptq["rung"], ROWS[:2], trust_remote_code="off", log=lambda *_: None)
    m = S.decoded_hf_model(tiny_gptq["rung"], "cpu", False, dtype=torch.float32)
    src = AutoModelForCausalLM.from_pretrained(tiny_gptq["src"], dtype=torch.float32).eval()
    with torch.no_grad():
        lq = np.concatenate([m(torch.tensor([x])).logits[0].numpy() for x in ROWS[:2]])
        lr = np.concatenate([src(torch.tensor([x])).logits[0].numpy() for x in ROWS[:2]])
    d = V.kld_top1(lr, lq)
    assert r["n"] == d["n"] == 80
    assert r["kld"] == pytest.approx(d["kld"], rel=0.05, abs=1e-4)


def test_gptqmodel_loader_and_the_decoded_loader_give_the_same_logits(tiny_gptq, monkeypatch):
    """Where gptqmodel is installed (the box): its own loader -- the one that honours `dynamic`, as vLLM does -- and the
    decoded loader the tool gate now uses produce the same logits on the CPU."""
    pytest.importorskip("gptqmodel")
    import torch
    import torch._dynamo
    monkeypatch.setattr(torch._dynamo.config, "disable", True)     # its CPU kernel torch.compile()s (needs a C++ compiler)
    import pollard_stream as S
    from gptqmodel import GPTQModel
    g = GPTQModel.load(tiny_gptq["rung"], device="cpu")
    m = S.decoded_hf_model(tiny_gptq["rung"], "cpu", False, dtype=torch.float32)
    ids = torch.tensor([ROWS[1]])
    with torch.no_grad():
        a = g(ids).logits[0].float().numpy()
        b = m(ids).logits[0].float().numpy()
    assert V.kld_top1(a, b)["kld"] < 1e-3 and V.kld_top1(a, b)["top1"] == 100.0


def test_toolcall_hf_engine_loads_gptq_through_the_decoded_loader(tiny_gptq, monkeypatch):
    import pollard_stream as S
    import pollard_toolcall as TC
    with open(os.path.join(tiny_gptq["rung"], "chat_template.jinja"), "w") as f:
        f.write("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    seen = []
    real = S.decoded_hf_model
    monkeypatch.setattr(S, "decoded_hf_model", lambda *a, **k: seen.append(a[0]) or real(*a, **k))
    eng = TC.HFEngine(tiny_gptq["rung"], "cpu")
    assert seen == [tiny_gptq["rung"]], "from_pretrained would build every linear at 4 bits"
    eng.close()
    assert TC.decoded_gb(tiny_gptq["rung"]) > TC.model_gb(tiny_gptq["rung"]) * 0.9


# ----------------------------------------------------------------------------------- 4. EXL3 as a SYSTEM task

def test_exl3_env_as_system_finds_ninja_and_the_owners_extension_cache(monkeypatch):
    cache = r"C:\Users\jwate\AppData\Local\torch_extensions\torch_extensions\Cache"
    monkeypatch.setattr(R.os.path, "isdir", lambda p: p == cache)
    import glob as _glob
    monkeypatch.setattr(_glob, "glob", lambda pat: [cache] if "torch_extensions" in pat else [])
    base = {"USERNAME": "BOX$", "USERPROFILE": r"C:\Windows\system32\config\systemprofile",
            "Path": r"C:\Windows\system32;C:\Windows", "POLLARD_HOME": r"C:\Users\jwate\pollard"}
    env = R.exl3_env(base, exe=r"C:\pollard\venv312\Scripts\python.exe", platform_name="win32")
    assert env["Path"].split(";")[0] == r"C:\pollard\venv312\Scripts", "ninja lives in the venv's Scripts"
    assert env["TORCH_EXTENSIONS_DIR"] == cache
    assert base["Path"] == r"C:\Windows\system32;C:\Windows", "the caller's env is not mutated"
    # not twice on PATH; an explicit TORCH_EXTENSIONS_DIR wins; a normal user keeps torch's own default
    again = R.exl3_env(env, exe=r"C:\pollard\venv312\Scripts\python.exe", platform_name="win32")
    assert again["Path"].lower().count(r"venv312\scripts") == 1
    own = R.exl3_env(dict(base, TORCH_EXTENSIONS_DIR=r"D:\ext"), exe=r"C:\pollard\venv312\Scripts\python.exe",
                     platform_name="win32")
    assert own["TORCH_EXTENSIONS_DIR"] == r"D:\ext"
    user = R.exl3_env(dict(base, USERNAME="jwate", USERPROFILE=r"C:\Users\jwate"),
                      exe=r"C:\pollard\venv312\Scripts\python.exe", platform_name="win32")
    assert "TORCH_EXTENSIONS_DIR" not in user
    # POSIX: the venv's bin goes on PATH, nothing else changes
    nix = R.exl3_env({"PATH": "/usr/bin"}, exe="/opt/venv/bin/python", platform_name="linux")
    assert nix["PATH"] == "/opt/venv/bin:/usr/bin" and "TORCH_EXTENSIONS_DIR" not in nix


def test_exl3_steps_run_in_the_exl3_env(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,), env=None: ran.append((cmd[0], env)))
    monkeypatch.setattr(R, "exl3_env", lambda base=None, **k: {"EXL3": "1", **(base or {})})
    c = {"state": str(tmp_path / "state.json"), "mdir": str(tmp_path)}
    steps = [R.Step(id="exl3/4.0/build", lane="exl3", rung="4.0", kind="build", requires="any",
                    cmd=["pollard-exl3", "--bpw", "4.0"]),
             R.Step(id="gptq/int4/build", lane="gptq", rung="int4", kind="build", requires="any",
                    cmd=["pollard-export", "--bits", "4"])]
    R.execute(steps, c, {}, run=True, have={"any": True, "cuda": True}, out=lambda *_: None)
    got = dict(ran)
    assert got["pollard-exl3"] == {"EXL3": "1"} and got["pollard-export"] is None


# ------------------------------------------------------------------------------ 5. coherence: ik binary + budget

def test_coherence_serves_an_ik_only_rung_with_ik_llama_server(monkeypatch, tmp_path):
    import pollard_auto
    import pollard_bench as b
    ik = tmp_path / "ik" / "build" / "bin"
    ik.mkdir(parents=True)
    exe = ".exe" if os.name == "nt" else ""
    (ik / f"llama-server{exe}").write_text("")
    stock = tmp_path / "stock-llama-server"
    stock.write_text("")
    monkeypatch.setattr(b, "SERVER_BIN", None)
    monkeypatch.setattr(b, "find_llama_bin", lambda name: str(stock))
    monkeypatch.setattr(pollard_auto, "find_ik_build", lambda *_: str(tmp_path / "ik"))
    monkeypatch.setattr(pollard_auto, "_bin_in", lambda d, n: os.path.join(d, "build", "bin", n + exe))
    monkeypatch.setattr(b, "_needs_ik", lambda m: m.endswith("IQ1_KT.gguf"))
    assert b._server_bin("M-Pollard-IQ1_KT.gguf") == str(ik / f"llama-server{exe}")
    assert b._server_bin("M-Pollard-IQ3_S.gguf") == str(stock)
    assert b._server_bin() == str(stock)
    src = open(b.__file__, encoding="utf-8").read()
    served = src[src.index("def _served("):]
    assert "_server_bin(model)" in served[:served.index("cmd = [")], "the gate's server is picked per model"


def test_a_thinking_model_gets_the_tool_test_budget_in_the_coherence_gate():
    import pollard_modelkind as MK
    import pollard_toolcall as TC
    assert MK.TOKENS_THINKING == TC.N_PREDICT_REASONING == 1024
    assert MK.TOKENS_PLAIN < MK.TOKENS_THINKING


# ------------------------------------------------------------------------------------ 6. the release's card step

def _safetensors(path, shapes):
    hdr, off = {}, 0
    for k, shp in shapes.items():
        n = int(np.prod(shp)) * 2
        hdr[k] = {"dtype": "BF16", "shape": list(shp), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + b"\0" * off)


def test_card_step_names_the_hf_id_the_imatrix_and_the_real_param_count(tmp_path):
    src = tmp_path / "src" / "LFM2.5-24B-A2B"
    src.mkdir(parents=True)
    # a MoE whose dense config estimate is tiny next to what it holds
    (src / "config.json").write_text(json.dumps({"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100}))
    _safetensors(src / "model-00001-of-00002.safetensors", {"a": (1000, 1000), "b": (10,)})
    _safetensors(src / "model-00002-of-00002.safetensors", {"c": (500, 2000)})
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 4000020},
        "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                       "c": "model-00002-of-00002.safetensors"}}))
    assert R.source_params(str(src)) == 2_000_010
    import pollard_card as C
    assert C.parse_params_b(None, json.load(open(src / "config.json"))) < 0.01, "the estimate that went on the card"
    c = {"src": str(src), "hf_dir": str(src), "name": "LFM2.5-24B-A2B", "mdir": str(tmp_path / "m"),
         "release_dir": str(tmp_path / "m" / "release"), "f16": str(tmp_path / "f16.gguf"),
         "smooth_dir": str(src) + "-smoothed", "calib": str(tmp_path / "calib.txt"), "imatrix": str(tmp_path / "LFM2.5-24B-A2B-f16.imatrix"), "hf_repo": None}
    cfg = {"base_model": "LiquidAI/LFM2.5-24B-A2B"}
    cmd = R.card_cmd(cfg, c, ["gguf"])
    arg = lambda k: cmd[cmd.index(k) + 1]                      # noqa: E731
    assert arg("--base-model") == "LiquidAI/LFM2.5-24B-A2B", "never the local source path"
    assert arg("--imatrix-file") == "LFM2.5-24B-A2B.imatrix", "the name upload() ships it under"
    assert C.parse_params_b(arg("--params"), {}) == pytest.approx(0.002, abs=1e-3)
    # no imatrix in the plan: no claim
    assert "--imatrix-file" not in R.card_cmd(cfg, dict(c, imatrix=None), ["gguf"])


def test_base_model_id_from_the_plan_the_hf_id_the_config_or_the_workspace_name(tmp_path):
    d = tmp_path / "C__pollard__src"                           # a path, not an id
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}))
    assert R.base_model_id({"base_model": "Org/M"}, {"src": str(d), "hf_dir": str(d)}) == "Org/M"
    assert R.base_model_id({}, {"src": "Qwen/Qwen2.5-3B", "hf_dir": str(d)}) == "Qwen/Qwen2.5-3B"
    assert R.base_model_id({}, {"src": str(d), "hf_dir": str(d)}) == "Qwen/Qwen2.5-0.5B-Instruct"
    ws = tmp_path / "LiquidAI__LFM2.5-24B-A2B"
    ws.mkdir()
    (ws / "config.json").write_text(json.dumps({"_name_or_path": str(ws)}))     # a local path there: ignored
    assert R.base_model_id({}, {"src": str(ws), "hf_dir": str(ws)}) == "LiquidAI/LFM2.5-24B-A2B"
    bare = tmp_path / "LFM2.5-24B-A2B"
    bare.mkdir()
    assert R.base_model_id({}, {"src": str(bare), "hf_dir": str(bare)}) is None
    assert R.base_model_id({"base_model": str(bare)}, {"src": str(bare), "hf_dir": str(bare)}) is None


# ------------------------------------------------------------------------------- 7. tool gate on the CPU

def test_cpu_estimate_scales_with_the_decoded_size():
    import pollard_toolcall as TC
    lo, hi = TC.cpu_estimate(1.0, 60, 384, gbps=20)
    assert hi == pytest.approx(60 * 384 / 20 / 60) and lo == pytest.approx(hi / 4)
    assert TC.cpu_estimate(48.0, 60, 1024, gbps=20)[1] > 1000          # a 24B decoded: hours, and it says so
