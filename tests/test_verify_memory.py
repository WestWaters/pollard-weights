"""The streamed verify and the tool-call gate size memory from what they will actually HOLD (the Windows box, 32 GB
with ~15.5 GB available, 2026-10-19): pollard-verify takes --max-ram-gb (the release passes the plan's cap) and runs
the source pass and the rung pass one after the other under one plan -- bf16 blocks when fp32 does not fit, each
block freed as soon as every row went through it, the source's final states cached on disk; pollard-toolcall's CPU
fallback sizes a rung from its DECODED bf16 footprint, never from its packed files, and records not-measured instead
of starting what cannot fit. Tiny models and fakes only."""
from __future__ import annotations

import json
import os
import sys
import types
import weakref
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_stream as S  # noqa: E402


# ------------------------------------------------------------------------------- the plan (no torch)

class FakeSrc:
    def __init__(self, shapes):
        self.shapes = shapes

    def shape(self, k):
        return self.shapes[k]


def fake_backbone(layers=40, per_layer=600_000_000, hidden=2048, vocab=65536):
    """LFM2.5-24B-A2B-sized shapes: 40 blocks of ~0.6 B parameters (64 experts each), 2048 wide, 65k vocabulary."""
    shapes = {"model.embed_tokens.weight": (vocab, hidden), "model.norm.weight": (hidden,)}
    keys = []
    for i in range(layers):
        k = f"model.layers.{i}.mlp.experts.weight"
        shapes[k] = (per_layer // hidden, hidden)
        keys.append([k])
    tc = types.SimpleNamespace(vocab_size=vocab, hidden_size=hidden, num_attention_heads=32, head_dim=64)
    return types.SimpleNamespace(n_layers=layers, layer_keys=keys, src=FakeSrc(shapes),
                                 outer_keys=["model.embed_tokens.weight", "model.norm.weight"],
                                 kmap={"model.embed_tokens.weight": ("model.embed_tokens.weight", None)},
                                 hidden_size=lambda: hidden, text_config=lambda: tc)


def test_a_24b_moe_verify_plans_both_passes_under_12_gb():
    bb = fake_backbone()
    plan = S.verify_memory(bb, 16, 2048, 12.0, log=lambda *_: None)
    assert plan["peak_gb"] <= 12.0, "the protocol's 16 x 2048 rows, both passes, under the box's 12 GB cap"
    tight = S.verify_memory(bb, 16, 2048, 6.0, log=lambda *_: None)
    assert plan["compute_bytes"] == 4 and tight["compute_bytes"] == 2, "fp32 whenever it fits, else bf16 blocks"
    assert tight["peak_gb"] <= 6.0 < plan["peak_gb"]
    with pytest.raises(S.OutOfMemoryBudget, match="--max-ram-gb"):
        S.verify_memory(bb, 16, 2048, 2.0, log=lambda *_: None)


def test_on_cuda_the_block_lives_in_vram():
    bb = fake_backbone()
    cpu = S.verify_memory(bb, 16, 2048, 64.0, log=lambda *_: None)
    gpu = S.verify_memory(bb, 16, 2048, 64.0, device="cuda:0", log=lambda *_: None)
    assert gpu["load_gb"] < cpu["load_gb"] and gpu["vram_gb"] > 0 and cpu["vram_gb"] == 0


def test_the_budget_is_the_cap_not_a_third_of_it(monkeypatch):
    monkeypatch.setattr(S, "free_ram_gb", lambda: 15.5)
    assert S.ram_budget(12.0) == 12.0
    assert S.ram_budget(None) == pytest.approx(15.5 * 0.85)


# ------------------------------------------------------------------------------- pollard-verify / pollard-release

def test_pollard_verify_takes_max_ram_gb_and_hands_it_to_the_streamed_measure(monkeypatch, tmp_path):
    import pollard_verify as V
    seen = {}

    class Stop(Exception):
        pass

    def fake(model, source, eval_path, device="cpu", length=None, chunks=8, trust_remote_code="auto",
             max_ram_gb=None):
        seen.update(max_ram_gb=max_ram_gb, device=device)
        raise Stop
    monkeypatch.setattr(V, "measure_stream", fake)
    ev = tmp_path / "ev.txt"
    ev.write_text("x")
    monkeypatch.setattr(sys, "argv", ["pollard-verify", "--lane", "mx", "--model", str(tmp_path), "--source",
                                      str(tmp_path), "--eval", str(ev), "--stream", "on", "--device", "cpu",
                                      "--max-ram-gb", "12.5"])
    with pytest.raises(Stop):
        V.main()
    assert seen == {"max_ram_gb": 12.5, "device": "cpu"}


def test_the_release_passes_the_plans_ram_cap_to_every_streamed_verify(tmp_path, monkeypatch):
    import pollard_release as R
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    d = tmp_path / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    cfg = R.load_plan(None)
    cfg["max_ram_gb"] = 12
    _, _, steps = R.plan_steps(cfg, str(d))
    by = {s.id: s for s in steps}
    for sid in ("gptq/int4/verify", "mx/NVFP4/verify"):
        cmd = by[sid].cmd
        assert cmd[cmd.index("--max-ram-gb") + 1] == "12", sid
    cfg["max_ram_gb"] = None
    _, _, steps = R.plan_steps(cfg, str(d))
    assert "--max-ram-gb" not in {s.id: s for s in steps}["gptq/int4/verify"].cmd


# ------------------------------------------------------------------------------- a tiny model, RAM-capped

def _tiny_pair(tmp_path):
    torch = pytest.importorskip("torch")
    T = pytest.importorskip("transformers")
    torch.manual_seed(7)
    cfg = T.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                        tie_word_embeddings=False)
    m = T.LlamaForCausalLM(cfg).float().eval()
    src, rung = tmp_path / "src", tmp_path / "rung"
    m.save_pretrained(src)
    with torch.no_grad():
        for p in m.model.layers[1].parameters():
            p.add_(torch.randn_like(p) * 0.02)
    m.save_pretrained(rung)
    return str(src), str(rung)


ROWS = [[1, 5, 9, 17, 33, 2, 64, 99, 7, 12, 40, 3, 88, 21, 6, 15], [1, 44, 2, 9, 71, 30, 8, 5, 19, 120, 11, 4, 3, 60,
                                                                    13, 27]]


def test_the_two_passes_never_sit_in_memory_together(tmp_path, monkeypatch):
    src, rung = _tiny_pair(tmp_path)
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    full = S.stream_fidelity(src, rung, ROWS, starts=[8, 8], trust_remote_code="off", log=lambda *_: None)
    assert full["memory_plan"]["compute_bytes"] == 4 and 0 < full["kld"] < 0.5

    engines = weakref.WeakSet()
    loads = []
    real_init, real_load = S.Engine.__init__, S.Engine.load_layer

    def init(self, *a, **k):
        real_init(self, *a, **k)
        engines.add(self)

    def load(self, i, tensors=None, guard_gb=None):
        live = [e for e in engines if getattr(e, "kwargs", None)]
        assert live == [self], "one pass at a time: the other engine's recorded calls are gone"
        loaded = sum(1 for L in self.bb.layers for p in L.parameters() if not p.is_meta)
        assert loaded == 0, "every earlier block was freed as soon as all rows went through it"
        loads.append((i, isinstance(tensors, S._LazyRung)))
        return real_load(self, i, tensors, guard_gb)
    monkeypatch.setattr(S.Engine, "__init__", init)
    monkeypatch.setattr(S.Engine, "load_layer", load)
    # a RAM-capped planner: the cap decides bf16 blocks, the measurement still runs
    real_plan = S.verify_memory

    def capped(bb, nrows, seqlen, budget_gb, **kw):
        p = real_plan(bb, nrows, seqlen, 1e9, **kw)
        p["compute_bytes"] = 2
        return p
    monkeypatch.setattr(S, "verify_memory", capped)
    small = S.stream_fidelity(src, rung, ROWS, starts=[8, 8], trust_remote_code="off", log=lambda *_: None,
                              max_ram_gb=12.0)
    assert [i for i, _ in loads] == [0, 1, 2, 0, 1, 2], "the source's blocks, then the rung's"
    assert [lazy for _, lazy in loads] == [False] * 3 + [True] * 3, "the rung's tensors are read lazily"
    assert small["memory_plan"]["compute_bytes"] == 2
    assert small["n"] == full["n"] and small["kld"] == pytest.approx(full["kld"], rel=0.1, abs=2e-4)
    cache = os.path.join(str(tmp_path / "home"))
    left = [f for _, _, fs in os.walk(cache) for f in fs if f.startswith("verify-state-")]
    assert not left, "the source-state cache is deleted afterwards"


def test_the_cap_holds_the_verify_cleanly_when_nothing_fits(tmp_path, monkeypatch):
    src, rung = _tiny_pair(tmp_path)
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    with pytest.raises(S.OutOfMemoryBudget, match="streamed verify needs"):
        S.stream_fidelity(src, rung, ROWS, trust_remote_code="off", log=lambda *_: None, max_ram_gb=0.01)


# ------------------------------------------------------------------------------- pollard-toolcall's CPU fallback

def test_a_pollard_mx_rung_is_sized_decoded_and_never_started_when_that_does_not_fit(monkeypatch, tmp_path):
    import pollard_mxfp6
    import pollard_toolcall as TC
    d = tmp_path / "mx-rung"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["Lfm2MoeForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"\0" * 1000)
    monkeypatch.setattr(pollard_mxfp6, "read_meta", lambda p: {"modules": {"x": "mxfp4"}})
    monkeypatch.setattr(TC, "_has_module", lambda n: n in ("transformers", "torch"))
    monkeypatch.setattr(TC, "gpu_users", lambda *a: ["Game.exe"])          # the son's GPU: CPU fallback
    monkeypatch.setattr(TC, "gpu_free_gb", lambda *a: 15.0)
    monkeypatch.setattr(TC, "model_gb", lambda m: 13.0)                     # packed, on disk
    monkeypatch.setattr(TC, "decoded_gb", lambda m: 48.0)                   # bf16, what the loader holds
    monkeypatch.setattr(TC, "_free_ram_gb", lambda: 15.5)
    started = []
    monkeypatch.setattr(TC, "HFEngine", lambda *a, **k: started.append(a))
    with pytest.raises(TC.NotMeasured, match=r"48\.0GB model needs ~56\.2GB of RAM, 15\.5GB free"):
        TC.preflight("hf", str(d), kind={"tools": {"family": "hermes"}})
    assert not started
    monkeypatch.setattr(TC, "_free_ram_gb", lambda: 60.0)
    ok = TC.preflight("hf", str(d), kind={"tools": {"family": "hermes"}})
    assert ok["device"] == "cpu" and ok["loader"] == "decoded" and ok["decoded_gb"] == 48.0


def test_decoded_size_is_the_architectures_bf16_parameters_not_the_packed_files(tmp_path):
    pytest.importorskip("torch")
    T = pytest.importorskip("transformers")
    import pollard_toolcall as TC
    cfg = T.LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False)
    cfg.torch_dtype = "bfloat16"
    d = tmp_path / "rung"
    d.mkdir()
    cfg.save_pretrained(d)
    c = json.loads((d / "config.json").read_text())
    c["quantization_config"] = {"quant_method": "gptq", "bits": 4}
    (d / "config.json").write_text(json.dumps(c))
    (d / "model.safetensors").write_bytes(b"\0" * 10)                      # "packed": 10 bytes on disk
    with __import__("torch").device("meta"):
        n = sum(p.numel() for p in T.LlamaForCausalLM(cfg).parameters())
    assert TC.decoded_gb(str(d)) == pytest.approx(n * 2 / 1e9)
    empty = tmp_path / "unreadable"
    empty.mkdir()
    (empty / "model.safetensors").write_bytes(b"\0" * 1000)
    assert TC.decoded_gb(str(empty)) == pytest.approx(TC.model_gb(str(empty)) * 4), \
        "unknowable: bounded from above, never the packed size"
