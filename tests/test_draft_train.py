"""pollard-draft train: Pollard's own quant-aware drafter heads. Fakes only for the pipeline (no server, no trainer, no
GPU); the tensor paths (rung weights into an HF model, hidden-state capture) run on a tiny random model when torch +
transformers are installed and skip otherwise."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_draft as D  # noqa: E402
import pollard_draft_train as DT  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402

QWEN4B = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 2560, "intermediate_size": 9728,
          "num_hidden_layers": 36, "vocab_size": 151936, "tie_word_embeddings": True}
QWEN06 = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
          "num_hidden_layers": 28, "vocab_size": 151936, "tie_word_embeddings": True}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POLLARD_HEAVY_LOCK", raising=False)
    import pollard_convert
    monkeypatch.setattr(pollard_convert, "find_converter", lambda d=None: (None, "none here"))
    return tmp_path


# ------------------------------------------------------------------ when, and with what recipe

def test_needs_training_only_when_nothing_redistributable(tmp_path):
    assert DT.needs_training({"pick": None})[0]
    go, why = DT.needs_training({"pick": {"id": "AngelSlim/x", "license": "other", "redistributable": False}})
    assert go and "may not be redistributed" in why
    go, why = DT.needs_training({"pick": {"id": "z-lab/x", "license": "mit", "redistributable": True}})
    assert not go and "redistributable" in why
    assert DT.needs_training({"pick": {"id": "z-lab/x", "redistributable": True}}, prefer_own=True)[0]
    p = tmp_path / "drafter.json"
    p.write_text(json.dumps({"pick": None}))
    assert DT.needs_training(str(p))[0]
    assert DT.needs_training(str(tmp_path / "missing.json"))[0], "find never ran -> nothing found"


def test_layer_ids_match_speculators_default():
    assert DT.default_layer_ids(36) == [2, 18, 33]
    assert DT.default_layer_ids(28) == [2, 14, 25]


def test_recipe_scales_to_model_and_card():
    assert 3.5 < DT.params_b(QWEN4B) < 4.5 and DT.params_b(QWEN06) < 1.0
    big = DT.recipe("eagle3", QWEN4B, vram_gb=80)
    assert big["total_seq_len"] == 8192 and big["draft_vocab_size"] == 32000 and big["epochs"] == 10
    box = DT.recipe("eagle3", QWEN4B, vram_gb=15.5)               # RTX 5070 Ti, what nvidia-smi reports free
    assert box["fits"] and box["total_seq_len"] <= 2048 and box["vram_est_gb"] <= 15.5 * 0.9
    df = DT.recipe("dflash", QWEN4B, vram_gb=15.5)
    assert df["num_layers"] <= 3 and df["draft_vocab_size"] is None
    assert not df["fits"] and df["vram_est_gb"] > 15.5 * 0.9, "a DFlash head on a 4B target wants a bigger card"
    assert DT.recipe("dflash", QWEN4B, vram_gb=48)["fits"]
    sm = DT.recipe("eagle3", QWEN06, smoke=True, device="mps")
    assert sm["max_steps"] <= DT.SMOKE_MAX_STEPS and sm["samples"] <= 64 and "reasoning" not in sm["mix"]
    assert sm["attn_impl"] == "sdpa" and sm["loss_impl"] == "eager" and sm["num_workers"] == 0
    assert DT.recipe("eagle3", QWEN4B)["mix"]["agent"] >= DT.recipe("eagle3", QWEN4B)["mix"]["chat"], \
        "tool turns weigh as much as chat: that is where drafters are run"


def test_train_argv_is_the_cached_file_backend_run():
    rec = DT.recipe("eagle3", QWEN06, smoke=True, device="mps")
    argv = DT.train_argv("py", "eagle3", "/v", "/d", "/s", [2, 14, 25], rec)
    assert argv[:4] == ["py", "-m", "speculators", "train"]
    s = " ".join(argv)
    assert "--hidden-states-backend file" in s and "--on-missing raise" in s, "never calls a vLLM server"
    assert "--target-layer-ids 2 14 25" in s and "--verifier-name-or-path /v" in s
    assert "--max-steps 40" in s and "--draft-vocab-size 32000" in s and "--draft-attn-impl sdpa" in s
    d = " ".join(DT.train_argv("py", "dflash", "/v", "/d", "/s", [1], DT.recipe("dflash", QWEN4B, vram_gb=80)))
    assert "--draft-vocab-size" not in d and "--max-steps" not in d and "--draft-attn-impl" not in d


# ------------------------------------------------------------------ data

def test_prompt_records_mix_carries_tools_offline():
    recs = DT.prompt_records(12, {"chat": 1, "agent": 1, "codechat": 1}, offline=True)
    assert len(recs) == 12 and {r["domain"] for r in recs} == {"chat", "agent", "codechat"}
    agent = [r for r in recs if r["domain"] == "agent"]
    assert agent and all(r.get("tools") for r in agent) and any(
        m.get("tool_calls") for r in agent for m in r["messages"]), "agent turns keep their tool calls + results"


def test_regenerate_keeps_only_what_the_rung_said():
    recs = DT.prompt_records(6, {"chat": 1}, offline=True)
    calls = []

    def complete(messages, tools=None, thinking=None):
        calls.append(messages)
        if len(calls) % 3 == 0:
            raise RuntimeError("server hiccup")
        return {"role": "assistant", "content": f"rung answer {len(calls)}"}
    out, ok, dropped = DT.regenerate(recs, complete)
    assert ok == 4 and dropped == 2 and len(out) == 4
    assert all(r["messages"][-1]["content"].startswith("rung answer") for r in out), \
        "a failed request drops the record -- never trains on the dataset's reference answer"


class FakeTok:
    """Character-level 'tokenizer': one id per character."""
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": [ord(c) for c in text]}


class FakeRenderer:
    def __call__(self, messages, tools=None, thinking=None):
        return "".join(f"<{m['role']}>{m.get('content', '')}</>" for m in messages)

    def _try(self, messages, add_generation_prompt=False, tools=None, **kw):
        return self(messages) + ("<assistant>" if add_generation_prompt else "")

    def kwargs(self, thinking):
        return {}


def test_render_rows_masks_only_the_answer():
    recs = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}],
             "domain": "chat"},
            {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}]}]
    rows = DT.render_rows(recs, FakeRenderer(), FakeTok(), 4096)
    assert len(rows) == 1, "a turn with (almost) nothing to learn is dropped"
    r = rows[0]
    text = "".join(chr(i) for i, m in zip(r["input_ids"], r["loss_mask"]) if m)
    assert text == "hello there</>" and len(r["input_ids"]) == len(r["loss_mask"])
    short = DT.render_rows(recs[:1], FakeRenderer(), FakeTok(), 30)[0]
    assert len(short["input_ids"]) == 30 and sum(short["loss_mask"]) == 14, "history trimmed, the answer kept whole"
    assert DT.render_rows(recs[:1], FakeRenderer(), FakeTok(), 20) == [], "truncated past the answer: nothing to learn"


def test_hs_bytes():
    assert DT.hs_bytes([{"input_ids": [0] * 100}], 3, 2560) == 100 * 4 * 2560 * 2


# ------------------------------------------------------------------ safety

def test_preflight_refuses_what_does_not_belong_here():
    with pytest.raises(DT.Refused, match="smoke test only"):
        DT.preflight("mps", False, 0.6, None, 100)
    with pytest.raises(DT.Refused, match="1B target"):
        DT.preflight("mps", True, 4.0, 40, 24)
    with pytest.raises(DT.Refused, match="capped"):
        DT.preflight("cpu", True, 0.6, 5000, 24)
    DT.preflight("mps", True, 0.6, 40, 24, avail_ram=64)
    with pytest.raises(DT.Refused, match="not ours to take"):
        DT.preflight("cuda", False, 4.0, None, 8000, users=["Cyberpunk2077.exe"], free_vram=15.5)
    with pytest.raises(DT.Refused, match="no NVIDIA GPU"):
        DT.preflight("cuda", False, 4.0, None, 8000, users=[], free_vram=None)
    with pytest.raises(DT.Refused, match="VRAM"):
        DT.preflight("cuda", False, 4.0, None, 8000, need_vram_gb=20, users=[], free_vram=15.5)
    with pytest.raises(DT.Refused, match="hidden states need"):
        DT.preflight("cuda", False, 4.0, None, 8000, need_disk=200e9, free_disk=50e9, users=[], free_vram=15.5)
    with pytest.raises(DT.Refused, match="RAM"):
        DT.preflight("mps", True, 0.6, 40, 24, need_ram_gb=40, avail_ram=8)


def test_gpu_guard_honours_the_shared_ignore_list(monkeypatch):
    import pollard_toolcall as T
    table = ("| Processes:                                                                              |\n"
             "|    0   N/A  N/A      4242    C+G   C:\\Games\\Steam\\game.exe                    N/A      |\n"
             "|    0   N/A  N/A      5151    C+G   C:\\Apps\\Overlay\\overlay.exe                 N/A      |\n")
    users = T.gpu_users(table, "")
    assert "game.exe" in users and "overlay.exe" in users
    monkeypatch.setenv("POLLARD_GPU_IGNORE", "overlay")
    users = T.gpu_users(table, "")
    assert users == ["game.exe"], "the owner may ignore a desktop app -- a game still counts"
    with pytest.raises(DT.Refused, match="game.exe"):
        DT.gpu_guard(users=users, free=15.5)
    DT.gpu_guard(users=[], free=15.5)


def test_heavy_lock_waits_and_releases(tmp_path, monkeypatch):
    lock = str(tmp_path / "mac-heavy.lock")
    os.mkdir(lock)                                               # someone else's job holds it
    threading.Timer(0.3, lambda: os.rmdir(lock)).start()
    t0 = time.time()
    with DT.heavy_lock(lock, poll=0.05, log=None):
        assert os.path.isdir(lock) and time.time() - t0 >= 0.25
    assert not os.path.exists(lock)
    monkeypatch.setenv("POLLARD_HEAVY_LOCK", lock)
    with pytest.raises(RuntimeError):
        with DT.heavy_lock(poll=0.05, log=None):
            raise RuntimeError("boom")
    assert not os.path.exists(lock), "released on failure too"
    monkeypatch.delenv("POLLARD_HEAVY_LOCK")
    with DT.heavy_lock(log=None):                                # no lock configured: a no-op
        pass


def test_gpu_watch_stops_training_when_a_game_starts():
    class Proc:
        pid, done = 999, False

        def poll(self):
            return 0 if self.done else None

        def terminate(self):
            self.done = True
    seen = iter([[], ["python.exe"], ["game.exe"]])
    p = Proc()
    with DT.GpuWatch(p, every=0.01, users=lambda: next(seen, ["game.exe"])) as w:
        for _ in range(200):
            if p.done:
                break
            time.sleep(0.01)
    assert p.done and "game.exe" in w.reason and "resumes from checkpoint" in w.reason


# ------------------------------------------------------------------ checkpoints

def test_latest_checkpoint_and_finalize(tmp_path):
    s = tmp_path / "ckpt"
    for e in ("0", "1", "3"):
        (s / e).mkdir(parents=True)
        (s / e / "config.json").write_text(json.dumps({"speculators_config": {"verifier": {"name_or_path": "/v"}}}))
        (s / e / "model.safetensors").write_bytes(b"w" + e.encode())
    (s / "2").mkdir()                                            # an interrupted epoch: no config
    assert DT.latest_checkpoint(str(s)) == str(s / "3")
    os.symlink(s / "1", s / "checkpoint_best")
    assert DT.latest_checkpoint(str(s)) == os.path.realpath(s / "1")
    assert DT.latest_checkpoint(str(tmp_path / "none")) is None
    out = DT.finalize(str(s / "1"), str(tmp_path / "head"), "Qwen/Qwen3-4B", {"rung_tag": "Q4_K_M"})
    cfg = json.loads((Path(out) / "config.json").read_text())
    assert cfg["speculators_config"]["verifier"]["name_or_path"] == "Qwen/Qwen3-4B", \
        "the shipped head names the target, not the local verifier-head dir it trained against"
    assert json.loads((Path(out) / "pollard_train.json").read_text())["rung_tag"] == "Q4_K_M"
    assert (Path(out) / "model.safetensors").read_bytes() == b"w1"


# ------------------------------------------------------------------ the whole pipeline, faked

def _train_args(tmp_path, **kw):
    p = argparse.ArgumentParser()
    p.add_argument("--target")
    p.add_argument("--kind", default="eagle3")
    p.add_argument("--run", action="store_true")
    p.add_argument("--epochs", type=int)
    DT.add_args(p)
    a = p.parse_args(["--target", "Qwen/Qwen3-0.6B"])
    a.__dict__.update(kw)
    return a


def test_if_needed_skips_when_a_redistributable_head_exists(home, capsys):
    found = home / "drafter.json"
    found.write_text(json.dumps({"pick": {"id": "z-lab/Qwen3-4B-DFlash-b16", "license": "mit",
                                          "redistributable": True}}))
    a = _train_args(home, if_needed=str(found), run=True)
    assert DT.cmd_train(a) == 0
    assert "not training" in capsys.readouterr().out


def test_cmd_train_runs_every_stage_against_the_rung(home, monkeypatch, capsys):
    tdir = home / "Qwen3-0.6B"
    tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(QWEN06))
    rung = home / "Qwen3-0.6B-Q4_K_M.gguf"
    rung.write_bytes(b"x")
    bf16 = home / "Qwen3-0.6B-BF16.gguf"
    bf16.write_bytes(b"x")
    seen = {}

    class Ren:
        tok = FakeTok()
        rinfo = {"mode": "toggle"}

        def __init__(self, *a):
            pass

        def __call__(self, messages, tools=None, thinking=None):
            return FakeRenderer()(messages)

        def _try(self, *a, **k):
            return FakeRenderer()._try(*a, **k)

        def kwargs(self, thinking):
            seen.setdefault("thinking", set()).add(thinking)
            return {}
    import pollard_calib
    monkeypatch.setattr(pollard_calib, "Renderer", Ren)
    monkeypatch.setattr(DT, "pick_device", lambda want: "mps")
    monkeypatch.setattr(DT, "has_speculators", lambda py: True)
    monkeypatch.setattr(DT, "preflight", lambda *a, **k: seen.setdefault("preflight", (a, k)))

    def complete_for(a, renderer, rec):
        import contextlib
        seen["regen_by"] = a.rung
        return contextlib.nullcontext(lambda m, tools=None, thinking=None: {"role": "assistant",
                                                                           "content": "rung says " + m[-1]["role"]}), \
            "llama-server on rung"
    monkeypatch.setattr(DT, "_complete_for", complete_for)
    monkeypatch.setattr(DT, "write_dataset", lambda rows, d: seen.setdefault("rows", rows))
    monkeypatch.setattr(DT, "_load_target", lambda tdir, dev, g=None, trc=False: seen.setdefault("taps_from", g) or "M")

    def capture(rows, model, ids, hs, dev, progress=None):
        seen["capture"] = (len(rows), ids, dev)
    monkeypatch.setattr(DT, "capture", capture)
    monkeypatch.setattr(DT, "write_verifier_head", lambda m, src, out: seen.setdefault("verifier", out))
    monkeypatch.setattr(DT, "_free", lambda d: None)

    class Popen:
        def __init__(self, argv, **kw):
            seen["argv"] = argv
            ck = Path(argv[argv.index("--save-path") + 1]) / "0"
            ck.mkdir(parents=True)
            (ck / "config.json").write_text(json.dumps({"architectures": ["Eagle3Speculator"],
                                                        "speculators_model_type": "eagle3",
                                                        "speculators_config": {"verifier": {"name_or_path": "x"}}}))
            (ck / "model.safetensors").write_bytes(b"w")
            self.pid = 1

        def wait(self):
            return 0

        def poll(self):
            return 0
    monkeypatch.setattr(DT.subprocess, "Popen", Popen)
    monkeypatch.setattr(D, "cmd_convert", lambda ca: seen.setdefault("convert", ca) and 0)

    def convert(ca):
        seen["convert"] = ca
        g = os.path.join(ws.model_dir(ca.key, create=True), "draft", "h-Q4.gguf")
        os.makedirs(os.path.dirname(g), exist_ok=True)
        Path(g).write_bytes(b"g")
        D.record(ca.key, g, "EAGLE3-Q4_K_M", "gguf", "eagle3", ca.meta)
        return 0
    monkeypatch.setattr(D, "cmd_convert", convert)
    monkeypatch.setattr(D, "cmd_measure", lambda ma: seen.setdefault("measure", []).append(ma) or 0)
    a = _train_args(home, target_dir=str(tdir), rung=str(rung), rung_tag="Q4_K_M", smoke=True, run=True,
                    offline=True, measure_target=[f"BF16={bf16}"], work=str(home / "work"))
    assert DT.cmd_train(a) == 0
    out = capsys.readouterr().out
    assert seen["regen_by"] == str(rung) and seen["taps_from"] == str(rung), "answers AND taps from the rung"
    assert seen["capture"][1] == [2, 14, 25] and seen["capture"][2] == "mps"
    assert all(sum(r["loss_mask"]) for r in seen["rows"])
    argv = seen["argv"]
    assert argv[argv.index("--verifier-name-or-path") + 1] == seen["verifier"], "fit to the rung's own distribution"
    regen = [json.loads(x) for x in open(home / "work" / "regen.jsonl")]
    assert regen and all(r["answered_by"] == "Q4_K_M" and r["messages"][-1]["content"].startswith("rung says")
                         for r in regen)
    assert False in seen["thinking"], "a toggle model answers non-reasoning prompts with thinking off"
    ca = seen["convert"]
    assert ca.kind == "eagle3" and ca.no_gguf is False and ca.drafter.endswith("head")
    assert ca.meta["source"] == "pollard-trained" and ca.meta["license"] == "apache-2.0" and \
        ca.meta["quant_aware_vs"] == "Q4_K_M" and ca.meta["redistributable"] is True
    prov = json.loads(Path(ca.drafter, "pollard_train.json").read_text())
    assert prov["trainer"]["licence"] == "Apache-2.0" and prov["taps"] == "rung" and prov["smoke"] is True
    ma = seen["measure"][0]
    assert ma.target_tag == ["Q4_K_M", "BF16"] and ma.target == [str(rung), str(bf16)], \
        "acceptance vs the quantized rung AND vs bf16"
    assert "quantized rung Q4_K_M" in out and "Apache-2.0" in out


def test_cmd_train_plan_only_and_dflash_is_hf_only(home, monkeypatch, capsys):
    tdir = home / "T"
    tdir.mkdir()
    (tdir / "config.json").write_text(json.dumps(QWEN06))
    a = _train_args(home, target_dir=str(tdir), run=False, smoke=True)
    monkeypatch.setattr(DT, "pick_device", lambda want: "cpu")
    assert DT.cmd_train(a) == 0
    out = capsys.readouterr().out
    assert "plan only" in out and "NOT quant-aware" in out
    spec = {"architectures": ["DFlashSpeculator"], "speculators_model_type": "dflash",
            "transformer_layer_config": {"hidden_size": 1024}}
    assert D.gguf_convertible(spec, "dflash") is False, "llama.cpp's DFlash converter does not read speculators"
    assert D.classify(spec) == "dflash"


def test_rung_from_manifest(home):
    p = home / "M-Pollard-Q4_K_M.gguf"
    p.write_bytes(b"x")
    ws.record_build("/x/M-f16.gguf", "gguf", str(p), tag="Q4_K_M")
    assert DT.rung_from_manifest("/x/M-f16.gguf", "q4_k_m") == str(p)
    with pytest.raises(DT.Refused):
        DT.rung_from_manifest("/x/M-f16.gguf", "IQ2_XS")


# ------------------------------------------------------------------ cards + release

def _trained_build(key, tmp, tag="EAGLE3-Q4_K_M", fmt="gguf"):
    g = os.path.join(ws.model_dir(key, create=True), "draft", tag + (".gguf" if fmt == "gguf" else ""))
    os.makedirs(os.path.dirname(g), exist_ok=True)
    if fmt == "gguf":
        Path(g).write_bytes(b"x" * 10)
    else:
        os.makedirs(g, exist_ok=True)
    D.record(key, g, tag, fmt, "eagle3", DT.build_meta({"rung_tag": "Q4_K_M", "taps": "rung", "samples": 8000}))
    return g


def test_draft_card_says_pollard_trained_quant_aware(home):
    key = "Qwen/Qwen3-4B"
    _trained_build(key, home)
    res = {"EAGLE3-Q4_K_M": {"draft": {"kind": "eagle3", "source": "pollard-trained", "quant_aware_vs": "Q4_K_M",
                                       "targets": {"Q4_K_M": {"tau": 2.9, "accept_len": 1.9, "accept_rate": 0.63},
                                                   "f16": {"tau": 2.8, "accept_len": 1.8, "accept_rate": 0.6}}},
                             "verified": {"pass": True}}}
    md = D.draft_card(key, res, "PollardWeights/Qwen3-4B-Pollard-DRAFT", {"gguf": "PollardWeights/Qwen3-4B-Pollard"},
                      D.draft_builds(key), verified_only=False)
    assert "license: apache-2.0" in md and "- pollard-trained" in md and "- quant-aware" in md
    assert "Pollard-trained, quant-aware vs Q4_K_M" in md and "taps from that rung's own weights" in md
    assert "speculators (Apache-2.0)" in md and "github.com/vllm-project/speculators" in md
    assert "| EAGLE3-Q4_K_M | Q4_K_M | **2.90** |" in md and "| EAGLE3-Q4_K_M | f16 | **2.80** |" in md
    assert "base_model:\n- Qwen/Qwen3-4B\ntags" in md, "no third-party drafter in base_model"
    sec = "\n".join(D.card_section(res, "PollardWeights/Qwen3-4B-Pollard-DRAFT",
                                   {"gguf": "PollardWeights/Qwen3-4B-Pollard"}, lane="gguf"))
    assert "Pollard-trained, quant-aware vs Q4_K_M" in sec and "--spec-type draft-eagle3" in sec


def test_measure_meta_carries_the_training_provenance(home):
    g = _trained_build("Qwen/Qwen3-4B", home)
    m = D.build_meta(g)
    assert m["source"] == "pollard-trained" and m["quant_aware_vs"] == "Q4_K_M" and m["taps"] == "rung"
    assert m["trained_by"].startswith("speculators")


def test_convert_skips_when_the_release_trained_its_own(home, capsys):
    key = "Qwen/Qwen3-4B"
    a = argparse.Namespace(skip_if_trained=True, key=key, target=key)
    _trained_build(key, home)
    assert D.cmd_convert(a) == 0 and "nothing to convert" in capsys.readouterr().out


@pytest.fixture
def model(home):
    d = home / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


def test_release_train_step_is_cuda_quant_aware_and_conditional(model):
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"]["enabled"] = True
    cfg["lanes"]["draft"]["prefer_own"] = True
    cfg["lanes"]["draft"]["train"]["samples"] = 4000
    c, _, steps = R.plan_steps(cfg, model)
    by = {s.id: s for s in steps}
    ids = [s.id for s in steps]
    assert ids.index("draft/find") < ids.index("draft/train") < ids.index("draft/convert") < ids.index("draft/rungs")
    assert by["draft/find"].ok_codes == (0, 1), "nothing found is an answer, not a failed lane"
    t = by["draft/train"]
    assert t.requires == "cuda"
    cmd = t.cmd
    assert cmd[:2] == ["pollard-draft", "train"] and "--prefer-own" in cmd and "--run" in cmd
    assert cmd[cmd.index("--if-needed") + 1].endswith("drafter.json")
    assert cmd[cmd.index("--rung-tag") + 1] == "Q4_K_M" and cmd[cmd.index("--rung-key") + 1] == c["f16"]
    assert cmd[cmd.index("--kind") + 1] == "eagle3" and cmd[cmd.index("--taps") + 1] == "rung"
    assert cmd[cmd.index("--measure-target") + 1] == "f16=" + c["f16"] and cmd[cmd.index("--samples") + 1] == "4000"
    assert "--skip-if-trained" in by["draft/convert"].cmd
    plain = R.load_plan(None)
    plain["lanes"]["draft"]["enabled"] = True
    _, _, st2 = R.plan_steps(plain, model)
    assert "--prefer-own" not in {x for s in st2 if s.id == "draft/train" for x in s.cmd}


def test_release_measures_a_trained_head_against_f16_too(model):
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"]["enabled"] = True
    c = R._ctx(cfg, model)
    os.makedirs(c["gguf_dir"], exist_ok=True)
    q = os.path.join(c["gguf_dir"], "Tiny-1B-Pollard-Q4_K_M.gguf")
    Path(q).write_bytes(b"x")
    ws.record_build(c["f16"], "gguf", q, tag="Q4_K_M")
    _trained_build(c["src"], None)
    steps = R.draft_rung_steps(cfg, c, D.draft_builds(c["src"]))
    m = steps[0].cmd
    tags = [m[i + 1] for i, x in enumerate(m) if x == "--target-tag"]
    assert tags == ["Q4_K_M", "f16"]


def test_train_cli_routes_run_to_the_pipeline(monkeypatch, capsys):
    got = {}
    monkeypatch.setattr(DT, "cmd_train", lambda a: got.setdefault("a", a) and 0)
    assert D.main(["train", "--target", "Qwen/Qwen3-4B", "--rung", "q.gguf", "--run"]) == 0
    assert got["a"].rung == "q.gguf" and got["a"].kind == "eagle3"
    assert D.main(["train", "--target", "Qwen/Qwen3-4B"]) == 0
    assert "recipe only" in capsys.readouterr().out


# ------------------------------------------------------------------ tensor paths (tiny random model)

def _tiny_qwen3(tmp_path):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    cfg = tr.Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
    torch.manual_seed(0)
    return torch, tr, tr.Qwen3ForCausalLM(cfg).eval()


def test_capture_layout_is_what_speculators_reads(tmp_path):
    torch, _, model = _tiny_qwen3(tmp_path)
    st = pytest.importorskip("safetensors.torch")
    rows = [{"input_ids": [1, 5, 9, 33, 70], "loss_mask": [0, 0, 1, 1, 1]}, {"input_ids": [3, 4], "loss_mask": [0, 1]}]
    hs = tmp_path / "hs"
    assert DT.capture(rows, model, [0, 2], str(hs), "cpu", progress=None) == 2
    d = st.load_file(str(hs / "hs_0.safetensors"))
    assert d["token_ids"].tolist() == rows[0]["input_ids"]
    assert tuple(d["hidden_states"].shape) == (5, 3, 64) and d["hidden_states"].dtype == torch.bfloat16
    with torch.no_grad():
        out = model(torch.tensor([rows[0]["input_ids"]]), output_hidden_states=True)
        emb = model.get_input_embeddings()(torch.tensor([rows[0]["input_ids"]]))[0]
    assert torch.allclose(d["hidden_states"][:, 0].float(), emb.float(), atol=2e-2), "id 0 = the embeddings"
    assert torch.allclose(d["hidden_states"][:, 1].float(), out.hidden_states[2][0].float(), atol=2e-2), \
        "id k = the stream entering layer k (vLLM's aux-hidden-state convention)"
    final = d["hidden_states"][:, 2].float()
    normed = model.model.norm(final.unsqueeze(0))[0]
    assert torch.allclose(normed, out.hidden_states[-1][0].float(), atol=5e-2), "last slot = pre-norm final state"
    assert DT.capture(rows, model, [0, 2], str(hs), "cpu", progress=None) == 0, "resumable"


def test_rung_weights_round_trip_through_a_gguf(tmp_path):
    torch, tr, model = _tiny_qwen3(tmp_path)
    np = pytest.importorskip("numpy")
    import gguf
    path = str(tmp_path / "tiny.gguf")
    w = gguf.GGUFWriter(path, "qwen3")
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, 4)
    for name, p in model.state_dict().items():
        if name == "lm_head.weight":
            continue                                             # tied: the GGUF stores token_embd once
        g = tmap.get_name(name, try_suffixes=(".weight", ".bias"))
        w.add_tensor(g, p.detach().float().numpy() + np.float32(0.5))   # "the rung": every weight shifted
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    fresh = tr.Qwen3ForCausalLM(model.config).eval()
    n, miss = DT.load_rung_weights(fresh, path)
    assert not miss and n == len(fresh.state_dict())
    for k, v in model.state_dict().items():
        assert torch.allclose(fresh.state_dict()[k], v + 0.5), k
    with pytest.raises(DT.Refused):
        DT.load_rung_weights(fresh, path, state=("nosuch-arch", {}))


def test_unpermute_inverts_llama_converter_permute():
    np = pytest.importorskip("numpy")
    w = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    n_head = 2
    perm = w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
    assert (DT._unpermute(perm, n_head) == w).all()


def test_write_verifier_head_holds_embed_lm_head_norm(tmp_path):
    torch, _, model = _tiny_qwen3(tmp_path)
    st = pytest.importorskip("safetensors.torch")
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    (src / "model.safetensors.index.json").write_text("{}")
    out = DT.write_verifier_head(model, str(src), str(tmp_path / "ver"))
    d = st.load_file(os.path.join(out, "model.safetensors"))
    assert set(d) == {"embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    assert os.path.exists(os.path.join(out, "tokenizer.json")) and \
        not os.path.exists(os.path.join(out, "model.safetensors.index.json"))


def test_write_dataset_is_the_prepared_format(tmp_path):
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    rows = [{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1]}, {"input_ids": [5, 3], "loss_mask": [0, 1]}]
    d = DT.write_dataset(rows, str(tmp_path / "data"))
    ds = datasets.load_from_disk(d)
    assert ds.column_names == ["input_ids", "loss_mask", "seq_len"] and list(ds["seq_len"][:]) == [4, 2]
    freq = torch.load(os.path.join(d, "token_freq.pt"))
    assert freq == {3: 2, 4: 1}, "only supervised (assistant) tokens are counted"


# ------------------------------------------------------------------ the box recipe (not run here)

def _box():
    import importlib.util
    spec = importlib.util.spec_from_file_location("box_train", ROOT / "experiments" / "draft_train" / "box_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_box_recipe_is_cpu_answers_gpu_train_and_a_past_dated_task():
    B = _box()
    lines = B.schtasks_lines()
    assert "/ru SYSTEM" in lines[0] and "/sd 01/01/2026" in lines[0] and lines[1].startswith("schtasks /run")
    assert lines[2].startswith("schtasks /delete")
    a = argparse.Namespace(target="Qwen/Qwen3-4B", rung="q4.gguf", bf16="bf16.gguf", samples=8000, threads=19,
                           trainer_python="spec.exe", llama_server=None, max_steps=None, run=True)
    cmd = B.argv_for(a)
    assert cmd[0] == "train" and cmd[cmd.index("--kind") + 1] == "eagle3" and cmd[cmd.index("--ngl") + 1] == "0"
    assert cmd[cmd.index("--threads") + 1] == "19" and "--run" in cmd and "BF16=bf16.gguf" in cmd
    rec = DT.recipe("eagle3", QWEN4B, vram_gb=15.5)
    assert rec["fits"] and rec["total_seq_len"] == 2048


def test_box_refuses_without_nvidia(monkeypatch):
    B = _box()
    monkeypatch.setattr(B.shutil, "which", lambda n: None)
    with pytest.raises(SystemExit, match="nvidia-smi"):
        B.main(["--home", "/nonexistent"])
