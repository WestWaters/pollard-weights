"""pollard-draft: the speculative drafter lane. Fakes only -- no hub, no server, no model: the hub answers come from a
dict, llama-server's timings are literal JSON, commands are recorded."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_draft as D  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_verify as V  # noqa: E402
import pollard_workspace as ws  # noqa: E402

QWEN4B = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 2560,
          "num_hidden_layers": 36, "vocab_size": 151936}
ZLAB = {"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 5,
        "num_target_layers": 36, "vocab_size": 151936,
        "dflash_config": {"mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]}}
SPEC_DFLASH2 = {"architectures": ["DFlash2DraftModel"], "speculators_model_type": "dflash2",
                "transformer_layer_config": {"hidden_size": 2560, "vocab_size": 151936},
                "speculators_config": {"verifier": {"name_or_path": "Qwen/Qwen3-4B"}}}
EAGLE_RH = {"architectures": ["Eagle3Speculator"], "speculators_model_type": "eagle3", "draft_vocab_size": 32000,
            "transformer_layer_config": {"hidden_size": 2560, "vocab_size": 151936, "num_hidden_layers": 1},
            "speculators_config": {"verifier": {"name_or_path": "Qwen/Qwen3-4B"}}}
EAGLE_YUHUILI = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 1,
                 "vocab_size": 128256, "draft_vocab_size": 32000}
WRONG = {"architectures": ["DFlashDraftModel"], "hidden_size": 4096, "num_target_layers": 36, "vocab_size": 151936,
         "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]}}
SMALL = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28,
         "vocab_size": 151936}


def fake_hub(models):
    """A fetch() over {repo: {"config": ..., "license": ..., "downloads": ...}} answering the URLs pollard-draft asks."""
    def fetch(path):
        if "/resolve/main/config.json" in path:
            repo = path.split("huggingface.co/")[1].split("/resolve/")[0]
            if repo in models and models[repo].get("config") is not None:
                return models[repo]["config"]
            raise FileNotFoundError(path)
        if path.startswith("/api/models/"):
            repo = path[len("/api/models/"):]
            m = models.get(repo, {})
            return {"id": repo, "downloads": m.get("downloads", 0), "cardData": {"license": m.get("license")},
                    "siblings": [{"rfilename": f} for f in m.get("files", ["config.json", "model.safetensors"])]}
        if path.startswith("/api/models?"):
            import urllib.parse
            q = urllib.parse.parse_qs(path.split("?", 1)[1])
            term = q["search"][0].lower()
            return [{"id": r, "downloads": m.get("downloads", 0)} for r, m in models.items()
                    if term in r.lower()]
        raise AssertionError(path)
    return fetch


HUB = {
    "Qwen/Qwen3-4B": {"config": QWEN4B, "license": "apache-2.0", "downloads": 10**6},
    "Qwen/Qwen3-0.6B": {"config": SMALL, "license": "apache-2.0", "downloads": 10**7},
    "Qwen/Qwen3-1.7B": {"config": {**SMALL, "hidden_size": 2048}, "license": "apache-2.0"},
    "z-lab/Qwen3-4B-DFlash-b16": {"config": ZLAB, "license": "mit", "downloads": 13000},
    "RedHatAI/Qwen3-4B-speculator.dflash2": {"config": SPEC_DFLASH2, "license": "apache-2.0", "downloads": 300},
    "RedHatAI/Qwen3-4B-speculator.eagle3": {"config": EAGLE_RH, "license": "apache-2.0", "downloads": 90000},
    "someone/Qwen3-4B-DFlash-wrong": {"config": WRONG, "license": "mit", "downloads": 10**6},
    "AtomicChat/Qwen3-4B-DFlash-GGUF": {"config": None, "downloads": 50000, "files": ["x.gguf"]},
    "custom/Qwen3-4B_eagle3": {"config": {**EAGLE_RH}, "license": "other", "downloads": 10**5},
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    import pollard_convert                     # whatever converter this machine has must not steer the ranking
    monkeypatch.setattr(pollard_convert, "find_converter", lambda d=None: (None, "none here"))
    return tmp_path


# ------------------------------------------------------------------ what a checkpoint is

def test_classify_every_published_layout():
    assert D.classify(ZLAB) == "dflash"
    assert D.classify(SPEC_DFLASH2) == "dflash"
    assert D.classify({"architectures": ["Qwen3DSparkModel"]}) == "dspark"
    assert D.classify({"architectures": ["DSparkDraftModel"], "speculators_model_type": "dspark"}) == "dspark"
    assert D.classify(EAGLE_RH) == "eagle3"
    assert D.classify({"architectures": ["LlamaForCausalLMEagle3"]}) == "eagle3"     # AngelSlim
    assert D.classify(EAGLE_YUHUILI) == "eagle3", "yuhuili: plain Llama arch, one layer + a draft vocab"
    assert D.classify(QWEN4B) is None, "an ordinary model is a `simple` candidate, not a head"


def test_compat_checks_the_numbers_the_runtime_needs():
    assert D.compat(ZLAB, QWEN4B, "dflash")[0]
    ok, problems, _ = D.compat(WRONG, QWEN4B, "dflash")
    assert not ok and any("hidden 4096" in p for p in problems)
    ok, problems, _ = D.compat({**ZLAB, "num_target_layers": 28}, QWEN4B, "dflash")
    assert not ok and any("28-layer" in p for p in problems)
    ok, problems, _ = D.compat({**ZLAB, "vocab_size": 128256}, QWEN4B, "dflash")
    assert not ok and any("vocab" in p for p in problems)
    ok, _, notes = D.compat(EAGLE_RH, QWEN4B, "eagle3")
    assert ok and any("draft vocab 32000" in n for n in notes) and any("Qwen/Qwen3-4B" in n for n in notes)
    assert not D.compat(EAGLE_YUHUILI, QWEN4B, "eagle3")[0]
    assert D.compat(SMALL, QWEN4B, "simple")[0]
    assert not D.compat(QWEN4B, SMALL, "simple")[0], "a drafter bigger than its target is no drafter"
    # a VLM target keeps its LM shape under text_config
    vlm = {"architectures": ["X"], "text_config": QWEN4B}
    assert D.compat(ZLAB, vlm, "dflash")[0]


def test_gguf_convertibility_follows_llama_cpp_converter():
    assert D.gguf_convertible(ZLAB, "dflash") is True
    assert D.gguf_convertible(SPEC_DFLASH2, "dflash") is False, "DFlash converter does not flatten speculators"
    assert D.gguf_convertible({"architectures": ["DSparkDraftModel"], "transformer_layer_config": {}}, "dspark")
    assert D.gguf_convertible(EAGLE_RH, "eagle3") is True
    assert D.gguf_convertible(EAGLE_YUHUILI, "eagle3") is True


def test_name_score():
    assert D.name_score("z-lab/Qwen3-4B-DFlash-b16", "Qwen/Qwen3-4B") == 2
    assert D.name_score("RedHatAI/Qwen3-4B-speculator.dflash2", "Qwen/Qwen3-4B") == 2
    assert D.name_score("AngelSlim/Qwen3-4B_eagle3", "Qwen/Qwen3-4B") == 2
    assert D.name_score("taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3", "Qwen/Qwen3-4B") == 1
    assert D.name_score("z-lab/Qwen3-8B-DFlash-b16", "Qwen/Qwen3-4B") == 0


# ------------------------------------------------------------------ find

def test_find_picks_the_compatible_redistributable_gguf_able_drafter(home, monkeypatch):
    res = D.find("Qwen/Qwen3-4B", fetch=fake_hub(HUB))
    ids = [c["id"] for c in res["candidates"]]
    assert res["pick"]["id"] == "z-lab/Qwen3-4B-DFlash-b16"
    assert res["pick"]["redistributable"] and res["pick"]["gguf"] is True
    by = {c["id"]: c for c in res["candidates"]}
    assert not by["someone/Qwen3-4B-DFlash-wrong"]["ok"], "downloads never outrank a shape mismatch"
    assert not by["AtomicChat/Qwen3-4B-DFlash-GGUF"]["ok"]
    assert by["custom/Qwen3-4B_eagle3"]["ok"] and not by["custom/Qwen3-4B_eagle3"]["redistributable"]
    assert ids.index("z-lab/Qwen3-4B-DFlash-b16") < ids.index("RedHatAI/Qwen3-4B-speculator.dflash2"), \
        "a drafter llama.cpp can convert ranks before one it cannot"
    simple = [c for c in res["candidates"] if c["kind"] == "simple"]
    assert simple and simple[0]["id"] == "Qwen/Qwen3-0.6B", "same family, < 1/3 the size"
    assert res["warning"] is None


def test_find_pinned_drafter_and_unknown_target_warns(home):
    res = D.find("Qwen/Qwen3-4B", fetch=fake_hub(HUB), drafter="RedHatAI/Qwen3-4B-speculator.eagle3")
    assert [c["id"] for c in res["candidates"]] == ["RedHatAI/Qwen3-4B-speculator.eagle3"]
    assert res["pick"]["kind"] == "eagle3"
    gated = D.find("meta/Gated-8B", fetch=fake_hub({}), drafter="RedHatAI/Qwen3-4B-speculator.eagle3")
    assert gated["warning"] and "NOT checked" in gated["warning"]
    # a pinned ordinary model is assessed as a `simple` drafter
    res = D.find("Qwen/Qwen3-4B", fetch=fake_hub(HUB), drafter="Qwen/Qwen3-0.6B")
    assert res["pick"]["kind"] == "simple"


def test_find_cli_writes_drafter_json(home, monkeypatch, capsys):
    monkeypatch.setattr(D, "hf_json", fake_hub(HUB))
    out = home / "drafter.json"
    assert D.main(["find", "--target", "Qwen/Qwen3-4B", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["pick"]["id"] == "z-lab/Qwen3-4B-DFlash-b16"
    assert "pick: z-lab/Qwen3-4B-DFlash-b16 (dflash)" in capsys.readouterr().out
    monkeypatch.setattr(D, "hf_json", fake_hub({"Qwen/Qwen3-4B": HUB["Qwen/Qwen3-4B"]}))
    assert D.main(["find", "--target", "Qwen/Qwen3-4B", "--no-simple"]) == 1, "nothing fits -> exit 1"


# ------------------------------------------------------------------ convert

def test_convert_cmds_bf16_once_then_quantize_each_rung():
    steps = D.convert_cmds("/d", "/t", "dflash", "/out", "Qwen/Qwen3-4B", ["Q8_0", "Q4_K_M"], "py", "conv.py", "q")
    assert [s[0] for s in steps] == ["DFlash-BF16", "DFlash-Q8_0", "DFlash-Q4_K_M"]
    conv = steps[0][2]
    assert conv[:3] == ["py", "conv.py", "/d"] and conv[conv.index("--target-model-dir") + 1] == "/t"
    assert conv[conv.index("--outtype") + 1] == "bf16"
    assert steps[2][1].endswith("Qwen3-4B-Pollard-DRAFT-DFlash-Q4_K_M.gguf")
    assert steps[2][2] == ["q", steps[0][1], steps[2][1], "Q4_K_M"]
    simple = D.convert_cmds("/d", None, "simple", "/out", "Qwen/Qwen3-8B", ["Q8_0"], "py", "c", "q", "Qwen3-0.6B")
    assert "--target-model-dir" not in simple[0][2], "a whole small model carries its own tokenizer"
    assert simple[1][1].endswith("Qwen3-8B-Pollard-DRAFT-Qwen3-0.6B-Q8_0.gguf")


def test_stage_hf_copies_or_converts(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(EAGLE_YUHUILI))
    (src / "model.safetensors").write_bytes(b"x")
    (src / "README.md").write_text("card")
    (src / "pic.png").write_bytes(b"x")
    dest, how = D.stage_hf(str(src), str(tmp_path / "hf"), "dflash")
    assert how == "copied as published"
    assert sorted(os.listdir(dest)) == ["config.json", "model.safetensors"], "weights + config, not someone's card"
    ran = []

    class R0:
        returncode = 0
    dest, how = D.stage_hf(str(src), str(tmp_path / "hf2"), "eagle3", "Qwen/Qwen3-4B", "speculators",
                           run=lambda c: ran.append(c) or R0())
    assert how == "speculators convert" and ran[0][:2] == ["speculators", "convert"]
    assert ran[0][ran[0].index("--verifier") + 1] == "Qwen/Qwen3-4B"


def test_convert_records_rungs_in_the_manifest(home, monkeypatch):
    ddir = home / "drafter"
    ddir.mkdir()
    (ddir / "config.json").write_text(json.dumps(ZLAB))
    (ddir / "model.safetensors").write_bytes(b"w" * 10)
    pick = home / "drafter.json"
    pick.write_text(json.dumps({"pick": {"id": str(ddir), "kind": "dflash", "license": "mit"}}))
    import pollard_convert
    import pollard_calc
    monkeypatch.setattr(pollard_convert, "find_converter", lambda d: ("/conv.py", "ok"))
    monkeypatch.setattr(pollard_calc, "find_llama_bin", lambda n, arch=None: "/bin/" + n)
    ran = []

    def fake_run(cmd, *a, **k):
        ran.append(cmd)
        out = cmd[cmd.index("--outfile") + 1] if "--outfile" in cmd else cmd[2]
        Path(out).write_bytes(b"g" * 5)

        class R0:
            returncode = 0
        return R0()
    monkeypatch.setattr(D.subprocess, "run", fake_run)
    assert D.main(["convert", "--target", "Qwen/Qwen3-4B", "--target-dir", str(ddir), "--from", str(pick)]) == 0
    builds = {b["tag"]: b for b in D.draft_builds("Qwen/Qwen3-4B")}
    assert set(builds) == {"DFlash-BF16", "DFlash-Q8_0", "DFlash-Q4_K_M", "DFlash-hf"}
    assert builds["DFlash-BF16"]["publish"] is False, "the BF16 GGUF is the intermediate, not a rung"
    assert builds["DFlash-Q4_K_M"]["redistributable"] is True and builds["DFlash-Q4_K_M"]["license"] == "mit"
    assert builds["DFlash-hf"]["format"] == "hf" and builds["DFlash-hf"]["via"].endswith("DFlash-BF16.gguf")
    assert all(b["lane"] == "draft" for b in builds.values())
    assert len(ran) == 3


# ------------------------------------------------------------------ measure

def test_stats_pool_tokens_not_prompt_ratios():
    rows = [{"kind": "chat", "n": 100, "drafted": 70, "accepted": 50, "ms": 1000, "tps": 100},
            {"kind": "code", "n": 20, "drafted": 7, "accepted": 0, "ms": 500, "tps": 40}]
    s = D.summarise(rows)
    assert s["tau"] == round(120 / 70, 3) and s["accept_len"] == round(50 / 70, 3)
    assert s["accept_rate"] == round(50 / 77, 4) and s["tok_s"] == 80.0
    assert s["by_kind"]["chat"]["tau"] == 2.0 and s["by_kind"]["code"]["tau"] == 1.0
    assert D.stats([{"kind": "x", "n": 0, "drafted": 0, "accepted": 0}]) is None


def test_row_from_llama_server_timings():
    t = {"predicted_n": 128, "predicted_ms": 2000.0, "predicted_per_second": 64.0, "draft_n": 120,
         "draft_n_accepted": 87}
    r = D.row_from_timings({"id": "p", "kind": "tool"}, t)
    assert (r["n"], r["drafted"], r["accepted"], r["tps"]) == (128, 120, 87, 64.0)
    assert D.row_from_timings({"id": "p", "kind": "chat"}, {"predicted_n": 5})["drafted"] == 0


def test_prompt_set_covers_agent_turns_and_limit_spreads_kinds():
    kinds = {p["kind"] for p in D.PROMPTS}
    assert kinds == {"chat", "code", "reasoning", "tool"}
    assert any(m.get("role") == "tool" for p in D.PROMPTS for m in p["messages"]), "a turn after a tool result"
    assert {p["kind"] for p in D.pick_prompts(4)} == kinds
    body = D.request_body(D.PROMPTS[-1], 64)
    assert body["temperature"] == 0 and body["tools"] and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_server_cmd_per_kind():
    c = D.gguf_server_cmd("srv", "t.gguf", "d.gguf", "dflash", 9, ngl="0")
    assert c[c.index("--spec-type") + 1] == "draft-dflash" and c[c.index("--spec-draft-n-max") + 1] == "7"
    assert c[c.index("-md") + 1] == "d.gguf" and c[c.index("-ngld") + 1] == "0", "drafter follows -ngl (shared box)"
    e = D.gguf_server_cmd("srv", "t.gguf", "d.gguf", "eagle3", 9, n_max=4, ngld="all")
    assert e[e.index("--spec-type") + 1] == "draft-eagle3" and e[e.index("--spec-draft-n-max") + 1] == "4"
    assert "-md" not in D.gguf_server_cmd("srv", "t.gguf", None, "dflash", 9), "the plain baseline"


def test_vllm_and_mlx_counters():
    before = D.vllm_counters('vllm:spec_decode_num_accepted_tokens_total{model_name="x"} 10\n'
                             'vllm:spec_decode_num_draft_tokens_total{model_name="x"} 30\n'
                             'vllm:spec_decode_num_drafts_total{model_name="x"} 10\n')
    after = D.vllm_counters('vllm:spec_decode_num_accepted_tokens_total{model_name="x"} 110\n'
                            'vllm:spec_decode_num_draft_tokens_total{model_name="x"} 230\n'
                            'vllm:spec_decode_num_drafts_total{model_name="x"} 60\n')
    s = D.vllm_stats(before, after)
    assert s == {"tau": 3.0, "accept_len": 2.0, "accept_rate": 0.5}
    assert D.vllm_stats(after, after) is None
    assert D.mlx_tokens_per_cycle({"recent_requests": [{"tokens_per_cycle": 3.5}, {"x": {"tokens_per_cycle": 4}}]}) \
        == [3.5, 4.0]
    assert D.vllm_spec_config("dflash", "org/d", 7) == {"method": "dflash", "model": "org/d",
                                                         "num_speculative_tokens": 7}


def _fake_measure(rows_by_target):
    def fake(target, drafter, kind, prompts, *a, **k):
        if rows_by_target[target] is None:
            raise D.NotMeasured("needs ~9.0GB of RAM, 3.0GB free")
        return {"spec": rows_by_target[target]}
    return fake


def test_measure_records_worst_pairing_and_not_measured(home, monkeypatch):
    good = [{"id": "a", "kind": "chat", "n": 100, "drafted": 90, "accepted": 60, "ms": 1000, "tps": 100}]
    weak = [{"id": "a", "kind": "chat", "n": 100, "drafted": 90, "accepted": 40, "ms": 1000, "tps": 100}]
    monkeypatch.setattr(D, "measure_gguf", _fake_measure({"q6.gguf": good, "iq3.gguf": weak, "big.gguf": None}))
    res = str(home / "results-draft.json")
    args = ["measure", "--drafter", "d.gguf", "--kind", "dflash", "--rung", "DFlash-Q4_K_M", "--results", res,
            "--redistributable", "true", "--hw", "M4"]
    assert D.main(args + ["--target", "q6.gguf", "--target-tag", "Q6_K", "--target", "iq3.gguf",
                          "--target-tag", "IQ3_S"]) == 0
    d = json.load(open(res))["DFlash-Q4_K_M"]["draft"]
    assert set(d["targets"]) == {"Q6_K", "IQ3_S"}, "a second target merges, never replaces"
    assert d["targets"]["Q6_K"]["tau"] == 2.5 and d["tau"] == round(100 / 60, 3), "headline = worst pairing"
    assert d["kind"] == "dflash" and d["n_max"] == 7 and d["hw"] == "M4" and d["redistributable"] is True
    assert D.main(args + ["--target", "big.gguf", "--target-tag", "Q8_0"]) == 0, "not measured is not a crash"
    d = json.load(open(res))["DFlash-Q4_K_M"]["draft"]
    assert d["targets"]["Q8_0"]["status"] == "not-measured" and "RAM" in d["targets"]["Q8_0"]["reason"]
    assert d["tau"] == round(100 / 60, 3) and set(d["targets"]) == {"Q6_K", "IQ3_S", "Q8_0"}


def test_measure_only_not_measured_fails_the_gate(home, monkeypatch):
    monkeypatch.setattr(D, "measure_gguf", _fake_measure({"t.gguf": None}))
    res = str(home / "r.json")
    D.main(["measure", "--drafter", "d.gguf", "--kind", "eagle3", "--rung", "EAGLE3-Q8_0", "--results", res,
            "--target", "t.gguf", "--target-tag", "Q6_K", "--redistributable", "true"])
    r = json.load(open(res))
    assert r["EAGLE3-Q8_0"]["draft"]["status"] == "not-measured"
    gates = V.load_gates(None)
    th = V.thresholds_for(gates, "EAGLE3-Q8_0", "draft")
    g = V.evaluate_gate(V.metrics_from_results(r, "EAGLE3-Q8_0"), th)
    assert not g["pass"] and [c["check"] for c in g["checks"] if not c["ok"]] == ["drafter tokens per target pass"]


# ------------------------------------------------------------------ gate

def test_draft_gate_is_acceptance_and_licence_not_kld():
    gates = V.load_gates(None)
    th = V.thresholds_for(gates, "DFlash-mlx-8bit", "draft", ["agentic"])
    assert th["draft_tau_min"] == 1.5 and th["draft_redistributable"] is True
    assert not ({"kld_max", "top1_min", "toolcall_retention_min", "runtime", "coherence"} & set(th)), \
        "bit-width / tool-call rows are about target rungs"
    ok = {"r": {"draft": {"tau": 3.1, "accept_rate": 0.6, "redistributable": True}}}
    assert V.evaluate_gate(V.metrics_from_results(ok, "r"), th)["pass"]
    slow = {"r": {"draft": {"tau": 1.2, "redistributable": True}}}
    assert not V.evaluate_gate(V.metrics_from_results(slow, "r"), th)["pass"]
    closed = {"r": {"draft": {"tau": 3.1, "redistributable": False}}}
    g = V.evaluate_gate(V.metrics_from_results(closed, "r"), th)
    assert not g["pass"] and g["checks"][-1]["check"].startswith("drafter licence")


def test_verify_cli_draft_lane_runs_nothing(home, monkeypatch, capsys):
    res = home / "r.json"
    res.write_text(json.dumps({"DFlash-Q8_0": {"draft": {"tau": 3.0, "redistributable": True}}}))
    monkeypatch.setattr(sys, "argv", ["pollard-verify", "--lane", "draft", "--model", str(home / "d.gguf"),
                                      "--rung", "DFlash-Q8_0", "--results", str(res), "--gate"])
    V.main()
    assert "GATE PASSED" in capsys.readouterr().out
    assert json.loads(res.read_text())["DFlash-Q8_0"]["verified"]["pass"] is True


# ------------------------------------------------------------------ cards + serve lines

RESULTS = {"DFlash-Q4_K_M": {"draft": {"kind": "dflash", "source": "z-lab/Qwen3-4B-DFlash-b16", "hw": "Apple M4",
                                       "tau": 3.0, "targets": {"Q6_K": {"tau": 3.11, "accept_len": 2.11,
                                                                       "accept_rate": 0.31, "tok_s": 61.4,
                                                                       "by_kind": {"chat": {"tau": 1.9}}}}},
                             "verified": {"pass": True}},
           "DFlash-Q8_0": {"draft": {"kind": "dflash", "targets": {"Q6_K": {"status": "not-measured",
                                                                            "reason": "no RAM"}}},
                           "verified": {"pass": False}}}


def test_serve_lines_per_kind():
    t = {"gguf": "P/M-Pollard", "gptq": "P/M-Pollard-GPTQ", "mlx": "P/M-Pollard-MLX", "base": "Q/M"}
    d = D.serve_lines("dflash", t, "P/M-Pollard-DRAFT")
    assert "--spec-type draft-dflash --spec-draft-n-max 7" in d["llama.cpp"]
    assert '"method": "dflash"' in d["vLLM"] and "P/M-Pollard-GPTQ" in d["vLLM"]
    assert "dflash serve --model P/M-Pollard-MLX --draft P/M-Pollard-DRAFT" in d["MLX"]
    assert "--speculative-algorithm DFLASH" in d["SGLang"]
    e = D.serve_lines("eagle3", t, "P/D")
    assert "draft-eagle3" in e["llama.cpp"] and '"method": "eagle3"' in e["vLLM"] and "no MLX runtime" in e["MLX"]
    s = D.serve_lines("simple", t, "P/D")
    assert "--draft-model P/D" in s["MLX"] and '"method": "draft_model"' in s["vLLM"] and "SGLang" not in s


def test_target_card_section_shows_measured_and_this_lanes_line():
    sec = "\n".join(D.card_section(RESULTS, "P/M-Pollard-DRAFT", {"gguf": "P/M-Pollard", "mlx": "P/M-MLX"}, "gguf"))
    assert "## Speculative decoding" in sec and "z-lab/Qwen3-4B-DFlash-b16" in sec
    assert "| DFlash-Q4_K_M | Q6_K | **3.11** |" in sec and "Apple M4" in sec
    assert "DFlash-Q8_0" not in sec, "a rung that failed its gate is not quoted"
    assert "**llama.cpp**" in sec and "**MLX**" not in sec, "only this lane's runtime"
    assert D.card_section({}, "x", {}) == []


def test_draft_repo_card(home):
    builds = [{"tag": "DFlash-Q4_K_M", "name": "M-Pollard-DRAFT-DFlash-Q4_K_M.gguf", "format": "gguf",
               "bytes": 3.2e8, "kind": "dflash", "source": "z-lab/Qwen3-4B-DFlash-b16", "license": "mit"},
              {"tag": "DFlash-Q8_0", "name": "M-Pollard-DRAFT-DFlash-Q8_0.gguf", "format": "gguf", "bytes": 5.8e8},
              {"tag": "DFlash-BF16", "format": "gguf", "publish": False}]
    md = D.draft_card("Qwen/Qwen3-4B", RESULTS, "P/Qwen3-4B-Pollard-DRAFT",
                      {"gguf": "P/Qwen3-4B-Pollard", "base": "Qwen/Qwen3-4B"}, builds)
    assert md.startswith("---\nlicense: mit\n") and "- z-lab/Qwen3-4B-DFlash-b16" in md
    assert "M-Pollard-DRAFT-DFlash-Q4_K_M.gguf" in md and "DFlash-Q8_0.gguf" not in md, "verified rungs only"
    assert "-md M-Pollard-DRAFT-DFlash-Q4_K_M.gguf" in md and "## Measured acceptance" in md
    assert "BF16" not in md


def test_train_recipe_is_licence_clean_and_names_the_quantized_rung():
    cmds = D.train_recipe("Qwen/Qwen3-4B", "dflash", regen_url="http://box:8080")
    assert cmds[0].startswith("pip install speculators") and "Apache-2.0" in cmds[0]
    assert any("regenerate-responses" in c and "http://box:8080/v1/chat/completions" in c for c in cmds)
    assert any(c.startswith("speculators train") and "--speculator-type dflash" in c for c in cmds)
    assert cmds[-1].startswith("pollard-draft convert")
    assert "--draft-vocab-size 32000" in " ".join(D.train_recipe("M", "eagle3"))
    assert D.TRAINERS["speculators"]["licence"] == "Apache-2.0" and D.TRAINERS["specforge"]["licence"] == "MIT"


def test_train_prints_without_run(capsys):
    assert D.main(["train", "--target", "Qwen/Qwen3-4B"]) == 0
    assert "recipe only" in capsys.readouterr().out


# ------------------------------------------------------------------ release wiring

@pytest.fixture
def model(home):
    d = home / "Tiny-1B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}))
    return str(d)


ALL = {"any": True, "apple": True, "cuda": True}


def test_release_draft_lane_order_and_cards(model):
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"]["enabled"] = True
    c, lanes, steps = R.plan_steps(cfg, model)
    got = [s.id for s in steps]
    assert "draft/note" not in got
    assert got.index("gguf/rungs") < got.index("draft/find") < got.index("draft/convert") < got.index("draft/rungs") \
        < got.index("gguf/card"), "the drafter is built + gated before any card quotes it"
    by = {s.id: s for s in steps}
    assert R.REQUIRES["draft"] == "any"
    for lane in ("gguf", "mlx", "gptq"):
        card = by[f"{lane}/card"].cmd
        assert card[card.index("--draft-repo") + 1].endswith("Tiny-1B-Pollard-DRAFT")
    dc = by["draft/card"].cmd
    assert dc[:2] == ["pollard-draft", "card"] and "--verified-only" in dc
    assert "exl3" not in dc[dc.index("--lanes") + 1]
    conv = by["draft/convert"].cmd
    assert conv[conv.index("--rungs") + 1] == "Q8_0,Q4_K_M"
    # off by default: no draft step, no draft flag on cards
    _, _, plain = R.plan_steps(R.load_plan(None), model)
    assert not any(s.lane == "draft" for s in plain)
    assert "--draft-results" not in {x for s in plain if s.kind == "card" for x in s.cmd}


def test_release_draft_alone_and_pinned_drafter(model):
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"].update({"enabled": True, "drafter": "z-lab/Qwen3-4B-DFlash-b16"})
    _, _, steps = R.plan_steps(cfg, model, {"draft"})
    got = [s.id for s in steps if s.lane == "draft"]
    assert got == ["draft/find", "draft/train", "draft/convert", "draft/rungs", "draft/card", "draft/upload"]
    find = steps[[s.id for s in steps].index("draft/find")].cmd
    assert find[find.index("--drafter") + 1] == "z-lab/Qwen3-4B-DFlash-b16"


def test_release_draft_rungs_measure_against_every_gguf_rung(model):
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"].update({"enabled": True, "limit": 6})
    c = R._ctx(cfg, model)
    os.makedirs(c["gguf_dir"], exist_ok=True)
    for tag in ("Q6_K", "IQ4_XS"):
        p = os.path.join(c["gguf_dir"], f"Tiny-1B-Pollard-{tag}.gguf")
        Path(p).write_bytes(b"x")
        ws.record_build(c["f16"], "gguf", p, tag=tag)
    dq = os.path.join(c["mdir"], "draft", "d-Q4.gguf")
    os.makedirs(os.path.dirname(dq), exist_ok=True)
    Path(dq).write_bytes(b"x")
    D.record(c["src"], dq, "DFlash-Q4_K_M", "gguf", "dflash", {"license": "mit"})
    D.record(c["src"], dq + ".bf16", "DFlash-BF16", "gguf", "dflash", {"publish": False})
    hf = os.path.join(c["mdir"], "draft", "hf")
    os.makedirs(hf)
    D.record(c["src"], hf, "DFlash-hf", "hf", "dflash", {"via": dq + ".bf16"})
    steps = R.draft_rung_steps(cfg, c, D.draft_builds(c["src"]))
    assert [s.id for s in steps] == ["draft/DFlash-Q4_K_M/measure", "draft/DFlash-Q4_K_M/verify",
                                     "draft/DFlash-hf/measure", "draft/DFlash-hf/verify"]
    m = steps[0].cmd
    assert m.count("--target") == 2 and {m[i + 1] for i, x in enumerate(m) if x == "--target-tag"} == {"Q6_K", "IQ4_XS"}
    assert m[m.index("--limit") + 1] == "6" and m[m.index("--ngl") + 1] == "0"
    assert steps[2].cmd[steps[2].cmd.index("--via") + 1] == dq + ".bf16"
    v = steps[1].cmd
    assert v[:4] == ["pollard-verify", "--lane", "draft", "--model"] and "--gate" in v
    cfg["lanes"]["draft"]["targets"] = ["Q6_K"]
    assert R.draft_targets(cfg, c) == [("Q6_K", os.path.join(c["gguf_dir"], "Tiny-1B-Pollard-Q6_K.gguf"))]


def test_release_execute_expands_draft_rungs(model, monkeypatch):
    ran = []
    monkeypatch.setattr(R, "run_cmd", lambda cmd, ok=(0,): ran.append(cmd))
    cfg = R.load_plan(None)
    cfg["lanes"]["draft"]["enabled"] = True
    c, _, steps = R.plan_steps(cfg, model, {"draft"})
    dq = os.path.join(c["mdir"], "draft", "d.gguf")
    os.makedirs(os.path.dirname(dq), exist_ok=True)
    Path(dq).write_bytes(b"x")
    D.record(c["src"], dq, "DFlash-Q8_0", "gguf", "dflash", {})
    st = R.execute([s for s in steps if s.kind != "upload"], c, cfg, run=True, have=ALL, out=lambda *_: None)
    assert st["draft/DFlash-Q8_0/measure"] == "done" and st["draft/DFlash-Q8_0/verify"] == "done"
    assert [x[:2] for x in ran if "draft" in " ".join(x)] == [["pollard-draft", "find"], ["pollard-draft", "train"],
                                    ["pollard-draft", "convert"],
                                    ["pollard-draft", "measure"], ["pollard-verify", "--lane"],
                                    ["pollard-draft", "card"]]


def test_release_upload_draft_layout(model, monkeypatch):
    cfg = R.load_plan(None)
    c = R._ctx(cfg, model)
    d = os.path.join(c["mdir"], "draft")
    hf = os.path.join(d, "hf")
    os.makedirs(hf)
    Path(hf, "config.json").write_text("{}")
    g = os.path.join(d, "Tiny-1B-Pollard-DRAFT-DFlash-Q4_K_M.gguf")
    Path(g).write_bytes(b"x")
    bf = os.path.join(d, "bf16.gguf")
    Path(bf).write_bytes(b"x")
    D.record(c["src"], g, "DFlash-Q4_K_M", "gguf", "dflash", {})
    D.record(c["src"], hf, "DFlash-hf", "hf", "dflash", {})
    D.record(c["src"], bf, "DFlash-BF16", "gguf", "dflash", {"publish": False})
    for p in (g, hf, bf):
        ws.record_gate(p, {"pass": True})
    out = os.path.join(c["release_dir"], "draft")
    os.makedirs(out)
    Path(out, "README.md").write_text("card")
    calls = []

    class Api:
        def create_repo(self, *a, **k):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
            calls.append(("file", path_in_repo))

        def upload_folder(self, folder_path, repo_id, path_in_repo=None):
            calls.append(("folder", folder_path, path_in_repo))

        def list_repo_files(self, repo):
            return ["README.md", "config.json", "Tiny-1B-Pollard-DRAFT-DFlash-Q4_K_M.gguf"]
    import types
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    st = R.Step(id="draft/upload", lane="draft", kind="upload", repo="P/Tiny-1B-Pollard-DRAFT", out=out)
    assert R.upload(st, c, cfg, True) == "uploaded"
    assert ("file", "Tiny-1B-Pollard-DRAFT-DFlash-Q4_K_M.gguf") in calls
    assert ("folder", os.path.abspath(hf), None) in calls, "the HF drafter at the root: vLLM loads it by repo id"
    assert not any("bf16" in str(x) for x in calls)


def test_pollard_card_appends_the_speculative_section(tmp_path, monkeypatch):
    import pollard_card as C
    home = tmp_path / "home"
    src = tmp_path / "Org__M"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024,
                                                 "num_hidden_layers": 4, "vocab_size": 1000}))
    (src / "README.md").write_text("---\nlicense: mit\n---\n")
    d = home / "models" / "Org__M"
    d.mkdir(parents=True)
    builds = [{"name": "M-Pollard-Q6_K.gguf", "lane": "gguf", "tag": "Q6_K", "path": str(d / "M-Pollard-Q6_K.gguf"),
               "bytes": 9e8}]
    (d / "MANIFEST.json").write_text(json.dumps({"model": "Org/M", "builds": builds}))
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"M-Pollard-Q6_K.gguf": {"ppl": 10.1}}))
    dres = tmp_path / "results-draft.json"
    dres.write_text(json.dumps(RESULTS))
    out = tmp_path / "README.md"
    monkeypatch.setenv("POLLARD_HOME", str(home))
    base = ["pollard-card", "--model", str(src), "--builds-from", "Org/M", "--repo", "PollardWeights/M-Pollard",
            "--results", str(res), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", base + ["--draft-results", str(dres), "--draft-repo", "PollardWeights/M-DRAFT"])
    C.main()
    md = out.read_text()
    assert "## Speculative decoding" in md and "| DFlash-Q4_K_M | Q6_K | **3.11** |" in md
    assert "--spec-type draft-dflash" in md and "PollardWeights/M-DRAFT" in md
    monkeypatch.setattr(sys, "argv", base + ["--draft-results", str(tmp_path / "missing.json"),
                                             "--draft-repo", "PollardWeights/M-DRAFT"])
    C.main()
    assert "## Speculative decoding" not in out.read_text(), "no measured drafter -> no section"
