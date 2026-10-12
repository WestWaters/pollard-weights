"""pollard-card: "Use it in your harness" and the agent-ready column, rendered from the same table as pollard serve.

What it protects: the card and the CLI must say the same thing about how to point an agent at a rung, so the
section is rendered from pollard_harness.HARNESSES rather than written by hand. The Ollama / LM Studio rows
only appear when a rung stock llama.cpp opens exists (both bundle stock llama.cpp; an ik_llama-only rung
fails there). A decision model gets the OMP judge block and a /v1/systemone note instead of chat harnesses
it cannot serve. And when results carry a pollard-toolcall result per rung, the files table says which rungs
still write valid tool calls -- without it, no column (an all-"--" column reads as "measured, nothing there").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_harness as H  # noqa: E402


def test_section_lists_every_chat_harness_and_ollama_only_with_a_stock_rung():
    md = "\n".join(H.card_section("Org/M-Pollard", "M-Pollard-Q4_K_M.gguf", "Q4_K_M", stock_tag="Q4_K_M"))
    assert md.startswith("## Use it in your harness")
    for h in H.HARNESSES.values():
        assert h["label"] in md
    assert "ollama run hf.co/Org/M-Pollard:Q4_K_M" in md
    assert "pollard serve Org/M-Pollard:Q4_K_M --harness claude-code" in md
    assert "llama-server -hf Org/M-Pollard:Q4_K_M -c 65536 --jinja --reasoning-format deepseek" in md
    md = "\n".join(H.card_section("Org/M-Pollard", "M-Pollard-IQ2_KT.gguf", "IQ2_KT", needs="ik_llama.cpp"))
    assert "Ollama" not in md and "LM Studio" not in md
    assert "ik_llama.cpp" in md and "-hf" not in md


def test_decision_card_shows_the_omp_judge_not_chat_harnesses():
    md = "\n".join(H.card_section("Org/clef-Pollard", "clef-Q6_K.gguf", "Q6_K", decision=True, stock_tag="Q6_K"))
    assert "api: typesafe" in md and "judge: pollard-judge/clef-Q6_K" in md and "/v1/systemone" in md
    assert "Claude Code" in md and "ANTHROPIC_BASE_URL" not in md and "ollama run" not in md


def test_agent_ready_cell():
    assert C.agent_ready_cell({"pass": True, "rate": 0.95}) == "yes (95%)"
    assert C.agent_ready_cell({"rate": 0.5}) == "no (50%)"                  # pollard-toolcall's 0.75 bar
    assert C.agent_ready_cell({"pass": False}) == "no"
    assert C.agent_ready_cell(None) == "--"


def _card(tmp_path, monkeypatch, results):
    home = tmp_path / "home"
    src = tmp_path / "Org__M"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024,
                                                 "num_hidden_layers": 4, "vocab_size": 1000}))
    (src / "README.md").write_text("---\nlicense: mit\n---\n")
    d = home / "models" / "Org__M"
    d.mkdir(parents=True)
    builds = [{"name": f"M-Pollard-{t}.gguf", "lane": "gguf", "tag": t, "path": str(d / f"M-Pollard-{t}.gguf"),
               "bytes": n} for t, n in (("Q6_K", 900_000_000), ("Q4_K_M", 600_000_000))]
    (d / "MANIFEST.json").write_text(json.dumps({"model": "Org/M", "builds": builds}))
    res = tmp_path / "results.json"
    res.write_text(json.dumps(results))
    out = tmp_path / "README.md"
    monkeypatch.setenv("POLLARD_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["pollard-card", "--model", str(src), "--builds-from", "Org/M",
                                      "--repo", "PollardWeights/M-Pollard", "--results", str(res), "--out", str(out)])
    C.main()
    return out.read_text()


def test_card_has_agent_ready_column_and_harness_section_after_how_to_run(tmp_path, monkeypatch):
    md = _card(tmp_path, monkeypatch, {
        "M-Pollard-Q6_K.gguf": {"ppl": 10.1, "kld": 0.01, "toolcall": {"pass": True, "rate": 0.95}},
        "M-Pollard-Q4_K_M.gguf": {"ppl": 10.6, "kld": 0.05, "toolcall": {"pass": False, "rate": 0.6}}})
    assert "| agent-ready |" in md and "| yes (95%) |" in md and "| no (60%) |" in md
    assert md.index("## How to run") < md.index("## Use it in your harness") < md.index("## Errata")
    assert "pollard serve PollardWeights/M-Pollard:" in md


def test_card_without_toolcall_results_has_no_agent_ready_column(tmp_path, monkeypatch):
    md = _card(tmp_path, monkeypatch, {"M-Pollard-Q6_K.gguf": {"ppl": 10.1, "kld": 0.01}})
    assert "agent-ready" not in md and "## Use it in your harness" in md
