"""pollard serve / pollard_harness: one table of agent harnesses, the runtime a rung needs, and configs that parse.

What it protects: a rung that loads is not a rung an agent can use. Claude Code's prompt is ~20k+ tokens
before the first turn, so llama-server's default context truncates it silently; thinking left in `content`
is parsed as the answer or a broken tool call; and an `IQ4_XS` whose allocation put an ik_llama-only atom on
a protected tensor does not open in stock llama.cpp, Ollama or LM Studio. pollard serve must size the
context to the harness, never hand an ik-only rung to stock llama.cpp, refuse to point a chat harness at a
decision model (it only answers /v1/systemone), and --write must back up and never emit a config the
harness cannot parse. Everything here runs with --print-only or mocks: no server is started.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_harness as H  # noqa: E402

ALL = ["omp", "claude-code", "codex", "opencode", "cline", "aider", "continue", "goose", "openhands",
       "open-webui", "ollama", "lm-studio"]


def test_table_has_every_harness_and_every_field():
    assert set(H.HARNESSES) == set(ALL)
    for name, h in H.HARNESSES.items():
        assert h["endpoint"] in H.ENDPOINTS, name
        assert h["min_ctx"] >= 8192 and callable(h["render"]) and h["card"], name
        assert isinstance(h["verified"], bool), name
    # the harnesses whose system prompt is the size of a novel get a context that holds it
    assert H.HARNESSES["claude-code"]["min_ctx"] >= 65536
    assert H.HARNESSES["claude-code"]["endpoint"] == "anthropic" and H.HARNESSES["codex"]["endpoint"] == "responses"
    # the ones Pollard does not serve never carry a reasoning flag -- it is not ours to set
    assert H.HARNESSES["ollama"]["reasoning"] is None and H.HARNESSES["lm-studio"]["reasoning"] is None


def _c(**kw):
    return H.context("Model-Pollard-Q4_K_M", 32768, 8080, **kw)


def test_rendered_configs_parse_in_their_own_format():
    c = _c()
    omp = yaml.safe_load(H.render("omp", c))
    prov = omp["providers"]["pollard"]
    assert prov["api"] == "openai-completions" and prov["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert prov["auth"] == "none" and prov["models"][0]["contextWindow"] == 32768
    codex = tomllib.loads(H.render("codex", c))
    assert codex["model_providers"]["pollard"]["wire_api"] == "responses"
    assert codex["profiles"]["pollard"]["model"] == "Model-Pollard-Q4_K_M"
    oc = json.loads("\n".join(l for l in H.render("opencode", c).splitlines() if not l.startswith("//")))
    p = oc["provider"]["pollard"]
    assert p["npm"] == "@ai-sdk/openai-compatible" and p["options"]["baseURL"].endswith("/v1")
    assert p["models"]["Model-Pollard-Q4_K_M"]["limit"]["context"] == 32768
    cont = yaml.safe_load(H.render("continue", c))
    assert cont["models"][0]["provider"] == "llama.cpp" and cont["models"][0]["apiBase"] == "http://127.0.0.1:8080"
    cc = H.render("claude-code", c)
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8080\n" in cc and "ANTHROPIC_MODEL=Model-Pollard-Q4_K_M" in cc
    assert "OPENAI_API_BASE=http://127.0.0.1:8080/v1" in H.render("aider", c)
    for name in ALL:                                       # every renderer at least runs and names the model
        assert "Model-Pollard-Q4_K_M" in H.render(name, c) or name == "ollama"


def test_decision_model_renders_the_omp_judge_on_typesafe():
    """OMP's typesafe API POSTs to {baseUrl}/v1/systemone, so its baseUrl is the server ROOT."""
    c = H.context("clef-flash-Q6_K", 4096, 8080, decision=True)
    d = yaml.safe_load(H.render("omp", c).split("# ~/.omp/agent/config.yml")[0])
    prov = d["providers"]["pollard-judge"]
    assert prov["api"] == "typesafe" and prov["baseUrl"] == "http://127.0.0.1:8080"
    roles = yaml.safe_load(H.render("omp", c).split("# ~/.omp/agent/config.yml")[1])
    assert roles["modelRoles"]["judge"] == "pollard-judge/clef-flash-Q6_K"


def test_ollama_uses_the_published_tag_or_a_modelfile():
    assert "ollama run hf.co/Org/M-Pollard:Q4_K_M" in H.render("ollama", _c(repo="Org/M-Pollard", tag="Q4_K_M"))
    assert "num_ctx 32768" in H.render("ollama", _c(repo="Org/M-Pollard", tag="Q4_K_M"))
    assert "FROM /x/m.gguf" in H.render("ollama", _c(file="/x/m.gguf"))


# -- runtime choice ----------------------------------------------------------------------------------

def test_ik_only_rung_never_goes_to_stock_llama_cpp():
    stock_only = lambda name: "/usr/bin/llama-server" if name == "llama.cpp" else None
    rt, binary, fatal = H.choose_runtime("ik_llama", "qwen3", exe=stock_only, supports_arch=lambda b, a: True)
    assert binary is None and fatal and "ik_llama" in fatal
    both = lambda name: {"llama.cpp": "/usr/bin/llama-server", "ik_llama": "/opt/ik_llama.cpp/bin/llama-server"}[name]
    rt, binary, fatal = H.choose_runtime("ik_llama", "qwen3", exe=both, supports_arch=lambda b, a: True)
    assert rt == "ik_llama" and binary == "/opt/ik_llama.cpp/bin/llama-server" and fatal is None
    # an explicit --server that is plainly stock is refused for an ik rung, too
    rt, binary, fatal = H.choose_runtime("ik_llama", "qwen3", server="/usr/bin/llama-server", exe=both,
                                         supports_arch=lambda b, a: True)
    assert fatal and binary is None


def test_stock_rung_prefers_upstream_and_checks_the_arch():
    both = lambda name: {"llama.cpp": "/usr/bin/llama-server", "ik_llama": "/opt/ik_llama.cpp/bin/llama-server"}[name]
    rt, binary, fatal = H.choose_runtime("stock", "qwen3", exe=both, supports_arch=lambda b, a: True)
    assert rt == "llama.cpp" and fatal is None
    rt, binary, fatal = H.choose_runtime("newer", "spark2_5", exe=both, supports_arch=lambda b, a: False)
    assert fatal and "update llama.cpp" in fatal
    assert H.choose_runtime("fork", "k2-horizon", exe=both)[2]


# -- preflight ----------------------------------------------------------------------------------------

def _info(tmp_path, name="M-Q4_K_M.gguf"):
    p = tmp_path / name
    p.write_bytes(b"GGUF")
    return {"path": str(p), "name": name, "tag": "Q4_K_M", "repo": None, "url": None, "source": "file"}


def test_preflight_blocks_what_cannot_work(tmp_path):
    info = _info(tmp_path)
    fatal, _ = H.preflight(info, "codex", 32768, "stock", "qwen3", None, False, True, "llama.cpp", "x", None)
    assert any("decision model" in f for f in fatal)
    fatal, _ = H.preflight(info, "ollama", 8192, "ik_llama", "qwen3", None, False, False, None, None, None)
    assert any("Ollama" in f for f in fatal)
    fatal, _ = H.preflight(info, "omp", 32768, "stock", "dream", None, True, False, "llama.cpp", "x", None)
    assert any("diffusion" in f for f in fatal)
    fatal, _ = H.preflight(dict(info, path=str(tmp_path / "gone.gguf")), "omp", 32768, "stock", "qwen3", None,
                           False, False, "llama.cpp", "x", None)
    assert any("no such file" in f for f in fatal)


def test_preflight_warns_on_context_and_ik_endpoints(tmp_path):
    info = _info(tmp_path)
    fatal, warn = H.preflight(info, "claude-code", 8192, "ik_llama", "qwen3", 32768, False, False,
                              "ik_llama", "/opt/ik_llama/llama-server", None)
    assert not fatal
    assert any("below Claude Code's working minimum" in w for w in warn)
    assert any("unverified on ik_llama" in w for w in warn)
    _, warn = H.preflight(info, "claude-code", 65536, "stock", "qwen3", 32768, False, False, "llama.cpp", "x", None)
    assert any("beyond the model's trained 32768" in w for w in warn)


# -- server command -----------------------------------------------------------------------------------

def test_server_cmd_sizes_context_and_splits_reasoning():
    argv, dropped = H.server_cmd("/b/llama-server", "/m.gguf", "claude-code", 65536, 8080, alias="m")
    s = " ".join(argv)
    assert "-c 65536" in s and "--jinja" in s and "--reasoning-format deepseek" in s and "-np 1" in s
    assert "--alias m" in s and "--no-context-shift" in s and not dropped
    argv, dropped = H.server_cmd("/b/llama-server", "/m.gguf", "omp", 4096, 8080, alias="m", decision=True)
    s = " ".join(argv)
    assert "-b 4096 -ub 4096" in s and "--jinja" not in s and "--reasoning-format" not in s
    # a fork that does not know a flag: the flag is dropped, not passed (a hard startup failure otherwise)
    argv, dropped = H.server_cmd("/b/llama-server", "/m.gguf", "omp", 32768, 8080, alias="m",
                                 supports=lambda f: f != "--reasoning-format")
    assert "--reasoning-format" not in argv and dropped == ["--reasoning-format"]


# -- resolving the target -----------------------------------------------------------------------------

def test_resolve_workspace_rung_from_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path))
    d = tmp_path / "models" / "Org__Model"
    d.mkdir(parents=True)
    gg = d / "Model-Pollard-GGUF-Q4_K_M.gguf"
    gg.write_bytes(b"GGUF")
    (d / "MANIFEST.json").write_text(json.dumps({"model": "Org/Model", "builds": [
        {"name": gg.name, "lane": "gguf", "tag": "Q4_K_M", "path": str(gg), "bytes": 4}]}))
    got = H.resolve("Org/Model:q4_k_m")
    assert got["path"] == str(gg) and got["source"] == "workspace"
    assert H.resolve(gg.name)["path"] == str(gg)                        # a bare build name, any MANIFEST


def test_resolve_published_rung_through_the_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path))
    fetch = lambda url: {"siblings": [{"rfilename": "M-Pollard-Q6_K.gguf"}, {"rfilename": "M-Pollard-Q4_K_M.gguf"},
                                      {"rfilename": "mmproj-M-Q4_K_M.gguf"}, {"rfilename": "README.md"}]}
    got = H.resolve("PollardWeights/M-Pollard:Q4_K_M", fetch=fetch)
    assert got["file_in_repo"] == "M-Pollard-Q4_K_M.gguf" and got["source"] == "hf"
    assert got["url"] == "https://huggingface.co/PollardWeights/M-Pollard/resolve/main/M-Pollard-Q4_K_M.gguf"
    with pytest.raises(SystemExit):
        H.resolve("PollardWeights/M-Pollard:IQ2_XXS", fetch=fetch)


# -- the CLI, --print-only ----------------------------------------------------------------------------

def test_print_only_starts_nothing_and_prints_the_snippet(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path))
    monkeypatch.setattr(H.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))
    rc = H.main([str(tmp_path / "Qwen3-8B-Pollard-Q4_K_M.gguf"), "--harness", "claude-code", "--print-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "-c 65536" in out and "ANTHROPIC_BASE_URL=http://127.0.0.1:8080" in out
    assert "BLOCKED: no such file" in out
    rc = H.main([str(tmp_path / "clef-flash-Pollard-Q6_K.gguf"), "--harness", "omp", "--print-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "api: typesafe" in out and "judge: pollard-judge/clef-flash-Pollard-Q6_K" in out
    assert "/v1/systemone" in out and "-c 4096" in out
    H.main([str(tmp_path / "clef-flash-Pollard-Q6_K.gguf"), "--harness", "codex", "--print-only"])
    out = capsys.readouterr().out
    assert "cannot chat" in out and "wire_api" not in out                # no config for a harness that cannot use it


def test_pollard_serve_subcommand_dispatches(monkeypatch):
    import pollard_auto
    seen = {}
    monkeypatch.setattr(H, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    monkeypatch.setattr(sys, "argv", ["pollard", "serve", "x.gguf", "--print-only"])
    with pytest.raises(SystemExit):
        pollard_auto.main()
    assert seen["argv"] == ["x.gguf", "--print-only"]


def test_serve_is_a_declared_console_script():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert py["project"]["scripts"]["pollard-serve"] == "pollard_harness:main"
    assert "pollard_harness" in py["tool"]["setuptools"]["py-modules"]


# -- --write ------------------------------------------------------------------------------------------

def test_write_omp_merges_and_backs_up(tmp_path):
    p = tmp_path / ".omp" / "agent" / "models.yml"
    p.parent.mkdir(parents=True)
    p.write_text("providers:\n  other:\n    baseUrl: https://x/v1\n    api: openai-completions\n")
    done = H.write_config("omp", _c(), home=str(tmp_path))
    (path, bak), = done
    assert bak and Path(bak).read_text().startswith("providers:\n  other:")
    d = yaml.safe_load(p.read_text())
    assert set(d["providers"]) == {"other", "pollard"} and d["providers"]["pollard"]["api"] == "openai-completions"
    assert not (tmp_path / ".omp" / "agent" / "config.yml").exists()  # a chat model's role is left to the user


def test_write_omp_decision_sets_the_judge_role(tmp_path):
    cfg = tmp_path / ".omp" / "agent" / "config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("modelRoles:\n  default: anthropic/claude\n")
    H.write_config("omp", H.context("clef", 4096, 8080, decision=True), home=str(tmp_path))
    roles = yaml.safe_load(cfg.read_text())["modelRoles"]
    assert roles == {"default": "anthropic/claude", "judge": "pollard-judge/clef"}
    prov = yaml.safe_load((cfg.parent / "models.yml").read_text())["providers"]["pollard-judge"]
    assert prov["api"] == "typesafe" and prov["baseUrl"] == "http://127.0.0.1:8080"


def test_write_codex_appends_tables_and_refuses_a_redeclaration(tmp_path):
    p = tmp_path / ".codex" / "config.toml"
    p.parent.mkdir(parents=True)
    p.write_text('model = "gpt-5"\n# mine\n[mcp_servers.x]\ncommand = "y"\n')
    H.write_config("codex", _c(), home=str(tmp_path))
    d = tomllib.loads(p.read_text())
    assert d["model"] == "gpt-5" and d["mcp_servers"]["x"]["command"] == "y"     # top-level key untouched
    assert d["model_providers"]["pollard"]["base_url"] == "http://127.0.0.1:8080/v1" and "# mine" in p.read_text()
    with pytest.raises(ValueError):
        H.write_config("codex", _c(), home=str(tmp_path))


def test_write_opencode_refuses_jsonc_and_merges_json(tmp_path):
    p = tmp_path / ".config" / "opencode" / "opencode.json"
    p.parent.mkdir(parents=True)
    p.write_text('{\n  // my comment\n  "theme": "x"\n}\n')
    with pytest.raises(ValueError):
        H.write_config("opencode", _c(), home=str(tmp_path))
    p.write_text('{"theme": "x", "provider": {"other": {}}}')
    H.write_config("opencode", _c(), home=str(tmp_path))
    d = json.loads(p.read_text())
    assert d["theme"] == "x" and set(d["provider"]) == {"other", "pollard"}


def test_write_continue_replaces_its_own_entry_only(tmp_path):
    p = tmp_path / ".continue" / "config.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("name: mine\nmodels:\n  - name: Other\n    provider: openai\n")
    H.write_config("continue", _c(), home=str(tmp_path))
    H.write_config("continue", _c(), home=str(tmp_path))
    names = [m["name"] for m in yaml.safe_load(p.read_text())["models"]]
    assert names == ["Other", "Model-Pollard-Q4_K_M (Pollard)"]
    assert len(list(p.parent.glob("config.yaml.pollard-bak-*"))) >= 1


def test_write_refuses_formats_it_does_not_merge(tmp_path):
    for name in ("claude-code", "cline", "goose", "aider", "ollama"):
        with pytest.raises(ValueError):
            H.write_config(name, _c(), home=str(tmp_path))
    assert not any(tmp_path.iterdir())
