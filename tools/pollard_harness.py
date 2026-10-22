#!/usr/bin/env python3
"""pollard-serve -- serve a Pollard build for an agent harness: the right runtime, a context the harness
actually fits in, and the config snippet that points the harness at it.

A rung that loads is not a rung an agent can use. Three things go wrong silently between the two, and all
three are settings rather than weights:

  * context. Claude Code's system prompt plus its tool schemas is ~20k+ tokens before the first user turn.
    llama-server's default context is the model's trained length on new builds and 4096 on older ones and
    on ik_llama; a harness whose prompt does not fit is truncated or rejected, and what the user sees is
    "the model ignores my tools".
  * reasoning. Without --reasoning-format the thinking block can land in `content`, where the harness
    parses it as the answer or as a malformed tool call.
  * runtime. A rung whose allocation put an ik_llama-only atom on a protected tensor (an `IQ4_XS` can be
    one) does not open in stock llama.cpp, nor in Ollama or LM Studio, which bundle it.

So the harnesses live in ONE table (HARNESSES): which endpoint each speaks, the smallest context it is
usable at, how llama-server must split reasoning for it, and how to render its config. The CLI and the
model card (pollard-card's "Use it in your harness") both render from it, so the two never disagree.

    pollard serve ~/pollard/models/Org__Model/Model-Pollard-GGUF-Q4_K_M.gguf --harness claude-code
    pollard serve Org/Model:Q4_K_M --harness omp            # a workspace rung (MANIFEST) by model + tag
    pollard serve PollardWeights/Model-Pollard:Q4_K_M --harness codex --write
    pollard serve clef-flash-Q6_K.gguf --harness omp        # a decision model -> the OMP judge block
    pollard serve model.gguf --harness opencode --print-only   # plan + snippet, start nothing

`--write` merges the snippet into the harness's own config file (after copying the original beside it)
for the formats it can merge without guessing: omp (models.yml / config.yml), opencode (opencode.json),
codex (config.toml, appended tables only) and continue (config.yaml). Everything else is printed.
"""
from __future__ import annotations

import argparse, glob, json, os, re, shutil, subprocess, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PORT = 8080
DECISION_CTX = 4096          # pollard-decision's gate default; a decision prompt is one state + its questions

ENDPOINTS = {"chat": "/v1/chat/completions", "anthropic": "/v1/messages", "responses": "/v1/responses",
             "systemone": "/v1/systemone", "own": None}


# -- renderers ----------------------------------------------------------------------------------------
# Each takes the serve context c = {base, model, ctx, repo, tag, file, decision, provider} and returns
# the snippet text. `base` is the server root (http://127.0.0.1:8080), never with /v1 -- each harness
# appends what it expects, because they disagree.

def _omp(c):
    if c["decision"]:
        # OMP's `typesafe` API is a judgment API, not a chat transport: it POSTs the System One body to
        # {baseUrl}/v1/systemone and backs the `judge` model role. So baseUrl is the server ROOT here,
        # where the chat provider below wants .../v1 (docs/models.md, oh-my-pi, read 2026-10-11).
        return (f"# ~/.omp/agent/models.yml\n"
                f"providers:\n"
                f"  {c['provider']}:\n"
                f"    baseUrl: {c['base']}\n"
                f"    api: typesafe\n"
                f"    auth: none\n"
                f"    models:\n"
                f"      - id: {c['model']}\n"
                f"        name: {c['model']} (Pollard, /v1/systemone)\n"
                f"\n"
                f"# ~/.omp/agent/config.yml\n"
                f"modelRoles:\n"
                f"  judge: {c['provider']}/{c['model']}\n")
    return (f"# ~/.omp/agent/models.yml\n"
            f"providers:\n"
            f"  {c['provider']}:\n"
            f"    baseUrl: {c['base']}/v1\n"
            f"    api: openai-completions\n"
            f"    auth: none\n"
            f"    models:\n"
            f"      - id: {c['model']}\n"
            f"        name: {c['model']} (Pollard)\n"
            f"        contextWindow: {c['ctx']}\n"
            f"\n"
            f"# then pick it with /model, or make it a role in ~/.omp/agent/config.yml:\n"
            f"#   modelRoles:\n"
            f"#     default: {c['provider']}/{c['model']}\n")


def _claude_code(c):
    # Claude Code reads these from the environment (docs.claude.com, "LLM gateway" settings). Kept as env
    # vars rather than written to ~/.claude/settings.json on purpose: that file is global, and pointing
    # every Claude Code session on the machine at a local server is not what "try this model" means.
    return (f"export ANTHROPIC_BASE_URL={c['base']}\n"
            f"export ANTHROPIC_AUTH_TOKEN=pollard          # any non-empty value; llama-server has no key\n"
            f"export ANTHROPIC_MODEL={c['model']}\n"
            f"export ANTHROPIC_DEFAULT_HAIKU_MODEL={c['model']}   # background tasks too, or they ask for a Claude model\n"
            f"claude\n")


def _codex(c):
    # A provider table plus a profile table, both appendable: Codex's top-level `model =` key has to come
    # before every table, so appending it to an existing file would land it inside the last table.
    return (f"# ~/.codex/config.toml\n"
            f"[model_providers.{c['provider']}]\n"
            f"name = \"Pollard (llama-server)\"\n"
            f"base_url = \"{c['base']}/v1\"\n"
            f"wire_api = \"responses\"\n"
            f"\n"
            f"[profiles.{c['provider']}]\n"
            f"model = \"{c['model']}\"\n"
            f"model_provider = \"{c['provider']}\"\n"
            f"\n"
            f"# run: codex --profile {c['provider']}\n")


def _opencode_obj(c):
    return {"$schema": "https://opencode.ai/config.json",
            "provider": {c["provider"]: {
                "npm": "@ai-sdk/openai-compatible", "name": "Pollard (llama-server)",
                "options": {"baseURL": f"{c['base']}/v1"},
                "models": {c["model"]: {"name": c["model"], "limit": {"context": c["ctx"], "output": 8192}}}}}}


def _opencode(c):
    return ("// ~/.config/opencode/opencode.json\n" + json.dumps(_opencode_obj(c), indent=2) +
            f"\n// pick it with /models, or: opencode -m {c['provider']}/{c['model']}\n")


def _cline(c):
    # UNVERIFIED against Cline's / Roo's docs: the field names below are the settings panel's labels as
    # commonly reported. Cline stores them in VS Code extension state, not a file, so this is never written.
    return (f"Cline / Roo Code -> settings -> API Provider: \"OpenAI Compatible\"\n"
            f"  Base URL:        {c['base']}/v1\n"
            f"  API Key:         pollard   (any value)\n"
            f"  Model ID:        {c['model']}\n"
            f"  Context Window:  {c['ctx']}\n")


def _aider(c):
    return (f"export OPENAI_API_BASE={c['base']}/v1\n"
            f"export OPENAI_API_KEY=pollard\n"
            f"aider --model openai/{c['model']}\n")


def _continue_obj(c):
    return {"name": f"{c['model']} (Pollard)", "provider": "llama.cpp", "model": c["model"],
            "apiBase": c["base"], "roles": ["chat", "edit", "apply"],
            "defaultCompletionOptions": {"contextLength": c["ctx"]}}


def _continue(c):
    o = _continue_obj(c)
    return (f"# ~/.continue/config.yaml\n"
            f"models:\n"
            f"  - name: {o['name']}\n"
            f"    provider: llama.cpp\n"
            f"    model: {o['model']}\n"
            f"    apiBase: {o['apiBase']}\n"
            f"    roles: [chat, edit, apply]\n"
            f"    defaultCompletionOptions:\n"
            f"      contextLength: {c['ctx']}\n")


def _goose(c):
    # UNVERIFIED against Goose's docs: the OpenAI provider's host / base-path variable names as commonly
    # reported. Printed only.
    return (f"export GOOSE_PROVIDER=openai\n"
            f"export OPENAI_HOST={c['base']}\n"
            f"export OPENAI_BASE_PATH=v1/chat/completions\n"
            f"export OPENAI_API_KEY=pollard\n"
            f"export GOOSE_MODEL={c['model']}\n"
            f"goose session\n")


def _openhands(c):
    # UNVERIFIED against OpenHands' docs (LiteLLM-style `openai/<name>` + a base URL is the known pattern).
    # OpenHands runs in Docker, where 127.0.0.1 is the container: serve with --host 0.0.0.0.
    base = c["base"].replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    return (f"OpenHands -> Settings -> LLM -> Advanced:\n"
            f"  Custom Model:  openai/{c['model']}\n"
            f"  Base URL:      {base}/v1\n"
            f"  API Key:       pollard\n"
            f"(OpenHands runs in Docker: start the server with --host 0.0.0.0 so the container can reach it)\n")


def _open_webui(c):
    return (f"Open WebUI -> Admin Settings -> Connections -> OpenAI API -> add:\n"
            f"  URL:  {c['base']}/v1      (from a Docker install: http://host.docker.internal:{c['port']}/v1,\n"
            f"                             and serve with --host 0.0.0.0)\n"
            f"  Key:  pollard\n"
            f"The model appears as \"{c['model']}\".\n")


def _ollama(c):
    # Ollama runs its own bundled llama.cpp, so pollard serve starts nothing for it. Its default num_ctx is
    # small (a few thousand tokens) and it does not say when a prompt overflowed -- set it explicitly.
    if c.get("repo") and c.get("tag"):
        return (f"ollama run hf.co/{c['repo']}:{c['tag']}\n"
                f">>> /set parameter num_ctx {c['ctx']}\n")
    return (f"# Modelfile\n"
            f"FROM {c['file']}\n"
            f"PARAMETER num_ctx {c['ctx']}\n"
            f"\n"
            f"ollama create {c['model'].lower()} -f Modelfile && ollama run {c['model'].lower()}\n")


def _lm_studio(c):
    # `lms import` is LM Studio's CLI for adding a local GGUF (UNVERIFIED flag set; the GUI path is the
    # "My Models" folder). LM Studio's server is OpenAI-compatible on :1234 by default.
    src = f"lms get hf.co/{c['repo']}:{c['tag']}" if c.get("repo") and c.get("tag") else f"lms import {c['file']}"
    return (f"{src}\n"
            f"lms load {c['model']} --context-length {c['ctx']}\n"
            f"lms server start          # OpenAI-compatible at http://localhost:1234/v1\n")


# -- THE table ----------------------------------------------------------------------------------------
# endpoint: what the harness calls on llama-server ("own" = the harness runs its own runtime).
# min_ctx:  the smallest context it is usable at; a harness whose prompt overflows fails silently.
# reasoning: llama-server --reasoning-format. "deepseek" puts thinking in reasoning_content, out of the
#           `content` the harness parses for answers and tool calls. None = not ours to set.
# config / fmt / write: where its config lives and whether --write may merge into it.
# verified: whether the config keys were checked against the harness's own docs (False = known pattern).
HARNESSES: dict[str, dict] = {
    "omp": {"label": "Oh My Pi (`omp`)", "endpoint": "chat", "min_ctx": 32768, "reasoning": "deepseek",
            "config": "~/.omp/agent/models.yml", "fmt": "yaml", "write": True, "verified": True,
            "render": _omp, "card": "`~/.omp/agent/models.yml`: provider `api: openai-completions`, "
                                    "`baseUrl: http://127.0.0.1:8080/v1`, `auth: none`"},
    "claude-code": {"label": "Claude Code", "endpoint": "anthropic", "min_ctx": 65536, "reasoning": "deepseek",
                    "config": None, "fmt": "env", "write": False, "verified": True, "render": _claude_code,
                    "card": "`ANTHROPIC_BASE_URL=http://127.0.0.1:8080` `ANTHROPIC_AUTH_TOKEN=pollard` "
                            "`ANTHROPIC_MODEL=<name>`"},
    "codex": {"label": "Codex CLI", "endpoint": "responses", "min_ctx": 32768, "reasoning": "deepseek",
              "config": "~/.codex/config.toml", "fmt": "toml", "write": True, "verified": True, "render": _codex,
              "card": "`~/.codex/config.toml`: `[model_providers.pollard]` `base_url = \"http://127.0.0.1:8080/v1\"`, "
                      "`wire_api = \"responses\"`"},
    "opencode": {"label": "OpenCode", "endpoint": "chat", "min_ctx": 32768, "reasoning": "deepseek",
                 "config": "~/.config/opencode/opencode.json", "fmt": "json", "write": True, "verified": True,
                 "render": _opencode, "card": "`opencode.json`: provider `npm: @ai-sdk/openai-compatible`, "
                                              "`options.baseURL: http://127.0.0.1:8080/v1`, `limit.context`"},
    "cline": {"label": "Cline / Roo Code", "endpoint": "chat", "min_ctx": 32768, "reasoning": "deepseek",
              "config": None, "fmt": "ui", "write": False, "verified": False, "render": _cline,
              "card": "API Provider \"OpenAI Compatible\", Base URL `http://127.0.0.1:8080/v1`"},
    "aider": {"label": "Aider", "endpoint": "chat", "min_ctx": 16384, "reasoning": "deepseek",
              "config": None, "fmt": "env", "write": False, "verified": True, "render": _aider,
              "card": "`OPENAI_API_BASE=http://127.0.0.1:8080/v1 aider --model openai/<name>`"},
    "continue": {"label": "Continue", "endpoint": "chat", "min_ctx": 32768, "reasoning": "deepseek",
                 "config": "~/.continue/config.yaml", "fmt": "yaml", "write": True, "verified": True,
                 "render": _continue, "card": "`config.yaml`: `provider: llama.cpp`, `apiBase: http://127.0.0.1:8080`"},
    "goose": {"label": "Goose", "endpoint": "chat", "min_ctx": 32768, "reasoning": "deepseek",
              "config": None, "fmt": "env", "write": False, "verified": False, "render": _goose,
              "card": "`GOOSE_PROVIDER=openai` `OPENAI_HOST=http://127.0.0.1:8080`"},
    "openhands": {"label": "OpenHands", "endpoint": "chat", "min_ctx": 32768, "reasoning": "deepseek",
                  "config": None, "fmt": "ui", "write": False, "verified": False, "render": _openhands,
                  "card": "model `openai/<name>`, base URL `http://host.docker.internal:8080/v1`"},
    "open-webui": {"label": "Open WebUI", "endpoint": "chat", "min_ctx": 8192, "reasoning": "deepseek",
                   "config": None, "fmt": "ui", "write": False, "verified": True, "render": _open_webui,
                   "card": "Admin Settings -> Connections -> OpenAI API: `http://127.0.0.1:8080/v1`"},
    "ollama": {"label": "Ollama", "endpoint": "own", "min_ctx": 8192, "reasoning": None,
               "config": None, "fmt": "cli", "write": False, "verified": True, "render": _ollama, "stock_only": True,
               "card": "`ollama run hf.co/<repo>:<TAG>`, then `/set parameter num_ctx 8192` or more"},
    "lm-studio": {"label": "LM Studio / Jan", "endpoint": "own", "min_ctx": 8192, "reasoning": None,
                  "config": None, "fmt": "cli", "write": False, "verified": False, "render": _lm_studio,
                  "stock_only": True, "card": "load the GGUF, set the context length to 8192 or more"},
}


def ctx_label(n):
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def context(model, ctx, port=DEFAULT_PORT, host="127.0.0.1", repo=None, tag=None, file=None,
            decision=False):
    """The values every renderer reads."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "", None) else host
    return {"base": f"http://{shown}:{port}", "port": port, "model": model, "ctx": ctx, "repo": repo,
            "tag": tag, "file": file or f"{model}.gguf", "decision": decision,
            "provider": "pollard-judge" if decision else "pollard"}


def render(name, c):
    return HARNESSES[name]["render"](c)


# -- resolving what to serve --------------------------------------------------------------------------

def _manifest_builds(model):
    try:
        import pollard_workspace as ws
        return ws.read_manifest(model).get("builds", [])
    except Exception:
        return []


def _all_workspace_builds():
    try:
        import pollard_workspace as ws
        return [(slug, b) for slug in ws.list_models() for b in _manifest_builds(slug) if isinstance(b, dict)]
    except Exception:
        return []


def _pick(builds, tag):
    t = tag.lower()
    for b in builds:
        if str(b.get("tag", "")).lower() == t or str(b.get("name", "")).lower() in (t, t + ".gguf"):
            return b
    return None


def hf_gguf_files(repo, fetch=None):
    """The .gguf filenames of a published repo, through the Hub's public API (stdlib, no login)."""
    import urllib.request
    fetch = fetch or (lambda url: json.loads(urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "pollard-serve"}), timeout=20).read().decode("utf-8")))
    info = fetch(f"https://huggingface.co/api/models/{repo}")
    return [s["rfilename"] for s in info.get("siblings", []) if str(s.get("rfilename", "")).endswith(".gguf")]


def resolve(target, fetch=None):
    """{path, name, tag, repo, url, source} for a .gguf path, a workspace rung (MODEL:TAG) or a published
    rung (ORG/REPO:TAG). Published rungs include files_in_repo (all shards, starting with shard one).
    A local path wins; then the workspace MANIFEST; then the Hub."""
    t = os.path.expanduser(target)
    if not os.path.isfile(t) and os.sep not in t and "/" not in t and t.endswith(".gguf"):
        # a bare build name not in the cwd: the workspace knows where `pollard` put it
        for _slug, b in _all_workspace_builds():
            if str(b.get("name", "")).lower() == t.lower():
                return {"path": b.get("path"), "name": b.get("name"), "tag": b.get("tag"), "repo": None,
                        "url": None, "source": "workspace"}
    if t.endswith(".gguf") or os.path.isfile(t):
        name = os.path.basename(t)
        m = re.search(r"[-_.]((?:I?Q\d[A-Z0-9_]*)|F16|BF16|F32)\.gguf$", name, re.I)
        return {"path": os.path.abspath(t), "name": name, "tag": m.group(1) if m else None, "repo": None,
                "url": None, "source": "file"}
    model, _, tag = target.rpartition(":")
    if not model:
        # a bare build name, searched across every MANIFEST in the workspace
        for _slug, b in _all_workspace_builds():
            if str(b.get("name", "")).lower() in (target.lower(), target.lower() + ".gguf"):
                return {"path": b.get("path"), "name": b.get("name"), "tag": b.get("tag"), "repo": None,
                        "url": None, "source": "workspace"}
        raise SystemExit(f"pollard serve: {target!r} is not a .gguf path, a MODEL:TAG workspace rung, "
                         "or an ORG/REPO:TAG published rung")
    b = _pick([x for x in _manifest_builds(model) if x.get("lane", "gguf") == "gguf"], tag)
    if b:
        return {"path": b.get("path"), "name": b.get("name"), "tag": b.get("tag") or tag, "repo": None,
                "url": None, "source": "workspace"}
    if "/" in model:
        try:
            files = hf_gguf_files(model, fetch)
        except Exception as e:                                                   # noqa: BLE001
            return {"path": None, "name": f"{model.split('/')[-1]}-{tag}.gguf", "tag": tag, "repo": model,
                    "url": None, "source": "hf", "error": f"could not list {model} on the Hub ({e})"}
        hit = [f for f in files if re.search(rf"[-_.]{re.escape(tag)}(-\d+-of-\d+)?\.gguf$", f, re.I)
               and "mmproj" not in f.lower()]
        if not hit:
            raise SystemExit(f"pollard serve: {model} has no {tag} GGUF (has: "
                             + ", ".join(os.path.basename(f) for f in files[:12]) + ")")
        f = sorted(hit)[0]
        selected = [f]
        split = re.fullmatch(r"(.*)-(\d+)-of-(\d+)\.gguf", f)
        if split:
            prefix, index, total = split.groups()
            count = int(total)
            # Bound the filename's count before allocating its expected sibling list.
            if not 1 <= int(index) <= count or count > len(files):
                raise SystemExit(f"pollard serve: incomplete split GGUF {f!r} in {model}")
            # Match this exact basename AND subdirectory. Other rungs or split counts
            # cannot supply a missing sibling, even when they share the model prefix.
            selected = [f"{prefix}-{i:0{len(index)}d}-of-{total}.gguf" for i in range(1, count + 1)]
            missing = set(selected) - set(files)
            if missing:
                raise SystemExit(f"pollard serve: incomplete split GGUF {f!r} in {model}"
                                 + f" (missing: {', '.join(sorted(missing))})")
            f = selected[0]
        return {"path": None, "name": os.path.basename(f), "tag": tag, "repo": model, "file_in_repo": f,
                "files_in_repo": selected,
                "url": f"https://huggingface.co/{model}/resolve/main/{f}", "source": "hf"}
    raise SystemExit(f"pollard serve: no build tagged {tag!r} in the workspace MANIFEST for {model!r}")


def is_decision(path, name):
    """A decision model by its GGUF (a `.decision.` head block, decision tags) or, for a file not on disk,
    by the names the family ships under -- pollard-modelkind's own test, plus clef."""
    if path and os.path.isfile(path):
        try:
            from pollard_modelkind import classify
            return bool(classify(path).get("decision"))
        except Exception:
            pass
    try:
        from pollard_modelkind import _DECISION_NAME
        if _DECISION_NAME.search(name or ""):
            return True
    except Exception:
        pass
    return bool(re.search(r"(^|[^a-z])clef([^a-z]|$)", (name or "").lower()))


def gguf_facts(src):
    """(runtime verdict, arch, trained ctx, diffusion) read off the header; ('unknown', ...) when unreadable.
    `src` may be a path or a Hub URL (range requests, a few MB)."""
    verdict, arch, n_train, diffusion = "unknown", "", None, False
    if not src:
        return verdict, arch, n_train, diffusion
    try:
        import pollard_ggufcompat as gc
        verdict, _reasons = gc.runtime_of(src)
        arch = gc.read_header(src)[0]
    except Exception:
        pass
    if src and os.path.isfile(src):
        try:
            from pollard_calc import read_gguf_meta
            meta = read_gguf_meta(src)
            arch = arch or str(meta.get("general.architecture", ""))
            n_train = meta.get(f"{arch}.context_length")
        except Exception:
            pass
    try:
        from pollard_modelkind import DIFFUSION_ARCHS
        diffusion = arch.lower().replace("_", "-") in DIFFUSION_ARCHS
    except Exception:
        pass
    return verdict, arch, (int(n_train) if n_train else None), diffusion


# -- runtime choice + preflight -----------------------------------------------------------------------

def choose_runtime(verdict, arch, server=None, exe=None, supports_arch=None):
    """(runtime name, binary or None, fatal message or None).

    The ordering is pollard_studio.runtimes.for_build's -- the same one Studio uses -- and the stock/ik
    verdict comes from the file's tensor types (pollard-ggufcheck), never its name. An ik_llama-only rung
    is NEVER handed to stock llama.cpp: it refuses every ggml type above 42 with "invalid ggml type", which
    reads as a broken build rather than the wrong runtime."""
    from pollard_studio import runtimes as rt
    exe = exe or (lambda name: rt._exe(rt.SPECS[name]))
    if supports_arch is None:
        from pollard_calc import binary_supports_arch as supports_arch
    if verdict == "fork" and not server:
        return None, None, (f"architecture `{arch}` is not in upstream llama.cpp; only its vendor fork loads it. "
                            "Pass --server <that fork's llama-server>.")
    order = [n for n in rt.for_build("gguf", "IQ_KT" if verdict == "ik_llama" else None)
             if n in ("llama.cpp", "ik_llama")]
    if verdict == "ik_llama":
        order = ["ik_llama"]
    elif verdict == "stock":
        # stock first even when ik_llama is installed: upstream carries /v1/messages and /v1/responses,
        # which the harnesses need and the fork is not known to.
        order.sort(key=lambda n: n != "llama.cpp")
    for name in order:
        binary = server or exe(name)
        if not binary:
            continue
        if name == "ik_llama" and "ik_llama" not in binary.replace("\\", "/").lower():
            return None, None, ("this rung carries ik_llama-only atoms and --server does not name an ik_llama "
                                "build; stock llama.cpp cannot load it. Point --server or IK_LLAMA_SERVER at "
                                "ik_llama's llama-server.")
        if arch and not supports_arch(binary, arch):
            return name, binary, (f"{binary} does not know architecture `{arch}`" +
                                  (" -- update llama.cpp (upstream has it)" if verdict == "newer" else
                                   " -- it needs llama.cpp with Pollard's runtime patch: `pollard-runtime --update`"
                                   if verdict == "pollard" else ""))
        return name, binary, None
    if verdict == "ik_llama":
        return "ik_llama", None, ("this rung carries ik_llama-only atoms (ggml types above 42); stock llama.cpp, "
                                  "Ollama and LM Studio cannot load it. Build ik_llama.cpp and set IK_LLAMA_SERVER "
                                  "to its llama-server.")
    return order[0] if order else "llama.cpp", None, "llama-server not found -- run ./install.sh or put it on PATH"


def server_cmd(binary, path, harness, ctx, port, host="127.0.0.1", ngl=99, alias=None, decision=False,
               supports=lambda flag: True, hf=None):
    """The llama-server argv for this harness. Optional flags are dropped (and reported) when the build
    does not take them -- an unknown flag is a hard startup failure, and the forks disagree on which exist."""
    h = HARNESSES[harness]
    argv = [binary or "llama-server"] + (["-hf", hf] if hf and not path else ["-m", path or "model.gguf"])
    argv += ["--host", host, "--port", str(port), "-c", str(ctx), "-ngl", str(ngl),
             # one slot: upstream's default makes several slots share one KV pool while each believes it
             # owns the whole context (ggml-org/llama.cpp#23652). A harness sizes its prompt to the
             # advertised context, which is exactly the overflow that aborts the server mid-session.
             "-np", "1"]
    dropped = []
    opt = []
    if alias:
        opt.append(["--alias", alias])
    if decision:
        # clef evaluates a whole request in one batch; pollard-decision serves it the same way
        argv += ["-b", str(ctx), "-ub", str(ctx)]
    else:
        opt.append(["--jinja"])                          # the model's own template: tool-call parsing needs it
        if h.get("reasoning"):
            opt.append(["--reasoning-format", h["reasoning"]])
        # ik defaults context-shift ON: an over-long agent prompt would be silently cut at the FRONT --
        # the system prompt and tool schemas -- instead of failing where the harness can see it
        opt.append(["--no-context-shift"])
    for o in opt:
        if supports(o[0]):
            argv += o
        else:
            dropped.append(o[0])
    return argv, dropped


def preflight(info, harness, ctx, verdict, arch, n_train, diffusion, decision, runtime, binary, fatal_rt):
    """(fatal, warnings). Cheap: nothing here loads weights."""
    fatal, warn = [], []
    h = HARNESSES[harness]
    if info.get("error"):
        # without the file listing there is no file to fetch or header to read
        fatal.append(info["error"])
    if info["source"] != "hf" and not (info.get("path") and os.path.isfile(info["path"])):
        fatal.append(f"no such file: {info.get('path')}")
    if diffusion:
        fatal.append(f"`{arch}` is a diffusion LLM: llama-server cannot decode it (llama-diffusion-cli is "
                     "one process per generation, not a server a harness can call)")
    if decision and not (harness == "omp"):
        fatal.append("this is a decision model: it answers /v1/systemone and cannot chat, so a chat harness "
                     "cannot drive it. Use --harness omp (the judge role, api: typesafe).")
    if h.get("stock_only") and verdict in ("ik_llama", "fork", "pollard"):
        fatal.append(f"{h['label']} runs its own bundled llama.cpp, which cannot load this rung "
                     f"({'ik_llama-only atoms' if verdict == 'ik_llama' else f'architecture `{arch}`'}). "
                     "Use a harness pointed at `pollard serve` instead.")
    if h["endpoint"] != "own" and fatal_rt:
        fatal.append(fatal_rt)
    if verdict == "unknown":
        warn.append("could not read the GGUF header, so the runtime (stock vs ik_llama) is unverified")
    if runtime == "ik_llama" and h["endpoint"] in ("anthropic", "responses"):
        warn.append(f"{h['label']} calls {ENDPOINTS[h['endpoint']]}; that endpoint is upstream llama.cpp's and "
                    "is unverified on ik_llama's server")
    if ctx < h["min_ctx"] and not decision:
        warn.append(f"context {ctx} is below {h['label']}'s working minimum {h['min_ctx']}: its prompt will be "
                    "truncated or rejected, usually without a clear error")
    if n_train and ctx > n_train:
        warn.append(f"context {ctx} is beyond the model's trained {n_train}; quality past {n_train} is "
                    "not what the model was trained for")
    return fatal, warn


# -- --write: merge into the harness's own config -----------------------------------------------------

def backup(path):
    """Copy the original beside itself before touching it; returns the backup path (None if no file)."""
    if not os.path.exists(path):
        return None
    dst = f"{path}.pollard-bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, dst)
    return dst


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".pollard-tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _yaml():
    import yaml                                             # PyYAML comes with the gguf dependency
    return yaml


def _write_yaml_merge(path, mutate):
    yaml = _yaml()
    data = {}
    if os.path.exists(path):
        data = yaml.safe_load(open(path, encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a YAML mapping; not touching it")
    mutate(data)
    bak = backup(path)
    _atomic_write(path, yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    return bak


def write_config(harness, c, home=None):
    """Merge the snippet into the harness's config. Returns [(path, backup)]. Raises on a format it will
    not merge safely (the caller prints instead)."""
    home = home or os.path.expanduser("~")
    h = HARNESSES[harness]
    if not h.get("write"):
        raise ValueError(f"{h['label']} is configured by {h['fmt']}, not a file this tool merges -- printed instead")
    p = lambda rel: os.path.join(home, rel)
    done = []
    if harness == "omp":
        prov = {"baseUrl": c["base"] if c["decision"] else c["base"] + "/v1",
                "api": "typesafe" if c["decision"] else "openai-completions", "auth": "none",
                "models": [{"id": c["model"], "name": f"{c['model']} (Pollard)"} |
                           ({} if c["decision"] else {"contextWindow": c["ctx"]})]}
        path = p(".omp/agent/models.yml")
        done.append((path, _write_yaml_merge(path, lambda d: d.setdefault("providers", {}).__setitem__(c["provider"], prov))))
        if c["decision"]:
            # the judge role IS the point of serving a decision model to OMP; a chat model's role is left
            # to the user, because writing `default` would switch every session's model
            cpath = p(".omp/agent/config.yml")
            done.append((cpath, _write_yaml_merge(cpath, lambda d: d.setdefault("modelRoles", {}).__setitem__(
                "judge", f"{c['provider']}/{c['model']}"))))
        return done
    if harness == "continue":
        path = p(".continue/config.yaml")
        entry = _continue_obj(c)

        def mut(d):
            models = [m for m in (d.get("models") or []) if not (isinstance(m, dict) and m.get("name") == entry["name"])]
            d["models"] = models + [entry]
        done.append((path, _write_yaml_merge(path, mut)))
        return done
    if harness == "opencode":
        path = p(".config/opencode/opencode.json")
        data = {}
        if os.path.exists(path):
            try:
                data = json.load(open(path, encoding="utf-8"))
            except ValueError:
                # opencode also reads JSONC; rewriting it through json would drop the comments
                raise ValueError(f"{path} is not plain JSON (comments?); not rewriting it -- printed instead")
        new = _opencode_obj(c)
        data.setdefault("$schema", new["$schema"])
        data.setdefault("provider", {})[c["provider"]] = new["provider"][c["provider"]]
        bak = backup(path)
        _atomic_write(path, json.dumps(data, indent=2) + "\n")
        return [(path, bak)]
    if harness == "codex":
        import tomllib
        path = p(".codex/config.toml")
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        cur = tomllib.loads(old) if old else {}
        if c["provider"] in (cur.get("model_providers") or {}) or c["provider"] in (cur.get("profiles") or {}):
            raise ValueError(f"{path} already has a `{c['provider']}` provider/profile; edit it by hand "
                             "(TOML tables cannot be re-declared, and rewriting the file would drop its comments)")
        add = "\n".join(l for l in _codex(c).splitlines() if not l.startswith("# ~/"))
        text = (old.rstrip("\n") + "\n\n" if old else "") + add.strip() + "\n"
        tomllib.loads(text)                                  # never write a file Codex cannot parse
        bak = backup(path)
        _atomic_write(path, text)
        return [(path, bak)]
    raise ValueError(f"no writer for {harness}")


# -- card ---------------------------------------------------------------------------------------------

def card_section(repo, name, tag=None, decision=False, stock_tag=None, needs=None, port=DEFAULT_PORT):
    """`## Use it in your harness` for pollard-card, from the same table the CLI prints.

    `stock_tag` is a rung stock llama.cpp opens -- the Ollama / LM Studio rows appear only with one, since
    both bundle stock llama.cpp. `needs` names the runtime the featured rung requires when stock cannot
    open it ("ik_llama.cpp", a vendor fork, "a current llama.cpp"); None = stock."""
    alias = os.path.splitext(name or "model.gguf")[0]
    out = ["## Use it in your harness", ""]
    if decision:
        c = context(alias, DECISION_CTX, port, repo=repo, tag=tag, file=name, decision=True)
        out += ["A decision model has no chat endpoint, so chat harnesses (Claude Code, Codex, OpenCode, ...) "
                "cannot drive it. It serves **`/v1/systemone`**, which is what "
                "[Oh My Pi](https://github.com/can1357/oh-my-pi)'s `api: typesafe` provider calls "
                "(`{baseUrl}/v1/systemone`) for its **`judge`** role:", "",
                "```yaml", render("omp", c).rstrip(), "```", "",
                "```bash", f"pollard serve {repo}:{tag} --harness omp" if tag else
                f"pollard serve {name} --harness omp", "```", ""]
        return out
    serve_ref = f"{repo}:{tag}" if tag else name
    out += ["Any agent harness that speaks the OpenAI, Anthropic or Responses API can drive this model through "
            "llama-server. [Pollard Weights](https://github.com/WestWaters/pollard-weights)' `pollard serve` "
            "picks the runtime this rung needs, sizes the context to the harness and prints its exact config "
            "(`--write` merges it in, after backing the original up):", "",
            "```bash", f"pollard serve {serve_ref} --harness claude-code      # or omp, codex, opencode, aider, ...",
            "```", ""]
    out += ["| harness | calls | min. context | setup |", "|---|---|---:|---|"]
    for key, h in HARNESSES.items():
        if h.get("stock_only"):
            if not stock_tag:
                continue
            setup = h["card"].replace("<repo>", repo).replace("<TAG>", stock_tag)
        else:
            setup = h["card"].replace("<name>", alias)
        ep = "own runtime" if h["endpoint"] == "own" else f"`{ENDPOINTS[h['endpoint']]}`"
        out.append(f"| {h['label']} | {ep} | {ctx_label(h['min_ctx'])} | {setup} |")
    out.append("")
    ctx = HARNESSES["claude-code"]["min_ctx"]
    if needs:
        out += [f"Without Pollard, start the llama-server from {needs} yourself (this rung needs it) with a "
                "context the harness fits in and the model's own template:", "", "```bash",
                f"llama-server -m {name} -c {ctx} --jinja --alias {alias} -np 1 --port {port}", "```", ""]
    else:
        out += ["Without Pollard, start llama-server yourself with a context the harness fits in, the model's "
                "own template (tool calls are parsed through it) and the thinking kept out of the answer:", "",
                "```bash",
                f"llama-server -hf {repo}:{tag} -c {ctx} --jinja --reasoning-format deepseek --alias {alias} "
                f"-np 1 --port {port}" if tag else
                f"llama-server -m {name} -c {ctx} --jinja --reasoning-format deepseek --alias {alias} -np 1 --port {port}",
                "```", ""]
    out += ["_Too small a context fails silently: the harness's system prompt and tool schemas are truncated and "
            "the model looks like it ignores its tools. Claude Code needs 64k._", ""]
    return out


# -- CLI ----------------------------------------------------------------------------------------------

def _wait_healthy(proc, base, timeout):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pollard serve", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("target", help="a .gguf path, a workspace rung MODEL:TAG, or a published rung ORG/REPO:TAG")
    ap.add_argument("--harness", default="omp", choices=sorted(HARNESSES))
    ap.add_argument("--ctx", type=int, default=0, help="context size (default: the harness's working minimum)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 for a harness in Docker (Open WebUI, OpenHands)")
    ap.add_argument("--ngl", type=int, default=99, help="layers on the GPU (0 = CPU only)")
    ap.add_argument("--server", help="the llama-server binary to use (else found: runtime/, $POLLARD_HOME/bin, PATH)")
    ap.add_argument("--decision", action="store_true", help="treat it as a decision model even if not detected")
    ap.add_argument("--write", action="store_true", help="merge the snippet into the harness's config file "
                    "(the original is copied to <file>.pollard-bak-<time> first)")
    ap.add_argument("--print-only", action="store_true", help="resolve, check and print; start nothing")
    ap.add_argument("--timeout", type=int, default=600, help="seconds to wait for the server to load")
    a = ap.parse_args(argv)
    try:
        # the config snippet has to reach a piped / redirected terminal while the server is still running
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    h = HARNESSES[a.harness]
    info = resolve(a.target)
    decision = a.decision or is_decision(info.get("path"), info.get("name"))
    verdict, arch, n_train, diffusion = gguf_facts(info.get("path") or info.get("url"))
    if h["endpoint"] == "own":
        runtime, binary, fatal_rt = None, None, None
    elif a.print_only and not a.server:
        # say what WOULD run without demanding it exist here: --print-only is also how a card is checked
        try:
            runtime, binary, fatal_rt = choose_runtime(verdict, arch, supports_arch=lambda b, x: True)
        except Exception:
            runtime, binary, fatal_rt = None, None, None
    else:
        runtime, binary, fatal_rt = choose_runtime(verdict, arch, a.server)
    ctx = a.ctx or (DECISION_CTX if decision else h["min_ctx"])
    alias = os.path.splitext(info.get("name") or "model")[0]
    c = context(alias, ctx, a.port, a.host, info.get("repo"), info.get("tag"),
                info.get("path") or info.get("name"), decision)
    fatal, warn = preflight(info, a.harness, ctx, verdict, arch, n_train, diffusion, decision, runtime, binary,
                            fatal_rt)

    print(f"== pollard serve :: {info.get('name')}  ->  {h['label']}")
    print(f"   runtime: {runtime or ('(the harness runs its own)' if h['endpoint'] == 'own' else '--')}"
          f"   verdict: {verdict}" + (f"   arch: {arch}" if arch else "") + f"   ctx: {ctx}"
          + ("   decision model -> /v1/systemone" if decision else ""))
    for w in warn:
        print(f"   WARNING: {w}")
    for f in fatal:
        print(f"   BLOCKED: {f}")
    if fatal and not a.print_only:
        return 2

    if decision and a.harness != "omp":
        return 0 if a.print_only else 2                  # nothing a chat harness could use: print no config

    argv_srv = None
    if h["endpoint"] != "own":
        supports = lambda flag: True
        if not a.print_only and binary:
            from pollard_bench import _server_supports
            supports = lambda flag: _server_supports(binary, flag)
        argv_srv, dropped = server_cmd(binary, info.get("path"), a.harness, ctx, a.port, a.host, a.ngl, alias,
                                       decision, supports)
        for d in dropped:
            print(f"   note: {binary} does not take {d}; left out")
        print("\n   " + " ".join(argv_srv))
    print(f"\n-- {h['label']}" + (f"  ({h['config']})" if h.get("config") else "") +
          ("" if h.get("verified") else "  [config keys not verified against this harness's docs]") + " --")
    print(render(a.harness, c))

    def do_write():
        try:
            for path, bak in write_config(a.harness, c):
                print(f"   wrote {path}" + (f"  (original kept at {bak})" if bak else "  (new file)"))
        except Exception as e:                                                   # noqa: BLE001
            print(f"   not written: {e}")

    if a.print_only or h["endpoint"] == "own":
        if a.write:
            do_write()
        return 1 if (fatal and not a.print_only) else 0

    path = info.get("path")
    if info["source"] == "hf" and not path:
        files = info.get("files_in_repo", [info["file_in_repo"]])
        print(f"   downloading {', '.join(files)} from {info['repo']} ...")
        if len(files) > 1:
            from huggingface_hub import snapshot_download
            # One snapshot keeps every shard at the same revision and in the same
            # directory, where llama-server discovers the siblings of shard one.
            snapshot = snapshot_download(info["repo"], allow_patterns=[glob.escape(f) for f in files])
            missing = [f for f in files if not os.path.isfile(os.path.join(snapshot, f))]
            if missing:
                raise SystemExit("pollard serve: incomplete split GGUF download (missing: "
                                 + ", ".join(missing) + ")")
            path = os.path.join(snapshot, info["file_in_repo"])
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(info["repo"], info["file_in_repo"])
        argv_srv[argv_srv.index("-m") + 1] = path
    log = os.path.join(os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard"), "cache",
                       f"serve-{a.port}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        # the log, not DEVNULL: a server that dies on load says exactly why ("invalid ggml type", "failed to
        # allocate", "unknown model architecture") and that line is the whole diagnosis
        proc = subprocess.Popen(argv_srv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    base = c["base"]
    if not _wait_healthy(proc, base, a.timeout):
        proc.terminate()
        tail = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-8:]
        print("   llama-server did not come up. Last lines:\n     " + "\n     ".join(tail))
        return 1
    print(f"   serving at {base} ({ENDPOINTS[h['endpoint']]})  -- log: {log}  -- Ctrl-C to stop")
    if a.write:
        do_write()
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
