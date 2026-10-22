#!/usr/bin/env python3
"""pollard-modelkind -- what KIND of model is this, so the rest of Pollard stops guessing.

A build is measured and gated as though every model were a plain text model. It is not. A
reasoning-tuned model opens with a thinking block and only answers afterwards; an agentic model is
trained to emit tool calls; an instruct model was tuned away from raw-text modelling entirely. Score
one of those on raw Wikipedia and the numbers are meaningless -- gemma-4-12B-it reports perplexity
~664 there while a plain 7B reports 5.4 on the same corpus and binary -- and gate it with a short
token budget and it gets cut off mid-thought and fails a build that was about to answer correctly.

None of that needs guessing: the chat template says so. It is shipped in the GGUF and in the HF
checkout, and it names the turn structure the model was trained on.

    pollard-modelkind --model <gguf-or-hf-dir>            # what it is, and what to measure it with

Everything here is model-side and self-contained.
"""
from __future__ import annotations

import argparse, json, os, re

# GGUF general.architecture / HF model_type values that llama.cpp decodes by diffusion (llama_model_is_diffusion).
# Normalised with "_" -> "-" so "diffusion_gemma" (HF) and "diffusion-gemma" (GGUF) both match.
DIFFUSION_ARCHS = frozenset({"dream", "llada", "llada-moe", "rnd1", "diffusion-gemma"})

# How many tokens the coherence gate must allow. A thinking model spends its first hundred-odd
# tokens reasoning, so a short budget reads as incoherence. 220 was not enough: LFM2.5-24B-A2B's IQ3_S
# (2026-10-17) was still "Analyze the user's input" at token 220 and read as BELOW FLOOR, then passed
# the same gate at 1024. A thinking model gets the tool-call test's budget (pollard-toolcall's
# N_PREDICT_REASONING): the same room to think before it answers.
TOKENS_PLAIN, TOKENS_THINKING = 96, 1024


def _template_and_arch(model):
    """(chat_template, architectures, config) from a GGUF or an HF checkout."""
    if str(model).endswith(".gguf") and os.path.isfile(model):
        try:
            from pollard_calc import read_gguf_meta
            m = read_gguf_meta(model)
            arch = m.get("general.architecture", "")
            return (m.get("tokenizer.chat_template", "") or "",
                    [arch] if arch else [], {k: v for k, v in m.items() if isinstance(k, str)})
        except Exception:
            return "", [], {}
    tpl = ""
    # chat_template.jinja is a plain file, not JSON -- newer checkouts ship it that way and reading
    # only the JSON ones reports a chat model as a base model.
    jinja = os.path.join(model, "chat_template.jinja")
    if os.path.isfile(jinja):
        try:
            tpl = open(jinja, encoding="utf-8").read()
        except OSError:
            pass
    for name in ("tokenizer_config.json", "chat_template.json"):
        q = os.path.join(model, name)
        if os.path.isfile(q):
            try:
                d = json.load(open(q, encoding="utf-8"))
                tpl = d.get("chat_template") or tpl
            except (ValueError, OSError):
                pass
    archs, conf = [], {}
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        try:
            conf = json.load(open(cfg, encoding="utf-8"))
            archs = list(conf.get("architectures") or [])
            archs += list((conf.get("text_config") or {}).get("architectures") or [])
        except (ValueError, OSError):
            pass
    return tpl or "", archs, conf



def _modality_evidence(model_dir):
    """(encoder blocks per modality, projection tensors per modality). None if weights unreadable.

    Config flags say what a family supports; the checkpoint says what THIS release shipped -- and a
    model can ship a modality two ways: an encoder tower, or (encoder-free) projection weights."""
    import glob
    pats = {"image": ("vision", "visual", "image", "vit"),
            "audio": ("audio", "speech", "whisper"),
            "video": ("video", "temporal")}
    counts = {k: 0 for k in pats}
    layers = {k: set() for k in pats}
    projs = {k: 0 for k in pats}
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not files:
        return None, None
    try:
        from safetensors import safe_open
    except ImportError:
        return None, None
    try:
        for f in files:
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    low = k.lower()
                    if "language_model" in low:
                        continue                       # the text stack, whatever else it mentions
                    for m, words in pats.items():
                        if any(w in low for w in words):
                            counts[m] += 1
                            # An encoder TOWER is repeated blocks. A projection layer and a patch
                            # embedder are not an encoder, however many tensors they add up to.
                            hit = re.search(r"(?:layers?|blocks?|h)\.(\d+)\.", low)
                            if hit:
                                layers[m].add(int(hit.group(1)))
                            if re.search(r"embed|proj|patch", low):
                                projs[m] += 1
    except Exception:
        return None, None
    return ({m: (len(layers[m]) if layers[m] else 0) for m in counts}, projs)


# ----------------------------------------------------------------- reasoning (thinking) models
#
# A reasoning model emits a thinking block before every answer, and that block is most of the tokens it
# ever produces. Calibrate it on reference answers with no thinking in them and the imatrix protects the
# tensors of a model that does not exist (the LFM2.5-24B build: zero `<think>` blocks in the corpus,
# although the model always reasons). So the template is asked, not guessed:
#
#   1. RENDER it (jinja2, no tokenizer needed): an assistant turn carrying reasoning_content / thinking /
#      reasoning, and the text around that marker IS the model's own delimiter, exactly as it emits it
#      (`<think>\n`, `<seed:think>`, `[THINK]`, `<|channel|>analysis<|message|>`, `<|START_THINKING|>`...).
#   2. Render the generation prompt with each toggle variable the template reads (enable_thinking,
#      thinking, thinking_mode, reasoning_effort, ...) on and off: a prompt that changes is a toggle; a
#      prompt that already ends in the opener is always-on.
#   3. Static fallbacks when jinja2 is missing or the template will not render: known delimiter pairs,
#      the variables named in the text, then names/tags/config (Qwen3 / R1 / QwQ / Magistral / gpt-oss /
#      LFM2.5 / Ring / Kimi-Thinking / Phi-reasoning ...), with the known NON-thinking releases excluded.

#: (opener, closer) pairs seen in shipped templates -- the static fallback when rendering is unavailable
THINK_PAIRS = [
    ("<|channel|>analysis<|message|>", "<|end|>"),           # gpt-oss / harmony (closer then opens final)
    ("<seed:think>", "</seed:think>"),                       # ByteDance Seed-OSS
    ("<mm:think>", "</mm:think>"),                           # MiniMax M3
    ("<ifm|think>", "</ifm|think>"),                         # IFM K2 Horizon
    ("<|START_THINKING|>", "<|END_THINKING|>"),              # Cohere Command A reasoning
    ("<|channel>thought", "<channel|>"),                     # Gemma 4
    ("[THINK]", "[/THINK]"),                                 # Magistral / Ministral reasoning
    ("\u25c1think\u25b7", "\u25c1/think\u25b7"),                     # Kimi-VL thinking (left/right triangles)
    ("<thinking>", "</thinking>"),                           # Apriel
    ("<reasoning>", "</reasoning>"),
    ("<think>", "</think>"),                                  # Qwen3 / R1 / GLM / LFM2.5 / Phi-4-reasoning / ...
]
HARMONY_FINAL = "<|start|>assistant<|channel|>final<|message|>"
#: template variables that switch thinking (bool-like) or grade it (effort)
TOGGLE_VARS = ("enable_thinking", "thinking", "thinking_mode", "reasoning", "think", "enable_reasoning",
               "use_thinking", "is_thinking")
EFFORT_VARS = ("reasoning_effort", "thinking_budget", "thinking_effort")
#: names that are reasoning models even when their template says little (R1-distill strips the block)
_REASONING_NAME = re.compile(
    r"thinking|thinker|reasoning|reasoner|(^|[-_/])r1([-_.]|$)|qwq|magistral|phi-?4-?(mini-)?(flash-)?reasoning|"
    r"(^|[-_/])ring[-_]|gpt-?oss|openthinker|deepscaler|skywork-or1|acereason|olmo-?3.*think|"
    r"lfm2\.?5|k2-?thinking|kimi.*thinking|glm-?(4\.[5-9]|5)|seed-?oss|nemotron.*(nano-?v2|super|ultra)|"
    r"exaone-?4|hunyuan|minimax-?m[123]|step-?3", re.I)
#: releases that are explicitly NOT thinking although their family is (Qwen3-*-Instruct-2507 etc.)
_NON_REASONING_NAME = re.compile(r"instruct-?2507|non-?thinking|no-?think|nothink|(^|[-_])coder([-_]|$)", re.I)
#: HF card tags
_REASONING_TAGS = ("reasoning", "thinking", "chain-of-thought", "cot", "reasoning-model", "deepseek-r1")
_MARK_R, _MARK_A, _MARK_U = "RZQXREASON", "AZQXANSWER", "UZQXUSER"


def _jinja_env(mutable=False):
    """A jinja2 environment that renders HF chat templates the way transformers does (no tokenizer).

    `mutable=True` allows list.append / dict.update inside the template (still sandboxed): a few shipped
    templates (Kimi-K2) build lists that way, which llama.cpp's own jinja engine accepts and transformers'
    immutable sandbox refuses."""
    import jinja2
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment, SandboxedEnvironment

    class _Generation(Extension):                       # transformers' {% generation %} block
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return jinja2.nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    def strftime_now(fmt):
        import datetime
        return datetime.datetime.now().strftime(fmt)

    cls = SandboxedEnvironment if mutable else ImmutableSandboxedEnvironment
    env = cls(trim_blocks=True, lstrip_blocks=True, extensions=[_Generation, jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def _special_tokens(model, conf):
    """bos/eos strings the template may reference."""
    out = {"bos_token": "", "eos_token": ""}
    if isinstance(conf, dict):                                       # a GGUF has no strings for them here
        pass
    if os.path.isdir(str(model)):
        q = os.path.join(model, "tokenizer_config.json")
        if os.path.isfile(q):
            try:
                d = json.load(open(q, encoding="utf-8"))
                for k in out:
                    v = d.get(k)
                    out[k] = (v.get("content") if isinstance(v, dict) else v) or ""
            except (ValueError, OSError):
                pass
    return out


class ChatTemplate:
    """A model's own chat template rendered with jinja2 -- no transformers, no tokenizer download.

    The calibration renderer falls back to this when transformers is absent, and the reasoning probe
    always uses it. `render(messages, add_generation_prompt, tools, **template_vars)`."""

    def __init__(self, source, specials=None, mutable=False):
        self.source = source
        self.specials = dict(specials or {})
        self._tpl = _jinja_env(mutable).from_string(source)

    @classmethod
    def from_model(cls, model):
        tpl, _, conf = _template_and_arch(model)
        if not tpl:
            return None
        if isinstance(tpl, list):                                   # named templates: take "default"
            tpl = next((t.get("template") for t in tpl if t.get("name") == "default"), tpl[0].get("template"))
        try:
            return cls(tpl, _special_tokens(model, conf))
        except Exception:                                            # noqa: BLE001 -- jinja missing / bad syntax
            return None

    def variables(self):
        """Undeclared variables the template reads (enable_thinking, reasoning_effort, ...)."""
        try:
            from jinja2 import meta
            return set(meta.find_undeclared_variables(_jinja_env().parse(self.source)))
        except Exception:                                            # noqa: BLE001
            return set()

    def render(self, messages, add_generation_prompt=False, tools=None, **kw):
        ctx = dict(self.specials)
        ctx.update(kw)
        return self._tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                **ctx)


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:                                                # noqa: BLE001
        return None


def _probe_template(ct):
    """What rendering says: delimiters around the reasoning, the generation-prompt prefill, toggles."""
    out = {"rendered": False, "open": None, "mid": None, "close": None, "toggle": None,
           "on": None, "off": None, "default_on": None, "prefill": "", "effort": None}
    user = {"role": "user", "content": _MARK_U}
    asst = {"role": "assistant", "content": _MARK_A, "reasoning_content": _MARK_R, "thinking": _MARK_R,
            "reasoning": _MARK_R}
    tv = ct.variables()
    toggles = [v for v in TOGGLE_VARS if v in tv]
    efforts = [v for v in EFFORT_VARS if v in tv]
    out["effort"] = efforts[0] if efforts else None
    base = _safe(ct.render, [user], add_generation_prompt=True)
    # Which variable, with which values, changes the generation prompt? That is the toggle.
    for var in toggles:
        on = _safe(ct.render, [user], add_generation_prompt=True, **{var: True})
        off = _safe(ct.render, [user], add_generation_prompt=True, **{var: False})
        if on is not None and off is not None and on != off:
            out.update(toggle=var, on={var: True}, off={var: False})
            out["default_on"] = (base == on) if base is not None else None
            break
    if out["toggle"] is None and toggles:
        # Some templates switch on a string mode ("thinking_mode": "thinking"/"chat").
        for var in toggles:
            for on_v, off_v in (("thinking", "chat"), ("on", "off"), ("enabled", "disabled")):
                on = _safe(ct.render, [user], add_generation_prompt=True, **{var: on_v})
                off = _safe(ct.render, [user], add_generation_prompt=True, **{var: off_v})
                if on is not None and off is not None and on != off:
                    out.update(toggle=var, on={var: on_v}, off={var: off_v},
                               default_on=(base == on) if base is not None else None)
                    break
            if out["toggle"]:
                break
    if out["toggle"] is None and out["effort"] and "<|channel|>" not in ct.source:
        # an effort grade that also has an OFF level (Hy3: reasoning_effort no_think / low / high)
        var = out["effort"]
        hi = _safe(ct.render, [user], add_generation_prompt=True, **{var: "high"})
        for off_v in ("no_think", "none", "off", "disabled"):
            off = _safe(ct.render, [user], add_generation_prompt=True, **{var: off_v})
            if hi is not None and off is not None and off != hi and off_v in ct.source:
                out.update(toggle=var, on={var: "high"}, off={var: off_v},
                           default_on=(base == hi) if base is not None else None)
                break
    on_kw = dict(out["on"] or {})
    if out["effort"] == "reasoning_effort":
        on_kw["reasoning_effort"] = "high"
    out["on_kw"] = on_kw
    gen = _safe(ct.render, [user], add_generation_prompt=True, **on_kw) or base or ""
    full = _safe(ct.render, [user, asst], add_generation_prompt=False, **on_kw)
    if full is None:                                                 # a template that rejects our extra keys
        full = _safe(ct.render, [user, {"role": "assistant", "content": _MARK_A,
                                        "reasoning_content": _MARK_R}], add_generation_prompt=False, **on_kw)
    if full and _MARK_R in full and _MARK_A in full and full.index(_MARK_R) < full.index(_MARK_A):
        out["rendered"] = True
        head = full[:full.index(_MARK_R)]
        if gen and head.startswith(gen):
            out["open"] = head[len(gen):]               # what the model itself emits before reasoning
        elif _MARK_U in head:                           # header differs: take what follows the user turn
            out["open"] = head[head.rfind(_MARK_U) + len(_MARK_U):][-48:]
        out["mid"] = full[full.index(_MARK_R) + len(_MARK_R):full.index(_MARK_A)]
    out["gen_tail"] = (gen or "")[-200:]
    # The generation prompt may already contain the opener (always-on: R1-0528, Qwen3-Thinking-2507).
    tail = (gen or "").rstrip()
    for o, _ in THINK_PAIRS:
        if tail.endswith(o):
            out["prefill"] = o
            break
    return out


def _derive_open(close):
    """The opener that pairs with a closer: </x> -> <x>, [/X] -> [X], <|close|>x -> <|open|>x, END -> START."""
    if not close:
        return None
    if close.startswith("</"):
        return "<" + close[2:]
    if close.startswith("[/"):
        return "[" + close[2:]
    for a, b in (("close", "open"), ("END", "START"), ("end", "begin")):
        if a in close:
            return close.replace(a, b, 1)
    return None


def _static_pair(low_tpl, tpl):
    for o, c in THINK_PAIRS:
        # both halves, or a closer that names thinking itself (LFM2.5 only ever strips `</think>`);
        # a bare `<|end|>` is every Phi template's end-of-turn, not a reasoning channel
        if o.lower() in low_tpl and (c.lower() in low_tpl or "think" in o.lower()):
            return o, c
        if "think" in c.lower() and c.lower() in low_tpl:
            return o, c
    m = re.search(r"</([a-z:|_]*think[a-z_]*)>", tpl, re.I)
    if m:
        return f"<{m.group(1)}>", f"</{m.group(1)}>"
    return None, None


def _card_tags(model):
    tags = []
    if os.path.isdir(str(model)):
        rd = os.path.join(model, "README.md")
        if os.path.isfile(rd):
            try:
                head = open(rd, encoding="utf-8", errors="replace").read(4000)
                fm = head.split("---")[1] if head.startswith("---") and head.count("---") >= 2 else ""
                tags += re.findall(r"-\s*([a-z0-9_\-]+)", fm.lower())
            except OSError:
                pass
    return tags


def _model_names(model, conf):
    """Every name this checkpoint goes by: the path's last part, an HF-cache repo name (a snapshot dir is
    a hash), the GGUF general.* names and the config's _name_or_path."""
    out = []
    p = str(model or "").rstrip("/\\")
    if p:
        out.append(os.path.basename(p))
        m = re.search(r"models--([^/\\]+?)--([^/\\]+)", p)
        if m:
            out.append(m.group(2))
    for k in ("general.name", "general.basename", "general.finetune", "_name_or_path"):
        v = (conf or {}).get(k)
        if v:
            out.append(os.path.basename(str(v).rstrip("/\\")))
    return [x for x in out if x]


def reasoning_info(model, tpl=None, archs=None, conf=None) -> dict:
    """Is this a reasoning (thinking) model, how does it delimit the thinking, and can it be switched off?

    {"reasoning": bool, "mode": "always"|"toggle"|"none", "style": "harmony"|"tag",
     "open": str, "close": str, "mid": str (closer + separator before the answer, as rendered),
     "prefill": opener the generation prompt already contains, "toggle": {"var","on","off","default_on"},
     "effort": var, "source": "render"|"static"|"name", "why": [...]}"""
    if tpl is None:
        tpl, archs, conf = _template_and_arch(model)
    tpl = tpl if isinstance(tpl, str) else ""
    conf = conf or {}
    low = tpl.lower()
    why = []
    info = {"reasoning": False, "mode": "none", "style": None, "open": None, "close": None, "mid": None,
            "prefill": "", "toggle": None, "effort": None, "source": None, "why": why}
    names_base = " ".join(_model_names(model, conf))
    negative = bool(_NON_REASONING_NAME.search(names_base))
    gencfg = {}
    if os.path.isdir(str(model)):
        p = os.path.join(model, "generation_config.json")
        if os.path.isfile(p):
            try:
                gencfg = json.load(open(p, encoding="utf-8"))
            except (ValueError, OSError):
                gencfg = {}

    probe = None
    ct = None
    if tpl:
        try:
            ct = ChatTemplate(tpl, _special_tokens(model, conf))
        except Exception:                                            # noqa: BLE001 -- no jinja2 / bad syntax
            ct = None
        if ct is not None:
            probe = _safe(_probe_template, ct)
    harmony = "<|channel|>" in tpl and "analysis" in low
    s_open, s_close = _static_pair(low, tpl)
    # without rendering: variables the template TESTS (bare "think"/"thinking"/"reasoning" are words in tags
    # and prose -- `</think>` is not a switch -- so those count only in a jinja test)
    vars_named = [v for v in TOGGLE_VARS + EFFORT_VARS if len(v) > 9 and re.search(r"(?<![\w<>/|:])%s(?![\w>|])" % v, tpl)]
    vars_named += [v for v in ("thinking", "reasoning") if re.search(
        r"\b(if|elif|and|or|not)\s+(not\s+)?%s\b(?!\s*\.)|(?<![.\w])%s\s+is\s+(not\s+)?defined|(?<![.\w])%s\s*[!=]="
        % (v, v, v), tpl)]

    if harmony:
        info.update(style="harmony", open="<|channel|>analysis<|message|>", close="<|end|>",
                    mid="<|end|>" + HARMONY_FINAL)
    elif probe and probe.get("rendered") and (probe.get("mid") or "").strip():
        o = (probe.get("open") or "").strip() or None
        c = probe["mid"].strip()
        # what rendered may carry more than the delimiter (a response header after the closer, a turn
        # header before the opener): keep the known delimiter when it is inside, the rendered text if not
        if s_close and s_close in c:
            c = s_close
        if s_open and (o is None or s_open in o):
            o = s_open
        info.update(style="tag", open=o or (probe.get("prefill") or None), close=c, mid=probe["mid"],
                    source="render")
    elif s_close:
        info.update(style="tag", open=s_open, close=s_close, source="static")
    if info["close"] and not info["open"]:
        info["open"] = _derive_open(info["close"].split("<|open|>")[0] if "<|open|>" in info["close"]
                                    else info["close"])
    if probe:
        info["prefill"] = probe.get("prefill") or ""
        if not info["prefill"] and info["open"] and (probe.get("gen_tail") or "").rstrip().endswith(
                info["open"].strip()):
            info["prefill"] = info["open"]
        if probe.get("toggle"):
            info["toggle"] = {"var": probe["toggle"], "on": probe["on"], "off": probe["off"],
                              "default_on": probe.get("default_on")}
        info["effort"] = probe.get("effort")
    elif vars_named:
        tv = [v for v in vars_named if v in TOGGLE_VARS]
        if tv and info["close"]:
            info["toggle"] = {"var": tv[0], "on": {tv[0]: True}, "off": {tv[0]: False}, "default_on": None}
        ev = [v for v in vars_named if v in EFFORT_VARS]
        info["effort"] = ev[0] if ev else None

    tags = [t for t in _card_tags(model) if t in _REASONING_TAGS]
    named = bool(_REASONING_NAME.search(names_base))
    has_delims = bool(info["close"])
    if harmony:
        info.update(reasoning=True, mode="always", source=info["source"] or "render")
        why.append("harmony channels (analysis -> final): reasons before every answer"
                   + (f"; graded by {info['effort']}" if info["effort"] else ""))
    elif info["toggle"] and has_delims:
        info.update(reasoning=True, mode="toggle")
        t = info["toggle"]
        why.append(f"template switches thinking with `{t['var']}`"
                   + ("" if t.get("default_on") is None else f" (default {'on' if t['default_on'] else 'off'})"))
    elif info["prefill"]:
        info.update(reasoning=True, mode="always")
        why.append(f"generation prompt already opens the thinking block ({info['prefill']!r}): always-on")
    elif has_delims and negative:
        why.append(f"template knows {info['close']!r} but the release name says non-thinking")
    elif has_delims and (named or tags or not negative):
        # The template handles thinking (strips it from history, renders reasoning_content) and nothing
        # turns it off: the model emits it itself. LFM2.5, Kimi-K2-Thinking, MiniMax-M2, StepFun 3.5.
        info.update(reasoning=True, mode="always")
        why.append(f"template handles {info['open'] or ''}...{info['close']} with no switch to turn it off "
                   "-> the model opens it itself")
    elif named or tags:
        info.update(reasoning=True, mode="always", style="tag", open="<think>", close="</think>",
                    source="name")
        why.append("name/tags read as a reasoning release" + (f" (tags: {', '.join(tags)})" if tags else "")
                   + "; delimiters assumed <think>...</think>")
    if info["reasoning"] and negative and not (harmony or info["toggle"] or info["prefill"]):
        info.update(reasoning=False, mode="none")
        why.append("overridden: release name marks it non-thinking")
    if not info["reasoning"]:
        for k in ("enable_thinking", "thinking", "reasoning"):
            if gencfg.get(k) or conf.get(k):
                info.update(reasoning=True, mode="always", style="tag",
                            open=info["open"] or "<think>", close=info["close"] or "</think>", source="config")
                why.append(f"generation/model config sets {k}")
                break
    raw = (probe or {}).get("open") if (probe or {}).get("rendered") else None
    info["open_raw"] = raw if (raw and info["open"] and info["open"] in raw) else (
        (info["open"] + ("\n" if info["style"] == "tag" else "")) if info["open"] else "")
    if info["reasoning"] and not info["mid"] and info["close"]:
        # the common emission (Qwen3 / R1 / GLM): "<think>\n" thinking "\n</think>\n\n" answer
        info["mid"] = ("\n" + info["close"] + "\n\n") if info["style"] == "tag" else info["close"]
    if info["reasoning"] and info["style"] == "tag" and info["open"] and not info["source"]:
        info["source"] = "static"
    return info


def split_reasoning(text, info=None):
    """(reasoning, answer, unclosed) from raw generated text, for ANY delimiter style.

    `unclosed` = a thinking block opened and never closed -- the budget ran out mid-thought, so there is
    no answer to score. A closer with no opener (the opener was prefilled by the template) still splits."""
    t = text or ""
    pairs = []
    if info and info.get("close"):
        if info.get("style") == "harmony":
            pairs.append(("<|channel|>analysis<|message|>", "<|end|>"))
        else:
            pairs.append((info.get("open") or "", info["close"]))
    pairs += [p for p in THINK_PAIRS if p not in pairs]
    if "<|channel|>" in t:                                           # harmony: analysis ... final
        m = re.search(r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", t, re.S)
        r = " ".join(re.findall(r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)(?:<\|end\|>|$)", t, re.S))
        if m:
            return r.strip(), m.group(1).strip(), False
        if r:
            return r.strip(), "", True
    for o, c in pairs:
        if c and c in t:
            head, _, tail = t.rpartition(c)
            if o and o in head:
                head = head.split(o, 1)[1] if head.count(o) == 1 else head.replace(o, "")
            return head.strip(), tail.strip(), False
        if o and o in t:
            return t.split(o, 1)[1].strip(), t.split(o, 1)[0].strip(), True
    return "", t.strip(), False


def strip_reasoning(text, info=None):
    """The answer only -- every thinking block removed, whatever the model's delimiters."""
    return split_reasoning(text, info)[1]


# ----------------------------------------------------------------- tool calling (the parser family)
#
# A model that calls tools writes the call in ITS OWN syntax -- Hermes `<tool_call>{json}</tool_call>`, Qwen3-Coder
# XML, LFM2's pythonic list, gpt-oss harmony channels, Mistral `[TOOL_CALLS]` -- and the server has to be told
# which parser reads it (vLLM `--tool-call-parser`, llama.cpp `--jinja` + its own template sniffing). Users mis-set
# this constantly, and a gate that grades calls in a format the model was never trained on measures the prompt.
# So the template is asked, the way llama.cpp asks it (common/chat.cpp):
#
#   1. SNIFF the template source for the markers llama.cpp's specialised handlers key on (same order, first match
#      wins: Ministral-3, gpt-oss, Muse, Functionary v3.2, Kimi-K2, Kimi-K3, Cohere2, LFM2, LFM2.5, GigaChat,
#      MiniMax-M3, DeepSeek-V3.2/V4, Gemma 4, MiniCPM5, Qwen3-Coder), then markers unique to other families.
#   2. RENDER it (jinja2): one dummy tool, one assistant turn carrying a dummy `tool_calls`, a tool result. The text
#      between the user turn and the tool result IS the model's call syntax, the same diff llama.cpp's autoparser
#      runs; the rendered call is classified by signature. Rendering the tool list with and without tools says
#      whether the template describes tools at all, and where it puts them.
#   3. A template that renders no call (some drop `tool_calls` and let the model write calls into content: LFM2,
#      Granite 3.x, R1-distill) keeps the sniffed family; one with neither is `none`.

#: family -> how it is served and written. vllm = `--tool-call-parser` value; vllm_ok True = in vLLM's documented
#: parser list, False = in vLLM's source tree (CLI name or model mapping unconfirmed), None = no vLLM parser.
#: shape = where the name/arguments live: json (name inside one JSON object), tag+json (name in markup, JSON args),
#: tag+tagged (one markup element per argument), pythonic (`[fn(a=1)]`), kv (Gemma `{a:<|"|>x<|"|>}`).
TOOL_FAMILIES = {
    "hermes": dict(vllm="hermes", vllm_ok=True, shape="json",
                   fmt='<tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>'),
    "qwen3_coder": dict(vllm="qwen3_coder", vllm_ok=True, alt="qwen3_xml", shape="tag+tagged",
                        fmt="<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>"),
    "lfm2": dict(vllm="lfm2", vllm_ok=False, shape="pythonic",
                 fmt='<|tool_call_start|>[NAME(key="value")]<|tool_call_end|>'),
    "harmony": dict(vllm="openai", vllm_ok=True, shape="tag+json",
                    fmt="<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{...}<|call|>"),
    "mistral": dict(vllm="mistral", vllm_ok=True, shape="tag+json",
                    fmt='[TOOL_CALLS]NAME[ARGS]{...}  (tokenizer v11+)  |  [TOOL_CALLS][{"name": NAME, "arguments": {...}}]'),
    "llama3_json": dict(vllm="llama3_json", vllm_ok=True, shape="json",
                        fmt='{"name": NAME, "parameters": {...}}'),
    "pythonic": dict(vllm="pythonic", vllm_ok=True, alt="llama4_pythonic", shape="pythonic",
                     fmt='[NAME(key="value")]'),
    "xlam": dict(vllm="xlam", vllm_ok=True, shape="json", fmt='[{"name": NAME, "arguments": {...}}]'),
    "deepseek_v3": dict(vllm="deepseek_v3", vllm_ok=True, shape="tag+json",
                        fmt="<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>function<\uff5ctool\u2581sep\uff5c>NAME\n```json\n{...}\n```"
                            "<\uff5ctool\u2581call\u2581end\uff5c><\uff5ctool\u2581calls\u2581end\uff5c>"),
    "deepseek_v31": dict(vllm="deepseek_v31", vllm_ok=True, shape="tag+json",
                         fmt="<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>NAME<\uff5ctool\u2581sep\uff5c>{...}<\uff5ctool\u2581call\u2581end\uff5c>"
                             "<\uff5ctool\u2581calls\u2581end\uff5c>"),
    "deepseek_v32": dict(vllm="deepseek_v32", vllm_ok=False, shape="tag+tagged",
                         fmt='<\uff5cDSML\uff5cfunction_calls|tool_calls><\uff5cDSML\uff5cinvoke name="NAME"><\uff5cDSML\uff5cparameter name="KEY" '
                             'string="true">VALUE</\uff5cDSML\uff5cparameter></\uff5cDSML\uff5cinvoke>...'),
    "glm45": dict(vllm="glm45", vllm_ok=True, shape="tag+tagged",
                  fmt="<tool_call>NAME\n<arg_key>KEY</arg_key>\n<arg_value>VALUE</arg_value>\n</tool_call>"),
    "glm47": dict(vllm="glm47", vllm_ok=True, shape="tag+tagged",
                  fmt="<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>"),
    "kimi_k2": dict(vllm="kimi_k2", vllm_ok=True, shape="tag+json",
                    fmt="<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:0<|tool_call_argument_begin|>{...}"
                        "<|tool_call_end|><|tool_calls_section_end|>"),
    "kimi_k3": dict(vllm="kimi_k3", vllm_ok=False, shape="tag+tagged",
                    fmt='<|open|>call tool="NAME" index="1"<|sep|><|open|>argument key="KEY" ...<|sep|>VALUE<|close|>argument'),
    "granite": dict(vllm="granite", vllm_ok=True, shape="json",
                    fmt='<|tool_call|>[{"name": NAME, "arguments": {...}}]'),
    # Granite 3.0-3.2 instruct: the same template text (it never renders tool_calls back), but the model writes the
    # call as PLAIN `<tool_call>` + a JSON list (granite-3.1-1b-a400m, measured) -- llama-server's autoparser reads
    # neither, vLLM's granite parser reads both spellings
    "granite_tc": dict(vllm="granite", vllm_ok=True, shape="json",
                       fmt='<tool_call>[{"arguments": {...}, "name": NAME}]'),
    "functiongemma": dict(vllm="functiongemma", vllm_ok=True, shape="kv",
                          fmt="<start_function_call>call:NAME{key:<escape>value<escape>}<end_function_call>"),
    "gemma4": dict(vllm="gemma4", vllm_ok=False, shape="kv",
                   fmt='<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>'),
    "cohere": dict(vllm="cohere_command3", vllm_ok=False, shape="json",
                   fmt='<|START_ACTION|>[{"tool_call_id": "0", "tool_name": NAME, "parameters": {...}}]<|END_ACTION|>'),
    "cohere_cmdr": dict(vllm=None, vllm_ok=None, shape="json",
                        fmt='Action: ```json\n[{"tool_name": NAME, "parameters": {...}}]\n```'),
    "minimax_m1": dict(vllm="minimax", vllm_ok=False, shape="json",
                       fmt='<tool_calls>\n{"name": NAME, "arguments": {...}}\n</tool_calls>'),
    "minimax_m2": dict(vllm="minimax_m2", vllm_ok=False, shape="tag+tagged",
                       fmt='<minimax:tool_call>\n<invoke name="NAME">\n<parameter name="KEY">VALUE</parameter>\n</invoke>\n'
                           '</minimax:tool_call>'),
    "minimax_m3": dict(vllm="minimax_m3", vllm_ok=False, shape="tag+tagged",
                       fmt=']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="NAME">]<]minimax[>[<KEY>VALUE...'),
    "seed_oss": dict(vllm="seed_oss", vllm_ok=False, shape="tag+tagged",
                     fmt="<seed:tool_call>\n<function=NAME>\n<parameter=KEY>VALUE</parameter>\n</function>\n</seed:tool_call>"),
    "functionary_v31": dict(vllm=None, vllm_ok=None, shape="tag+json", fmt="<function=NAME>{...}</function>"),
    "functionary_v32": dict(vllm=None, vllm_ok=None, shape="tag+json", fmt=">>>NAME\n{...}"),
    "firefunction": dict(vllm=None, vllm_ok=None, shape="json",
                         fmt=' functools[{"name": NAME, "arguments": {...}}]'),
    "apriel": dict(vllm=None, vllm_ok=None, shape="json",
                   fmt='<tool_calls>[{"name": NAME, "arguments": {...}}]</tool_calls>'),
    "apertus": dict(vllm=None, vllm_ok=None, shape="json", fmt='<|tools_prefix|>[{NAME: {...}}]<|tools_suffix|>'),
    "nemotron_v2": dict(vllm=None, vllm_ok=None, shape="json",
                        fmt='<TOOLCALL>[{"name": NAME, "arguments": {...}}]</TOOLCALL>'),
    "gigachat": dict(vllm=None, vllm_ok=None, shape="json",
                     fmt='<|function_call|>{"name": NAME, "arguments": {...}}  (v3: a "function call" role turn)'),
    "minicpm5": dict(vllm=None, vllm_ok=None, shape="tag+tagged",
                     fmt='<function name="NAME"><param name="KEY">VALUE</param></function>'),
    "hunyuan": dict(vllm=None, vllm_ok=None, shape="tag+tagged",
                    fmt="<tool_calls:opensource><tool_call:opensource>NAME<tool_sep:opensource><arg_key:opensource>KEY"
                        "</arg_key:opensource><arg_value:opensource>VALUE</arg_value:opensource>..."),
    "solar_open": dict(vllm=None, vllm_ok=None, shape="tag+json",
                       fmt="<|tool_calls|><|tool_call:begin|>ID<|tool_call:name|>NAME<|tool_call:args|>{...}<|tool_call:end|>"),
    "muse_glimmer": dict(vllm=None, vllm_ok=None, shape="tag+tagged",
                         fmt='<atem:function_calls><atem:invoke name="NAME"><atem:parameter name="KEY">VALUE'
                             '</atem:parameter></atem:invoke></atem:function_calls>'),
    "json": dict(vllm=None, vllm_ok=None, shape="json", fmt="a JSON call in an unlisted wrapper"),
    "unknown": dict(vllm=None, vllm_ok=None, shape=None, fmt="an unlisted call syntax"),
}


def llama_cpp_handler(src):
    """The handler llama.cpp's common_chat_try_specialized_template picks for this template source (same checks,
    same order -- first match wins), or "autoparser" when none matches (render-and-diff classification)."""
    src = src or ""

    def has(*xs):
        return all(x in src for x in xs)
    if has("[SYSTEM_PROMPT]", "[TOOL_CALLS]", "[ARGS]") and "[CALL_ID]" not in src:
        return "Ministral/Magistral 3", "mistral"
    if "<|channel|>" in src:
        return "GPT-OSS", "harmony"
    if has("<atem:function_calls>", "<|eom|>"):
        return "Muse Glimmer", "muse_glimmer"
    if has(">>>all", ">>>${recipient}"):
        return "Functionary v3.2", "functionary_v32"
    if has("<|tool_calls_section_begin|>", "<|tool_call_begin|>"):
        return "Kimi K2", "kimi_k2"
    if has("<|open|>", "<|close|>", "<|end_of_msg|>"):
        return "Kimi K3", "kimi_k3"
    if has("<|START_TEXT|>", "<|START_ACTION|>"):
        return "Cohere2 MoE", "cohere"
    if has("<|tool_list_start|>", "<|tool_list_end|>"):
        return "LFM2", "lfm2"
    if "List of tools: [" in src and "<|tool_list_start|>" not in src:
        return "LFM2.5", "lfm2"
    if has("<|role_sep|>", "<|message_sep|>") and "<|function_call|>" not in src:
        return "GigaChatV3", "gigachat"
    if has("]<]minimax[>[", "<tool_call>", "<invoke name="):
        return "MiniMax-M3", "minimax_m3"
    if has("dsml_token", "DSML") and ("function_calls" in src or "tool_calls" in src):
        return "DeepSeek V3.2/V4", "deepseek_v32"
    if "'<|tool_call>call:'" in src:
        return "Gemma4", "gemma4"
    if has("Tool usage guidelines:", '<function name="', '<param name="'):
        return "MiniCPM5", "minicpm5"
    if has("<tool_call>", "<function=", "<parameter="):
        return "Qwen3-Coder", "qwen3_coder"
    return "autoparser", None


def _sniff_tool_family(src):
    """(family, strong) from the template source alone. llama.cpp's specialised list first; then markers unique
    to one family (strong); then shapes several families share (weak -- rendering decides)."""
    handler, fam = llama_cpp_handler(src)
    if fam:
        return fam, True
    s = src or ""
    for marker, fam in (("<seed:tool_call>", "seed_oss"), ("<minimax:tool_call>", "minimax_m2"),
                        ("<start_function_call>", "functiongemma"), ("<|tool_call|>", "granite"),
                        ("<|START_ACTION|>", "cohere"), ("[TOOL_CALLS]", "mistral"), ("functools[", "firefunction"), ("<TOOLCALL>", "nemotron_v2"),
                        ("<|function_call|>", "gigachat"), ("tool_call:opensource", "hunyuan"),
                        ("<|tool_call:begin|>", "solar_open"), ("<|tools_prefix|>", "apertus")):
        if marker in s:
            return fam, True
    if "tool\u2581calls\u2581begin" in s:
        return ("deepseek_v3" if "```json" in s else "deepseek_v31"), True
    if "Action:" in s and "tool_name" in s:
        return "cohere_cmdr", True
    if "<function=" in s and "<|python_tag|>" in s:
        return "functionary_v31", True
    if "<arg_key>" in s and "<tool_call>" in s:
        return "glm45", False
    if "<tool_calls>" in s:
        return "apriel", False
    if "<tool_call>" in s:
        return "hermes", False
    if "<|python_tag|>" in s or ('"parameters"' in s and re.search(r"\btools\b", s)):
        return "llama3_json", False
    return None, False


_TC_NAME, _TC_VAL, _TC_USER, _TC_RES, _TC_SYS, _TC_DESC = (
    "pzq_lookup", "PZQCITY", "PZQUSER", "PZQRESULT", "PZQSYS", "PZQDESC")
_TC_TOOLS = [{"type": "function", "function": {
    "name": _TC_NAME, "description": f"Look up {_TC_DESC}.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city"},
                                                     "days": {"type": "integer", "description": "how many days"}},
                   "required": ["city"]}}}]
#: list containers seen around a rendered tool list
_TOOL_LIST_MARKERS = ("<tools>", "<|tool_list_start|>", "List of tools:", "[AVAILABLE_TOOLS]", "<|tools_prefix|>",
                      "# Tools", "namespace functions", "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>", "<available_tools>",
                      "You have access to the following functions", "Here is a list of functions in JSON format")


#: what ends a rendered call, and what ends the assistant turn after it (the example stops at the latter)
_CALL_CLOSERS = (r"</tool_call>|<\|tool_call_end\|>|<\|tool_calls_section_end\|>|<\|call\|>|tool\u2581calls?\u2581end\uff5c>|"
                 r"<\|END_ACTION\|>|</tool_calls>|</TOOLCALL>|<tool_call\|>|</minimax:tool_call>|</seed:tool_call>|"
                 r"</\uff5cDSML\uff5c(?:function|tool)_calls>|</atem:function_calls>|</function>|<\|tool_call:end\|>|"
                 r"</tool_calls?:opensource>|<end_function_call>|<\|tools_suffix\|>|<\|close\|>tools<\|sep\|>|"
                 r"</invoke>|```|[\]\}]")
_TURN_ENDS = ("<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|eot|>", "<|end|>", "<\uff5cend\u2581of\u2581sentence\uff5c>", "[TOOL_RESULTS]",
              "<|observation|>", "</s>", "<seed:eos>", "[e~[", "</assistant>", "<|message_sep|>", "<|END_OF_TURN_TOKEN|>",
              "<turn|>", "<|im_system|>", "<\uff5chy_eos", "<|end_of_text|>", "<|calls|>", "<end_of_sentence>", "\n\n<sep>",
              "<|tool_response", "<|close|>message", "]<]minimax[>[</tool_call>")


def _classify_call_text(seg, name=_TC_NAME):
    """The family a rendered call belongs to, by signature. None when the name is not in the text at all."""
    if not seg or name not in seg:
        return None
    n = re.escape(name)
    checks = [
        (lambda: f"to=functions.{name}" in seg, "harmony"),
        (lambda: "<|tool_calls_section_begin|>" in seg, "kimi_k2"),
        (lambda: '<|open|>call tool="' in seg, "kimi_k3"),
        (lambda: "DSML" in seg, "deepseek_v32"),
        (lambda: "tool\u2581call\u2581begin" in seg, "deepseek"),
        (lambda: "]<]minimax[>[" in seg, "minimax_m3"),
        (lambda: "<minimax:tool_call>" in seg, "minimax_m2"),
        (lambda: "<seed:tool_call>" in seg, "seed_oss"),
        (lambda: "<atem:function_calls>" in seg, "muse_glimmer"),
        (lambda: "tool_call:opensource" in seg, "hunyuan"),
        (lambda: "<|tool_call:begin|>" in seg, "solar_open"),
        (lambda: "<start_function_call>" in seg, "functiongemma"),
        (lambda: "<|tool_call>call:" in seg, "gemma4"),
        (lambda: "<|tool_call_start|>" in seg, "lfm2"),
        (lambda: "<|START_ACTION|>" in seg, "cohere"),
        (lambda: bool(re.search(r"Action:\s*```json", seg)), "cohere_cmdr"),
        (lambda: "[TOOL_CALLS]" in seg, "mistral"),
        (lambda: "<|tool_call|>" in seg, "granite"),
        (lambda: "<TOOLCALL>" in seg, "nemotron_v2"),
        (lambda: "<|tools_prefix|>" in seg, "apertus"),
        (lambda: "<|function_call|>" in seg or "function call<|role_sep|>" in seg, "gigachat"),
        (lambda: bool(re.search(r'<function name="%s"' % n, seg)), "minicpm5"),
        (lambda: f"<function={name}>" in seg, "function="),
        (lambda: f">>>{name}" in seg, "functionary_v32"),
        (lambda: "functools[" in seg, "firefunction"),
        (lambda: "<arg_key>" in seg, "glm"),
        (lambda: "<tool_calls>" in seg, "tool_calls"),
        (lambda: bool(re.search(r"<tool_call>\s*\[\s*\{", seg)) and "</tool_call>" not in seg, "granite_tc"),
        (lambda: "<tool_call>" in seg and '"name"' in seg, "hermes"),
        (lambda: bool(re.search(r"\[\s*%s\(" % n, seg)), "pythonic"),
        (lambda: '"parameters"' in seg and '"name"' in seg, "llama3_json"),
        (lambda: bool(re.search(r'\[\s*\{\s*"name"\s*:\s*"%s"' % n, seg)), "xlam"),
        (lambda: '"name"' in seg, "json"),
    ]
    for test, fam in checks:
        if not test():
            continue
        if fam == "deepseek":
            return "deepseek_v3" if "```json" in seg else "deepseek_v31"
        if fam == "function=":
            return "qwen3_coder" if "<parameter=" in seg else "functionary_v31"
        if fam == "glm":
            return "glm47" if re.search(r"%s<arg_key>" % n, seg) else "glm45"
        if fam == "tool_calls":
            return "apriel" if re.search(r"<tool_calls>\s*\[", seg) else "minimax_m1"
        return fam
    return "unknown"


def _tc_messages(arg_string):
    args = {"city": _TC_VAL, "days": 7}
    call = {"id": "pzqcall01", "type": "function",                 # 9 chars: Mistral templates insist
            "function": {"name": _TC_NAME, "arguments": json.dumps(args) if arg_string else args}}
    return [{"role": "user", "content": _TC_USER},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "pzqcall01", "name": _TC_NAME, "content": _TC_RES},
            {"role": "assistant", "content": "PZQFINAL"}]


def _render_tool_probe(ct):
    """What rendering says about tools: the call segment, the tool list with and without tools, the placement."""
    out = {"rendered": False, "seg": None, "call": None, "tools_rendered": None, "placement": None,
           "container": None, "args_string": None}
    full = None
    for as_string in (False, True):                          # object arguments first, then a JSON string
        r = _safe(ct.render, _tc_messages(as_string), tools=_TC_TOOLS)
        if r and _TC_USER in r:
            full = r
            out["args_string"] = as_string
            if _TC_VAL in r:
                break
    if full is not None:
        out["rendered"] = True
        ui = full.index(_TC_USER) + len(_TC_USER)
        ri = full.find(_TC_RES, ui)
        seg = full[ui:ri if ri >= 0 else len(full)]
        out["seg"] = seg
        if _TC_NAME in seg:
            gen = _safe(ct.render, [{"role": "user", "content": _TC_USER}], add_generation_prompt=True,
                        tools=_TC_TOOLS) or ""
            call = full[len(gen):] if gen and full.startswith(gen) else seg
            cut = call.find(_TC_RES)
            call = call[:cut] if cut >= 0 else call
            # trim to the call itself: drop an empty thinking block in front, stop after the last argument's closer
            end = max(call.rfind(_TC_VAL), call.rfind("7"))
            # stop at the end of the call: the last call closer before the turn ends
            tail = call[end:]
            stops = [tail.find(t) for t in _TURN_ENDS if t in tail]
            stop = min(stops) if stops else len(tail)
            ends = [m.end() for m in re.finditer(_CALL_CLOSERS, tail[:stop])]
            call = call[:end + (max(ends) if ends else stop)]
            call = re.sub(r"^\s*(<think>\s*</think>|</think>|<\|START_THINKING\|><\|END_THINKING\|>)\s*", "", call)
            out["call"] = call.strip()
    # the tool list: rendered at all, and where
    msgs = [{"role": "system", "content": _TC_SYS}, {"role": "user", "content": _TC_USER}]
    with_t = _safe(ct.render, msgs, add_generation_prompt=True, tools=_TC_TOOLS)
    if with_t is None:
        msgs = msgs[1:]
        with_t = _safe(ct.render, msgs, add_generation_prompt=True, tools=_TC_TOOLS)
    if with_t is not None:
        ti = with_t.find(_TC_NAME)
        out["tools_rendered"] = ti >= 0
        if ti >= 0:
            ui, si = with_t.find(_TC_USER), with_t.find(_TC_SYS)
            eot = ""
            pair = _safe(ct.render, [{"role": "user", "content": "UZQX"}, {"role": "assistant", "content": "AZQX"}])
            if pair and "AZQX" in pair:
                eot = pair[pair.index("AZQX") + 4:].strip()
            if ui >= 0 and ti > ui:
                out["placement"] = "after the user message"
            elif si >= 0 and (not eot or eot not in with_t[min(si, ti):max(si, ti)]):
                out["placement"] = "system prompt"
            elif ui >= 0 and (not eot or eot not in with_t[ti:ui]):
                out["placement"] = "user turn"
            else:
                out["placement"] = "own turn"
            near = with_t[max(0, ti - 600):ti]
            hits = [(near.rfind(mk), mk) for mk in _TOOL_LIST_MARKERS if mk in near]
            out["container"] = max(hits)[1] if hits else None
    return out


def toolcall_info(model=None, tpl=None, archs=None, conf=None) -> dict:
    """Does this model call tools, in which syntax, and how must it be served?

    {"family", "supports_tools", "supports_tool_calls", "source": static|render|static+render,
     "vllm_parser", "vllm_parser_confirmed", "vllm_flags", "llama_cpp": {"flags", "handler", "format"},
     "tool_list": {"placement", "container"}, "call_format", "shape", "example", "why": [...]}"""
    if tpl is None:
        tpl, archs, conf = _template_and_arch(model)
    if isinstance(tpl, list):                                    # named templates: prefer "tool_use"
        tpl = next((t.get("template") for t in tpl if t.get("name") == "tool_use"),
                   next((t.get("template") for t in tpl if t.get("name") == "default"), tpl[0].get("template")))
    tpl = tpl if isinstance(tpl, str) else ""
    why = []
    info = {"family": "none", "supports_tools": False, "supports_tool_calls": False, "source": None,
            "vllm_parser": None, "vllm_parser_confirmed": None, "vllm_flags": None,
            "llama_cpp": None, "tool_list": {"placement": None, "container": None},
            "call_format": None, "shape": None, "example": None, "reader": None, "why": why}
    if not tpl:
        why.append("no chat template -> no tool calling")
        return info
    sniffed, strong = _sniff_tool_family(tpl)
    handler, _ = llama_cpp_handler(tpl)
    probe = None
    for mutable in (False, True):
        try:
            ct = ChatTemplate(tpl, _special_tokens(model, conf or {}), mutable=mutable)
        except Exception:                                        # noqa: BLE001 -- no jinja2 / bad syntax
            break
        got = _safe(_render_tool_probe, ct)
        if got and (probe is None or got.get("rendered") or probe.get("tools_rendered") is None):
            probe = got
        if probe and probe.get("rendered"):
            break
    rendered = _classify_call_text((probe or {}).get("seg"))
    if rendered:
        info["supports_tool_calls"] = True
        info["example"] = ((probe.get("call") or "").replace(_TC_NAME, "get_weather")
                           .replace(_TC_VAL, "Paris")[:400]) or None
    tools_seen = (probe or {}).get("tools_rendered")
    if tools_seen is not None:
        info["supports_tools"] = bool(tools_seen)
        info["tool_list"] = {"placement": probe.get("placement"), "container": probe.get("container")}
    else:                                                       # no rendering: what the source reads
        info["supports_tools"] = bool(re.search(r"\btools\b", tpl)) and bool(sniffed)
    if strong and sniffed:
        fam = sniffed
        info["source"] = "static+render" if rendered == sniffed else "static"
        why.append(f"template markers -> {sniffed}" + (f" (llama.cpp: specialised '{handler}')"
                                                       if handler != "autoparser" else ""))
        if rendered and rendered != sniffed:
            why.append(f"rendered call reads as {rendered}; the markers win, as in llama.cpp")
        if not rendered:
            why.append("the template does not render tool_calls back -- the model writes calls into its content")
        info["supports_tool_calls"] = True
    elif rendered:
        fam, info["source"] = rendered, "render"
        why.append(f"rendered assistant tool call reads as {rendered}")
    elif sniffed and info["supports_tools"]:
        fam, info["source"] = sniffed, "static"
        info["supports_tool_calls"] = True
        why.append(f"template describes tools and mentions {sniffed}-style calls (not rendered)")
    else:
        fam = "none"
        why.append("template renders no tool list" if not info["supports_tools"] else
                   "template lists tools but writes no call syntax of its own")
    if fam == "granite" and any(re.search(r"granite[-_ ]3\.[0-2]\b", n or "", re.I)
                                for n in _safe(_model_names, model, conf or {}) or []):
        fam = "granite_tc"
        why.append("Granite 3.0-3.2 writes the call as a plain `<tool_call>[...]` (not the template's <|tool_call|>)")
    info["family"] = fam
    info["reader"] = None
    if tpl:
        for mutable in (False, True):
            try:
                rd = _safe(derive_reader, ChatTemplate(tpl, _special_tokens(model, conf or {}), mutable=mutable))
            except Exception:                                    # noqa: BLE001 -- no jinja2 / bad syntax
                rd = None
            if rd:
                info["reader"] = rd
                break
    if fam != "none":
        meta = TOOL_FAMILIES.get(fam, TOOL_FAMILIES["unknown"])
        ex, lead = info["example"] or "", re.split(r"NAME|\{", meta["fmt"])[0].strip()
        if lead and lead in ex:                                  # start the example at the call, not the header
            info["example"] = ex[ex.index(lead):]
        info["call_format"], info["shape"] = meta["fmt"], meta["shape"]
        info["vllm_parser"], info["vllm_parser_confirmed"] = meta["vllm"], meta["vllm_ok"]
        if meta["vllm"]:
            info["vllm_flags"] = f"--enable-auto-tool-choice --tool-call-parser {meta['vllm']}"
        if handler != "autoparser":
            fmt = f"specialised: {handler}"
        else:
            shape = {"json": "JSON_NATIVE", "tag+json": "TAG_WITH_JSON", "tag+tagged": "TAG_WITH_TAGGED"}.get(
                meta["shape"])
            fmt = f"autoparser ({shape})" if shape else "autoparser (no native shape -- check the parse)"
        info["llama_cpp"] = {"flags": "--jinja", "handler": handler, "format": fmt}
        if not info["supports_tools"]:
            why.append("the template does not render a tool list -- tools must be described in the system prompt")
    return info


def serve_with_line(t, model="<model>"):
    """One line: how to serve this model so its tool calls are parsed -- llama.cpp and vLLM."""
    if not t or t.get("family", "none") == "none":
        return "no tool calling in the chat template"
    parts = ["llama-server --jinja"]
    lc = t.get("llama_cpp") or {}
    if lc.get("format"):
        parts[0] += f"  ({lc['format']})"
    if t.get("vllm_parser"):
        parts.append(f"vllm serve {model} {t['vllm_flags']}" + ("" if t.get("vllm_parser_confirmed") else
                                                          "  (parser name unconfirmed for this model)"))
    else:
        parts.append("vLLM: no tool parser for this syntax")
    return "  |  ".join(parts)


def format_tool_call(family, name, arguments):
    """One call written in a family's own syntax -- for a template that does not render tool_calls back (LFM2,
    Granite 3.x, R1-distill): the calibration text must still carry the call the way the model writes it.
    Returns None for a family with no exact writer here (the caller keeps its JSON fallback)."""
    args = arguments if isinstance(arguments, dict) else {}
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except ValueError:
            args = {}
    js = json.dumps(args, ensure_ascii=False)
    obj = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)

    def py(v):
        return repr(v) if not isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    def val(v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    pyc = f"[{name}(" + ", ".join(f"{k}={py(v)}" for k, v in args.items()) + ")]"
    w = {
        "hermes": f"<tool_call>\n{obj}\n</tool_call>",
        "qwen3_coder": f"<tool_call>\n<function={name}>\n" + "".join(
            f"<parameter={k}>\n{val(v)}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>",
        "lfm2": f"<|tool_call_start|>{pyc}<|tool_call_end|>",
        "pythonic": pyc,
        "mistral": f"[TOOL_CALLS]{name}[ARGS]{js}",
        "llama3_json": json.dumps({"name": name, "parameters": args}, ensure_ascii=False),
        "xlam": f"[{obj}]",
        "granite": f"<|tool_call|>[{obj}]",
        "granite_tc": f"<tool_call>[{obj}]",
        "deepseek_v3": f"<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>function<\uff5ctool\u2581sep\uff5c>{name}\n```json\n{js}\n```"
                       "<\uff5ctool\u2581call\u2581end\uff5c><\uff5ctool\u2581calls\u2581end\uff5c>",
        "deepseek_v31": f"<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>{name}<\uff5ctool\u2581sep\uff5c>{js}<\uff5ctool\u2581call\u2581end\uff5c>"
                        "<\uff5ctool\u2581calls\u2581end\uff5c>",
        "glm45": f"<tool_call>{name}\n" + "".join(f"<arg_key>{k}</arg_key>\n<arg_value>{val(v)}</arg_value>\n"
                                                   for k, v in args.items()) + "</tool_call>",
        "glm47": f"<tool_call>{name}" + "".join(f"<arg_key>{k}</arg_key><arg_value>{val(v)}</arg_value>"
                                                for k, v in args.items()) + "</tool_call>",
        "kimi_k2": f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>{js}"
                   "<|tool_call_end|><|tool_calls_section_end|>",
        "minimax_m2": f'<minimax:tool_call>\n<invoke name="{name}">\n' + "".join(
            f'<parameter name="{k}">{val(v)}</parameter>\n' for k, v in args.items()) + "</invoke>\n</minimax:tool_call>",
        "seed_oss": f"<seed:tool_call>\n<function={name}>\n" + "".join(
            f"<parameter={k}>{val(v)}</parameter>\n" for k, v in args.items()) + "</function>\n</seed:tool_call>",
        "cohere": "<|START_ACTION|>[\n    " + json.dumps({"tool_call_id": "0", "tool_name": name, "parameters": args},
                                                         ensure_ascii=False) + "\n]<|END_ACTION|>",
        "apriel": f"<tool_calls>[{obj}]</tool_calls>",
        "nemotron_v2": f"<TOOLCALL>[{obj}]</TOOLCALL>",
        "functionary_v31": f"<function={name}>{js}</function>",
        "firefunction": f" functools[{obj}]",
        "functiongemma": f"<start_function_call>call:{name}{{" + ",".join(
            f"{k}:<escape>{v}<escape>" if isinstance(v, str) else f"{k}:{json.dumps(v)}" for k, v in args.items())
            + "}<end_function_call>",
        "gemma4": f"<|tool_call>call:{name}{{" + ",".join(
            f'{k}:<|"|>{v}<|"|>' if isinstance(v, str) else f"{k}:{json.dumps(v)}" for k, v in args.items())
            + "}<tool_call|>",
    }
    return w.get(family)


# ----------------------------------------------------------------- reading calls back (the inverse of the writers)
#
# A server parses the model's calls for us (llama-server --jinja, mlx_lm.server, vllm --tool-call-parser). A model run
# IN-PROCESS (exllamav3, transformers) hands back raw text, so the call has to be read the way that family's server
# parser would read it: keyed on the family toolcall_info named, strict about the family's own markers (call-shaped
# text in another syntax is NOT a call -- a server would not accept it either), typed through the tool's schema the
# way vLLM's tagged-argument parsers do. parse_tool_calls(format_tool_call(f, n, a), f) == [(n, a)] for every family
# with a writer; the families below without a reader are reported, never guessed.

#: every opener a call can start with, any family -- "the model tried to call" even when nothing parsed
CALL_MARKERS = re.compile(
    r"<tool_call>|<tool_calls>|\[TOOL_CALLS\]|<\|tool_call|<function[=\s]|<\|python_tag\|>|<\|START_ACTION\|>|"
    r"<TOOLCALL>|<minimax:tool_call>|<seed:tool_call>|<invoke name=|<start_function_call>|<\|tool_calls_section_begin"
    r"|tool\u2581calls?\u2581begin|DSML|to=functions\.|functools\[|<\|function_call\|>|<\|tools_prefix\|>|"
    r"tool_call:opensource|>>>[A-Za-z_]|Action:\s*```|\"name\"\s*:|\w+\([^()]*=")


def call_shaped(text):
    """Does this answer look like an attempted call (in any family's syntax)? A call the parser could not read is a
    format failure (`unparsed`), not prose -- the served and the in-process paths both grade it that way."""
    return bool(CALL_MARKERS.search(text or ""))


def _json_values(s, opener="{["):
    """Every JSON value starting at a `{`/`[` in s, left to right (raw_decode; non-JSON skipped)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(s):
        if s[i] in opener:
            try:
                val, end = dec.raw_decode(s, i)
                out.append(val)
                i = end
                continue
            except ValueError:
                pass
        i += 1
    return out


def _call_from_obj(o):
    """(name, args) from one JSON call object, whichever spelling the family uses; None when it is not a call."""
    if not isinstance(o, dict):
        return None
    if isinstance(o.get("function"), dict):                          # OpenAI shape
        o = o["function"]
    name = o.get("name") or o.get("tool_name")
    if not isinstance(name, str):
        if len(o) == 1 and isinstance(next(iter(o.values())), dict):  # apertus {NAME: {...}}
            return next(iter(o.keys())), next(iter(o.values()))
        return None
    args = next((o[k] for k in ("arguments", "parameters", "args", "input") if k in o), {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return name, args                                        # graded as bad-args, as a server would pass it
    return name, args


def _calls_in_json(s):
    out = []
    for v in _json_values(s):
        for o in (v if isinstance(v, list) else [v]):
            c = _call_from_obj(o)
            if c:
                out.append(c)
    return out


def _coerce(raw, spec):
    """One tagged argument's text -> its value, by the schema type (vLLM qwen3_coder / glm / minimax readers)."""
    t = (spec or {}).get("type")
    if t == "string":
        return raw
    if t == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            pass
    if t == "number":
        try:
            f = float(raw.strip())
            return int(f) if f.is_integer() and re.fullmatch(r"-?\d+", raw.strip()) else f
        except ValueError:
            pass
    if t == "boolean" and raw.strip().lower() in ("true", "false"):
        return raw.strip().lower() == "true"
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:
        import ast
        return ast.literal_eval(raw)
    except Exception:                                                # noqa: BLE001
        return raw


def _tagged(name, pairs, schemas):
    props = ((schemas or {}).get(name) or {})
    return name, {k.strip(): _coerce(v, props.get(k.strip())) for k, v in pairs}


def _strip1(v):
    """A tagged value without the one newline the writers put either side of it."""
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _pythonic(s):
    """[f(a=1), g(b="x")] -> [(name, args)] via ast (JSON true/false/null accepted too)."""
    import ast
    s = s.strip()
    if not s.startswith("["):
        s = "[" + s + "]"
    try:
        tree = ast.parse(s, mode="eval").body
    except SyntaxError:
        return []
    out = []

    def lit(node):
        if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[node.id]
        if isinstance(node, ast.List):
            return [lit(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {lit(k): lit(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)
    for call in (tree.elts if isinstance(tree, ast.List) else [tree]):
        if not isinstance(call, ast.Call):
            return []
        fn = call.func
        name = fn.id if isinstance(fn, ast.Name) else (ast.unparse(fn) if isinstance(fn, ast.Attribute) else None)
        try:
            out.append((name, {k.arg: lit(k.value) for k in call.keywords}))
        except Exception:                                            # noqa: BLE001 -- a non-literal argument
            return []
    return out


def _balanced_list(t, start):
    """The `[...]` starting at t[start], string-aware; None when it never closes."""
    depth, q, esc = 0, None, False
    for i in range(start, len(t)):
        c = t[i]
        if q:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c in "[(":
            depth += 1
        elif c in "])":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None


def _kv_object(body, quote):
    """Gemma's `{key:<|"|>v<|"|>,n:3}` (or FunctionGemma's <escape>) -> a dict."""
    s = body.replace(quote, '"')
    s = re.sub(r'([{,]\s*)([A-Za-z_][\w\-]*)\s*:', r'\1"\2":', s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


_FW = "\uff5c"                                                     # DeepSeek's full-width bar
_TCB, _TCE, _TSEP = f"<{_FW}tool\u2581call\u2581begin{_FW}>", f"<{_FW}tool\u2581call\u2581end{_FW}>", f"<{_FW}tool\u2581sep{_FW}>"


def _p_hermes(t, sc):
    out = []
    for m in re.finditer(r"<tool_call>(.*?)(?:</tool_call>|$)", t, re.S):
        out += _calls_in_json(m.group(1))
    return out


def _p_function_eq(t, sc):                                         # qwen3_coder / seed_oss
    out = []
    for m in re.finditer(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", t, re.S):
        pairs = [(k, _strip1(v)) for k, v in re.findall(
            r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", m.group(2), re.S)]
        out.append(_tagged(m.group(1).strip(), pairs, sc))
    return out


def _p_glm(t, sc):
    out = []
    for m in re.finditer(r"<tool_call>\s*([^\s<]+)\s*(.*?)(?:</tool_call>|$)", t, re.S):
        pairs = [(k, _strip1(v)) for k, v in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                                                        m.group(2), re.S)]
        out.append(_tagged(m.group(1), pairs, sc))
    return out


def _p_invoke(t, sc, inv=r'<invoke name="([^"]+)">', par=r'<parameter name="([^"]+)">(.*?)</parameter>',
              end=r"</invoke>"):
    out = []
    for m in re.finditer(inv + r"(.*?)(?:" + end + r"|$)", t, re.S):
        out.append(_tagged(m.group(1), [(k, _strip1(v)) for k, v in re.findall(par, m.group(2), re.S)], sc))
    return out


def _p_dsml(t, sc):
    out = []
    for m in re.finditer(rf'<{_FW}DSML{_FW}invoke name="([^"]+)">(.*?)(?:</{_FW}DSML{_FW}invoke>|$)', t, re.S):
        args = {}
        props = (sc or {}).get(m.group(1)) or {}
        for k, isstr, v in re.findall(rf'<{_FW}DSML{_FW}parameter name="([^"]+)"(?:\s+string="(\w+)")?>(.*?)'
                                      rf'</{_FW}DSML{_FW}parameter>', m.group(2), re.S):
            args[k] = v if isstr == "true" else _coerce(v, props.get(k))
        out.append((m.group(1), args))
    return out


def _p_pythonic_wrapped(t, sc, start=r"<\|tool_call_start\|>", end=r"<\|tool_call_end\|>"):
    out = []
    for m in re.finditer(start + r"(.*?)(?:" + end + r"|$)", t, re.S):
        out += _pythonic(m.group(1))
    return out


def _p_pythonic(t, sc):
    m = re.search(r"\[\s*[A-Za-z_][\w.]*\s*\(", t)
    if not m:
        return []
    body = _balanced_list(t, m.start())
    return _pythonic(body) if body else []


def _p_mistral(t, sc):
    out = []
    for seg in t.split("[TOOL_CALLS]")[1:]:
        s = seg.strip()
        if s[:1] in "[{":
            out += _calls_in_json(s)
            continue
        m = re.match(r"([\w.\-]+)\s*(?:\[CALL_ID\]\s*\S+?\s*)?(?:\[ARGS\])?\s*(?=\{)", s)
        if m:
            vals = _json_values(s[m.end():], "{")
            if vals and isinstance(vals[0], dict):
                out.append((m.group(1), vals[0]))
    return out


def _p_llama3_json(t, sc):
    s = t.strip()
    if s.startswith("<|python_tag|>"):
        s = s[len("<|python_tag|>"):].strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    if not s.startswith("{"):                                        # vLLM: the reply must OPEN with the call
        return []
    return _calls_in_json(s)


def _p_xlam(t, sc):
    s = re.sub(r"^```(?:json)?\s*", "", t.strip())
    if s[:1] not in "[{":
        return []
    return _calls_in_json(s)


def _p_after(marker, end=None):
    def p(t, sc):
        out = []
        for seg in t.split(marker)[1:]:
            if end and end in seg:
                seg = seg.split(end, 1)[0]
            out += _calls_in_json(seg)
        return out
    return p


def _p_harmony(t, sc):
    out = []
    for m in re.finditer(r"to=functions\.([\w\-.]+)(.*?)<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|<\|return\|>|$)",
                         t, re.S):
        vals = _json_values(m.group(3), "{")
        out.append((m.group(1), vals[0] if vals else m.group(3).strip()))
    return out


def _p_deepseek_v3(t, sc):
    return [(n.strip(), json.loads(a)) for n, a in re.findall(
        re.escape(_TCB) + r"\s*function\s*" + re.escape(_TSEP) + r"\s*([^\n]+?)\s*\n\s*```(?:json)?\s*(\{.*?\})\s*```",
        t, re.S) if _ok_json(a)]


def _p_deepseek_v31(t, sc):
    return [(n.strip(), json.loads(a)) for n, a in re.findall(
        re.escape(_TCB) + r"\s*([^\n<]+?)\s*" + re.escape(_TSEP) + r"\s*(\{.*?\})\s*" + re.escape(_TCE), t, re.S)
        if _ok_json(a)]


def _ok_json(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def _p_kimi_k2(t, sc):
    out = []
    for n, a in re.findall(r"<\|tool_call_begin\|>\s*(?:functions\.)?([\w\-.]+?)(?::\d+)?\s*"
                           r"<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>", t, re.S):
        vals = _json_values(a, "{")
        if vals:
            out.append((n, vals[0]))
    return out


def _p_kv(open_, close, quote):
    def p(t, sc):
        out = []
        for m in re.finditer(re.escape(open_) + r"call:([\w\-.]+)\s*(\{.*?\})\s*(?:" + re.escape(close) + "|$)", t, re.S):
            d = _kv_object(m.group(2), quote)
            if d is not None:
                out.append((m.group(1), d))
        return out
    return p


def _p_functionary_v31(t, sc):
    return [(n, json.loads(a)) for n, a in re.findall(r"<function=([\w\-.]+)>\s*(\{.*?\})\s*</function>", t, re.S)
            if _ok_json(a)]


def _p_functionary_v32(t, sc):
    out = []
    for m in re.finditer(r">>>\s*([\w\-.]+)\s*\n", t):
        if m.group(1) == "all":
            continue
        vals = _json_values(t[m.end():], "{")
        if vals:
            out.append((m.group(1), vals[0]))
    return out


def _p_solar(t, sc):
    return [(n.strip(), json.loads(a)) for n, a in re.findall(
        r"<\|tool_call:name\|>(.*?)<\|tool_call:args\|>\s*(\{.*?\})\s*(?:<\|tool_call:end\|>|$)", t, re.S)
        if _ok_json(a)]


def _p_granite(t, sc):
    """Granite: `<|tool_call|>` (3.3) or a plain `<tool_call>` (3.0-3.2) followed by a JSON list of calls -- vLLM's
    granite parser accepts both spellings, and so does this reader."""
    m = re.search(r"<\|tool_call\|>|<tool_call>", t)
    if not m:
        return []
    rest = t[m.end():]
    if "</tool_call>" in rest and m.group(0) == "<tool_call>" and not rest.lstrip().startswith("["):
        return []                                                    # a Hermes block, not Granite's list
    return _calls_in_json(rest.split("</tool_call>", 1)[0] if "</tool_call>" in rest else rest)


def _p_cohere_cmdr(t, sc):
    m = re.search(r"Action:\s*```(?:json)?\s*(.*?)```", t, re.S)
    return _calls_in_json(m.group(1)) if m else []


#: family -> the reader that takes its calls back out of raw text (what that family's server parser accepts)
CALL_READERS = {
    "hermes": _p_hermes, "qwen3_coder": _p_function_eq, "seed_oss": _p_function_eq,
    "glm45": _p_glm, "glm47": _p_glm,
    "minimax_m2": _p_invoke,
    "minicpm5": lambda t, sc: _p_invoke(t, sc, r'<function name="([^"]+)">', r'<param name="([^"]+)">(.*?)</param>',
                                        r"</function>"),
    "deepseek_v32": _p_dsml,
    "lfm2": _p_pythonic_wrapped, "pythonic": _p_pythonic,
    "mistral": _p_mistral, "llama3_json": _p_llama3_json, "xlam": _p_xlam,
    "granite": lambda t, sc: _p_granite(t, sc),
    "granite_tc": lambda t, sc: _p_granite(t, sc),
    "harmony": _p_harmony,
    "deepseek_v3": _p_deepseek_v3, "deepseek_v31": _p_deepseek_v31,
    "kimi_k2": _p_kimi_k2,
    "cohere": _p_after("<|START_ACTION|>", "<|END_ACTION|>"), "cohere_cmdr": _p_cohere_cmdr,
    "minimax_m1": _p_after("<tool_calls>", "</tool_calls>"), "apriel": _p_after("<tool_calls>", "</tool_calls>"),
    "nemotron_v2": _p_after("<TOOLCALL>", "</TOOLCALL>"),
    "functionary_v31": _p_functionary_v31, "functionary_v32": _p_functionary_v32,
    "firefunction": _p_after("functools"),
    "functiongemma": _p_kv("<start_function_call>", "<end_function_call>", "<escape>"),
    "gemma4": _p_kv("<|tool_call>", "<tool_call|>", '<|"|>'),
    "gigachat": _p_after("<|function_call|>"),
    "solar_open": _p_solar,
    "apertus": _p_after("<|tools_prefix|>", "<|tools_suffix|>"),
    "json": lambda t, sc: _calls_in_json(t),
}


def tool_schemas(tools):
    """{name: properties} from an OpenAI `tools` list -- how tagged arguments are typed back."""
    out = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if fn.get("name"):
            out[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    return out


def parse_tool_calls(text, family, tools=None, reader=None):
    """(calls, content, attempted) from a model's raw answer (thinking already removed).

    calls     [{"name": str, "arguments": dict | str}] in the order written -- what a server returns in tool_calls
    content   the text with the calls taken out (prose around them)
    attempted True when the answer carries call markers at all (parsed or not)

    reader    a derive_reader spec (toolcall_info's "reader"): the model's OWN template's call syntax, read off a
              rendering of a known call. Used when the family has no reader here, and whenever the family's reader
              finds no call in text that carries the template's markers.

    Raises KeyError for a family with no reader here and no template reader (the caller reports it, never guesses)."""
    calls, content, attempted, _ = read_tool_calls(text, family, tools, reader)
    return calls, content, attempted


def read_tool_calls(text, family, tools=None, reader=None):
    """parse_tool_calls + which reader read the calls: "family:<name>", "template" or None."""
    fam_reader = CALL_READERS.get(family)
    if fam_reader is None and not reader:
        raise KeyError(family)
    t = text or ""
    calls, how = [], None
    if fam_reader is not None:
        try:
            calls = [{"name": n, "arguments": a} for n, a in fam_reader(t, tool_schemas(tools)) if n]
        except Exception:                                            # noqa: BLE001 -- unreadable = no call
            calls = []
        how = f"family:{family}" if calls else None
    if not calls and reader:
        try:
            calls = [{"name": n, "arguments": a} for n, a in parse_with_reader(t, reader, tools) if n]
        except Exception:                                            # noqa: BLE001
            calls = []
        how = "template" if calls else None
    attempted = call_shaped(t) or bool(calls) or bool(reader and _reader_marker_in(t, reader))
    content = t
    if calls:
        first = CALL_MARKERS.search(t)
        mk = _reader_marker_in(t, reader) if reader else None
        starts = [x for x in ((first.start() if first else None), mk) if x is not None]
        content = t[:min(starts)].strip() if starts else ""
    return calls, content, attempted, how


def _reader_marker_in(text, spec):
    """where the template reader's opening marker first appears in text, or None."""
    op = (spec or {}).get("open") if (spec or {}).get("shape") in ("json", "pythonic") else (spec or {}).get("call_open")
    if not op or not op.strip():
        return None
    m = re.search(_flex(op.strip()), text or "")
    return m.start() if m else None


# ----------------------------------------------------------------- the template-derived reader (any model)
#
# The family readers above cover the syntaxes Pollard knows by name. A model whose template writes calls in a syntax
# with no name here -- or a known family whose reader finds nothing -- is still readable: its OWN template is asked to
# render a known call (one call, then two in one turn) and the wrapper, the separators and the shape of the call are
# read off that rendering. The model's output is then parsed with exactly those markers. One reader, derived from the
# template, used on every path (served replies whose server parser did not read the call, in-process generations).

_RD_NAME, _RD_NAME2 = "pzq_lookup", "pzq_other"
_RD_A1, _RD_B1, _RD_A2, _RD_B2 = "PZQCITY", "PZQUNIT", "PZQTOWN", "PZQMODE"
_RD_TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"Look up {_TC_DESC}.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city"},
                                                     "unit": {"type": "string", "description": "the unit"}},
                   "required": ["city"]}}} for n in (_RD_NAME, _RD_NAME2)]


def _rd_messages(calls, as_string):
    tc = [{"id": f"pzqcall0{i + 1}", "type": "function",
           "function": {"name": n, "arguments": json.dumps(a) if as_string else a}} for i, (n, a) in enumerate(calls)]
    msgs = [{"role": "user", "content": _TC_USER}, {"role": "assistant", "content": "", "tool_calls": tc}]
    msgs += [{"role": "tool", "tool_call_id": c["id"], "name": c["function"]["name"], "content": _TC_RES} for c in tc]
    return msgs + [{"role": "assistant", "content": "PZQFINAL"}]


def _rd_call_text(ct, calls):
    """The assistant's call turn as the template writes it (generation prompt cut off, thinking stub dropped, cut at
    the end of the last call), or None when the template does not render these calls."""
    last = list(calls[-1][1].values())[-1]
    for as_string in (False, True):
        full = _safe(ct.render, _rd_messages(calls, as_string), tools=_RD_TOOLS)
        if not full or _TC_USER not in full or calls[-1][0] not in full:
            continue
        gen = _safe(ct.render, [{"role": "user", "content": _TC_USER}], add_generation_prompt=True,
                    tools=_RD_TOOLS) or ""
        ui = full.index(_TC_USER) + len(_TC_USER)
        call = full[len(gen):] if gen and full.startswith(gen) else full[ui:]
        cut = call.find(_TC_RES)
        call = call[:cut] if cut >= 0 else call
        end = call.rfind(last)
        if end < 0:
            continue
        end += len(last)
        tail = call[end:]
        stops = [tail.find(t) for t in _TURN_ENDS if t in tail]
        stop = min(stops) if stops else len(tail)
        ends = [m.end() for m in re.finditer(_CALL_CLOSERS, tail[:stop])]
        call = call[:end + (max(ends) if ends else 0)]
        call = re.sub(r"^\s*(<think>\s*</think>|</think>|<\|START_THINKING\|><\|END_THINKING\|>)\s*", "", call)
        return call.strip()
    return None


def _lcs_suffix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[len(a) - 1 - n] == b[len(b) - 1 - n]:
        n += 1
    return a[len(a) - n:] if n else ""


def _opener(lit):
    """A per-call opener read as the common tail of two renderings starts where a marker starts: closing brackets
    left over from whatever preceded it (`>` of an end tag) are dropped, so `><<act:` reads as `<<act:`."""
    t = lit.lstrip()
    s = t.lstrip(">]}) \t\r\n")
    return s if s else t


def _flex(lit):
    """A literal marker as a regex: whitespace runs flexible, digits any number (call indices), the rest exact."""
    out, i = [], 0
    while i < len(lit):
        c = lit[i]
        if c.isspace():
            j = i
            while j < len(lit) and lit[j].isspace():
                j += 1
            out.append(r"\s*")
            i = j
        elif c.isdigit():
            j = i
            while j < len(lit) and lit[j].isdigit():
                j += 1
            out.append(r"\d+")
            i = j
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _json_spans(s):
    """[(start, end, value)] of every JSON object / array in s, left to right."""
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(s):
        if s[i] in "{[":
            try:
                val, end = dec.raw_decode(s, i)
                out.append((i, end, val))
                i = end
                continue
            except ValueError:
                pass
        i += 1
    return out


def _is_args(v, want):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return False
    return isinstance(v, dict) and v == want


def derive_reader(ct):
    """The call syntax read off the template's own rendering of a known call -- a spec parse_with_reader reads model
    output with -- or None when the template renders no call. Shapes:
      json      the call is a JSON object (or list of them) carrying the name: {"open", "close"}
      name+json the name sits in markup, the arguments are a JSON object: {"call_open", "mid"}
      tagged    one markup element per argument: {"call_open", "name_tail", "kv_open", "kv_mid", "val_close",
                "call_close"}
      pythonic  [name(key="value")]: {"open"}"""
    a1 = {"city": _RD_A1, "unit": _RD_B1}
    a2 = {"city": _RD_A2, "unit": _RD_B2}
    one = _rd_call_text(ct, [(_RD_NAME, a1)])
    if not one or _RD_NAME not in one:
        return None
    two = _rd_call_text(ct, [(_RD_NAME, a1), (_RD_NAME2, a2)])
    if two and _RD_NAME2 not in two:
        two = None
    n0 = one.index(_RD_NAME)
    spans = _json_spans(one)
    # json: a JSON value whose object(s) carry the name
    for s0, s1, v in spans:
        objs = v if isinstance(v, list) else [v]
        if any((_call_from_obj(o) or (None,))[0] == _RD_NAME for o in objs):
            return {"shape": "json", "open": one[:s0].strip(), "close": one[s1:].strip(), "list": isinstance(v, list),
                    "example": one}
    # name+json: the arguments are a JSON object after the name
    for s0, s1, v in spans:
        if s0 > n0 and (_is_args(v, a1) or (isinstance(v, dict) and v.get("city") == _RD_A1)):
            pre, mid = one[:n0], one[n0 + len(_RD_NAME):s0]
            call_open = pre
            if two:
                t_n2 = two.index(_RD_NAME2)
                first = next((e for b, e, w in _json_spans(two) if b < t_n2 and (_is_args(w, a1) or
                              (isinstance(w, dict) and w.get("city") == _RD_A1))), None)
                if first is not None:
                    call_open = _lcs_suffix(pre, two[first:t_n2]) or pre
            return {"shape": "name+json", "call_open": _opener(call_open), "mid": mid, "example": one}
    # pythonic: name( ... key="value" ... )
    m = re.search(r"\[?\s*" + re.escape(_RD_NAME) + r"\s*\(", one)
    if m and re.search(r"city\s*=", one[m.end():]):
        return {"shape": "pythonic", "open": one[:m.start()].strip(), "example": one}
    # tagged: key / value markup per argument. The name nearest the first argument is the call's own (a header
    # may repeat it); the value closer is what the last value and the middle values share (a separator follows
    # only between them); a closer that repeats the key (`<city>v</city>`) is kept as a key back-reference.
    v1 = one.find(_RD_A1)
    k1 = one.rfind("city", 0, v1) if v1 >= 0 else -1
    n0 = one.rfind(_RD_NAME, 0, k1) if k1 >= 0 else -1
    k2 = one.find("unit", v1 + len(_RD_A1)) if v1 >= 0 else -1
    v2 = one.find(_RD_B1, k2 + 4) if k2 >= 0 else -1
    if min(n0, k1, v1, k2, v2) < 0:
        return None
    K = "\x00KEY\x00"
    lead, between = one[n0 + len(_RD_NAME):k1], one[v1 + len(_RD_A1):k2]
    kv_open = _lcs_suffix(lead, between)
    name_tail = lead[:len(lead) - len(kv_open)]
    kv_mid = one[k1 + 4:v1]
    mid_t = between[:len(between) - len(kv_open)].replace("city", K)
    post_t = one[v2 + len(_RD_B1):].replace("unit", K, 1)
    n = 0
    while n < min(len(mid_t), len(post_t)) and mid_t[n] == post_t[n]:
        n += 1
    val_close, sep, call_close = mid_t[:n], mid_t[n:], post_t[n:]
    pre = one[:n0]
    call_open = pre
    if two:
        t_n2 = two.rfind(_RD_NAME2, 0, two.find(_RD_A2))
        e1 = two.rfind(_RD_B1, 0, t_n2)
        if t_n2 >= 0 and e1 >= 0:
            call_open = _lcs_suffix(pre, two[e1 + len(_RD_B1):t_n2]) or pre
    return {"shape": "tagged", "call_open": _opener(call_open), "name_tail": name_tail, "kv_open": kv_open,
            "kv_mid": kv_mid, "val_close": val_close, "sep": sep, "call_close": call_close.strip().replace(K, "unit"),
            "example": one}


_RD_NAME_RE = r"([A-Za-z_][\w.\-:/]*)"


def parse_with_reader(text, spec, tools=None):
    """[(name, args)] read from raw model text with a derive_reader spec -- only text carrying the template's own
    markers counts (a bare JSON reply is a call only when the template's call is a bare JSON value too)."""
    t = text or ""
    sc = tool_schemas(tools)
    shape = (spec or {}).get("shape")
    if shape == "json":
        op = spec.get("open") or ""
        if op:
            m = re.search(_flex(op), t)
            if not m:
                return []
            region = t[m.end():]
        else:
            region = re.sub(r"^```(?:json)?\s*", "", t.strip())
            if region[:1] not in "[{":
                return []
        return _calls_in_json(region)
    if shape == "name+json":
        out = []
        co = (spec.get("call_open") or "").strip()
        pat = (_flex(co) if co else r"^\s*") + _RD_NAME_RE + _flex(spec.get("mid") or "")
        for m in re.finditer(pat, t):                         # no opener: the reply must START with the call
            rest = t[m.end():]
            lead = len(rest) - len(rest.lstrip())
            if rest[lead:lead + 1] != "{":
                continue
            try:
                v, _ = json.JSONDecoder().raw_decode(rest, lead)
            except ValueError:
                continue
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            out.append((m.group(1), v))
        return out
    if shape == "pythonic":
        op = spec.get("open") or ""
        if op:
            m = re.search(_flex(op), t)
            if not m:
                return []
            t = t[m.end():]
        s = t.lstrip()
        if s.startswith("["):
            body = _balanced_list(s, 0)
            return _pythonic(body) if body else []
        m = re.search(r"[A-Za-z_][\w.]*\s*\(", s)
        if not m or m.start() > 0:
            return []
        body = _balanced_list(s, 0)
        return _pythonic(body) if body else []
    if shape == "tagged":
        out = []
        co, nt = spec.get("call_open") or "", spec.get("name_tail") or ""
        ko, km, vc, sp, cc = (spec.get(k) or "" for k in ("kv_open", "kv_mid", "val_close", "sep", "call_close"))
        if not (co.strip() or nt.strip()):
            return []                                           # no markup to anchor on: never guess
        name_re = r"([^\s<>\"'=]+?)" if nt.strip() else _RD_NAME_RE
        heads = list(re.finditer(_flex(co) + name_re + _flex(nt), t))
        K = "\x00KEY\x00"
        vc_re = "(?P=key)".join(_flex(x) for x in vc.split(K)) if vc.strip() else \
            (r"(?=" + _flex(sp + ko) + r"|$)" if (sp + ko).strip() else r"$")
        kv_re = _flex(ko) + r"(?P<key>[\w.\-]+)" + _flex(km) + r"(?P<val>.*?)" + vc_re
        for i, m in enumerate(heads):
            stop = heads[i + 1].start() if i + 1 < len(heads) else len(t)
            seg = t[m.end():stop]
            em = re.search(_flex(cc), seg) if cc.strip() else None
            body = seg[:em.start()] if em else seg
            name = m.group(1).strip()
            props = (sc or {}).get(name) or {}
            pairs = [(x.group("key"), x.group("val")) for x in re.finditer(kv_re, body, re.S)]
            out.append((name, {k: _coerce(_strip1(v), props.get(k)) for k, v in pairs}))
        return out
    return []


_DECISION_TAGS =("decision-model", "calibrated-probabilities", "system-one")
_DECISION_NAME = re.compile(r"(^|[^a-z])(jev|openjev|nanojev|laya|semif)([^a-z]|$)|decision[-_ ]?model", re.I)


def _d3_format(model, conf) -> str | None:
    """"d3-code-readout-v1" for a d3 package (vllm-sr/d3 family: decision_config.json next to the weights, or a GGUF
    `pollard-d3 convert` tagged), else None. Its answer is a separate readout over the last token, not a letter."""
    try:
        import pollard_d3 as D3
    except ImportError:
        return None
    meta = conf if isinstance(conf, dict) and "general.architecture" in conf else None
    return D3.FORMAT_ID if D3.is_d3(str(model), meta) else None


def _is_decision(model, conf) -> bool:
    """A decision HEAD first (clef: joint_head_config.json in the HF dir, a "<arch>.decision.*" block
    in the GGUF), then card tags (HF README front-matter or GGUF general.tags), then the names this
    family ships under. Never from the body's architecture alone: OpenJev is a stock qwen35 body."""
    if os.path.isdir(str(model)) and os.path.isfile(os.path.join(model, "joint_head_config.json")):
        return True
    if _d3_format(model, conf):                           # d3: decision_config.json, code-readout v1
        return True
    if isinstance(conf, dict) and any(".decision." in str(k) for k in conf):
        return True
    tags = []
    if isinstance(conf, dict):
        t = conf.get("general.tags")
        if isinstance(t, (list, tuple)):
            tags += [str(x).lower() for x in t]
        for k in ("general.name", "general.basename", "general.finetune"):
            if _DECISION_NAME.search(str(conf.get(k, ""))):
                return True
    if os.path.isdir(str(model)):
        rd = os.path.join(model, "README.md")
        if os.path.isfile(rd):
            try:
                head = open(rd, encoding="utf-8", errors="replace").read(4000)
                fm = head.split("---")[1] if head.startswith("---") and head.count("---") >= 2 else ""
                tags += re.findall(r"-\s*([a-z0-9_\-]+)", fm.lower())
            except OSError:
                pass
    if any(t in _DECISION_TAGS for t in tags):
        return True
    return bool(_DECISION_NAME.search(os.path.basename(str(model).rstrip("/"))))


_CODER_NAME = re.compile(r"coder|codestral|starcoder|codegen|code[-_]?llama|devstral|codegemma|granite-\d+b-code", re.I)
_TRANSLATE_NAME = re.compile(r"translat|nllb|madlad|tower(instruct|plus)?|seamless|hy-?mt|opus-mt|(^|[-_])mt(\d|[-_]|$)", re.I)


def calib_profile(kind: dict, name: str = "") -> tuple:
    """(profile, why) -- the pollard-calib domain mix for this kind of model. The imatrix spends its
    protection where the calibration activations are, so a translator calibrated on English prose
    or an agent model that never saw a tool call is protected for the wrong job."""
    base = os.path.basename(str(name).rstrip("/\\"))
    if _TRANSLATE_NAME.search(base):
        return "translation", f"name '{base}' reads as a translation model"
    if _CODER_NAME.search(base):
        return "code", f"name '{base}' reads as a code model"
    if kind.get("agentic"):
        return "agent", "chat template drives tool calls"
    return "general", "no stronger signal than a general chat/base model"


def classify(model) -> dict:
    """What this model is, and what that implies for measuring it."""
    tpl, archs, conf = _template_and_arch(model)
    low, joined = tpl.lower(), " ".join(archs)
    why = []

    instruct = bool(tpl)
    if instruct:
        why.append("ships a chat template -> instruct-tuned, not a raw-text model")
    # Reasoning: rendered from the template itself (delimiters, toggle, always-on), names as fallback.
    rinfo = reasoning_info(model, tpl, archs, conf)
    thinking = rinfo["reasoning"]
    if thinking:
        why.append(f"reasoning model ({rinfo['mode']}): {rinfo['open'] or ''}...{rinfo['close'] or ''}"
                   + (f" -- {rinfo['why'][0]}" if rinfo["why"] else ""))
    # Tool calling: the parser family the template writes calls in (rendered, llama.cpp-style sniffing first).
    tinfo = _safe(toolcall_info, model, tpl, archs, conf) or {"family": "none"}
    agentic = len(re.findall(r"tool_call|tool_response|function_call|\btools\b", low)) >= 3 \
        or tinfo.get("family", "none") != "none"
    if agentic:
        fam = tinfo.get("family", "none")
        why.append(f"template drives tool calls ({len(re.findall('tool', low))} mentions)"
                   + (f"; call syntax: {fam}" if fam != "none" else ""))
    # Modalities. HF puts each one in its own sub-config (vision_config, audio_config, ...), the
    # template carries its placeholder tokens, and the architecture name usually says so too. A
    # model that takes or emits something other than text cannot be scored like text at all.
    blob = (low + " " + joined.lower() + " " +
            " ".join(k.lower() for k in conf) + " " +
            " ".join(str(v).lower()[:200] for k, v in conf.items() if "config" in str(k).lower()))
    MODALITY = {
        "image":  (r"vision_config|image_token|<image>|pixel_values|mm_?proj|imagetext|vision2seq|"
                   r"visiontext|image_size"),
        "audio":  r"audio_config|audio_token|<audio>|whisper|wav2vec|speech_encoder|mel_bins|qwen2audio",
        "video":  r"video_config|video_token|<video>|num_frames|frame_rate|videollava|video_encoder",
        "speech_out": r"vocoder|codec_config|snac|speech_decoder|tts|audio_head",
    }
    modalities = [m for m, pat in MODALITY.items() if re.search(pat, blob)]
    # A declared modality is not a usable one. gemma-4-12B-it carries vision_config, audio_config
    # and the projection layers, but the encoder TOWERS are not in the release -- the checkpoint
    # holds 666 language-model tensors, one embed_vision, one embed_audio and a 9-tensor embedder.
    # Reporting it as image+audio+video would put a capability on the card the weights cannot do.
    declared = list(modalities)
    if modalities and os.path.isdir(model):
        have, proj = _modality_evidence(model)
        if have is not None:
            # A modality is usable if the checkpoint carries EITHER an encoder tower (repeated
            # blocks, the classic design) OR the projection weights an encoder-free model uses
            # instead. Gemma 4 is encoder-free on purpose: Google replaced a 550M vision encoder
            # with one 35M matmul and dropped the audio conformer entirely, projecting 40ms/16kHz
            # chunks straight into the embedding space. Demanding blocks there would report a
            # genuinely multimodal model as text-only.
            modalities = [m for m in modalities
                          if have.get(m, 0) >= 2 or proj.get(m, 0) >= 1 or m == "speech_out"]
            for m in declared:
                if m not in modalities:
                    why.append(f"{m}: declared in the config, but neither an encoder tower nor "
                               f"projection weights are in the checkpoint -- not usable here")
                elif have.get(m, 0) < 2:
                    why.append(f"{m}: encoder-free -- projection weights, no tower "
                               f"(needs the mmproj alongside the model)")
    vision = "image" in modalities
    for m in modalities:
        why.append(f"{m} signals in the template/config/architecture")
    # A diffusion LLM (Dream / LLaDA / RND1 / DiffusionGemma) is not decoded left-to-right: it
    # unmasks a canvas over N steps. Perplexity and next-token KL are computed against an objective
    # it was never trained on, and llama-server cannot decode it at all -- generation, and so the
    # coherence gate and chat, go through llama-diffusion-cli. The architecture string says so.
    diffusion = any(str(x).lower().replace("_", "-") in DIFFUSION_ARCHS for x in archs) or \
        str((conf or {}).get("model_type", "")).lower().replace("_", "-") in DIFFUSION_ARCHS
    if diffusion:
        why.append("diffusion architecture -> masked-denoising decoder; AR perplexity/KL are not "
                   "its objective; generate with llama-diffusion-cli, not llama-server")
    # A DECISION model (Jev / OpenJev / Laya / d1-style) answers typed questions from the logits at the
    # first output position and never writes prose. Its product is a probability, so the coherence
    # gate and a text perplexity say nothing about a rung; calibration against f16 does. The card
    # says so ("decision-model" tag), and so do the names these ship under.
    decision = _is_decision(model, conf)
    decision_format = _d3_format(model, conf) if decision else None
    if decision_format:
        why.append(f"d3 decision model ({decision_format}) -> non-causal body, last-token pooling, FP32 readout over "
                   "255 answer codes; gate = readout KL / agreement / ECE vs bf16 (pollard-d3), not first-position "
                   "letters; the readout ships byte-exact in every lane")
        if thinking or agentic:
            why.append("the chat template's thinking / tool-call blocks are the base model's: a d3 model never "
                       "generates, so neither is measured (no reasoning share, no tool-call gate)")
        thinking = agentic = False
    elif decision:
        why.append("decision model -> answers from first-position logits; gate = typed-set "
                   "agreement + option KL vs f16 (pollard-decision), not free-text coherence")
    if not instruct:
        why.append("no chat template -> base model; raw-text perplexity is meaningful here")

    # Anything beyond text changes what a "score" even means: a text corpus exercises none of the
    # encoder that makes the model what it is, so a text-only number is partial by construction.
    multimodal = [m for m in modalities if m != "speech_out"]
    out = {
        "instruct": instruct, "thinking": thinking, "agentic": agentic, "vision": vision,
        "diffusion": diffusion, "decision": decision, "decision_format": decision_format,
        "runtime": "llama-diffusion-cli" if diffusion else "llama-server",
        "modalities": modalities, "base": not instruct,
        # A model tuned away from raw text must be scored on text it was tuned FOR, or the number
        # describes the mismatch rather than the build.
        "eval": ("diffusion" if diffusion else
                 "decision" if decision else
                 "multimodal" if multimodal else
                 "raw-text" if not instruct else "in-domain"),
        "gate_tokens": TOKENS_THINKING if thinking else TOKENS_PLAIN,
        "reasoning": {k: v for k, v in rinfo.items() if k != "why"},
        "tools": {k: v for k, v in tinfo.items() if k != "why"},
        "why": why,
    }
    names = [str(x) for x in (model, (conf or {}).get("general.name"), (conf or {}).get("_name_or_path")) if x]
    out["calib_profile"], out["calib_profile_why"] = calib_profile(out, names[0] if names else "")
    for extra in names[1:]:                                  # a GGUF's general.name can say "coder" too
        p, w = calib_profile({}, extra)
        if p != "general" and out["calib_profile"] in ("general", "agent"):
            out["calib_profile"], out["calib_profile_why"] = p, w
            break
    return out


def describe(k: dict) -> str:
    tags = [n for n in ("base", "instruct", "thinking", "agentic", "diffusion", "decision") if k.get(n)]
    tags += k.get("modalities", [])
    return "+".join(tags) or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True, help="a .gguf or an HF checkout")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args()
    k = classify(a.model)
    if a.json:
        print(json.dumps(k, indent=1)); return
    print(f"== pollard-modelkind :: {a.model}")
    print(f"   kind        : {describe(k)}")
    for w in k["why"]:
        print(f"                 - {w}")
    note = {"in-domain": "   (raw Wikipedia would measure the mismatch, not the build)",
            "multimodal": "   (a TEXT corpus exercises none of the " +
                          "/".join(m for m in k["modalities"] if m != "speech_out") +
                          " encoder -- any text-only score is partial by construction)",
            "raw-text": ""}
    print(f"   eval corpus : {k['eval']}{note.get(k['eval'], '')}")
    print(f"   calib mix   : pollard-calib --profile {k['calib_profile']}   ({k['calib_profile_why']})")
    r = k.get("reasoning") or {}
    if r.get("reasoning"):
        tog = r.get("toggle") or {}
        print(f"   reasoning   : {r['mode']}  {r.get('open') or ''} ... {r.get('close') or ''}"
              + (f"  (switch: {tog.get('var')})" if tog else "")
              + (f"  (prefilled by the template: {r['prefill']!r})" if r.get("prefill") else "")
              + f"  [{r.get('source')}]")
        print(f"   calib       : pollard-calib adds a reasoning share rendered with these delimiters"
              + (" (thinking on AND off)" if r["mode"] == "toggle" else ""))
    t = k.get("tools") or {}
    if t.get("family", "none") != "none":
        print(f"   tool calls  : {t['family']}  [{t.get('source')}]"
              + ("" if t.get("supports_tools") else "  (template renders no tool list)"))
        if t.get("example"):
            print("   call syntax : " + t["example"].replace("\n", "\\n")[:160].encode("ascii", "backslashreplace").decode())
        print("   serve with  : " + serve_with_line(t))
    print(f"   gate budget : {k['gate_tokens']} tokens"
          + ("   (thinking block runs first)" if k["thinking"] else ""))


if __name__ == "__main__":
    main()
