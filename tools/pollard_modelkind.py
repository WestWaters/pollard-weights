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
# tokens reasoning, so a short budget reads as incoherence.
TOKENS_PLAIN, TOKENS_THINKING = 96, 220


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


def _jinja_env():
    """A jinja2 environment that renders HF chat templates the way transformers does (no tokenizer)."""
    import jinja2
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment

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

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_Generation, jinja2.ext.loopcontrols])
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

    def __init__(self, source, specials=None):
        self.source = source
        self.specials = dict(specials or {})
        self._tpl = _jinja_env().from_string(source)

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


_DECISION_TAGS = ("decision-model", "calibrated-probabilities", "system-one")
_DECISION_NAME = re.compile(r"(^|[^a-z])(jev|openjev|nanojev|laya|semif)([^a-z]|$)|decision[-_ ]?model", re.I)


def _is_decision(model, conf) -> bool:
    """A decision HEAD first (clef: joint_head_config.json in the HF dir, a "<arch>.decision.*" block
    in the GGUF), then card tags (HF README front-matter or GGUF general.tags), then the names this
    family ships under. Never from the body's architecture alone: OpenJev is a stock qwen35 body."""
    if os.path.isdir(str(model)) and os.path.isfile(os.path.join(model, "joint_head_config.json")):
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
    agentic = len(re.findall(r"tool_call|tool_response|function_call|\btools\b", low)) >= 3
    if agentic:
        why.append(f"template drives tool calls ({len(re.findall('tool', low))} mentions)")
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
    if decision:
        why.append("decision model -> answers from first-position logits; gate = typed-set "
                   "agreement + option KL vs f16 (pollard-decision), not free-text coherence")
    if not instruct:
        why.append("no chat template -> base model; raw-text perplexity is meaningful here")

    # Anything beyond text changes what a "score" even means: a text corpus exercises none of the
    # encoder that makes the model what it is, so a text-only number is partial by construction.
    multimodal = [m for m in modalities if m != "speech_out"]
    out = {
        "instruct": instruct, "thinking": thinking, "agentic": agentic, "vision": vision,
        "diffusion": diffusion, "decision": decision,
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
    print(f"   gate budget : {k['gate_tokens']} tokens"
          + ("   (thinking block runs first)" if k["thinking"] else ""))


if __name__ == "__main__":
    main()
