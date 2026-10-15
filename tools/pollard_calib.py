#!/usr/bin/env python3
"""pollard-calib -- build a calibration corpus for the KIND of model being quantized (Calib 3.0 + profiles).

The calibration corpus is what lets imatrix / pollard-sensitivity route to the tensors that
matter -- and the biggest measured quality lever the good quantizers (Unsloth Dynamic 3.0) pay
for is NOT exotic types, it's a bigger, domain-MIXED calib (chat + code + math + multilingual +
prose) with enough coverage. Undercovered experts hard-fail low-bit ("Missing importance matrix
... bailing out") and bloat the build; a prose-only calib misranks. This tool assembles a balanced,
high-coverage corpus so every downstream imatrix/sensitivity pass sees all the domains.

PROFILES weight the domains for what the model is FOR -- a translation model calibrated on English
prose spends its protected bits on the wrong tensors:

  general      prose + code + math + chat + multilingual, equal weight (the Calib 3.0 default)
  translation  parallel sentence pairs rendered through the model's OWN chat template, many languages
  agent        tool-call traces, with the tool schemas passed through the model's chat template
  code         raw code + code-chat turns
  auto         picked from pollard-modelkind (agentic -> agent, coder -> code, translator -> translation)

--model <hf-dir-or-id> renders every chat-shaped sample through that model's own tokenizer chat
template (transformers), so the imatrix sees the exact turn tokens the model will serve with. Without
it a generic ChatML rendering is used.

--self-generate replaces the reference answers with the f16/bf16 model's OWN responses (AngelSlim's
target-generated calibration): the prompts go to a llama-server / vLLM / mlx_lm server (--server URL)
or to a local llama-server this tool starts on --self-generate-gguf, and the model's answers are
rendered back through its template. The calibration then matches the distribution the model actually
produces, not a dataset author's.

Every dataset is open-licensed and pinned to a revision; the ids, revisions and licences land in a
<out>.sources.json sidecar and (with --record-model) in the workspace MANIFEST, so a build can say
exactly what it was calibrated on.

It emits a raw text corpus (newline-separated samples) that works for BOTH `llama-imatrix -f`
(reads raw text) and `pollard-export --calib` / gptqmodel (one sample per line). Prefers real HF
datasets when `datasets` is importable and reachable; ALWAYS falls back to bundled seeds so it
produces a valid (smaller) corpus offline. Optionally writes a held-out split so you can check the
imatrix isn't overfit (KL on unseen text should track KL on the calib text).
Held-out splits deduplicate sample content across domains before splitting;
offline seeds therefore yield fewer samples than --per-domain requests.

  pollard-calib --out calib.txt                          # balanced default (~all domains)
  pollard-calib --out calib.txt --per-domain 400 --held-out calib.heldout.txt
  pollard-calib --out calib.txt --domains code,math,chat # only these (e.g. a coding model)
  pollard-calib --out calib.txt --profile agent --model Qwen/Qwen3-8B
  pollard-calib --out calib.txt --profile auto --model ./my-model --self-generate --server http://127.0.0.1:8080
  pollard-calib --out calib.txt --min-chars 200 --seed 0
The imatrix/sensitivity COMPUTE (GPU) is a separate step -- this only builds the corpus.
"""
import argparse, json, math, os, random, re, sys, textwrap, unicodedata
from pathlib import Path

DOMAINS = ["prose", "code", "codechat", "math", "chat", "multilingual", "translation", "agent"]

# Domain weights per profile. --per-domain is the sample count for weight 1.
PROFILES = {
    "general":     {"prose": 1, "code": 1, "math": 1, "chat": 1, "multilingual": 1},
    "translation": {"translation": 4, "multilingual": 2, "chat": 1, "prose": 1},
    "agent":       {"agent": 4, "chat": 2, "code": 1, "math": 1},
    "code":        {"code": 3, "codechat": 3, "chat": 1, "math": 1},
}
PROFILE_CHOICES = sorted(PROFILES) + ["auto"]

# Languages the multilingual and translation domains cover (FLORES/fineweb-2 codes). Wide on purpose:
# a calib that only ever sees Latin script never exercises the embedding rows of the others.
LANGS = ["deu_Latn", "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "nld_Latn", "pol_Latn",
         "rus_Cyrl", "ukr_Cyrl", "tur_Latn", "arb_Arab", "hin_Deva", "ben_Beng", "cmn_Hani",
         "jpn_Jpan", "kor_Hang", "vie_Latn", "ind_Latn", "tha_Thai", "swh_Latn", "ces_Latn",
         "ell_Grek", "heb_Hebr", "fas_Arab"]
LANG_NAME = {"eng_Latn": "English", "deu_Latn": "German", "fra_Latn": "French", "spa_Latn": "Spanish",
             "ita_Latn": "Italian", "por_Latn": "Portuguese", "nld_Latn": "Dutch", "pol_Latn": "Polish",
             "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian", "tur_Latn": "Turkish", "arb_Arab": "Arabic",
             "hin_Deva": "Hindi", "ben_Beng": "Bengali", "cmn_Hani": "Chinese", "jpn_Jpan": "Japanese",
             "kor_Hang": "Korean", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian", "tha_Thai": "Thai",
             "swh_Latn": "Swahili", "ces_Latn": "Czech", "ell_Grek": "Greek", "heb_Hebr": "Hebrew",
             "fas_Arab": "Persian", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
             "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ar": "Arabic",
             "hi": "Hindi", "ko": "Korean", "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "sw": "Swahili"}
# fineweb-2 names Mandarin by its macro-language; FLORES+ splits it by script
_FINEWEB_CODE = {"cmn_Hani": "cmn_Hani", "arb_Arab": "arb_Arab"}

# HF sources per domain. Proven-working source FIRST, then fallbacks, then the bundled seed.
# Every source is OPEN-LICENSED and PINNED to a revision so a build can say exactly what it saw.
# Removed 2026-10-14: tatsu-lab/alpaca (CC-BY-NC-4.0), codeparrot/codeparrot-clean-valid and
# bigcode/the-stack-smol-xs (no licence declared), papluca/language-identification (no licence) --
# and the multilingual domain's wikitext fallback, which made "multilingual" quietly English.
# Updated for `datasets` 5.x (bare "wikitext" and script-based flores200 broke -- do not restore them).
_HF = {
    "prose": [dict(id="Salesforce/wikitext", config="wikitext-103-raw-v1", split="train", kind="text",
                   field="text", license="cc-by-sa-3.0", revision="b08601e04326c79dfdd32d625aee71d232d685c3")],
    "code": [dict(id="ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", kind="text", field="solution",
                  license="mit", revision="5f839b1f368a76b161028bb9edff055db34022b2")],
    "codechat": [dict(id="m-a-p/CodeFeedback-Filtered-Instruction", split="train", kind="chat",
                      field=("query", "answer"), license="apache-2.0",
                      revision="a08c213a9748c66c15d0225814be80a2e77adf4a"),
                 dict(id="ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", kind="chat",
                      field=("problem", "solution"), license="mit",
                      revision="5f839b1f368a76b161028bb9edff055db34022b2")],
    "math": [dict(id="openai/gsm8k", config="main", split="train", kind="pair", field=("question", "answer"),
                  license="mit", revision="740312add88f781978c0658806c59bc2815b9866")],
    "chat": [dict(id="HuggingFaceH4/ultrachat_200k", split="train_sft", kind="messages", field="messages",
                  license="mit", revision="8049631c405ae6576f93f445c6b8166f76f5505a")],
    "multilingual": [dict(id="HuggingFaceFW/fineweb-2", split="train", kind="perlang", field="text",
                          license="odc-by", revision="af9c13333eb981300149d5ca60a8e9d659b276b9"),
                     dict(id="CohereLabs/aya_dataset", split="train", kind="pair", field=("inputs", "targets"),
                          license="apache-2.0", revision="f9ea04583f02a8f86404ff6c58bf75fe637df8a2")],
    # FLORES+ DEV split only: devtest is what `pollard-taskeval --suite mt --mt-set flores` scores on,
    # and WMT24++ (the default MT eval) is never used for calibration -- calibrating on the test set
    # would make the retention table measure memory.
    "translation": [dict(id="openlanguagedata/flores_plus", split="dev", kind="parallel", field="text",
                         license="cc-by-sa-4.0", revision="e707e62e762a9e2faca3933aa315f4665cc3cd8f",
                         note="gated (auto-approve): needs an HF login that accepted the terms"),
                    dict(id="google/smol", split="train", kind="smol", field=("src", "trg"),
                         license="cc-by-4.0", revision="fdaff3a1a019f89fa30a562a85d7c1d3e9150444")],
    "agent": [dict(id="NousResearch/hermes-function-calling-v1", config="func_calling_singleturn",
                   split="train", kind="hermes", field="conversations", license="apache-2.0",
                   revision="dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
              dict(id="NousResearch/hermes-function-calling-v1", config="func_calling",
                   split="train", kind="hermes", field="conversations", license="apache-2.0",
                   revision="dae3e1d28cfbcf4b915c04ea1e072030529b4bda")],
}
_SMOL_LANGS = ["es", "ja", "ko", "hi", "id", "th", "sw", "fa", "bn", "ta"]

# Bundled seeds -- always available so the tool never emits an empty domain. Short but real,
# spanning structure the imatrix needs to see (prose cadence, code tokens, digits/operators,
# instruction/answer turns, non-English scripts). Repeated+shuffled up to --per-domain.
_SEED = {
    "prose": [
        "The river carved the valley over ten thousand years, leaving terraces of pale stone.",
        "In economics, a market clears when the quantity supplied equals the quantity demanded.",
        "She opened the letter slowly, unsure whether the news would change everything or nothing.",
        "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    ],
    "code": [
        "def binary_search(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] == x: return mid\n        if a[mid] < x: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
        "import numpy as np\narr = np.arange(12).reshape(3, 4)\nprint(arr.sum(axis=0), arr.mean())",
        "SELECT user_id, COUNT(*) AS n FROM events WHERE ts > NOW() - INTERVAL '7 days' GROUP BY user_id ORDER BY n DESC LIMIT 10;",
        "const memo = new Map();\nfunction fib(n){ if(n<2) return n; if(memo.has(n)) return memo.get(n); const v=fib(n-1)+fib(n-2); memo.set(n,v); return v; }",
    ],
    "math": [
        "Question: If 3x + 7 = 22, what is x?\nAnswer: 3x = 15, so x = 5.",
        "Question: A train travels 240 km in 3 hours. What is its average speed?\nAnswer: 240 / 3 = 80 km/h.",
        "Question: What is the derivative of f(x) = x^3 - 2x?\nAnswer: f'(x) = 3x^2 - 2.",
        "Question: Compute 17 * 24.\nAnswer: 17 * 24 = 408.",
    ],
    "multilingual": [
        "La lluvia caía suavemente sobre los tejados de la ciudad dormida.",
        "Die Wissenschaft schreitet voran, indem sie Vermutungen streng überprüft.",
        "科学は仮説を厳密に検証することで前進する。",
        "La connaissance partagée est la seule qui grandit lorsqu'on la donne.",
        "المعرفة التي تُشارَك هي الوحيدة التي تكبر حين تُعطى.",
        "Знание, которым делятся, это единственное, что растёт, когда его отдают.",
        "知识是唯一一种分享之后反而会增长的东西。",
        "जो ज्ञान बाँटा जाता है, वही देने पर बढ़ता है।",
        "지식은 나눌수록 커지는 유일한 것이다.",
        "Kiến thức là thứ duy nhất càng chia sẻ càng lớn lên.",
        "Bilgi, paylaşıldıkça büyüyen tek şeydir.",
        "Wiedza to jedyna rzecz, która rośnie, gdy się nią dzielimy.",
    ],
}
# Chat-shaped seeds are kept as structured records and rendered through the model's own template.
_SEED_CHAT = [
    [{"role": "user", "content": "Explain recursion in one sentence."},
     {"role": "assistant", "content": "Recursion is when a function solves a problem by calling itself on a smaller version of that problem until it reaches a base case."}],
    [{"role": "user", "content": "Give three tips for better sleep."},
     {"role": "assistant", "content": "Keep a consistent schedule, avoid screens before bed, and keep the room cool and dark."}],
    [{"role": "user", "content": "Summarize the water cycle."},
     {"role": "assistant", "content": "Water evaporates, condenses into clouds, falls as precipitation, and flows back to the sea, repeating continuously."}],
    [{"role": "user", "content": "What is the capital of France?"},
     {"role": "assistant", "content": "The capital of France is Paris."}],
]
_SEED_CODECHAT = [
    [{"role": "user", "content": "Write a Python function that reverses a linked list."},
     {"role": "assistant", "content": "```python\ndef reverse(head):\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev\n```\nIt walks the list once, pointing each node back at the previous one."}],
    [{"role": "user", "content": "Why does `[] == []` print False in JavaScript?"},
     {"role": "assistant", "content": "Arrays are objects, and `==` on objects compares references. Two literals create two distinct arrays, so the references differ."}],
    [{"role": "user", "content": "Fix this SQL: SELECT name FROM users WHERE age > 30 GROUP BY name HAVING COUNT(*) > 1 ORDER BY"},
     {"role": "assistant", "content": "The ORDER BY clause is missing its column:\n```sql\nSELECT name FROM users WHERE age > 30 GROUP BY name HAVING COUNT(*) > 1 ORDER BY name;\n```"}],
]
# Translation seeds: one "lang<TAB>english<TAB>translation" string per line.
_SEED_PAIRS = [
    "es\tThe weather is nice today.\tHoy hace buen tiempo.",
    "fr\tWhere is the train station?\tOù est la gare ?",
    "de\tI would like a cup of coffee, please.\tIch hätte gern eine Tasse Kaffee, bitte.",
    "it\tThe museum opens at nine in the morning.\tIl museo apre alle nove del mattino.",
    "pt\tThank you very much for your help.\tMuito obrigado pela sua ajuda.",
    "ru\tThe book is on the table.\tКнига лежит на столе.",
    "zh\tI am learning to cook.\t我在学做饭。",
    "ja\tThis city is very beautiful.\tこの町はとても美しいです。",
    "ar\tThe children are playing in the garden.\tالأطفال يلعبون في الحديقة.",
    "hi\tMy brother lives in Delhi.\tमेरा भाई दिल्ली में रहता है।",
    "ko\tThe meeting starts at three.\t회의는 세 시에 시작합니다.",
    "tr\tThe shop is closed on Sundays.\tDükkan pazar günleri kapalıdır.",
]
_SEED_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                      "unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Find flights between two airports.",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                                                      "date": {"type": "string"}},
                    "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {"name": "create_ticket", "description": "Open a support ticket.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "priority": {"type": "integer"}},
                    "required": ["title"]}}},
]
_SEED_AGENT = [
    ("What's the weather in Lisbon in celsius?", "get_weather", {"city": "Lisbon", "unit": "c"},
     {"temp": 21, "sky": "clear"}, "It is 21 C and clear in Lisbon right now."),
    ("Find me a flight from JFK to SFO on 2026-11-02.", "search_flights",
     {"origin": "JFK", "destination": "SFO", "date": "2026-11-02"},
     {"flights": [{"id": "UA123", "depart": "08:05"}]}, "UA123 departs JFK at 08:05 on November 2nd."),
    ("My invoice page crashes, please open a high priority ticket.", "create_ticket",
     {"title": "Invoice page crashes", "priority": 1}, {"ticket": 4821},
     "I opened ticket #4821 for the invoice page crash at high priority."),
]


# ----------------------------------------------------------------- rendering (the model's own template)

def chatml(messages, tools=None):
    """Generic rendering when the model's own template is unavailable: ChatML, tools Hermes-style."""
    out = []
    if tools:
        out.append("<|im_start|>system\nYou may call these tools:\n<tools>\n"
                   + json.dumps(tools, ensure_ascii=False) + "\n</tools><|im_end|>")
    for m in messages:
        content = m.get("content") or ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            content += ("\n" if content else "") + "<tool_call>\n" + json.dumps(
                {"name": fn.get("name"), "arguments": fn.get("arguments")}, ensure_ascii=False) + "\n</tool_call>"
        role = m.get("role", "user")
        if role == "tool":
            role, content = "user", f"<tool_response>\n{content}\n</tool_response>"
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(out)


class Renderer:
    """Renders a structured chat record through the target model's own chat template.

    The imatrix is a statistic of the activations the model sees; the activations it sees in service
    begin with its own turn tokens, not a dataset author's `<|user|>`. With no --model (or no
    transformers) the generic ChatML rendering above is used and the sidecar says so."""

    def __init__(self, model=None, trust_remote_code=False):
        self.tok, self.model, self.why = None, model, "generic ChatML (no --model)"
        if not model:
            return
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
            if getattr(tok, "chat_template", None):
                self.tok, self.why = tok, f"{model}'s own chat template"
            else:
                self.why = f"generic ChatML ({model} ships no chat template)"
        except Exception as e:                                   # noqa: BLE001
            self.why = f"generic ChatML (could not load {model}'s tokenizer: {repr(e)[:80]})"

    def __call__(self, messages, tools=None):
        if self.tok is None:
            return chatml(messages, tools)
        msgs = [dict(m) for m in messages]
        for m in msgs:                                          # templates differ on argument encoding
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except ValueError:
                        pass
        for attempt in (msgs, _flatten_tools(msgs)):
            try:
                kw = {"tools": tools} if tools else {}
                return self.tok.apply_chat_template(attempt, tokenize=False, **kw)
            except Exception:                                    # noqa: BLE001
                continue
        return chatml(messages, tools)


def _flatten_tools(messages):
    """For templates without a tool role / tool_calls: the calls and results as plain text turns."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["content"] = (m.get("content") or "") + "".join(
                "\n" + json.dumps({"name": tc.get("function", {}).get("name"),
                                   "arguments": tc.get("function", {}).get("arguments")}, ensure_ascii=False)
                for tc in m.pop("tool_calls"))
        if m.get("role") == "tool":
            m = {"role": "user", "content": "Tool result: " + str(m.get("content", ""))}
        out.append(m)
    return out


def render(record, renderer):
    """A record is plain text (str) or {"messages": [...], "tools": [...]}."""
    if isinstance(record, str):
        return record
    return renderer(record["messages"], record.get("tools"))


# ----------------------------------------------------------------- dataset extractors

def _hermes_record(conv, tools_json=None):
    """NousResearch hermes-function-calling row -> structured messages + tool schemas, so the calls
    reach the model's OWN tool-call syntax instead of Hermes' XML."""
    tools = None
    if tools_json:
        try:
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
        except ValueError:
            tools = None
    msgs = []
    for turn in conv or []:
        who, val = turn.get("from"), str(turn.get("value", ""))
        if who == "system":
            if tools is None:
                m = re.search(r"<tools>\s*(.*?)\s*</tools>", val, re.S)
                if m:
                    try:
                        tools = json.loads(m.group(1))
                    except ValueError:
                        pass
            continue                                            # Hermes' own system prompt: not the model's
        if who == "human":
            msgs.append({"role": "user", "content": val})
        elif who == "gpt":
            calls = []
            for blob in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", val, re.S):
                try:
                    c = json.loads(blob)
                except ValueError:
                    try:
                        import ast
                        c = ast.literal_eval(blob)
                    except Exception:                            # noqa: BLE001
                        continue
                calls.append({"type": "function", "function": {"name": c.get("name"),
                                                                "arguments": c.get("arguments", {})}})
            text = re.sub(r"<tool_call>.*?</tool_call>", "", val, flags=re.S).strip()
            m = {"role": "assistant", "content": text}
            if calls:
                m["tool_calls"] = calls
            msgs.append(m)
        elif who == "tool":
            msgs.append({"role": "tool", "content": re.sub(r"</?tool_response>", "", val).strip()})
    if isinstance(tools, dict):
        tools = [tools]
    if not msgs or not any(m["role"] == "assistant" for m in msgs):
        return None
    return {"messages": msgs, "tools": tools or None}


def translation_record(src_lang, tgt_lang, src, tgt):
    return {"messages": [
        {"role": "user", "content": f"Translate the following {LANG_NAME.get(src_lang, src_lang)} text into "
                                    f"{LANG_NAME.get(tgt_lang, tgt_lang)}.\n\n{src}"},
        {"role": "assistant", "content": tgt}]}


def _stream(load_dataset, src, config=None):
    cfg = config if config is not None else src.get("config")
    kw = {"split": src["split"], "streaming": True}
    if src.get("revision"):
        kw["revision"] = src["revision"]
    return load_dataset(src["id"], cfg, **kw) if cfg else load_dataset(src["id"], **kw)


def _extract(src, ex, min_chars):
    kind, field = src["kind"], src["field"]
    if kind == "text":
        t = str(ex.get(field, "")).strip()
        return t if len(t) >= min_chars else None
    if kind == "pair":
        t = "\n".join(str(ex.get(f, "")).strip() for f in field)
        return t if len(t) >= min_chars else None
    if kind == "chat":
        q, a = (str(ex.get(f, "")).strip() for f in field)
        if not q or not a or len(q) + len(a) < min_chars:
            return None
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
    if kind == "messages":
        msgs = [{"role": m.get("role"), "content": m.get("content", "")} for m in ex.get(field) or []
                if m.get("role") in ("system", "user", "assistant")]
        if not msgs or sum(len(m["content"]) for m in msgs) < min_chars:
            return None
        return {"messages": msgs}
    if kind == "hermes":
        return _hermes_record(ex.get(field), ex.get("tools"))
    return None


def _from_hf(domain, need, min_chars):
    """Try each HF source for a domain; return a list of records or [] if unavailable.
    The source that actually delivered is remembered in _USED for the sidecar / MANIFEST."""
    try:
        from datasets import load_dataset
    except Exception:
        return []
    for src in _HF.get(domain, []):
        out = []
        try:
            if src["kind"] == "perlang":                         # round-robin over languages
                per = max(1, math.ceil(need / len(LANGS)))
                for lang in LANGS:
                    got = 0
                    for ex in _stream(load_dataset, src, _FINEWEB_CODE.get(lang, lang)):
                        t = str(ex.get(src["field"], "")).strip()
                        if len(t) >= min_chars:
                            out.append(t[:4000]); got += 1
                        if got >= per:
                            break
            elif src["kind"] == "parallel":                      # FLORES+: zip eng with each language by id
                per = max(1, math.ceil(need / len(LANGS)))
                eng = {ex["id"]: ex["text"] for ex in _stream(load_dataset, src, "eng_Latn")}
                for i, lang in enumerate(LANGS):
                    got = 0
                    for ex in _stream(load_dataset, src, lang):
                        if ex.get("id") in eng:
                            pair = (eng[ex["id"]], ex["text"])
                            # both directions, alternating, so decoder AND encoder sides see each script
                            out.append(translation_record("eng_Latn", lang, *pair) if (got + i) % 2 == 0
                                       else translation_record(lang, "eng_Latn", pair[1], pair[0]))
                            got += 1
                        if got >= per:
                            break
            elif src["kind"] == "smol":
                per = max(1, math.ceil(need / len(_SMOL_LANGS)))
                for lang in _SMOL_LANGS:
                    got = 0
                    for ex in _stream(load_dataset, src, f"smolsent__en_{lang}"):
                        a, b = (str(ex.get(f, "")).strip() for f in src["field"])
                        if a and b:
                            out.append(translation_record("eng_Latn", lang, a, b)); got += 1
                        if got >= per:
                            break
            else:
                for ex in _stream(load_dataset, src):
                    r = _extract(src, ex, min_chars)
                    if r is not None:
                        out.append(r)
                    if len(out) >= need:
                        break
            if out:
                langs = {"perlang": LANGS, "parallel": LANGS, "smol": _SMOL_LANGS}.get(src["kind"])
                _USED[domain] = {**src, "config": ",".join(("smolsent__en_" + x) if src["kind"] == "smol" else x
                                                          for x in langs)} if langs else src
                return out[:need]
        except Exception as e:
            print(f"  [{domain}] HF source {src['id']} unavailable ({repr(e)[:60]}) -- trying next/seed",
                  file=sys.stderr)
    return []


_USED = {}


def _seed_records(domain):
    if domain == "chat":
        return [{"messages": m} for m in _SEED_CHAT]
    if domain == "codechat":
        return [{"messages": m} for m in _SEED_CODECHAT]
    if domain == "translation":
        out = []
        for line in _SEED_PAIRS:
            lang, en, tr = line.split("\t")
            out += [translation_record("eng_Latn", lang, en, tr), translation_record(lang, "eng_Latn", tr, en)]
        return out
    if domain == "agent":
        out = []
        for i, (q, name, args, result, answer) in enumerate(_SEED_AGENT):
            call = {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
            out.append({"tools": _SEED_TOOLS, "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": json.dumps(result)},
                {"role": "assistant", "content": answer}]})
        return out
    return list(_SEED.get(domain, []))


def build_domain(domain, need, min_chars, rng):
    """`need` samples for one domain: real HF data first, topped up (shuffled/repeated) from seeds."""
    got = _from_hf(domain, need, min_chars)
    src = "hf" if got else "seed"
    seeds = _seed_records(domain)
    while len(got) < need and seeds:
        batch = seeds[:]
        rng.shuffle(batch)
        got.extend(batch)
    return got[:need], src


# ----------------------------------------------------------------- self-generation (AngelSlim-style)

def openai_complete(base_url, model="pollard", max_tokens=512, temperature=0.0, timeout=600):
    """A completion function over any OpenAI-compatible server (llama-server, vLLM, mlx_lm.server)."""
    import urllib.request

    def complete(messages, tools=None):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            msg = json.load(r)["choices"][0]["message"]
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return out
    return complete


def self_generate(records, complete, log=None):
    """Replace the LAST assistant turn of every chat record with the model's own answer to the turns
    before it. Plain-text records (prose, raw code) pass through: there is no prompt to answer.
    A record whose request fails keeps its reference answer, and the count says so."""
    out, done, failed = [], 0, 0
    for r in records:
        if isinstance(r, str) or not r.get("messages"):
            out.append(r); continue
        msgs = r["messages"]
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        if last is None:
            out.append(r); continue
        try:
            ans = complete([dict(m) for m in msgs[:last]], r.get("tools"))
            if not (ans.get("content") or ans.get("tool_calls")):
                raise ValueError("empty answer")
            out.append({**r, "messages": msgs[:last] + [ans]})
            done += 1
        except Exception as e:                                   # noqa: BLE001
            out.append(r); failed += 1
            if log and failed <= 3:
                log(f"  self-generate: kept the reference answer ({repr(e)[:80]})")
    return out, done, failed


# ----------------------------------------------------------------- split + provenance

def split_domains(rows, fraction, rng):
    """Split unique content, retaining original formatting in both corpora."""
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("held-out fraction must be between 0 and 1")
    seen = set()
    train, held, stats = [], [], []
    for domain, source, samples in rows:
        unique = []
        for sample in samples:
            key = " ".join(unicodedata.normalize("NFC", sample).split())
            if key and key not in seen:
                seen.add(key)
                unique.append(sample)
        if len(unique) < 2:
            raise ValueError(f"{domain} needs at least two unique samples for a held-out split")
        rng.shuffle(unique)
        count = min(len(unique) - 1, max(1, int(len(unique) * fraction)))
        held.extend((domain, sample) for sample in unique[:count])
        train.extend((domain, sample) for sample in unique[count:])
        stats.append((domain, source, len(unique) - count, count))
    return train, held, stats


def same_output(first, second):
    """Catch relative paths, symlinks and existing hardlinks before writing."""
    if Path(first).resolve() == Path(second).resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False


def resolve_profile(profile, model=None):
    """'auto' -> the profile pollard-modelkind picks for this model (general when it cannot tell)."""
    if profile != "auto":
        return profile, "requested"
    if not model:
        return "general", "auto: no --model to classify, general mix"
    try:
        from pollard_modelkind import classify
        k = classify(model)
        return k.get("calib_profile", "general"), "auto: " + k.get("calib_profile_why", "pollard-modelkind")
    except Exception as e:                                       # noqa: BLE001
        return "general", f"auto: could not classify ({repr(e)[:60]}), general mix"


def domain_counts(profile, per_domain, domains=None):
    """{domain: n samples}. An explicit --domains list keeps the old equal-weight behaviour."""
    if domains:
        return {d: per_domain for d in domains}
    return {d: max(1, int(round(per_domain * w))) for d, w in PROFILES[profile].items()}


def provenance(profile, why, counts, stats, renderer_why, selfgen=None):
    """What this corpus was made of -- written beside it and into the MANIFEST."""
    doms = {}
    for d, src, ntr, nh in stats:
        s = _USED.get(d) if src == "hf" else None
        doms[d] = {"requested": counts.get(d), "train": ntr, "held_out": nh,
                   "source": ({k: s.get(k) for k in ("id", "config", "split", "revision", "license")} if s
                              else {"id": "unrecorded"} if src == "hf"
                              else {"id": "pollard-calib bundled seeds", "license": "apache-2.0"})}
    return {"tool": "pollard-calib", "profile": profile, "profile_why": why, "rendering": renderer_why,
            "self_generated": selfgen, "domains": doms}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--out", help="output corpus (raw text; default: workspace calibration/)")
    ap.add_argument("--profile", default="general", choices=PROFILE_CHOICES,
                    help="domain mix for what the model is for (default general; auto = from pollard-modelkind)")
    ap.add_argument("--domains", help=f"explicit comma list from {DOMAINS} at equal weight (overrides --profile)")
    ap.add_argument("--model", help="HF dir/id whose OWN chat template renders the chat-shaped samples "
                                    "(and which --profile auto classifies)")
    ap.add_argument("--trust-remote-code", action="store_true", help="allow the tokenizer's own code")
    ap.add_argument("--per-domain", type=int, default=300, help="target samples per domain (weight 1)")
    ap.add_argument("--min-chars", type=int, default=120, help="drop samples shorter than this")
    ap.add_argument("--held-out", help="also write a disjoint held-out corpus (overfit check)")
    ap.add_argument("--held-frac", type=float, default=0.1, help="fraction of each domain held out")
    ap.add_argument("--self-generate", action="store_true",
                    help="replace reference answers with the f16 model's own (needs --server or --self-generate-gguf)")
    ap.add_argument("--server", help="OpenAI-compatible base URL serving the f16/bf16 model (llama-server/vLLM/mlx_lm)")
    ap.add_argument("--self-generate-gguf", help="start a local llama-server on this f16 GGUF for --self-generate")
    ap.add_argument("--gen-max-tokens", type=int, default=512)
    ap.add_argument("--ngl", default="0", help="GPU layers for the local llama-server (default 0 = CPU; shared boxes)")
    ap.add_argument("--record-model", help="workspace model key: log the corpus + its sources in that MANIFEST")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.per_domain < 1 or a.min_chars < 1:
        ap.error("--per-domain and --min-chars must be positive")
    if a.held_out and (not math.isfinite(a.held_frac) or not 0 < a.held_frac < 1):
        ap.error("--held-frac must be between 0 and 1")
    if a.self_generate and not (a.server or a.self_generate_gguf):
        ap.error("--self-generate needs --server URL or --self-generate-gguf f16.gguf")

    rng = random.Random(a.seed)
    doms = None
    if a.domains is not None:
        doms = [d.strip() for d in a.domains.split(",") if d.strip()]
        bad = [d for d in doms if d not in DOMAINS]
        if bad:
            sys.exit(f"ERROR: unknown domain(s) {bad}; choose from {DOMAINS}")
        if not doms or len(doms) != len(set(doms)):
            ap.error("--domains must contain distinct domains")
    profile, why = resolve_profile(a.profile, a.model)
    if doms:
        profile, why = "custom", f"--domains {','.join(doms)}"
    counts = domain_counts(profile if not doms else "general", a.per_domain, doms)
    if not a.out:
        try:
            import pollard_workspace as ws
            a.out = os.path.join(ws.calibration_dir(create=True), "calib-3.0.txt")
        except Exception:
            a.out = "calib-3.0.txt"
    if a.held_out and same_output(a.out, a.held_out):
        ap.error("calibration and held-out paths must be different files")
    for path in (a.out, a.held_out):
        if path and os.path.dirname(os.path.abspath(path)):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    renderer = Renderer(a.model, a.trust_remote_code)
    complete, selfgen = None, None
    srv_ctx = None
    if a.self_generate:
        if a.server:
            complete = openai_complete(a.server, max_tokens=a.gen_max_tokens)
        else:
            from pollard_taskeval import served
            srv_ctx = served(a.self_generate_gguf, a.ngl, 8099, 8192)
            complete = openai_complete(srv_ctx.__enter__(), max_tokens=a.gen_max_tokens)

    train, held, stats = [], [], []
    rows = []
    try:
        for d, n in counts.items():
            samples, src = build_domain(d, n, a.min_chars, rng)
            if complete is not None:
                samples, ok, bad = self_generate(samples, complete, log=lambda m: print(m, file=sys.stderr))
                selfgen = selfgen or {"server": a.server or a.self_generate_gguf, "answers": 0, "kept_reference": 0}
                selfgen["answers"] += ok
                selfgen["kept_reference"] += bad
            samples = [render(s, renderer) for s in samples]
            rows.append((d, src, samples))
            if not a.held_out:
                rng.shuffle(samples)
                train += [(d, s) for s in samples]
                stats.append((d, src, len(samples), 0))
    finally:
        if srv_ctx is not None:
            srv_ctx.__exit__(None, None, None)
    if a.held_out:
        try:
            train, held, stats = split_domains(rows, a.held_frac, rng)
        except ValueError as exc:
            ap.error(str(exc))
        for d, _, ntr, nh in stats:
            if ntr + nh < counts.get(d, a.per_domain):
                print(f"  [{d}] retained {ntr + nh} unique samples of {counts.get(d, a.per_domain)} requested",
                      file=sys.stderr)

    rng.shuffle(train)
    rng.shuffle(held)
    chars = sum(len(s) for _, s in train)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(s for _, s in train))
    print(f"== pollard-calib :: Calib 3.0 corpus, profile {profile} ({why})")
    print(f"   chat samples rendered with {renderer.why}")
    for d, src, ntr, nh in stats:
        print(f"   {d:13s} {ntr:5d} train + {nh:4d} held  [{src}]")
    print(f"   TOTAL {len(train)} samples  |  ~{chars/1e6:.2f}M chars  |  ~{chars//4/1e3:.0f}K tokens (est)")
    print(f"   wrote {a.out}")
    if selfgen:
        print(f"   self-generated {selfgen['answers']} answers ({selfgen['kept_reference']} kept the reference)")
    if a.held_out:
        with open(a.held_out, "w", encoding="utf-8") as f:
            f.write("\n".join(s for _, s in held))
        print(f"   wrote {a.held_out} ({len(held)} held-out samples -- overfit check)")
    prov = provenance(profile, why, counts, stats, renderer.why, selfgen)
    prov.update({"path": os.path.abspath(a.out), "held_out": os.path.abspath(a.held_out) if a.held_out else None,
                 "samples": len(train), "chars": chars, "seed": a.seed})
    with open(a.out + ".sources.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1)
    if a.record_model:
        try:
            import pollard_workspace as ws
            ws.record_calibration(a.record_model, a.out, prov)
        except Exception as e:                                   # noqa: BLE001
            print(f"   (could not record the corpus in the MANIFEST: {e})", file=sys.stderr)
    if any(src == "seed" for _, src, _, _ in stats):
        print("   NOTE: some domains fell back to bundled seeds (no `datasets` / offline). Install "
              "`datasets` on the box for full-scale real-data coverage.", file=sys.stderr)


if __name__ == "__main__":
    main()
