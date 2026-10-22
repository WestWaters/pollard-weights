#!/usr/bin/env python3
"""pollard-decision -- gate and score a DECISION model (Jev / OpenJev / d1-style) the way it is used:
one forward pass, typed options lettered [A] [B] [C], the answer read off the FIRST output position.

A decision model never writes prose. Its product is the probability it puts on each option, so the
coherence gate (loops, control tokens, "did it stop") measures nothing about it, and next-token
perplexity on a text corpus measures the body it was fine-tuned away from. What a rung can lose is
exactly what these models are sold on: calibration. So the gate asks a fixed set of typed questions in
the same layout OpenJev's helper uses (State / Question / Options / "Answer with the letter of the best
option only."), reads the letter probabilities at position one through llama-server's logprobs, and
reports, per build: accuracy, how much mass lands on letters at all (a crushed embedding answers with
something else), and -- with --ref -- agreement with f16, the option-distribution KL to f16, and the
mean absolute probability drift. That last trio is the KLD board for a decision model.

    pollard-decision --gguf OpenJev-Q4_K_M.gguf --ref OpenJev-Q8_0.gguf
    pollard-bench --gguf OpenJev-Q4_K_M.gguf --ref OpenJev-Q8_0.gguf --gate   # same thing, from the board

Stdlib + the server client in pollard_bench. Readout is the helper's: letter logits / T, softmax over
the option letters only, missing letters floored at -30.
"""
from __future__ import annotations

import argparse, json, math, os, re, sys

LETTERS = [chr(c) for c in range(65, 91)]
READOUT_T = 1.1          # OpenJev helper default
MISSING = -30.0

#: (state, question, options, answer_index). Plain, unambiguous, multilingual-free: the point is to
#: measure what a rung did to the model's probabilities, not to test its knowledge.
DECISION_SET = [
    ("Customer message: I was charged twice for my order last week and nobody has replied.",
     "Which team should handle this?", ["billing", "shipping", "technical", "sales"], 0),
    ("Customer message: The tracking page says delivered but the parcel never arrived.",
     "Which team should handle this?", ["billing", "shipping", "technical", "sales"], 1),
    ("Customer message: The app crashes every time I open the settings screen.",
     "Which team should handle this?", ["billing", "shipping", "technical", "sales"], 2),
    ("Review: Absolutely loved it, would buy again in a heartbeat.",
     "What is the sentiment?", ["positive", "negative", "neutral"], 0),
    ("Review: Broke after two days and support ignored me.",
     "What is the sentiment?", ["positive", "negative", "neutral"], 1),
    ("Review: It arrived on Tuesday in a brown box.",
     "What is the sentiment?", ["positive", "negative", "neutral"], 2),
    ("Message: URGENT: production database is down, all customers affected.",
     "How urgent is this?", ["can wait", "this week", "today", "right now"], 3),
    ("Message: Minor typo on the about page, fix whenever.",
     "How urgent is this?", ["can wait", "this week", "today", "right now"], 0),
    ("Email subject: You have won a free cruise, click here to claim now!!!",
     "Is this spam?", ["yes", "no"], 0),
    ("Email subject: Agenda for Thursday's planning meeting",
     "Is this spam?", ["yes", "no"], 1),
    ("Text: Je voudrais un cafe, s'il vous plait.",
     "What language is the text?", ["English", "French", "German", "Spanish"], 1),
    ("Text: Ich hatte gern einen Kaffee, bitte.",
     "What language is the text?", ["English", "French", "German", "Spanish"], 2),
    ("Page: a login form with username and password fields and a Sign in button.",
     "What should the agent do to log in?", ["click Sign in", "fill username then password then click Sign in",
                                           "scroll down", "close the tab"], 1),
    ("Page: a cookie banner covers the article; buttons: Accept all, Reject all, Settings.",
     "What should the agent do to read the article?", ["click Settings", "click Reject all", "reload", "type a search"], 1),
    ("Statement: Water boils at 100 degrees Celsius at sea level.",
     "Is the statement true?", ["yes", "no"], 0),
    ("Statement: The Moon is larger than the Earth.",
     "Is the statement true?", ["yes", "no"], 1),
    ("Code: def add(a, b): return a - b",
     "Does the function do what its name says?", ["yes", "no"], 1),
    ("Ticket: User asks how to reset their password.",
     "Which category?", ["how-to", "bug", "billing", "feature request"], 0),
    ("Ticket: Please add dark mode to the dashboard.",
     "Which category?", ["how-to", "bug", "billing", "feature request"], 3),
    ("Headline: Central bank raises interest rates by half a point.",
     "Which section?", ["sports", "economy", "entertainment", "science"], 1),
]


def build_prompt(state: str, question: str, options: list[str]) -> str:
    """OpenJev helper layout, verbatim in spirit: state, question, lettered options, letter-only ask."""
    lines = "\n".join(f"[{LETTERS[i]}] {o}" for i, o in enumerate(options))
    return (f"State:\n{state}\n\nQuestion: {question}\nOptions:\n{lines}\n\n"
            f"Answer with the letter of the best option only.")


def letter_logprobs(top: list[dict]) -> dict[str, float]:
    """{letter: logprob} from a top_logprobs list [{token, logprob}]. The exact letter token wins;
    whitespace / bracket variants (" A", "[A", "A]") fill in only when the bare letter is absent."""
    exact, loose = {}, {}
    for t in top or []:
        tok = str(t.get("token", ""))
        lp = float(t.get("logprob", MISSING))
        if len(tok) == 1 and tok in LETTERS:
            exact[tok] = max(lp, exact.get(tok, MISSING))
        else:
            s = tok.strip().strip("[]().:")
            if len(s) == 1 and s in LETTERS:
                loose[s] = max(lp, loose.get(s, MISSING))
    out = dict(loose); out.update(exact)
    return out


def option_probs(lps: dict[str, float], n: int, T: float = READOUT_T) -> list[float]:
    z = [lps.get(LETTERS[i], MISSING) / T for i in range(n)]
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]


def letter_mass(top: list[dict], n: int) -> float:
    """How much of the model's top-k probability sits on the option letters at all -- a build whose
    embedding lost the letters answers with something else, and no readout can fix that."""
    lps = letter_logprobs(top)
    return sum(math.exp(lps[LETTERS[i]]) for i in range(n) if LETTERS[i] in lps)


def kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(max(pi, 1e-9) / max(qi, 1e-9)) for pi, qi in zip(p, q))


CONFIDENT = 0.95       # "over95": a wrong answer given at >= this confidence


def _argmax_conf(d: dict):
    if not d:
        return None, 0.0
    k = max(d, key=d.get)
    return k, float(d[k])


def calibration(items, n_bins: int = 10) -> dict:
    """Calibration of [(dist {option: p}, gold option)] against the gold labels.

    Agreement and option KL say whether a rung moved; they do not say whether its probabilities still MEAN
    what they say, and a decision model is sold on exactly that ("calibrated probabilities"). A rung can
    agree with f16 on every argmax while pushing 0.7s to 0.97s -- an app that thresholds on confidence
    then acts on answers it should have escalated. So, per build:

      ece     10-bin, top-label: |accuracy - mean confidence| per confidence bin, weighted by bin size
      brier   multi-class: sum over options of (p - onehot)^2, averaged
      nll     -log p(gold), averaged (floored at 1e-12)
      over95  of the WRONG answers, the share given at >= 0.95 confidence (the ones a threshold lets through)
    """
    n = len(items)
    if not n:
        return {}
    bins = [[0, 0.0, 0.0] for _ in range(n_bins)]          # count, sum confidence, sum correct
    brier = nll = 0.0
    wrong = wrong_conf = 0
    for dist, gold in items:
        top, conf = _argmax_conf(dist)
        ok = top is not None and str(top) == str(gold)
        b = bins[min(int(conf * n_bins), n_bins - 1)]
        b[0] += 1; b[1] += conf; b[2] += 1.0 if ok else 0.0
        keys = set(dist) | {str(gold)}
        brier += sum((float(dist.get(k, 0.0)) - (1.0 if k == str(gold) else 0.0)) ** 2 for k in keys)
        nll += -math.log(max(float(dist.get(str(gold), 0.0)), 1e-12))
        if not ok:
            wrong += 1
            wrong_conf += conf >= CONFIDENT
    ece = sum(abs(c_ok - c_conf) for cnt, c_conf, c_ok in bins if cnt) / n
    return {"ece": ece, "brier": brier / n, "nll": nll / n,
            "over95": (wrong_conf / wrong) if wrong else 0.0, "over95_n": wrong_conf, "wrong": wrong}


def bootstrap_ci(flags, b: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap 95% CI for the mean of per-question 0/1 flags (here: rung agrees with the reference).
    Resampling QUESTIONS keeps each rung/reference pair together -- the pairing is the point. Seeded, so a
    board is reproducible."""
    import random
    n = len(flags)
    if not n:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return (means[int(alpha / 2 * b)], means[min(int((1 - alpha / 2) * b), b - 1)])


def _ref_calibration(out: dict, items, ref_items, agree_flags) -> None:
    """With a reference: its calibration, the rung's ECE change, flips (argmax changes) and the agreement CI."""
    rc = calibration(ref_items)
    out.update({"ref_ece": rc.get("ece"), "delta_ece": out["ece"] - rc.get("ece", 0.0),
                "flips": sum(1 for f in agree_flags if not f), "agreement_ci": list(bootstrap_ci(agree_flags))})


def score_rows(rows: list[dict], ref: list[dict] | None = None) -> dict:
    """rows: [{probs, mass, answer}] for the build; ref: same for the f16. Returns the board."""
    n = len(rows)
    acc = sum(1 for r in rows if max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["answer"]) / n
    mass = sum(r["mass"] for r in rows) / n
    out = {"n": n, "accuracy": acc, "letter_mass": mass,
           "p_correct": sum(r["probs"][r["answer"]] for r in rows) / n}
    if ref:
        agree = sum(1 for r, s in zip(rows, ref)
                    if max(range(len(r["probs"])), key=r["probs"].__getitem__)
                    == max(range(len(s["probs"])), key=s["probs"].__getitem__)) / n
        out.update({"ref_accuracy": sum(1 for s in ref if max(range(len(s["probs"])), key=s["probs"].__getitem__) == s["answer"]) / n,
                    "ref_letter_mass": sum(s["mass"] for s in ref) / n,
                    "agreement": agree,
                    "option_kl": sum(kl(s["probs"], r["probs"]) for r, s in zip(rows, ref)) / n,
                    "mean_abs_drift": sum(sum(abs(a - b) for a, b in zip(r["probs"], s["probs"])) / len(r["probs"])
                                          for r, s in zip(rows, ref)) / n})
    as_items = lambda rs: [({str(i): p for i, p in enumerate(r["probs"])}, str(r["answer"])) for r in rs]
    out.update(calibration(as_items(rows)))
    if ref:
        am = lambda p: max(range(len(p)), key=p.__getitem__)
        _ref_calibration(out, as_items(rows), as_items(ref),
                         [1 if am(r["probs"]) == am(s["probs"]) else 0 for r, s in zip(rows, ref)])
    return out


def verdict(board: dict) -> tuple[str, str]:
    """PASS / WEAK / FAIL and why. With a reference the bar is fidelity to f16 (this is a KLD board);
    without one, that the build still answers with letters and gets the easy set mostly right."""
    # Letter mass is judged against the reference: a decision model puts ~all of it on letters and a
    # rung that lost them has lost the format (embedding / output tensor). A general chat model used as
    # a decision model never had it in the first place -- that is not the rung's fault.
    ref_mass = board.get("ref_letter_mass")
    if ref_mass is not None:
        if ref_mass >= 0.2 and board["letter_mass"] < 0.5 * ref_mass:
            return "FAIL", (f"letter mass fell from {ref_mass:.0%} (reference) to {board['letter_mass']:.0%} -- the build "
                            "no longer answers in the question's format (protect the token embedding / output tensor)")
    elif board["letter_mass"] < 0.5:
        return "FAIL", (f"only {board['letter_mass']:.0%} of the mass lands on option letters -- the build no longer "
                        "answers the question's format (protect the token embedding / output tensor)")
    if "agreement" in board:
        if board["agreement"] >= 0.9 and board["option_kl"] <= 0.05:
            return "PASS", (f"agrees with the reference on {board['agreement']:.0%}, option KL {board['option_kl']:.4f}, "
                            f"mean drift {board['mean_abs_drift']:.3f}")
        if board["agreement"] >= 0.8 and board["option_kl"] <= 0.15:
            return "WEAK", (f"agreement {board['agreement']:.0%}, option KL {board['option_kl']:.4f} -- decisions mostly "
                            "hold, probabilities have moved; check the calibration your app relies on")
        return "FAIL", (f"agreement {board['agreement']:.0%}, option KL {board['option_kl']:.4f} -- the rung changed "
                        "what the model decides")
    if board["accuracy"] >= 0.7:
        return "PASS", f"accuracy {board['accuracy']:.0%} on the typed set, {board['letter_mass']:.0%} letter mass"
    return "WEAK", f"accuracy {board['accuracy']:.0%} on the typed set (no reference to compare against)"


def ask(base: str, prompt: str, post) -> dict:
    """One decision through llama-server's chat endpoint: one token, greedy, top-20 logprobs."""
    got = post(base, "/v1/chat/completions",
               {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": 20, "cache_prompt": False, "seed": 0,
                "reasoning_format": "none"})
    ch = (got.get("choices") or [{}])[0]
    content = ((ch.get("logprobs") or {}).get("content") or [{}])[0]
    return {"token": content.get("token", ""), "top": content.get("top_logprobs") or []}


def native(base: str) -> bool:
    """True when the server says the model is a NATIVE decision model (a decision head, e.g. clef):
    /v1/models lists "decisions" in output_modalities. Such a server only answers /v1/systemone --
    clef cannot generate text at all, so the letter readout has nothing to read."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_bench import _get
    try:
        models = _get(base, "/v1/models").get("data") or []
    except Exception:
        return False
    return any(isinstance(m, dict) and "decisions" in ((m.get("architecture") or {}).get("output_modalities") or [])
               for m in models)


def ask_systemone(base: str, state: str, question: str, options: list[str], post) -> list[float]:
    """One typed choice question through /v1/systemone; the option probabilities in option order.
    One question per request: clef decides the questions of a request jointly, and the board compares
    each question on its own."""
    got = post(base, "/v1/systemone",
               {"state": state, "questions": {"q": {"type": "choice", "instructions": question,
                                                    "criteria": {o: None for o in options}}}})
    probs = (((got.get("answers") or {}).get("q") or {}).get("probabilities") or {})
    p = [float(probs.get(o, 0.0)) for o in options]
    s = sum(p)
    return [v / s for v in p] if s > 0 else [1.0 / len(options)] * len(options)


def run(base: str, post, items=None) -> list[dict]:
    rows = []
    head = native(base)
    for state, q, opts, ans in (items or DECISION_SET):
        if head:
            # the head's distribution is over the options by construction: there is no letter mass to lose
            rows.append({"question": q, "options": opts, "answer": ans, "first": "",
                         "probs": ask_systemone(base, state, q, opts, post), "mass": 1.0})
            continue
        r = ask(base, build_prompt(state, q, opts), post)
        rows.append({"question": q, "options": opts, "answer": ans, "first": r["token"],
                     "probs": option_probs(letter_logprobs(r["top"]), len(opts)),
                     "mass": letter_mass(r["top"], len(opts))})
    return rows


TYPED_DATASET = "LocalLLaMA/typed-decisions"
TYPED_ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=LocalLLaMA/typed-decisions"
                  "&config=all&split=test&offset={off}&length=100")


def load_typed(cache=None, fetch=None):
    """The typed-decisions test split (Apache-2.0): 400 states x 5 typed questions = 2000 decisions, each row
    already a /v1/systemone request, gold = a soft distribution per question. Fetched once with the stdlib
    from the Hub's dataset server and cached, with the dataset revision, so every board says what it ran."""
    home = os.environ.get("POLLARD_HOME") or os.path.expanduser("~/pollard")
    cache = cache or os.path.join(home, "cache", "typed-decisions-test.json")
    if os.path.isfile(cache):
        return json.load(open(cache, encoding="utf-8"))
    import urllib.request
    fetch = fetch or (lambda url: json.loads(urllib.request.urlopen(url, timeout=60).read().decode("utf-8")))
    rows, off = [], 0
    while True:
        page = fetch(TYPED_ROWS_URL.format(off=off))
        for r in page.get("rows", []):
            r = r["row"]
            parse = lambda v: json.loads(v) if isinstance(v, str) else v
            rows.append({"id": r["id"], "workflow": r["workflow"], "state": r["state"],
                         "questions": parse(r["questions"]), "gold": parse(r["gold"])})
        off += 100
        if off >= int(page.get("num_rows_total") or 0) or not page.get("rows"):
            break
    try:
        rev = fetch(f"https://huggingface.co/api/datasets/{TYPED_DATASET}").get("sha", "")
    except Exception:
        rev = ""
    data = {"dataset": TYPED_DATASET, "split": "test", "revision": rev, "rows": rows}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(data, open(cache, "w", encoding="utf-8"))
    return data


def _answer_dist(q, ans):
    """A /v1/systemone answer as {option: p}, keyed the way typed-decisions' gold is: choice by option name,
    noul as true/false, score by level index as a string."""
    t = q.get("type")
    if t == "noul":
        p = float((ans or {}).get("noul", 0.5))
        return {"true": p, "false": 1.0 - p}
    probs = (ans or {}).get("probabilities") or {}
    if t == "score":
        return {str(int(float(k))): float(v) for k, v in probs.items()}
    return {str(k): float(v) for k, v in probs.items()}


def _dist_kl(p, q):
    keys = set(p) | set(q)
    return sum(p.get(k, 0.0) * math.log(max(p.get(k, 0.0), 1e-9) / max(q.get(k, 0.0), 1e-9)) for k in keys)


def run_typed(base, post, rows):
    """Every row as one /v1/systemone request (its five questions answered together, as clef decides them)."""
    out = []
    for r in rows:
        got = post(base, "/v1/systemone", {"state": r["state"], "questions": r["questions"]})
        answers = got.get("answers") or {}
        for qid, q in r["questions"].items():
            out.append({"row": r["id"], "workflow": r["workflow"], "qid": qid, "type": q.get("type"),
                        "dist": _answer_dist(q, answers.get(qid)), "gold": r["gold"].get(qid) or {}})
    return out


def score_typed(rows, ref=None):
    """The decision board on typed-decisions, in score_rows' shape so verdict() and the card read it unchanged:
    accuracy / KL against the gold distribution, and -- with a reference -- agreement, option KL and drift."""
    argmax = lambda d: max(d, key=d.get) if d else None
    n = len(rows)
    out = {"n": n, "letter_mass": 1.0, "set": "typed-decisions",
           "accuracy": sum(1 for r in rows if argmax(r["dist"]) == str(r["gold"].get("label"))) / n,
           "kl_from_gold": sum(_dist_kl(r["gold"].get("probabilities") or {}, r["dist"]) for r in rows) / n,
           "p_correct": sum(r["dist"].get(str(r["gold"].get("label")), 0.0) for r in rows) / n}
    if ref:
        keys = lambda a, b: sorted(set(a) | set(b))
        out.update({
            "ref_letter_mass": 1.0,
            "ref_accuracy": sum(1 for s in ref if argmax(s["dist"]) == str(s["gold"].get("label"))) / n,
            "ref_kl_from_gold": sum(_dist_kl(s["gold"].get("probabilities") or {}, s["dist"]) for s in ref) / n,
            "agreement": sum(1 for r, s in zip(rows, ref) if argmax(r["dist"]) == argmax(s["dist"])) / n,
            "option_kl": sum(_dist_kl(s["dist"], r["dist"]) for r, s in zip(rows, ref)) / n,
            "mean_abs_drift": sum(sum(abs(r["dist"].get(k, 0.0) - s["dist"].get(k, 0.0)) for k in keys(r["dist"], s["dist"]))
                                  / max(len(keys(r["dist"], s["dist"])), 1) for r, s in zip(rows, ref)) / n})
    # calibration against the gold LABEL (typed-decisions' gold is soft; its label is the argmax of it)
    items = lambda rs: [(r["dist"], str(r["gold"].get("label"))) for r in rs]
    out.update(calibration(items(rows)))
    if ref:
        _ref_calibration(out, items(rows), items(ref),
                         [1 if argmax(r["dist"]) == argmax(s["dist"]) else 0 for r, s in zip(rows, ref)])
    return out


def _typed_rows_out(rows, ref):
    """Per workflow x type summary rows for the printed table (2000 single rows would be noise)."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["workflow"], r["type"]), []).append(i)
    argmax = lambda d: max(d, key=d.get) if d else None
    out = []
    for (wf, t), idx in sorted(groups.items()):
        if ref:
            ag = sum(1 for i in idx if argmax(rows[i]["dist"]) == argmax(ref[i]["dist"])) / len(idx)
            kv = sum(_dist_kl(ref[i]["dist"], rows[i]["dist"]) for i in idx) / len(idx)
            out.append({"prompt": f"{wf}/{t}"[:48], "loop": ag < 0.9, "reason": f"agree {ag:.0%}",
                        "sample": f"n={len(idx)}  option KL {kv:.4f}"})
        else:
            acc = sum(1 for i in idx if argmax(rows[i]["dist"]) == str(rows[i]["gold"].get("label"))) / len(idx)
            out.append({"prompt": f"{wf}/{t}"[:48], "loop": False, "reason": f"gold acc {acc:.0%}",
                        "sample": f"n={len(idx)}"})
    return out


_PPLX = re.compile(r"pplx|decider|(^|[^a-z0-9])d3([^a-z0-9]|$)", re.I)


def _source_config(model: str, source: str | None) -> dict | None:
    """The source checkpoint's config.json: --source (a dir or the file), else one beside the GGUF."""
    for cand in ([source] if source else []) + [os.path.dirname(os.path.abspath(model))]:
        if not cand:
            continue
        p = os.path.join(cand, "config.json") if os.path.isdir(cand) else cand
        if os.path.isfile(p):
            try:
                return json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def pregate_checks(meta: dict, cfg: dict | None = None) -> list[str]:
    """Things the GGUF's own metadata says will make the board measure the wrong model. Warnings, not
    failures: each is a converter/runtime gap, and the gate still runs so the damage is measured.

      * clef-type (a decision head): Unsloth's FastDecisionModel exports the calibrated per-type temperatures
        as `<arch>.decision.temperature.<type>`. Upstream's converter drops them, the head then answers at
        T=1, and every probability is off while the argmax -- and so agreement -- can look fine.
      * pplx-decider / d3-style: the source runs noncausal attention (config `attention_mode`). Upstream
        llama.cpp runs pplx-decider v1.1 causally unless the GGUF says `<arch>.attention.causal = false`;
        ~3 of 25 decisions flip.
    """
    warns = []
    arch = str(meta.get("general.architecture", "") or "")
    names = " ".join(str(meta.get(k, "")) for k in ("general.name", "general.basename", "general.architecture"))
    head = any(".decision." in str(k) for k in meta)
    if head or "clef" in names.lower():
        temps = [k for k in meta if str(k).startswith(f"{arch}.decision.temperature.")]
        if not temps:
            warns.append(f"no `{arch}.decision.temperature.*` keys: the calibrated per-type temperatures were "
                         "dropped in conversion (upstream's converter does; Unsloth FastDecisionModel exports them), "
                         "so the head answers at T=1 and its probabilities are uncalibrated")
    if _PPLX.search(names):
        mode = str((cfg or {}).get("attention_mode", "")).lower().replace("-", "_")
        noncausal = "noncausal" in mode or "non_causal" in mode or "bidirectional" in mode
        causal_key = meta.get(f"{arch}.attention.causal",
                              next((v for k, v in meta.items() if str(k).endswith("causal_attn")), None))
        if cfg is None:
            warns.append("pplx-decider / d3-style model and no source config.json to read `attention_mode` from "
                         "(pass --source): cannot tell whether llama.cpp will run it with the right attention")
        elif noncausal and causal_key is not False:
            warns.append(f"source attention_mode is `{cfg.get('attention_mode')}` but the GGUF has no "
                         f"`{arch}.attention.causal = false`: llama.cpp will run it causally (~3/25 decisions "
                         "flip on pplx-decider v1.1)")
    return warns


#: architectures whose graph ends in a decision head: the output is the head's option scores, never per-token LM
#: logits (src/models/clef-omni.cpp sets only t_embd -> the head). llama-perplexity has nothing to read on these --
#: on clef-omni llama-imatrix's ppl pass died "get_logits_ith: invalid logits id 0, reason: no logits" (2026-10-17).
DECISION_HEAD_ARCHS = ("clef", "clef-omni")
NOT_APPLICABLE = "not applicable: decision-only model"


def decision_only_meta(meta: dict) -> str | None:
    """Why a GGUF (its metadata) is a DECISION-ONLY model -- no per-token logits, so KLD / PPL cannot exist and its
    fidelity is the decision board -- or None. A letter-readout decision model (Jev / OpenJev: a stock body read at
    position one) is NOT decision-only: it has logits and is measured on them as well."""
    meta = meta or {}
    arch = str(meta.get("general.architecture", "") or "")
    if arch in DECISION_HEAD_ARCHS:
        return f"`{arch}` ends in a decision head (no per-token logits)"
    if any(".decision." in str(k) for k in meta):
        return "the GGUF carries a decision head (`.decision.*` keys; no per-token logits)"
    if any(str(t).startswith("dec.blk.") for t in (meta.get("_tensor_params") or {})):
        return "the GGUF carries a decision-head stack (`dec.blk.N`; no per-token logits)"
    try:
        import pollard_d3 as D3
        if D3.gguf_is_d3(meta):
            return f"{D3.FORMAT_ID}: a non-causal body read by a separate readout (no next-token logits)"
    except ImportError:
        pass
    return None


def decision_only(model: str) -> str | None:
    """decision_only_meta for a GGUF file; for an HF folder: clef's joint_head_config.json, or a d3 package
    (decision_config.json). None for everything else, and for anything unreadable."""
    model = str(model or "")
    if os.path.isdir(model):
        if os.path.isfile(os.path.join(model, "joint_head_config.json")):
            return "joint_head_config.json: a decision head on the body (no per-token logits)"
        try:
            import pollard_d3 as D3
            if D3.decision_config(model) is not None:
                return f"{D3.FORMAT_ID}: a non-causal body read by a separate readout (no next-token logits)"
        except ImportError:
            pass
        return None
    if not model.endswith(".gguf") or not os.path.isfile(model):
        return None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pollard_calc import read_gguf_meta
        return decision_only_meta(read_gguf_meta(model))
    except (Exception, SystemExit):                                # noqa: BLE001 -- unreadable: not decided here
        return None


def resolve_ngl(gguf: str, ngl) -> str:
    """`--ngl auto` -> the layers that fit THIS file in free VRAM (pollard_llamaserve, the release's own logic); a
    number as given. The build and the reference are served one after the other, so each gets its own fit."""
    try:
        import pollard_llamaserve as LS
        return str(LS.resolve_ngl(gguf, ngl))
    except Exception:                                              # noqa: BLE001 -- never crash on the GPU probe
        return "0" if str(ngl).strip().lower() == "auto" else str(ngl)


def _ngl_arg(v: str) -> str:
    s = str(v).strip().lower()
    if s == "auto" or s.lstrip("-").isdigit():
        return s
    raise argparse.ArgumentTypeError(f"--ngl takes a number of GPU layers or `auto`, not {v!r}")


def pregate(model: str, source: str | None = None) -> list[str]:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pollard_calc import read_gguf_meta
        meta = read_gguf_meta(model)
    except Exception:
        return []
    return pregate_checks(meta, _source_config(model, source))


def gate(model: str, ngl, ctx: int = 4096, ref: str | None = None, qset: str = "builtin",
         limit: int = 0, source: str | None = None) -> dict:
    """Pre-gate metadata checks, then the board (see _gate). Warnings ride along in the result."""
    warns = pregate(model, source)
    res = _gate(model, ngl, ctx, ref, qset, limit)
    if warns:
        res["warnings"] = warns
    return res


def _gate(model: str, ngl, ctx: int = 4096, ref: str | None = None, qset: str = "builtin",
          limit: int = 0) -> dict:
    """Serve the build (and the reference), run the set on both, return the board + verdict in the
    same shape pollard-bench prints ("verdict", "rows", "reason").

    qset "typed-decisions" replays the 2000-decision benchmark through /v1/systemone (native decision
    models only). The 20 built-in questions are easy enough that every clef rung scored 100%; on 2000
    decisions agreement and option KL separate the rungs.

    `ngl` may be `auto`: each file is fitted to free VRAM when it is served (resolve_ngl)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_bench import _served, _post

    def fit(path):
        return resolve_ngl(path, ngl)

    if qset == "typed-decisions":
        data = load_typed()
        if limit:                                       # a smoke run; a board is only comparable at the full 400
            data = dict(data, rows=data["rows"][:limit])
        # clef evaluates a whole prompt in one batch: let it be as large as the context
        extra = ("-b", str(ctx), "-ub", str(ctx))
        with _served(model, fit(model), ctx, extra=extra) as base:
            if base is None or not native(base):
                return {"verdict": "NO_SERVER" if base is None else "UNSUPPORTED", "config": None,
                        "rows": [{"prompt": "(not run)", "loop": None, "sample": "",
                                  "reason": "llama-server would not start" if base is None
                                  else "typed-decisions needs a native decision model (/v1/systemone)"}]}
            rows = run_typed(base, _post, data["rows"])
        ref_rows = None
        if ref:
            with _served(ref, fit(ref), ctx, extra=extra) as rbase:
                if rbase is not None:
                    ref_rows = run_typed(rbase, _post, data["rows"])
        board = score_typed(rows, ref_rows)
        board["set"] = f"{data['dataset']}@{(data.get('revision') or '')[:8]}" + (f" (first {limit} rows)" if limit else "")
        v, why = verdict(board)
        return {"verdict": v, "config": "decision", "reason": why, "board": board,
                "rows": _typed_rows_out(rows, ref_rows), "sampling": ["--temp", "0"]}
    with _served(model, fit(model), ctx) as base:
        if base is None:
            return {"verdict": "NO_SERVER", "config": None, "rows": [{"prompt": "(not run)", "loop": None,
                    "reason": "llama-server would not start", "sample": ""}]}
        rows = run(base, _post)
    ref_rows = None
    if ref:
        with _served(ref, fit(ref), ctx) as rbase:
            if rbase is not None:
                ref_rows = run(rbase, _post)
    board = score_rows(rows, ref_rows)
    v, why = verdict(board)
    out_rows = []
    for i, r in enumerate(rows):
        pred = max(range(len(r["probs"])), key=r["probs"].__getitem__)
        ok = pred == r["answer"]
        note = f"p={r['probs'][pred]:.2f} -> [{LETTERS[pred]}] {r['options'][pred]}"
        if ref_rows:
            rp = max(range(len(ref_rows[i]["probs"])), key=ref_rows[i]["probs"].__getitem__)
            note += f"  f16: [{LETTERS[rp]}] p={ref_rows[i]['probs'][rp]:.2f}  kl={kl(ref_rows[i]['probs'], r['probs']):.3f}"
        out_rows.append({"prompt": r["question"][:48], "loop": (not ok) if not ref_rows else (pred != rp),
                         "reason": (("agrees" if pred == rp else "DISAGREES with reference") if ref_rows
                                    else ("correct" if ok else "WRONG")) + ("" if ok else " (wrong answer)"),
                         "sample": note})
    return {"verdict": v, "config": "decision", "reason": why, "board": board, "rows": out_rows,
            "sampling": ["--temp", "0"]}


def board_calibration_line(b: dict) -> str:
    """The calibration half of the board, one line."""
    if "ece" not in b:
        return ""
    line = (f"calibration: ECE {b['ece']:.4f}  Brier {b['brier']:.4f}  NLL {b['nll']:.4f}  "
            f"over95 {b['over95']:.0%} of {b['wrong']} wrong")
    if "flips" in b:
        lo, hi = b["agreement_ci"]
        line += (f"  | vs ref: dECE {b['delta_ece']:+.4f}  flips {b['flips']}/{b['n']}  "
                 f"agreement 95% CI [{lo:.1%}, {hi:.1%}]")
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", help="reference GGUF (f16 / Q8_0) for agreement + option KL")
    ap.add_argument("--ngl", type=_ngl_arg, default="0",
                    help="GPU layers for llama-server, or `auto` = what fits free VRAM, per file (default 0)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", dest="qset", default="builtin", choices=["builtin", "typed-decisions"],
                    help="question set: the 20 built-in questions, or the typed-decisions benchmark (2000 "
                         "decisions, native decision models; fetched once from the Hub and cached)")
    ap.add_argument("--limit", type=int, default=0, help="typed-decisions: only the first N states (smoke run)")
    ap.add_argument("--source", help="the source checkpoint (HF dir or its config.json), for the pre-gate checks "
                    "(attention mode); default: a config.json beside the GGUF")
    ap.add_argument("--results", help="merge this rung's decision board into this results.json (pollard-card "
                                      "--results: the Decision fidelity table and the retention table)")
    ap.add_argument("--rung", help="the rung's name/tag in --results (default: the GGUF file name)")
    a = ap.parse_args()
    from pollard_calc import set_model_arch
    set_model_arch(a.gguf)                 # llama-server from the engine that carries this architecture's patches
    res = gate(a.gguf, a.ngl, a.ctx, a.ref, a.qset, a.limit, a.source)
    if a.results and res.get("board"):
        import pollard_workspace as ws
        board = {k: v for k, v in res["board"].items() if not isinstance(v, (list, dict))}
        board.update({"verdict": res["verdict"], "set": board.get("set") or a.qset})
        ws.merge_results(a.results, a.rung or os.path.basename(a.gguf), {"decision": board})
    if a.json:
        print(json.dumps(res, indent=1)); return
    b = res.get("board", {})
    for r in res["rows"]:
        print(f"  [{'ok  ' if not r['loop'] else 'BAD '}] {r['prompt']:<48} {r['reason']:<28} {r['sample']}")
    if b:
        line = f"accuracy {b['accuracy']:.0%}  letter mass {b['letter_mass']:.0%}  p(correct) {b['p_correct']:.2f}"
        if "agreement" in b:
            line += (f"  | vs ref: agreement {b['agreement']:.0%}  option KL {b['option_kl']:.4f}  "
                     f"drift {b['mean_abs_drift']:.3f}  (ref accuracy {b['ref_accuracy']:.0%})")
        print("  " + line)
        cal = board_calibration_line(b)
        if cal:
            print("  " + cal)
    for w in res.get("warnings") or []:
        print(f"  WARNING: {w}")
    print(f"VERDICT: {res['verdict']} -- {res.get('reason', '')}")
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
