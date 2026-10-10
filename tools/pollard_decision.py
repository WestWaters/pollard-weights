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

import argparse, json, math, os, sys

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


def gate(model: str, ngl: int, ctx: int = 4096, ref: str | None = None, qset: str = "builtin",
         limit: int = 0) -> dict:
    """Serve the build (and the reference), run the set on both, return the board + verdict in the
    same shape pollard-bench prints ("verdict", "rows", "reason").

    qset "typed-decisions" replays the 2000-decision benchmark through /v1/systemone (native decision
    models only). The 20 built-in questions are easy enough that every clef rung scored 100%; on 2000
    decisions agreement and option KL separate the rungs."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_bench import _served, _post
    if qset == "typed-decisions":
        data = load_typed()
        if limit:                                       # a smoke run; a board is only comparable at the full 400
            data = dict(data, rows=data["rows"][:limit])
        # clef evaluates a whole prompt in one batch: let it be as large as the context
        extra = ("-b", str(ctx), "-ub", str(ctx))
        with _served(model, ngl, ctx, extra=extra) as base:
            if base is None or not native(base):
                return {"verdict": "NO_SERVER" if base is None else "UNSUPPORTED", "config": None,
                        "rows": [{"prompt": "(not run)", "loop": None, "sample": "",
                                  "reason": "llama-server would not start" if base is None
                                  else "typed-decisions needs a native decision model (/v1/systemone)"}]}
            rows = run_typed(base, _post, data["rows"])
        ref_rows = None
        if ref:
            with _served(ref, ngl, ctx, extra=extra) as rbase:
                if rbase is not None:
                    ref_rows = run_typed(rbase, _post, data["rows"])
        board = score_typed(rows, ref_rows)
        board["set"] = f"{data['dataset']}@{(data.get('revision') or '')[:8]}" + (f" (first {limit} rows)" if limit else "")
        v, why = verdict(board)
        return {"verdict": v, "config": "decision", "reason": why, "board": board,
                "rows": _typed_rows_out(rows, ref_rows), "sampling": ["--temp", "0"]}
    with _served(model, ngl, ctx) as base:
        if base is None:
            return {"verdict": "NO_SERVER", "config": None, "rows": [{"prompt": "(not run)", "loop": None,
                    "reason": "llama-server would not start", "sample": ""}]}
        rows = run(base, _post)
    ref_rows = None
    if ref:
        with _served(ref, ngl, ctx) as rbase:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", help="reference GGUF (f16 / Q8_0) for agreement + option KL")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", dest="qset", default="builtin", choices=["builtin", "typed-decisions"],
                    help="question set: the 20 built-in questions, or the typed-decisions benchmark (2000 "
                         "decisions, native decision models; fetched once from the Hub and cached)")
    ap.add_argument("--limit", type=int, default=0, help="typed-decisions: only the first N states (smoke run)")
    a = ap.parse_args()
    res = gate(a.gguf, a.ngl, a.ctx, a.ref, a.qset, a.limit)
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
    print(f"VERDICT: {res['verdict']} -- {res.get('reason', '')}")
    sys.exit(0 if res["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
