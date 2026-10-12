#!/usr/bin/env python3
"""pollard-tune -- fine-tune a model on YOUR data, prove it beat the base (and a frontier model), then ship it.

The lane Coinbase described in its "Owning Intelligence" series (2026-10): a 9B open model post-trained on
their own labelled fraud outcomes beat a frontier "decision agent" on precision, recall, F1 and -- the one that
mattered -- dollar-weighted recall, at less than half the latency. What made it work was not the trainer: it
was a held-out benchmark built from matured labels, a fixed decision policy so model changes were the only
variable, metrics weighted by what a miss costs, and versioning weights + prompt + policy as one unit.
This tool makes those the defaults:

    pollard tune --base Qwen/Qwen3.5-9B --data cases.csv --task classify --label-field fraud \\
                 --weight-field amount --instructions "You review card payments for fraud." --out runs/fraud
    pollard tune --base Qwen/Qwen3.5-9B --data cases.csv ... --out runs/fraud \\
                 --compare-url http://127.0.0.1:8081 --compare-model big-local-model      # + a third column
    pollard tune --base ... --data ... --out runs/fraud --ship --harness omp     # + GGUF ladder, card, serve line

Steps (each also runs alone: `pollard-tune prep|train|eval|ship --out RUN`):

prep   -- (a) chat SFT jsonl {messages}, (b) classifier jsonl/csv (input -> label [+ reason, weight, time]),
          (c) typed-decisions rows {state, questions, gold}. Validates, dedupes, splits train/valid/test with a
          fixed seed (stratified by label; by time with --time-field), reports class balance, and REFUSES to
          train when a test prompt also appears in train (--test FILE you supplied, or a bad merge).
train  -- LoRA / QLoRA SFT. Backends: mlx (mlx-lm, Apple Silicon), trl (TRL + PEFT, CUDA/CPU), unsloth (CUDA;
          for typed-decisions it trains Unsloth's FastDecisionModel clef head). Checkpoints + --resume, loss log,
          then the adapter is merged into a full HF checkpoint with forge.json lineage.
eval   -- base vs tuned (vs --compare-url) on the held-out test split. classify: per-class P/R/F1, confusion,
          weighted recall (--weight-field: Coinbase's dollar-weighted recall), latency P50/P99. chat: exact /
          contains (+ --judge-url). decision: accuracy, KL from the gold distribution, ECE.
ship   -- pollard-smoke preflight, `pollard --hf <merged> --run --ladder`, the card re-rendered with the eval
          table (--extra-md), then the `pollard serve ... --harness` line (or llama-server where serve is absent).

Everything runs on this machine. The only time a row leaves it is --compare-url / --judge-url pointing off-box,
and that needs --allow-remote.
"""
from __future__ import annotations

import argparse, csv, glob, hashlib, json, math, os, random, re, shutil, subprocess, sys, time
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASKS = ("chat", "classify", "decision")
BACKENDS = ("mlx", "trl", "unsloth")
DEFAULT_INSTRUCTIONS = "You review one case and classify it."
DECISION_SYSTEM = ("You make one typed decision about a state. Answer with exactly one option name and nothing else.")
INPUT_KEYS = ("input", "text", "prompt", "case", "transaction", "content", "features", "question")
LABEL_KEYS = ("label", "output", "answer", "target", "class", "decision", "completion")
LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


# ===================================================================================== data
def _norm_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def read_rows(path):
    """Rows from .jsonl / .json (a list, or {"rows": [...]}) / .csv. A bad line is an error with its number:
    silently skipping it would train on a different set than the one the user counted."""
    if not os.path.isfile(path):
        sys.exit(f"--data {path}: no such file")
    low = path.lower()
    if low.endswith(".csv") or low.endswith(".tsv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f, delimiter="\t" if low.endswith(".tsv") else ",")]
    if low.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        return d["rows"] if isinstance(d, dict) and "rows" in d else list(d)
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            sys.exit(f"{path}:{i}: not JSON ({e})")
    return rows


def detect_task(rows):
    r = rows[0] if rows else {}
    if isinstance(r.get("messages"), list):
        return "chat"
    if "questions" in r and "state" in r:
        return "decision"
    return "classify"


def _pick_key(row, keys, given, what):
    if given:
        if given not in row:
            sys.exit(f"--{what}-field {given!r} is not a column (have: {', '.join(row)})")
        return given
    for k in keys:
        if k in row:
            return k
    return None


def _render_input(v):
    """A structured case (a dict of transaction features) is rendered as sorted-key JSON, so the same case
    always becomes the same prompt -- dedupe and the leak check compare prompts, not dict orderings."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)


def _maybe_json(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def classify_system(instructions, labels, with_reason):
    tail = (" On the first line write exactly one label from: " + ", ".join(labels)
            + (". On the second line give a one-sentence reason." if with_reason else ". Write nothing else."))
    return (instructions or DEFAULT_INSTRUCTIONS).rstrip() + tail


def normalize_classify(rows, a):
    """-> examples [{key, prompt_key, messages, label, weight, time, reason}] and the label set."""
    first = rows[0]
    lk = _pick_key(first, LABEL_KEYS, a.label_field, "label")
    if not lk:
        sys.exit("classify: no label column (looked for " + ", ".join(LABEL_KEYS) + "); pass --label-field")
    ik = _pick_key(first, INPUT_KEYS, a.input_field, "input")
    # the weight (an amount) stays IN the prompt: it is the cost of a miss AND a feature a reviewer would read.
    # Excluding it once collapsed 900 synthetic cases to 281 "duplicates" that differed only by amount.
    side = {lk, a.reason_field, a.time_field, a.id_field}
    out, bad = [], 0
    for r in rows:
        lab = r.get(lk)
        if lab is None or str(lab).strip() == "":
            bad += 1
            continue
        if ik:
            text = _render_input(_maybe_json(r.get(ik)))
        else:                                       # a tabular case: every non-side column, as key: value lines
            text = "\n".join(f"{k}: {v}" for k, v in r.items() if k not in side and k is not None)
        if not text.strip():
            bad += 1
            continue
        w = None
        if a.weight_field:
            try:
                w = float(r.get(a.weight_field) or 0)
            except (TypeError, ValueError):
                sys.exit(f"--weight-field {a.weight_field}: {r.get(a.weight_field)!r} is not a number")
        out.append({"input": text, "label": str(lab).strip(), "reason": (str(r.get(a.reason_field) or "").strip()
                    if a.reason_field else ""), "weight": w, "time": r.get(a.time_field) if a.time_field else None,
                    "id": r.get(a.id_field) if a.id_field else None})
    labels = sorted({e["label"] for e in out})
    if len(labels) < 2:
        sys.exit(f"classify: need at least 2 labels, found {labels}")
    sysmsg = classify_system(a.instructions, labels, bool(a.reason_field))
    for e in out:
        ans = e["label"] + (("\n" + e["reason"]) if e["reason"] else "")
        e["messages"] = [{"role": "system", "content": sysmsg}, {"role": "user", "content": e["input"]},
                         {"role": "assistant", "content": ans}]
        e["prompt_key"] = _h(_norm_text(e["input"]))
        e["key"] = _h(_norm_text(e["input"]) + "\x00" + e["label"])
    return out, labels, sysmsg, bad


def normalize_chat(rows, a):
    out, bad = [], 0
    for r in rows:
        m = r.get("messages")
        if not (isinstance(m, list) and m and all(isinstance(x, dict) and "role" in x and "content" in x for x in m)):
            bad += 1
            continue
        if m[-1]["role"] != "assistant":
            bad += 1                                 # nothing to learn from, and nothing to score against
            continue
        prompt = json.dumps([[x["role"], _norm_text(x["content"])] for x in m[:-1]])
        out.append({"messages": m, "label": None, "weight": None, "time": r.get(a.time_field) if a.time_field else None,
                    "prompt_key": _h(prompt), "key": _h(prompt + "\x00" + _norm_text(m[-1]["content"]))})
    return out, [], None, bad


def decision_options(q):
    t = q.get("type")
    crit = q.get("criteria")
    if t == "noul":
        return ["true", "false"]
    if t == "score":
        n = len(crit) if isinstance(crit, list) else len(crit or {})
        return [str(i) for i in range(n)]
    return list(crit.keys()) if isinstance(crit, dict) else [str(c) for c in (crit or [])]


def decision_prompt(state, q):
    """One typed question as a chat turn: the state, the question, and its options with their criteria."""
    crit = q.get("criteria")
    opts = decision_options(q)
    if isinstance(crit, dict):
        desc = [crit.get(o, "") for o in opts]
    elif isinstance(crit, list):
        desc = [str(c) for c in crit]
    else:
        desc = [""] * len(opts)
    st = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    lines = "\n".join(f"- {o}: {d}" if d else f"- {o}" for o, d in zip(opts, desc))
    return f"State:\n{st}\n\nQuestion ({q.get('type')}): {q.get('instructions', '')}\nOptions:\n{lines}"


def normalize_decision(rows, a):
    """A typed-decisions row is the unit (dedupe, split and leak check by STATE), expanded to one example per
    question for the generation path. Each example keeps the gold distribution for KL / ECE."""
    out, bad = [], 0
    for r in rows:
        qs, gold = _maybe_json(r.get("questions")), _maybe_json(r.get("gold"))
        state = r.get("state")
        if not isinstance(qs, dict) or not isinstance(gold, dict) or state is None:
            bad += 1
            continue
        skey = _h(_norm_text(state if isinstance(state, str) else json.dumps(state, sort_keys=True)))
        for qid, q in qs.items():
            g = gold.get(qid) or {}
            if "label" not in g:
                continue
            opts = decision_options(q)
            out.append({"messages": [{"role": "system", "content": DECISION_SYSTEM},
                                     {"role": "user", "content": decision_prompt(state, q)},
                                     {"role": "assistant", "content": str(g["label"])}],
                        "label": str(g["label"]), "options": opts, "qtype": q.get("type"), "qid": qid,
                        "gold": {str(k): float(v) for k, v in (g.get("probabilities") or {}).items()},
                        "workflow": r.get("workflow"), "row": r.get("id"), "weight": None,
                        "time": r.get(a.time_field) if a.time_field else None,
                        "prompt_key": skey, "key": _h(skey + "\x00" + qid), "_raw": r})
    return out, [], DECISION_SYSTEM, bad


def dedupe(ex):
    """Exact duplicates go; the SAME prompt with DIFFERENT answers is a conflict -- kept once (first seen) and
    counted, because a model cannot learn both and the test split must not hold the other copy."""
    seen_key, seen_prompt, out = set(), {}, []
    dup = conflict = 0
    for e in ex:
        if e["key"] in seen_key:
            dup += 1
            continue
        if e["prompt_key"] in seen_prompt and e.get("qid") is None:
            conflict += 1
            continue
        seen_key.add(e["key"])
        seen_prompt.setdefault(e["prompt_key"], e)
        out.append(e)
    return out, dup, conflict


def split(ex, seed=0, valid=0.1, test=0.1, stratify=True, by_time=False):
    """Group by prompt (a decision state's five questions travel together), then: oldest -> train and newest ->
    test with --time-field (Coinbase replayed a past window; a random split lets a model see the future), else
    a seeded shuffle, stratified by label so a 5% class still lands in every split."""
    groups = OrderedDict()
    for e in ex:
        groups.setdefault(e["prompt_key"], []).append(e)
    keys = list(groups)
    rng = random.Random(seed)

    def cut(ks):
        n = len(ks)
        nt = int(round(n * test)) if n >= 3 else 0
        nv = int(round(n * valid)) if n >= 3 else 0
        if test > 0 and nt == 0 and n >= 3:
            nt = 1
        if valid > 0 and nv == 0 and n >= 4:
            nv = 1
        return ks[:n - nt - nv], ks[n - nt - nv:n - nt], ks[n - nt:]

    if by_time:
        keys.sort(key=lambda k: str(groups[k][0].get("time")))
        tr, va, te = cut(keys)
    else:
        if stratify and all(groups[k][0].get("label") is not None for k in keys) and not groups[keys[0]][0].get("qid"):
            by = OrderedDict()
            for k in keys:
                by.setdefault(groups[k][0]["label"], []).append(k)
            tr, va, te = [], [], []
            for lab in sorted(by):
                ks = by[lab][:]
                rng.shuffle(ks)
                a_, b_, c_ = cut(ks)
                tr += a_; va += b_; te += c_
            rng.shuffle(tr)
        else:
            rng.shuffle(keys)
            tr, va, te = cut(keys)
    flat = lambda ks: [e for k in ks for e in groups[k]]
    return flat(tr), flat(va), flat(te)


def leaks(train, test):
    tk = {e["prompt_key"] for e in train}
    return [e for e in test if e["prompt_key"] in tk]


def balance_counts(ex):
    return dict(Counter(e["label"] for e in ex if e.get("label") is not None))


def oversample(train, seed=0, ratio=0.5, cap=10):
    """Repeat minority-class rows (train split only) up to `ratio` x the majority count, at most `cap` copies each.
    Coinbase balanced its RL reward across fraud and legitimate rows for the same reason: at a 5% base rate
    the cheapest way to look accurate is to never flag anything."""
    c = balance_counts(train)
    if len(c) < 2:
        return train, {}
    top = max(c.values())
    rng = random.Random(seed)
    out, added = list(train), {}
    for lab, n in c.items():
        want = min(int(top * ratio), n * cap) - n
        if want <= 0:
            continue
        pool = [e for e in train if e["label"] == lab]
        extra = [pool[i % len(pool)] for i in range(want)]
        out += extra
        added[lab] = want
    rng.shuffle(out)
    return out, added


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _public(e):
    return {k: v for k, v in e.items() if k not in ("_raw", "input", "reason")}


def prep(a):
    """Data -> RUN/data/{train,valid}.jsonl (pure {messages}: what every backend reads), heldout.jsonl (the test
    split with labels, weights and gold for eval), prep.json (counts, balance, hashes) and RUN/agent.json."""
    rows = read_rows(a.data)
    if not rows:
        sys.exit(f"--data {a.data}: no rows")
    task = a.task if a.task and a.task != "auto" else detect_task(rows)
    norm = {"classify": normalize_classify, "chat": normalize_chat, "decision": normalize_decision}[task]
    ex, labels, sysmsg, bad = norm(rows, a)
    ex, dup, conflict = dedupe(ex)
    if len(ex) < 10:
        sys.exit(f"only {len(ex)} usable examples after validation/dedupe -- too few to split and evaluate")
    given_test = None
    if a.test:
        trows = read_rows(a.test)
        given_test, _, _, _ = norm(trows, a)
        tr, va, _ = split(ex, a.seed, a.valid_frac, 0.0, task == "classify", bool(a.time_field))
        te = given_test
    else:
        tr, va, te = split(ex, a.seed, a.valid_frac, a.test_frac, task == "classify", bool(a.time_field))
    leaked = leaks(tr + va, te)
    if leaked and not a.allow_leak:
        sys.exit(f"REFUSING to train: {len(leaked)} of {len(te)} test prompts also appear in train/valid "
                 f"(first: {leaked[0]['prompt_key']}). A leaked test set measures memory, not the model. "
                 "Remove them from --data, or --allow-leak to proceed knowingly.")
    if not te:
        sys.exit("the test split is empty -- add data or raise --test-frac")
    added = {}
    if task == "classify":
        c = balance_counts(tr)
        ratio = min(c.values()) / max(c.values()) if c else 1
        if a.balance == "on" or (a.balance == "auto" and ratio < 0.25):
            # auto stops at 1:2 -- a fully balanced train set teaches a 5%-base-rate prior of 50% and buys recall
            # with precision; `on` balances fully
            tr, added = oversample(tr, a.seed, 1.0 if a.balance == "on" else 0.5)
    out = os.path.abspath(a.out)
    dd = os.path.join(out, "data")
    os.makedirs(dd, exist_ok=True)
    _write_jsonl(os.path.join(dd, "train.jsonl"), [{"messages": e["messages"]} for e in tr])
    _write_jsonl(os.path.join(dd, "valid.jsonl"), [{"messages": e["messages"]} for e in (va or tr[:max(1, len(tr) // 20)])])
    _write_jsonl(os.path.join(dd, "heldout.jsonl"), [_public(e) for e in te])
    positive = a.positive
    if task == "classify" and not positive:
        allc = balance_counts(tr + va + te)
        positive = min(allc, key=lambda k: (allc[k], k))   # the rare class is the one a reviewer exists to catch
    info = {"task": task, "source": os.path.basename(a.data), "rows_in": len(rows), "invalid": bad,
            "duplicates": dup, "conflicts": conflict, "seed": a.seed, "split": "time" if a.time_field else
            ("stratified" if task == "classify" else "random"),
            "counts": {"train": len(tr), "valid": len(va), "test": len(te)},
            "balance": {"train": balance_counts(tr), "valid": balance_counts(va), "test": balance_counts(te)},
            "oversampled": added, "labels": labels, "positive": positive, "weight_field": a.weight_field,
            "test_source": os.path.basename(a.test) if a.test else "split", "leaks": len(leaked),
            "test_hash": _h("".join(sorted(e["key"] for e in te)))}
    json.dump(info, open(os.path.join(dd, "prep.json"), "w"), indent=1)
    # The agent is weights + prompt + decision policy, versioned as ONE unit (Coinbase part 3). Serving with a
    # different system prompt than the one trained on is a different agent -- so it ships beside the weights.
    agent = {"task": task, "system": sysmsg, "labels": labels, "positive": positive,
             "output": ("label" + (" + reason" if getattr(a, "reason_field", None) else "")) if task == "classify"
             else ("option name" if task == "decision" else "free text"),
             "policy": {"flag_if": positive} if task == "classify" else None}
    json.dump(agent, open(os.path.join(out, "agent.json"), "w"), indent=1)
    if task == "decision":
        # Unsloth's FastDecisionModel builds from typed-decisions ROWS, not rendered turns; keep them beside the split
        _write_jsonl(os.path.join(dd, "decision-all.jsonl"), [{"_raw": r} for r in rows])
    _print_prep(info)
    return info


def _print_prep(i):
    print(f"== pollard-tune prep :: {i['source']} -> {i['task']}  ({i['rows_in']} rows; {i['invalid']} invalid, "
          f"{i['duplicates']} duplicate, {i['conflicts']} conflicting)")
    c = i["counts"]
    print(f"   split ({i['split']}, seed {i['seed']}): train {c['train']}  valid {c['valid']}  test {c['test']}"
          + ("" if i["test_source"] == "split" else f"  (test from {i['test_source']})"))
    if i["labels"]:
        for part in ("train", "test"):
            b = i["balance"][part]
            tot = sum(b.values()) or 1
            print(f"   {part:5} balance: " + "  ".join(f"{k} {v} ({100*v/tot:.1f}%)" for k, v in sorted(b.items())))
        print(f"   positive class (precision/recall/weighted recall are about it): {i['positive']}")
    if i["oversampled"]:
        print("   oversampled in train only: " + ", ".join(f"{k} +{v}" for k, v in i["oversampled"].items()))
    print(f"   leak check: {i['leaks']} test prompts seen in train/valid")


# ===================================================================================== train
def _has(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _cuda():
    if not _has("torch"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pick_backend(want, task, has=_has, cuda=_cuda, platform=sys.platform):
    """auto: a decision set on a CUDA box with Unsloth -> its FastDecisionModel (a clef head with calibrated
    temperatures); Apple Silicon -> mlx; otherwise TRL+PEFT; Unsloth last for chat/classify (it needs CUDA)."""
    if want and want != "auto":
        need = {"mlx": "mlx_lm", "trl": "trl", "unsloth": "unsloth"}[want]
        if not has(need) or (want == "trl" and not has("peft")):
            sys.exit(f"--backend {want} is not installed here: pip install "
                     + {"mlx": "mlx-lm", "trl": "trl peft", "unsloth": "unsloth"}[want])
        return want
    if task == "decision" and has("unsloth") and cuda():
        return "unsloth"
    if platform == "darwin" and has("mlx_lm"):
        return "mlx"
    if has("trl") and has("peft"):
        return "trl"
    if has("unsloth") and cuda():
        return "unsloth"
    sys.exit("no training backend installed. Apple Silicon: pip install mlx-lm   CUDA/CPU: pip install trl peft"
             "   (optional, CUDA: pip install unsloth)")


def _n_lines(p):
    return sum(1 for l in open(p, encoding="utf-8") if l.strip()) if os.path.exists(p) else 0


def plan_iters(n_train, epochs, batch, iters=None):
    return int(iters) if iters else max(1, math.ceil(n_train * float(epochs) / max(1, batch)))


LOSS_RE = re.compile(r"Iter (\d+): (Train|Val) loss ([0-9.]+|nan)", re.I)


def parse_mlx_line(line):
    m = LOSS_RE.search(line)
    if not m:
        return None
    return {"step": int(m.group(1)), "split": m.group(2).lower(), "loss": float(m.group(3))}


def _spawn(cmd, log_cb=None, env=None):
    """Run a training subprocess, stream its output, hand every line to log_cb. Returns the exit code."""
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    # unbuffered: through a pipe the child block-buffers, and a 2-hour run then shows no loss until it ends
    env = dict(env or os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in p.stdout:
        sys.stdout.write("   | " + line)
        if log_cb:
            log_cb(line)
    return p.wait()


def _latest_ckpt(adapter_dir):
    """mlx-lm saves NNNNNNN_adapters.safetensors every --save-every; the largest N is where a resume starts."""
    best = (0, None)
    for f in glob.glob(os.path.join(adapter_dir, "*_adapters.safetensors")):
        m = re.match(r"(\d+)_adapters\.safetensors$", os.path.basename(f))
        if m and int(m.group(1)) > best[0]:
            best = (int(m.group(1)), f)
    return best


def local_model(model):
    """A Hub id as the local snapshot dir. mlx_lm fuse insists on a COMPLETE snapshot and failed on a cache that
    lacked only README/LICENSE (Qwen2.5-0.5B-Instruct, first smoke run); training on the same id had worked.
    Handing both the directory sidesteps that, and the lineage still records the Hub id."""
    if os.path.isdir(str(model)):
        return model
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(model, "config.json")
        if isinstance(hit, str) and os.path.exists(os.path.join(os.path.dirname(hit), "tokenizer_config.json")) \
                and glob.glob(os.path.join(os.path.dirname(hit), "*.safetensors")):
            return os.path.dirname(hit)
        from huggingface_hub import snapshot_download
        return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                        "*.tiktoken", "*.py"])
    except Exception:
        return model


def train_mlx(a, run, n_train, spawn=_spawn):
    dd, ad = os.path.join(run, "data"), os.path.join(run, "adapter")
    os.makedirs(ad, exist_ok=True)
    total = plan_iters(n_train, a.epochs, a.batch, a.iters)
    base = local_model(a.base)
    if a.qlora:
        # QLoRA on mlx = LoRA on a 4-bit mlx copy of the base, then fuse --dequantize. The merged weights are the
        # dequantized 4-bit base + the adapter, not the original bf16 -- recorded in lineage so nobody confuses them.
        qd = os.path.join(run, "base-q4")
        if not os.path.exists(os.path.join(qd, "config.json")):
            rc = spawn([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", base, "--mlx-path", qd, "-q"])
            if rc:
                sys.exit("mlx_lm convert -q failed")
        base = qd
    cfg = os.path.join(run, "lora.yaml")
    open(cfg, "w").write(f"lora_parameters:\n  rank: {a.rank}\n  scale: {a.alpha / a.rank}\n  dropout: {a.dropout}\n")
    done, ck = _latest_ckpt(ad) if a.resume else (0, None)
    prog = os.path.join(ad, "pollard-progress.json")
    if a.resume and os.path.exists(prog) and os.path.exists(os.path.join(ad, "adapters.safetensors")):
        # a finished run leaves adapters.safetensors past the last numbered checkpoint; a fuse that failed after
        # it (first smoke run: an offline HF cache) must not retrain the last 20 steps on --resume
        fin = json.load(open(prog)).get("done", 0)
        if fin > done:
            done, ck = fin, os.path.join(ad, "adapters.safetensors")
    left = max(0, total - done)
    log_p = os.path.join(run, "train-log.jsonl")
    losses = []
    if a.resume and os.path.exists(log_p):          # the losses before the resume belong to the same run
        losses = [r for r in (json.loads(l) for l in open(log_p) if l.strip()) if r["step"] <= done]
    open(log_p, "w").write("".join(json.dumps(r) + "\n" for r in losses))

    def cb(line):
        r = parse_mlx_line(line)
        if r:
            r["step"] += done
            losses.append(r)
            open(log_p, "a").write(json.dumps(r) + "\n")

    if left:
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", base, "--train", "--data", dd,
               "--adapter-path", ad, "--iters", str(left), "--batch-size", str(a.batch),
               "--learning-rate", str(a.lr), "--max-seq-length", str(a.max_len), "--num-layers", str(a.lora_layers),
               "--steps-per-report", str(a.log_every), "--steps-per-eval", str(a.eval_every or max(left, 1)),
               "--save-every", str(a.save_every), "--seed", str(a.seed), "--mask-prompt", "-c", cfg]
        if a.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if ck:
            cmd += ["--resume-adapter-file", ck]
            print(f"   resuming from {os.path.basename(ck)} (step {done} of {total})")
        if spawn(cmd, cb):
            sys.exit("mlx_lm lora failed (see the log above); fix and re-run with --resume")
        json.dump({"done": total}, open(prog, "w"))
    merged = os.path.join(run, a.name)
    fuse = [sys.executable, "-m", "mlx_lm", "fuse", "--model", base, "--adapter-path", ad, "--save-path", merged]
    if a.qlora:
        fuse.append("--dequantize")
    if spawn(fuse):
        sys.exit("mlx_lm fuse failed -- the adapter is intact in " + ad)
    return {"backend": "mlx", "steps": total, "losses": losses, "adapter": ad, "merged": merged,
            "base_used": "4-bit mlx copy (QLoRA, dequantized on merge)" if a.qlora else "bf16/fp16 base"}


def _to_prompt_completion(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows]


def train_trl(a, run, n_train):
    """TRL SFT with a PEFT LoRA (4-bit QLoRA on CUDA with --qlora). Prompt/completion conversational rows, so
    the loss is on the answer only -- the same thing --mask-prompt does on mlx."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    dd, ad = os.path.join(run, "data"), os.path.join(run, "adapter")
    tok = AutoTokenizer.from_pretrained(a.base)
    kw = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if a.qlora:
        if not torch.cuda.is_available():
            sys.exit("--qlora on the trl backend needs CUDA (bitsandbytes); drop it, or use --backend mlx on a Mac")
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    total = plan_iters(n_train, a.epochs, a.batch, a.iters)
    cfg = SFTConfig(output_dir=ad, max_steps=total, per_device_train_batch_size=a.batch, learning_rate=a.lr,
                    logging_steps=a.log_every, save_steps=a.save_every, eval_strategy="steps" if a.eval_every else "no",
                    eval_steps=a.eval_every or None, seed=a.seed, report_to="none", lr_scheduler_type="cosine",
                    warmup_ratio=0.03, max_length=a.max_len, completion_only_loss=True,
                    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                    gradient_checkpointing=a.grad_checkpoint)
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    tr = SFTTrainer(model=model, args=cfg, processing_class=tok, peft_config=lora,
                    train_dataset=Dataset.from_list(_to_prompt_completion(os.path.join(dd, "train.jsonl"))),
                    eval_dataset=Dataset.from_list(_to_prompt_completion(os.path.join(dd, "valid.jsonl"))))
    resume = a.resume and bool(glob.glob(os.path.join(ad, "checkpoint-*")))
    tr.train(resume_from_checkpoint=True if resume else None)
    losses = []
    with open(os.path.join(run, "train-log.jsonl"), "a") as f:
        for h in tr.state.log_history:
            for k, s in (("loss", "train"), ("eval_loss", "val")):
                if k in h:
                    r = {"step": h.get("step"), "split": s, "loss": float(h[k])}
                    losses.append(r)
                    f.write(json.dumps(r) + "\n")
    tr.model.save_pretrained(ad)
    merged = os.path.join(run, a.name)
    _merge_peft(a.base, ad, merged, tok)
    return {"backend": "trl", "steps": total, "losses": losses, "adapter": ad, "merged": merged,
            "base_used": "4-bit NF4 base (QLoRA), merged into a bf16 reload" if a.qlora else "bf16 base"}


def _merge_peft(base, adapter, merged, tok):
    """Merge into a FRESH bf16 load of the base -- merging into the 4-bit training copy would bake the NF4
    rounding into a checkpoint that Pollard is then about to quantize a second time."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.save_pretrained(merged)
    tok.save_pretrained(merged)


def train_unsloth(a, run, n_train, task):
    """Unsloth on CUDA. typed-decisions -> FastDecisionModel: a clef-layout joint head trained on the typed
    questions, temperatures calibrated on the held-out valid rows, merged to 16-bit with joint_head.safetensors
    + joint_head_config.json beside it. Anything else -> FastLanguageModel + TRL SFT."""
    if task == "decision":
        from datasets import Dataset
        from transformers import TrainingArguments
        from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported
        raw = lambda p: [json.loads(l)["_raw"] for l in open(p) if l.strip()]
        model, tok = FastDecisionModel.from_pretrained(model_name=a.base, max_seq_length=a.max_len, load_in_4bit=a.qlora)
        model = FastDecisionModel.get_peft_model(model, r=a.rank, lora_alpha=a.alpha, lora_dropout=0,
                                                 use_gradient_checkpointing="unsloth", random_state=a.seed)
        enc = lambda rows: Dataset.from_list([{**r, "questions": json.dumps(r["questions"]) if not isinstance(r["questions"], str) else r["questions"],
                                               "gold": json.dumps(r["gold"]) if not isinstance(r["gold"], str) else r["gold"]} for r in rows])
        tr_items, rep = FastDecisionModel.build_dataset(enc(raw(os.path.join(run, "data", "decision-train.jsonl"))), tok, model)
        va_items, _ = FastDecisionModel.build_dataset(enc(raw(os.path.join(run, "data", "decision-valid.jsonl"))), tok, model)
        total = plan_iters(len(tr_items), a.epochs, a.batch, a.iters)
        t = DecisionTrainer(model=model, processing_class=tok, train_dataset=tr_items, eval_dataset=va_items,
                            args=TrainingArguments(per_device_train_batch_size=a.batch, max_steps=total, learning_rate=a.lr,
                                                   lr_scheduler_type="cosine", warmup_steps=10, logging_steps=a.log_every,
                                                   bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
                                                   output_dir=os.path.join(run, "adapter"), report_to="none", seed=a.seed,
                                                   save_steps=a.save_every))
        t.train(resume_from_checkpoint=True if a.resume and glob.glob(os.path.join(run, "adapter", "checkpoint-*")) else None)
        FastDecisionModel.calibrate(model, tok, va_items)
        merged = os.path.join(run, a.name)
        model.save_pretrained_merged(merged, tok, save_method="merged_16bit")
        for f in ("joint_head.safetensors", "joint_head_config.json"):     # the head must travel with the body
            src = os.path.join(run, "adapter", f)
            if not os.path.exists(os.path.join(merged, f)) and os.path.exists(src):
                shutil.copy(src, merged)
        losses = [{"step": h.get("step"), "split": "train", "loss": h["loss"]} for h in t.state.log_history if "loss" in h]
        return {"backend": "unsloth-decision", "steps": total, "losses": losses, "merged": merged,
                "adapter": os.path.join(run, "adapter"), "skipped": rep.get("skipped"), "base_used": "unsloth"}
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_name=a.base, max_seq_length=a.max_len, load_in_4bit=a.qlora)
    model = FastLanguageModel.get_peft_model(model, r=a.rank, lora_alpha=a.alpha, lora_dropout=0,
                                             use_gradient_checkpointing="unsloth", random_state=a.seed)
    total = plan_iters(n_train, a.epochs, a.batch, a.iters)
    t = SFTTrainer(model=model, processing_class=tok,
                   train_dataset=Dataset.from_list(_to_prompt_completion(os.path.join(run, "data", "train.jsonl"))),
                   args=SFTConfig(output_dir=os.path.join(run, "adapter"), max_steps=total, per_device_train_batch_size=a.batch,
                                  learning_rate=a.lr, logging_steps=a.log_every, save_steps=a.save_every, seed=a.seed,
                                  report_to="none", max_length=a.max_len, completion_only_loss=True))
    t.train(resume_from_checkpoint=True if a.resume and glob.glob(os.path.join(run, "adapter", "checkpoint-*")) else None)
    merged = os.path.join(run, a.name)
    model.save_pretrained_merged(merged, tok, save_method="merged_16bit")
    losses = [{"step": h.get("step"), "split": "train", "loss": h["loss"]} for h in t.state.log_history if "loss" in h]
    return {"backend": "unsloth", "steps": total, "losses": losses, "merged": merged,
            "adapter": os.path.join(run, "adapter"), "base_used": "unsloth"}


def clef_temperatures(merged):
    """The calibrated per-type temperatures of a clef-layout model (joint_head_config.json), or {}."""
    p = os.path.join(merged, "joint_head_config.json")
    if not os.path.isfile(p):
        return {}
    cfg = json.load(open(p))
    t = cfg.get("temperatures") or cfg.get("temperature") or {}
    if isinstance(t, (int, float)):
        return {"all": float(t)}
    return {str(k): float(v) for k, v in t.items()} if isinstance(t, dict) else {}


def _run_dir(a):
    run = os.path.abspath(a.out)
    if not os.path.isfile(os.path.join(run, "data", "prep.json")):
        sys.exit(f"{run} has no prepared data -- run `pollard-tune prep --data ... --out {a.out}` (or the one-shot) first")
    return run


def _name(a, run):
    return a.name or f"{os.path.basename(str(a.base).rstrip('/'))}-{os.path.basename(run)}"


def train(a, backends=None):
    run = _run_dir(a)
    info = json.load(open(os.path.join(run, "data", "prep.json")))
    if not a.base:
        sys.exit("--base is required (an HF model id or local dir)")
    a.name = _name(a, run)
    task = info["task"]
    be = pick_backend(a.backend, task)
    if task == "decision" and be == "unsloth":
        # FastDecisionModel splits per state itself; hand it the raw rows of OUR train/valid split so the
        # test states stay unseen.
        _split_raw_decision(run)
    elif task == "decision":
        print("   decision data on the GENERATION path (Unsloth FastDecisionModel needs unsloth + CUDA): each typed "
              "question becomes a chat turn answered with the option name. No clef head, no calibrated temperatures;"
              " eval reads option probabilities from the LM's own likelihoods.")
    n_train = _n_lines(os.path.join(run, "data", "train.jsonl"))
    print(f"== pollard-tune train :: {a.base} -> {os.path.join(run, a.name)}  [{be}{', QLoRA' if a.qlora else ', LoRA'}"
          f" r{a.rank} a{a.alpha}, lr {a.lr}, {plan_iters(n_train, a.epochs, a.batch, a.iters)} steps x batch {a.batch}]",
          flush=True)
    fns = backends or {"mlx": lambda: train_mlx(a, run, n_train), "trl": lambda: train_trl(a, run, n_train),
                       "unsloth": lambda: train_unsloth(a, run, n_train, task)}
    t0 = time.time()
    res = fns[be]()
    res["seconds"] = round(time.time() - t0, 1)
    res["base"] = a.base
    merged = res["merged"]
    for f in ("agent.json",):
        if os.path.exists(os.path.join(run, f)):
            shutil.copy(os.path.join(run, f), merged)
    temps = clef_temperatures(merged)
    if temps:
        res["temperatures"] = temps
    _record_lineage(a, info, res)
    tr = [x["loss"] for x in res["losses"] if x["split"] == "train"]
    va = [x["loss"] for x in res["losses"] if x["split"] == "val"]
    print(f"   done in {res['seconds']}s: train loss {tr[0] if tr else '?'} -> {tr[-1] if tr else '?'}"
          f"; val loss {va[0] if va else '?'} -> {va[-1] if va else '?'}; merged -> {merged}")
    json.dump({k: v for k, v in res.items() if k != "losses"}, open(os.path.join(run, "train.json"), "w"), indent=1)
    return res


def _split_raw_decision(run):
    dd = os.path.join(run, "data")
    test_rows = {json.loads(l).get("row") for l in open(os.path.join(dd, "heldout.jsonl")) if l.strip()}
    # the train/valid jsonl hold rendered turns only, so re-derive raw rows from the decision examples file
    src = os.path.join(dd, "decision-all.jsonl")
    if not os.path.exists(src):
        sys.exit("decision raw rows missing (re-run prep)")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    keep = [r for r in rows if r["_raw"].get("id") not in test_rows]
    cut = max(1, len(keep) // 10)
    _write_jsonl(os.path.join(dd, "decision-train.jsonl"), keep[cut:])
    _write_jsonl(os.path.join(dd, "decision-valid.jsonl"), keep[:cut])


def _record_lineage(a, info, res):
    from pollard_forge import _carry_lineage, _lineage
    merged = res["merged"]
    if os.path.isdir(str(a.base)):
        _carry_lineage(a.base, merged)
    tr = [x["loss"] for x in res["losses"] if x["split"] == "train"]
    va = [x["loss"] for x in res["losses"] if x["split"] == "val"]
    _lineage(merged, {"op": "tune", "from": a.base, "backend": res["backend"], "task": info["task"],
                      "method": "qlora" if a.qlora else "lora", "rank": a.rank, "alpha": a.alpha, "lr": a.lr,
                      "steps": res["steps"], "batch": a.batch, "max_len": a.max_len, "seed": a.seed,
                      "data": [{"source": info["source"], "weight": 1.0}], "splits": info["counts"],
                      "test_hash": info["test_hash"], "oversampled": info.get("oversampled") or {},
                      "train_loss_start": tr[0] if tr else None, "train_loss_end": tr[-1] if tr else None,
                      "val_loss_start": va[0] if va else None, "val_loss_end": va[-1] if va else None,
                      "base_used": res.get("base_used"), "temperatures": res.get("temperatures"),
                      "seconds": res.get("seconds")})


# ===================================================================================== eval
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(t):
    t = THINK_RE.sub("", t or "")
    return t.split("</think>")[-1] if "</think>" in t else t


def parse_label(text, labels):
    """The label a reply commits to: its first line if that IS a label, else the earliest whole-word label in the
    reply (longest first at a tie, so 'not fraud' beats the 'fraud' inside it). None = unparsed, scored wrong."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
    t = strip_think(text).strip()
    nl = {norm(l): l for l in labels}
    first = norm(t.splitlines()[0]) if t else ""
    if first in nl:
        return nl[first]
    full = norm(t)
    hits = []
    for k, l in nl.items():
        m = re.search(r"(^| )" + re.escape(k) + r"( |$)", full)
        if m:
            hits.append((m.start(), -len(k), l))
    return min(hits)[2] if hits else None


def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def classification_metrics(gold, pred, labels, positive=None, weights=None):
    n = len(gold)
    conf = {g: {p: 0 for p in labels + ["<unparsed>"]} for g in labels}
    for g, p in zip(gold, pred):
        conf.setdefault(g, {q: 0 for q in labels + ["<unparsed>"]})
        conf[g][p if p in labels else "<unparsed>"] += 1
    per = {}
    for l in labels:
        tp = conf.get(l, {}).get(l, 0)
        fp = sum(conf[g].get(l, 0) for g in conf if g != l)
        fn = sum(v for k, v in conf.get(l, {}).items() if k != l)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[l] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}
    out = {"n": n, "accuracy": sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0,
           "macro_f1": sum(v["f1"] for v in per.values()) / len(per) if per else 0.0,
           "unparsed": sum(1 for p in pred if p not in labels) / n if n else 0.0,
           "per_class": per, "confusion": conf}
    if positive:
        pc = per.get(positive, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        out.update({"positive": positive, "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"]})
        if weights is not None:
            # Coinbase's fourth metric: of all the fraudulent VALUE, what share did it catch. A model can hold
            # recall by catching many small cases and still miss the ones that cost money.
            tot = sum(w or 0 for g, w in zip(gold, weights) if g == positive)
            hit = sum(w or 0 for g, p, w in zip(gold, pred, weights) if g == positive and p == positive)
            out["weighted_recall"] = hit / tot if tot else 0.0
    return out


def chat_metrics(refs, preds, judged=None):
    n = len(refs)
    norm = lambda s: re.sub(r"\s+", " ", strip_think(s)).strip().lower()
    out = {"n": n, "exact": sum(1 for r, p in zip(refs, preds) if norm(r) == norm(p)) / n if n else 0.0,
           "contains": sum(1 for r, p in zip(refs, preds) if norm(r) and norm(r) in norm(p)) / n if n else 0.0}
    if judged is not None:
        out["judge_pass"] = sum(1 for j in judged if j) / n if n else 0.0
    return out


def ece(confs, correct, bins=10):
    """Expected calibration error over argmax confidence: a decision model that says 0.9 should be right 90%."""
    if not confs:
        return None
    tot, n = 0.0, len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(acc - conf)
    return tot


def decision_metrics(rows):
    """rows: [{dist, gold:{probabilities}, label}] -> pollard_decision.score_typed's board + ECE."""
    import pollard_decision as pd
    typed = [{"row": r.get("row"), "workflow": r.get("workflow"), "qid": r.get("qid"), "type": r.get("qtype"),
              "dist": r["dist"], "gold": {"label": r["label"], "probabilities": r.get("gold") or {}}} for r in rows]
    board = pd.score_typed(typed)
    confs, correct = [], []
    for r in rows:
        d = r["dist"]
        if d:
            top = max(d, key=d.get)
            confs.append(d[top]); correct.append(1 if top == r["label"] else 0)
    board["ece"] = ece(confs, correct)
    return board


# ---------------------------------------------------------------- predictors
class Endpoint:
    """Any OpenAI-compatible chat endpoint: a frontier API, vLLM, llama-server, a bigger local model."""

    def __init__(self, url, model, key_env=None, max_tokens=64, post=None, extra=None):
        self.base = url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.model, self.max_tokens = model, max_tokens
        self.key = os.environ.get(key_env) if key_env else None
        self.post = post or self._post
        self.name = model
        # provider-specific request fields (--compare-body). A thinking model with an 8-token budget returns
        # "<think>..." and nothing else; qwen3:8b on Ollama needed {"reasoning_effort": "none"} to answer at all.
        self.extra = extra or {}

    def _post(self, url, body):
        import urllib.request
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
        return json.loads(urllib.request.urlopen(req, timeout=300).read().decode())

    def generate(self, messages):
        t0 = time.perf_counter()
        r = self.post(self.base + "/v1/chat/completions", {"model": self.model, "messages": messages,
                                                             "temperature": 0, "max_tokens": self.max_tokens, **self.extra})
        dt = time.perf_counter() - t0
        return (r.get("choices") or [{}])[0].get("message", {}).get("content") or "", dt

    def option_probs(self, messages, options):
        text, dt = self.generate(messages)
        lab = parse_label(text, options)
        eps = 1e-3
        return ({o: (1 - eps * (len(options) - 1)) if o == lab else eps for o in options} if lab else
                {o: 1 / len(options) for o in options}), dt


def _template(tok, messages):
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _softmax_dict(d):
    m = max(d.values())
    e = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


class MLXModel:
    def __init__(self, path, max_tokens=64, name=None):
        from mlx_lm import load
        self.model, self.tok = load(path)
        self.max_tokens, self.name = max_tokens, name or os.path.basename(str(path).rstrip("/"))

    def generate(self, messages):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = _template(self.tok, messages)
        t0 = time.perf_counter()
        out = generate(self.model, self.tok, prompt, max_tokens=self.max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
        return out, time.perf_counter() - t0

    def option_probs(self, messages, options):
        import mlx.core as mx
        prompt = _template(self.tok, messages)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        t0 = time.perf_counter()
        lp = {}
        for o in options:
            oi = self.tok.encode(o, add_special_tokens=False)
            logits = self.model(mx.array([ids + oi]))[0].astype(mx.float32)
            ls = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            lp[o] = float(sum(ls[len(ids) - 1 + j, t].item() for j, t in enumerate(oi)))
        return _softmax_dict(lp), time.perf_counter() - t0


class HFModel:
    def __init__(self, path, max_tokens=64, name=None, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16 if self.dev == "cuda" else torch.float16 if self.dev == "mps" else torch.float32).to(self.dev).eval()
        self.max_tokens, self.name = max_tokens, name or os.path.basename(str(path).rstrip("/"))

    def generate(self, messages):
        t = self.torch
        ids = self.tok(_template(self.tok, messages), return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
        t0 = time.perf_counter()
        with t.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), time.perf_counter() - t0

    def option_probs(self, messages, options):
        t = self.torch
        ids = self.tok(_template(self.tok, messages), add_special_tokens=False).input_ids
        t0 = time.perf_counter()
        lp = {}
        with t.no_grad():
            for o in options:
                oi = self.tok(o, add_special_tokens=False).input_ids
                x = t.tensor([ids + oi], device=self.dev)
                ls = t.log_softmax(self.model(x).logits[0].float(), dim=-1)
                lp[o] = float(sum(ls[len(ids) - 1 + j, tid] for j, tid in enumerate(oi)))
        return _softmax_dict(lp), time.perf_counter() - t0


def evaluate_model(m, rows, task, labels=None, positive=None, judge=None):
    """One model over the held-out rows. Returns {metrics, latency, predictions}."""
    preds, lat, outs = [], [], []
    for i, r in enumerate(rows):
        msgs = r["messages"][:-1]
        if task == "decision":
            dist, dt = m.option_probs(msgs, r["options"])
            outs.append({"dist": dist})
            preds.append(dist)
        else:
            text, dt = m.generate(msgs)
            outs.append({"text": text})
            preds.append(text)
        lat.append(dt)
        if (i + 1) % 25 == 0:
            print(f"     {m.name}: {i + 1}/{len(rows)}", flush=True)
    if task == "classify":
        pl = [parse_label(t, labels) for t in preds]
        for o, p in zip(outs, pl):
            o["label"] = p
        ws = [r.get("weight") for r in rows]
        met = classification_metrics([r["label"] for r in rows], pl, labels, positive,
                                     ws if any(w is not None for w in ws) else None)
    elif task == "chat":
        refs = [r["messages"][-1]["content"] for r in rows]
        judged = [judge(r["messages"][:-1], ref, p) for r, ref, p in zip(rows, refs, preds)] if judge else None
        met = chat_metrics(refs, preds, judged)
    else:
        met = decision_metrics([{**r, "dist": d} for r, d in zip(rows, preds)])
    met["latency_p50_s"] = percentile(lat, 50)
    met["latency_p99_s"] = percentile(lat, 99)
    return {"model": m.name, "metrics": met, "predictions": outs}


def make_judge(url, model, key_env=None, post=None):
    ep = Endpoint(url, model, key_env, max_tokens=8, post=post)

    def judge(prompt_msgs, ref, cand):
        q = (f"Conversation so far:\n{json.dumps(prompt_msgs, ensure_ascii=False)[:4000]}\n\nReference answer:\n{ref}\n\n"
             f"Candidate answer:\n{strip_think(cand)}\n\nDoes the candidate answer as well as the reference? "
             "Reply with exactly PASS or FAIL.")
        text, _ = ep.generate([{"role": "user", "content": q}])
        return "PASS" in text.upper()
    return judge


def _is_remote(url):
    from urllib.parse import urlparse
    host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    return host not in LOOPBACK


HEADLINE = {"classify": [("precision", "precision ({pos})"), ("recall", "recall ({pos})"), ("f1", "F1 ({pos})"),
                         ("weighted_recall", "weighted recall ({pos}, by {wf})"), ("macro_f1", "macro F1"),
                         ("accuracy", "accuracy"), ("unparsed", "unparsed replies")],
            "chat": [("exact", "exact match"), ("contains", "contains reference"), ("judge_pass", "judge pass")],
            "decision": [("accuracy", "accuracy (argmax = gold label)"), ("kl_from_gold", "KL from gold (lower better)"),
                         ("p_correct", "p(gold label)"), ("ece", "ECE (lower better)")]}


def comparison_table(task, results, info):
    """results: [(column, eval_result)] -> markdown table, one metric per row, plus tuned-minus-base deltas."""
    pos, wf = info.get("positive"), info.get("weight_field")
    cols = [c for c, _ in results]
    has_base = "base" in cols and "tuned" in cols
    hdr = "| metric | " + " | ".join(cols) + (" | tuned - base |" if has_base else " |")
    lines = [hdr, "|---|" + "---:|" * (len(cols) + (1 if has_base else 0))]
    rows = HEADLINE[task] + [("latency_p50_s", "latency P50 (s)"), ("latency_p99_s", "latency P99 (s)")]
    for key, label in rows:
        vals = [r["metrics"].get(key) for _, r in results]
        if all(v is None for v in vals):
            continue
        fmt = lambda v: "-" if v is None else (f"{v:.3f}" if key.startswith("latency") or key in ("kl_from_gold", "ece")
                                               else f"{100 * v:.1f}%")
        line = f"| {label.format(pos=pos, wf=wf)} | " + " | ".join(fmt(v) for v in vals) + " |"
        if has_base:
            b, t = results[cols.index("base")][1]["metrics"].get(key), results[cols.index("tuned")][1]["metrics"].get(key)
            if b is None or t is None:
                d = "-"
            elif key.startswith("latency") or key in ("kl_from_gold", "ece"):
                d = f"{t - b:+.3f}"
            else:
                d = f"{100 * (t - b):+.1f} pts"
            line += f" {d} |"
        lines.append(line)
    return "\n".join(lines)


def confusion_md(m):
    conf = m.get("confusion") or {}
    if not conf:
        return ""
    preds = list(next(iter(conf.values())).keys())
    preds = [p for p in preds if p != "<unparsed>" or any(conf[g].get(p) for g in conf)]
    out = ["| gold \\ predicted | " + " | ".join(preds) + " |", "|---|" + "---:|" * len(preds)]
    for g in conf:
        out.append(f"| {g} | " + " | ".join(str(conf[g].get(p, 0)) for p in preds) + " |")
    return "\n".join(out)


def report_md(task, results, info, lineage=None):
    """The card section (`## ` headings only, as pollard-card --extra-md requires): the measured comparison and
    how the model was tuned. Sources by file name only -- a card never carries a home path."""
    n = results[0][1]["metrics"].get("n") if results else 0
    md = ["## Fine-tune evaluation", "",
          f"Held-out test split: {n} {'decisions' if task == 'decision' else 'examples'} never seen in training "
          f"({info.get('split')} split, seed {info.get('seed')}, leak check: {info.get('leaks', 0)} overlaps). Same "
          "prompt and decision policy for every column; greedy decoding.", "",
          comparison_table(task, results, info), ""]
    if task == "classify":
        for col, r in results:
            if col == "tuned":
                md += [f"Confusion matrix (tuned):", "", confusion_md(r["metrics"]), ""]
    if lineage:
        md += ["## How it was tuned", "", "| | |", "|---|---|"]
        for k in ("from", "backend", "method", "rank", "alpha", "lr", "steps", "batch", "max_len"):
            if lineage.get(k) is not None:
                md.append(f"| {k} | {lineage[k]} |")
        sp = lineage.get("splits") or {}
        md.append(f"| data | {info.get('source')} -- train {sp.get('train')} / valid {sp.get('valid')} / test {sp.get('test')} |")
        if lineage.get("val_loss_start") is not None:
            md.append(f"| validation loss | {lineage['val_loss_start']} -> {lineage['val_loss_end']} |")
        md.append("")
    return "\n".join(md)


def _load_models(a, run, tuned_path, task):
    """Which runtime loads the base and tuned models for eval: mlx on a Mac with mlx-lm, else transformers."""
    use = a.eval_backend
    if use == "auto":
        use = "mlx" if (sys.platform == "darwin" and _has("mlx_lm")) else "hf"
    cls = MLXModel if use == "mlx" else HFModel
    mt = 8 if task == "decision" else a.max_new_tokens
    return (lambda: cls(local_model(a.base), mt, name="base")), (lambda: cls(tuned_path, mt, name="tuned"))


def evaluate(a, loaders=None, compare=None):
    run = _run_dir(a)
    info = json.load(open(os.path.join(run, "data", "prep.json")))
    task = info["task"]
    rows = [json.loads(l) for l in open(os.path.join(run, "data", "heldout.jsonl")) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    tp = json.load(open(os.path.join(run, "train.json"))) if os.path.exists(os.path.join(run, "train.json")) else {}
    a.base = a.base or tp.get("base")
    tuned_path = tp.get("merged") or os.path.join(run, _name(a, run))
    for url in (a.compare_url, a.judge_url):
        if url and _is_remote(url) and not a.allow_remote:
            sys.exit(f"{url} is not on this machine: the held-out rows ({len(rows)} of YOUR data) would be sent there. "
                     "Pass --allow-remote if that is intended (e.g. a frontier API you are cleared to send them to).")
    judge = make_judge(a.judge_url, a.judge_model, a.api_key_env) if a.judge_url else None
    base_l, tuned_l = loaders or _load_models(a, run, tuned_path, task)
    results = []
    ed = os.path.join(run, "eval")
    os.makedirs(ed, exist_ok=True)
    print(f"== pollard-tune eval :: {len(rows)} held-out {task} examples")
    cols = []
    if not a.skip_base:
        cols.append(("base", base_l))
    cols.append(("tuned", tuned_l))
    if a.compare_url:
        cols.append((a.compare_model or "compare", lambda: compare or Endpoint(
            a.compare_url, a.compare_model, a.api_key_env, 8 if task == "decision" else a.max_new_tokens,
            extra=json.loads(a.compare_body) if a.compare_body else None)))
    for col, load in cols:
        m = load()
        r = evaluate_model(m, rows, task, info.get("labels"), info.get("positive"), judge)
        r["column"] = col
        json.dump(r, open(os.path.join(ed, f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', col)}.json"), "w"), indent=1)
        results.append((col, r))
        del m
    lineage = None
    fj = os.path.join(tuned_path, "forge.json")
    if os.path.exists(fj):
        lineage = next((s for s in reversed(json.load(open(fj))["steps"]) if s.get("op") == "tune"), None)
    table = comparison_table(task, results, info)
    md = report_md(task, results, info, lineage)
    open(os.path.join(ed, "report.md"), "w").write(md + "\n")
    summary = {"task": task, "n": len(rows), "columns": {c: r["metrics"] for c, r in results}}
    json.dump(summary, open(os.path.join(ed, "summary.json"), "w"), indent=1)
    if lineage is not None:                          # the eval result belongs in the lineage too
        from pollard_forge import _lineage
        head = {c: {k: r["metrics"].get(k) for k, _ in HEADLINE[task] if r["metrics"].get(k) is not None}
                for c, r in results}
        _lineage(tuned_path, {"op": "tune-eval", "n": len(rows), "test_hash": info.get("test_hash"), "results": head})
    print(table)
    print(f"   wrote {os.path.join(ed, 'report.md')} (card section) and summary.json")
    return summary


# ===================================================================================== ship
def _tool(cmd):
    """`pollard-*` launchers resolve through pollard_auto's fallback; bare `pollard` the same way (an editable
    checkout without its console scripts on PATH still runs the module)."""
    import pollard_auto
    if cmd and cmd[0] == "pollard" and not shutil.which("pollard"):
        return [sys.executable, os.path.abspath(pollard_auto.__file__)] + list(cmd[1:])
    return pollard_auto._resolve_tool(cmd)


def serve_available():
    import importlib.util
    try:
        return importlib.util.find_spec("pollard_harness") is not None
    except (ImportError, ValueError):
        return False


PREFERRED_RUNGS = ("Q4_K_M", "IQ4_XS", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0")


def pick_rung(builds):
    gg = [b for b in builds if b.get("lane", "gguf") == "gguf" and b.get("path")]
    for t in PREFERRED_RUNGS:
        for b in gg:
            if str(b.get("tag", "")).upper() == t or f"-{t}" in str(b.get("name", "")).upper():
                return b
    return min(gg, key=lambda b: b.get("bytes") or 1e18) if gg else None


def serve_line(rung_path, harness, decision=False, available=None):
    """The hand-off: `pollard serve` when this install has it (pw-harness branch), else the llama-server line
    it would have run, so the step degrades instead of failing."""
    available = serve_available() if available is None else available
    if available:
        return f"pollard serve {rung_path} --harness {harness}" + (" --decision" if decision else "")
    ctx = 4096 if decision else 32768
    return (f"llama-server -m {rung_path} -c {ctx} --jinja --port 8080 "
            f"   # `pollard serve --harness {harness}` is not in this install yet; this is the server it would start")


def ship(a, run_cmd=None):
    run = _run_dir(a)
    tp = json.load(open(os.path.join(run, "train.json")))
    merged = tp["merged"]
    a.base = a.base or tp.get("base")
    info = json.load(open(os.path.join(run, "data", "prep.json")))
    report = os.path.join(run, "eval", "report.md")
    run_cmd = run_cmd or (lambda cmd: subprocess.run(_tool(cmd)).returncode)
    print(f"== pollard-tune ship :: {merged}")
    # preflight first: seconds, no weights -- the build must not die an hour in on a converter gap
    if run_cmd(["pollard-smoke", "--model", merged]) != 0 and not a.force:
        sys.exit("pollard-smoke says this checkpoint will not build here (above). Fix that, or --force.")
    rc = run_cmd(["pollard", "--hf", merged, "--run", "--ladder"] + (["--force"] if a.force else []))
    if rc != 0:
        sys.exit(f"`pollard --hf {merged} --run --ladder` failed (exit {rc})")
    f16 = os.path.join(os.path.dirname(merged), os.path.basename(merged.rstrip("/")) + "-f16.gguf")
    temps = tp.get("temperatures") or {}
    if temps and os.path.exists(f16):
        _check_decision_temps(f16, temps)
    card = ["pollard-card", "--model", f16 if os.path.exists(f16) else merged, "--base-model", a.base or merged,
            "--title", os.path.basename(merged.rstrip("/")), "--out", os.path.join(run, "README.md")]
    if os.path.exists(report):
        card += ["--extra-md", report]
    if a.license:
        card += ["--license", a.license]
    if run_cmd(card) != 0:
        print("   (card step failed -- the builds are fine; re-run pollard-card with --extra-md eval/report.md)")
    try:
        import pollard_workspace as ws
        builds = ws.read_manifest(f16).get("builds", [])
    except Exception:
        builds = []
    rung = pick_rung(builds)
    line = serve_line(rung["path"] if rung else f"<{os.path.basename(merged)} rung>.gguf", a.harness,
                      info["task"] == "decision")
    print("\n   serve it:\n     " + line)
    agent = os.path.join(merged, "agent.json")
    if os.path.exists(agent):
        print(f"   the system prompt it was trained with is in {agent} -- serve with THAT prompt; a different one is a "
              "different agent (and the eval no longer describes it)")
    return line


def _check_decision_temps(f16, temps):
    """Unsloth's exporter writes `<arch>.decision.temperature.<type>`; the stock clef converter drops them and the
    head then answers at T=1. Say so with the values to put back, instead of shipping uncalibrated quietly."""
    try:
        from pollard_calc import read_gguf_meta
        meta = read_gguf_meta(f16)
    except Exception:
        return
    if not any(".decision.temperature." in str(k) for k in meta):
        print("   WARNING: the f16 GGUF has no `.decision.temperature.*` keys -- the converter dropped the calibrated "
              "temperatures (" + ", ".join(f"{k}={v:.3f}" for k, v in temps.items()) + "). Export with Unsloth's "
              "GGUF exporter, or re-add them, before trusting the probabilities.")


# ===================================================================================== CLI
def _common(p):
    p.add_argument("--out", required=True, help="the run directory (data, adapter, merged model, eval)")
    p.add_argument("--base", help="HF model id or local dir to fine-tune")
    p.add_argument("--name", help="merged model name (default: <base>-<run dir name>)")


def _prep_args(p):
    p.add_argument("--data", required=True, help=".jsonl / .json / .csv: chat {messages}, classifier rows, or typed-decisions")
    p.add_argument("--task", default="auto", choices=("auto",) + TASKS)
    p.add_argument("--test", help="your own test file (else split from --data); refused if it overlaps train")
    p.add_argument("--input-field"); p.add_argument("--label-field"); p.add_argument("--reason-field")
    p.add_argument("--weight-field", help="per-row cost (e.g. amount): adds weighted recall on the positive class")
    p.add_argument("--time-field", help="split by time: oldest -> train, newest -> test")
    p.add_argument("--id-field", help="a column that is an id, kept out of the prompt")
    p.add_argument("--positive", help="the class precision/recall are about (default: the rarest)")
    p.add_argument("--instructions", help="the task instruction for classify (the system prompt's first sentence)")
    p.add_argument("--valid-frac", type=float, default=0.1); p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--balance", default="auto", choices=("auto", "on", "off"),
                   help="oversample minority labels in TRAIN. auto: up to 1:2 when the rarest is under 1/4 of the largest; on: 1:1")
    p.add_argument("--allow-leak", action="store_true"); p.add_argument("--seed", type=int, default=0)


def _train_args(p):
    p.add_argument("--backend", default="auto", choices=("auto",) + BACKENDS)
    p.add_argument("--qlora", action="store_true", help="train on a 4-bit base (mlx: mlx 4-bit; trl: NF4 on CUDA)")
    p.add_argument("--rank", type=int, default=16); p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--dropout", type=float, default=0.0); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2); p.add_argument("--iters", type=int, help="steps (overrides --epochs)")
    p.add_argument("--batch", type=int, default=4); p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lora-layers", type=int, default=16, help="mlx: how many top layers get adapters (-1 = all)")
    p.add_argument("--log-every", type=int, default=10); p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=100); p.add_argument("--resume", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    if not any(x.dest == "seed" for x in p._actions):
        p.add_argument("--seed", type=int, default=0)


def _eval_args(p):
    p.add_argument("--compare-url", help="OpenAI-compatible endpoint to score as a third column")
    p.add_argument("--compare-model", help="model name at --compare-url")
    p.add_argument("--compare-body", help='extra JSON merged into each compare request, e.g. \'{"reasoning_effort": "none"}\'')
    p.add_argument("--api-key-env", default=None, help="env var holding the endpoint key (e.g. OPENAI_API_KEY)")
    p.add_argument("--judge-url"); p.add_argument("--judge-model", help="chat: a judge model for PASS/FAIL")
    p.add_argument("--allow-remote", action="store_true", help="allow sending held-out rows to a non-local endpoint")
    p.add_argument("--eval-backend", default="auto", choices=("auto", "mlx", "hf"))
    p.add_argument("--max-new-tokens", type=int, default=64); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-base", action="store_true")


def _ship_args(p):
    p.add_argument("--harness", default="omp", help="agent harness for the serve line (pollard serve --harness)")
    p.add_argument("--license", help="card license (default: read off the base model)")
    p.add_argument("--force", action="store_true")


def build_parser():
    ap = argparse.ArgumentParser(prog="pollard-tune", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep", help="validate, dedupe, split, leak-check"); _common(p); _prep_args(p)
    t = sub.add_parser("train", help="LoRA/QLoRA SFT + merge"); _common(t); _train_args(t)
    e = sub.add_parser("eval", help="base vs tuned (vs --compare-url) on the test split"); _common(e); _eval_args(e)
    s = sub.add_parser("ship", help="pollard ladder + card with the eval + serve line"); _common(s); _ship_args(s)
    r = sub.add_parser("run", help="prep -> train -> eval [-> ship]"); _common(r); _prep_args(r); _train_args(r)
    _eval_args(r); _ship_args(r)
    r.add_argument("--ship", action="store_true", help="also build the GGUF ladder + card and print the serve line")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("prep", "train", "eval", "ship", "run", "-h", "--help"):
        argv = ["run"] + argv                  # `pollard tune --base ... --data ...` is the one-shot
    a = build_parser().parse_args(argv)
    if a.cmd == "prep":
        prep(a)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "eval":
        evaluate(a)
    elif a.cmd == "ship":
        ship(a)
    else:
        if not a.base:
            sys.exit("--base is required")
        prep(a)
        train(a)
        evaluate(a)
        if a.ship:
            ship(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
