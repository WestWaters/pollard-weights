"""pollard-tune: data prep (dedupe, stratified/time split, leak refusal, balance), backend choice, the mlx
command line + resume, metrics (per-class P/R/F1, dollar-weighted recall, latency, ECE), the eval table, the
lineage step, and the ship hand-off to `pollard serve` -- all on tiny synthetic data with mocked trainers and
models, so none of it needs mlx / trl / torch installed.

What it protects: a test split that leaks into train (the number then measures memory), a 5% class that
vanishes from a split, a weighted recall computed over the wrong rows, a resume that retrains finished steps,
and a ship step that dies where `pollard serve` is not on main yet.
"""
from __future__ import annotations

import argparse, csv, json, os, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_tune as T  # noqa: E402


def _args(**kw):
    p = T.build_parser()
    base = ["run", "--out", kw.pop("out"), "--data", kw.pop("data"), "--base", kw.pop("base", "tiny/base")]
    a = p.parse_args(base + kw.pop("extra", []))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _fraud_csv(path, n=200, fraud_every=10, dup=0):
    rows = []
    for i in range(n):
        fraud = i % fraud_every == 0
        rows.append({"txn_id": f"t{i}", "amount": 1000 + i if fraud else 10 + i, "age_days": 1 if fraud else 300 + i,
                     "country_mismatch": "yes" if fraud else "no", "label": "fraud" if fraud else "legit",
                     "ts": f"2026-01-{1 + i % 28:02d}T{i % 24:02d}:00"})
    rows += rows[:dup]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


# ------------------------------------------------------------------ prep
def test_prep_stratifies_dedupes_and_writes_the_agent(tmp_path):
    data = _fraud_csv(tmp_path / "f.csv", dup=7)
    a = _args(out=str(tmp_path / "run"), data=str(data),
              extra=["--label-field", "label", "--weight-field", "amount", "--id-field", "txn_id", "--balance", "off"])
    info = T.prep(a)
    assert info["duplicates"] == 7 and info["task"] == "classify"
    # 20 fraud of 200: every split must still hold some (a random split can leave the test set with none)
    assert info["balance"]["test"].get("fraud", 0) >= 2 and info["balance"]["valid"].get("fraud", 0) >= 1
    assert info["positive"] == "fraud" and info["leaks"] == 0
    held = [json.loads(l) for l in open(tmp_path / "run" / "data" / "heldout.jsonl")]
    assert all(r["weight"] is not None for r in held)
    # the id never reaches the prompt; the amount (the weight) does -- it is a feature as well as a cost
    user = held[0]["messages"][1]["content"]
    assert "txn_id" not in user and "amount:" in user
    train = [json.loads(l) for l in open(tmp_path / "run" / "data" / "train.jsonl")]
    assert set(train[0]) == {"messages"}               # what mlx-lm / TRL read, nothing else
    agent = json.load(open(tmp_path / "run" / "agent.json"))
    assert agent["labels"] == ["fraud", "legit"] and "fraud, legit" in agent["system"]


def test_prep_refuses_a_test_file_that_leaks_into_train(tmp_path):
    data = _fraud_csv(tmp_path / "f.csv")
    leak = _fraud_csv(tmp_path / "t.csv", n=30)                 # same first 30 cases as the training file
    a = _args(out=str(tmp_path / "run"), data=str(data), extra=["--label-field", "label", "--test", str(leak)])
    with pytest.raises(SystemExit, match="REFUSING to train"):
        T.prep(a)
    a.allow_leak = True
    assert T.prep(a)["leaks"] > 0


def test_time_split_puts_the_newest_cases_in_test(tmp_path):
    data = _fraud_csv(tmp_path / "f.csv", n=100)
    a = _args(out=str(tmp_path / "run"), data=str(data), extra=["--label-field", "label", "--time-field", "ts"])
    info = T.prep(a)
    assert info["split"] == "time"
    rows = T.read_rows(str(data))
    newest = sorted(r["ts"] for r in rows)[-info["counts"]["test"]:]
    held = [json.loads(l)["time"] for l in open(tmp_path / "run" / "data" / "heldout.jsonl")]
    assert sorted(held) == newest


def test_auto_balance_oversamples_train_only(tmp_path):
    data = _fraud_csv(tmp_path / "f.csv", n=200, fraud_every=20)        # 5% fraud, like Coinbase's onramp set
    a = _args(out=str(tmp_path / "run"), data=str(data), extra=["--label-field", "label"])
    info = T.prep(a)
    tb = info["balance"]["train"]
    assert info["oversampled"].get("fraud") and tb["fraud"] / tb["legit"] == pytest.approx(0.5, abs=0.05)
    assert info["balance"]["test"]["fraud"] / sum(info["balance"]["test"].values()) < 0.1


def test_conflicting_labels_for_the_same_case_are_counted_and_dropped():
    ex = [{"key": "a1", "prompt_key": "a", "label": "x"}, {"key": "a2", "prompt_key": "a", "label": "y"},
          {"key": "a1", "prompt_key": "a", "label": "x"}, {"key": "b1", "prompt_key": "b", "label": "x"}]
    out, dup, conflict = T.dedupe(ex)
    assert [e["key"] for e in out] == ["a1", "b1"] and dup == 1 and conflict == 1


def test_chat_rows_need_a_final_assistant_turn(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(30)]
    rows.append({"messages": [{"role": "user", "content": "dangling"}]})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    a = _args(out=str(tmp_path / "run"), data=str(p))
    info = T.prep(a)
    assert info["task"] == "chat" and info["invalid"] == 1 and sum(info["counts"].values()) == 30


TYPED = {"id": "s0", "workflow": "invoice", "state": json.dumps({"amount": 12}),
         "questions": {"route": {"type": "choice", "instructions": "Who?", "criteria": {"ap": "payables", "fraud": "risk"}},
                       "dup": {"type": "noul", "instructions": "Duplicate?"},
                       "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "mid", "high"]}},
         "gold": {"route": {"label": "ap", "probabilities": {"ap": 0.8, "fraud": 0.2}},
                  "dup": {"label": "false", "probabilities": {"true": 0.1, "false": 0.9}},
                  "risk": {"label": "0", "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}}}}


def test_typed_decisions_split_by_state_and_keep_gold(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({**TYPED, "id": f"s{i}", "state": json.dumps({"amount": i})}) for i in range(20)))
    a = _args(out=str(tmp_path / "run"), data=str(p))
    info = T.prep(a)
    assert info["task"] == "decision"
    held = [json.loads(l) for l in open(tmp_path / "run" / "data" / "heldout.jsonl")]
    train_users = {json.loads(l)["messages"][1]["content"].split("\n\nQuestion")[0]
                   for l in open(tmp_path / "run" / "data" / "train.jsonl")}
    # all five questions of a state travel together: none of a test state's turns is in train
    assert not {h["messages"][1]["content"].split("\n\nQuestion")[0] for h in held} & train_users
    assert {h["qtype"] for h in held} == {"choice", "noul", "score"}
    assert next(h for h in held if h["qtype"] == "score")["options"] == ["0", "1", "2"]
    assert (tmp_path / "run" / "data" / "decision-all.jsonl").exists()


# ------------------------------------------------------------------ backend + mlx
def test_backend_choice():
    has = lambda mods: (lambda m: m in mods)
    assert T.pick_backend("auto", "classify", has({"mlx_lm"}), lambda: False, "darwin") == "mlx"
    assert T.pick_backend("auto", "classify", has({"trl", "peft"}), lambda: True, "linux") == "trl"
    assert T.pick_backend("auto", "decision", has({"unsloth", "trl", "peft"}), lambda: True, "linux") == "unsloth"
    # a decision set on a Mac goes the generation path on mlx, not a CUDA-only trainer
    assert T.pick_backend("auto", "decision", has({"mlx_lm", "unsloth"}), lambda: False, "darwin") == "mlx"
    with pytest.raises(SystemExit, match="not installed"):
        T.pick_backend("trl", "chat", has({"trl"}), lambda: False, "linux")        # peft missing
    with pytest.raises(SystemExit, match="no training backend"):
        T.pick_backend("auto", "chat", has(set()), lambda: False, "linux")


def test_mlx_line_parser():
    assert T.parse_mlx_line("Iter 40: Val loss 0.230, Val took 3.386s") == {"step": 40, "split": "val", "loss": 0.23}
    assert T.parse_mlx_line("Iter 10: Train loss 1.252, Learning Rate 1.000e-04")["split"] == "train"
    assert T.parse_mlx_line("Loading pretrained model") is None


def _prepped(tmp_path, n=120):
    data = _fraud_csv(tmp_path / "f.csv", n=n)
    a = _args(out=str(tmp_path / "run"), data=str(data),
              extra=["--label-field", "label", "--weight-field", "amount", "--iters", "30", "--backend", "mlx"])
    T.prep(a)
    return a


def test_mlx_train_command_resume_and_lineage(tmp_path, monkeypatch):
    a = _prepped(tmp_path)
    run = str(tmp_path / "run")
    monkeypatch.setattr(T, "local_model", lambda m: m)
    calls = []

    def spawn(cmd, cb=None, env=None):
        calls.append(cmd)
        if "lora" in cmd:
            n = int(cmd[cmd.index("--iters") + 1])
            for s in (10, 20, 30)[:n // 10]:
                cb(f"Iter {s}: Train loss {1.0 / s:.3f}, Learning Rate 1e-4")
            cb(f"Iter {n}: Val loss 0.050, Val took 1s")
            open(os.path.join(cmd[cmd.index("--adapter-path") + 1], "adapters.safetensors"), "w").write("x")
        if "fuse" in cmd:
            out = cmd[cmd.index("--save-path") + 1]
            os.makedirs(out, exist_ok=True)
            json.dump({"model_type": "qwen2"}, open(os.path.join(out, "config.json"), "w"))
        return 0

    a.name = "m"
    res = T.train(a, backends={"mlx": lambda: T.train_mlx(a, run, 99, spawn=spawn)})
    lora = next(c for c in calls if "lora" in c)
    assert "--mask-prompt" in lora and lora[lora.index("--iters") + 1] == "30"
    assert "rank: 16" in open(os.path.join(run, "lora.yaml")).read()
    steps = json.load(open(os.path.join(res["merged"], "forge.json")))["steps"]
    t = steps[-1]
    assert t["op"] == "tune" and t["backend"] == "mlx" and t["val_loss_end"] == 0.05 and t["splits"]["test"] > 0
    assert os.path.exists(os.path.join(res["merged"], "agent.json"))
    # resume after a finished run whose FUSE failed: no retraining, straight to fuse, losses kept
    calls.clear()
    a.resume = True
    res2 = T.train(a, backends={"mlx": lambda: T.train_mlx(a, run, 99, spawn=spawn)})
    assert not any("lora" in c for c in calls) and any("fuse" in c for c in calls)
    assert [x for x in res2["losses"] if x["split"] == "val"]


def test_mlx_resume_from_a_numbered_checkpoint(tmp_path, monkeypatch):
    a = _prepped(tmp_path)
    run = str(tmp_path / "run")
    monkeypatch.setattr(T, "local_model", lambda m: m)
    ad = os.path.join(run, "adapter")
    os.makedirs(ad, exist_ok=True)
    open(os.path.join(ad, "0000020_adapters.safetensors"), "w").write("x")
    seen = []

    def spawn(cmd, cb=None, env=None):
        seen.append(cmd)
        if "fuse" in cmd:
            os.makedirs(cmd[cmd.index("--save-path") + 1], exist_ok=True)
        return 0

    a.resume, a.name = True, "m"
    T.train_mlx(a, run, 99, spawn=spawn)
    lora = next(c for c in seen if "lora" in c)
    assert lora[lora.index("--iters") + 1] == "10"           # 30 planned, 20 done
    assert lora[lora.index("--resume-adapter-file") + 1].endswith("0000020_adapters.safetensors")


def test_plan_iters():
    assert T.plan_iters(100, 2, 4) == 50 and T.plan_iters(100, 2, 4, iters=7) == 7 and T.plan_iters(0, 1, 4) == 1


# ------------------------------------------------------------------ metrics
def test_parse_label_prefers_first_line_and_longest_match():
    labels = ["fraud", "not fraud"]
    assert T.parse_label("not fraud\nthe account is old", labels) == "not fraud"
    assert T.parse_label("Fraud.", labels) == "fraud"
    assert T.parse_label("<think>is it fraud?</think>\nnot fraud", labels) == "not fraud"
    assert T.parse_label("I think this is not fraud at all", labels) == "not fraud"
    assert T.parse_label("cannot tell", labels) is None


def test_classification_metrics_and_dollar_weighted_recall():
    gold = ["fraud", "fraud", "fraud", "legit", "legit", "legit"]
    pred = ["fraud", "legit", None, "fraud", "legit", "legit"]
    w = [1000.0, 10.0, 90.0, 5.0, 5.0, 5.0]
    m = T.classification_metrics(gold, pred, ["fraud", "legit"], "fraud", w)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(1 / 3)
    # caught 1 of 3 cases but 1000 of 1100 dollars: recall and dollar-weighted recall tell different stories
    assert m["weighted_recall"] == pytest.approx(1000 / 1100)
    assert m["unparsed"] == pytest.approx(1 / 6) and m["confusion"]["fraud"]["<unparsed>"] == 1
    assert m["per_class"]["legit"]["support"] == 3


def test_percentile_and_ece():
    xs = [0.1 * i for i in range(1, 101)]
    assert T.percentile(xs, 50) == pytest.approx(5.0) and T.percentile(xs, 99) == pytest.approx(9.9)
    assert T.percentile([], 50) is None
    assert T.ece([1.0, 1.0], [1, 1]) == 0.0
    assert T.ece([0.9, 0.9], [0, 0]) == pytest.approx(0.9)


def test_decision_metrics_reuse_the_typed_board():
    rows = [{"label": "ap", "gold": {"ap": 0.8, "fraud": 0.2}, "dist": {"ap": 0.8, "fraud": 0.2}, "qtype": "choice"},
            {"label": "0", "gold": {"0": 0.7, "1": 0.3}, "dist": {"0": 0.2, "1": 0.8}, "qtype": "score"}]
    m = T.decision_metrics(rows)
    assert m["accuracy"] == 0.5 and m["set"] == "typed-decisions" and m["kl_from_gold"] > 0 and m["ece"] is not None


# ------------------------------------------------------------------ eval end to end (fake models)
class Fake:
    def __init__(self, name, fn, dt=0.01):
        self.name, self.fn, self.dt = name, fn, dt

    def generate(self, messages):
        return self.fn(messages[-1]["content"]), self.dt

    def option_probs(self, messages, options):
        return {o: 1 / len(options) for o in options}, self.dt


def test_eval_writes_table_report_and_lineage(tmp_path, monkeypatch):
    a = _prepped(tmp_path)
    run = tmp_path / "run"
    merged = run / "m"
    merged.mkdir()
    json.dump({"tool": "pollard-forge", "steps": [{"op": "tune", "from": "tiny/base", "backend": "mlx", "rank": 16,
                                                   "splits": {"train": 1, "valid": 1, "test": 1}}]},
              open(merged / "forge.json", "w"))
    json.dump({"merged": str(merged), "base": "tiny/base"}, open(run / "train.json", "w"))
    base = lambda: Fake("base", lambda u: "legit")
    tuned = lambda: Fake("tuned", lambda u: "fraud" if "yes" in u else "legit", 0.005)
    ep = T.Endpoint("http://127.0.0.1:9", "frontier", post=lambda url, body: {
        "choices": [{"message": {"content": "fraud\nlooks risky"}}]})
    a.compare_url, a.compare_model = "http://127.0.0.1:9", "frontier"
    s = T.evaluate(a, loaders=(base, tuned), compare=ep)
    c = s["columns"]
    assert c["tuned"]["recall"] == 1.0 and c["tuned"]["weighted_recall"] == 1.0
    assert c["base"]["recall"] == 0.0 and c["frontier"]["recall"] == 1.0 and c["frontier"]["precision"] < 1.0
    md = open(run / "eval" / "report.md").read()
    assert md.startswith("## Fine-tune evaluation") and "| recall (fraud) | 0.0% | 100.0% |" in md
    assert "weighted recall (fraud, by amount)" in md and "## How it was tuned" in md
    assert str(tmp_path) not in md                       # a card never carries a local path
    steps = json.load(open(merged / "forge.json"))["steps"]
    assert steps[-1]["op"] == "tune-eval" and steps[-1]["results"]["tuned"]["recall"] == 1.0


def test_eval_refuses_to_send_rows_off_box_without_consent(tmp_path):
    a = _prepped(tmp_path)
    json.dump({"merged": str(tmp_path / "run" / "m"), "base": "x"}, open(tmp_path / "run" / "train.json", "w"))
    a.compare_url, a.compare_model = "https://api.example.com", "frontier"
    with pytest.raises(SystemExit, match="allow-remote"):
        T.evaluate(a, loaders=(lambda: None, lambda: None))
    assert not T._is_remote("http://localhost:8080") and T._is_remote("api.example.com")


def test_endpoint_strips_v1_and_scores_options():
    seen = {}

    def post(url, body):
        seen["url"] = url
        return {"choices": [{"message": {"content": "fraud"}}]}
    ep = T.Endpoint("http://127.0.0.1:8080/v1/", "m", post=post)
    p, _ = ep.option_probs([{"role": "user", "content": "x"}], ["ap", "fraud"])
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions" and p["fraud"] > 0.99


# ------------------------------------------------------------------ ship
def test_ship_preflights_builds_cards_and_hands_off(tmp_path, monkeypatch, capsys):
    a = _prepped(tmp_path)
    run = tmp_path / "run"
    merged = run / "Model-ft"
    merged.mkdir()
    json.dump({"task": "classify"}, open(merged / "agent.json", "w"))
    json.dump({"merged": str(merged), "base": "tiny/base"}, open(run / "train.json", "w"))
    (run / "eval").mkdir()
    (run / "eval" / "report.md").write_text("## Fine-tune evaluation\n")
    cmds = []
    import pollard_workspace as ws
    monkeypatch.setattr(ws, "read_manifest", lambda k: {"builds": [
        {"lane": "gguf", "tag": "Q6_K", "path": "/w/Model-ft-Q6_K.gguf", "bytes": 3},
        {"lane": "gguf", "tag": "Q4_K_M", "path": "/w/Model-ft-Q4_K_M.gguf", "bytes": 2}]})
    a.harness = "claude-code"
    line = T.ship(a, run_cmd=lambda c: cmds.append(c) or 0)
    assert cmds[0][:2] == ["pollard-smoke", "--model"]
    assert cmds[1] == ["pollard", "--hf", str(merged), "--run", "--ladder"]
    card = cmds[2]
    assert card[0] == "pollard-card" and card[card.index("--extra-md") + 1].endswith("report.md")
    assert "Q4_K_M" in line and "claude-code" in line


def test_serve_line_degrades_without_pollard_serve():
    assert T.serve_line("/w/m-Q4_K_M.gguf", "omp", available=True) == "pollard serve /w/m-Q4_K_M.gguf --harness omp"
    fallback = T.serve_line("/w/m-Q4_K_M.gguf", "omp", available=False)
    assert fallback.startswith("llama-server -m /w/m-Q4_K_M.gguf") and "not in this install" in fallback
    assert "--decision" in T.serve_line("/w/x.gguf", "omp", decision=True, available=True)


def test_ship_stops_when_preflight_fails(tmp_path):
    a = _prepped(tmp_path)
    run = tmp_path / "run"
    json.dump({"merged": str(run / "m"), "base": "b"}, open(run / "train.json", "w"))
    with pytest.raises(SystemExit, match="pollard-smoke"):
        T.ship(a, run_cmd=lambda c: 1)


def test_pick_rung_prefers_q4_k_m():
    b = [{"lane": "gguf", "tag": "IQ4_XS", "path": "a"}, {"lane": "gguf", "tag": "Q4_K_M", "path": "b"},
         {"lane": "mlx", "tag": "4bit", "path": "c"}]
    assert T.pick_rung(b)["path"] == "b" and T.pick_rung([]) is None


def test_clef_temperatures(tmp_path):
    assert T.clef_temperatures(str(tmp_path)) == {}
    json.dump({"temperatures": {"choice": 1.3, "noul": 0.9}}, open(tmp_path / "joint_head_config.json", "w"))
    assert T.clef_temperatures(str(tmp_path)) == {"choice": 1.3, "noul": 0.9}


def test_one_shot_argv_routes_to_run(monkeypatch):
    seen = []
    monkeypatch.setattr(T, "prep", lambda a: seen.append("prep"))
    monkeypatch.setattr(T, "train", lambda a: seen.append("train"))
    monkeypatch.setattr(T, "evaluate", lambda a: seen.append("eval"))
    monkeypatch.setattr(T, "ship", lambda a: seen.append("ship"))
    T.main(["--base", "b", "--data", "d.csv", "--out", "r", "--ship"])
    assert seen == ["prep", "train", "eval", "ship"]


def test_forge_card_renders_a_tune_step(tmp_path):
    import pollard_forge as G
    d = tmp_path / "m"
    d.mkdir()
    json.dump({"model_type": "qwen2", "num_hidden_layers": 2}, open(d / "config.json", "w"))
    json.dump({"tool": "pollard-forge", "steps": [
        {"op": "tune", "from": "Qwen/Qwen2.5-0.5B-Instruct", "backend": "mlx", "method": "lora", "rank": 16, "steps": 30,
         "task": "classify", "data": [{"source": "fraud.csv", "weight": 1.0}], "splits": {"train": 80, "valid": 10, "test": 20},
         "val_loss_start": 0.5, "val_loss_end": 0.1},
        {"op": "tune-eval", "n": 20, "results": {"base": {"f1": 0.2}, "tuned": {"f1": 0.9}}}]}, open(d / "forge.json", "w"))
    G.cmd_card(argparse.Namespace(model=str(d), name="M", license="apache-2.0", repo=None, description=None))
    card = open(d / "README.md").read()
    assert "| 1 | tune | LoRA r16 (mlx, classify) on fraud.csv" in card and "Qwen/Qwen2.5-0.5B-Instruct" in card
    assert "| 2 | tune-eval |" in card and "f1" in card


def test_endpoint_merges_provider_fields():
    seen = {}
    ep = T.Endpoint("http://127.0.0.1:11434", "qwen3:8b", extra={"reasoning_effort": "none"},
                    post=lambda url, body: seen.update(body) or {"choices": [{"message": {"content": "legit"}}]})
    assert ep.generate([{"role": "user", "content": "x"}])[0] == "legit"
    assert seen["reasoning_effort"] == "none" and seen["temperature"] == 0


def test_ship_keeps_the_rungs_a_failed_flagship_step_left(tmp_path, monkeypatch):
    """The Mac smoke run: Q6_K built, then the IQ1_KT flagship step died (no ik_llama) and `pollard` exited 1."""
    a = _prepped(tmp_path)
    run = tmp_path / "run"
    json.dump({"merged": str(run / "m"), "base": "b"}, open(run / "train.json", "w"))
    import pollard_workspace as ws
    monkeypatch.setattr(ws, "read_manifest", lambda k: {"builds": [{"lane": "gguf", "tag": "Q6_K", "path": "/w/m-Q6_K.gguf"}]})
    line = T.ship(a, run_cmd=lambda c: 1 if c[0] == "pollard" else 0)
    assert "/w/m-Q6_K.gguf" in line
    monkeypatch.setattr(ws, "read_manifest", lambda k: {"builds": []})
    with pytest.raises(SystemExit, match="built no rung"):
        T.ship(a, run_cmd=lambda c: 1 if c[0] == "pollard" else 0)
