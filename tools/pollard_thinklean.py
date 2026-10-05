#!/usr/bin/env python3
"""pollard-thinklean: a logit-bias preset that trims reasoning-model overthinking.

Lotfi, Kirichenko, Li & Liu (Meta FAIR, arXiv 2606.00206) found that where a quantized reasoning model
disagrees most with its full-precision self, it reaches for hedging tokens ("wait", "but",
"alternatively", "perhaps", "maybe"...). A fixed negative logit bias on those tokens shortened chains
12-23% with accuracy kept or improved, across GPTQ/AWQ/FlatQuant and 5 models; BF16 shortened too.
r/LocalLLaMA reproduced it on Qwen3.5-4B GGUFs at -2 (11-19% fewer tokens, MATH-500 subset).

Token ids are tokenizer-specific, so a preset has to be generated per model. This prints the
llama.cpp flags (and llama-server JSON) for one model:

    pollard-thinklean --model microsoft/FrogNano-4B-2609 --bias -2
    pollard-thinklean --model ./my-model --json preset.json

Only single-token spellings are biased (a multi-token word would bias its first piece, which is
shared with innocent words).

Measure it before anything ships (the card prints only measured numbers):

    pollard-thinklean --model <hub id> --gguf rung.gguf --measure 50 --json thinklean.json

serves the GGUF with llama-server, answers the same MATH-500 problems with and without the preset,
and records generated tokens and exact-match accuracy (last \\boxed{} vs the reference). Pass the JSON
to pollard-card --thinklean to get the card section.
"""
import argparse, json, os, re, socket, subprocess, sys, time, urllib.request

# The paper's hedging family. Each is tried bare, with a leading space and capitalized; only spellings
# that are ONE token in this tokenizer are used.
HEDGES = ["wait", "but", "alternatively", "perhaps", "maybe", "however", "hmm", "actually",
          "reconsider", "recheck", "double-check", "wrong", "mistake", "hold", "though",
          "rethink", "recalculate", "doubt", "unsure", "possibly", "hmmm", "oops"]   # not "verify"/"instead": those start useful checks


def single_token_ids(tok, word):
    ids = set()
    for s in {word, " " + word, word.capitalize(), " " + word.capitalize(), word.upper(), " " + word.upper()}:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


MATH500 = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"


def _boxed(text):
    """Contents of the LAST \\boxed{...}, braces balanced."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1] if depth == 0 else None


def _norm(a):
    a = (a or "").strip().replace(" ", "").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a).replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    return a.removeprefix("$").removesuffix("$")


def _free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def measure(gguf, ids, bias, n, ngl="99", max_tokens=6144, seed=0):
    """Same problems, same seed, with and without the preset: tokens generated and exact-match accuracy."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pollard_calc import find_llama_bin
    server = find_llama_bin("llama-server")
    if not server:
        sys.exit("llama-server not found -- pollard-runtime --update")
    rows = [json.loads(l) for l in urllib.request.urlopen(MATH500, timeout=120).read().decode().splitlines() if l.strip()]
    rows = rows[:: max(1, len(rows) // n)][:n]                       # spread across subjects and levels
    port = _free_port()
    srv = subprocess.Popen([server, "-m", gguf, "--port", str(port), "-ngl", str(ngl), "-c", str(max_tokens + 1024),
                            "--jinja", "-np", "1"] + (["-t", os.environ["POLLARD_THREADS"]] if os.environ.get("POLLARD_THREADS") else []),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read():
                    break
            except Exception:                                              # noqa: BLE001
                time.sleep(1)
        out = {}
        for arm, lb in (("default", []), ("think-lean", [[i, bias] for i in ids])):
            toks, right, done = [], 0, 0
            for k, r in enumerate(rows):
                body = {"messages": [{"role": "user", "content": r["problem"] + "\n\nPut the final answer in \\boxed{}."}],
                        "max_tokens": max_tokens, "temperature": 0, "seed": seed, "logit_bias": lb}
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                d = json.load(urllib.request.urlopen(req, timeout=3600))
                msg = d["choices"][0]["message"]
                text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
                toks.append(d["usage"]["completion_tokens"])
                right += _norm(_boxed(text)) == _norm(r["answer"])
                done += 1
                print(f"   {arm:10s} {k + 1:3d}/{len(rows)}  tokens {toks[-1]:5d}  acc {right}/{done}", flush=True)
            out[arm] = {"n": done, "accuracy": round(right / done, 4), "mean_tokens": round(sum(toks) / len(toks), 1),
                        "median_tokens": sorted(toks)[len(toks) // 2]}
        d0, d1 = out["default"], out["think-lean"]
        out["token_change_pct"] = round((d1["mean_tokens"] - d0["mean_tokens"]) / d0["mean_tokens"] * 100, 1)
        out["accuracy_change_pts"] = round((d1["accuracy"] - d0["accuracy"]) * 100, 1)
        out["gguf"] = os.path.basename(gguf)
        out["bench"] = f"MATH-500, {len(rows)} problems, greedy, max {max_tokens} tokens"
        return out
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="hub id or local checkout (tokenizer only is loaded)")
    ap.add_argument("--bias", type=float, default=-2.0, help="logit bias per token (paper swept -0.5..-4)")
    ap.add_argument("--words", help="comma list to override the hedge list")
    ap.add_argument("--json", help="also write the preset (and any measurement) here")
    ap.add_argument("--gguf", help="with --measure: the quantized file to test")
    ap.add_argument("--measure", type=int, metavar="N", help="answer N MATH-500 problems with and without the preset")
    ap.add_argument("--ngl", default="99", help="with --measure: GPU layers (0 on a shared GPU)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    words = [w.strip() for w in a.words.split(",")] if a.words else HEDGES
    table = {w: single_token_ids(tok, w) for w in words}
    ids = sorted({i for v in table.values() for i in v})
    if not ids:
        sys.exit("no single-token hedges in this tokenizer -- nothing to bias")
    b = int(a.bias) if float(a.bias).is_integer() else a.bias
    flags = " ".join(f"--logit-bias {i}{'+' if b >= 0 else ''}{b}" for i in ids)
    preset = {"model": a.model, "bias": b, "token_ids": ids, "by_word": table,
              "llama_cpp_flags": flags, "llama_server": {"logit_bias": [[i, b] for i in ids]},
              "source": "arXiv 2606.00206 (Meta FAIR): quantized reasoning models overthink; bias hedges"}
    print(f"# think-lean preset for {a.model}: {len(ids)} token ids, bias {b}")
    for w, v in table.items():
        if v:
            print(f"#   {w:14s} {v}")
    print(flags)
    if a.measure:
        if not a.gguf:
            sys.exit("--measure needs --gguf")
        preset["measured"] = measure(a.gguf, ids, b, a.measure, a.ngl, a.max_tokens)
        m = preset["measured"]
        print(f"# measured on {m['bench']}: tokens {m['token_change_pct']:+.1f}%, accuracy {m['accuracy_change_pts']:+.1f} pts "
              f"({m['default']['accuracy']:.0%} -> {m['think-lean']['accuracy']:.0%})")
    if a.json:
        json.dump(preset, open(a.json, "w"), indent=1)
        print(f"# written {a.json}")


if __name__ == "__main__":
    main()
