#!/usr/bin/env python3
"""dsv41_build_calib.py — in-domain calibration rows for DeepSeek-V4.1-Flash (variant of build_calib.py: DSML rendering via encoding.py) (run inside the brain
image: needs transformers for the chat template + tokenizer).

Sources: replayed production chat payloads (OpenAI-style message lists + the reference
completion), rendered through the model's own chat template so special tokens, role
markers and tool-call syntax appear exactly as in serving. Streams are concatenated with
EOS and cut into fixed rows (default 2048 tokens). Deterministic sampling (seed) with a
per-source quota so agentic/tool-heavy, prose and tool-call-heavy traffic are all covered
(MoE calibration needs DIVERSE routing; a narrow corpus leaves experts uncovered).

  build_calib.py --tokenizer DIR --out calib.safetensors --rows 384 --cols 2048 \
      --src general:0.55:replay_general.jsonl.gz --src prose:0.20:replay_prose.jsonl.gz \
      --src tools:0.25:replay3_general.jsonl.gz
Output: safetensors {"input_ids": int32 [rows, cols]} + a JSON sidecar with provenance counts.
"""
import argparse, gzip, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding"))
from encoding import encode_messages  # DeepSeek-V4.1 DSML renderer (official, from the HF repo)
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rows", type=int, default=384)
ap.add_argument("--cols", type=int, default=2048)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--src", action="append", required=True, help="name:weight:path.jsonl.gz")
ap.add_argument("--max-payload-tokens", type=int, default=6144, help="truncate very long sessions (keeps the tail)")
ap.add_argument("--exclude", action="append", default=[], help="held-out jsonl.gz whose texts must NOT enter calibration (matched by ref_completion / last user message hash)")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
eos = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
rng = random.Random(a.seed)
need_total = a.rows * a.cols

def render(rec):
    """DeepSeek-V4.1 has no Jinja chat template; render through DeepSeek's own encoding.py (DSML, thinking_mode="chat" =
    the non-thinking serving mode we bench with), then tokenize. Falls back to role: content lines if the encoder rejects a record."""
    msgs = rec.get("messages") or []
    ref = rec.get("ref_completion")
    msgs = [m for m in msgs if isinstance(m, dict) and m.get("role")]
    if ref and isinstance(ref, str) and ref.strip():
        msgs = msgs + [{"role": "assistant", "content": ref}]
    if not msgs:
        return None
    try:
        text = encode_messages(msgs, thinking_mode="chat", add_default_bos_token=False)
        if not isinstance(text, str): text = text[0]
    except Exception:
        text = "\n".join(f"{m.get('role')}: {m.get('content') or json.dumps(m.get('tool_calls'))}" for m in msgs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > a.max_payload_tokens:
        ids = ids[-a.max_payload_tokens:]
    return ids

import hashlib
def _h(x): return hashlib.md5(x.encode()).hexdigest()
def _keys(d):
    ks = set()
    r = d.get("ref_completion") or ""
    if r: ks.add("r:" + _h(r[:2000]))
    us = [m for m in (d.get("messages") or []) if isinstance(m, dict) and m.get("role") == "user"]
    if us: ks.add("u:" + _h(json.dumps(us[-1:])))
    return ks
excl = set()
for path in a.exclude:
    for l in gzip.open(path, "rt"):
        excl |= _keys(json.loads(l))
print(f"exclusion keys from held-out: {len(excl)}", flush=True)
streams, prov = {}, {}
for spec in a.src:
    name, w, path = spec.split(":", 2)
    quota = int(need_total * float(w) * 1.15)          # over-provision, trimmed at row cut
    recs = [json.loads(l) for l in gzip.open(path, "rt")]
    rng.shuffle(recs)
    buf, used = [], 0
    skipped = 0
    for r in recs:
        if excl and (_keys(r) & excl):
            skipped += 1; continue
        ids = render(r)
        if not ids:
            continue
        buf.extend(ids + [eos]); used += 1
        if len(buf) >= quota:
            break
    streams[name] = buf[:quota]
    prov[name] = {"records_used": used, "tokens": len(streams[name]), "weight": float(w), "excluded_heldout_matches": skipped}
    print(f"{name}: {used} records -> {len(streams[name])} tokens (quota {quota}); excluded {skipped} held-out matches", flush=True)

# interleave sources in proportion so every row mix is diverse, then cut rows
order = []
for name, s in streams.items():
    n_rows = max(1, round(a.rows * prov[name]["weight"]))
    order += [name] * n_rows
rng.shuffle(order)
rows, pos = [], {k: 0 for k in streams}
for name in order:
    s = streams[name]; p = pos[name]
    if p + a.cols > len(s):
        continue
    rows.append(s[p:p + a.cols]); pos[name] = p + a.cols
    if len(rows) == a.rows:
        break
assert len(rows) == a.rows, f"only {len(rows)} rows available; lower --rows or add sources"
ids = torch.tensor(rows, dtype=torch.int32)
save_file({"input_ids": ids}, a.out)
json.dump({"rows": a.rows, "cols": a.cols, "seed": a.seed, "eos": eos, "sources": prov,
           "row_source_counts": {k: order[:len(rows)].count(k) for k in streams}},
          open(a.out + ".json", "w"), indent=1)
print(f"wrote {a.out}: {tuple(ids.shape)} tokens={ids.numel()} unique={ids.unique().numel()}")
