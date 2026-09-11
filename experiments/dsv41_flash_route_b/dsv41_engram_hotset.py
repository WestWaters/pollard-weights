#!/usr/bin/env python3
"""dsv41_engram_hotset.py — "Pollard-guess" the Engram hot set: hash a large in-domain corpus with DeepSeek's own NgramHashState (no forward
needed, hashes are a pure function of token n-grams), count row accesses per Engram layer, emit the top-K row ids per layer plus a coverage
report (in-sample and against a held-out file). Runs on a Spark in the vllm image (torch + transformers) with the reference code
(inference/engram.py, model.py for ModelArgs) and encoding/ (DSML renderer) next to it.
usage: python3 dsv41_engram_hotset.py --ref-dir /data/dsv41kit/inference --tokenizer /data/dsv41cap/meta/tok --config /data/dsv41cap/meta/config.json
       --corpus a.jsonl.gz [b.jsonl.gz ...] --held held.jsonl.gz --top 20000000 --out /data/dsv41engram/hotset
"""
import argparse, dataclasses, gzip, json, os, sys, time, importlib
ap = argparse.ArgumentParser()
ap.add_argument("--ref-dir", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--corpus", nargs="+", required=True); ap.add_argument("--held", default=""); ap.add_argument("--top", type=int, default=20_000_000)
ap.add_argument("--max-tokens", type=int, default=0, help="stop after this many corpus tokens (0 = all)"); ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--out", required=True); a = ap.parse_args()
import torch
from safetensors.torch import save_file
sys.path.insert(0, a.ref_dir); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding"))
# the reference model.py imports the TileLang `kernel` module; register a stub so we can import ModelArgs on CPU
import types; sys.modules.setdefault("kernel", types.SimpleNamespace(act_quant=None, fp4_act_quant=None, fp4_gemm=None, fp8_gemm=None, hc_split_sinkhorn=None, sparse_attn=None))
_ip = types.ModuleType("image_processor"); _ip.IMAGE_START, _ip.IMAGE, _ip.IMAGE_NEW_LINE, _ip.IMAGE_END = range(4); sys.modules.setdefault("image_processor", _ip)
_vs = types.ModuleType("vision")
class _NoVision:
    def __init__(self, *a, **k): raise RuntimeError("vision tower unused here")
_vs.Aligner = _NoVision
_vs.ViT = _NoVision
sys.modules.setdefault("vision", _vs)   # same stubs as dsv41_capture.py (model.py imports both; unused here)
dsm = importlib.import_module("model"); dse = importlib.import_module("engram")
from encoding import encode_messages
from transformers import AutoTokenizer
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
cfg = json.load(open(a.config)); margs = dsm.ModelArgs(**{k: v for k, v in cfg.items() if k in {f.name for f in dataclasses.fields(dsm.ModelArgs)}})
layout = dse.EngramLayout.from_args(margs); tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
hs = dse.NgramHashState(dataclasses.replace(margs, max_batch_size=1), layout, tok)
n_layers = len(layout.layer_ids)
_off = hs.offsets if torch.is_tensor(hs.offsets) else torch.as_tensor(hs.offsets); _pr = hs.primes if torch.is_tensor(hs.primes) else torch.as_tensor(hs.primes)
table_rows = int(_off.max().item()) + int(_pr.max().item()) + 1   # upper bound of the last bucket end (offsets per col, primes per n-gram order)
log(f"engram layers {layout.layer_ids}, hash cols {layout.n_hash_cols if hasattr(layout,'n_hash_cols') else '?'}, table rows ~{table_rows/1e6:.0f}M")
def render(rec):
    msgs = [m for m in (rec.get("messages") or []) if isinstance(m, dict) and m.get("role")]
    ref = rec.get("ref_completion")
    if ref and isinstance(ref, str) and ref.strip(): msgs = msgs + [{"role": "assistant", "content": ref}]
    if not msgs: return None
    try:
        text = encode_messages(msgs, thinking_mode="chat", add_default_bos_token=False)
        if not isinstance(text, str): text = text[0]
    except Exception:
        text = "\n".join(f"{m.get('role')}: {m.get('content') or json.dumps(m.get('tool_calls'))}" for m in msgs)
    return tok(text, add_special_tokens=False)["input_ids"]
def hash_file(path, counts=None, limit=0):
    """stream a jsonl.gz, hash each conversation in chunks; accumulate per-layer counts (int32 [table_rows]) or return the flat id list."""
    tot = 0; ids_out = [[] for _ in range(n_layers)]
    for line in gzip.open(path, "rt"):
        try: ids = render(json.loads(line))
        except Exception: continue
        if not ids: continue
        t = torch.tensor(ids, dtype=torch.long); W = min(a.chunk, int(getattr(margs, "max_seq_len", a.chunk))); OV = 8   # NgramHashState caches [B, max_seq_len] from start_pos 0
        for s in range(0, t.numel(), W - OV):
            seg = t[s:s + W].unsqueeze(0)
            if seg.shape[1] <= OV and s > 0: break
            h = hs(seg, 0, None)[0]                          # [L, n_layers, cols]; each window hashed from position 0 (lookback = OV overlap)
            if s > 0: h = h[OV:]                             # drop the overlap positions already counted by the previous window
            for li in range(n_layers):
                v = h[:, li, :].reshape(-1); v = v[v >= 0]
                if counts is not None: counts[li].index_add_(0, v, torch.ones_like(v, dtype=torch.int32))
                else: ids_out[li].append(v)
        tot += t.numel()
        if limit and tot >= limit: break
    return tot, (None if counts is not None else [torch.cat(x) for x in ids_out])
os.makedirs(a.out, exist_ok=True)
counts = [torch.zeros(table_rows, dtype=torch.int32) for _ in range(n_layers)]; total = 0; t0 = time.time()
for p in a.corpus:
    n, _ = hash_file(p, counts, a.max_tokens - total if a.max_tokens else 0); total += n; log(f"{os.path.basename(p)}: {n/1e6:.1f}M tokens (cum {total/1e6:.1f}M, {time.time()-t0:.0f}s)")
    if a.max_tokens and total >= a.max_tokens: break
report = {"corpus_tokens": total, "layers": {}}
held_ids = None
if a.held: _, held_ids = hash_file(a.held); log(f"held-out {os.path.basename(a.held)}: {held_ids[0].numel()/1e6:.1f}M accesses/layer")
for li, L in enumerate(layout.layer_ids):
    c = counts[li]; nz = (c > 0).sum().item(); acc = c.sum().item()
    order = torch.argsort(c, descending=True)[:a.top]; cs = c[order].cumsum(0)
    rep = {"accesses": acc, "unique_rows": nz, "coverage_in_sample": {str(k): round(cs[min(k, cs.numel()) - 1].item() / acc, 4) for k in (1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000) if k <= cs.numel()}}
    if held_ids is not None:
        hv = held_ids[li]; hu, hc = torch.unique(hv, return_counts=True)
        rep["coverage_held_out"] = {}
        for k in (1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000):
            if k > order.numel(): continue
            hot = order[:k].sort().values; hit = torch.isin(hu, hot); rep["coverage_held_out"][str(k)] = round(hc[hit].sum().item() / hv.numel(), 4)
    save_file({"hot_ids": order.to(torch.int64).contiguous(), "counts": c[order].contiguous()}, os.path.join(a.out, f"engram_hot_L{L:02d}.safetensors"))
    report["layers"][str(L)] = rep; log(f"L{L:02d}: {acc/1e6:.1f}M accesses, {nz/1e6:.2f}M unique rows; in-sample {rep['coverage_in_sample']}; held-out {rep.get('coverage_held_out')}")
if n_layers == 2: log(f"layers share ids? identical hot-1M sets: {torch.equal(torch.argsort(counts[0], descending=True)[:1_000_000].sort().values, torch.argsort(counts[1], descending=True)[:1_000_000].sort().values)}")
json.dump(report, open(os.path.join(a.out, "hotset_report.json"), "w"), indent=1); log("HOTSET-DONE")
