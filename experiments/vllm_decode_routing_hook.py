#!/usr/bin/env python3
"""vllm_decode_routing_hook.py — capture MoE routing (expert picks + routing-weight mass) per layer, split PREFILL vs DECODE,
from a running vLLM server. Answers the e10 question ("does decode concentrate more than prefill?") on models that only run
on vLLM — the fused MoE path exports nothing per token, so this wraps the router's expert selection.

STATUS: written against vLLM 0.28 (`FusedMoERouter.select_experts` in
`vllm/model_executor/layers/fused_moe/router/fused_moe_router.py`); UNTESTED on a live server at the time of this commit —
verification on a GLM-5.3 TP8 deployment is scheduled; results will be appended to the GLM-5.3 note.

How it works: wraps `select_experts` to record the returned (topk_weights, topk_ids); classifies each token as decode or
prefill from the forward context's attention metadata (`max_query_len == 1` → pure decode batch; otherwise per-request
`query_start_loc` gives the query length of each request and single-token requests are decode); accumulates per-layer
counts and mass; dumps JSON every VLLM_ROUTING_DUMP_EVERY calls (default 500) to VLLM_ROUTING_DUMP_DIR (one file per rank).

IMPORTANT: Python-level hooks do not run inside CUDA-graph replay. Capture with graphs off (`--enforce-eager`, or a
compilation config with cudagraph_mode NONE) — slower, but this is a measurement run, not serving.

Install (no code change to vLLM): put this file and a `sitecustomize.py` containing `import vllm_decode_routing_hook` in a
directory on PYTHONPATH of the vLLM workers, e.g. for docker:
    -v /path/hook:/hook -e PYTHONPATH=/hook -e VLLM_ROUTING_DUMP_DIR=/capture -e VLLM_ROUTING_DUMP_EVERY=500
Then drive the server with real prompts (prefill) that generate a few hundred tokens each (decode). Analyse with:
    python3 vllm_decode_routing_hook.py --analyse /capture/*.json     # n_eff, top-K coverage, decode/prefill ratio per layer
"""
import json, math, os, sys, time


def _install():
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.router import fused_moe_router as fmr
    except Exception as e:  # noqa: BLE001
        print(f"[routing-hook] not installed: {e}", file=sys.stderr); return
    dump_dir = os.environ.get("VLLM_ROUTING_DUMP_DIR", "/tmp/vllm_routing"); os.makedirs(dump_dir, exist_ok=True)
    every = int(os.environ.get("VLLM_ROUTING_DUMP_EVERY", "500"))
    state = {"layers": {}, "order": {}, "calls": 0, "started": time.time()}
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or str(os.getpid())

    def decode_mask(n_tokens, device):
        """bool[n_tokens]: True for decode tokens. Falls back to 'all prefill' if metadata is unavailable."""
        try:
            from vllm.forward_context import get_forward_context
            md = get_forward_context().attn_metadata
            if isinstance(md, dict): md = next(iter(md.values()))
            mql = getattr(md, "max_query_len", None)
            if mql is not None and int(mql) == 1: return torch.ones(n_tokens, dtype=torch.bool, device=device)
            qsl = getattr(md, "query_start_loc", None)
            if qsl is not None:
                qsl = qsl.to("cpu"); lens = (qsl[1:] - qsl[:-1]); m = torch.zeros(n_tokens, dtype=torch.bool)
                for r in range(len(lens)):
                    if int(lens[r]) == 1 and int(qsl[r]) < n_tokens: m[int(qsl[r])] = True
                return m.to(device)
        except Exception:  # noqa: BLE001
            pass
        return torch.zeros(n_tokens, dtype=torch.bool, device=device)

    def record(router, topk_weights, topk_ids):
        key = getattr(router, "layer_name", None) or getattr(router, "prefix", None)
        if key is None:
            key = state["order"].setdefault(id(router), f"router{len(state['order'])}")
        E = int(getattr(router, "global_num_experts", 0) or getattr(router, "num_experts", 0) or int(topk_ids.max()) + 1)
        L = state["layers"].setdefault(key, {"experts": E, "decode": {"count": [0.0] * E, "mass": [0.0] * E, "tokens": 0},
                                              "prefill": {"count": [0.0] * E, "mass": [0.0] * E, "tokens": 0}})
        if topk_ids.shape[1] and topk_ids.max() >= E:  # grow if the expert count guess was low
            for ph in ("decode", "prefill"):
                L[ph]["count"] += [0.0] * (int(topk_ids.max()) + 1 - E); L[ph]["mass"] += [0.0] * (int(topk_ids.max()) + 1 - E)
            L["experts"] = E = int(topk_ids.max()) + 1
        dm = decode_mask(topk_ids.shape[0], topk_ids.device)
        for ph, sel in (("decode", dm), ("prefill", ~dm)):
            if not bool(sel.any()): continue
            ids = topk_ids[sel].reshape(-1).to(torch.int64); w = topk_weights[sel].reshape(-1).to(torch.float32)
            cnt = torch.bincount(ids, minlength=E).to("cpu"); mass = torch.zeros(E, dtype=torch.float32, device=ids.device).index_add_(0, ids, w).to("cpu")
            for e in range(E):
                L[ph]["count"][e] += float(cnt[e]); L[ph]["mass"][e] += float(mass[e])
            L[ph]["tokens"] += int(sel.sum())
        state["calls"] += 1
        if state["calls"] % every == 0:
            tmp = os.path.join(dump_dir, f"routing_rank{rank}.json.tmp")
            json.dump({"meta": {"rank": rank, "calls": state["calls"], "elapsed_s": time.time() - state["started"], "vllm": _vllm_version()}, "layers": state["layers"]}, open(tmp, "w"))
            os.replace(tmp, os.path.join(dump_dir, f"routing_rank{rank}.json"))

    orig = fmr.FusedMoERouter.select_experts

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            tw, ti = out[0], out[1]
            if hasattr(ti, "shape") and ti.ndim == 2: record(self, tw, ti)
        except Exception as e:  # noqa: BLE001 — never break serving
            if state["calls"] == 0: print(f"[routing-hook] record failed: {e}", file=sys.stderr)
        return out

    fmr.FusedMoERouter.select_experts = wrapped
    print(f"[routing-hook] installed; dumping every {every} calls to {dump_dir}", file=sys.stderr)


def _vllm_version():
    try:
        import vllm; return vllm.__version__
    except Exception:  # noqa: BLE001
        return "?"


def analyse(paths, Ks=(8, 16, 32, 64, 128, 192)):
    """Merge rank dumps (same layer keys → sum) and print per-layer n_eff and top-K coverage for decode vs prefill."""
    layers = {}
    for p in paths:
        d = json.load(open(p))
        for k, L in d["layers"].items():
            A = layers.setdefault(k, {"experts": L["experts"], "decode": {"count": [0.0] * L["experts"], "mass": [0.0] * L["experts"], "tokens": 0},
                                     "prefill": {"count": [0.0] * L["experts"], "mass": [0.0] * L["experts"], "tokens": 0}})
            for ph in ("decode", "prefill"):
                for e in range(L["experts"]):
                    A[ph]["count"][e] += L[ph]["count"][e]; A[ph]["mass"][e] += L[ph]["mass"][e]
                A[ph]["tokens"] += L[ph]["tokens"]
    def stats(c):
        tot = sum(c)
        if tot <= 0: return None
        p = sorted((x / tot for x in c), reverse=True); H = -sum(x * math.log(x) for x in p if x > 0)
        return {"n_eff": math.exp(H), "cov": {K: sum(p[:K]) for K in Ks if K <= len(p)}}
    print("layer | decode tokens n_eff top64 top128 | prefill tokens n_eff top64 top128 | n_eff ratio decode/prefill")
    ratios = []
    for k in sorted(layers, key=lambda s: (len(s), s)):
        L = layers[k]; sd, sp_ = stats(L["decode"]["count"]), stats(L["prefill"]["count"])
        f = lambda s, ph: (f"{L[ph]['tokens']:>8} {s['n_eff']:6.1f} {s['cov'].get(64, 0)*100:5.1f}% {s['cov'].get(128, 0)*100:5.1f}%" if s else "        —")
        r = (sd["n_eff"] / sp_["n_eff"]) if sd and sp_ else float("nan"); ratios.append(r) if sd and sp_ else None
        print(f"{k:>28} | {f(sd, 'decode')} | {f(sp_, 'prefill')} | {r:.2f}")
    if ratios: print(f"median n_eff ratio decode/prefill over {len(ratios)} layers: {sorted(ratios)[len(ratios)//2]:.2f}  (<1 = decode concentrates more)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--analyse":
        analyse(sys.argv[2:])
    else:
        print(__doc__)
else:
    _install()
