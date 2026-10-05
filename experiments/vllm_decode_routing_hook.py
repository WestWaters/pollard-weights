#!/usr/bin/env python3
"""vllm_decode_routing_hook.py — capture MoE routing (expert picks + routing-weight mass) per layer, split PREFILL vs DECODE,
from a running vLLM server. Answers the e10 question ("does decode concentrate more than prefill?") on models that only run
on vLLM — the fused MoE path exports nothing per token, so this wraps the router's expert selection.

STATUS: CPU fixtures and a single-GB10 generation check on a Nemotron-H MoE
using vLLM 0.29.1rc1.dev513+g0961bbae2 with FlashInfer attention. This is not
validation of GLM-5.3 TP8, expert parallelism, or arbitrary vLLM versions.

How it works: wraps `select_experts` to record the returned (topk_weights, topk_ids).
Scheduler flags distinguish actual prefill from decode where available. The
FlashInfer bridge preserves these before they become kernel-dispatch counts.
Other metadata falls back to query boundaries and sequence lengths; missing
metadata/padding stays unknown. Disable speculation,
and validate controlled request schedules before drawing conclusions. Accumulates
counts and mass; dumps JSON every VLLM_ROUTING_DUMP_EVERY calls (default 500) to
VLLM_ROUTING_DUMP_DIR (one file per process). See notes/vllm-routing-capture.md.

IMPORTANT: Python-level hooks do not run inside CUDA-graph replay. Capture with graphs off (`--enforce-eager`, or a
compilation config with cudagraph_mode NONE) — slower, but this is a measurement run, not serving.

Install (no code change to vLLM): put this file and a `sitecustomize.py` containing `import vllm_decode_routing_hook` in a
directory on PYTHONPATH of the vLLM workers, e.g. for docker:
    -v /path/hook:/hook -e PYTHONPATH=/hook -e VLLM_ROUTING_DUMP_DIR=/capture -e VLLM_ROUTING_DUMP_EVERY=500
Then drive the server with real prompts (prefill) that generate a few hundred tokens each (decode). Analyse with:
    python3 vllm_decode_routing_hook.py --analyse /capture/*.json --rank-mode replicated
"""
import atexit, json, math, os, sys, time, uuid


def phase_mask(metadata, n_tokens, device):
    """0=prefill, 1=single-token decode, -1=unknown/padding.

    Query length and backend kernel splits are heuristics, not scheduler phase
    flags. Preserve explicit is_prefilling flags where available. Disable speculative
    decoding for this capture: multi-token verification is not a prefill.
    Missing or inconsistent metadata must not inflate the prefill histogram.
    """
    import torch
    unknown = torch.full((n_tokens,), -1, dtype=torch.int8, device=device)
    if isinstance(metadata, dict):
        # Hybrid models also carry state-space metadata, whose request ordering
        # is backend-specific. Use attention groups with full query boundaries.
        groups = [m for m in metadata.values() if getattr(m, "query_start_loc", None) is not None
                  or type(m).__name__ == "FlashInferMetadata"]
        masks = [phase_mask(m, n_tokens, device) for m in groups]
        if not masks or any(not torch.equal(masks[0], m) for m in masks[1:]):
            return unknown
        return masks[0]
    try:
        scheduler_mask = getattr(metadata, "_pollard_phase_mask", None)
        if scheduler_mask is not None:
            actual = int(metadata.num_actual_tokens)
            if scheduler_mask.ndim != 1 or scheduler_mask.numel() != actual or actual > n_tokens:
                return unknown
            if not bool(((scheduler_mask == -1) | (scheduler_mask == 0) | (scheduler_mask == 1)).all()):
                return unknown
            mask = unknown.clone()
            mask[:actual] = scheduler_mask.to(device)
            return mask
        if type(metadata).__name__ == "FlashInferMetadata":
            # Legacy fallback without the metadata bridge: this is a KERNEL
            # dispatch split. A one-token prefill continuation can appear here
            # as decode. Do not treat this as validated scheduler accounting.
            decode = int(metadata.num_decode_tokens)
            prefill = int(metadata.num_prefill_tokens)
            actual = int(metadata.num_actual_tokens)
            if min(decode, prefill) < 0 or decode + prefill != actual or actual > n_tokens:
                return unknown
            mask = unknown.clone()
            mask[:decode] = 1
            mask[decode:actual] = 0
            return mask
        starts = metadata.query_start_loc.to("cpu").tolist()
        actual = int(getattr(metadata, "num_actual_tokens", starts[-1]))
        if not starts or starts[0] != 0 or starts[-1] != actual or actual > n_tokens:
            return unknown
        seq_lens = getattr(metadata, "seq_lens", None)
        seq_lens = seq_lens.to("cpu").tolist() if seq_lens is not None else None
        prefilling = getattr(metadata, "is_prefilling", None)
        if prefilling is not None:
            if prefilling.dtype != torch.bool or prefilling.ndim != 1 or prefilling.numel() != len(starts) - 1:
                return unknown
            prefilling = prefilling.to("cpu").tolist()
        if seq_lens is not None and len(seq_lens) < len(starts) - 1:
            return unknown
        mask = unknown.to("cpu").clone()
        for r, (start, stop) in enumerate(zip(starts, starts[1:])):
            if stop <= start:
                return unknown
            size = stop - start
            # A one-token NEW prompt is not a decode. Without sequence lengths
            # a one-token query cannot be classified safely.
            if seq_lens is not None and seq_lens[r] < size:
                return unknown
            phase = (int(not prefilling[r]) if prefilling is not None else
                     (0 if size > 1 else (-1 if seq_lens is None else int(seq_lens[r] > size))))
            mask[start:stop] = phase
        return mask.to(device)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return unknown


class Capture:
    """Per-process router observations; no generated text or request IDs."""
    def __init__(self, dump_dir, every=500):
        if every <= 0:
            raise ValueError("VLLM_ROUTING_DUMP_EVERY must be positive")
        self.dump_dir, self.every = dump_dir, every
        self.layers, self.order = {}, {}
        self.calls, self.errors = 0, 0
        self.mixed_router_calls = 0
        self.started = time.time()
        self._set_identity()

    def _set_identity(self):
        pid = os.getpid()
        if getattr(self, "pid", None) != pid:
            self.instance = uuid.uuid4().hex
        self.pid = pid
        self.rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        # RANK can be absent or identical on several workers. Never overwrite
        # another worker's capture, even when processes share a dump directory.
        self.path = os.path.join(self.dump_dir, f"routing_rank{self.rank or 'unknown'}_pid{self.pid}_{self.instance}.json")

    def reset(self):
        """Start a measured interval after model initialization/warmup."""
        self._set_identity()
        self.layers.clear()
        self.order.clear()
        self.calls = self.errors = 0
        self.mixed_router_calls = 0
        self.started = time.time()
        if os.path.exists(self.path):
            os.unlink(self.path)  # only this capture's old cumulative snapshot

    def record(self, router, weights, ids, metadata):
        import torch
        if os.getpid() != self.pid:
            # sitecustomize may run before vLLM forks. Inherited state/path must
            # not cause several workers to overwrite the parent's file.
            self.reset()
        if ids.ndim != 2 or weights.shape != ids.shape:
            raise ValueError("routing ids and weights must have the same 2D shape")
        if not ids.numel():
            return
        if int(ids.min()) < 0 or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("invalid expert ids or routing weights")
        key = getattr(router, "layer_name", None) or getattr(router, "prefix", None)
        if key is None:
            key = self.order.setdefault(id(router), f"router{len(self.order)}")
        experts = max(int(getattr(router, "global_num_experts", 0) or
                          getattr(router, "num_experts", 0) or 0), int(ids.max()) + 1)
        layer = self.layers.setdefault(key, {"experts": experts, **{
            ph: {"count": [0.0] * experts, "mass": [0.0] * experts, "tokens": 0}
            for ph in ("decode", "prefill", "unknown")}})
        # Use the stored width, not this call's inferred maximum. A later batch
        # can visit previously unseen experts when the router has no E attribute.
        experts = max(experts, layer["experts"])
        for ph in ("decode", "prefill", "unknown"):
            for field in ("count", "mass"):
                layer[ph][field].extend([0.0] * (experts - len(layer[ph][field])))
        layer["experts"] = experts
        phases = phase_mask(metadata, ids.shape[0], ids.device)
        observed = set()
        for ph, value in (("decode", 1), ("prefill", 0), ("unknown", -1)):
            select = phases == value
            if not bool(select.any()):
                continue
            observed.add(ph)
            chosen = ids[select].reshape(-1).to(torch.int64)
            w = weights[select].reshape(-1).to(torch.float32)
            count = torch.bincount(chosen, minlength=experts).to("cpu").tolist()
            mass = torch.zeros(experts, dtype=torch.float32, device=chosen.device).index_add_(0, chosen, w).to("cpu").tolist()
            for e in range(experts):
                layer[ph]["count"][e] += count[e]
                layer[ph]["mass"][e] += mass[e]
            layer[ph]["tokens"] += int(select.sum())
        if {"decode", "prefill"} <= observed:
            self.mixed_router_calls += 1
        self.calls += 1
        if self.calls % self.every == 0:
            self.flush()

    def flush(self):
        if os.getpid() != self.pid:
            return  # an unused fork must not write inherited observations
        if not self.calls and not self.errors:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        data = {"meta": {"rank": self.rank, "pid": os.getpid(), "calls": self.calls,
                         "mixed_router_calls": self.mixed_router_calls,
                         "errors": self.errors, "elapsed_s": time.time() - self.started,
                         "vllm": _vllm_version(), "phase_method": "scheduler-flag-or-backend-query-heuristic"},
                "layers": self.layers}
        with open(self.path + ".tmp", "w") as f:
            json.dump(data, f, allow_nan=False)
        os.replace(self.path + ".tmp", self.path)


def assign_layer_names(model):
    """Call on each worker's model before capture, so rank keys are stable."""
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    count = 0
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        if isinstance(router, FusedMoERouter):
            router.layer_name = name
            count += 1
    return count


class RoutingWorkerExtension:
    """Named collective RPC methods; no insecure callable serialization needed.

    Pass worker_extension_cls='vllm_decode_routing_hook.RoutingWorkerExtension'
    to LLM, or the corresponding --worker-extension-cls option to vllm serve.
    """
    def pollard_start_capture(self):
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
        count = assign_layer_names(self.model_runner.get_model())
        capture = getattr(FusedMoERouter.select_experts, "_pollard_capture", None)
        if capture is None:
            raise RuntimeError("routing hook is not installed in this worker")
        capture.reset()
        return {"named_routers": count}

    def pollard_flush_capture(self):
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
        capture = getattr(FusedMoERouter.select_experts, "_pollard_capture", None)
        if capture is None:
            raise RuntimeError("routing hook is not installed in this worker")
        capture.flush()
        return {"calls": capture.calls, "errors": capture.errors, "layers": len(capture.layers),
                "file": os.path.basename(capture.path)}

    def pollard_remove_capture(self):
        """Flush and restore the original router for a same-engine control."""
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
        wrapped = FusedMoERouter.select_experts
        if not hasattr(wrapped, "_pollard_original"):
            raise RuntimeError("routing hook is not installed in this worker")
        wrapped._pollard_capture.flush()
        FusedMoERouter.select_experts = wrapped._pollard_original
        bridge = getattr(wrapped, "_pollard_builder_bridge", None)
        if bridge is not None:
            cls, original, installed = bridge
            if cls.build is installed:
                cls.build = original
        return {"removed": True}


def preserve_flashinfer_scheduler_phase(builder_cls, on_error=None):
    """Preserve scheduler phase before FlashInfer reduces it to a kernel split.

    A one-token prefill continuation can use a decode kernel. Backend token
    counts therefore do not establish request phase. This bridge requires the
    common metadata's explicit is_prefilling flags; otherwise it marks unknown.
    """
    import torch
    original = builder_cls.build

    def build(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        common = kwargs.get("common_attn_metadata")
        if common is None and len(args) > 1:
            common = args[1]  # build(common_prefix_len, common_attn_metadata, ...)
        try:
            actual = int(result.num_actual_tokens)
            mask = torch.full((actual,), -1, dtype=torch.int8)
            if getattr(common, "is_prefilling", None) is not None:
                mask = phase_mask(common, actual, "cpu")
            result._pollard_phase_mask = mask
        except Exception:  # Recording must not alter the backend's return path.
            if on_error is not None:
                on_error()
        return result

    builder_cls.build = build
    return builder_cls, original, build


def _install():
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.router import fused_moe_router as fmr
    except Exception as e:  # noqa: BLE001
        print(f"[routing-hook] not installed: {e}", file=sys.stderr); return
    capture = Capture(os.environ.get("VLLM_ROUTING_DUMP_DIR", "/tmp/vllm_routing"),
                      int(os.environ.get("VLLM_ROUTING_DUMP_EVERY", "500")))
    if getattr(fmr.FusedMoERouter.select_experts, "_pollard_capture", False):
        return
    # Save short runs below the periodic interval on normal process exit. SIGKILL
    # cannot flush; use a small dump interval for interrupted experiments.
    atexit.register(capture.flush)

    orig = fmr.FusedMoERouter.select_experts
    bridge = None
    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
        def bridge_error():
            capture.errors += 1
        bridge = preserve_flashinfer_scheduler_phase(FlashInferMetadataBuilder, bridge_error)
    except ImportError:
        pass  # Other backends still use their own query metadata.

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            tw, ti = out[0], out[1]
            from vllm.forward_context import get_forward_context
            try:
                md = get_forward_context().attn_metadata
            except Exception:
                md = None
            capture.record(self, tw, ti, md)
        except Exception as e:  # noqa: BLE001 — never break serving
            capture.errors += 1
            if capture.errors == 1: print(f"[routing-hook] record failed: {e}", file=sys.stderr)
        return out

    wrapped._pollard_capture = capture
    wrapped._pollard_original = orig
    wrapped._pollard_builder_bridge = bridge
    fmr.FusedMoERouter.select_experts = wrapped
    print(f"[routing-hook] installed; dumping every {capture.every} calls to {capture.dump_dir}", file=sys.stderr)


def _vllm_version():
    try:
        import vllm; return vllm.__version__
    except Exception:  # noqa: BLE001
        return "?"


def analyse(paths, Ks=(8, 16, 32, 64, 128, 192), rank_mode=None):
    """Merge disjoint rank observations, or select one verified TP replica."""
    if len(paths) > 1 and rank_mode not in ("replicated", "disjoint"):
        raise ValueError("multiple captures: choose --rank-mode replicated (TP replicas) or disjoint (different tokens)")
    captures = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        if data.get("meta", {}).get("errors", 0):
            raise ValueError("capture has recording errors; inspect the private worker log before analysis")
        if not data.get("layers"):
            raise ValueError("capture has no routing observations")
        captures.append(data)
    if rank_mode == "replicated" and len(captures) > 1:
        # Count replicated TP router observations once. Do not guess a rank
        # layout from identical-looking distribution shapes alone.
        first = captures[0]["layers"]
        for other in captures[1:]:
            if set(first) != set(other["layers"]):
                raise ValueError("replicated captures have different layer keys")
            for key, layer in first.items():
                candidate = other["layers"][key]
                if layer["experts"] != candidate["experts"]:
                    raise ValueError("replicated captures have different expert counts")
                for ph in ("decode", "prefill", "unknown"):
                    left, right = layer.get(ph), candidate.get(ph)
                    if left is None and right is None:
                        continue
                    if left is None or right is None or left["tokens"] != right["tokens"] or left["count"] != right["count"]:
                        raise ValueError("replicated captures differ; check capture boundaries and rank layout")
                    if len(left["mass"]) != len(right["mass"]) or any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
                                                                  for a, b in zip(left["mass"], right["mass"])):
                        raise ValueError("replicated captures have different routing mass")
        captures = captures[:1]
    layers = {}
    for d in captures:
        for k, L in d["layers"].items():
            A = layers.setdefault(k, {"experts": L["experts"], "decode": {"count": [0.0] * L["experts"], "mass": [0.0] * L["experts"], "tokens": 0},
                                     "prefill": {"count": [0.0] * L["experts"], "mass": [0.0] * L["experts"], "tokens": 0},
                                     "unknown": {"count": [0.0] * L["experts"], "mass": [0.0] * L["experts"], "tokens": 0}})
            if A["experts"] != L["experts"]:
                raise ValueError(f"expert count differs for {k}; do not merge incompatible captures")
            for ph in ("decode", "prefill", "unknown"):
                if ph not in L:
                    continue
                for e in range(L["experts"]):
                    A[ph]["count"][e] += L[ph]["count"][e]; A[ph]["mass"][e] += L[ph]["mass"][e]
                A[ph]["tokens"] += L[ph]["tokens"]
    def stats(c):
        tot = sum(c)
        if tot <= 0: return None
        p = sorted((x / tot for x in c), reverse=True); H = -sum(x * math.log(x) for x in p if x > 0)
        return {"n_eff": math.exp(H), "cov": {K: sum(p[:K]) for K in Ks}}
    print("layer | decode tokens n_eff top64 top128 | prefill tokens n_eff top64 top128 | n_eff ratio decode/prefill")
    print(f"rank mode: {rank_mode or 'single'}; token totals are router observations, not request totals")
    ratios = []
    for k in sorted(layers, key=lambda s: (len(s), s)):
        L = layers[k]; sd, sp_ = stats(L["decode"]["count"]), stats(L["prefill"]["count"])
        f = lambda s, ph: (f"{L[ph]['tokens']:>8} {s['n_eff']:6.1f} {s['cov'].get(64, 0)*100:5.1f}% {s['cov'].get(128, 0)*100:5.1f}%" if s else "        —")
        r = (sd["n_eff"] / sp_["n_eff"]) if sd and sp_ else float("nan"); ratios.append(r) if sd and sp_ else None
        print(f"{k:>28} | {f(sd, 'decode')} | {f(sp_, 'prefill')} | {r:.2f}")
        if L["unknown"]["tokens"]:
            print(f"  {k}: {L['unknown']['tokens']} unclassified/padded observations excluded from phase comparison")
    if ratios: print(f"median n_eff ratio decode/prefill over {len(ratios)} layers: {sorted(ratios)[len(ratios)//2]:.2f}  (<1 = decode concentrates more)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--analyse":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--analyse", nargs="+", required=True)
        parser.add_argument("--rank-mode", choices=("replicated", "disjoint"))
        args = parser.parse_args()
        analyse(args.analyse, rank_mode=args.rank_mode)
    else:
        print(__doc__)
else:
    _install()
