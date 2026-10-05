#!/usr/bin/env python3
"""Compare routing on original synthetic workloads, not quality or throughput.

Uses the routing-check worker extension. Raw snapshots are kept separately from
the aggregate report, without request text or generated text. Synthetic domain
differences do not establish behavior on real traffic or a quantization benefit.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from vllm_routing_check import check_capture, summarize_outputs

DOMAINS = ("code", "reasoning", "general")
CONTEXTS = {"short": 256, "long": 1536}
KS = (8, 16, 32, 64, 128)


def workload_text(domain, variant, paragraphs):
    """Original, deterministic public fixtures. No copied or private corpus."""
    if domain not in DOMAINS or variant not in range(4) or paragraphs < 1:
        raise ValueError("invalid workload settings")
    sections = []
    for i in range(paragraphs):
        n = i + variant * 101
        if domain == "code":
            sections.append(f"def update_{n}(records, limit={n % 19 + 2}):\n"
                            "    kept = [r for r in records if r['active']]\n"
                            "    return sorted(kept, key=lambda r: r['score'])[:limit]\n"
                            f"# Review function {n} for empty inputs, ties, and mutation.\n")
        elif domain == "reasoning":
            sections.append(f"Warehouse {n} starts with {n % 29 + 14} crates. "
                            f"It receives {n % 11 + 3} deliveries of {n % 7 + 2} crates "
                            f"and sends out {n % 13 + 5} crates. Each crate contains "
                            "six sealed boxes. Track the remaining crates and boxes, "
                            "showing each step and stating any assumptions.\n")
        else:
            sections.append(f"At station {n}, the volunteer archivists catalogued "
                            "letters and photographs from the old coastal railway. "
                            "A visitor asked how the weather affected daily journeys. "
                            "The curator compared passenger diaries with maintenance "
                            "notes and explained why accounts sometimes disagreed.\n")
    instruction = {"code": "Review the code and suggest a safe improvement.",
                   "reasoning": "Explain the arithmetic for the final warehouse.",
                   "general": "Summarize the archive account in plain language."}[domain]
    return "\n".join(sections) + "\n" + instruction


def workload_tokens(tokenizer, domain, target, variant):
    """Build a token-length fixture without depending on a chat template.

    Truncation deliberately makes this a controlled token-stream experiment,
    not a scored instruction-following task. Actual encoded IDs are hashed.
    """
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("target must be a positive token count")
    sections = 1
    while True:
        ids = tokenizer.encode(workload_text(domain, variant, sections), add_special_tokens=False)
        if len(ids) >= target:
            return ids[:target]
        sections *= 2


def normalized(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("expected finite nonnegative routing values")
    total = sum(values)
    if total <= 0:
        raise ValueError("routing histogram is empty")
    return [v / total for v in values]


def coverage(values, indices):
    p = normalized(values)
    return sum(p[i] for i in indices)


def top_indices(values, k):
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def js_bits(left, right):
    if len(left) != len(right):
        raise ValueError("expert widths disagree")
    p, q = normalized(left), normalized(right)
    middle = [(a + b) / 2 for a, b in zip(p, q)]
    return sum(0.5 * v * math.log2(v / m) for dist in (p, q)
               for v, m in zip(dist, middle) if v > 0)


def describe_layers(data):
    results = []
    # Router module names are model identifiers, not host filesystem paths.
    for ordinal, (name, layer) in enumerate(sorted(data["layers"].items())):
        row = {"layer": ordinal, "router_name": name, "experts": layer["experts"]}
        for phase in ("prefill", "decode"):
            h = layer[phase]
            row[phase] = {"tokens": h["tokens"],
                          "count": list(h["count"]), "mass": list(h["mass"]),
                          "selection_coverage": {str(k): coverage(h["count"], top_indices(h["count"], k)) for k in KS},
                          "weight_coverage": {str(k): coverage(h["mass"], top_indices(h["mass"], k)) for k in KS}}
        results.append(row)
    return results


def compare_layers(source, target):
    if source["layers"].keys() != target["layers"].keys():
        raise ValueError("capture layer names disagree")
    comparisons = []
    for ordinal, name in enumerate(sorted(source["layers"])):
        a, b = source["layers"][name], target["layers"][name]
        if a["experts"] != b["experts"]:
            raise ValueError("capture expert widths disagree")
        row = {"layer": ordinal}
        for phase in ("prefill", "decode"):
            row[phase] = {"selection_js_bits": js_bits(a[phase]["count"], b[phase]["count"]),
                          "source_cache_target_selection_coverage": {
                              str(k): coverage(b[phase]["count"], top_indices(a[phase]["count"], k)) for k in KS}}
        comparisons.append(row)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moe-backend")
    parser.add_argument("--batched", action="store_true", help="four concurrent requests instead of serial requests")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if not 2 <= args.max_tokens <= 128:
        parser.error("--max-tokens must be between 2 and 128")
    import vllm
    from vllm import LLM, SamplingParams
    options = dict(model=args.model, trust_remote_code=False, tensor_parallel_size=1,
                   enforce_eager=True, max_model_len=2048, max_num_seqs=4 if args.batched else 1,
                   max_num_batched_tokens=1024, gpu_memory_utilization=0.55,
                   enable_prefix_caching=False, enable_chunked_prefill=True,
                   compilation_config={"mode": 0}, seed=0,
                   kernel_config={"enable_flashinfer_autotune": False},
                   worker_extension_cls="vllm_decode_routing_hook.RoutingWorkerExtension")
    if args.moe_backend:
        options["moe_backend"] = args.moe_backend
    llm = LLM(**options)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    profiles, captured = [], {}
    requests = {}
    for domain, (context, length) in itertools.product(DOMAINS, CONTEXTS.items()):
        key = f"{domain}-{context}"
        ids = [workload_tokens(tokenizer, domain, length, variant) for variant in range(4)]
        requests[key] = [{"prompt_token_ids": x} for x in ids]
    def generate(key):
        if args.batched:
            return llm.generate(requests[key], params)
        return [llm.generate([x], params)[0] for x in requests[key]]
    raw_dir = args.capture_dir / "snapshots"
    raw_dir.mkdir(exist_ok=True)
    for key in requests:
        workers = llm.collective_rpc("pollard_start_capture")
        if any(w["named_routers"] == 0 for w in workers):
            raise ValueError("no named routers")
        result = summarize_outputs(generate(key))
        reports = llm.collective_rpc("pollard_flush_capture")
        result["workers"] = check_capture(args.capture_dir, reports, result)
        if len(reports) != 1:
            raise ValueError("this harness supports one worker; do not merge ranks implicitly")
        data = json.loads((args.capture_dir / reports[0]["file"]).read_text())
        # reset() deletes the worker's previous snapshot, so preserve each profile first.
        (raw_dir / f"{key}.json").write_text(json.dumps(data) + "\n")
        captured[key] = data
        result.update(profile=key, prompt_ids_sha256=hashlib.sha256(
            json.dumps(requests[key], separators=(",", ":")).encode()).hexdigest(),
            mixed_router_calls=data["meta"].get("mixed_router_calls"),
            layers=describe_layers(data))
        profiles.append(result)
        print(json.dumps({"completed": key, "prompt_tokens": result["prompt_tokens"],
                          "decode_tokens": result["expected_decode_tokens"]}), flush=True)
    removal = llm.collective_rpc("pollard_remove_capture")
    controls = [[summarize_outputs(generate(key)) for key in requests] for _ in range(2)]
    summary = {"schema_version": 1, "vllm": vllm.__version__, "synthetic_workloads": True,
               "quality_benchmark": False, "performance_benchmark": False,
               "serial": not args.batched, "tensor_parallel_size": 1,
               "max_num_batched_tokens": 1024, "max_tokens": args.max_tokens,
               "flashinfer_autotune": False, "prefix_caching": False,
               "profiles": profiles, "hook_removal": removal,
               "control_repeats_match": all(a["output_sha256"] == b["output_sha256"] for a, b in zip(*controls)),
               "capture_matches_controls": all(a["output_sha256"] == b["output_sha256"] for run in controls for a, b in zip(profiles, run)),
               "same_engine_controls": controls,
               "comparisons": [{"source": a, "target": b, "layers": compare_layers(captured[a], captured[b])}
                               for a, b in itertools.permutations(requests, 2)]}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
