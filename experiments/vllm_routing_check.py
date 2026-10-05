#!/usr/bin/env python3
"""Check routing capture on fixed synthetic requests, not model quality or speed.

Run once with the hook installed on worker PYTHONPATH and --capture-dir pointing
to the workers' shared dump directory. Run again without the hook and without
--capture-dir; compare output_sha256 for both batches. No request text, generated
text, model paths, hostnames or worker filenames are included in the summary.
"""
import argparse
import hashlib
import json
from pathlib import Path

PROMPTS = [
    "Explain why a database transaction needs isolation. Give a concrete example.",
    "Write a Python function that merges two sorted lists without changing its inputs.",
    "A gardener has a rectangular plot measuring 12 by 8 metres. Explain how to calculate the perimeter and area.",
    "Tell a short story about a lighthouse keeper repairing a broken radio.",
]


def summarize_outputs(outputs):
    sequences = [list(x.outputs[0].token_ids) for x in outputs]
    if any(not ids for ids in sequences):
        raise ValueError("expected at least one generated token per request")
    return {"requests": len(outputs),
            "prompt_tokens": sum(len(x.prompt_token_ids) for x in outputs),
            "output_tokens": sum(map(len, sequences)),
            # The last generated token is returned, not fed back to the router.
            "expected_decode_tokens": sum(len(ids) - 1 for ids in sequences),
            "output_sha256": hashlib.sha256(json.dumps(sequences, separators=(",", ":")).encode()).hexdigest()}


def check_capture(directory, workers, expected):
    if not workers:
        raise ValueError("no worker reports")
    reports = []
    for worker in workers:
        name = worker["file"]
        if Path(name).name != name or not name.startswith("routing_") or not name.endswith(".json"):
            raise ValueError("expected a routing capture basename")
        data = json.loads((Path(directory) / name).read_text())
        if worker["errors"] or data["meta"]["errors"]:
            raise ValueError("worker reported recording errors")
        if not data["layers"] or len(data["layers"]) != worker["layers"]:
            raise ValueError("worker has missing routing layers")
        phases = {"prefill": expected["prompt_tokens"],
                  "decode": expected["expected_decode_tokens"], "unknown": 0}
        for layer in data["layers"].values():
            if any(layer[phase]["tokens"] != count for phase, count in phases.items()):
                raise ValueError("per-layer token counts disagree with the requests")
        reports.append({"layers": len(data["layers"]), "calls": worker["calls"],
                        "errors": 0, "tokens_per_layer": phases})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture-dir", help="shared worker dump directory; omit for an unhooked control")
    parser.add_argument("--output", type=Path, required=True, help="aggregate JSON summary")
    parser.add_argument("--moe-backend", help="optional runtime-specific backend, e.g. marlin")
    parser.add_argument("--serial", action="store_true", help="one request at a time to control batch shape")
    parser.add_argument("--same-engine-control", action="store_true", help="remove the hook and repeat twice in the same engine")
    args = parser.parse_args()
    if args.same_engine_control and not args.capture_dir:
        parser.error("--same-engine-control requires --capture-dir")
    # Keep vLLM optional for CPU fixtures and --help.
    import vllm
    from vllm import LLM, SamplingParams
    options = dict(model=args.model, trust_remote_code=False, tensor_parallel_size=1,
                   enforce_eager=True, max_model_len=2048, max_num_seqs=1 if args.serial else 4,
                   gpu_memory_utilization=0.55, enable_prefix_caching=False,
                   enable_chunked_prefill=True, compilation_config={"mode": 0}, seed=0,
                   kernel_config={"enable_flashinfer_autotune": False})
    if args.moe_backend:
        options["moe_backend"] = args.moe_backend
    if args.capture_dir:
        options["worker_extension_cls"] = "vllm_decode_routing_hook.RoutingWorkerExtension"
    llm = LLM(**options)
    def generate(max_tokens):
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        if args.serial:
            return [llm.generate([prompt], params)[0] for prompt in PROMPTS]
        return llm.generate(PROMPTS, params)
    batches = []
    for max_tokens in (1, 48):
        if args.capture_dir:
            named = llm.collective_rpc("pollard_start_capture")
            if any(x["named_routers"] == 0 for x in named):
                raise ValueError("no named MoE routers in this model")
        outputs = generate(max_tokens)
        result = summarize_outputs(outputs)
        result["max_tokens"] = max_tokens
        if args.capture_dir:
            result["workers"] = check_capture(args.capture_dir, llm.collective_rpc("pollard_flush_capture"), result)
        batches.append(result)
    summary = {"schema_version": 1, "vllm": vllm.__version__, "capture": bool(args.capture_dir),
               "flashinfer_autotune": False,
               "serial": args.serial,
               "quality_benchmark": False, "performance_benchmark": False, "batches": batches}
    if args.same_engine_control:
        summary["hook_removal"] = llm.collective_rpc("pollard_remove_capture")
        controls = [[summarize_outputs(generate(n)) for n in (1, 48)] for _ in range(2)]
        summary["same_engine_controls"] = controls
        summary["control_repeats_match"] = all(x["output_sha256"] == y["output_sha256"]
                                               for x, y in zip(*controls))
        summary["capture_matches_controls"] = all(x["output_sha256"] == y["output_sha256"]
                                                  for control in controls for x, y in zip(batches, control))
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
