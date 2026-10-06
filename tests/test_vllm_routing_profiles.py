"""Workload provenance and routing comparisons do not need a GPU."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

EXPERIMENTS = Path(__file__).parents[1] / "experiments"
spec = importlib.util.spec_from_file_location("routing_profiles", EXPERIMENTS / "vllm_routing_profiles.py")
P = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(EXPERIMENTS))
try:
    spec.loader.exec_module(P)
finally:
    sys.path.pop(0)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


def test_workloads_are_deterministic_distinct_and_bounded():
    streams = [P.workload_tokens(Tokenizer(), d, n, v)
               for d in P.DOMAINS for n in P.CONTEXTS.values() for v in range(4)]
    assert len(streams) == 24
    assert len({tuple(x) for x in streams}) == 24
    assert all(len(x) in P.CONTEXTS.values() for x in streams)
    assert streams[0] == P.workload_tokens(Tokenizer(), "code", 256, 0)


@pytest.mark.parametrize("target", [0, -1, True, 1.5])
def test_rejects_invalid_token_targets(target):
    with pytest.raises(ValueError):
        P.workload_tokens(Tokenizer(), "code", target, 0)


@pytest.mark.parametrize("values", [[], [0, 0], [-1, 2], [float("nan"), 1], [float("inf")]])
def test_rejects_invalid_histograms(values):
    with pytest.raises(ValueError):
        P.normalized(values)


def test_js_has_known_values_and_width_check():
    assert P.js_bits([1, 0], [1, 0]) == 0
    assert P.js_bits([1, 0], [0, 1]) == 1
    assert P.js_bits([1, 2], [2, 1]) == pytest.approx(P.js_bits([2, 1], [1, 2]))
    with pytest.raises(ValueError):
        P.js_bits([1], [1, 2])


def capture(counts):
    return {"layers": {"model.public.layer": {"experts": len(counts), **{
        phase: {"tokens": 4, "count": counts, "mass": counts} for phase in ("prefill", "decode")}}}}


def test_curves_and_transfer_use_source_not_target_top_k():
    # Source puts all demand in expert 0. Target puts it in expert 8.
    source = capture([1] + [0] * 8)
    target = capture([0] * 8 + [1])
    row = P.compare_layers(source, target)[0]
    assert row["decode"]["selection_js_bits"] == 1
    assert row["decode"]["source_cache_target_selection_coverage"]["8"] == 0
    assert row["decode"]["source_cache_target_selection_coverage"]["16"] == 1
    assert P.describe_layers(target)[0]["decode"]["selection_coverage"]["8"] == 1
    assert "model.public" not in json.dumps(row)


def test_rejects_incompatible_layers():
    with pytest.raises(ValueError):
        P.compare_layers(capture([1, 2]), capture([1, 2, 3]))
    with pytest.raises(ValueError):
        P.compare_layers(capture([1]), {"layers": {}})
