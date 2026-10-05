"""The live-check accounting and output must be testable without vLLM."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("routing_check", Path(__file__).parents[1] / "experiments/vllm_routing_check.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_last_generated_token_is_not_a_decode_observation():
    outputs = [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(token_ids=ids)])
               for ids in ([3], [4, 5, 6])]
    result = C.summarize_outputs(outputs)
    assert result["prompt_tokens"] == 4 and result["output_tokens"] == 4
    assert result["expected_decode_tokens"] == 2
    assert len(result["output_sha256"]) == 64
    assert "token_ids" not in json.dumps(result)


@pytest.mark.parametrize("failure", [None, "unknown", "errors", "empty", "path"])
def test_capture_checks_counts_and_rejects_bad_reports(tmp_path, failure):
    worker = {"file": "routing_fixture.json", "errors": 0, "calls": 3, "layers": 1}
    data = {"meta": {"errors": 0}, "layers": {"layer": {
        "prefill": {"tokens": 4}, "decode": {"tokens": 2}, "unknown": {"tokens": 0}}}}
    if failure == "unknown": data["layers"]["layer"]["unknown"]["tokens"] = 1
    if failure == "errors": data["meta"]["errors"] = 1
    if failure == "empty": data["layers"] = {}
    if failure == "path": worker["file"] = "../routing_fixture.json"
    (tmp_path / "routing_fixture.json").write_text(json.dumps(data))
    if failure:
        with pytest.raises(ValueError):
            C.check_capture(tmp_path, [worker], {"prompt_tokens": 4, "expected_decode_tokens": 2})
    else:
        report = C.check_capture(tmp_path, [worker], {"prompt_tokens": 4, "expected_decode_tokens": 2})
        assert report[0]["tokens_per_layer"] == {"prefill": 4, "decode": 2, "unknown": 0}
        assert "routing_fixture" not in json.dumps(report)
