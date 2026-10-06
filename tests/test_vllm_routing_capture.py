"""Routing fixtures need no vLLM server; tensor cases use optional CPU torch."""
import importlib.util
import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "pollard_routing_fixture", Path(__file__).parents[1] / "experiments/vllm_decode_routing_hook.py")
H = importlib.util.module_from_spec(spec)
# Import the helpers without installing hooks into a real vLLM runtime. Some
# test environments have vLLM installed but intentionally expose no GPU.
_import = builtins.__import__


def _without_vllm(name, *args, **kwargs):
    if name == "vllm" or name.startswith("vllm."):
        raise ModuleNotFoundError("vLLM installation is disabled for unit fixtures")
    return _import(name, *args, **kwargs)


with patch("builtins.__import__", side_effect=_without_vllm):
    spec.loader.exec_module(H)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def metadata(torch, starts, seq_lens=None, actual=None):
    result = SimpleNamespace(query_start_loc=torch.tensor(starts))
    if seq_lens is not None:
        result.seq_lens = torch.tensor(seq_lens)
    if actual is not None:
        result.num_actual_tokens = actual
    return result


def test_mixed_batch_new_prompt_decode_and_padding(torch):
    md = metadata(torch, [0, 3, 4, 5], [3, 20, 1], actual=5)
    assert H.phase_mask(md, 7, "cpu").tolist() == [0, 0, 0, 1, 0, -1, -1]


@pytest.mark.parametrize("md", [None, {}, SimpleNamespace(max_query_len=1)])
def test_missing_metadata_is_not_prefill(torch, md):
    assert H.phase_mask(md, 2, "cpu").tolist() == [-1, -1]


def test_single_queries_without_sequence_lengths_are_unknown(torch):
    assert H.phase_mask(metadata(torch, [0, 1, 3]), 3, "cpu").tolist() == [-1, 0, 0]


@pytest.mark.parametrize("starts,seq_lens,actual", [([1, 2], [2], 2), ([0, 3], [2], 3),
                                                   ([0, 2, 1], [2, 2], 1), ([0, 3], [3], 2)])
def test_invalid_metadata_is_unknown(torch, starts, seq_lens, actual):
    assert H.phase_mask(metadata(torch, starts, seq_lens, actual), 4, "cpu").tolist() == [-1] * 4


def test_attention_groups_must_agree(torch):
    a, b = metadata(torch, [0, 1], [1]), metadata(torch, [0, 1], [5])
    assert H.phase_mask({"a": a, "b": b}, 1, "cpu").tolist() == [-1]


def test_hybrid_state_space_metadata_is_not_used_as_attention(torch):
    md = {"attention": metadata(torch, [0, 2, 3], [2, 8]),
          "mamba": SimpleNamespace(num_prefills=1, num_decodes=1)}
    assert H.phase_mask(md, 3, "cpu").tolist() == [0, 0, 1]


def test_flashinfer_uses_its_explicit_decode_first_split(torch):
    md = type("FlashInferMetadata", (), {})()
    md.num_decode_tokens, md.num_prefill_tokens, md.num_actual_tokens = 2, 3, 5
    assert H.phase_mask({"attn": md, "mamba": SimpleNamespace()}, 7, "cpu").tolist() == [1, 1, 0, 0, 0, -1, -1]
    md.num_prefill_tokens = 4
    assert H.phase_mask(md, 7, "cpu").tolist() == [-1] * 7


def test_scheduler_phase_corrects_single_token_prefill_continuation(torch):
    md = metadata(torch, [0, 1, 2, 4], [30, 20, 42], actual=4)
    md.is_prefilling = torch.tensor([True, False, True])
    assert H.phase_mask(md, 5, "cpu").tolist() == [0, 1, 0, 0, -1]
    md.is_prefilling = torch.tensor([True])
    assert H.phase_mask(md, 5, "cpu").tolist() == [-1] * 5


def test_flashinfer_bridge_preserves_request_phase_not_kernel_split(torch):
    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            result = type("FlashInferMetadata", (), {})()
            result.num_actual_tokens = 2
            result.num_decode_tokens = 2  # two one-token queries, both decode kernels
            result.num_prefill_tokens = 0
            return result
    cls, original, installed = H.preserve_flashinfer_scheduler_phase(Builder)
    md = metadata(torch, [0, 1, 2], [30, 20], actual=2)
    md.is_prefilling = torch.tensor([True, False])
    result = Builder().build(0, md)
    assert H.phase_mask(result, 3, "cpu").tolist() == [0, 1, -1]
    assert result.num_decode_tokens == 2  # the serving backend's split is unchanged
    md.is_prefilling = None
    result = Builder().build(common_prefix_len=0, common_attn_metadata=md)
    assert H.phase_mask(result, 2, "cpu").tolist() == [-1, -1]
    assert cls.build is installed
    cls.build = original


def test_expert_width_grows_and_never_shrinks(torch, tmp_path):
    capture = H.Capture(str(tmp_path), every=100)
    router = SimpleNamespace(prefix="model.layers.0.mlp")
    md = metadata(torch, [0, 1], [5])
    for expert in (1, 3, 0):
        capture.record(router, torch.tensor([[1.0]]), torch.tensor([[expert]]), md)
    layer = capture.layers[router.prefix]
    assert layer["experts"] == 4
    assert layer["decode"]["count"] == [1, 1, 0, 1]
    assert layer["decode"]["tokens"] == 3
    assert not list(tmp_path.glob("*.json"))
    capture.flush()
    data = json.loads(Path(capture.path).read_text())
    assert data["meta"]["calls"] == 3
    assert data["layers"] == capture.layers
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_routing_batch_is_harmless(torch, tmp_path):
    capture = H.Capture(str(tmp_path))
    capture.record(SimpleNamespace(), torch.empty((0, 2)), torch.empty((0, 2), dtype=torch.int64), None)
    assert capture.calls == 0 and not capture.layers


def test_counts_only_router_calls_with_both_phases(torch, tmp_path):
    capture = H.Capture(str(tmp_path), every=100)
    router = SimpleNamespace(prefix="layer0", global_num_experts=2)
    mixed = metadata(torch, [0, 2, 3], [2, 8])
    capture.record(router, torch.ones((3, 1)), torch.zeros((3, 1), dtype=torch.int64), mixed)
    capture.record(router, torch.ones((1, 1)), torch.zeros((1, 1), dtype=torch.int64), None)
    assert capture.mixed_router_calls == 1
    capture.flush()
    assert json.loads(Path(capture.path).read_text())["meta"]["mixed_router_calls"] == 1
    capture.reset()
    assert capture.mixed_router_calls == 0


def test_unknown_tokens_are_kept_separate(torch, tmp_path):
    capture = H.Capture(str(tmp_path), every=1)
    capture.record(SimpleNamespace(prefix="layer0", global_num_experts=4),
                   torch.tensor([[0.25, 0.75]]), torch.tensor([[1, 3]]), None)
    layer = capture.layers["layer0"]
    assert layer["prefill"]["tokens"] == layer["decode"]["tokens"] == 0
    assert layer["unknown"]["count"] == [0, 1, 0, 1]
    assert sum(layer["unknown"]["mass"]) == 1.0


def dump(tmp_path, name, count=2):
    data = {"layers": {"layer0": {"experts": 2, "decode": {
        "count": [count, 0], "mass": [float(count), 0], "tokens": count},
        "prefill": {"count": [0, 2], "mass": [0, 2.0], "tokens": 2}}}}
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def test_rank_mode_is_explicit(tmp_path):
    paths = [dump(tmp_path, "a.json"), dump(tmp_path, "b.json")]
    with pytest.raises(ValueError, match="rank-mode"):
        H.analyse(paths)


def test_replicated_ranks_count_once(tmp_path, capsys):
    H.analyse([dump(tmp_path, "a.json"), dump(tmp_path, "b.json")], rank_mode="replicated")
    output = capsys.readouterr().out
    assert "replicated" in output
    assert "100.0%" in output  # top64 covers all experts in this two-expert fixture


def test_replicated_ranks_with_different_boundaries_fail(tmp_path):
    with pytest.raises(ValueError, match="captures differ"):
        H.analyse([dump(tmp_path, "a.json"), dump(tmp_path, "b.json", 3)], rank_mode="replicated")


def test_capture_names_do_not_collide_without_rank(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(H.os, "getpid", lambda: 10)
    a = H.Capture(str(tmp_path))
    monkeypatch.setattr(H.os, "getpid", lambda: 11)
    assert a.path != H.Capture(str(tmp_path)).path


def test_fork_adopts_child_identity_and_discards_inherited_counts(torch, tmp_path, monkeypatch):
    monkeypatch.setattr(H.os, "getpid", lambda: 10)
    capture = H.Capture(str(tmp_path), every=1)
    router = SimpleNamespace(prefix="layer0", num_experts=4)
    capture.record(router, torch.tensor([[1.0]]), torch.tensor([[2]]), None)
    parent = capture.path
    monkeypatch.setattr(H.os, "getpid", lambda: 11)
    capture.record(router, torch.tensor([[1.0]]), torch.tensor([[3]]), None)
    assert capture.path != parent and capture.calls == 1
    assert capture.layers["layer0"]["unknown"]["count"] == [0, 0, 0, 1]
    assert json.loads(Path(parent).read_text())["meta"]["pid"] == 10
    assert json.loads(Path(capture.path).read_text())["meta"]["pid"] == 11


def test_captures_with_recording_errors_are_not_analyzed(tmp_path):
    path = Path(dump(tmp_path, "error.json"))
    data = json.loads(path.read_text())
    data["meta"] = {"errors": 1}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="recording errors"):
        H.analyse([str(path)])


def test_invalid_dump_interval(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        H.Capture(str(tmp_path), every=0)


def test_installed_wrapper_returns_unchanged_output_and_flushes(torch, tmp_path, monkeypatch):
    class Router:
        prefix = "layer0"
        num_experts = 4
        def select_experts(self, weights, ids):
            return weights, ids
    # Exercise the installation path with real CPU tensors and a fake vLLM
    # boundary. Syntax checks alone cannot catch undefined names in _install.
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe", "vllm.model_executor.layers.fused_moe.router",
             "vllm.model_executor.layers.fused_moe.router.fused_moe_router", "vllm.forward_context"]
    modules = {name: ModuleType(name) for name in names}
    modules[names[-2]].FusedMoERouter = Router
    modules[names[-3]].fused_moe_router = modules[names[-2]]
    modules[names[-1]].get_forward_context = lambda: SimpleNamespace(attn_metadata=metadata(torch, [0, 1], [8]))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = ModuleType("vllm.v1.attention.backends.flashinfer")
    class Builder:
        def build(self, common_prefix_len, common_attn_metadata):
            return SimpleNamespace(num_actual_tokens=1)
    backend.FlashInferMetadataBuilder = Builder
    original_build = Builder.build
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setenv("VLLM_ROUTING_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ROUTING_DUMP_EVERY", "1")
    H._install()
    assert Builder.build is not original_build
    wrapped = Router.select_experts
    router = Router()
    model = SimpleNamespace(named_modules=lambda: [("model.layers.3.mlp", SimpleNamespace(router=router))])
    assert H.assign_layer_names(model) == 1
    assert router.layer_name == "model.layers.3.mlp"
    H._install()
    assert Router.select_experts is wrapped
    weights, ids = torch.tensor([[1.0]]), torch.tensor([[2]])
    result = Router().select_experts(weights, ids)
    assert result[0] is weights and result[1] is ids
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["layers"]["layer0"]["decode"]["tokens"] == 1
    extension = H.RoutingWorkerExtension()
    extension.model_runner = SimpleNamespace(get_model=lambda: model)
    assert extension.pollard_start_capture() == {"named_routers": 1}
    assert wrapped._pollard_capture.calls == 0
    assert not list(tmp_path.glob("*.json"))
    router.select_experts(weights, ids)
    assert extension.pollard_flush_capture()["layers"] == 1
    # A failed observation must not fail the original router call.
    Router().select_experts(torch.tensor([[1.0]]), torch.tensor([[-1]]))
    assert wrapped._pollard_capture.errors == 1
    wrapped._pollard_capture.flush()
    H.atexit.unregister(wrapped._pollard_capture.flush)
    assert extension.pollard_remove_capture() == {"removed": True}
    assert Router.select_experts is wrapped._pollard_original
    assert Builder.build is original_build
    with pytest.raises(RuntimeError, match="not installed"):
        extension.pollard_start_capture()
