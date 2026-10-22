"""Cards serve MIXED-precision checkpoints with `vllm serve <dir> --linear-backend humming` (vLLM >= 0.32.0).

vLLM's Humming kernels (vllm-project/humming, in vLLM 0.32.0) run a per-layer mix when the checkpoint loads through
its own quant method and `--linear-backend humming` routes the linears there. `--quantization humming` is a different
thing: it reads only the first compressed-tensors config group that targets Linear, or Quark's global_quant_config --
one scheme for every layer of a mix. So: a GPTQ `dynamic` mix, a Quark per-module `layer_quant_config` (pollard-mx
--mix / MXFP6), a multi-group compressed-tensors rung get the flag; uniform rungs keep the plain line; no card ever
writes `--quantization humming`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402
import pollard_mxfp6 as M6  # noqa: E402

FLAG = "--linear-backend humming"


def _rung(tmp_path, name, qcfg, sidecar=None):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"quantization_config": qcfg}))
    if sidecar is not None:
        (d / M6.SIDECAR).write_text(json.dumps(sidecar))
    return str(d)


def test_which_checkpoints_are_mixed(tmp_path):
    gptq_mix = _rung(tmp_path, "g4", {"quant_method": "gptq", "bits": 4,
                                      "dynamic": {r"+:^model\.layers\.0\.mlp\.down_proj$": {"bits": 8}}})
    gptq_uni = _rung(tmp_path, "g8", {"quant_method": "gptq", "bits": 8, "dynamic": {r"-:^lm_head$": {}}})
    quark = _rung(tmp_path, "qk", {"quant_method": "quark", "global_quant_config": {},
                                   "layer_quant_config": {"*layers.0.mlp.down_proj": {}}})
    ct_mix = _rung(tmp_path, "ct2", {"quant_method": "compressed-tensors", "format": "mixed-precision",
                                     "config_groups": {"group_0": {}, "group_1": {}}})
    ct_uni = _rung(tmp_path, "ct1", {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                     "config_groups": {"group_0": {"targets": ["Linear"]}}})
    mx6 = _rung(tmp_path, "mx6", {"quant_method": "compressed-tensors", "config_groups": {"group_0": {}}},
                {"format": "mxfp6-e2m3", "layout": "compressed-tensors"})
    assert C.checkpoint_is_mixed(gptq_mix) and C.checkpoint_is_mixed(quark) and C.checkpoint_is_mixed(ct_mix)
    assert C.checkpoint_is_mixed(mx6)
    assert not C.checkpoint_is_mixed(gptq_uni) and not C.checkpoint_is_mixed(ct_uni)
    assert not C.checkpoint_is_mixed(str(tmp_path / "missing")) and not C.checkpoint_is_mixed(None)


def test_gptq_lane_folder_serves_a_mix_with_humming(tmp_path):
    mix = _rung(tmp_path, "M-Pollard-GPTQ-int4", {"quant_method": "gptq", "bits": 4,
                                                  "dynamic": {r"+:.*\.layers\.0\..*": {"bits": 8}}})
    sec = "\n".join(C.lane_folder_section("gptq", [{"path": mix, "tag": "int4", "bytes": 1}], {}, "O/M-Pollard"))
    assert f"vllm serve ./M-Pollard/GPTQ/int4 {FLAG}" in sec and "vLLM >= 0.32.0" in sec
    assert "--quantization humming" in sec and "Not `--quantization humming`" in sec, "the note says why not"
    uni = _rung(tmp_path, "M-Pollard-GPTQ-int8", {"quant_method": "gptq", "bits": 8})
    sec = "\n".join(C.lane_folder_section("gptq", [{"path": uni, "tag": "int8", "bytes": 1}], {}, "O/M-Pollard"))
    assert "vllm serve ./M-Pollard/GPTQ/int8" in sec and FLAG not in sec and "humming" not in sec


def test_mx_mix_and_mxfp6_serve_lines(tmp_path):
    meta = {"format": "mix", "runtimes": [["vLLM", "yes", "quark"]]}
    assert M6.serve_lines(meta, "./mix-5bpw")[0].startswith(f"vllm serve ./mix-5bpw {FLAG}")
    assert FLAG in M6.serve_lines(dict(meta, format="mxfp6-e3m2"), "./e3m2")[0]
    assert M6.serve_lines(dict(meta, format="fp8"), "./fp8") == ["vllm serve ./fp8"]
    # an llm-compressor rung (no pollard_mx.json) with several config groups
    ct = _rung(tmp_path, "M-Pollard-MX-NVFP4", {"quant_method": "compressed-tensors",
                                                "config_groups": {"group_0": {}, "group_1": {}}})
    md = "\n".join(C.rungs_section([{"path": ct, "name": "M-Pollard-MX-NVFP4", "tag": "NVFP4", "bytes": 1}],
                                   "mx", "O/M-Pollard-MX"))
    assert f"vllm serve ./nvfp4 {FLAG}" in md


def test_the_repo_level_vllm_line(tmp_path):
    mix = _rung(tmp_path, "g4", {"quant_method": "gptq", "bits": 4, "dynamic": {r"+:x": {"bits": 8}}})
    block = "\n".join(C.vllm_serve_block("O/M-Pollard", ["gptq"], [{"path": mix}]))
    assert f"vllm serve O/M-Pollard {FLAG}" in block and "0.32.0" in block
    assert FLAG not in "\n".join(C.vllm_serve_block("O/M-Pollard", ["gptq"], []))
    tools = {"vllm_parser": "hermes", "vllm_flags": "--enable-auto-tool-choice --tool-call-parser hermes"}
    line = "\n".join(C.tool_serve_lines(tools, ["gptq"], "O/M-Pollard", "x.gguf", [{"path": mix}]))
    assert f"--tool-call-parser hermes {FLAG}" in line


def test_no_card_line_ever_says_quantization_humming():
    for f in ("pollard_card.py", "pollard_mxfp6.py", "pollard_mx.py", "pollard_gptq.py", "pollard_gptqpack.py"):
        src = (ROOT / "tools" / f).read_text(encoding="utf-8")
        assert "vllm serve" not in src or "--quantization humming`," in src or "--quantization humming" not in src
    lines = M6.serve_lines({"format": "mix", "runtimes": [["vLLM", "yes", ""]]}, "./d") + [M6.humming_serve("r")]
    assert all("--quantization" not in x for x in lines)
