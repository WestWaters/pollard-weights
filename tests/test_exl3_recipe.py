"""pollard-exl3 --recipe pollard: Pollard's measured allocation emitted as an ExLlamaV3 `-rcp` recipe.yaml.

The recipe must load in EXL3's OWN parser: tests/fixtures/exl3_recipe_parser_v1_6_2.py is v1.6.2's
create_q_strategy_from_recipe + convert.py's -rcp checks, vendored verbatim (MIT). The checkpoint here is a
synthetic safetensors file written by hand (headers only matter), so this runs without torch/exllamav3; the
YAML round trip needs pyyaml. The real head-to-head (Pollard recipe vs sc_optimize at the same bpw, KL on
held-out) is a box run: notes/exl3-recipe-bridge.md."""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_exl3 as E  # noqa: E402

H, I, L = 64, 128, 3


def _write_safetensors(path, shapes):
    """A valid safetensors file (F16 zeros) from {name: shape} -- no library needed."""
    hdr, off = {}, 0
    for name, shape in shapes.items():
        n = 2
        for d in shape:
            n *= d
        hdr[name] = {"dtype": "F16", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(hdr).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + b"\0" * off)


def _model(tmp_path, moe_layer=2, experts=4):
    d = tmp_path / "m"
    d.mkdir()
    shapes = {"model.embed_tokens.weight": (256, H), "model.norm.weight": (H,), "lm_head.weight": (256, H),
              "model.visual.blocks.0.attn.qkv.weight": (3 * H, H), "mtp.layers.0.mlp.up_proj.weight": (I, H)}
    for li in range(L):
        p = f"model.layers.{li}"
        shapes[f"{p}.input_layernorm.weight"] = (H,)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[f"{p}.self_attn.{n}.weight"] = (H, H)
        if li == moe_layer:
            shapes[f"{p}.mlp.gate.weight"] = (experts, H)                         # router: not budgeted
            shapes[f"{p}.mlp.shared_expert_gate.weight"] = (1, H)
            shapes[f"{p}.mlp.experts.gate_up_proj"] = (experts, 2 * I, H)        # fused experts
            shapes[f"{p}.mlp.experts.down_proj"] = (experts, H, I)
        else:
            shapes[f"{p}.mlp.gate_proj.weight"] = (I, H)
            shapes[f"{p}.mlp.up_proj.weight"] = (I, H)
            shapes[f"{p}.mlp.down_proj.weight"] = (H, I)
    _write_safetensors(d / "model.safetensors", shapes)
    (d / "config.json").write_text(json.dumps({"architectures": ["FooForCausalLM"], "num_hidden_layers": L}))
    return str(d)


SENS = {"attn": {"0": 0.01, "1": 0.30, "2": 0.02}, "ffn": {"0": 0.50, "1": 0.02, "2": 0.10}, "probe": "rtn4"}


def test_budgeted_keys_follow_exl3_naming(tmp_path):
    keys, how = E.exl3_budgeted_keys(_model(tmp_path), prefer_exl3=False)
    names = {k for k, _ in keys}
    assert len(names) == len(keys), "no tensor listed twice"
    assert how == "safetensors headers"
    assert "model.layers.0.self_attn.q_proj" in names and "model.layers.0.mlp.down_proj" in names
    assert {f"model.layers.2.mlp.experts.{e}.{n}" for e in range(4) for n in ("gate_proj", "up_proj", "down_proj")} <= names
    for bad in ("model.layers.2.mlp.gate", "model.layers.2.mlp.shared_expert_gate", "lm_head", "model.embed_tokens",
                "model.norm", "model.visual.blocks.0.attn.qkv", "mtp.layers.0.mlp.up_proj",
                "model.layers.0.input_layernorm", "model.layers.2.mlp.experts.gate_up_proj"):
        assert bad not in names, bad
    assert dict(keys)["model.layers.2.mlp.experts.0.gate_proj"] == I * H


@pytest.mark.parametrize("bpw", (2.5, 3.0, 4.0, 5.5))
def test_recipe_hits_the_budget_ties_fused_groups_and_follows_sensitivity(tmp_path, bpw):
    keys, _ = E.exl3_budgeted_keys(_model(tmp_path), prefer_exl3=False)
    r = E.pollard_recipe(keys, SENS, bpw)
    t = r["tensors"]
    assert set(t) == {k for k, _ in keys}
    assert all(isinstance(v, int) and 2 <= v <= 8 for v in t.values())
    total = sum(n for _, n in keys)
    achieved = sum(t[k] * n for k, n in keys) / total
    assert achieved <= bpw + 1e-9 and achieved > bpw - 0.6 and abs(achieved - r["achieved_bpw"]) < 1e-3
    for li in range(L):                                   # EXL3 fuses q/k/v and gate/up: one bitrate each
        assert len({t[f"model.layers.{li}.self_attn.{n}"] for n in ("q_proj", "k_proj", "v_proj")}) == 1
    for e in range(4):
        assert t[f"model.layers.2.mlp.experts.{e}.gate_proj"] == t[f"model.layers.2.mlp.experts.{e}.up_proj"]
    # the measured-hot groups get at least the bits of the cold ones
    assert t["model.layers.0.mlp.down_proj"] >= t["model.layers.1.mlp.down_proj"]
    assert t["model.layers.1.self_attn.o_proj"] >= t["model.layers.0.self_attn.o_proj"]


def test_written_yaml_round_trips_and_passes_the_schema_checks(tmp_path):
    yaml = pytest.importorskip("yaml")
    keys, _ = E.exl3_budgeted_keys(_model(tmp_path), prefer_exl3=False)
    r = E.pollard_recipe(keys, SENS, 3.0)
    r["model"] = "/some/path: with colon"
    p = E.write_recipe(r, str(tmp_path / "r.yaml"))
    back = yaml.safe_load(open(p))
    assert back["tensors"] == r["tensors"] and back["target_bpw"] == 3.0 and back["head_bits"] == 6
    assert back["model"] == "/some/path: with colon"
    assert E.check_recipe(back, [k for k, _ in keys]) == []
    broken = dict(back, tensors={**back["tensors"], "model.layers.9.self_attn.q_proj": 4})
    assert any("match no budgeted" in e for e in E.check_recipe(broken, [k for k, _ in keys]))
    short = dict(back, tensors={k: v for k, v in list(back["tensors"].items())[1:]})
    assert any("missing" in e for e in E.check_recipe(short, [k for k, _ in keys]))
    assert E.check_recipe(dict(back, tensors={k: 9 for k in back["tensors"]}), [k for k, _ in keys])


def _exl3_parser():
    spec = importlib.util.spec_from_file_location("exl3_parser", ROOT / "tests" / "fixtures" / "exl3_recipe_parser_v1_6_2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Mod:
    def __init__(self, key=None, numel=0, qmap="x", qbits_key="bits", modules=()):
        self.key, self.numel, self.qmap, self.qbits_key, self.modules = key, numel, qmap, qbits_key, list(modules)

    def weights_numel(self):
        return self.numel


def test_recipe_loads_in_exl3s_own_parser(tmp_path):
    X = _exl3_parser()

    class Linear(_Mod):
        pass
    X.Linear = Linear
    keys, _ = E.exl3_budgeted_keys(_model(tmp_path), prefer_exl3=False)
    # the model graph EXL3 would build: budgeted linears, a router with no qmap, the head on head_bits
    blocks = [_Mod(modules=[Linear(k, n) for k, n in keys]),
              _Mod(modules=[Linear("model.layers.2.mlp.gate", 4 * H, qmap=None)]),
              Linear("lm_head", 256 * H, qbits_key="head_bits")]

    class Model:
        modules = blocks
    r = E.pollard_recipe(keys, SENS, 3.5)
    targets, bpw = X.create_q_strategy_from_recipe(Model, None, None, r["tensors"], r["head_bits"], 4)
    assert targets["lm_head"] == 6 and abs(bpw - r["achieved_bpw"]) < 1e-3
    assert all(targets[k] == v for k, v in r["tensors"].items())
    with pytest.raises(ValueError, match="missing"):
        X.create_q_strategy_from_recipe(Model, None, None, dict(list(r["tensors"].items())[1:]), 6, 4)
    with pytest.raises(ValueError, match="not matching"):
        X.create_q_strategy_from_recipe(Model, None, None, {**r["tensors"], "lm_head": 4}, 6, 4)
    pytest.importorskip("yaml")
    p = E.write_recipe(r, str(tmp_path / "r.yaml"))
    tensors, bits, head = X.parse_recipe_args(p)
    assert tensors == r["tensors"] and bits == r["achieved_bpw"] and head == 6


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "tools"))
    return subprocess.run([sys.executable, str(ROOT / "tools" / "pollard_exl3.py"), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_cli_modes(tmp_path):
    m = _model(tmp_path)
    sens = tmp_path / "s.json"
    sens.write_text(json.dumps(SENS))
    out = tmp_path / "o"
    r = _cli("--model", m, "--out", str(out), "--bpw", "3", "--recipe", "pollard", "--sensitivity", str(sens),
             "--plan-only")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.pollard-recipe.yaml").exists() and "-rcp" in r.stdout
    r = _cli("--model", m, "--out", str(out), "--bpw", "3", "--recipe", "pollard", "--plan-only")
    assert r.returncode != 0 and "--sensitivity" in (r.stderr + r.stdout)
    r = _cli("--model", m, "--out", str(out), "--bpw", "3", "--recipe", "exl3", "--plan-only")
    assert r.returncode != 0 and "sc_optimize.py" in r.stdout
    r = _cli("--model", m, "--out", str(out), "--bpw", "3", "--plan-only")
    assert r.returncode == 0 and "-rcp" not in r.stdout and "-b 3.0" in r.stdout
