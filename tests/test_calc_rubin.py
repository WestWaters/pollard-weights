"""pollard-calc / pollard-fit on Rubin-class and rack-scale memory.

What broke: the card table stopped at B200, so `--gpu rubin` / `--gpu b300` fell through to "not in the
card table"; `--ram` only took a number, so `pollard-fit --ram rubin` died on float('rubin'); the fit
ladder labelled every pool past 2 TB as "<N>x Spark-class" (a 20 TB model read "160x Spark-class");
and there was no fp8 KV option although vLLM's `--kv-cache-dtype fp8` is what the MX lane serves with.
Memory per part: Rubin R100 / VR200 288 GB HBM4, B300 / GB300 288 GB, GB200 192 GB nameplate (~186
usable), Vera Rubin NVL72 = 72 x 288 = 20,736 GB.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_calc as C

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")


@pytest.mark.parametrize("name,gb", [("rubin", 288), ("r100", 288), ("vr200", 288), ("b300", 288),
                                     ("gb300", 288), ("gb200", 192), ("nvl72", 20736),
                                     ("vr-nvl72", 20736), ("gb200-nvl72", 13824), ("RUBIN", 288)])
def test_rubin_and_rack_presets(name, gb):
    assert C.parse_gpu(name) == gb


def test_presets_stack_like_any_card():
    assert C.parse_gpu("rubinx8") == 8 * 288
    assert C.parse_gpu("b300x8") == 8 * 288
    assert C.parse_gpu("vr-nvl72x2") == 2 * 20736


def test_mem_parser_takes_numbers_and_presets():
    assert C.parse_mem_gb("16") == 16.0
    assert C.parse_mem_gb(24) == 24.0
    assert C.parse_mem_gb("rubin") == 288
    assert C.parse_mem_gb("nvl72") == 20736
    assert C.parse_mem_gb("not-a-card") is None


def test_fp8_kv_is_one_byte_per_element():
    a = C.analyse({"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
                   "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256})
    assert C.KV_BYTES["fp8"] == 1.0
    assert C.kv_cache_bytes(a, 1000, C.KV_BYTES["fp8"]) * 2 == C.kv_cache_bytes(a, 1000, C.KV_BYTES["f16"])


def _ladder(capsys, total_gb):
    a = {"layers": 1, "kv_heads": 0, "head_dim": 0}
    C.fit_report(a, total_gb, 0, 2.0, "f16")
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  [")]


def test_ladder_has_rubin_rows(capsys):
    rows = _ladder(capsys, 10)
    assert any("288 GB (Rubin R100 / B300" in r for r in rows)
    assert any("8x Rubin" in r for r in rows)


def test_rack_scale_is_not_labelled_spark(capsys):
    """A 19 TB model (past 64x Rubin, inside one rack): rows past one node are Rubin GPUs then an NVL72 rack, never Spark boxes."""
    rows = _ladder(capsys, 19_000)
    big = [r for r in rows if "TB" in r and "Spark" not in r and ("Rubin" in r)]
    assert not any("Spark-class" in r for r in rows)
    assert any("Vera Rubin NVL72 rack" in r and r.startswith("  [YES]") for r in rows)
    assert big


def test_ladder_extends_by_racks(capsys):
    rows = _ladder(capsys, 60_000)
    assert any("4x Vera Rubin NVL72 racks" in r and r.startswith("  [YES]") for r in rows)


def _config(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, '
                 '"num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256}')
    return str(p)


def test_calc_cli_takes_ram_preset_and_fp8_kv(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "pollard_calc.py"), "--config", _config(tmp_path),
                        "--ram", "rubin", "--ctx", "131072", "--kv-quant", "fp8"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "288 GB RAM" in r.stdout
    assert "KV cache (fp8)" in r.stdout


def test_calc_cli_rejects_unknown_ram(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "pollard_calc.py"), "--config", _config(tmp_path),
                        "--ram", "warp-drive"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "preset" in r.stderr


def test_fit_ram_goes_through_the_calc_parser():
    """pollard-fit must not float() --ram itself any more -- it shares calc's preset table."""
    src = open(os.path.join(TOOLS, "pollard_fit.py"), encoding="utf-8").read()
    assert "parse_mem_gb(a.ram)" in src
    assert "a.ram = float(a.ram)" not in src
