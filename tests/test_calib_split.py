"""Held-out calibration data must not contain repeated training samples."""
import importlib.util
from pathlib import Path
import random
import sys
import unicodedata

import pytest

spec = importlib.util.spec_from_file_location(
    "calib_split", Path(__file__).parents[1] / "tools" / "pollard_calib.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def keys(rows):
    return {" ".join(unicodedata.normalize("NFC", text).split()) for _, text in rows}


def test_deduplicates_whitespace_unicode_and_cross_domain_content():
    rows = [("prose", "hf", ["café story", "cafe\u0301  story", "second story", "third story"]),
            ("chat", "hf", ["café\nstory", "chat one", "chat two", "chat two"])]
    train, held, stats = C.split_domains(rows, .1, random.Random(0))
    assert not keys(train) & keys(held)
    assert len(keys(train) | keys(held)) == len(train) + len(held) == 5
    assert stats == [("prose", "hf", 2, 1), ("chat", "hf", 1, 1)]
    assert (train, held, stats) == C.split_domains(rows, .1, random.Random(0))


def test_preserves_code_indentation():
    code = "def f():\n    return 1"
    train, held, _ = C.split_domains([("code", "hf", [code, "return 2"])], .5, random.Random(1))
    assert code in [text for _, text in train + held]


@pytest.mark.parametrize("fraction", [0, 1, -.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        C.split_domains([], fraction, random.Random(0))


def test_rejects_insufficient_unique_content():
    with pytest.raises(ValueError, match="two unique"):
        C.split_domains([("prose", "hf", ["same", " same\n", ""])], .1, random.Random(0))


def test_leaves_at_least_one_training_sample():
    train, held, _ = C.split_domains([("prose", "hf", ["one", "two"])], .99, random.Random(0))
    assert len(train) == len(held) == 1


def run(monkeypatch, out, *args):
    monkeypatch.setattr(sys, "argv", ["pollard-calib", "--out", str(out), *args])
    C.main()


def test_offline_seed_split_is_disjoint(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(C, "_from_hf", lambda *args: [])
    out, held = tmp_path / "train", tmp_path / "held"
    run(monkeypatch, out, "--domains", "prose", "--held-out", str(held))
    train_text, held_text = out.read_text(), held.read_text()
    assert set(train_text.splitlines()).isdisjoint(held_text.splitlines())
    assert len(train_text.splitlines()) == 3
    assert len(held_text.splitlines()) == 1
    assert "4 unique samples of 300 requested" in capsys.readouterr().err
    run(monkeypatch, out, "--domains", "prose", "--held-out", str(held))
    assert (out.read_text(), held.read_text()) == (train_text, held_text)


def test_without_held_out_keeps_requested_sample_count(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "_from_hf", lambda *args: [])
    out = tmp_path / "train"
    run(monkeypatch, out, "--domains", "prose", "--per-domain", "30")
    assert len(out.read_text().splitlines()) == 30


def test_bad_split_does_not_write_any_output(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "build_domain", lambda *args: (["same"] * 30, "hf"))
    out, held = tmp_path / "train", tmp_path / "held"
    with pytest.raises(SystemExit):
        run(monkeypatch, out, "--held-out", str(held))
    assert not out.exists() and not held.exists()


@pytest.mark.parametrize("kind", ["same", "symlink", "hardlink"])
def test_rejects_aliasing_output_paths_before_loading(monkeypatch, tmp_path, kind):
    out, held = tmp_path / "train", tmp_path / "held"
    out.write_text("keep existing content")
    if kind == "same":
        held = out
    elif kind == "symlink":
        held.symlink_to(out)
    else:
        held.hardlink_to(out)
    def unexpected(*args):
        pytest.fail("dataset loading should not start")
    monkeypatch.setattr(C, "build_domain", unexpected)
    with pytest.raises(SystemExit):
        run(monkeypatch, out, "--held-out", str(held))
    assert out.read_text() == "keep existing content"


@pytest.mark.parametrize("args", [
    ["--per-domain", "0"], ["--min-chars", "0"], ["--domains", ""],
    ["--domains", "code,code"], ["--held-out", "held", "--held-frac", "nan"],
])
def test_invalid_arguments_do_not_write(monkeypatch, tmp_path, args):
    out = tmp_path / "train"
    with pytest.raises(SystemExit):
        run(monkeypatch, out, *args)
    assert not out.exists()
