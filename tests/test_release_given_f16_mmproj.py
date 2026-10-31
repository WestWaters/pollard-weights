"""A release given an f16 skips the HF conversion -- the step that writes the mmproj -- so LFM3-Audio / -Omni went out
as the language model alone (2026-10-28). Given an f16 and a checkpoint with weights, the GGUF lane exports the mmproj
itself, before the ladder and therefore before the complete-model gate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


def _plan(tmp_path, monkeypatch, given_f16, weights=True):
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    import pollard_release as R
    src = tmp_path / "LFM3-Audio-2B"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "lfm3_audio", "num_hidden_layers": 2}))
    if weights:
        (src / "model.safetensors").write_bytes(b"")
    cfg = R.load_plan(None, "text")
    cfg.update(calib_file=str(tmp_path / "c.txt"), heldout_file=str(tmp_path / "h.txt"),
               sensitivity=str(tmp_path / "s.json"))
    if given_f16:
        f16 = tmp_path / "LFM3-Audio-2B-f16.gguf"
        f16.write_bytes(b"")
        cfg["f16"] = str(f16)
    _, _, steps = R.plan_steps(cfg, str(src), {"gguf"})
    return [s.id for s in steps if s.lane == "gguf"]


def test_given_f16_exports_the_mmproj_before_the_ladder(tmp_path, monkeypatch):
    ids = _plan(tmp_path, monkeypatch, given_f16=True)
    assert "gguf/mmproj" in ids
    assert ids.index("gguf/mmproj") < min(i for i, s in enumerate(ids) if s.startswith("gguf/ladder"))


def test_the_hf_path_writes_it_itself(tmp_path, monkeypatch):
    assert "gguf/mmproj" not in _plan(tmp_path, monkeypatch, given_f16=False)


def test_no_checkpoint_weights_no_export(tmp_path, monkeypatch):
    assert "gguf/mmproj" not in _plan(tmp_path, monkeypatch, given_f16=True, weights=False)


def test_a_text_only_checkpoint_needs_no_converter(tmp_path, monkeypatch):
    import pollard_release as R
    src = tmp_path / "m"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(R, "_tower_components", lambda d: {})
    c = {"gguf_dir": str(tmp_path / "g"), "f16": str(tmp_path / "g" / "m-f16.gguf"), "hf_dir": str(src)}
    assert R.write_mmproj(c, out=lambda *a: None)


def test_a_missing_converter_fails_the_step_instead_of_the_release(tmp_path, monkeypatch):
    import pollard_auto as PA
    import pollard_release as R
    src = tmp_path / "m"
    src.mkdir()
    monkeypatch.setattr(R, "_tower_components", lambda d: {"audio": {"encoder.x": [1]}})
    monkeypatch.setattr(PA, "_find_convert", lambda d=None: (_ for _ in ()).throw(SystemExit("no converter")))
    c = {"gguf_dir": str(tmp_path / "g"), "f16": str(tmp_path / "g" / "m-f16.gguf"), "hf_dir": str(src)}
    assert R.write_mmproj(c, out=lambda *a: None) is False
