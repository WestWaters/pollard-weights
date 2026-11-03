"""The MLX lane for speech models mlx-vlm does not carry (Qwen3-ASR, granite-speech): built whole through mlx-audio with
Pollard's per-module allocation, gated on transcription. All offline: mlx-audio is stubbed."""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_audiobench as AB  # noqa: E402
import pollard_mlx as M  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_verify as V  # noqa: E402


def test_convert_keeps_mlx_audios_own_recipe_and_applies_pollards_bits(monkeypatch):
    seen = {}
    conv = types.ModuleType("mlx_audio.convert")
    conv.build_quant_predicate = lambda model, name=None: (lambda p, m: not p.startswith("audio_tower"))

    def convert(**k):
        pred = conv.build_quant_predicate(None)
        seen["tower"] = pred("audio_tower.layers.0.fc1", None)
        seen["layer"] = pred("model.layers.0.mlp.down_proj", None)
        seen["kw"] = k
    conv.convert = convert
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.convert", conv)
    own = conv.build_quant_predicate
    M.mlx_audio_convert("src", "out", q_bits=4, q_group_size=64,
                        quant_predicate=lambda p, m: {"bits": 8, "group_size": 64})
    assert seen["tower"] is False                                   # the model's own recipe keeps the tower
    assert seen["layer"] == {"bits": 8, "group_size": 64}           # Pollard's allocation everywhere else
    assert seen["kw"]["q_bits"] == 4 and seen["kw"]["mlx_path"] == "out"
    assert conv.build_quant_predicate is own                         # restored after the build


def test_transcribe_mlx_loads_once_and_reads_text(monkeypatch):
    loads = []

    class Model:
        def generate(self, audio, **kw):
            assert kw["temperature"] == 0.0
            return types.SimpleNamespace(text=f" heard {Path(audio).name} ")
    utils = types.ModuleType("mlx_audio.stt.utils")
    utils.load_model = lambda d: loads.append(d) or Model()
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    monkeypatch.setattr(AB, "_MLX_MODELS", {})
    assert AB.transcribe_mlx("/rung", "/s/a.wav") == "heard a.wav"
    assert AB.transcribe_mlx("/rung", "/s/b.wav") == "heard b.wav"
    assert loads == ["/rung"]


def test_measure_asr_mlx_needs_no_mmproj(tmp_path, monkeypatch):
    s = tmp_path / "set"
    s.mkdir()
    rows = [{"id": f"c{i}", "audio": f"c{i}.wav", "text": "one two three four", "speaker": str(i), "lang": "en"}
            for i in range(3)]
    (s / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(AB, "transcribe_mlx", lambda d, a, system=None, language=None: "one two three four")
    monkeypatch.setattr(AB, "transcribe_llama", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not llama")))
    out = V.measure_asr(str(tmp_path / "MLX-4bit"), None, clips=3, set_dir=str(s), mlx=True)
    assert out["asr_wer"] == 0.0 and "mmproj" not in out["asr"]


def test_release_gates_an_mlx_audio_rung_on_transcription(tmp_path, monkeypatch):
    hf = tmp_path / "hf"
    hf.mkdir()
    monkeypatch.setattr(M, "mlx_engine", lambda d: ("mlx_audio", "stt:qwen3_asr"))
    assert R.mlx_speech_model({"hf_dir": str(hf)}) is True
    assert V.mlx_speech(str(hf)) is True
    assert R.mlx_speech_model({"hf_dir": str(hf)}, "stt") is True
    monkeypatch.setattr(M, "mlx_engine", lambda d: ("mlx_audio", "sts:lfm_audio"))   # LFM3-Audio: no stt gate
    assert R.mlx_speech_model({"hf_dir": str(hf)}) is True
    assert R.mlx_speech_model({"hf_dir": str(hf)}, "stt") is False
    monkeypatch.setattr(M, "mlx_engine", lambda d: ("mlx_vlm", "lfm3"))
    assert R.mlx_speech_model({"hf_dir": str(hf)}) is False
    assert R.mlx_speech_model({"hf_dir": str(tmp_path / "missing")}) is False


def test_mlx_audio_routes_only_a_model_it_really_carries(tmp_path, monkeypatch):
    """mlx-audio guesses a family for checkpoints it lacks (LFM3-Omni -> qwen2_audio, LFM3-Audio -> lfm_audio = LFM2):
    only an exact model_type match routes there, the rest are held, never built as the wrong model."""
    conv = types.ModuleType("mlx_audio.convert")
    guess = {"qwen3_asr": "qwen3_asr", "lfm3_omni": "qwen2_audio", "lfm3_audio": "lfm_audio"}
    conv.detect_model_domain = lambda cfg, p: "stt"
    conv.get_model_type = lambda cfg, p, dom: guess[cfg["model_type"]]
    conv.get_model_class = lambda mt, dom: object
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.convert", conv)
    for mt, want in (("qwen3_asr", "stt:qwen3_asr"), ("lfm3_omni", None), ("lfm3_audio", None)):
        d = tmp_path / mt
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": mt}))
        assert M.mlx_audio_type(str(d)) == want, mt
