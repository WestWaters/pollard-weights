"""The speech-to-text gate: an audio-in GGUF rung must transcribe a fixed LibriSpeech subset under asr_wer_max before
it publishes (Qwen3-ASR's IQ2_XXS was "verified" at 100% WER with coherence off). All offline: no download, no
llama.cpp -- the transcriber is stubbed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_audiobench as AB  # noqa: E402
import pollard_release as R  # noqa: E402
import pollard_verify as V  # noqa: E402


def _gate(metrics, rung="Qwen3-ASR-1.7B-Pollard-IQ3_S", gates=None):
    g = gates if gates is not None else V.load_gates(None)
    th = V.thresholds_for(g, rung, "gguf", [])
    if metrics.get("asr_expected") and "asr_wer_max" not in th:
        th["asr_wer_max"] = V.ASR_WER_MAX
    ro, strict = V.gate_policy(g)
    return V.evaluate_gate(metrics, th, g.get("missing", "fail"), ro, strict)


def _asr_row(res):
    rows = [c for c in res["checks"] if c["key"] == "asr_wer_max"]
    return rows[0] if rows else None


BASE = {"kld": 0.05, "top1": 95.0, "coherence": "PASS", "runtime": "stock"}


def test_default_gates_hold_audio_rungs_to_a_wer_ceiling_and_it_blocks():
    assert V.load_gates(None)["lanes"]["gguf"]["asr_wer_max"] == 10
    assert "asr_wer_max" not in V.REPORT_ONLY
    ok = _gate(dict(BASE, asr_wer=1.72, asr_expected=True))
    assert ok["pass"] and _asr_row(ok)["ok"]
    broken = _gate(dict(BASE, asr_wer=100.0, asr_expected=True))
    assert not broken["pass"] and _asr_row(broken)["blocking"]


def test_text_model_gets_no_asr_row_and_unmeasured_audio_model_fails():
    assert _asr_row(_gate(dict(BASE))) is None and _gate(dict(BASE))["pass"]
    nm = _gate(dict(BASE, asr_expected=True, asr_reason="no mmproj beside the rung or its f16"))
    assert not nm["pass"] and "no mmproj" in _asr_row(nm)["why"]


def test_a_gates_file_without_the_key_still_holds_an_audio_model():
    g = {"missing": "fail", "default": {}, "lanes": {"gguf": {"runtime": "stock", "coherence": False}}}
    res = _gate({"runtime": "stock", "asr_wer": 49.6, "asr_expected": True}, gates=g)
    assert not res["pass"] and _asr_row(res)["value"] == 49.6


def test_results_json_carries_the_number_back_into_the_gate():
    m = V.metrics_from_results({"r": {"asr": {"wer": 3.1}}}, "r")
    assert m["asr_wer"] == 3.1 and m["asr_expected"] is True


def test_measure_asr_scores_the_cached_set_through_rung_and_mmproj(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    rung, mm = d / "M-Pollard-IQ3_S.gguf", d / "mmproj-M-BF16.gguf"
    rung.write_bytes(b"")
    mm.write_bytes(b"")
    s = tmp_path / "set"
    s.mkdir()
    rows = [{"id": f"c{i}", "audio": f"c{i}.wav", "text": "the cat sat on the mat", "speaker": str(i % 2),
             "lang": "en"} for i in range(4)]
    (s / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    calls = []

    def fake(gguf, mmproj, audio, system=None, prompt="", ngl="99", bin_dir=None, jinja=False):
        calls.append((gguf, mmproj, prompt, jinja))
        return "the cat sat on the mat" if not audio.endswith("c3.wav") else "the cat sat"
    monkeypatch.setattr(AB, "transcribe_llama", fake)
    monkeypatch.setattr(AB, "fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must use the cache")))
    out = V.measure_asr(str(rung), None, prompt="Transcribe.", jinja=True, clips=4, set_dir=str(s))
    assert out["asr_expected"] and out["asr"]["mmproj"] == mm.name and out["asr"]["clips"] == 4
    assert abs(out["asr_wer"] - 100 * 3 / 24) < 0.01                 # 3 deletions over 24 reference words
    assert all(c[1] == str(mm) and c[2] == "Transcribe." and c[3] for c in calls)


def test_measure_asr_without_mmproj_or_set_reports_why(tmp_path, monkeypatch):
    rung = tmp_path / "M-Pollard-IQ3_S.gguf"
    rung.write_bytes(b"")
    assert "no mmproj" in V.measure_asr(str(rung), None, set_dir=str(tmp_path / "s"))["asr_reason"]
    (tmp_path / "mmproj-M-BF16.gguf").write_bytes(b"")
    monkeypatch.setattr(AB, "fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    out = V.measure_asr(str(rung), None, set_dir=str(tmp_path / "s"))
    assert "asr_wer" not in out and "offline" in out["asr_reason"]


def test_release_turns_the_gate_on_for_audio_input_models_only(tmp_path, monkeypatch):
    import pollard_modelkind as MK
    f16 = tmp_path / "M-f16.gguf"
    f16.write_bytes(b"")
    c = {"f16": str(f16)}
    monkeypatch.setattr(MK, "classify", lambda src: {"modalities": ["audio"]})
    assert R.asr_args({}, c) == ["--asr"]
    monkeypatch.setattr(MK, "classify", lambda src: {"modalities": ["image"]})
    assert R.asr_args({}, c) == []
    assert R.asr_args({"measure": {"asr": False}}, c) == []
    assert R.asr_args({"measure": {"asr": {"prompt": "Transcribe.", "jinja": True}}}, c) == [
        "--asr", "--asr-prompt", "Transcribe.", "--asr-jinja"]


def test_an_f16_that_reads_as_text_still_gets_the_gate_from_the_hf_checkout(tmp_path, monkeypatch):
    """Qwen3-ASR's f16 GGUF holds only the text decoder: the HF checkout (or the mmproj) says it listens."""
    import pollard_modelkind as MK
    f16, hf = tmp_path / "M-f16.gguf", tmp_path / "hf"
    f16.write_bytes(b"")
    hf.mkdir()
    monkeypatch.setattr(MK, "classify", lambda src: {"modalities": ["audio"] if src == str(hf) else []})
    assert R.asr_args({}, {"f16": str(f16), "hf_dir": str(hf)}) == ["--asr"]
    assert R.asr_args({}, {"f16": str(f16)}) == []
