"""pollard-audioeval: scoring (normalisation per script, CER cap), transcript extraction from the CLIs' output, the
sentence table, and the per-language / worst-language report -- no llama.cpp binaries."""
from __future__ import annotations

import os
import sys
import types
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_audioeval as A  # noqa: E402


def test_normalize_drops_punctuation_and_cjk_spaces():
    assert A.normalize("Hello,  World!", "en") == "hello world"
    assert A.normalize("明日の 天気は、曇り。", "ja") == "明日の天気は曇り"
    assert A.normalize("ＡＢＣ", "en") == "abc"                     # NFKC folds full-width


def test_cer_and_cap():
    assert A.cer("The quick fox.", "the quick fox", "en") == 0.0
    assert A.cer("abcd", "abce", "en") == 0.25
    assert A.cer("ab", "a long runaway transcript", "en") == 1.0   # capped
    assert A.cer("", "", "en") == 0.0


def test_clean_transcript():
    assert A.clean_transcript("language Japanese<asr_text>明日の天気。\n") == "明日の天気。"
    assert A.clean_transcript("log line\nThe quick brown fox. Transcribe the audio.\n",
                              "Transcribe the audio.") == "The quick brown fox."
    assert A.clean_transcript("<think></think>Hello there.") == "Hello there."


def test_sentence_table_covers_every_tts_language():
    langs = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "it", "ja", "ko", "nl", "pl", "pt", "sv", "tr",
             "zh"}
    assert langs <= set(A.SENTENCES)
    for code, (name, sents) in A.SENTENCES.items():
        assert name and len(sents) >= 4, code
        assert all(A.normalize(s, code) for s in sents), code


def test_wav_seconds(tmp_path):
    p = tmp_path / "x.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)
    assert abs(A.wav_seconds(str(p)) - 0.5) < 1e-9
    assert A.wav_seconds(str(tmp_path / "missing.wav")) == 0.0


def test_report_names_the_worst_language(monkeypatch, tmp_path):
    def fake_speak(bin_dir, model, mmproj, text, lang, out, ngl, voice="female", timeout=600, seed=None):
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * (16000 if model == "rung" else 8000))
        return True

    def fake_transcribe(bin_dir, model, mmproj, audio, ngl, system=None, prompt="Transcribe the audio.", n=160,
                        timeout=600):
        lang = os.path.basename(audio)[:2]
        i = int(os.path.basename(audio)[2])
        text = A.SENTENCES[lang][1][i]
        return text if lang == "en" else text[: len(text) // 2]   # ja loses half of every sentence

    monkeypatch.setattr(A, "speak", fake_speak)
    monkeypatch.setattr(A, "transcribe", fake_transcribe)
    a = types.SimpleNamespace(langs="en,ja", gguf="rung", mmproj="mm", ref="ref", ref_mmproj=None, judge="j",
                              judge_mmproj="jm", probes="out,in", ngl="0", bin=None, keep=str(tmp_path))
    res = A.run(a)
    assert res["langs"]["en"]["speak_cer_mean"] == 0.0
    assert res["speak_cer"]["worst_lang"] == "ja" and res["speak_cer"]["worst"] > 0.4
    assert res["listen_cer"]["worst_lang"] == "ja"
    assert abs(res["speak_dur_drift"]["mean"] - 0.6931) < 1e-3     # the rung's audio is twice as long


def test_repetition_flags_loops_not_normal_speech():
    s = A.SENTENCES["en"][1][2]
    assert A.repetition(s, "en") == 0.0
    assert A.repetition(" ".join([s.split(",")[0]] * 4), "en") > 0.5
    assert A.repetition("明日の天気は曇り" * 3, "ja") > 0.5


def test_long_probe_scores_the_chunked_passage(monkeypatch, tmp_path):
    said = []

    def fake_speak(bin_dir, model, mmproj, text, lang, out, ngl, voice="female", timeout=600, seed=None):
        said.append(text)
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        return True

    spoken = {}
    monkeypatch.setattr(A, "speak", lambda *a_, **k: (spoken.__setitem__(a_[5], a_[3]), fake_speak(*a_, **k))[1])
    monkeypatch.setattr(A, "transcribe", lambda b, m, mm, audio, ngl, **k: spoken[audio])
    a = types.SimpleNamespace(langs="en", gguf="rung", mmproj="mm", ref="ref", ref_mmproj=None, judge="j",
                              judge_mmproj="jm", probes="long", ngl="0", bin=None, keep=str(tmp_path))
    res = A.run(a)
    row = res["langs"]["en"]
    assert row["long_cer"] == 0.0 and row["long_repeat"] == 0.0 and row["long_dur_drift"] == 0.0
    assert max(len(t) for t in said) < 260                     # chunks, never the whole passage at once
