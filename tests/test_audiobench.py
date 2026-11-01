"""pollard-audiobench: the leaderboard's scoring (normalization, corpus-level WER, CER for scripts without spaces),
the speaker-block bootstrap, the speaker-spread sample, transcript extraction, and the asr run end to end with a stub
transcriber -- no network, no binaries, no optional packages."""
from __future__ import annotations

import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_audiobench as B  # noqa: E402


def test_units_words_and_characters():
    assert B.units("Hello, World!", "en") == ["hello", "world"]
    assert B.units("技術 決定", "ja") == ["技", "術", "決", "定"]   # no spaces counted


def test_corpus_wer_is_total_edits_over_total_words():
    rows = [{"id": "a", "speaker": "1", "edits": 1, "ref_units": 10},
            {"id": "b", "speaker": "2", "edits": 0, "ref_units": 90}]
    assert B.corpus_error(rows) == 0.01                       # not the mean of 10% and 0%
    assert B.edits("a b c".split(), "a x c".split()) == 1
    assert B.edits([], ["x"]) == 1


def test_bootstrap_resamples_speakers_and_brackets_the_value():
    rows = [{"id": f"{s}-{i}", "speaker": str(s), "edits": s % 3, "ref_units": 10} for s in range(20) for i in range(3)]
    lo, hi = B.bootstrap_ci(rows)
    assert lo <= B.corpus_error(rows) <= hi and lo < hi
    assert B.bootstrap_ci(rows) == (lo, hi)                   # seeded: identical on every machine


def test_spread_is_deterministic_and_balances_speakers():
    items = [{"id": f"{s}-{i}", "speaker": s} for s in ("big",) for i in range(50)] + \
            [{"id": f"{s}-0", "speaker": s} for s in "abcdefghij"]
    pick = B.spread(items, 11, lambda x: x["speaker"])
    assert pick == B.spread(items, 11, lambda x: x["speaker"])
    assert sum(1 for p in pick if p["speaker"] == "big") <= 2   # one speaker cannot fill the sample


def test_clean_transcript_forms():
    assert B.clean_transcript("language English<asr_text>The quick fox.\n") == "The quick fox."
    assert B.clean_transcript("log\ntext_output=the quick fox\n") == "the quick fox"
    assert B.clean_transcript("x\nHello there. Transcribe the audio.", "Transcribe the audio.") == "Hello there."


def test_asr_run_end_to_end_with_a_stub(tmp_path, monkeypatch):
    rows = [{"id": "s1-0", "audio": "a.wav", "text": "IS THERE NOT A MERIDIAN EVERYWHERE", "speaker": "1", "lang": "en"},
            {"id": "s2-0", "audio": "b.wav", "text": "AT ONCE THE GOAT GAVE A LEAP", "speaker": "2", "lang": "en"}]
    with open(tmp_path / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    said = {"a.wav": "Is there not a meridian everywhere?", "b.wav": "At once the goat gave a leap."}
    seen = []

    def stub(g, m, audio, *a, **k):
        seen.append((k.get("prompt"), k.get("jinja")))
        return said[os.path.basename(audio)]
    monkeypatch.setattr(B, "transcribe_llama", stub)
    a = types.SimpleNamespace(set=str(tmp_path), limit=None, lang=None, audiocpp_family=None, model=None, backend=None,
                              gguf="rung.gguf", mmproj="mm.gguf", system=None, ngl="0", bin=None,
                              prompt="can you transcribe the speech into a written format?", jinja=True,
                              out=str(tmp_path / "r.json"))
    res = B.run_asr(a)
    assert res["metric"] == "wer" and res["value"] == 0.0 and res["clips"] == 2 and res["speakers"] == 2
    assert json.load(open(tmp_path / "r.json"))["rows"][0]["hyp"].startswith("Is there")
    assert seen[0] == ("can you transcribe the speech into a written format?", True)    # the card's prompt reaches the CLI


def test_language_codes_of_speech_sets():
    assert B._iso({"language_id": "cmn"}) == "zh" and B._iso({"language_id": "jpn"}) == "ja"
    assert B._iso({}) == "en" and B._iso({"language_id": "deu"}) == "de"


def test_tts_run_scores_the_judges_transcripts(tmp_path, monkeypatch):
    rows = [{"id": "a", "text": "Get the trust fund to the bank early.", "lang": "en", "speaker": "v1"},
            {"id": "b", "text": "The cat sat on the mat today.", "lang": "en", "speaker": "v2"}]
    with open(tmp_path / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    spoken = {}

    def synth(family, model, text, out, *a, **k):
        spoken[out] = text
        open(out, "wb").write(b"RIFF")
        return True
    heard = {"Get the trust fund to the bank early.": "get the trust fund to the bank early",
             "The cat sat on the mat today.": "the cat sat on a mat today"}          # one word wrong in seven
    monkeypatch.setattr(B, "synth_audiocpp", synth)
    monkeypatch.setattr(B, "whisper_transcribe", lambda m, wav, lang="en": heard[spoken[wav]])
    a = types.SimpleNamespace(set=str(tmp_path), limit=None, lang=None, keep=str(tmp_path / "wav"), clone=False,
                              audiocpp_family="ming3_tts", model="rung.gguf", backend=None, language="en", voice=None,
                              gguf=None, mmproj=None, ngl="0", bin=None, judge_whisper="ggml-large-v3.bin",
                              out=str(tmp_path / "r.json"))
    os.makedirs(a.keep, exist_ok=True)
    res = B.run_tts(a)
    assert res["metric"] == "wer" and res["failed"] == 0 and res["clips"] == 2
    assert abs(res["value"] - round(100 * 1 / 15, 2)) < 1e-9          # 1 edit over 8 + 7 reference words
    assert res["judge"].startswith("whisper-large-v3")
