"""pollard-calib on an audio-only chat template (Qwen3-ASR-1.7B, 2026-10-28): its chat_template.json places audio and
drops every text part, so every record rendered to the same empty turns and the chat domain collapsed to one sample
('needs at least two unique samples for a held-out split'). Such a template falls back to generic ChatML."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")              # the fallback path is exercised against its AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pollard_calib as C  # noqa: E402


class _Tpl:
    def __init__(self, keeps_text):
        self.keeps_text, self.source = keeps_text, "{{ messages }}"

    def render(self, messages, add_generation_prompt=False, tools=None, **kw):
        if not self.keeps_text:                         # Qwen3-ASR: empty turns whatever the text
            return "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n"
        return "".join(f"{m['role'].upper()}: {m['content']}\n" for m in messages)


def _renderer(monkeypatch, keeps_text):
    import pollard_modelkind
    monkeypatch.setattr(pollard_modelkind.ChatTemplate, "from_model", classmethod(lambda cls, m: _Tpl(keeps_text)))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError("no tokenizer"))))
    return C.Renderer(model="/x/audio-model")


def test_an_audio_only_template_falls_back_to_chatml_and_keeps_the_text(monkeypatch):
    r = _renderer(monkeypatch, keeps_text=False)
    assert "does not render message text" in r.why
    a = r._apply([{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}])
    b = r._apply([{"role": "user", "content": "Write a haiku."}, {"role": "assistant", "content": "Rain"}])
    assert "What is 2+2?" in a and a != b               # distinct records stay distinct


def test_a_template_that_renders_text_is_kept(monkeypatch):
    r = _renderer(monkeypatch, keeps_text=True)
    assert "own chat template" in r.why
    assert r._apply([{"role": "user", "content": "hi"}]) == "USER: hi\n"
