"""pollard_probe.load_backbone on audio-in LLMs (2026-10-28): granite-speech-5.0-2b and Qwen3-ASR-1.7B register
only as multimodal / speech-seq2seq classes, and Qwen3-ASR's checkpoint names its decoder `thinker.model.*` --
loaded as-is, every decoder weight came back MISSING (random) and the load still 'succeeded'."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pollard_probe as P  # noqa: E402

DECODER = ["model.language_model.layers.0.self_attn.q_proj.weight", "model.language_model.embed_tokens.weight",
           "lm_head.weight"]


class _Fake:
    """A from_pretrained that misses the decoder unless key_mapping maps the `thinker.*` names."""
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, dtype=None, output_loading_info=False, key_mapping=None, **kw):
        cls.calls.append(key_mapping)
        mapped = key_mapping is not None
        info = {"missing_keys": [] if mapped else DECODER + ["model.multi_modal_projector.linear_1.weight"],
                "unexpected_keys": [] if mapped else ["thinker.model.layers.0.self_attn.q_proj.weight"]}
        return object(), info


def test_only_text_decoder_weights_count_as_missing():
    info = {"missing_keys": DECODER + ["model.audio_tower.layers.0.fc1.weight",
                                       "model.multi_modal_projector.linear_1.weight"]}
    assert P._decoder_missing(info) == DECODER


def test_a_thinker_named_checkpoint_is_remapped_not_measured_as_random():
    _Fake.calls.clear()
    P._load_checked(_Fake, "Qwen/Qwen3-ASR-1.7B", None, {})
    assert _Fake.calls == [None, P.THINKER_KEY_MAP]


def test_a_decoder_that_still_did_not_load_is_refused():
    class _Never(_Fake):
        @classmethod
        def from_pretrained(cls, model_id, dtype=None, output_loading_info=False, key_mapping=None, **kw):
            return object(), {"missing_keys": DECODER, "unexpected_keys": ["thinker.model.norm.weight"]}
    with pytest.raises(ValueError, match="would be random"):
        P._load_checked(_Never, "x", None, {})
