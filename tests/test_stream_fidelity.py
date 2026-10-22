"""Streamed verification scores the same output distribution as native model forwards (CPU, no downloads)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import pollard_stream as S  # noqa: E402


@pytest.mark.parametrize("architecture,cap", [
    ("Gemma2", "default"), ("Gemma2", None), ("Gemma3Text", 30.0), ("Llama", None),
])
def test_streamed_fidelity_matches_native_logits(tmp_path, architecture, cap):
    torch = pytest.importorskip("torch")
    T = pytest.importorskip("transformers")
    torch.manual_seed(32)
    cfg = getattr(T, architecture + "Config")(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    if architecture.startswith("Gemma"):
        if cap == "default":
            assert cfg.final_logit_softcapping == 30.0
        else:
            cfg.final_logit_softcapping = cap
    model_class = getattr(T, ("Gemma3" if architecture == "Gemma3Text" else architecture) + "ForCausalLM")
    model = model_class(cfg).float().eval()
    rows = [[2, 5, 8, 12, 14, 43, 99, 32, 54, 86, 77, 37]]
    ids = torch.tensor(rows)
    start = 6
    source, rung = tmp_path / "source", tmp_path / "rung"
    with torch.no_grad():
        # Put the logits in the softcap-sensitive range, then change only the rung's output head.
        model.lm_head.weight.mul_(100)
        model.save_pretrained(source)
        reference_logits = model(ids).logits[0, start:].double()
        model.lm_head.weight.add_(torch.randn_like(model.lm_head.weight) * 0.2)
        model.save_pretrained(rung)
        rung_logits = model(ids).logits[0, start:].double()
    a, b = reference_logits.log_softmax(-1), rung_logits.log_softmax(-1)
    kl = (a.exp() * (a - b)).sum(-1)
    targets = ids[0, start + 1:]
    positions = torch.arange(len(targets))
    expected = {
        "kld": kl.mean().item(), "kld_p99": torch.quantile(kl, 0.99).item(),
        "top1": (a.argmax(-1) == b.argmax(-1)).double().mean().item() * 100,
        "ref_ppl": (-a[positions, targets]).mean().exp().item(),
        "ppl": (-b[positions, targets]).mean().exp().item(),
    }
    actual = S.stream_fidelity(str(source), str(rung), rows, starts=[start], block=2,
                               trust_remote_code="off", log=lambda *_: None)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=2e-4, abs=1e-7), key
    same = S.stream_fidelity(str(source), str(source), rows, starts=[start], block=2,
                             trust_remote_code="off", log=lambda *_: None)
    assert same["kld"] == pytest.approx(0, abs=1e-10)
    assert same["top1"] == 100.0
    assert same["ppl"] == pytest.approx(expected["ref_ppl"], rel=2e-4)
