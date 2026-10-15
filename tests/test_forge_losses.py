"""pollard-forge's distillation losses (AngelSlim's kl / rkl / cakld / top-k set): each matches its
definition, kl is unchanged from forge's original, and the CLI carries --loss / --lm-loss-weight into the
lineage. Needs torch (skips otherwise, like test_forge)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_forge as G  # noqa: E402


def _logits(seed, shape=(2, 5, 11)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * 2


def _kl(p_logits, q_logits):
    """KL(p || q) per token, mean."""
    lp, lq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def test_kl_is_forges_original_temperature_scaled_forward_kl():
    s, t = _logits(0), _logits(1)
    T = 2.0
    old = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True,
                   reduction="batchmean") * (T * T) / s.shape[1]
    assert torch.allclose(G.kd_loss(s, t, T), old, atol=1e-6)
    assert torch.allclose(G.kd_loss(s, t, 1.0, "kl"), _kl(t, s), atol=1e-6)


def test_rkl_and_cakld_definitions():
    s, t = _logits(2), _logits(3)
    y = torch.randint(0, 11, (2, 5), generator=torch.Generator().manual_seed(4))
    assert torch.allclose(G.kd_loss(s, t, 9.0, "rkl"), _kl(s, t), atol=1e-6), "rkl ignores T, like AngelSlim"
    conf = F.softmax(t, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1).flatten()
    lp, lq = F.log_softmax(t, -1).flatten(0, 1), F.log_softmax(s, -1).flatten(0, 1)
    fkl = (lp.exp() * (lp - lq)).sum(-1)
    bkl = (lq.exp() * (lq - lp)).sum(-1)
    want = (conf * bkl + (1 - conf) * fkl).mean()
    assert torch.allclose(G.kd_loss(s, t, 1.0, "cakld", y), want, atol=1e-6)
    with pytest.raises(ValueError, match="labels"):
        G.kd_loss(s, t, 1.0, "cakld")


def test_cakld_tracks_teacher_confidence():
    s = _logits(5, (1, 1, 6))
    sure = torch.full((1, 1, 6), -50.0); sure[..., 2] = 50.0         # teacher certain of token 2
    y = torch.tensor([[2]])
    assert torch.allclose(G.kd_loss(s, sure, 1.0, "cakld", y), G.kd_loss(s, sure, 1.0, "rkl"), atol=1e-4)
    y_other = torch.tensor([[4]])                                    # label the teacher gives ~0 to
    assert torch.allclose(G.kd_loss(s, sure, 1.0, "cakld", y_other), G.kd_loss(s, sure, 1.0, "kl"), atol=1e-4)


def test_topk_variants():
    s, t = _logits(6), _logits(7)
    V = s.shape[-1]
    assert torch.allclose(G.kd_loss(s, t, 1.0, "kl_top_k", topk=V), _kl(t, s), atol=1e-5), "k = V is plain KL"
    assert torch.allclose(G.kd_loss(s, t, 1.0, f"kl_top_{V}"), _kl(t, s), atol=1e-5)
    assert torch.allclose(G.kd_loss(s, t, 1.0, "r_kl_top_k", topk=V), _kl(s, t), atol=1e-5)
    small = G.kd_loss(s, t, 1.0, "kl_top_3")
    assert torch.isfinite(small) and small >= 0
    with pytest.raises(ValueError):
        G.kd_loss(s, t, 1.0, "kl_top_k", topk=0)


def test_every_loss_has_finite_gradients_and_padded_vocab_is_tolerated():
    t = _logits(8, (2, 5, 13))                                       # teacher vocab padded wider
    y = torch.randint(0, 11, (2, 5))
    for name in ("kl", "rkl", "cakld", "kl_top_k", "r_kl_top_k"):
        s = _logits(9).requires_grad_(True)
        G.kd_loss(s, t, 2.0, name, y, 4).backward()
        assert torch.isfinite(s.grad).all(), name


def test_cli_validates_loss_names():
    import argparse
    assert G._loss_name("CAKLD") == "cakld" and G._loss_name("kl_top_64") == "kl_top_64"
    with pytest.raises(argparse.ArgumentTypeError):
        G._loss_name("mse")


def test_distill_cli_records_the_loss_in_the_lineage(tmp_path):
    import json
    import os
    from transformers import AutoTokenizer
    tok = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        AutoTokenizer.from_pretrained(tok, local_files_only=True)
    except Exception:
        pytest.skip("no cached tokenizer to build a model with")
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n\n".join(f"sample {i}: the quick brown fox jumps over the lazy dog" for i in range(50)))
    teacher, student = str(tmp_path / "t"), str(tmp_path / "s")
    for out in (teacher, student):
        _cli(["new", "--family", "llama", "--layers", "1", "--hidden", "32", "--heads", "2", "--kv-heads", "1",
              "--ffn", "64", "--ctx", "64", "--tokenizer", tok, "--out", out])
    _cli(["train", "--model", student, "--teacher", teacher, "--loss", "cakld", "--lm-loss-weight", "1.0",
          "--kd", "1.0", "--data", str(corpus), "--steps", "2", "--seq", "16", "--batch", "2",
          "--eval-batches", "1", "--device", "cpu", "--log-every", "100"])
    step = json.load(open(os.path.join(student, "forge.json")))["steps"][-1]
    assert step["op"] == "distill" and step["loss"] == "cakld" and step["lm_loss_weight"] == 1.0


def _cli(args):
    old = sys.argv
    sys.argv = ["pollard-forge"] + list(args)
    try:
        G.main()
    finally:
        sys.argv = old
