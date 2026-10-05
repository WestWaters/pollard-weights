"""pollard-forge: new / prune / train / card on tiny random models (CPU, seconds)."""
import json, os, sys, tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_forge as G

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _tiny(family="llama", layers=4):
    from transformers import AutoModelForCausalLM
    cfg = G.build_config(family, layers, 64, 4, 2, 128, 256, ctx=128)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg)


def test_sizes_are_valid_shapes():
    for name, (L, H, nh, kv, ffn) in G.SIZES.items():
        assert H % nh == 0 and nh % kv == 0, name
    cfg = G.build_config("qwen3", 2, 128, 4, 2, 256, 1024)
    assert cfg.model_type == "qwen3" and cfg.head_dim == 32 and cfg.tie_word_embeddings


def test_parse_source():
    assert G.parse_source("hf:HuggingFaceFW/fineweb-edu@0.6") == {
        "kind": "hf", "name": "HuggingFaceFW/fineweb-edu", "config": None, "split": "train", "weight": 0.6,
        "spec": "hf:HuggingFaceFW/fineweb-edu"}
    s = G.parse_source("hf:org/ds:cfg:validation")
    assert (s["config"], s["split"], s["weight"]) == ("cfg", "validation", 1.0)
    with pytest.raises(SystemExit):
        G.parse_source("/no/such/file.txt")


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_prune_shapes_and_forward(family):
    m = _tiny(family, layers=6)
    bi = [0.5, 0.1, 0.4, 0.05, 0.3, 0.6]                      # layers 3 and 1 change the state least
    act = [torch.arange(128, dtype=torch.float32) for _ in range(6)]
    keep, drop = G.prune_model(m, bi, act, keep_layers=4, ffn_frac=0.5)
    assert sorted(drop) == [1, 3] and keep == [0, 2, 4, 5]
    assert m.config.num_hidden_layers == 4 and m.config.intermediate_size == 64
    layers, _ = G._decoder_layers(m)
    assert [l.self_attn.layer_idx for l in layers] == [0, 1, 2, 3]      # KV slots renumbered
    assert layers[0].mlp.gate_proj.weight.shape == (64, 64) and layers[0].mlp.down_proj.weight.shape == (64, 64)
    out = m(torch.randint(0, 256, (1, 12)))                              # still runs, generation cache included
    assert out.logits.shape == (1, 12, 256)
    m.generate(torch.randint(0, 256, (1, 4)), max_new_tokens=3, do_sample=False)


def test_prune_keeps_highest_activation_channels():
    m = _tiny(layers=2)
    up = m.model.layers[0].mlp.up_proj.weight.detach().clone()
    act = [torch.zeros(128), torch.zeros(128)]
    act[0][[5, 70, 100]] = 9.0; act[0][[1, 2]] = 5.0
    G.prune_model(m, [1, 1], act, keep_layers=2, ffn_frac=0.5, protect_ends=False)
    kept = m.model.layers[0].mlp.up_proj.weight
    for ch in (5, 70, 100, 1, 2):
        assert any(torch.equal(kept[i], up[ch]) for i in range(kept.shape[0])), f"channel {ch} dropped"


def test_train_and_card_roundtrip():
    from transformers import AutoTokenizer
    tokdir = None
    for cand in ("HuggingFaceTB/SmolLM2-135M-Instruct",):
        try:
            AutoTokenizer.from_pretrained(cand, local_files_only=True); tokdir = cand
        except Exception:
            pass
    if not tokdir:
        pytest.skip("no cached tokenizer to build a model with")
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "c.txt")
        open(corpus, "w").write("\n\n".join(f"sample {i}: the quick brown fox jumps over the lazy dog" for i in range(200)))
        out = os.path.join(d, "m")
        _cli(["new", "--family", "llama", "--layers", "2", "--hidden", "64", "--heads", "4", "--kv-heads", "2",
              "--ffn", "128", "--ctx", "128", "--tokenizer", tokdir, "--out", out])
        _cli(["train", "--model", out, "--data", corpus, "--steps", "6", "--seq", "32", "--batch", "2",
              "--lr", "3e-3", "--eval-batches", "1", "--device", "cpu", "--log-every", "100"])
        log = json.load(open(os.path.join(out, "forge.json")))
        assert [s["op"] for s in log["steps"]] == ["new", "train"]
        tr = log["steps"][1]
        assert tr["eval_loss_end"] < tr["eval_loss_start"], "a few steps on a repetitive corpus must lower the loss"
        with pytest.raises(SystemExit):                                    # never invent a license
            _cli(["card", "--model", out, "--name", "T"])
        _cli(["card", "--model", out, "--name", "T", "--license", "apache-2.0"])
        card = open(os.path.join(out, "README.md")).read()
        assert "license: apache-2.0" in card and "| 2 | train |" in card and d not in card, "no local paths in a card"


def _cli(args):
    old = sys.argv
    sys.argv = ["pollard-forge"] + list(args)
    try:
        G.main()
    finally:
        sys.argv = old
