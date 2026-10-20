"""Explicit precision tiers must obey the same weight budget as allocated tiers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from pollard_calc import analyse
from pollard_fit import plan_allocation


def arch():
    return analyse({"hidden_size": 4096, "num_hidden_layers": 32,
                    "num_attention_heads": 32, "num_key_value_heads": 8,
                    "intermediate_size": 14336, "vocab_size": 32000})


@pytest.mark.parametrize("emb", [None, "q8_0"])
def test_all_pinned_tiers_reject_an_impossible_budget(emb):
    tiers = {"bulk": "q8_0", "attn": "q8_0"}
    if emb:
        tiers["emb"] = emb
    with pytest.raises(SystemExit, match="fit"):
        plan_allocation(arch(), 8, 3, tiers=tiers, emb_imatrix_ok=False)


def test_all_pinned_tiers_still_fit_a_sufficient_budget():
    result = plan_allocation(arch(), 16, 3, tiers={"bulk": "q8_0", "attn": "q8_0"},
                             emb_imatrix_ok=False)
    assert result[2] <= 16 * 0.85 - 3
    assert all(t == "q8_0" for _, t in result[0])


def test_unpinned_embeddings_can_descend_to_fit_reserved_tiers():
    model = arch()
    # Reserve the body at q6_K, leaving room for a lower embedding rung only.
    tiers = {"bulk": "q6_K", "attn": "q6_K"}
    full = plan_allocation(model, 16, 0, tiers=tiers, emb_imatrix_ok=False)
    budget = full[2] - 0.05
    result = plan_allocation(model, budget / 0.85, 0, tiers=tiers, emb_imatrix_ok=False)
    assert result[2] <= budget
    assert result[1] != "q6_K"
