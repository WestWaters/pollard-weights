"""Conv/attention hybrids (LFM2/LFM3): attention is counted on the blocks that have it, and no attention block sits
below the base type the bulk settles on -- a text sensitivity that rates it cheap put LFM3-Audio-2B's eleven attention
blocks at iq2_xxs (KLD 0.226; held at the base type: 0.080 at the same size, 2026-10-29)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from pollard_calc import analyse
from pollard_fit import BPW, plan_allocation
F_BPW = BPW

ATTN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def lfm3(attn_layers=ATTN):
    return analyse({"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 32,
                    "num_key_value_heads": 8, "intermediate_size": 7680, "vocab_size": 128000,
                    "_attn_layers": attn_layers})


def cheap_attention():
    """The imatrix-route profile's shape: attention nearly free, the FFNs expensive."""
    return {"ffn": {str(i): 1.0 for i in range(24)}, "attn": {str(i): 0.001 for i in range(24)}}


def attn_types(overrides):
    out = {}
    for pat, t in overrides:
        if ".attn_" in pat and pat.endswith("attn_.*"):
            out[int(pat.split("\\.")[1])] = t
    return out


def test_attention_on_only_some_blocks_marks_a_hybrid():
    a = lfm3()
    assert a["hybrid"] and a["attn_layers"] == ATTN and a["n_full"] == len(ATTN)
    dense = lfm3(attn_layers=list(range(24)))
    assert not dense["hybrid"] and dense["attn_layers"] == []


def test_hybrid_attention_never_below_the_base_type():
    overrides, _, gb, _, (summary, src) = plan_allocation(lfm3(), 1.0, 0, sensitivity=cheap_attention())
    at = attn_types(overrides)
    assert sorted(at) == ATTN                                  # attention items only where attention exists
    bulk = [t for p, t in overrides if ".ffn_" in p]
    base = max(set(bulk), key=bulk.count)
    assert all(BPW[t] >= BPW[base] for t in at.values()), (at, base)
    assert "hybrid attention floor" in src
    assert gb <= 1.0 * 0.85 + 1e-6 or gb <= 1.0                # still inside the budget


def test_dense_models_are_untouched():
    dense = lfm3(attn_layers=list(range(24)))
    overrides, _, _, _, (_, src) = plan_allocation(dense, 1.0, 0, sensitivity=cheap_attention())
    assert "hybrid attention floor" not in src
    assert len(attn_types(overrides)) == 24


def test_four_bit_rungs_raise_attention_a_step_and_lower_nothing_else():
    """4-bit and up: attention one step above the base, its bytes ADDED -- paying with the early FFNs made LFM3-Omni
    cut sentences short (2026-10-30)."""
    import pollard_fit as F
    a = lfm3()
    with_floor = plan_allocation(a, 1.6, 0, sensitivity=cheap_attention())
    no_floor = plan_allocation(dict(a, hybrid=False, attn_layers=[]), 1.6, 0, sensitivity=cheap_attention())
    bulk = [t for p, t in with_floor[0] if ".ffn_" in p]
    base = max(set(bulk), key=bulk.count)
    assert F.BPW[base] >= F.BPW["iq4_xs"]
    step = F.LADDER[F.LADDER.index(base) - 1]
    at = attn_types(with_floor[0])
    assert all(at[i] == step or F.BPW[at[i]] > F.BPW[step] for i in ATTN)     # the real attention blocks
    ffn = lambda ov: {p: t for p, t in ov if ".ffn_" in p}
    assert ffn(with_floor[0]) == ffn(no_floor[0])
    assert "bytes added" in with_floor[4][1]


def mixer_lfm3():
    a = lfm3()
    a["mixers"] = {i: [(f"blk.{i}.shortconv.in_proj.weight", 6291456), (f"blk.{i}.shortconv.out_proj.weight", 2097152)]
                   for i in range(24) if i not in ATTN}
    return a


def test_hybrid_mixers_are_counted_at_the_base_type_not_contested():
    """3-bit: the shortconv projections ride at the base type and their bytes are budgeted (bidding them against the
    FFNs cost LFM3's Japanese long-form speech). 4-bit and up: they stay outside the plan at the rung's own type, as the
    rung that matched the old one in every language had them (2026-10-31)."""
    ov, _, gb, _, (_, src) = plan_allocation(mixer_lfm3(), 1.2, 0, sensitivity=cheap_attention())
    mix = {p: t for p, t in ov if "shortconv" in p}
    bulk = [t for p, t in ov if ".ffn_" in p]
    base = max(set(bulk), key=bulk.count)
    assert F_BPW[base] < F_BPW["iq4_xs"] and mix and set(mix.values()) == {base}, (set(mix.values()), base)
    assert "mixers pinned" in src and sorted(attn_types(ov)) == ATTN
    ov4, _, _, _, (_, src4) = plan_allocation(mixer_lfm3(), 1.6, 0, sensitivity=cheap_attention())
    assert not [p for p, _ in ov4 if "shortconv" in p] and "bytes added" in src4
