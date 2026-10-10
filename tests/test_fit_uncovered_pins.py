"""pollard-fit's uncovered-tensor guard actually takes effect.

Cloudflare/clef-flash, 2026-10-09: the IQ3_S / IQ2 rungs died on "Missing importance matrix for tensor
dec.blk.0.ffn_down.weight". The guard had found the decision head's tensors and pinned them -- at the END
of the --tensor-type list, where llama-quantize (first match wins) never reached them, because the body's
unanchored `blk\\.0\\.ffn_down\\.weight=iq2_s` matched `dec.blk.0.ffn_down.weight` first. And the IQ4_XS
rung built "fine" with the head at an uncalibrated low-bit type.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_fit as F  # noqa: E402

NAMES = ["token_embd.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight",
         "dec.blk.0.ffn_down.weight", "dec.blk.0.cross_attn_q.weight", "output.weight"]
COVERED = {"blk.0.ffn_down.weight", "blk.0.attn_q.weight"}


def _first_match(overrides, name, default):
    for p, t in overrides:
        if re.search(p, name):
            return t
    return default


def test_pins_win_under_first_match():
    ov = [(r"blk\.0\.ffn_down\.weight", "iq2_s"), (r"blk\.0\.attn_.*", "iq3_s")]
    out, n = F.pin_uncovered(ov, NAMES, COVERED, "IQ3_S")
    assert n == 2
    assert _first_match(out, "dec.blk.0.ffn_down.weight", "iq3_s") == F.EMB_FLOOR
    assert _first_match(out, "dec.blk.0.cross_attn_q.weight", "iq3_s") == F.EMB_FLOOR
    assert _first_match(out, "blk.0.ffn_down.weight", "iq3_s") == "iq2_s"        # the body keeps its plan


def test_uncovered_low_bit_that_would_not_crash_is_held_too():
    ov = [(r"blk\.0\.ffn_down\.weight", "iq4_xs")]
    out, n = F.pin_uncovered(ov, NAMES, COVERED, "IQ4_XS")
    assert _first_match(out, "dec.blk.0.ffn_down.weight", "iq4_xs") == F.EMB_FLOOR


def test_uncovered_already_at_or_above_floor_is_left_alone():
    ov = [(r"blk\.0\.ffn_down\.weight", "q8_0")]
    out, n = F.pin_uncovered(ov, ["dec.blk.0.ffn_down.weight"], COVERED, "Q8_0")
    assert n == 0 and out == ov


def test_embeddings_and_output_are_not_this_guards_business():
    out, n = F.pin_uncovered([], ["token_embd.weight", "output.weight"], set(), "IQ2_XXS")
    assert n == 0
