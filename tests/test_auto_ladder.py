"""pollard --ladder: the whole publish ladder from one command.

Until 2026-10-09 the one-shot built ONE rung (--ram 16) and every published ladder -- FrogMini,
FrogNano, clef-flash -- was finished by hand-written batch files with --ram values found by trial and
error. The ladder search uses pollard-fit's own planner; against the real clef-flash f16 (18.2 GB) it
picked Q6_K 7.5 / IQ4_XS 5.6 / IQ3_S 4.8 / IQ2_XXS 3.7 GB, the rungs chosen by hand that night.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as pa  # noqa: E402

# (ram, preset, projected GB) as the planner reported them for clef-flash, thinned
POINTS = [(9.5, "Q6_K", 7.5), (8.9, "Q6_K", 7.49), (8.0, "Q5_K_M", 6.8), (7.0, "IQ4_XS", 5.9),
          (6.6, "IQ4_XS", 5.61), (6.3, "IQ3_S", 5.4), (5.7, "IQ3_S", 4.83), (5.6, "IQ2_S", 4.7),
          (4.6, "IQ2_XXS", 3.9), (4.3, "IQ2_XXS", 3.65), (3.0, "Q2_K", 2.6)]
F16 = 18.2


def test_one_rung_per_label_nearest_its_size():
    got = pa.pick_ladder(POINTS, F16)
    assert {k: v[1] for k, v in got.items()} == {"Q6_K": "Q6_K", "IQ4_XS": "IQ4_XS",
                                                 "IQ3_S": "IQ3_S", "IQ2": "IQ2_XXS"}
    assert got["IQ4_XS"][0] == 6.6 and got["IQ3_S"][0] == 5.7 and got["IQ2"][0] == 4.3


def test_flagship_takes_the_two_bit_rung_where_ik_builds_it():
    got = pa.pick_ladder(POINTS, F16, ik_flagship=True)
    assert "IQ2" not in got and set(got) == {"Q6_K", "IQ4_XS", "IQ3_S"}


def test_unreachable_label_is_absent_not_mislabelled():
    got = pa.pick_ladder([p for p in POINTS if p[1] != "IQ3_S"], F16)
    assert "IQ3_S" not in got and got["IQ4_XS"][1] == "IQ4_XS"
