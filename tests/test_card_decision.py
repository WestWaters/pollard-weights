"""pollard-card shows a decision model as one: measured on decisions, served on /v1/systemone.

clef-flash, 2026-10-09: the card came out as a chat model -- pipeline_tag text-generation, tagged
conversational / ik_llama.cpp / trellis on a stock-only ladder, "Perplexity measured: yes" with every PPL
blank, and a llama-cli chat example for a server that only answers /v1/systemone.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_card as C  # noqa: E402

BUILDS = [{"name": "clef-flash-Pollard-Q6_K.gguf", "tag": "Q6_K", "bytes": 7_459_559_840},
          {"name": "clef-flash-Pollard-IQ2_XXS.gguf", "tag": "IQ2_XXS", "bytes": 3_358_742_944}]
RESULTS = {"clef-flash-Pollard-Q6_K.gguf": {"decision": {"n": 20, "agreement": 1.0, "option_kl": 4.5e-05,
                                                         "mean_abs_drift": 0.00075, "accuracy": 1.0}},
           "clef-flash-Pollard-IQ2_XXS.gguf": {"decision": {"n": 20, "agreement": 1.0, "option_kl": 0.0324,
                                                            "mean_abs_drift": 0.033, "accuracy": 1.0}}}


def _tags(fm):
    return [l[2:] for l in fm.splitlines() if l.startswith("- ")]


def test_decision_frontmatter():
    fm = C.frontmatter("Cloudflare/clef-flash", "apache-2.0", ["gguf"], "qwen3_5", "PollardWeights",
                       "text-classification", decision=True, ik=False)
    t = _tags(fm)
    assert "decision-model" in t and "systemone" in t
    assert "conversational" not in t and "ik_llama.cpp" not in t and "trellis" not in t
    assert "pipeline_tag: text-classification" in fm


def test_chat_model_frontmatter_unchanged():
    t = _tags(C.frontmatter("x/y", "mit", ["gguf"], "qwen3", None, "text-generation"))
    assert "conversational" in t and "ik_llama.cpp" in t and "decision-model" not in t


def test_decision_section_reports_every_rung_largest_first():
    md = "\n".join(C.decision_section(BUILDS, RESULTS))
    assert "## Decision fidelity" in md and "same 20 typed questions" in md
    assert md.index("Q6_K.gguf") < md.index("IQ2_XXS.gguf")
    assert "| 4.5e-05 |" in md and "| 0.032 |" in md and "| 100% |" in md


def test_decision_usage_is_systemone_not_chat():
    md = "\n".join(C.decision_usage("clef-flash-Pollard-Q6_K.gguf", "clef"))
    assert "/v1/systemone" in md and "llama-cli" not in md and "`clef`" in md


def test_kl_format_keeps_small_values_readable():
    assert C._fmt_kl(4.5e-05) == "4.5e-05" and C._fmt_kl(0.0033) == "0.0033" and C._fmt_kl(0.0324) == "0.032"
    assert C._fmt_kl(None) == "--"


def test_library_is_declared_not_guessed():
    """The Hub guessed the library from tags: "trellis" filed ik repos under Microsoft TRELLIS (3D), and a repo
    without it (clef-flash-Pollard, 2026-10-10) read "Downloads are not tracked for this model"."""
    assert "library_name: gguf" in C.frontmatter("x/y", "mit", ["gguf"], "qwen3", None)
    assert "library_name: gguf" in C.frontmatter("x/y", "mit", ["gguf"], "qwen3", None, decision=True, ik=False)
    assert "library_name: mlx" in C.frontmatter("x/y", "mit", ["mlx"], "qwen3", None)
    assert "library_name" not in C.frontmatter("x/y", "mit", ["gptq"], "qwen3", None)


def test_decision_reference_is_named_on_the_card():
    """Clef 27B was gated against Q8_0 (the 54 GB f16 cannot be served on the build box); the card said "f16"."""
    res = dict(RESULTS, _decision_ref="Q8_0")
    md = "\n".join(C.decision_section(BUILDS, res))
    assert "agrees with Q8_0" in md and "option KL vs Q8_0" in md and "f16" not in md
    assert "agrees with f16" in "\n".join(C.decision_section(BUILDS, RESULTS))
