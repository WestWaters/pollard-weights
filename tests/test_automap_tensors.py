"""automap should produce its own tensor listing, and choose a type that works.

Two things went wrong in the same flow and both were the tool refusing where it could have
decided. --tensors was REQUIRED, so every caller had to know to run `llama-quantize --dry-run`
first. And a dry-run at IQ1_S is refused outright without an importance matrix -- llama-quantize
prints the refusal instead of the tensor names, so the caller gets an empty file and no reason.

We only ever wanted names.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "tools/pollard_automap.py").read_text()


def test_tensors_is_no_longer_required():
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", "") == "--tensors"):
            for kw in node.keywords:
                assert kw.arg != "required" or kw.value.value is False, \
                    "--tensors is still required"
            return
    raise AssertionError("--tensors argument not found")


def test_it_makes_the_listing_when_none_was_given():
    assert "def tensor_list(" in SRC
    assert "a.tensors or tensor_list(" in SRC


def test_it_falls_back_to_a_type_that_needs_no_imatrix():
    """IQ1_S is refused without one. Q8_0 lists the same tensors and never is."""
    body = SRC[SRC.index("def tensor_list("):SRC.index("def parse_tensors")]
    assert "Q8_0" in body, "no imatrix-free fallback"
    assert "attempts" in body and "insert(0" in body, "the imatrix path should be tried first"


def test_it_verifies_it_actually_got_tensor_names():
    """An exit code of 0 is not evidence: the refusal prints and exits cleanly."""
    body = SRC[SRC.index("def tensor_list("):SRC.index("def parse_tensors")]
    assert re.search(r"re\.search\(r?\"+blk", body) or "blk\\\\.\\\\d+\\\\." in body, \
        "it should check the output contains tensor names"


def test_failure_says_what_the_binary_actually_said():
    """'could not list the tensors' with no quote from the tool is not a usable message."""
    body = SRC[SRC.index("def tensor_list("):SRC.index("def parse_tensors")]
    assert "last[:300]" in body


def test_the_new_code_stays_ascii():
    body = SRC[SRC.index("def tensor_list("):SRC.index("def parse_tensors")]
    assert all(ord(c) < 128 for c in body)
