"""Resume: a six-hour build that dies at 90% should not start over.

The weights are the easy half. What actually has to be preserved is `inps` -- the calibration set
propagated through every block quantized so far. Restoring weights without it puts the model in
the right state and the sequence in the wrong one, and the blocks after the resume point would be
solved against activations from an unquantized model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "tools/pollard_gptq.py").read_text()


def test_the_flags_exist():
    assert '"--work-dir"' in SRC and '"--resume"' in SRC


def test_the_flags_reach_the_solver():
    assert re.search(r"work_dir=a\.work_dir, resume=a\.resume", SRC)
    assert "def sequential_gptq(" in SRC
    sig = SRC[SRC.index("def sequential_gptq("):]
    sig = sig[:sig.index("):")]
    assert "work_dir=None" in sig and "resume=False" in sig


def test_activations_are_checkpointed_not_just_weights():
    """Restoring weights alone resumes the model but not the place in the sequence."""
    assert 'torch.save([t.half() for t in inps]' in SRC, "inps must be saved"
    assert 'torch.load(os.path.join(work_dir, "inps.pt")' in SRC, "and restored"


def test_a_checkpoint_is_written_after_the_block_is_finished():
    """A checkpoint that describes a half-quantized block is worse than none."""
    i = SRC.index("if work_dir:\n            # written AFTER")
    before = SRC[:i]
    # the propagation of this block's outputs must already have happened
    assert "inps = [fwd(layer, inp.to(dev)).cpu() for inp in inps]" in before


def test_progress_is_written_atomically():
    """A torn progress file resumes nowhere."""
    assert "os.replace(tmp, prog)" in SRC


def test_resuming_across_different_settings_is_refused():
    """Half a 4-bit model and half a 2-bit one loads fine and is merely wrong."""
    assert "def _ckpt_fingerprint(" in SRC
    assert 'st.get("fingerprint") != fp' in SRC
    assert "half one and half the other" in SRC


def test_the_fingerprint_covers_everything_that_changes_the_arithmetic():
    seg = SRC[SRC.index("def _ckpt_fingerprint("):]
    seg = seg[:seg.index("\n\n@")]
    for field in ("bits", "groupsize", "act_order", "qmode", "recipe", "n_calib", "n_layers"):
        assert field in seg, f"fingerprint ignores {field}"


def test_the_fingerprint_actually_separates_runs():
    seg = SRC[SRC.index("def _ckpt_fingerprint("):]
    seg = seg[:seg.index("\n\n@")]
    ns = {}
    exec(seg, ns)
    f = ns["_ckpt_fingerprint"]
    base = f(4, 128, False, "int", None, 128, 32)
    assert f(4, 128, False, "int", None, 128, 32) == base
    for changed in (f(2, 128, False, "int", None, 128, 32),
                    f(4, 64, False, "int", None, 128, 32),
                    f(4, 128, True, "int", None, 128, 32),
                    f(4, 128, False, "ternary", None, 128, 32),
                    f(4, 128, False, "int", {"x": 1}, 128, 32),
                    f(4, 128, False, "int", None, 256, 32),
                    f(4, 128, False, "int", None, 128, 40)):
        assert changed != base


def test_without_a_work_dir_nothing_is_written():
    """Checkpointing costs real disk -- the activations are large. It stays opt-in."""
    assert SRC.count("if work_dir:") >= 2
    assert "work_dir=None" in SRC


def test_the_resume_args_go_to_the_solver_not_to_endswith():
    """A regex edit once put them inside method.endswith(...), which parses and then fails at
    runtime. Check the call graph, not the text."""
    import ast
    tree = ast.parse(SRC)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "sequential_gptq"]
    assert calls, "no call to sequential_gptq found"
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert "work_dir" in kw and "resume" in kw
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(getattr(n, "func", None), "attr", "") == "endswith":
            assert not n.keywords, "str.endswith does not take keyword arguments"
