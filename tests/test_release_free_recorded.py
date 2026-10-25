"""--free-after-upload frees what the release RECORDED as built and confirmed on the Hub, wherever it is.

The Nemotron-3.5-Bolt release (2026-10-24) built its passthrough rungs from a vendor GGUF in
C:\\pollard\\downloads\\bolt-24gb-gguf\\ -- next to that source -- uploaded PT75 and confirmed it on the Hub, then
printed `(kept ...-Pollard-PT75.gguf: not a file this release built)`: the free step judged by directory, the rung was
outside the workspace, 38 GB stayed and the next rung was held for disk. Now the decision is the MANIFEST's: the
recorded output path, the size + sha256 the upload was confirmed with -- never the vendor source, never a file that
is not a recorded build output. Fakes only (test_release_free's fake Hub)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import pollard_release as R  # noqa: E402
import pollard_workspace as ws  # noqa: E402
from test_release_free import hub, inputs, model, plan  # noqa: E402,F401 -- shared fixtures

REPO = "PollardWeights/Tiny-1B-Pollard"


def _rung_beside_the_source(c, tag="PT75", body=b"GGUF" + b"\x07" * 4000):
    """A verified rung where the Bolt release wrote it: next to the vendor GGUF (the plan's f16), outside models/."""
    p = os.path.join(os.path.dirname(c["f16"]), f"Tiny-1B-Pollard-{tag}.gguf")
    Path(p).write_bytes(body)
    ws.record_build(c["f16"], "gguf", p, tag=tag)
    ws.record_gate(p, {"pass": True})
    return p


def _publish(c, cfg, path, tag="PT75", said=None):
    return R.publish_rung(R._publish_rung_step(c, "gguf", tag, path), c, cfg, True,
                          out=(said.append if said is not None else lambda *_: None))


def test_a_recorded_rung_outside_the_workspace_is_freed_after_a_confirmed_upload(model, inputs, hub):
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    p = _rung_beside_the_source(c)
    assert not p.startswith(ws.pollard_home())
    said = []
    assert _publish(c, cfg, p, said=said) == "done"
    assert "Tiny-1B-Pollard-PT75.gguf" in hub[REPO]
    assert not os.path.exists(p), "uploaded + confirmed by size + sha256: freed wherever it was written"
    assert any(x.strip().startswith("freed ") for x in said) and not any("(kept" in x for x in said)
    b = R.gguf_builds(c)[0]
    assert b["freed"].startswith("uploaded") and b["confirmed_files"]["Tiny-1B-Pollard-PT75.gguf"]["size"] == 4004
    assert os.path.exists(c["f16"]), "the vendor source beside it stays"


def test_the_source_gguf_is_never_freed_even_when_recorded_and_uploaded(model, inputs, hub):
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    src = c["f16"]
    ws.record_build(src, "gguf", src, tag="PT100")              # the 100% rung IS the source file
    ws.record_gate(src, {"pass": True})
    said = []
    assert _publish(c, cfg, src, tag="PT100", said=said) == "done"
    assert os.path.exists(src), "the vendor GGUF is an input -- never deleted"
    assert any("(kept" in x and "input" in x for x in said)
    assert not R.gguf_builds(c)[0].get("freed")


def test_a_file_not_recorded_as_a_build_output_is_never_freed(model, inputs, hub):
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    stray = os.path.join(os.path.dirname(c["f16"]), "Tiny-1B-Pollard-Q4_K.gguf")   # someone put it there
    Path(stray).write_bytes(b"GGUF" * 10)
    ok, why = R._freeable(c, c["f16"], {"path": stray, "lane": "gguf"})
    assert not ok and "not a build output" in why
    with pytest.raises(RuntimeError, match="not in the MANIFEST"):
        _publish(c, cfg, stray, tag="Q4_K")
    assert os.path.exists(stray)
    inws = os.path.join(ws.model_dir(c["f16"], create=True), "Tiny-1B-Pollard-Q5_K.gguf")
    Path(inws).write_bytes(b"GGUF" * 10)                                         # in models/ is not enough either
    assert not R._freeable(c, c["f16"], {"path": inws, "lane": "gguf"})[0]


def test_a_rung_that_changed_after_its_upload_was_confirmed_is_kept(model, inputs, hub, monkeypatch):
    """The recorded size is checked at free time: a file rewritten since the confirmation is not the one on the Hub."""
    cfg = plan(inputs)
    c, _, _ = R.plan_steps(cfg, model, {"gguf"})
    p = _rung_beside_the_source(c)
    orig = ws.update_build

    def rewrite_then_record(key, path, patch, remove=()):
        if "uploaded" in patch and os.path.exists(path):
            with open(path, "ab") as f:
                f.write(b"more")
        return orig(key, path, patch, remove)
    monkeypatch.setattr(ws, "update_build", rewrite_then_record)
    said = []
    assert _publish(c, cfg, p, said=said) == "done"
    assert os.path.exists(p) and any("changed size" in x for x in said)
