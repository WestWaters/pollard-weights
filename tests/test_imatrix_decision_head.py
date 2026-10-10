"""The auto-imatrix step sizes llama-imatrix for a decision head instead of crashing on it.

Cloudflare/clef-flash on the box, 2026-10-09: clef runs in embedding mode, llama.cpp forces
n_batch = n_ubatch = 512, and llama-imatrix died on GGML_ASSERT(params.n_ctx == n_seq * n_ctx) before
the first chunk -- Pollard stopped with "llama-imatrix failed (exit 3221226505)". With -c/-b/-ub 512 it
ran. A plain model gets no extra flags.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pollard_auto as pa  # noqa: E402


def test_decision_head_gets_embedding_shape(monkeypatch):
    monkeypatch.setattr(pa, "read_gguf_meta", lambda p: {"general.architecture": "clef",
                                                        "clef.decision.head_count": 8})
    assert pa._imatrix_shape("clef.gguf") == ["-c", "512", "-b", "512", "-ub", "512"]


def test_plain_model_untouched(monkeypatch):
    monkeypatch.setattr(pa, "read_gguf_meta", lambda p: {"general.architecture": "qwen35",
                                                        "qwen35.block_count": 32})
    assert pa._imatrix_shape("qwen.gguf") == []


def test_unreadable_gguf_untouched(monkeypatch):
    def boom(p):
        raise OSError("no such file")
    monkeypatch.setattr(pa, "read_gguf_meta", boom)
    assert pa._imatrix_shape("missing.gguf") == []
