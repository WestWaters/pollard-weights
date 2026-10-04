"""Build the local, pre-tokenized datasets for the tPD decomposition + the quant-arm eval sets.

Why local + pre-tokenized: tPD's data loader calls `load_dataset(name, split=...)` with no config
name and no mixing, so (a) wikitext-103 (needs a config name) and (b) a wikitext+chat *mix* cannot
be expressed in its YAML directly. We tokenize once with the Qwen3 tokenizer, pack into fixed-length
rows, and write Hub-layout parquet (`data/train-*.parquet`, `data/validation-*.parquet`) that
`load_dataset("<dir>", split="train")` picks up. `is_tokenized: true` in the config.

Document-level split (no leakage): every 10th target document goes to the HELD-OUT eval pool used
only by quant_arms.py; the rest feed the decomposition (train/validation) and calibration.
Non-target eval comes from wikitext-103 *test* + ultrachat *test_sft*; non-target train from the
train splits.

Outputs under --out:
  target_code/data/{train,validation}-00000-of-00001.parquet     (input_ids, len --train-seq)
  nontarget_mix/data/{train,validation}-00000-of-00001.parquet
  eval_sets.pt  {target_eval, nontarget_eval, target_calib, generic_calib: LongTensor[n, eval_seq], meta}

Usage (box):
  C:\\pollard\\tpd-venv\\Scripts\\python.exe prep_data.py --out C:/pollard/tpd_data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch

TARGET_SOURCES = [
    # (dataset, config, split, text column) -- first that loads wins. Both are ungated Python-only.
    ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
    ("codeparrot/codeparrot-clean", None, "train", "content"),
]
WIKI = ("Salesforce/wikitext", "wikitext-103-raw-v1")
CHAT = "HuggingFaceH4/ultrachat_200k"


# ----------------------------------------------------------------------------- packing helpers
def pack(token_streams: Iterable[list[int]], seq_len: int, max_rows: int, eos: int) -> list[list[int]]:
    """Concatenate docs separated by EOS and cut into fixed rows (drops the ragged tail)."""
    rows: list[list[int]] = []
    buf: list[int] = []
    for toks in token_streams:
        buf.extend(toks)
        buf.append(eos)
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(rows) >= max_rows:
                return rows
    return rows


def write_split_parquet(rows: list[list[int]], out_dir: Path, split: str) -> Path:
    """Write rows as Hub-layout parquet so `load_dataset(str(out_dir.parent), split=split)` works."""
    from datasets import Dataset

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{split}-00000-of-00001.parquet"
    Dataset.from_dict({"input_ids": rows}).to_parquet(str(path))
    return path


# ----------------------------------------------------------------------------- sources
def _stream(name: str, config: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(name, config, split=split, streaming=True)


def target_docs(log: list[str]) -> Iterator[str]:
    last_err: Exception | None = None
    for name, cfg, split, col in TARGET_SOURCES:
        try:
            ds = _stream(name, cfg, split)
            first = next(iter(ds))
            assert col in first, f"column {col} missing"
        except Exception as e:  # noqa: BLE001 - fall through to next source
            last_err = e
            log.append(f"target source {name} failed: {e!r}")
            continue
        log.append(f"target source = {name}:{split}[{col}]")
        for ex in ds:
            text = ex[col]
            if text and len(text) > 200:
                yield text
        return
    raise RuntimeError(f"no target source loaded; last error {last_err!r}")


def wiki_docs(split: str) -> Iterator[str]:
    """wikitext rows are paragraphs/headings; glue them into ~article-sized docs."""
    buf: list[str] = []
    for ex in _stream(WIKI[0], WIKI[1], split):
        t = ex["text"]
        if t.startswith(" = ") and not t.startswith(" = = ") and buf:
            doc = "".join(buf)
            if doc.strip():
                yield doc
            buf = []
        buf.append(t)
    if buf and "".join(buf).strip():
        yield "".join(buf)


def chat_docs(split: str, tok) -> Iterator[str]:
    for ex in _stream(CHAT, None, split):
        msgs = ex.get("messages")
        if not msgs:
            continue
        try:
            yield tok.apply_chat_template(msgs, tokenize=False)
        except Exception:  # noqa: BLE001 - tokenizer without a chat template
            yield "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def interleave(a: Iterator[list[int]], b: Iterator[list[int]]) -> Iterator[list[int]]:
    """~50/50 by token count: always draw from the stream that is behind."""
    na = nb = 0
    a_done = b_done = False
    while not (a_done and b_done):
        take_a = (na <= nb and not a_done) or b_done
        try:
            if take_a:
                t = next(a)
                na += len(t)
            else:
                t = next(b)
                nb += len(t)
            yield t
        except StopIteration:
            if take_a:
                a_done = True
            else:
                b_done = True


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="C:/pollard/tpd_data")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train-seq", type=int, default=128, help="tPD row length (config max_seq_len)")
    ap.add_argument("--target-train-rows", type=int, default=90_000, help="5k steps x 16 = 80k rows")
    ap.add_argument("--nontarget-train-rows", type=int, default=90_000)
    ap.add_argument("--val-rows", type=int, default=512)
    ap.add_argument("--eval-seq", type=int, default=512, help="quant-arm eval/calib row length")
    ap.add_argument("--eval-rows", type=int, default=128)
    ap.add_argument("--calib-rows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Document separator for packing: Qwen's eos is the chat token <|im_end|>; plain-text docs are
    # separated by <|endoftext|> in pretraining, so prefer that when the vocab has it.
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    if eos is None or eos == tok.unk_token_id:
        eos = tok.eos_token_id
    assert eos is not None
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    log: list[str] = []

    # --- target: python code, doc-level split. Every 10th doc -> held-out eval pool; of the rest,
    # docs fill the calibration pool first (disjoint from tPD training), then the tPD train pool.
    # Exact token accounting (+5% slack for the per-doc EOS / ragged tails).
    need_tpd = int((args.target_train_rows + args.val_rows) * args.train_seq * 1.05)
    need_calib = int(args.calib_rows * args.eval_seq * 1.05)
    need_held = int(args.eval_rows * args.eval_seq * 1.05)
    held: list[list[int]] = []
    calib_docs: list[list[int]] = []
    tpd_docs: list[list[int]] = []
    n_held = n_calib = n_tpd = 0
    for i, d in enumerate(target_docs(log)):
        if n_held >= need_held and n_calib >= need_calib and n_tpd >= need_tpd:
            break
        t = enc(d)
        if i % 10 == 0:
            if n_held < need_held:
                held.append(t)
                n_held += len(t) + 1
        elif n_calib < need_calib:
            calib_docs.append(t)
            n_calib += len(t) + 1
        elif n_tpd < need_tpd:
            tpd_docs.append(t)
            n_tpd += len(t) + 1
    random.shuffle(tpd_docs)

    tpd_rows = pack(tpd_docs, args.train_seq, args.target_train_rows + args.val_rows, eos)
    if len(tpd_rows) < args.target_train_rows + args.val_rows:
        log.append(f"WARNING target rows short: {len(tpd_rows)} < {args.target_train_rows + args.val_rows}")
    t_val, t_train = tpd_rows[: args.val_rows], tpd_rows[args.val_rows :]
    write_split_parquet(t_train, out / "target_code", "train")
    write_split_parquet(t_val, out / "target_code", "validation")
    target_calib = pack(calib_docs, args.eval_seq, args.calib_rows, eos)
    target_eval = pack(held, args.eval_seq, args.eval_rows, eos)

    # --- non-target: wikitext-103 + ultrachat, ~50/50 by tokens
    nt_train_stream = interleave(
        (enc(d) for d in wiki_docs("train")), (enc(d) for d in chat_docs("train_sft", tok))
    )
    nt_rows = pack(nt_train_stream, args.train_seq, args.nontarget_train_rows + args.val_rows, eos)
    write_split_parquet(nt_rows[args.val_rows :], out / "nontarget_mix", "train")
    write_split_parquet(nt_rows[: args.val_rows], out / "nontarget_mix", "validation")
    # generic calibration: wikitext validation + ultrachat train_gen (disjoint from train + eval)
    gen_stream = interleave(
        (enc(d) for d in wiki_docs("validation")),
        (enc(d) for d in chat_docs("train_gen", tok)),
    )
    generic_calib = pack(gen_stream, args.eval_seq, args.calib_rows, eos)
    nt_eval_stream = interleave(
        (enc(d) for d in wiki_docs("test")), (enc(d) for d in chat_docs("test_sft", tok))
    )
    nontarget_eval = pack(nt_eval_stream, args.eval_seq, args.eval_rows, eos)

    sets = {
        "target_eval": torch.tensor(target_eval, dtype=torch.long),
        "nontarget_eval": torch.tensor(nontarget_eval, dtype=torch.long),
        "target_calib": torch.tensor(target_calib, dtype=torch.long),
        "generic_calib": torch.tensor(generic_calib, dtype=torch.long),
    }
    meta = {
        "tokenizer": args.tokenizer,
        "train_seq": args.train_seq,
        "eval_seq": args.eval_seq,
        "rows": {k: int(v.shape[0]) for k, v in sets.items()},
        "tpd_rows": {"target_train": len(t_train), "nontarget_train": len(nt_rows) - args.val_rows},
        "log": log,
        "split_rule": "target: doc i%10==0 -> held-out eval; calib docs disjoint from tPD train docs",
    }
    empty = [k for k, v in meta["rows"].items() if v == 0] + [k for k, v in meta["tpd_rows"].items() if v == 0]
    assert not empty, f"empty sets {empty}: source too small for the requested rows; meta={meta}"
    sets["meta"] = meta  # type: ignore[assignment]
    torch.save(sets, out / "eval_sets.pt")
    (out / "prep_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
    # Abandoned `datasets` streaming iterators leave Arrow thread pools that can deadlock in their
    # destructor at interpreter shutdown (seen on macOS: 0% CPU forever after all files were written).
    # Everything is flushed and closed by now, so skip the shutdown sequence.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
