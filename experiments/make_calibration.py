#!/usr/bin/env python3
"""Build a mixed-domain calibration corpus for llama-imatrix.

Importance matrices are only as good as the text that excites the weights.
A 60 KB single-domain slice was our first pass; the better practice
(popularized by the dynamic-quant community) is a larger, deliberately
mixed corpus: encyclopedic prose, narrative prose, and code, so the
matrix sees every register the model will serve. This script downloads
public sources at build time (nothing is redistributed here), mixes them
in fixed proportions, and writes one file for `llama-imatrix -f`.

Usage:
  make_calibration.py --out calib.txt              # ~1.6 MB (~400K tokens)
  make_calibration.py --out calib.txt --chars 4000000
"""
import argparse
import io
import sys
import urllib.request

SOURCES = [
    # (name, url, share of corpus, skip_head_chars)
    ("wikitext (encyclopedic)",
     "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
     0.4, 0),
    ("gutenberg: Moby Dick (narrative)",
     "https://www.gutenberg.org/files/2701/2701-0.txt", 0.2, 60000),
    ("gutenberg: Pride & Prejudice (dialogue)",
     "https://www.gutenberg.org/files/1342/1342-0.txt", 0.2, 40000),
    ("llama.cpp source (code, MIT)",
     "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama.cpp",
     0.2, 0),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pollard-calib/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chars", type=int, default=1600000,
                    help="total corpus size in characters (default ~1.6M ≈ 400K tokens)")
    a = ap.parse_args()

    parts = []
    for name, url, share, skip in SOURCES:
        want = int(a.chars * share)
        try:
            raw = fetch(url)
        except Exception as e:
            sys.exit(f"ERROR fetching {name}: {e}")
        if url.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError:
                sys.exit("parquet source needs pyarrow: pip install pyarrow")
            text = "".join(pq.read_table(io.BytesIO(raw)).column("text").to_pylist())
        else:
            text = raw.decode("utf-8", "replace")
        text = text[skip:skip + want]
        parts.append(text)
        print(f"  {name:42s} {len(text):>9,} chars")

    corpus = "\n\n".join(parts)
    with open(a.out, "w") as f:
        f.write(corpus)
    print(f"\nwrote {a.out}: {len(corpus):,} chars (~{len(corpus)//4:,} tokens)")
    print("next: llama-imatrix -m model-f16.gguf -f", a.out, "-o model.imatrix -ngl 99")


if __name__ == "__main__":
    main()
