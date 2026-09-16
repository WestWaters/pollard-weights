#!/usr/bin/env python3
"""pollard-brainverify -- does this brain actually recall, and is the number real?

A brain that scores well during training can still be worthless, and a brain that is perfect can be
made to look broken by asking it the wrong way. Both happened here. This is the ONE measurement, with
the construction that produced the published numbers baked in, so nobody has to rebuild it and get it
subtly wrong -- which cost an evening and three different wrong answers on a brain that was fine.

  pollard-brainverify --brain FlyBrain-Pollard-CNSv1.pt --model Qwen/Qwen2.5-0.5B-Instruct \
      --filler corpus.txt

What it measures: a six-letter string is stated ONCE, buried under filler, and asked about far beyond
the attention window. Words are generated fresh every sample and never reused, so there is nothing to
memorise -- the only way to score is to store and retrieve.

Three numbers decide it, and all three must hold:

  FLOOR     the same task with no brain. Must be 0%. If it is not, the fact is reachable without
            memory and the headline number means nothing.
  RECALL    with the brain.
  CONTROLS  a word the document never contained, and a brain that read a DIFFERENT document. Both
            must be ~0%. A brain that scores well here is leaking, not remembering.

Two things that will silently ruin a measurement, both learned the hard way:

  * THE PROMPT IS PART OF THE EXPERIMENT. A token is filed under the words immediately before it, so
    the query has to reproduce that context. The document says "The secret word is <w>", and the
    query ends with those same words. Ask "what is the secret word?" alone and a brain measuring 100%
    measures 46% -- the memory is intact, the question simply arrives at the wrong address.
  * SCORE INSIDE ONE PASS. The question belongs in the same sequence as the document. Feeding the
    document and then running the question separately measures a brain that has read nothing.

Both are handled here. Do not reimplement this.
"""
from __future__ import annotations

import argparse
import random
import string
import sys

try:
    import torch
except ImportError:
    raise SystemExit("needs torch: pip install 'pollard-weights[flybrain]'") from None

# The verified construction. These are not defaults to tune -- they are the experiment.
QTAIL = " Question: what is the secret word? Answer: The secret word is"
WORD_LEN = 6
N_CHUNKS = 6


def build(tok, brain, filler, word, rng):
    """fact + filler + question, one sequence, with the QUESTION GUARANTEED PRESENT.

    Truncate the FILLER to fit the budget, never the question. Building the whole string and cutting
    to a token budget slices the question off the end of every sample, and the task becomes
    unanswerable -- which reads as a broken brain and is not.
    """
    head = tok(f"The secret word is {word}. ", return_tensors="pt").input_ids
    qt = tok(QTAIL, add_special_tokens=False, return_tensors="pt").input_ids
    budget = N_CHUNKS * brain.win - head.shape[1] - qt.shape[1]
    if budget < 1:
        raise SystemExit("window too small for the fact and the question")
    i = rng.randrange(0, max(len(filler) - 80_000, 1))
    fil = tok(filler[i:i + 80_000], add_special_tokens=False,
              return_tensors="pt").input_ids[:, :budget]
    return torch.cat([head, fil, qt], 1).to(brain.device)


def run_once(brain, ids, use_brain=True):
    """Read the document, return the retrieval at the question. One pass, memory intact."""
    brain.reset(1)
    raw = None
    for s0 in range(0, ids.shape[1], brain.win):
        ch = ids[:, s0:s0 + brain.win]
        if ch.shape[1] < 1:
            continue
        _logits, raw = brain._chunk(ch, write=use_brain)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--brain", required=True, help="the .pt to verify")
    ap.add_argument("--model", required=True, help="the backbone it was trained against")
    ap.add_argument("--filler", required=True, help="text to bury the fact under")
    ap.add_argument("--samples", type=int, default=128, help="default 128; 32 cannot tell 100 from 97")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from pollard_flybrain import FlyBrain, load_backbone
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_backbone(a.model, torch.float32, a.device)
    brain = FlyBrain.load(a.brain, device=a.device).bind(model, tok, verbose=True)
    filler = open(a.filler, encoding="utf-8", errors="replace").read()

    def word(rng):
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(WORD_LEN))

    hit = floor = ctl_word = ctl_doc = 0
    rng = random.Random(a.seed)
    for n in range(a.samples):
        w = word(rng)
        ids = build(tok, brain, filler, w, rng)
        want = tok(" " + w, add_special_tokens=False).input_ids[:brain.span]

        raw = run_once(brain, ids, use_brain=True)
        hit += brain.decode(raw)[0].tolist()[:len(want)] == want

        # floor: the same question with the memory never written
        raw0 = run_once(brain, ids, use_brain=False)
        floor += brain.decode(raw0)[0].tolist()[:len(want)] == want

        # control 1: score against a word the document never contained
        other = tok(" " + word(rng), add_special_tokens=False).input_ids[:brain.span]
        ctl_word += brain.decode(raw)[0].tolist()[:len(other)] == other

        # control 2: the brain reads a DIFFERENT document than the one we score against
        raw2 = run_once(brain, build(tok, brain, filler, word(rng), rng), use_brain=True)
        ctl_doc += brain.decode(raw2)[0].tolist()[:len(want)] == want

        if (n + 1) % 16 == 0:
            print(f"    {n+1}/{a.samples}  recall {100*hit/(n+1):.1f}%", flush=True)

    N = a.samples
    print(f"\n  === {a.brain}, {N} samples, words generated fresh ===")
    print(f"    FLOOR    no brain, fact outside the window   {100*floor/N:5.1f}%   <- must be 0")
    print(f"    RECALL   with the brain                      {100*hit/N:5.1f}%")
    print(f"    CONTROL  a word the document never held      {100*ctl_word/N:5.1f}%   <- must be 0")
    print(f"    CONTROL  brain read a DIFFERENT document     {100*ctl_doc/N:5.1f}%   <- must be 0")
    print(f"    state, constant at any length                {brain.state_bytes/1e6:.1f} MB")
    bad = [m for m, v in (("floor", floor), ("word control", ctl_word),
                          ("document control", ctl_doc)) if v / N > 0.02]
    if bad:
        print(f"\n  NOT A VALID MEASUREMENT: {', '.join(bad)} above 2% -- the number is leaking.")
        sys.exit(1)
    print("\n  controls clean: this is retrieval, not leakage.")


if __name__ == "__main__":
    main()
