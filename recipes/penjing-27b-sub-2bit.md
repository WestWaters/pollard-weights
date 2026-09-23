# Recipe: Penjing — a 27B at ~2 bits that still answers

Qwen3.8-27B (dense, hybrid-SSM: 48 SSM blocks + 16 attention, 65 total, 866
tensors) taken to roughly two bits per weight and measured against its own f16
rather than against a story. Every number below came off an RTX 5070 Ti (16 GB)
and is reproducible from this repo with the two commands in section 1.

This file is written to be handed to a coding agent ("follow
recipes/penjing-27b-sub-2bit.md") or followed by hand.

## 1. The build

```
pollard-fragile --gguf Qwen__Qwen3.8-27B-f16.gguf \
                --protect iq2_kt --out penjing.fragile.json

pollard-automap --model Qwen__Qwen3.8-27B-f16.gguf \
                --imatrix Qwen__Qwen3.8-27B.dat \
                --fragile penjing.fragile.json \
                --eval eval_heldout.txt \
                --body iq1_s --protect iq2_kt \
                --mix-only --ngl 99
```

That is the whole thing. No hand-written `--custom-q`, no hand-made tensor
list: `pollard-fragile` measures which tensor kinds have the heaviest tails,
`pollard-automap` classifies the architecture, builds the mix around that scan,
quantizes, scores perplexity and runs the coherence gate. `--mix-only` builds
the flagship mix and skips the rest of the ladder.

What automap decided here, verbatim from its log:

```
parsed: 852 tensors, 65 layers, dense +hybrid-SSM 48ssm/16attn
atoms : body=iq1_s (1.56 bpw)  protect=iq2_kt (2.125 bpw) -> imatrix-guided (trellis) build
flags : --output-tensor-type Q6_K --token-embedding-type Q4_K
custom-q:
  blk\.0\.=P, blk\.1\.=P, blk\.63\.=P, blk\.64\.=P,
  attn_k=iq1_s, attn_v=iq1_s,
  attn_q=P, attn_output=P, attn_gate=P, ffn_down=P,
  ssm_in=P, ssm_out=P, ssm_alpha=P, ssm_beta=P          (P = the --protect atom)
fragile : auto-protected -> nextn.eh_proj (kurtosis 1043.7, crest 136),
                            ssm_conv1d   (kurtosis   19.9, crest 100),
                            attn_output  (kurtosis   12.9, crest  91),
                            ssm_out      (kurtosis   10.2, crest  97)
imatrix : 496 tensors covered; 1 uncovered tensor pinned to q6_K up front,
          7 more repaired mid-build (see section 4)
```

Note the first line: automap parsed **852** tensors from a file that has **866**.
That gap is section 4's trap, and it is why the repair loop exists.

## 2. What it measures

KL divergence against the f16, via `llama-perplexity --kl-divergence-base ...
--kl-divergence -c 2048`. Same source, same imatrix, same corpus; the three
builds differ only in the `--protect` atom. **f16 baseline PPL 4.1422 +/- 0.1032.**

| protect atom | bpw | size | Mean KLD | Median | 90% | 95% | Same top p | PPL | runs on |
|---|---|---|---|---|---|---|---|---|---|
| `iq1_kt` (1.75) | 1.91 | 6.53 GB | 0.5513 +/- 0.0107 | 0.2003 | 1.3576 | 2.3335 | 75.88 +/- 0.42 % | 5.7864 | ik_llama only |
| **`iq2_kt` (2.125)** | **2.16** | **7.36 GB** | **0.4217 +/- 0.0096** | **0.1313** | **0.9969** | **1.7240** | **79.10 +/- 0.40 %** | **5.0817** | ik_llama only |
| `iq2_xxs` (2.06) | 2.12 | 7.25 GB | 0.5142 +/- 0.0104 | 0.1802 | 1.2258 | 2.0922 | 76.38 +/- 0.42 % | 5.4312 | **anywhere** |

All three pass the coherence gate. **Ship all of them.** The last column is a
property of each rung, not a reason to drop one -- see section 3.

A Pollard repo is a LADDER, and a ladder is allowed to have rungs for different
runtimes. Publish the trellis builds next to the stock-quantized ones and let
the card say which is which. Penjing's ladder:

```
$ pollard-ggufcheck --offline *.gguf
  stock llama.cpp   penjing-Q6_K.gguf
  stock llama.cpp   Qwen3.8-27B-Pollard-IQ4_XS.gguf
  stock llama.cpp   Qwen3.8-27B-Pollard-IQ3_S.gguf
  stock llama.cpp   Qwen3.8-27B-Pollard-IQ2_XXS.gguf
  stock llama.cpp   Qwen3.8-27B-Pollard-V3-iq2xxs.gguf
  needs ik_llama    Qwen3.8-27B-Pollard-V2-iq2kt.gguf  [IQ2_KT x344]
  needs ik_llama    Qwen3.8-27B-Pollard-FLAGSHIP.gguf  [IQ1_KT x344]

7 file(s): 2 ik_llama, 5 stock
```

`pollard-card` reads those same tensor types and writes the disclosure itself --
you do not hand-write it, and you do not guess from filenames:

> **Standard GGUF -- runs in stock llama.cpp / ik_llama.cpp, Ollama, LM Studio,
> except where noted.** `FLAGSHIP`, `V2-iq2kt` need ik_llama.cpp: their
> allocation puts ik_llama-only atoms on the tensors it protects. The rest run
> anywhere.

Build the stock rungs with the stock `llama-quantize` and the trellis rungs with
ik_llama's, into one repo. Someone who runs ik_llama takes the flagship; someone
on Ollama takes IQ4_XS or the `iq2_xxs` build at the same tier; the same repo
serves both, which is how anyone moving between runtimes expects a model to work.

## 3. The finding worth keeping: spend on the ALPHABET, not the bit count

`iq2_xxs` and `iq2_kt` are 0.065 bpw apart and produced builds 110 MB apart.
They are not close in quality:

- `iq2_kt` cuts mean divergence from the f16 by **23.5%** against `iq1_kt`
- `iq2_xxs`, at essentially the same size as `iq2_kt`, recovers barely a third
  of that -- 0.5142 against 0.5513, where `iq2_kt` reaches 0.4217
- the gap holds across the whole distribution -- median, 90th and 95th
  percentile all move together, and the error bars do not overlap

`iq2_kt` is a trellis/KT atom; `iq2_xxs` is a conventional one. At equal bits
the trellis codebook is worth about a fifth of the divergence. So buy the
better ALPHABET before buying more bits -- and if the budget allows exactly one
2-bit tier, make it a KT one.

The corollary is the negative result: going from 1.75 to 2.06 bpw bought almost
nothing (0.5513 -> 0.5142) while going from 1.75 to 2.125 bpw *in a KT atom*
bought a lot. Size alone does not predict quality here.

**And the better alphabet has a price that is not measured in bits.** KT atoms
are ik_llama's; stock llama.cpp caps ggml types at 43, and `IQ2_KT` is type
153. A stock runtime does not degrade on these builds, it refuses them:

```
tensor 'blk.0.ssm_alpha.weight' has invalid ggml type 153. should be in [0, 43)
```

That is not a reason to skip the trellis build -- it is the reason to build the
tier TWICE. `iq2_xxs` and `iq2_kt` sit at the same size and serve different
runtimes, so a complete ladder carries both: the KT rung for people on
ik_llama, the XXS rung at the same tier for everyone else. Neither is the
consolation prize.

What you must not do is ship an ik_llama-only file **without saying so** -- a
user whose runtime refuses to open it reads that as a corrupt download, not as
a runtime mismatch. Run `pollard-ggufcheck` before publishing and let
`pollard-card` write the sentence from the file's real tensor types.

## 4. Two traps this model exposes

**Block 64 is invisible to the imatrix, and it truncates the tensor list.**
Qwen3.8 carries an MTP/`nextn` head at `blk.64`. `llama-imatrix` never
exercises it -- it is not part of an ordinary forward pass -- so blocks 0-63
have coverage and block 64 has none. At a low-bit type, `llama-quantize` stops:

```
Missing importance matrix for tensor blk.64.nextn.eh_proj.weight in a very
low-bit quantization. The result will be garbage, so bailing out
```

Re-running the imatrix does not help; the tensor cannot be covered. Worse, the
abort also kills the `--dry-run` that produces the tensor list, which is why
automap sees 852 of 866 and cannot pin what it never saw. automap handles both
halves: it pins the uncovered tensors it *can* see before building, then parses
this exact error and repairs it in place -- here `blk.64.ffn_down`, `ffn_gate`,
`ffn_up`, `attn_k`, `attn_output`, `attn_q` and `attn_v`, seven tensors, each
pinned to `q6_K` and retried without losing the run.

`--custom-q` is FIRST-MATCH-WINS, so repairs are inserted at the front.

If you would rather hand automap a complete list than let it repair, pin the
block in the dry run yourself and check the count before building:

```
llama-quantize --dry-run --imatrix <dat> --custom-q "blk\.64\.=q6_K" \
               <f16> NUL IQ1_S > tensors.txt
findstr /c:"866/ 866" tensors.txt    # TENSOR_LIST_COMPLETE, vs 852 without the pin
pollard-automap --tensors tensors.txt ...
```

The cost is file size only: llama.cpp reports the whole block as
`unused tensor ... ignoring` at load.

**An instruct model cannot be gated as a base model.** This model's EOS is
`<|im_end|>`, which only exists inside a chat turn. Generate from a raw prompt
and it has no reachable way to stop: it runs to the token budget and repeats,
at ANY bit width. Scored that way the 1.91 bpw build was twice reported BELOW
FLOOR; scored through its own chat template it answers correctly and stops on
its own token. `pollard-bench --coherence` detects the template, serves the
model with `--jinja`, and scores the chat turn. If it cannot find
`llama-server` it reports UNGATED rather than silently falling back to a raw
completion -- point `--llama-server` at your ik_llama build.

Shipping sampler defaults the gate uses, and what the card should quote:

```
--temp 0.7 --repeat-penalty 1.15 --repeat-last-n 256 --top-k 40 --top-p 0.9
```

## 5. Reproducing it end to end

```
# 1. calibration + imatrix. ik_llama reads only the legacy .dat format.
#    Quantize a Q6_K host first: the f16 ran 54.7 GB against 16 GB VRAM at
#    254 s/pass; Q6_K is -0.17% measured, a third the size, and a rung you
#    ship anyway. --save-frequency matters -- it defaults to 0, and a run
#    that dies at chunk 484 with no checkpoint has to start over.
llama-quantize Qwen__Qwen3.8-27B-f16.gguf penjing-Q6_K.gguf Q6_K
llama-imatrix -m penjing-Q6_K.gguf -f calib.txt -o Qwen__Qwen3.8-27B.dat \
              --output-format dat --save-frequency 50

# 2. + 3. the fragile scan and the build -- section 1

# 4. the number that decides. The KLD base MUST be built at the context you
#    measure at; mismatched, llama-perplexity prints "failed to eval" and
#    still exits 0.
llama-perplexity -m <f16> -f wikitext2_test.txt -c 2048 \
                 --kl-divergence-base f16.klbase.dat
llama-perplexity -m <build> -f wikitext2_test.txt -c 2048 \
                 --kl-divergence-base f16.klbase.dat --kl-divergence

# 5. confirm the build is loadable by whatever you are publishing for.
#    Takes files positionally. --offline skips the HF lookup.
pollard-ggufcheck --offline <build>
```

Judge on KL divergence against your own f16 -- not on perplexity, and not on a
benchmark score. "Accuracy is Not All You Need" (NeurIPS 2024) measures
Spearman 0.981 between KLD and answer flips, and shows a 4-bit 70B sitting
within 0.78% of baseline accuracy while 8.1% of its answers had changed.

## 6. Porting this to another model

The recipe is `--body iq1_s --protect iq2_kt` plus a fragile scan, and that
much transfers. What does not transfer automatically:

- **Run `pollard-fragile` on the new model.** The protected kinds here
  (`ssm_conv1d`, `ssm_out`) are hybrid-SSM tensors; a pure transformer has
  neither, and will have its own heavy tails.
- **Check imatrix coverage before the build, not during it.** The line to read
  is automap's `imatrix : N tensors covered; PINNING ...`. A dense model with
  uncovered tensors is unusual and automap says so -- on Penjing it was a real
  architectural feature, but on most models it means a bad imatrix.
- **MoE models take a different recipe.** automap prints which one it chose
  (`Mix policy: [...]`); this run reports `dense recipe`.
- **`iq1_s` is a floor, not a default.** It was chosen here to land near
  Bonsai-2's 1.72 bpw footprint. On a smaller model it will not hold.
