# K2-Horizon GGUFs won't load on Windows/MSVC — and the fix

**Symptom.** On a Windows build of [`MBZUAI-IFM/llama.cpp`](https://github.com/MBZUAI-IFM/llama.cpp)
(the fork carrying the `k2-horizon` architecture), loading *any* K2-Horizon GGUF fails before a single
token is produced:

```
Failed to process regex: '(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|‌|‍)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+'
Regex error: regex_error(error_escape): The expression contained an invalid escaped character
```

`llama-quantize`, `llama-imatrix` and `llama-perplexity` all fail the same way, so on Windows the fork
can neither **build** nor **evaluate** a K2 model. The identical GGUF works on macOS.

**Cause.** `LLAMA_VOCAB_PRE_TYPE_K2_HORIZON` in `src/llama-vocab.cpp` declares a pre-tokenizer regex
using the Unicode property escapes `\p{L}`, `\p{M}` and the explicit `‌` / `‍` (ZWNJ/ZWJ).
`unicode_regex_split_custom` in `src/unicode.cpp` has hand-written splitters for the GPT-2, Llama-3,
Qwen2, Qwen3.5, Kimi-K2, AFMoE and other pre-tokenizers, but **no arm matched the K2-Horizon literal**,
so it fell through to the general-purpose `std::regex` / `std::wregex` fallback. That fallback is
platform-dependent: Apple's libc++ tolerates `\p{...}`, **MSVC's `std::regex` rejects it outright**.
Hence the macOS/Windows split — it is a portability bug, not a bad GGUF.

**Fix.** `k2-horizon-msvc-regex.patch` in this directory adds the missing custom splitter, so the
`std::regex` fallback is never reached for this tokenizer. It clones the existing Llama-3 splitter and
widens only the letter-run rule, because the K2 regex differs from Llama-3's in exactly one place —
`\p{L}+` becomes `(?:\p{L}|\p{M}|‌|‍)+`:

```cpp
auto _is_k2_letter = [&] (const size_t pos) -> bool {
    const uint32_t c = _get_cpt(pos);
    if (c == 0x200C || c == 0x200D) {   // ZWNJ / ZWJ join into the letter run
        return true;
    }
    const auto f = _get_flags(pos);
    return f.is_letter || f.is_accent_mark;   // \p{L} | \p{M}
};
```

Everything else — the contraction rule, `\p{N}{1,3}`, the punctuation run, the whitespace tail — is
byte-for-byte the Llama-3 logic, which already matches K2's remaining alternatives.

Apply it against the fork, then rebuild only the three tools you need:

```bash
git -C llama.cpp apply /path/to/k2-horizon-msvc-regex.patch
cmake --build build --config Release -j 8 \
      --target llama-quantize llama-imatrix llama-perplexity
```

**One caveat when regenerating this patch by hand:** the dispatch arm compares the regex by string
*equality*, so the literal must match `llama-vocab.cpp` character for character. Read it out of that
file programmatically rather than retyping it — a single wrong backslash silently reverts you to the
broken fallback.

**Verification** (RTX 5070 Ti 16 GB, MSVC 2022, CUDA 12.8 — full wikitext-2, ctx 512):

| check | before | after |
|---|---|---|
| K2-Horizon-3.7B f16 load | `error_escape` | loads |
| K2-Horizon-3.7B f16 PPL | — | **11.2798** ± 0.0808 |
| Qwen2.5-0.5B-Instruct Q8_0 control PPL | 13.6097 | **13.6097** |

The Qwen control is the point of the second row: the patch fixes K2 without disturbing any existing
splitter. This is worth upstreaming to the fork — it affects every Windows user of it.
