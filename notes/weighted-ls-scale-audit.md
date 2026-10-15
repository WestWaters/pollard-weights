# Weighted least-squares scale in Pollard's low-bit encoders -- audit (2026-10-15)

Question: is the importance-weighted least-squares (LS) scale -- Hy4's "Sherry" encoder lever,
`d = sum(w*u*x) / sum(w*u^2)` for fixed codes `u` -- actually used anywhere a Pollard build passes through?

| where | what it does | used by a shipping build? |
|---|---|---|
| `tools/pollard_gptq.py` `weighted_ls_scale` + `imatrix_quantize(scale="ls")` | closed-form weighted LS, iterated 5x, symmetric | **No.** The only caller (`--method imatrix`) uses the default `scale="sweep"`; no CLI flag selects `ls`. It is also the torch R&D harness, not an emitter. |
| `tools/pollard_lowbit.py` `q_ternary` / `q_binary` | abs-mean scale, unweighted | R&D only. Binary abs-mean IS the unweighted LS optimum; ternary abs-mean is not (zeros are counted), and neither uses the imatrix. |
| `runtime-patches/llama-stq-stq1_0-quants.patch` | STQ1_0 encoder: zero the least-important lane, scale = `sum(w*abs(x)) / sum(w)` over survivors -- the exact weighted LS for fixed ternary codes | **No.** It patches the stock `llama-stq` tree; the published ladder aliases `stq1_0 -> iq1_bn` (ik_llama, pollard-automap), so STQ1_0 itself never ships. |
| ggml `make_qx_quants` / `make_qkx2_quants` / `make_qkx3_quants` (K-quants, used for Q2_K..Q6_K and the K-quant parts of IQ4_XS etc.) | 19-candidate scale sweep, each candidate refined to the weighted LS `sumlx/suml2` with the imatrix as weights | **Yes** -- upstream, not Pollard code. The GGUF ladder already gets weighted LS through llama.cpp. |
| ggml `quantize_tq1_0` / `quantize_tq2_0` | amax, `quant_weights` ignored | not on the Pollard ladder |
| ik_llama `iq1_bn` / `IQ*_KT` encoders | not checked -- ik_llama source is not vendored here | the flagship rungs; check on the box |

Conclusion: the lever is implemented twice in Pollard and reaches no published file. Where it would matter
most (the ternary / 1-bit crush tier, where our own measurement says LS beats the sweep by ~33%), the shipping
path goes through ik_llama's `iq1_bn`, whose encoder was not audited. Next steps, if wanted: read ik_llama's
`iq1_bn` / `iq1_kt` quantize for an imatrix-weighted scale; expose `pollard-gptq --scale ls` for `--qmode
ternary` (the measured win) and leave 2-3 bit on the asymmetric sweep (LS loses there: -10% / -21%).
