# STQ1_0 QAT / distillation scripts (copied from the box, C:\pollard\qat, 2026-10-15)

The 2-bit-class STQ1_0 quantization-aware training runs from August–September 2026, kept here so the
upcoming `pollard-qat` builds on them instead of re-deriving them.

- `qat_stq1.py` — fake-quant STQ1_0 on the FFN weights (STE), matching the C encoder
  (`quantize_row_stq1_0_imatrix`: stride-16 groups of 4, per-block mean|x| scale, argmin zero).
- `qat_distill.py`, `qat_distill2.py` — KL distillation against the frozen ORIGINAL instruct teacher;
  `qat_distill2.py` samples a 50/50 wikitext + ChatML mix (raw wikitext alone erased chat behaviour).
- `prep_chat.py` — builds the ChatML half of the mix. `qat_continue.py`, `qat_real.py` — resume / export.
- `*.log` — the original run logs.

Result recorded at the time: 767 MB, PPL 20.37 (2.12x f16), mean KLD 0.869, top-1 59.95%, chat clean.
