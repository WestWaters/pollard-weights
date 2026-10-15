# EXL3 recipe bridge (`pollard-exl3 --recipe pollard|exl3|none`)

**What.** ExLlamaV3 (MIT, >= 1.6; v1.6.2 parser vendored in tests/fixtures) converts from a per-tensor bitrate
recipe: `convert.py -rcp recipe.yaml` with `tensors: {<module key>: bits}` covering exactly the budgeted
("bits") Linears, plus `target_bpw` / `achieved_bpw` / `head_bits`. EXL3's own `sc_optimize.py` fills it from
noise-injected KLD on a self-sampled trace. `--recipe pollard` fills it from **Pollard's** measured sensitivity:

* tensor list: exact from exllamav3's model graph when it imports (the box); otherwise from the safetensors
  headers (routers / norms / embeddings / towers / MTP excluded, fused experts expanded per expert, EXL3's names)
* model: `kld(t, K) = S_t * 4^(p - K)` (sc_optimize's law, alpha 2), S_t = the layer's measured attn/ffn group
  KL (pollard-probe / -sensitivity at p probe bits) split by tensor size
* solve: greedy marginal KL per stored bit at the --bpw budget (same accounting as `convert --bits`), EXL3's
  fused groups tied (`q+k+v`, `gate+up`, `in_proj_qkv+in_proj_z`), bits in [--min-bits 2, --max-bits 8]
* checked against EXL3's own parser logic before convert (missing / unknown keys abort convert there too)

Modes: `none` = the gold path (EXL3 budgeted allocator + smoothing + Calib 3.0) and stays the DEFAULT;
`pollard` = the recipe above; `exl3` = an sc_optimize recipe (`--exl3-recipe FILE`, or prints the sc_* pipeline);
a path = any recipe YAML (old behaviour).

## Head-to-head (box, overnight, GPU free -- EXL3 convert needs CUDA; NOT run yet)

Same smoothed source, same Calib 3.0 `-cd`, same bpw, same held-out eval; only the allocation differs.
Run when the 5070 Ti is free (60/40 rule: never while the son's session is resident -- `nvidia-smi` first).
Paths as on the box (`C:\pollard\pw`), schtasks with a past `/sd` + `/run` for anything long.

```bash
M=Qwen/Qwen3-8B; B=3.0
pollard-hf-smooth --model $M --out Qwen3-8B-smooth --calib cal.txt --device cpu   # the gold preconditioning
pollard-calib --out cal.txt --held-out held.txt --model $M                  # reasoning share automatic
pollard-probe --model Qwen3-8B-smooth --eval held.txt --out sens.json --device cpu

# A) gold path: EXL3 budgeted allocator
pollard-exl3 --model Qwen3-8B-smooth --calib-text cal.txt --bpw $B --out exl3-budgeted
# B) Pollard allocation -> EXL3 recipe
pollard-exl3 --model Qwen3-8B-smooth --calib-text cal.txt --bpw $B --recipe pollard --sensitivity sens.json --out exl3-pollard
# C) EXL3's own measured recipe (exllamav3 v1.6.2 checkout; needs a 6.0bpw and a 2.0bpw EXL3 first)
pollard-exl3 --model Qwen3-8B-smooth --bpw 6.0 --calib-text cal.txt --out exl3-6.0
pollard-exl3 --model Qwen3-8B-smooth --bpw 2.0 --calib-text cal.txt --out exl3-2.0
pollard-exl3 --model Qwen3-8B-smooth --bpw $B --recipe exl3 --out exl3-sc        # prints the sc_* commands:
#   python sc_trace.py -m exl3-6.0 -o sc/cal_trace.json -co sc/cal_trace.safetensors -cs 262144 -ambs 32
#   python sc_rfn_probe.py -mq exl3-2.0 -mr Qwen3-8B-smooth -d 0 -o sc/rfn_2.00bpw.json
#   python sc_measure.py -m Qwen3-8B-smooth -d 0 --shaped -tr sc/cal_trace.safetensors -rr sc/rfn_2.00bpw.json -rs 1.0,0.5 -o sc/noise_attrib.json
#   python sc_optimize.py -m sc/noise_attrib.json -rr sc/rfn_2.00bpw.json -b $B -hb 6 -al 2.0 -o sc/recipe.yaml
pollard-exl3 --model Qwen3-8B-smooth --calib-text cal.txt --bpw $B --recipe exl3 --exl3-recipe sc/recipe.yaml --out exl3-sc

# KL vs the bf16 source (pollard-verify's exl3 gate = assembled forward; same rows for all three), and
# EXL3's own eval on the held-out text as the cross-check (exllamav3 checkout: eval/model_diff.py / eval/ppl.py)
for D in exl3-budgeted exl3-pollard exl3-sc; do
  pollard-verify --model $D --source Qwen3-8B-smooth --end-to-end > verify-$D.txt
done
grep -H -i "kld\|top-1" verify-*.txt
```

Fairness notes: C) also gets EXL3's in-domain `-cd` trace if you pass `-cd sc/cal_trace.safetensors` -- run C
twice (Calib 3.0 `-cd` and its own trace) so the allocation and the calibration effect are separable. Repeat at
B = 2.5 and 4.0. Promote `--recipe pollard` past R&D only if it beats A at <= the same bpw (the SKILL's rule).
