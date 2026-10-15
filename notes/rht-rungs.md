# Online-RHT rungs that stock llama.cpp loads (`prism.hadamard`, #29600)

**Status:** tool built and verified exact on Qwen3-0.6B. The quick 0.6B A/B **loses to uniform+imatrix
at every rung tested**. The decision is made on the overnight Qwen3-8B run below.

## What the runtime does (read from llama.cpp `a9d27ac`, PR #29600, release **b11556**)

- `src/llama-model.cpp` `load_hparams_hadamard` reads `prism.hadamard.{version, block_size, transform,
  axis, sign_mode, sign_widths, sign_values, weight_names, inverse_weight_names, tied_output,
  gdn_v_grouped}`. It accepts one global power-of-two block size and one ±1 sign vector **per input
  width** (INT32 arrays). `transform` must be `normalized-sylvester-walsh-hadamard` and `axis` must be
  `input-last-dimension`.
- Allowed weights: `blk.N.{attn_q,attn_k,attn_v,attn_qkv,attn_gate,attn_output,ffn_gate,ffn_up,ffn_down,
  ffn_{gate,up,down}_exps,ffn_gate_up_exps,ffn_{gate,up,down}_shexp,ssm_out}.weight` and `output.weight`.
- Allowed archs: **llama, qwen3, qwen3moe, qwen35, qwen35moe, qwen3next**. Every other arch is refused
  at load time. Qwen2/Qwen2.5 is not on the list, so the test model is Qwen3-0.6B.
- `src/llama-graph.cpp` `build_hadamard_input`: x → x·s (signs) → block-WHT (`llama_mul_mat_hadamard`,
  matrix entry (−1)^popcount(r&c)/√B) → matmul. So the stored weight must be **W_f = W·D·H**, which gives
  W_f·(H·D·x) = W·x. Tied models use **v2**: `tied_output=true`, `token_embd.weight` in
  `inverse_weight_names` (the latent table is E·D·H, the graph applies h = s∘(H·z) after the lookup and
  folds the output projection through the same table).
- **An older llama.cpp ignores the keys and loads the folded weights unchanged.** Measured on the folded
  Qwen3-0.6B f16: perplexity on b11556+ is 17.78, and on Pollard's vendored pre-#29600 runtime it is
  **138,032,203**, with no error printed.
- **ik_llama.cpp** (`b25a013d`, #2550) has its own lenient loader for the keys. It is CPU-only (it warns
  that GPU offload may crash), and on any malformed manifest it warns and drops the rotations. Its
  default fused up/gate path (`fused_up_gate=true`) calls `ggml_fused_up_gate` and never applies the
  transform for non-Prism atoms. Treat these files as **ik-incompatible**, which is how ggufcheck now
  reports them.

## What was built (branch `feat/rht-rungs`, worktree `~/Desktop/pw-rht`)

- `tools/pollard_rht.py`, installed as `pollard-rht`:
  - `fold --gguf f16 --out X [--block 32..512] [--scope all|down|ffn|attn] [--seed] [--output auto|fold|skip]`
    streams one tensor at a time and folds W·D·H in the source dtype. It writes the manifest exactly as
    the loader reads it (v1, or v2 for tied models). Per tensor it runs an identity canary (W·x vs
    W_f·(H·D·x) on the stored weights). It refuses unsupported archs and refuses to fold twice.
  - `rungs --gguf folded-f16 --calib C [--targets IQ2_XXS,IQ2_S,Q2_K,Q3_K,IQ3_S,IQ4_XS] [--bin] [--threads] [--ref-imatrix orig.imatrix]`
    computes the imatrix **on the folded model**, then builds the rungs. Only binaries that carry #29600
    are accepted (checked by a string scan of libllama). `--ref-imatrix` prints the basis check.
    `--allocation gold --sensitivity S` runs the gold ladder planner (`pollard_auto.ladder_points/pick_ladder`
    → `pollard-fit`) on the folded weights.
  - `info FILE…` prints a file's manifest.
- `pollard-precondition`: new candidates `rht-B64/128/256`, `rht-down-only` (=`rht-down-B128`) and
  `rht-{down,attn,ffn}-B<n>`, plus `--bin`, `--threads` and `--ctx`. When the binaries carry #29600 and
  the arch is supported, `rht-B128` and `rht-down-only` join the default candidates. Otherwise they are
  dropped with the reason. "none" always competes, so the selector can never pick a worse build.
- `pollard-ggufcheck`: a new verdict, **`prism`**, prints `llama.cpp>=b11556 … older builds load it
  silently WRONG; NOT ik_llama.cpp` and exits 1, so the gate makes the card say it. `read_header` keeps
  its shape; `read_header_full` adds the manifest summary.
- `pollard-card` renders the requirement in four places: the top paragraph (all-RHT and mixed ladders),
  the "which file" list, the "runs in" column, and "How to run".
- `tests/test_rht.py` has 17 tests: the runtime's matrix, fold identity, expert stacks, the exact
  manifest types and keys, tied v2, scope, refusals, binary detection, ggufcheck and candidate names.

## Quick check: Qwen3-0.6B (Mac, Metal, stock llama.cpp built from `f2cc7282c`, which contains a9d27ac)

Setup:
- Eval: wikitext-2 test, **16 chunks × 512**, run as 4 quarters × 4 chunks to keep the base-logits file
  under 0.7 GB (disk guard). Mean KLD is against the f16 base.
- imatrix: `wiki_train.txt`, 64 × 512.
- Every RHT file is byte-for-byte the same size as its baseline (+30 KB of manifest).

**Identity (the PR's test).** The folded f16 (B128, all 196 linears plus `output.weight`) against the
original f16 gives **mean KLD 0.000005**, top-1 100.00%, and PPL(Q) 17.434 vs 17.436. The worst
per-tensor canary error is 3.5e-4, which is f16 rounding.

**The imatrix collects rotated inputs.** Per folded weight, the imatrix trace matches the original
within **2.7e-4**. Per-128-block energy is preserved (median deviation 8e-5), which only holds if the
hook sees H·D·x in the right channel order. The peak/mean channel importance flattens from a median of
**166 to 13** on 100% of folded weights.

**A/B at equal bytes.** Mean KLD, lower is better. Top-1 in brackets.

| rung | none + imatrix | rht-B128 (all) | rht-B64 | rht-B256 | rht-B32 | rht-down-only | rht-attn-B128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ2_XXS | **3.909** (24.0%) | 6.469 (10.7%) | | | | 5.151 (18.1%) | |
| IQ2_S | **1.502** (47.9%) | 2.694 (34.0%) | 2.358 | 2.456 | 2.436 | 1.904 (41.3%) | 2.097 |
| Q3_K (M) | **0.2399** (77.8%) | 0.3476 (73.0%) | 0.3348 | 0.3414 | 0.3667 | 0.2666 (76.5%) | 0.2858 |
| IQ4_XS | **0.1048** (84.5%) | 0.1768 (81.2%) | | | | 0.1326 (83.4%) | |
| Q4_K_M | **0.0673** (87.2%) | 0.0922 (85.6%) | | | | | |

Without an imatrix, RHT wins on the K-quants:

| rung (no imatrix) | none | rht-B128 |
|---|---:|---:|
| Q3_K (M) | 0.4476 | **0.4085** (−8.7%) |
| Q4_K_M | 0.1087 | **0.0997** (−8.4%) |
| IQ4_XS | **0.1455** | 0.1517 |

Selector end-to-end (`pollard-precondition --target q3_k --candidates none,rht-down-only --bin <b11556+>`,
small eval): none 0.2341, rht-down-only 0.2583, so it **picked none**. With the old runtime as `--bin`,
the rht candidates are dropped with the reason.

**What the 0.6B numbers say.** The fold is exact and the runtime path works. The loss comes from the
imatrix. llama.cpp's rounding uses only the **diagonal** importance. Block RHT spreads each salient input
channel across its 128-wide block (peak/mean 166 → 13), so imatrix-weighted rounding has nothing to
protect. Rotated IQ4_XS is even worse *with* its imatrix (0.177) than without (0.152). RHT pays off when
the rounding uses the full Hessian (QuIP#/QTIP LDLQ), and llama-quantize does not. Of the variants,
**down-only loses least** because it leaves the attention and gate/up inputs, where saliency is strongest,
untouched. This matches the earlier residual-rotation result on Qwen2.5-7B, where rotation lost to
imatrix at every tier. 0.6B numbers can mislead, which is why the 8B run below makes the call.

Next levers if 8B confirms the pattern (not built):
- **Per-tensor fold choice.** The manifest takes any subset of weight names (one global block), so fold
  only the tensors whose measured error drops.
- **Full-Hessian rounding on the folded weights** (GPTQ/LDLQ, then pack to GGUF atoms).

## OVERNIGHT: Qwen3-8B full ladder A/B on the box (CPU only)

Why the box: the Mac has about 11 GB free and Qwen3-8B f16 is 16.4 GB.

Box rules apply: **CPU only** (`CUDA_VISIBLE_DEVICES=-1`, `-ngl 0`, 19 of 32 threads, `/low`). Run
`nvidia-smi` first. Check that C: has **≥ 150 GB free** (16.4 GB HF + 16.4 GB f16 + 2 × 16.4 GB folded
f16 + 5 GB KLD base + about 4 × 20 GB of rungs). Launch through `schtasks` with a past `/sd`, then `/run`,
and delete the task afterwards.

**Before night 1:** the branch exists only on the Mac (not pushed). Mario decides how it reaches
`C:\pollard\pw`: merge and pull, or copy `tools/pollard_rht.py`, `pollard_precondition.py`,
`pollard_ggufcompat.py` and `pollard_card.py`.

```bat
:: ---- one-time setup (interactive, minutes) ----
nvidia-smi
wmic logicaldisk where "DeviceID='C:'" get FreeSpace
mkdir C:\pollard\rht8b C:\pollard\llama-b11559
curl -L -o C:\pollard\rht8b\lb.zip https://github.com/ggml-org/llama.cpp/releases/download/b11559/llama-b11559-bin-win-cpu-x64.zip
tar -xf C:\pollard\rht8b\lb.zip -C C:\pollard\llama-b11559
C:\pollard\llama-b11559\llama-quantize.exe --version
findstr /m /c:"prism.hadamard.version" C:\pollard\llama-b11559\llama.dll   &:: must print the dll -> #29600 present
pollard-smoke --model Qwen/Qwen3-8B                                          &:: PREFLIGHT, seconds
```

### Night 1: the current gold path (arm C) plus its artifacts

`C:\pollard\rht8b\night1.bat`:

```bat
@echo off
set CUDA_VISIBLE_DEVICES=-1
set POLLARD_THREADS=19
cd /d C:\pollard\rht8b
pollard --hf Qwen/Qwen3-8B --run --ladder --ngl 0 --no-card > gold.log 2>&1
pollard-ls >> gold.log 2>&1
```

```bat
schtasks /create /tn rht8b1 /tr "cmd /c start \"\" /low /b /wait C:\pollard\rht8b\night1.bat" /sc once /sd 01/01/2026 /st 00:00 /f
schtasks /run /tn rht8b1
:: morning: schtasks /delete /tn rht8b1 /f
```

From `gold.log` and `pollard-ls`, record:
- `F16`: the converted f16 GGUF
- `CALIB`: `pollard_calib.txt` beside it
- `IMAT`: the `.imatrix` (legacy .dat format, original basis)
- `SENS`: the pollard-probe sensitivity json
- the gold ladder rungs (Q6_K / IQ4_XS / IQ3_S, sizes chosen by the planner) and the IQ1_KT flagship
  (ik_llama)

### Night 2: arms A (uniform + imatrix), B (RHT uniform) and D (RHT + gold allocation), then KLD for everything

`C:\pollard\rht8b\night2.bat` (fill in the four `set` lines from night 1):

```bat
@echo off
setlocal enabledelayedexpansion
set CUDA_VISIBLE_DEVICES=-1
set POLLARD_THREADS=19
set PW=C:\pollard\pw
set LB=C:\pollard\llama-b11559
set W=C:\pollard\rht8b
set EVAL=C:\pollard\bench\wikitext2_test.txt
set F16=<from gold.log>
set CALIB=<from gold.log>
set IMAT=<from gold.log>
set SENS=<from gold.log>
set RUNGS=IQ2_XXS IQ2_S Q2_K Q3_K IQ3_S IQ4_XS
cd /d %W%
mkdir A B128 DOWN D128 DDOWN f16 2>nul

:: ---- arm A: uniform + imatrix (the baseline everyone ships), same imatrix as the gold path
for %%t in (%RUNGS%) do %LB%\llama-quantize.exe --imatrix %IMAT% %F16% A\A-%%t.gguf %%t 19

:: ---- arm B: online RHT, block 128 on every linear + LM head; imatrix ON THE FOLDED MODEL, same calib
python %PW%\tools\pollard_rht.py fold --gguf %F16% --out f16\rht128-f16.gguf --block 128
%LB%\llama-imatrix.exe -m f16\rht128-f16.gguf -f %CALIB% -o f16\rht128.imatrix -ngl 0 -t 19 --output-format gguf
python %PW%\tools\pollard_rht.py rungs --gguf f16\rht128-f16.gguf --imatrix f16\rht128.imatrix --targets IQ2_XXS,IQ2_S,Q2_K,Q3_K,IQ3_S,IQ4_XS --outdir B128 --bin %LB% --ngl 0 --threads 19

:: ---- arm B': ffn_down only (least-bad on 0.6B)
python %PW%\tools\pollard_rht.py fold --gguf %F16% --out f16\rhtdown-f16.gguf --block 128 --scope down
%LB%\llama-imatrix.exe -m f16\rhtdown-f16.gguf -f %CALIB% -o f16\rhtdown.imatrix -ngl 0 -t 19 --output-format gguf
python %PW%\tools\pollard_rht.py rungs --gguf f16\rhtdown-f16.gguf --imatrix f16\rhtdown.imatrix --targets IQ2_XXS,IQ2_S,Q2_K,Q3_K,IQ3_S,IQ4_XS --outdir DOWN --bin %LB% --ngl 0 --threads 19

:: ---- arm D: the gold planner + measured allocation (pollard-fit) on the RHT weights
python %PW%\tools\pollard_rht.py rungs --gguf f16\rht128-f16.gguf --imatrix f16\rht128.imatrix --allocation gold --sensitivity %SENS% --outdir D128 --bin %LB% --threads 19
python %PW%\tools\pollard_rht.py rungs --gguf f16\rhtdown-f16.gguf --imatrix f16\rhtdown.imatrix --allocation gold --sensitivity %SENS% --outdir DDOWN --bin %LB% --threads 19

:: ---- KLD vs f16: wikitext-2 test, 32 x 512 (base file ~5 GB)
%LB%\llama-perplexity.exe -m %F16% -f %EVAL% -c 512 --chunks 32 -ngl 0 -t 19 --kl-divergence-base %W%\base.kld > base.txt 2>&1
:: identity check first: the folded f16s must read ~0
for %%m in (f16\rht128-f16.gguf f16\rhtdown-f16.gguf) do %LB%\llama-perplexity.exe -m %%m -f %EVAL% -c 512 --chunks 32 -ngl 0 -t 19 --kl-divergence-base %W%\base.kld --kl-divergence > %%m.kld.txt 2>&1
for %%d in (A B128 DOWN D128 DDOWN) do for %%m in (%%d\*.gguf) do %LB%\llama-perplexity.exe -m %%m -f %EVAL% -c 512 --chunks 32 -ngl 0 -t 19 --kl-divergence-base %W%\base.kld --kl-divergence > %%m.kld.txt 2>&1
:: arm C (gold): stock rungs with the same binary; the IQ1_KT flagship with ik_llama's perplexity
:: (if ik refuses this base file, rebuild one with ik's llama-perplexity from %F16% and score the flagship on that)
for %%m in (<gold Q6_K> <gold IQ4_XS> <gold IQ3_S>) do %LB%\llama-perplexity.exe -m %%m -f %EVAL% -c 512 --chunks 32 -ngl 0 -t 19 --kl-divergence-base %W%\base.kld --kl-divergence > %W%\gold-%%~nm.kld.txt 2>&1
python %PW%\tools\pollard_ggufcompat.py --offline A\*.gguf B128\*.gguf DOWN\*.gguf D128\*.gguf DDOWN\*.gguf > ggufcheck.txt
python %W%\summ.py > SUMMARY.txt
del %W%\base.kld
```

`C:\pollard\rht8b\summ.py`:

```python
import glob, os, re
rows = []
for t in sorted(glob.glob(r"C:\pollard\rht8b\**\*.kld.txt", recursive=True)):
    s = open(t, errors="replace").read()
    k = re.search(r"Mean\s+KLD:\s*([0-9.]+)", s); p = re.search(r"Same top p:\s*([0-9.]+)", s)
    g = t[:-8]
    size = os.path.getsize(g) / 1e9 if os.path.exists(g) else float("nan")
    rows.append((os.path.relpath(g, r"C:\pollard\rht8b"), size, k and float(k.group(1)), p and float(p.group(1))))
for r in sorted(rows, key=lambda r: (r[1], r[0])):
    print(f"{r[0]:48s} {r[1]:7.2f} GB  KLD {r[2] if r[2] is not None else 'FAIL'}  top1 {r[3]}")
```

```bat
schtasks /create /tn rht8b2 /tr "cmd /c start \"\" /low /b /wait C:\pollard\rht8b\night2.bat > C:\pollard\rht8b\night2.log 2>&1" /sc once /sd 01/01/2026 /st 00:00 /f
schtasks /run /tn rht8b2
:: morning: schtasks /delete /tn rht8b2 /f ; type C:\pollard\rht8b\SUMMARY.txt
```

Reading the board:
- Compare **A vs B128 vs DOWN** at each rung (equal bytes).
- Compare **gold (C) vs D128/DDOWN** at the planner's sizes.
- The folded-f16 identity lines must read KLD of about 0. If they do not, stop: the run is invalid.

Runtime estimate (19 threads): 2 f16 imatrix runs (about 30–60 min each, depending on the Calib 3.0
length), 30-odd KLD runs at 16k tokens (about 2–4 min each), and the f16 base (about 10 min). That fits
one night.
