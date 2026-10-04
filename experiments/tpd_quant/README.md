# tpd_quant: does Targeted Parameter Decomposition give a better low-bit side path than SVD?

> **GPU etiquette.** The GPU stages run on Mario's Windows box (RTX 5070 Ti, **16 GB**), and only
> **when Mario says the GPU is free**. `run_box.ps1` refuses to start a GPU stage if `nvidia-smi` shows
> another compute process or more than 1.5 GB in use (`-Force` overrides this). Data prep is CPU-only and can
> run at any time. The kit uses its own venv, `C:\pollard\tpd-venv` (Python 3.13 + torch 2.8.0 cu128). It
> never uses `C:\pollard\venv312\Scripts\python.exe` and never installs anything into it. See `setup.md`.

## What and why

tPD (arXiv 2607.13047, code: `github.com/Antovigo/targeted-parameter-decomposition`, MIT, a fork of SPD)
splits chosen weight matrices into rank-1 components `U_i V_iᵀ` that a **target** dataset needs, plus a
full-rank residual `Δ = W − Σ U_i V_iᵀ` that holds everything the target doesn't need. Training uses
output-KL, stochastic and adversarial (persistent-PGD) mask reconstruction, and importance-minimality, with
one target batch and one non-target batch per step.

**Hypothesis.** Quantize `Δ` hard and keep `Σ_alive U_i V_iᵀ` in fp16 as a low-rank side path (like an
adapter). This should give better target-task KLD than an SVD side path of the same rank
(LoRC/SVDQuant-style) at the same total bits.

Setup: Qwen/Qwen3-0.6B, with the target being Python code (`codeparrot/codeparrot-clean-valid`, split at
document level). The non-target set is wikitext-103 plus UltraChat, about 50/50 by tokens. We decompose
`mlp.{gate,up,down}_proj` in layers 16–23: 24 matrices with C=256 each, 5k steps at batch 16×128.

### Arms (per matrix, for 2- and 3-bit, sym and asym, group 64, `quant_arms.py`)

| arm | weight | size |
|---|---|---|
| `A_rtn` | Q(W) | Q |
| `A_matched` | Q(W), with the groups that cut error most moved up to b+1 bits until the size equals B/C | = B = C |
| `B_svd` | SVD_r(W) [fp16] + Q(W − SVD_r(W)) | Q + side |
| `B_asvd` | same, but the SVD is of W·diag(√E_tgt[x²]) (activation-aware, ASVD-style) | Q + side |
| `C_tpd` | Σ_alive U_i V_iᵀ [fp16] + Q(Δ) | Q + side |
| `D_task` | Q(W) with a per-group clip that minimizes Σ_k imp[k](w−q)², imp[k] = E_tgt[Σ_i μ_i‖U_i‖²(V_i[k]x_k)²] | Q |
| `D_tact` | same, imp = E_tgt[x_k²] (control: target activations without the decomposition) | Q |
| `D_generic` | same, imp = E_generic[x_k²] | Q |

- `r` is the number of **alive** tPD components for that matrix: the maximum CI over target calibration
  data has to exceed the run's `ci_alive_threshold` (0.01). B and C are therefore **exactly** the same size.
- `μ_i` is the per-token causal importance from tPD's own trained CI function (`--ci-mode spd`).
- Q is group-wise RTN along d_in. Sym is the GPTQ-style grid with an fp16 scale. Asym is min-max with an fp16
  scale and a b-bit zero point. The side path costs `r·(d_in+d_out)·16` bits (`--side-bits 8` makes it
  int8 per row).
- Metrics, measured against the unquantized bf16 model: mean per-token KLD on held-out **target** code
  (64×512 tokens) and on **non-target** text (64×512), perplexity on both, and top-1 agreement. With
  `--humaneval N` it also reports greedy pass@1 on the first N HumanEval problems, for `A_rtn`/`B_svd`/
  `C_tpd` at 3-bit asym plus bf16. ⚠ This runs model-generated code in a subprocess with a 10 s timeout.
- "Seeds" are calibration-resampling seeds 0 and 1. Calibration-free arms (`A_rtn`, `B_svd`, `C_tpd`) are
  computed once and reused. For a stronger seed test, run a second decomposition with `-Seed 1`.
- `bpw` in the table counts the decomposed matrices only. Everything else stays bf16 in all arms.

Results go to `results.json` (all rows, per-matrix `alive_r`, the share of ‖W‖² captured by the tPD side
path vs the SVD side path, and the verdicts) and `results.md` (tables).

## Success criteria (`quant_arms.py` checks these automatically)

1. **C beats B.** For a given bit width and scheme, `C_tpd` has **at least 20% lower target KLD** (relative)
   than the better of `B_svd`/`B_asvd`, at identical size. Its **non-target KLD can be no more than 10%
   worse** than that B arm's.
2. **D beats generic.** `D_task` has lower target KLD than `D_generic` on **both** seeds. `D_tact` is the
   control: if `D_task ≈ D_tact`, the target-importance gain comes from target activations, not from the
   decomposition.
3. **If C ≈ B** (less than 5% apart), report it as: *functional decomposition adds nothing over SVD at this
   scale.*
4. Look at `A_matched` alongside C. It spends the same side-path bits on the quantizer instead. If it beats
   C, then the side path itself is the wrong place to spend the bits at this size.

## Run it

### One-time setup (CPU and network only): see `setup.md`

```bash
scp -P 3333 -i ~/.ssh/pollard_5070ti -r ~/Desktop/Pollard-Weights/experiments/tpd_quant jwate@localhost:C:/pollard/tpd_quant
```
```powershell
cd C:\pollard\tpd_quant
powershell -ExecutionPolicy Bypass -File setup.ps1          # tPD clone + C:\pollard\tpd-venv
powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage prep   # data -> C:/pollard/tpd_data (CPU)
```

### GPU stages (only after Mario says the GPU is free)

```powershell
powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage preflight   # 3 steps, prints peak VRAM + time/step
powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage decomp      # the 5k-step decomposition
powershell -ExecutionPolicy Bypass -File run_box.ps1 -Stage arms        # quant arms + verdicts
```

Or use the whole GPU chain detached, so it survives the SSH session:

```powershell
schtasks /Create /TN tpd_quant /SC ONCE /ST 23:59 /F /TR "powershell -ExecutionPolicy Bypass -File C:\pollard\tpd_quant\run_box.ps1 -Stage gpu"
schtasks /Run /TN tpd_quant
```

Logs are written to `C:/pollard/tpd_out/run_box_*.log`. tPD's own metrics go to
`C:/pollard/tpd_out/spd/qwen3code-s0/metrics.jsonl`, with a checkpoint every 1000 steps in `model_*.pth`.
Results are in `C:/pollard/tpd_out/results/qwen3code-s0/results.{json,md}`.

If preflight prints `TOO CLOSE`, or the run OOMs, use `-Batch 8` on both `preflight` and `decomp`.

Raw commands (what `run_box.ps1` runs):

```powershell
$Py = "C:\pollard\tpd-venv\Scripts\python.exe"
& $Py prep_data.py --out C:/pollard/tpd_data
& $Py run_tpd.py --data-root C:/pollard/tpd_data --out-dir C:/pollard/tpd_out --run-id qwen3code-s0 --batch 16 --preflight
& $Py run_tpd.py --data-root C:/pollard/tpd_data --out-dir C:/pollard/tpd_out --run-id qwen3code-s0 --batch 16
& $Py quant_arms.py --model Qwen/Qwen3-0.6B --decomp C:/pollard/tpd_out/spd/qwen3code-s0/model_5000.pth `
      --eval-sets C:/pollard/tpd_data/eval_sets.pt --out C:/pollard/tpd_out/results/qwen3code-s0 --seeds 0 1 [--humaneval 20]
```

### Expected runtime and VRAM (estimates; preflight gives the real numbers)

| stage | device | time | VRAM |
|---|---|---|---|
| setup + prep | CPU / network | ~10–20 min | – |
| preflight | GPU | ~2 min | it measures this |
| decomposition, 5k × (16×128 target + 16×128 non-target) | GPU | **~2–4 h** (about 1.5–2.5 s/step, each step ≈ 8–10 forwards + 5 backwards of 0.6B with a 152k vocab) | **~10–15 GB** at batch 16 (fp32 master weights 2.4 GB + 3 live graphs with vocab-sized KL); about 8 GB at batch 8 |
| quant arms (52 arm evals × 128 rows × 512 tokens, 2 seeds) | GPU | ~15–30 min | ~4–5 GB |

The decomposition dominates the cost. Preflight runs only 3 steps, so the step-time estimate it prints
includes model load and the step-0 eval, which makes it an upper bound.

## Files

| file | what |
|---|---|
| `config_qwen3_06b_code.yaml` | tPD config, validated against tPD's own pydantic `Config` (exact schema, `extra="forbid"`) |
| `run_tpd.py` | single-GPU launcher: puts this dir on PYTHONPATH, sets `SPD_OUT_DIR`, disables wandb, overrides via `--data-root/--batch/--seed/--steps/--set k=v`, `--preflight` |
| `tpd_models.py` | `Qwen3ForTPD.from_pretrained`: fp32 master weights and `use_cache=False` (see "tPD format notes") |
| `prep_data.py` | tokenizes and packs target and non-target data into Hub-layout parquet plus `eval_sets.pt` |
| `quant_arms.py` | the arms, metrics, verdicts, JSON and markdown output |
| `smoke_cpu.py` | CPU dry run (below) |
| `setup.ps1` / `setup.md` | box setup |
| `run_box.ps1` | box pipeline with the GPU-free guard |

## Smoke test (already passing on the Mac, CPU only)

```bash
.venv/bin/python experiments/tpd_quant/smoke_cpu.py                 # fake decomposition, ~10 s, ~0.45 GB RSS
<py3.13 venv with tPD>/bin/python experiments/tpd_quant/smoke_cpu.py --tpd-e2e   # + REAL tPD, ~35 s, ~1 GB
```

The default mode builds a random 2-layer Qwen3 and a fake checkpoint in tPD's exact state-dict layout (6
SVD-like, 4 small and 6 dead components per matrix), then runs every arm, the KLD/PPL code and the verdicts,
and checks that B and C are the same size, that 3-bit beats 2-bit, that KLD(ref, ref) = 0, that 10 components
are alive, and so on. `--tpd-e2e` also validates this YAML against tPD's `Config`, writes parquet the way
`prep_data.py` does, runs a **real 4-step tPD decomposition** through `run_tpd.py`, and runs
`quant_arms.py --ci-mode spd` on the resulting checkpoint. The smoke numbers come from a random model and
mean nothing.

## tPD format notes (read from the repo at `edbfb6c`)

- **Config schema**: `spd/configs.py::Config`. Unknown keys are an error. This YAML follows the paper's
  code-target reference (`spd/experiments/lm/targeted_decomposition/css/config_css_reference.yaml`): a
  global shared-transformer CI function, binomial sampling, `leaky_hard`, ImpMin + StochasticReconSubset +
  PersistentPGDRecon (adversarial) + UnmaskedRecon, KL output loss, and `use_delta_component: true`. These
  are our choices rather than the paper's: C=256 everywhere, 5k steps, lr 3e-4 (the paper used 1e-4–5e-4
  over 30–50k steps), batch 16, seq 128.
- **Module selection**: `module_info` entries are fnmatch patterns, each with its own C:
  `model.layers.1[6-9].mlp.*_proj` and `model.layers.2[0-3].mlp.*_proj`. Checked on a 28-layer Qwen3
  skeleton, these expand to exactly the 24 intended matrices.
- **Data**: `task_config` and `nontarget_task_config` are `LMTaskConfig`s, which take *one*
  `dataset_name`, have no HF config-name field and no mixing. That's why `prep_data.py` pre-tokenizes into
  local parquet directories (`is_tokenized: true`, `column_name: input_ids`). Local Hub-layout directories
  load fine with `load_dataset(dir, split=...)`, which the e2e smoke checks.
- **Checkpoint**: `$SPD_OUT_DIR/spd/<run_id>/model_<step>.pth` is the full `ComponentModel.state_dict()`.
  Components are under `_components.<module-path-with-dashes>.{U,V}`, with `U: (C, d_out)` and
  `V: (d_in, C)`. The component weight is `(V@U)ᵀ`, so `Δ = W − (V@U)ᵀ` (as in
  `ComponentModel.calc_weight_deltas`). `final_config.yaml` sits next to it. `quant_arms.py` reads U/V
  without importing spd, but `--ci-mode spd` loads `ComponentModel.from_pretrained(<pth>)` to get per-token CI.
- **Gradient checkpointing is not used**, on purpose. tPD swaps components in through forward hooks that
  only exist inside `ComponentModel.forward`. Checkpoint recomputation happens in backward, after those hooks
  have been removed, so the gradients would be silently wrong. Memory is controlled with `--batch`, and tPD's
  bf16 autocast is on.
- **Model dtype**: `pretrained_model_class: tpd_models.Qwen3ForTPD` loads fp32. Plain `AutoModelForCausalLM`
  would give bf16 under newer transformers, and tPD would then train its U/V in bf16.
- **Python 3.13 only** (`requires-python ==3.13.*`). Install from the `uv.lock` pins. An unpinned wandb
  breaks `import spd`.
- **Offline mode**: transformers 4.57.3 (the locked version) calls the Hub API when it loads some tokenizers
  even if `HF_HUB_OFFLINE=1` is set. Leave offline mode off.
