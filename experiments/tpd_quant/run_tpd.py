"""Launch a tPD decomposition with this kit's config, single GPU, no wandb, no torchrun.

  python run_tpd.py --config config_qwen3_06b_code.yaml --data-root C:/pollard/tpd_data \
      --out-dir C:/pollard/tpd_out --run-id qwen3code-s0 [--preflight] [--batch 8] [--seed 1]

Result: <out-dir>/spd/<run-id>/model_<steps>.pth + final_config.yaml (+ metrics.jsonl logs).
quant_arms.py takes that .pth via --decomp.

--preflight: runs 3 optimisation steps and prints peak CUDA memory, so a batch size that will not
fit the 16 GB card fails in ~1 minute instead of at step 1.

Gradient checkpointing is deliberately NOT enabled: tPD swaps components in via forward hooks that
exist only inside ComponentModel.forward's context manager. Checkpoint recomputation happens in
backward, after those hooks are removed, so the recomputed activations would come from the
*original* weights and the component gradients would be silently wrong. Memory is controlled
with --batch instead (tPD's bf16 autocast is on in the config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def build_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        root = args.data_root.rstrip("/\\")
        cfg["task_config"]["dataset_name"] = f"{root}/target_code"
        cfg["nontarget_task_config"]["dataset_name"] = f"{root}/nontarget_mix"
    if args.model:
        cfg["pretrained_model_name"] = args.model
        cfg["tokenizer_name"] = args.model
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch:
        for k in ("batch_size", "eval_batch_size", "nontarget_batch_size", "nontarget_eval_batch_size"):
            cfg[k] = args.batch
    if args.steps is not None:
        cfg["steps"] = args.steps
    for item in args.set or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key, yaml.safe_load(raw))
    if args.preflight:
        cfg["steps"] = 3
        cfg["save_freq"] = None
        cfg["n_eval_steps"] = 1
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config_qwen3_06b_code.yaml"))
    ap.add_argument("--data-root", default=None, help="dir produced by prep_data.py")
    ap.add_argument("--out-dir", default=None, help="sets SPD_OUT_DIR (default ~/spd_out)")
    ap.add_argument("--run-id", default=None, help="fixed run dir name (default: tPD random s-xxxx)")
    ap.add_argument("--model", default=None, help="override pretrained_model_name + tokenizer_name")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="override all 4 batch sizes")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--set", action="append", help="dotted.key=<yaml value>, repeatable")
    ap.add_argument("--preflight", action="store_true", help="3 steps + peak-memory report")
    ap.add_argument("--dump-config", default=None, help="write the final config JSON here and exit")
    args = ap.parse_args()

    cfg = build_config(args)
    if args.dump_config:
        Path(args.dump_config).write_text(json.dumps(cfg, indent=2))
        print(f"wrote {args.dump_config}")
        return

    # Must happen before `import spd` (spd.settings reads SPD_OUT_DIR at import time).
    if args.out_dir:
        os.environ["SPD_OUT_DIR"] = args.out_dir
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(HERE))  # tpd_models.Qwen3ForTPD

    import torch

    from spd.configs import Config
    from spd.experiments.lm.lm_decomposition import main as lm_main

    Config(**cfg)  # validate up front: fails fast on schema errors with a readable message
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lm_main(config_json="json:" + json.dumps(cfg), run_id=args.run_id)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
    print(f"[run_tpd] done: steps={cfg['steps']} batch={cfg['batch_size']} wall={dt:.0f}s "
          f"({dt / max(cfg['steps'], 1):.2f}s/step) peak_cuda_alloc={peak:.2f} GiB")
    if args.preflight and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        verdict = "OK" if peak < 0.85 * total else "TOO CLOSE - lower --batch"
        print(f"[run_tpd] preflight: {peak:.2f} / {total:.1f} GiB -> {verdict}")
        print(f"[run_tpd] full-run estimate at {yaml.safe_load(Path(args.config).read_text())['steps']}"
              f" steps: ~{dt / 3 * yaml.safe_load(Path(args.config).read_text())['steps'] / 3600:.1f} h"
              " (upper bound: includes model load + step-0 eval)")


if __name__ == "__main__":
    main()
