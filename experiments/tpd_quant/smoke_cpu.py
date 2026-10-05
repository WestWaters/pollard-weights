"""CPU dry run of the whole quant-arm pipeline -- proves the kit before any GPU time.

Default (any venv with torch + transformers + pyyaml, no downloads):
  * builds a 2-layer randomly initialised Qwen3 (vocab 512, d=64) on disk,
  * synthetic eval_sets.pt (target rows draw tokens 0..127, non-target rows 128..511),
  * a FAKE tPD checkpoint in the exact tPD state-dict layout (`_components.<path>.U/V`,
    `target_model.<path>.weight`, final_config.yaml): per matrix C=16 rank-1 components =
    6 perturbed SVD directions + 4 random small + 6 exactly-zero (dead),
  * unit checks (quantizer, bit accounting, SVD, KLD(ref,ref)=0, sandboxed program runner),
  * runs quant_arms.main() end to end (--ci-mode ones, 2 seeds, 2/3-bit, sym/asym, all 8 arms)
    and asserts on the JSON/markdown.

--tpd-e2e (needs the tPD package importable, i.e. run with the tpd venv):
  * validates config_qwen3_06b_code.yaml against tPD's pydantic Config,
  * writes tiny pre-tokenized parquet datasets with prep_data.write_split_parquet,
  * runs a REAL 4-step tPD decomposition through run_tpd.py on a tiny Qwen3 (Qwen3 tokenizer
    from the local HF cache), then quant_arms.py --ci-mode spd on the produced checkpoint.

  .venv/bin/python experiments/tpd_quant/smoke_cpu.py [--tpd-e2e] [--keep DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(min(4, os.cpu_count() or 1))  # the Mac is shared; stay light

import quant_arms as qa  # noqa: E402

MODS = ["model.layers.1.mlp.gate_proj", "model.layers.1.mlp.up_proj", "model.layers.1.mlp.down_proj"]


def tiny_qwen3(path: Path, vocab: int = 512, tokenizer_from: str | None = None) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=True, initializer_range=0.2, use_cache=False)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.save_pretrained(path)
    if tokenizer_from:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer_from).save_pretrained(path)


def fake_eval_sets(path: Path, seq: int = 64, lo_hi_t=(0, 128), lo_hi_n=(128, 512)) -> None:
    g = torch.Generator().manual_seed(0)
    mk = lambda n, lh: torch.randint(lh[0], lh[1], (n, seq), generator=g)  # noqa: E731
    torch.save({"target_eval": mk(8, lo_hi_t), "nontarget_eval": mk(8, lo_hi_n),
                "target_calib": mk(16, lo_hi_t), "generic_calib": mk(16, lo_hi_n), "meta": {"fake": True}}, path)


def fake_tpd_checkpoint(model_dir: Path, run_dir: Path, C: int = 16) -> None:
    """Exactly the tPD ComponentModel.state_dict() key layout quant_arms.load_tpd_checkpoint reads."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    sd = {}
    for n in MODS:
        W = model.get_submodule(n).weight.detach().float()
        d_out, d_in = W.shape
        Us, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U = torch.zeros(C, d_out)
        V = torch.zeros(d_in, C)
        for i in range(6):  # perturbed top singular directions -> a meaningful side path
            U[i] = Us[:, i] * S[i] + 0.05 * S[i] * torch.randn(d_out, generator=g) / math.sqrt(d_out)
            V[:, i] = Vh[i] + 0.05 * torch.randn(d_in, generator=g) / math.sqrt(d_in)
        for i in range(6, 10):  # small random components
            U[i] = 0.01 * torch.randn(d_out, generator=g)
            V[:, i] = 0.01 * torch.randn(d_in, generator=g)
        # 10..15 stay exactly zero -> dead
        key = n.replace(".", "-")
        sd[f"_components.{key}.U"] = U
        sd[f"_components.{key}.V"] = V
        sd[f"target_model.{n}.weight"] = W.clone()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sd, run_dir / "model_10.pth")
    (run_dir / "final_config.yaml").write_text("ci_alive_threshold: 0.01\nautocast_bf16: false\n")


def unit_checks() -> None:
    torch.manual_seed(0)
    W = torch.randn(32, 128)
    for scheme in ("sym", "asym"):
        e2 = (qa.fake_quant(W, 2, 64, scheme) - W).pow(2).mean()
        e3 = (qa.fake_quant(W, 3, 64, scheme) - W).pow(2).mean()
        e8 = (qa.fake_quant(W, 8, 64, scheme) - W).pow(2).mean()
        assert e8 < e3 < e2, (scheme, e2, e3, e8)
        Q = qa.fake_quant(W, 2, 64, scheme).reshape(32, 2, 64)
        n_levels = max(len(torch.unique(Q[r, g])) for r in range(32) for g in range(2))
        assert n_levels <= 4, f"{scheme}: {n_levels} levels in a 2-bit group"
        # clip search with uniform importance can only match or beat plain RTN (alpha=1 is in the grid)
        Qs = qa.fake_quant(W, 2, 64, scheme, imp=torch.ones(128), grid=10)
        assert (Qs - W).pow(2).sum() <= (qa.fake_quant(W, 2, 64, scheme) - W).pow(2).sum() + 1e-5
    assert qa.q_bits(32, 128, 2, 64, "sym") == 32 * 128 * 2 + 64 * 16
    assert qa.q_bits(32, 128, 2, 64, "asym") == 32 * 128 * 2 + 64 * 18
    L, R = qa.svd_side(W, 32)
    assert torch.allclose(L @ R, W, atol=1e-4), "full-rank SVD must reconstruct W"
    L, R = qa.svd_side(W, 5, col_scale=torch.rand(128) + 0.5)
    assert L.shape == (32, 5) and R.shape == (5, 128)
    budget = qa.side_bits_cost(32, 128, 3, 16)
    Qm, spent = qa.matched_quant(W, 2, 64, "asym", None, budget)
    assert 0 < spent <= budget and (Qm - W).pow(2).sum() < (qa.fake_quant(W, 2, 64, "asym") - W).pow(2).sum()
    assert qa.run_program("assert 1 + 1 == 2\n") and not qa.run_program("raise SystemExit(1)\n")
    assert not qa.run_program("while True: pass\n", timeout=1.0)
    print("[smoke] unit checks OK")


def check_results(res: dict, seeds: list[int], arms: list[str]) -> None:
    rows = res["rows"]
    want = 1 + len(seeds) * 2 * 2 * len(arms)
    assert len(rows) == want, f"{len(rows)} rows, expected {want}"
    base = rows[0]
    assert base["target_kld"] < 1e-6 and base["nontarget_kld"] < 1e-6, "KLD(ref, ref) must be 0"
    for r in rows:
        for k in ("target_kld", "nontarget_kld", "target_ppl", "nontarget_ppl", "bpw"):
            assert math.isfinite(r[k]) and r[k] >= 0, (r, k)
    get = lambda arm, b, s, seed=seeds[0]: next(r for r in rows if r["arm"] == arm and r["bits"] == b and r["scheme"] == s and r["seed"] == seed)  # noqa: E731
    for b in (2, 3):
        for s in ("sym", "asym"):
            assert abs(get("B_svd", b, s)["bpw"] - get("C_tpd", b, s)["bpw"]) < 1e-9, "B and C must be equal size"
            assert abs(get("B_asvd", b, s)["bpw"] - get("C_tpd", b, s)["bpw"]) < 1e-9
            # matched arm spends the same budget up to one group's cost
            assert get("A_matched", b, s)["bpw"] <= get("C_tpd", b, s)["bpw"] + 1e-9
            assert get("A_matched", b, s)["bpw"] > get("A_rtn", b, s)["bpw"]
            assert get("A_rtn", b, s)["target_kld"] > 0
        assert get("A_rtn", 3, "asym")["target_kld"] < get("A_rtn", 2, "asym")["target_kld"], "3-bit must beat 2-bit"
    assert get("B_svd", 2, "sym", seeds[1])["target_kld"] == get("B_svd", 2, "sym")["target_kld"], "calib-independent arm reused"
    v = res["verdicts"]
    assert len(v["C_vs_B"]) == 4 and len(v["D_task_vs_generic"]) == 4
    assert all(len(d["per_seed"]) == len(seeds) for d in v["D_task_vs_generic"])
    rs = [m["alive_r"] for m in res["modules"].values()]
    assert all(r == 10 for r in rs), f"alive r should be 10 (6 svd-ish + 4 small, 6 dead): {rs}"
    print("[smoke] result checks OK")


def run_fake(work: Path) -> None:
    model_dir, run_dir = work / "tiny_qwen3", work / "fake_tpd_run"
    tiny_qwen3(model_dir)
    fake_eval_sets(work / "eval_sets.pt")
    fake_tpd_checkpoint(model_dir, run_dir)
    seeds = [0, 1]
    res = qa.main([
        "--model", str(model_dir), "--decomp", str(run_dir / "model_10.pth"),
        "--eval-sets", str(work / "eval_sets.pt"), "--out", str(work / "results_fake"),
        "--ci-mode", "ones", "--dtype", "fp32", "--device", "cpu",
        "--seeds", *map(str, seeds), "--calib-rows", "8", "--eval-batch", "4", "--scale-grid", "8",
        "--n-target-eval", "8", "--n-nontarget-eval", "8",
    ])
    check_results(res, seeds, qa.ALL_ARMS)
    print((work / "results_fake" / "results.md").read_text())


def run_tpd_e2e(work: Path) -> None:
    import yaml
    from spd.configs import Config

    cfg_path = HERE / "config_qwen3_06b_code.yaml"
    Config.from_file(cfg_path)
    print(f"[smoke-e2e] {cfg_path.name} validates against tPD's Config schema")

    from prep_data import write_split_parquet

    model_dir = work / "tiny_qwen3_tok"
    tiny_qwen3(model_dir, vocab=151936, tokenizer_from="Qwen/Qwen3-4B")
    g = torch.Generator().manual_seed(0)
    data = work / "tpd_data"
    seq = 32
    for name, lo, hi in (("target_code", 0, 2000), ("nontarget_mix", 2000, 6000)):
        for split, n in (("train", 64), ("validation", 16)):
            write_split_parquet(torch.randint(lo, hi, (n, seq), generator=g).tolist(), data / name, split)
    from datasets import load_dataset

    assert len(load_dataset(str(data / "target_code"), split="validation")) == 16, "Hub-layout parquet not picked up"
    fake_eval_sets(work / "eval_sets_tok.pt", seq=32, lo_hi_t=(0, 2000), lo_hi_n=(2000, 6000))

    sets = [
        "module_info=[{module_pattern: 'model.layers.1.mlp.*_proj', C: 8}]",
        "ci_config.simple_transformer_ci_cfg={d_model: 32, n_blocks: 1, mlp_hidden_dim: [64], attn_config: {n_heads: 2, max_len: 64}}",
        f"task_config.max_seq_len={seq}", f"nontarget_task_config.max_seq_len={seq}",
        "autocast_bf16=false", "eval_freq=2", "slow_eval_freq=4", "n_eval_steps=1", "train_log_freq=1",
        "save_freq=null",
    ]
    cmd = [sys.executable, str(HERE / "run_tpd.py"), "--model", str(model_dir), "--data-root", str(data),
           "--out-dir", str(work / "spd_out"), "--run-id", "smoke", "--steps", "4", "--batch", "2"]
    for s in sets:
        cmd += ["--set", s]
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = (p.stdout + p.stderr)[-3000:]
    assert p.returncode == 0, f"run_tpd failed:\n{tail}"
    print(f"[smoke-e2e] real tPD run (4 steps) OK in {time.time() - t0:.0f}s; tail:\n{tail[-600:]}")
    ckpt = work / "spd_out" / "spd" / "smoke" / "model_4.pth"
    assert ckpt.exists(), list((work / "spd_out" / "spd" / "smoke").iterdir())
    final = yaml.safe_load((ckpt.parent / "final_config.yaml").read_text())
    assert final["pretrained_model_class"] == "tpd_models.Qwen3ForTPD"

    res = qa.main([
        "--model", str(model_dir), "--decomp", str(ckpt), "--eval-sets", str(work / "eval_sets_tok.pt"),
        "--out", str(work / "results_e2e"), "--ci-mode", "spd", "--dtype", "fp32", "--device", "cpu",
        "--seeds", "0", "1", "--calib-rows", "8", "--eval-batch", "2", "--scale-grid", "6",
        "--n-target-eval", "4", "--n-nontarget-eval", "4", "--bits", "2", "3",
    ])
    assert len(res["rows"]) == 1 + 2 * 2 * 2 * len(qa.ALL_ARMS)
    print(f"[smoke-e2e] quant_arms --ci-mode spd on the real tPD checkpoint OK "
          f"(alive r per matrix: {[m['alive_r'] for m in res['modules'].values()]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tpd-e2e", action="store_true")
    ap.add_argument("--keep", default=None, help="keep artifacts in this dir instead of a temp dir")
    args = ap.parse_args()
    t0 = time.time()
    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="tpd_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        unit_checks()
        run_fake(work)
        if args.tpd_e2e:
            run_tpd_e2e(work)
        print(f"[smoke] ALL PASSED in {time.time() - t0:.0f}s")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
