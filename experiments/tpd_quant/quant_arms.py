"""Quantization arms on tPD-decomposed matrices: does a *targeted* low-rank side path beat SVD?

For every decomposed matrix W (d_out x d_in) and every (bits, scheme) we build:

  A_rtn       Q(W)                                     plain group-wise fake-quant (RTN)
  A_matched   Q(W) with the most error-reducing groups upgraded to bits+1 until its size equals
              the B/C arms (equal-total-bits baseline that spends the side-path budget on Q instead)
  B_svd       SVD_r(W) [fp16] + Q(W - SVD_r(W))        activation-free (LoRC / SVDQuant-style)
  B_asvd      same, SVD of W*diag(s), s_k = sqrt(E_target[x_k^2])   activation-aware (ASVD-style)
  C_tpd       sum_{alive i} U_i V_i^T [fp16] + Q(Delta),  Delta = W - sum_alive U_i V_i^T
  D_task      Q(W), per-group clip chosen to minimise sum_k imp_task[k] (w-q)^2,
              imp_task[k] = E_target[ sum_i mu_i ||U_i||^2 (V_i[k] x_k)^2 ]
  D_tact      same, imp[k] = E_target[x_k^2]       (control: target activations, no decomposition)
  D_generic   same, imp[k] = E_generic[x_k^2]      (generic activation importance)

r (per matrix) = number of ALIVE tPD components (max CI over target calibration > threshold), so
B and C carry identical side-path sizes. Non-decomposed matrices stay at the model dtype.

Metrics (vs the unquantized model in --dtype, default bf16): mean per-token KLD on held-out TARGET
code and on NON-target text, perplexity on both, top-1 agreement; optional tiny HumanEval pass@1.

  python quant_arms.py --model Qwen/Qwen3-0.6B --decomp <SPD_OUT>/spd/<run>/model_5000.pth \
      --eval-sets C:/pollard/tpd_data/eval_sets.pt --out results/qwen3code_s0 [--humaneval 20]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# =============================================================================== quantizer
def _rtn_groups(G: torch.Tensor, bits: int, scheme: str, alpha: float) -> torch.Tensor:
    """Round-to-nearest on groups G[..., group]; alpha<1 shrinks the clip range (clip search)."""
    maxq = 2**bits - 1
    if scheme == "sym":
        # GPTQ-style symmetric grid: zero fixed at 2^(b-1), levels (q - 2^(b-1)) * scale
        amax = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) * alpha
        scale = 2 * amax / maxq
        zero = (maxq + 1) / 2
        q = torch.clamp(torch.round(G / scale) + zero, 0, maxq)
        return (q - zero) * scale
    if scheme == "asym":
        xmax = G.amax(-1, keepdim=True).clamp_min(0) * alpha
        xmin = G.amin(-1, keepdim=True).clamp_max(0) * alpha
        scale = ((xmax - xmin) / maxq).clamp_min(1e-12)
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(G / scale) + zero, 0, maxq)
        return (q - zero) * scale
    raise ValueError(f"unknown scheme {scheme}")


def fake_quant(
    W: torch.Tensor,
    bits: int,
    group: int = 64,
    scheme: str = "asym",
    imp: torch.Tensor | None = None,
    grid: int = 0,
    shrink_min: float = 0.5,
) -> torch.Tensor:
    """Group-wise (along d_in) fake quantization of W (d_out, d_in); returns dequantized fp32.

    grid>1: per (row, group) choose the clip ratio alpha in linspace(1, shrink_min, grid) that
    minimises sum_k imp[k] * (w_k - q_k)^2  (imp: (d_in,) per-input-channel importance).
    """
    Wf = W.float()
    d_out, d_in = Wf.shape
    assert d_in % group == 0, f"d_in={d_in} not divisible by group={group}"
    G = Wf.reshape(d_out, d_in // group, group)
    if not grid or grid <= 1:
        return _rtn_groups(G, bits, scheme, 1.0).reshape(d_out, d_in)
    if imp is None:
        imp_g = torch.ones(1, d_in // group, group, device=Wf.device)
    else:
        imp = imp.float().to(Wf.device)
        imp = imp + 1e-6 * imp.mean().clamp_min(1e-30)  # columns with ~0 importance still count a bit
        imp_g = imp.reshape(1, d_in // group, group)
    best_err: torch.Tensor | None = None
    best_Q: torch.Tensor | None = None
    for a in torch.linspace(1.0, shrink_min, grid).tolist():
        Q = _rtn_groups(G, bits, scheme, a)
        err = ((Q - G) ** 2 * imp_g).sum(-1)
        if best_err is None:
            best_err, best_Q = err, Q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_Q = torch.where(better[..., None], Q, best_Q)
    assert best_Q is not None
    return best_Q.reshape(d_out, d_in)


def q_bits(d_out: int, d_in: int, bits: int, group: int, scheme: str) -> int:
    """Storage of one quantized matrix: payload + per-group fp16 scale (+ b-bit zero for asym)."""
    n_groups = d_out * d_in // group
    per_group = 16 + (bits if scheme == "asym" else 0)
    return d_out * d_in * bits + n_groups * per_group


def side_bits_cost(d_out: int, d_in: int, r: int, side_bits: int) -> int:
    extra = 0 if side_bits >= 16 else 16 * (d_out + r)  # int8 side: fp16 per-row scales
    return r * (d_out + d_in) * side_bits + extra


def round_side(M: torch.Tensor, side_bits: int) -> torch.Tensor:
    """Storage rounding of a side-path factor: 16 -> fp16, 8 -> per-row absmax int8, 32 -> exact."""
    if side_bits >= 32:
        return M.float()
    if side_bits == 16:
        return M.half().float()
    if side_bits == 8:
        s = M.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        return torch.round(M / s).clamp(-127, 127) * s
    raise ValueError("side_bits must be 8, 16 or 32")


def matched_quant(
    W: torch.Tensor, bits: int, group: int, scheme: str, imp: torch.Tensor | None, extra_bits: int
) -> tuple[torch.Tensor, int]:
    """Q at `bits`, upgrading the groups with the largest imp-weighted error drop to bits+1 while
    `extra_bits` budget lasts. Returns (dequantized W, bits actually spent on upgrades)."""
    d_out, d_in = W.shape
    Wf = W.float()
    Qb = fake_quant(Wf, bits, group, scheme)
    if extra_bits <= 0:
        return Qb, 0
    Qb1 = fake_quant(Wf, bits + 1, group, scheme)
    G = Wf.reshape(d_out, -1, group)
    w = torch.ones(1, d_in // group, group, device=Wf.device) if imp is None else imp.float().to(Wf.device).reshape(1, -1, group)
    eb = ((Qb.reshape_as(G) - G) ** 2 * w).sum(-1)
    eb1 = ((Qb1.reshape_as(G) - G) ** 2 * w).sum(-1)
    gain = (eb - eb1).flatten()
    cost = group + (1 if scheme == "asym" else 0)
    n_up = min(int(extra_bits // cost), gain.numel())
    mask = torch.zeros_like(gain, dtype=torch.bool)
    if n_up > 0:
        mask[torch.topk(gain, n_up).indices] = True
    mask = mask.reshape(d_out, -1, 1)
    Q = torch.where(mask, Qb1.reshape_as(G), Qb.reshape_as(G)).reshape(d_out, d_in)
    return Q, n_up * cost


def svd_side(W: torch.Tensor, r: int, col_scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-r factors (L: d_out x r, R: r x d_in) with L @ R ~= W. col_scale -> ASVD-style
    (SVD of W diag(s), then fold 1/s back into R)."""
    Wf = W.float()
    if r <= 0:
        return Wf.new_zeros(Wf.shape[0], 0), Wf.new_zeros(0, Wf.shape[1])
    s = None
    if col_scale is not None:
        s = col_scale.float().to(Wf.device).clamp_min(1e-8)
        Wf = Wf * s[None, :]
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    L = U[:, :r] * S[:r]
    R = Vh[:r]
    if s is not None:
        R = R / s[None, :]
    return L, R


# =============================================================================== decomposition
@dataclass
class Decomposition:
    """tPD components per module path. U: (C, d_out), V: (d_in, C)  ->  U_i V_i^T = outer(U[i], V[:, i])."""

    U: dict[str, torch.Tensor]
    V: dict[str, torch.Tensor]
    target_weights: dict[str, torch.Tensor] = field(default_factory=dict)  # from checkpoint, for a sanity check
    config: dict = field(default_factory=dict)

    @property
    def modules(self) -> list[str]:
        return sorted(self.U)

    def side_factors(self, name: str, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = alive.nonzero().flatten().to(self.U[name].device)
        return self.U[name][idx].T.float(), self.V[name][:, idx].T.float()  # L (d_out x r), R (r x d_in)


def load_tpd_checkpoint(path: str | Path) -> Decomposition:
    """Read a tPD `model_<step>.pth` (ComponentModel.state_dict) without importing spd.

    Keys: `_components.<module-path-with-dashes>.{U,V}`, `target_model.<path>.weight`.
    """
    import yaml

    path = Path(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    U: dict[str, torch.Tensor] = {}
    V: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("_components.") and k[-2:] in (".U", ".V"):
            name = k[len("_components.") : -2].replace("-", ".")
            (U if k.endswith(".U") else V)[name] = v.float()
    assert U and set(U) == set(V), f"no/partial tPD components in {path} (keys like {list(sd)[:5]})"
    tw = {n: sd[f"target_model.{n}.weight"].float() for n in U if f"target_model.{n}.weight" in sd}
    cfg_path = path.parent / "final_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    return Decomposition(U=U, V=V, target_weights=tw, config=cfg)


def make_spd_ci_fn(ckpt: str | Path, device: str) -> tuple[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]], torch.nn.Module]:
    """Load the trained ComponentModel with tPD's own code; return (ci_fn(cache)->mu, fp32 target model)."""
    from spd.models.component_model import ComponentModel

    cm = ComponentModel.from_pretrained(str(ckpt))
    cm.to(device).eval()
    use_ac = device.startswith("cuda") and cm_autocast(ckpt)

    @torch.no_grad()
    def ci_fn(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
            out = cm.calc_causal_importances(pre_weight_acts=cache, sampling="continuous", detach_inputs=True)
        return {k: v.float().clamp(0, 1) for k, v in out.lower_leaky.items()}

    return ci_fn, cm.target_model


def cm_autocast(ckpt: str | Path) -> bool:
    import yaml

    p = Path(ckpt).parent / "final_config.yaml"
    return bool(yaml.safe_load(p.read_text()).get("autocast_bf16", False)) if p.exists() else False


# =============================================================================== statistics
@dataclass
class Stats:
    x2: dict[str, torch.Tensor]  # E[x_k^2]  (d_in,)
    imp_task: dict[str, torch.Tensor] | None = None  # E[x_k^2 sum_i mu_i |U_i|^2 V[k,i]^2]
    mu_max: dict[str, torch.Tensor] | None = None  # (C,)
    mu_mean: dict[str, torch.Tensor] | None = None


@torch.no_grad()
def collect_stats(
    model: torch.nn.Module,
    modules: list[str],
    tokens: torch.Tensor,
    batch_size: int,
    device: str,
    decomp: Decomposition | None = None,
    ci_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    autocast_bf16: bool = False,
) -> Stats:
    """One pass over `tokens`, capturing the inputs of every module in `modules`."""
    cache: dict[str, torch.Tensor] = {}
    handles = []
    for name in modules:
        mod = model.get_submodule(name)
        handles.append(mod.register_forward_pre_hook(lambda m, a, n=name: cache.__setitem__(n, a[0].detach())))
    x2 = {n: None for n in modules}
    task = {n: None for n in modules} if decomp is not None else None
    mu_max = {n: None for n in modules} if decomp is not None else None
    mu_sum = {n: None for n in modules} if decomp is not None else None
    n_tok = 0
    try:
        for i in range(0, tokens.shape[0], batch_size):
            b = tokens[i : i + batch_size].to(device)
            cache.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_bf16 and device.startswith("cuda")):
                model(b)
            n_tok += b.numel()
            mus = None
            if decomp is not None:
                mus = ci_fn(cache) if ci_fn is not None else None
            for n in modules:
                x = cache[n].float()
                xx = x * x  # (B, T, d_in)
                acc = xx.sum((0, 1))
                x2[n] = acc if x2[n] is None else x2[n] + acc
                if decomp is not None:
                    C = decomp.U[n].shape[0]
                    mu = mus[n].float() if mus is not None else torch.ones(*x.shape[:-1], C, device=x.device)
                    unorm2 = decomp.U[n].to(x.device).float().pow(2).sum(-1)  # (C,)
                    V2 = decomp.V[n].to(x.device).float().pow(2)  # (d_in, C)
                    t = (xx * ((mu * unorm2) @ V2.T)).sum((0, 1))
                    task[n] = t if task[n] is None else task[n] + t
                    mm = mu.amax((0, 1))
                    mu_max[n] = mm if mu_max[n] is None else torch.maximum(mu_max[n], mm)
                    ms = mu.sum((0, 1))
                    mu_sum[n] = ms if mu_sum[n] is None else mu_sum[n] + ms
    finally:
        for h in handles:
            h.remove()
    n_pos = n_tok
    st = Stats(x2={n: v / n_pos for n, v in x2.items()})
    if decomp is not None:
        st.imp_task = {n: v / n_pos for n, v in task.items()}
        st.mu_max = mu_max
        st.mu_mean = {n: v / n_pos for n, v in mu_sum.items()}
    return st


def alive_mask(decomp: Decomposition, stats: Stats | None, ci_mode: str, threshold: float) -> dict[str, torch.Tensor]:
    out = {}
    for n in decomp.modules:
        if ci_mode == "spd":
            assert stats is not None and stats.mu_max is not None
            out[n] = (stats.mu_max[n].cpu() > threshold)
        else:  # "ones": no CI available -> components with non-negligible norm are alive
            norm = decomp.U[n].norm(dim=1) * decomp.V[n].norm(dim=0)
            out[n] = norm > 1e-6 * norm.max().clamp_min(1e-30)
    return out


# =============================================================================== evaluation
class WeightPatcher:
    def __init__(self, model: torch.nn.Module, modules: list[str]):
        self.mods = {n: model.get_submodule(n) for n in modules}
        self.orig = {n: m.weight.data.clone() for n, m in self.mods.items()}

    @torch.no_grad()
    def set(self, ws: dict[str, torch.Tensor] | None) -> None:
        for n, m in self.mods.items():
            src = self.orig[n] if ws is None or n not in ws else ws[n]
            m.weight.data.copy_(src.to(m.weight.dtype))


@torch.no_grad()
def eval_arm(
    model: torch.nn.Module,
    patcher: WeightPatcher,
    ws: dict[str, torch.Tensor] | None,
    tokens: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """KLD(ref || arm) per token (all positions), NLL on next-token, top-1 agreement."""
    kl_sum = nll_sum = agree = 0.0
    n_kl = n_nll = 0
    for i in range(0, tokens.shape[0], batch_size):
        b = tokens[i : i + batch_size].to(device)
        patcher.set(None)
        ref_logits = model(b).logits
        if ws is not None:
            patcher.set(ws)
            logits = model(b).logits
        else:
            logits = ref_logits
        for row in range(b.shape[0]):  # one row at a time: bounds fp32 vocab-sized temporaries
            ref_lp = F.log_softmax(ref_logits[row].float(), -1)
            lp = F.log_softmax(logits[row].float(), -1)
            kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1).clamp_min(0)
            kl_sum += kl.sum().item()
            n_kl += kl.numel()
            tgt = b[row, 1:]
            nll_sum += -lp[:-1].gather(-1, tgt[:, None]).sum().item()
            n_nll += tgt.numel()
            agree += (lp.argmax(-1) == ref_lp.argmax(-1)).float().sum().item()
        del ref_logits, logits
    patcher.set(None)
    return {
        "kld": kl_sum / n_kl,
        "ppl": math.exp(nll_sum / n_nll),
        "top1": agree / n_kl,
    }


# =============================================================================== HumanEval (optional)
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert "]


def run_program(src: str, timeout: float = 10.0) -> bool:
    """Execute untrusted model code in a fresh interpreter subprocess with a timeout."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prog.py"
        p.write_text(src, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(p)], cwd=td, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


@torch.no_grad()
def humaneval_pass1(model, tok, patcher: WeightPatcher, ws, n: int, device: str, max_new: int = 384) -> float:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    patcher.set(ws)
    passed = 0
    try:
        for ex in list(ds)[:n]:
            ids = tok(ex["prompt"], return_tensors="pt").input_ids.to(device)
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                 pad_token_id=tok.eos_token_id)
            comp = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
            cut = min([comp.find(s) for s in STOP_SEQS if s in comp] + [len(comp)])
            prog = ex["prompt"] + comp[:cut] + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
            passed += run_program(prog)
    finally:
        patcher.set(None)
    return passed / n


# =============================================================================== arms
CALIB_INDEPENDENT = {"A_rtn", "B_svd", "C_tpd"}
ALL_ARMS = ["A_rtn", "A_matched", "B_svd", "B_asvd", "C_tpd", "D_task", "D_tact", "D_generic"]


def build_arm(
    arm: str,
    W: torch.Tensor,
    name: str,
    bits: int,
    scheme: str,
    args: argparse.Namespace,
    decomp: Decomposition,
    alive: torch.Tensor,
    st_t: Stats,
    st_g: Stats,
) -> tuple[torch.Tensor, int]:
    """Return (dequantized weight, total storage bits) for one matrix."""
    d_out, d_in = W.shape
    g, grid = args.group_size, args.scale_grid
    r = int(alive.sum())
    qb = q_bits(d_out, d_in, bits, g, scheme)
    side = side_bits_cost(d_out, d_in, r, args.side_bits)
    if arm == "A_rtn":
        return fake_quant(W, bits, g, scheme), qb
    if arm == "A_matched":
        Q, spent = matched_quant(W, bits, g, scheme, st_g.x2[name], side)
        return Q, qb + spent
    if arm in ("B_svd", "B_asvd"):
        cs = st_t.x2[name].sqrt() if arm == "B_asvd" else None
        L, R = svd_side(W, r, cs)
        P = round_side(L, args.side_bits) @ round_side(R, args.side_bits)
        return P + fake_quant(W - P, bits, g, scheme), qb + side
    if arm == "C_tpd":
        L, R = decomp.side_factors(name, alive)
        L, R = L.to(W.device), R.to(W.device)
        P = round_side(L, args.side_bits) @ round_side(R, args.side_bits)
        return P + fake_quant(W - P, bits, g, scheme), qb + side
    if arm == "D_task":
        assert st_t.imp_task is not None
        return fake_quant(W, bits, g, scheme, st_t.imp_task[name], grid), qb
    if arm == "D_tact":
        return fake_quant(W, bits, g, scheme, st_t.x2[name], grid), qb
    if arm == "D_generic":
        return fake_quant(W, bits, g, scheme, st_g.x2[name], grid), qb
    raise ValueError(arm)


def verdicts(rows: list[dict], seeds: list[int]) -> dict:
    out: dict = {"C_vs_B": [], "D_task_vs_generic": []}
    key = lambda r: (r["bits"], r["scheme"])  # noqa: E731
    configs = sorted({key(r) for r in rows if r["arm"] != "bf16"})
    get = lambda arm, cfg, seed: next((r for r in rows if r["arm"] == arm and key(r) == cfg and r["seed"] == seed), None)  # noqa: E731
    s0 = seeds[0]
    for cfg in configs:
        C = get("C_tpd", cfg, s0)
        Bs = [b for b in (get("B_svd", cfg, s0), get("B_asvd", cfg, s0)) if b is not None]
        if C and Bs:
            B = min(Bs, key=lambda r: r["target_kld"])
            rel = (B["target_kld"] - C["target_kld"]) / max(B["target_kld"], 1e-30)
            nt_ratio = C["nontarget_kld"] / max(B["nontarget_kld"], 1e-30)
            passed = rel >= 0.20 and nt_ratio <= 1.10
            note = "PASS" if passed else ("C~=B: functional decomposition adds nothing over SVD at this scale"
                                         if abs(rel) < 0.05 else "FAIL")
            out["C_vs_B"].append({
                "bits": cfg[0], "scheme": cfg[1], "best_B": B["arm"],
                "target_kld_C": C["target_kld"], "target_kld_B": B["target_kld"],
                "target_rel_improvement": rel, "nontarget_kld_ratio_C_over_B": nt_ratio,
                "pass": passed, "verdict": note,
            })
        per_seed = []
        for s in seeds:
            dt, dg = get("D_task", cfg, s), get("D_generic", cfg, s)
            if dt and dg:
                per_seed.append({"seed": s, "D_task": dt["target_kld"], "D_generic": dg["target_kld"],
                                 "D_tact": (get("D_tact", cfg, s) or {}).get("target_kld"),
                                 "task_wins": dt["target_kld"] < dg["target_kld"]})
        if per_seed:
            out["D_task_vs_generic"].append({
                "bits": cfg[0], "scheme": cfg[1], "per_seed": per_seed,
                "pass": len(per_seed) >= 2 and all(p["task_wins"] for p in per_seed),
            })
    return out


def to_markdown(res: dict) -> str:
    L = ["# tPD quant arms", "", f"model `{res['meta']['model']}` · decomp `{res['meta']['decomp']}` · "
         f"{res['meta']['n_modules']} matrices · mean alive r = {res['meta']['mean_r']:.1f} · "
         f"group {res['meta']['group_size']} · side path {res['meta']['side_bits']}-bit", "",
         "| seed | bits | scheme | arm | bpw (decomp. mats) | tgt KLD | non-tgt KLD | tgt PPL | non-tgt PPL | tgt top1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['seed']} | {r['bits']} | {r['scheme']} | {r['arm']} | {r['bpw']:.3f} | "
                 f"{r['target_kld']:.5f} | {r['nontarget_kld']:.5f} | {r['target_ppl']:.3f} | "
                 f"{r['nontarget_ppl']:.3f} | {r['target_top1']:.4f} |")
    L += ["", "## Verdicts", "", "**C (tPD side path) vs best B (SVD side path), equal size** — pass = "
          "target KLD ≥20% lower and non-target KLD ≤1.10× B", "",
          "| bits | scheme | best B | C tgt KLD | B tgt KLD | rel. improvement | non-tgt C/B | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for v in res["verdicts"]["C_vs_B"]:
        L.append(f"| {v['bits']} | {v['scheme']} | {v['best_B']} | {v['target_kld_C']:.5f} | "
                 f"{v['target_kld_B']:.5f} | {100 * v['target_rel_improvement']:+.1f}% | "
                 f"{v['nontarget_kld_ratio_C_over_B']:.3f} | {v['verdict']} |")
    L += ["", "**D: task-targeted vs generic importance for group scales** — pass = D_task lower target KLD on every seed (≥2 seeds)", "",
          "| bits | scheme | per seed (D_task / D_tact / D_generic) | pass |", "|---|---|---|---|"]
    for v in res["verdicts"]["D_task_vs_generic"]:
        ps = "; ".join(f"s{p['seed']}: {p['D_task']:.5f} / {p['D_tact'] if p['D_tact'] is None else format(p['D_tact'], '.5f')} / {p['D_generic']:.5f}" for p in v["per_seed"])
        L.append(f"| {v['bits']} | {v['scheme']} | {ps} | {'PASS' if v['pass'] else 'FAIL'} |")
    if res.get("humaneval"):
        L += ["", "## HumanEval pass@1 (greedy, first N problems)", "", "| arm | pass@1 |", "|---|---|"]
        for k, v in res["humaneval"].items():
            L.append(f"| {k} | {v:.3f} |")
    return "\n".join(L) + "\n"


# =============================================================================== main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local dir of the base model")
    ap.add_argument("--decomp", required=True, help="tPD checkpoint model_<step>.pth (final_config.yaml beside it)")
    ap.add_argument("--eval-sets", required=True, help="eval_sets.pt from prep_data.py")
    ap.add_argument("--out", required=True, help="output dir (results.json, results.md)")
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--schemes", nargs="+", default=["sym", "asym"], choices=["sym", "asym"])
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--side-bits", type=int, default=16, choices=[8, 16, 32])
    ap.add_argument("--scale-grid", type=int, default=20, help="clip-ratio grid for D arms")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="calibration resampling seeds")
    ap.add_argument("--calib-rows", type=int, default=64, help="rows sampled per seed from each calib pool")
    ap.add_argument("--n-target-eval", type=int, default=64)
    ap.add_argument("--n-nontarget-eval", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--ci-mode", choices=["spd", "ones"], default="spd",
                    help="spd: per-token CI from the trained tPD CI fn (needs the tPD package); "
                         "ones: mu=1, alive = non-zero-norm components (smoke tests)")
    ap.add_argument("--alive-threshold", type=float, default=None, help="default: ci_alive_threshold from the run config")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--humaneval", type=int, default=0, help="N HumanEval problems (0 = off); executes model code")
    ap.add_argument("--humaneval-arms", nargs="+", default=["A_rtn", "B_svd", "C_tpd"])
    ap.add_argument("--humaneval-config", default="3:asym", help="bits:scheme used for the HumanEval arms")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    t0 = time.time()
    dev = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    decomp = load_tpd_checkpoint(args.decomp)
    modules = decomp.modules
    thr = args.alive_threshold if args.alive_threshold is not None else float(decomp.config.get("ci_alive_threshold", 0.01))
    sets = torch.load(args.eval_sets, map_location="cpu", weights_only=False)
    tgt_eval = sets["target_eval"][: args.n_target_eval]
    nt_eval = sets["nontarget_eval"][: args.n_nontarget_eval]
    tgt_pool, gen_pool = sets["target_calib"], sets["generic_calib"]
    print(f"[quant_arms] {len(modules)} decomposed matrices; eval tgt {tuple(tgt_eval.shape)} non-tgt {tuple(nt_eval.shape)}")

    # ---- alive set + task importance need the CI function (tPD's ComponentModel) -> fp32 stats model
    ci_fn = None
    if args.ci_mode == "spd":
        ci_fn, stats_model = make_spd_ci_fn(args.decomp, dev)
        stats_ac = cm_autocast(args.decomp)
    else:
        from tpd_models import _load_causal_lm

        stats_model = _load_causal_lm(args.model, torch.float32).to(dev).eval()
        stats_ac = False
    # sanity: the checkpoint's frozen target weights must equal the model we quantize
    for n in modules:
        if n in decomp.target_weights:
            w_model = stats_model.get_submodule(n).weight.detach().float().cpu()
            err = (w_model - decomp.target_weights[n]).abs().max().item()
            assert err < 1e-2, f"{n}: checkpoint target weight != model weight (max abs diff {err})"
    st_alive = collect_stats(stats_model, modules, tgt_pool, args.eval_batch, dev, decomp, ci_fn, stats_ac)
    alive = alive_mask(decomp, st_alive, args.ci_mode, thr)
    per_seed_stats: dict[int, tuple[Stats, Stats]] = {}
    for s in args.seeds:
        g = torch.Generator().manual_seed(s)
        ti = torch.randperm(tgt_pool.shape[0], generator=g)[: args.calib_rows]
        gi = torch.randperm(gen_pool.shape[0], generator=g)[: args.calib_rows]
        st_t = collect_stats(stats_model, modules, tgt_pool[ti], args.eval_batch, dev, decomp, ci_fn, stats_ac)
        st_g = collect_stats(stats_model, modules, gen_pool[gi], args.eval_batch, dev, None, None, stats_ac)
        per_seed_stats[s] = (st_t, st_g)
    del stats_model, ci_fn
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- eval model in --dtype
    from tpd_models import _load_causal_lm

    model = _load_causal_lm(args.model, dtype).to(dev).eval()
    patcher = WeightPatcher(model, modules)
    W_full = {n: patcher.orig[n].float() for n in modules}
    n_weights = sum(W.numel() for W in W_full.values())

    mod_info = {}
    for n in modules:
        W = W_full[n]
        r = int(alive[n].sum())
        L, R = decomp.side_factors(n, alive[n])
        P = (L.to(W.device) @ R.to(W.device))
        Ls, Rs = svd_side(W, r)
        e = W.pow(2).sum().item()
        mod_info[n] = {
            "shape": list(W.shape), "C": int(decomp.U[n].shape[0]), "alive_r": r,
            "energy_tpd_side": P.pow(2).sum().item() / e,
            "energy_svd_side": (Ls @ Rs).pow(2).sum().item() / e if r else 0.0,
            "rel_err_W_minus_tpd": (W - P).norm().item() / math.sqrt(e),
            "rel_err_W_minus_svd": (W - Ls @ Rs).norm().item() / math.sqrt(e),
        }
    mean_r = sum(v["alive_r"] for v in mod_info.values()) / len(mod_info)
    print(f"[quant_arms] alive components per matrix: mean {mean_r:.1f} "
          f"(min {min(v['alive_r'] for v in mod_info.values())}, max {max(v['alive_r'] for v in mod_info.values())})")

    rows: list[dict] = []
    base_t = eval_arm(model, patcher, None, tgt_eval, args.eval_batch, dev)
    base_n = eval_arm(model, patcher, None, nt_eval, args.eval_batch, dev)
    rows.append({"seed": "-", "bits": 16, "scheme": "-", "arm": "bf16" if dtype == torch.bfloat16 else args.dtype,
                 "bpw": 16.0, "target_kld": base_t["kld"], "nontarget_kld": base_n["kld"],
                 "target_ppl": base_t["ppl"], "nontarget_ppl": base_n["ppl"], "target_top1": base_t["top1"]})
    he_bits, he_scheme = args.humaneval_config.split(":")
    he_store: dict[str, dict[str, torch.Tensor]] = {}

    for si, s in enumerate(args.seeds):
        st_t, st_g = per_seed_stats[s]
        for bits in args.bits:
            for scheme in args.schemes:
                for arm in args.arms:
                    if si > 0 and arm in CALIB_INDEPENDENT:
                        prev = next(r for r in rows if r["arm"] == arm and r["bits"] == bits and r["scheme"] == scheme and r["seed"] == args.seeds[0])
                        rows.append({**prev, "seed": s, "reused_from_seed": args.seeds[0]})
                        continue
                    ws, total_bits = {}, 0
                    for n in modules:
                        Wq, b = build_arm(arm, W_full[n], n, bits, scheme, args, decomp, alive[n], st_t, st_g)
                        ws[n] = Wq
                        total_bits += b
                    rt = eval_arm(model, patcher, ws, tgt_eval, args.eval_batch, dev)
                    rn = eval_arm(model, patcher, ws, nt_eval, args.eval_batch, dev)
                    row = {"seed": s, "bits": bits, "scheme": scheme, "arm": arm, "bpw": total_bits / n_weights,
                           "target_kld": rt["kld"], "nontarget_kld": rn["kld"], "target_ppl": rt["ppl"],
                           "nontarget_ppl": rn["ppl"], "target_top1": rt["top1"]}
                    rows.append(row)
                    print(f"[quant_arms] s{s} {bits}b {scheme:4s} {arm:9s} bpw {row['bpw']:.3f} "
                          f"tgtKLD {rt['kld']:.5f} ntKLD {rn['kld']:.5f} tgtPPL {rt['ppl']:.2f} ({time.time() - t0:.0f}s)")
                    if args.humaneval and si == 0 and arm in args.humaneval_arms and str(bits) == he_bits and scheme == he_scheme:
                        he_store[arm] = {n: w.to(dtype).cpu() for n, w in ws.items()}
                    del ws

    he = {}
    if args.humaneval:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        he["bf16" if dtype == torch.bfloat16 else args.dtype] = humaneval_pass1(model, tok, patcher, None, args.humaneval, dev)
        for arm, ws in he_store.items():
            he[f"{arm}@{he_bits}b-{he_scheme}"] = humaneval_pass1(model, tok, patcher, ws, args.humaneval, dev)
        print(f"[quant_arms] HumanEval pass@1 (n={args.humaneval}): {he}")

    res = {
        "meta": {"model": args.model, "decomp": str(args.decomp), "n_modules": len(modules), "mean_r": mean_r,
                 "group_size": args.group_size, "side_bits": args.side_bits, "ci_mode": args.ci_mode,
                 "alive_threshold": thr, "dtype": args.dtype, "seeds": args.seeds, "calib_rows": args.calib_rows,
                 "n_target_eval": int(tgt_eval.shape[0]), "n_nontarget_eval": int(nt_eval.shape[0]),
                 "eval_seq": int(tgt_eval.shape[1]), "wall_s": time.time() - t0},
        "modules": mod_info,
        "rows": rows,
        "verdicts": verdicts([r for r in rows if r["arm"] in ALL_ARMS], args.seeds),
        "humaneval": he,
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    (out_dir / "results.md").write_text(to_markdown(res))
    print(f"[quant_arms] wrote {out_dir / 'results.json'} and results.md ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    main()
