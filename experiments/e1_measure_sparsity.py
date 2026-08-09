"""E1 — How sparse are FFN activations, really?

Everything downstream depends on this. Flash-offloaded inference only wins if a SMALL, PREDICTABLE
subset of weights fires per token: that is what makes prefetching possible and what windowing
reuses. Apple's numbers come from ReLU-era models, where negatives are hard zeros. Modern models
use SwiGLU (Llama/Qwen) or newer activations (Kimi-K3: "situ") which produce *small* values rather
than zeros — so "sparsity" becomes a thresholding choice, not a property.

If real sparsity is low, NO predictor helps — not a trained MLP, not a distinguished-point table —
because there is no small active set to predict. That would end this project, cheaply, which is
exactly why it runs first.

Measures, per layer, on real text:
  * fraction of FFN neurons above a magnitude threshold (the "active set")
  * how much of the layer's total output magnitude that set carries (does it MATTER?)
  * token-to-token overlap of consecutive active sets (what windowing can reuse)

Run:  python e1_measure_sparsity.py --model Qwen/Qwen3-0.6B
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("needs: pip install torch transformers")

    print(f"loading {a.model} …")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    model.eval().to(a.device)

    # Hook the FFN down-projection INPUT: that vector is the per-neuron activation, and its
    # zero/near-zero entries are precisely the rows of the down-proj we could skip loading.
    acts, hooks = {}, []

    def mk(name):
        def hook(_m, inp, _out):
            acts[name] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        return hook

    for name, mod in model.named_modules():
        if name.endswith("mlp.down_proj") or name.endswith("mlp.c_proj"):
            hooks.append(mod.register_forward_hook(mk(name)))
    if not hooks:
        sys.exit("found no FFN down-projection to hook — check the architecture's module names")
    print(f"hooked {len(hooks)} FFN layers")

    text = ("The irrigation controller computes a watering schedule from the soil readings, so that "
            "each zone gets enough water without flooding the seedling beds. " * 20)
    ids = tok(text, return_tensors="pt").input_ids[:, :a.tokens].to(a.device)
    with torch.no_grad():
        model(ids)

    rows, prev = [], {}
    for name, A in sorted(acts.items()):
        mag = A.abs()
        # Threshold relative to each token's own max — scale-free, works across activations.
        thr = mag.max(dim=-1, keepdim=True).values * 0.01
        active = mag > thr
        frac = active.float().mean().item()
        # Does the active set carry the layer's actual output energy?
        energy = (mag * active).sum().item() / max(mag.sum().item(), 1e-9)
        # Consecutive-token overlap = what windowing can reuse for free.
        ov = None
        if A.shape[0] > 1:
            a0, a1 = active[:-1], active[1:]
            inter = (a0 & a1).sum(-1).float()
            union = (a0 | a1).sum(-1).float().clamp(min=1)
            ov = (inter / union).mean().item()
        rows.append({"layer": name, "active_frac": frac, "energy_kept": energy, "overlap": ov})
        prev[name] = active

    for h in hooks:
        h.remove()

    print(f"\n{'layer':<44}{'active%':>9}{'energy%':>9}{'overlap%':>10}")
    for r in rows:
        print(f"  {r['layer']:<42}{r['active_frac']*100:>8.1f}{r['energy_kept']*100:>9.1f}"
              f"{(r['overlap'] or 0)*100:>9.1f}")

    mean_active = sum(r["active_frac"] for r in rows) / len(rows)
    mean_overlap = sum((r["overlap"] or 0) for r in rows) / len(rows)
    print(f"\n  MEAN active {mean_active*100:.1f}%   overlap {mean_overlap*100:.1f}%")
    print("\n  VERDICT:", end=" ")
    if mean_active > 0.5:
        print("DENSE — no predictor can help. H1 is dead for this architecture.")
    elif mean_active > 0.25:
        print("WEAKLY sparse — flash offload wins little; marginal.")
    else:
        print(f"SPARSE ENOUGH — a predictor has something to predict. Proceed to E2.")

    out = Path(a.out or Path(__file__).parent.parent / "data" / f"e1_{a.model.replace('/','_')}.json")
    out.write_text(json.dumps({"model": a.model, "mean_active": mean_active,
                               "mean_overlap": mean_overlap, "layers": rows}, indent=2))
    print(f"  → {out}")


if __name__ == "__main__":
    main()
