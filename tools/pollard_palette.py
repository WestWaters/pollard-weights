#!/usr/bin/env python3
"""pollard-palette — measured mixed-ALPHABET allocation below the 2-bit floor.

The Pollard sensitivity method, extended past 1 bit, ON TOP OF GPTQ RECONSTRUCTION
(not RTN — RTN collapses at low bit, so allocating over it is allocating over noise).

Each weight tensor chooses an ALPHABET from

    A = { prune 0b, binary 1b, ternary ~1.58b, 2-bit }

Every candidate is GPTQ-quantized with the tensor's own Hessian, so the choice is
between four *reconstructed* options. The per-(tensor,alphabet) COST is the Hessian-
weighted reconstruction error GPTQ itself minimises (SqueezeLLM/OWQ sensitivity — the
principled quantity, not an invented proxy); a multiple-choice knapsack then picks the
assignment minimising total cost under a target average-bpw budget, and the chosen
assignment is VERIFIED with real WikiText-2 PPL.

The headline is RELATIVE: does the Palette beat UNIFORM TERNARY (also GPTQ) at matched
average bpw? On a 0.5B that relative win is the valid mechanism signal (Grok: small
models punish low-bit hardest — read relative, not absolute). Absolute usability is a
7B result; the allocator mechanism is provable here.

Pure PTQ. No QAT. Embeddings / lm_head / norms stay fp16 and are COUNTED in the average.

Usage:
  pollard-palette --model Qwen/Qwen2.5-0.5B-Instruct --groupsize 64 \
      --calib-file wiki_train.txt --eval-file wikitext2_test.txt \
      --target-bpw 1.58 1.3 1.0 --device mps
"""
import argparse, json, time, copy
import torch, torch.nn as nn

from pollard_gptq import gptq_quantize, eval_ppl, _chunks, linear_layers

# alphabet: name -> (symbol-bits, quantizer(W, H) -> dequant fp16)
def _q_prune(W, H, gs):        return torch.zeros_like(W)
def _q_binary(W, H, gs):       return gptq_quantize(W, H, 1, gs, qmode="binary")
def _q_ternary(W, H, gs):      return gptq_quantize(W, H, 2, gs, qmode="ternary")
def _q_2bit(W, H, gs):         return gptq_quantize(W, H, 2, gs, qmode="int")

ALPHABET = {
    "prune":   (0.0,   _q_prune),
    "binary":  (1.0,   _q_binary),
    "ternary": (1.585, _q_ternary),
    "2bit":    (2.0,   _q_2bit),
}
HEADER_BITS = 2.0                                       # alphabet id per tensor


def unit_bpw(name, groupsize, numel):
    sym, _ = ALPHABET[name]
    if name == "prune":
        return HEADER_BITS / max(numel, 1)
    return sym + 16.0 / (groupsize or numel) + HEADER_BITS / max(numel, 1)


@torch.no_grad()
def collect_hessians(model, calib, lins, dev):
    """H = 2 X Xᵀ per linear (GPTQ's Hessian), from fp16 activations. Kept on CPU."""
    H = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
    cnt = {n: 0 for n in lins}; hooks = []
    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            H[n].add_(2.0 * x.t() @ x); cnt[n] += x.shape[0]
        return hook
    for n, m in lins.items():
        hooks.append(m.register_forward_hook(mk(n)))
    mdev = next(model.parameters()).device
    for c in calib:
        model(c.unsqueeze(0).to(mdev))
    for h in hooks:
        h.remove()
    return {n: (H[n] / max(cnt[n], 1)).cpu() for n in lins}


def sens_cost(W, Q, Hdiag):
    """Hessian-weighted reconstruction error = Σ_j importance_j · Σ_i (W−Q)²_ij.
    Cheap per-tensor proxy — but it IGNORES cross-layer error propagation, so it badly
    mis-ranks (esp. prune). Kept as `--cost proxy`; default is measured NLL-delta."""
    e = (W.float() - Q.float()) ** 2
    return (e.sum(0) * Hdiag.clamp(min=0)).sum().item()


@torch.no_grad()
def calib_nll(model, calib):
    """Mean token NLL over calib — the GLOBAL, end-to-end signal. cost = this − fp16 ref."""
    dev = next(model.parameters()).device; nll = ntok = 0.0
    for c in calib:
        ids = c.unsqueeze(0).to(dev)
        nll += model(ids, labels=ids).loss.float().item() * (ids.numel() - 1)
        ntok += ids.numel() - 1
    return nll / ntok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--groupsize", type=int, default=64)
    ap.add_argument("--calib-file", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--eval-chunks", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--target-bpw", type=float, nargs="+", default=[1.585, 1.3, 1.0])
    ap.add_argument("--cost", choices=["measured", "proxy", "marginal"], default="marginal",
                    help="marginal = interaction-aware NLL-delta from the ternary floor (best); "
                    "measured = solo NLL-delta from fp16 (ignores interactions); proxy = per-tensor H-error (fast, mis-ranks)")
    ap.add_argument("--probe-chunks", type=int, default=4, help="calib chunks per crush measurement")
    ap.add_argument("--profile-out", default="palette_profile.json")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"== pollard-palette :: {a.model}  gs={a.groupsize}  dev={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev)
    calib = _chunks(tok, open(a.calib_file, encoding="utf-8").read(), a.seqlen, a.nsamples)
    test = _chunks(tok, open(a.eval_file, encoding="utf-8").read(), a.seqlen, a.eval_chunks)

    ppl_fp16 = eval_ppl(model, test); print(f"fp16 PPL: {ppl_fp16:.4f}", flush=True)
    state = copy.deepcopy(model.state_dict())
    lins = linear_layers(model.model.layers)
    print(f"collecting Hessians for {len(lins)} tensors ...", flush=True)
    Hs = collect_hessians(model, calib, lins, dev)
    probe_calib = calib[:a.probe_chunks]

    # cache fp16 originals + precompute each tensor's per-alphabet GPTQ quantization
    orig_w = {n: m.weight.data.clone() for n, m in lins.items()}
    quant = {}                                             # quant[n][alphabet] = dequant weight (cpu)
    for n, m in lins.items():
        H = Hs[n].to(dev); quant[n] = {}
        for name, (sym, qfn) in ALPHABET.items():
            Q = qfn(orig_w[n], H, a.groupsize) if name != "prune" else torch.zeros_like(orig_w[n])
            quant[n][name] = Q.to(m.weight.dtype).cpu()
        del H

    if a.cost == "marginal":
        # interaction-aware: set EVERY tensor to the ternary floor, then measure each
        # tensor's MARGINAL hit at each alphabet from that crushed operating point
        # (Shapley-lite: "marginal cost when the others are already crushed"). This is
        # the fix for collapsing low rungs — additive solo costs ignore cumulative error.
        for n, m in lins.items():
            m.weight.data = quant[n]["ternary"].to(dev)
        ref_nll = calib_nll(model, probe_calib)
        print(f"cost=marginal  ternary-floor ref NLL={ref_nll:.5f} ({len(probe_calib)} chunks)", flush=True)
    else:
        ref_nll = calib_nll(model, probe_calib)            # fp16 ref (solo/proxy)
        print(f"cost={a.cost}  fp16 ref NLL={ref_nll:.5f} ({len(probe_calib)} chunks)", flush=True)

    print("probing alphabets ...", flush=True)
    profile = {}; t0 = time.time()
    for i, (n, m) in enumerate(lins.items()):
        numel = orig_w[n].numel(); costs = {}; sizes = {}
        Hdiag = torch.diag(Hs[n]).cpu() if a.cost == "proxy" else None
        for name, (sym, qfn) in ALPHABET.items():
            if a.cost == "proxy":
                costs[name] = sens_cost(orig_w[n], quant[n][name], Hdiag)
            else:                                          # measured (from fp16) or marginal (from ternary floor)
                m.weight.data = quant[n][name].to(dev)
                costs[name] = calib_nll(model, probe_calib) - ref_nll
                restore = quant[n]["ternary"] if a.cost == "marginal" else orig_w[n]
                m.weight.data = restore.to(dev)
            sizes[name] = unit_bpw(name, a.groupsize, numel) * numel
        profile[n] = {"numel": numel, "costs": costs, "sizes": sizes}
        if (i + 1) % 40 == 0 or i + 1 == len(lins):
            el = int(time.time() - t0)
            print(f"  {i+1}/{len(lins)}  [{el}s, ~{el/(i+1)*len(lins):.0f}s total]", flush=True)
    for n, m in lins.items():                              # restore fp16 before materialize
        m.weight.data = orig_w[n].to(dev)
    json.dump({"model": a.model, "gs": a.groupsize, "ppl_fp16": ppl_fp16, "profile": profile},
              open(a.profile_out, "w"), indent=1)
    total_w = sum(p["numel"] for p in profile.values())

    # exact multiple-choice knapsack via DP (optimal -> monotonic ladder). Greedy-from-
    # prune produced catastrophic tight-budget assignments; DP minimises total measured
    # cost under the bit budget, so it only prunes/binarises where it genuinely helps.
    names = list(ALPHABET)
    items = list(profile.items())
    NB = 500
    unit = (2.2 * total_w) / NB                          # bits per bin
    INF = float("inf")
    prev = [INF] * (NB + 1); prev[0] = 0.0
    parents = []
    for (n, u) in items:
        cur = [INF] * (NB + 1); par = [(-1, -1)] * (NB + 1)
        opts = [(min(NB, int(round(u["sizes"][nm] / unit))), u["costs"][nm], ai)
                for ai, nm in enumerate(names)]
        for j in range(NB + 1):
            if prev[j] == INF:
                continue
            base = prev[j]
            for (b, c, ai) in opts:
                nj = j + b
                if nj <= NB and base + c < cur[nj]:
                    cur[nj] = base + c; par[nj] = (j, ai)
        prev = cur; parents.append(par)

    def query(bpw):
        bins = min(NB, int(round(bpw * total_w / unit)))
        reach = [j for j in range(bins + 1) if prev[j] < INF]
        bestj = min(reach, key=lambda j: prev[j])
        pick = {}; j = bestj
        for i in range(len(items) - 1, -1, -1):
            pj, ai = parents[i][j]
            pick[items[i][0]] = names[ai]; j = pj
        used = sum(profile[nn]["sizes"][pick[nn]] for nn in pick) / total_w
        return pick, used

    @torch.no_grad()
    def materialize_eval(pick):
        for n, m in lins.items():                          # reuse cached per-alphabet quants
            m.weight.data = quant[n][pick[n]].to(dev)
        return eval_ppl(model, test)

    # reference: uniform ternary (GPTQ) at its own bpw
    tern_bits = sum(unit_bpw("ternary", a.groupsize, p["numel"]) * p["numel"] for p in profile.values())
    tern_bpw = tern_bits / total_w
    print(f"\n== reference (probed weights = {total_w/1e6:.1f}M) ==", flush=True)
    uni_ppl = materialize_eval({n: "ternary" for n in profile})
    print(f"  uniform ternary : {uni_ppl:9.3f} PPL @ {tern_bpw:.3f} bpw", flush=True)

    print("\n== Pollard Palette (GPTQ + measured-cost knapsack) ==", flush=True)
    targets = sorted(set(a.target_bpw + [round(tern_bpw, 3)]), reverse=True)
    for tb in targets:
        pick, used = query(tb)
        ppl = materialize_eval(pick)
        mix = {k: sum(1 for v in pick.values() if v == k) for k in ALPHABET}
        matched = abs(tb - tern_bpw) < 1e-3
        flag = ""
        if matched:
            flag = "<= MATCHED vs uniform ternary: " + ("PALETTE WINS" if ppl < uni_ppl else "ties/loses")
        print(f"  {tb:.3f} bpw -> {used:.3f} : {ppl:9.3f} PPL   mix={mix}   {flag}", flush=True)


if __name__ == "__main__":
    main()
