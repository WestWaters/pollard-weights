#!/usr/bin/env python3
"""pollard-pack --target wafer — a WAFER CAPACITY & THROUGHPUT PLANNER for
Cerebras-class SRAM machines. It does NOT run on a wafer and it does NOT emit
wafer weights (Cerebras is a closed 16-bit stack with no user low-bit path). It
is an OFFLINE FORECAST: reuse Pollard's hot-set/cold-bulk ranking to produce a
dense/sparse tensor plan, then forecast wafers-per-model, active-bytes/token, and
a bandwidth-bound tokens/s uplift — the numbers that actually decide cost and
speed on an SRAM machine.

HONEST MODEL (verified against Cerebras docs):
  * Inference weights are SRAM-RESIDENT and 16-bit (FP16/BF16/cbfloat16) — a 30B
    model is ~60 GB resident REGARDLESS of source quantization. So bit-width crush
    does NOT shrink the on-wafer footprint. The only wafer lever is PRUNING:
    the cores skip zeros, so Pollard's SENSITIVITY RANKING is re-cast here as
    REAP-style EXPERT PRUNING (drop the least-important whole experts), not a
    bit-width plan. MoE-only; a dense model has no wafer lever.
  * SRAM is 44 GB/wafer (frozen since WSE-2); big models span MANY wafers. Fewer
    resident bytes => fewer wafers. That is the (only, exact) capacity lever.
  * Footprint ~ TOTAL params (=> wafers); throughput ~ ACTIVE params/token. But
    single-stream t/s is layer-DEPTH bound, not active-bytes bound, so this tool
    forecasts CAPACITY (wafers), NOT a tokens/s number. No wafer is benchmarked.

    pollard-pack --gguf model-f16.gguf --target wse3t --prune-experts 0.5
    pollard-pack --gguf moe.gguf --emit-plan plan.json --expert-usage router-usage.json
"""
import argparse, json, math, sys
from pollard_calc import read_gguf_meta, gguf_to_config, analyse

# VERIFIED Cerebras SRAM specs (see pollard_competitive_landscape / verify agent).
CHIPS = {
    "wse3":  {"name": "WSE-3 (CS-3)",       "sram_gb": 44.0, "bw_pbs": 21.0},
    "wse3t": {"name": "WSE-3 Turbo (CS-4)", "sram_gb": 44.0, "bw_pbs": 43.2},
}
BYTES16 = 2.0                                   # on-wafer weights are 16-bit, period
# Calibration anchors (published single-stream, for reference — NOT used to fabricate a
# t/s number; single-stream is layer-depth bound, so we forecast CAPACITY, not speed):
#   gpt-oss-120B (5.1B active/117B) = 3000 t/s CS-3 / 4400 CS-4; Qwen3-235B-A22B = ~1450;
#   Llama-70B dense = ~2500; Llama-405B dense = 969. (gpt-oss > 70B proves active-driven.)


def plan(a_arch, prune_experts):
    """The one HONEST wafer lever: REAP-style EXPERT PRUNING on MoE. Drop a fraction
    of the (sensitivity-ranked) experts entirely -> fewer resident params -> fewer
    wafers. Cerebras publishes this at up to ~50% experts at quality parity; Pollard's
    contribution is ranking WHICH experts to drop instead of uniform. Everything is
    16-bit resident either way (no low-bit datapath), so bit-width does NOT help; and
    per-token ACTIVE params are ~unchanged (top-k still routes to k of the survivors),
    so this is a CAPACITY/wafer-cost win, not a single-stream speed win. Dense models
    have NO wafer lever here. Pure arithmetic on the arch (no I/O)."""
    total = a_arch["total"]
    active = a_arch.get("active") or total          # params touched per token (unchanged by pruning)
    is_moe = (a_arch.get("n_experts") or 0) > 0

    if is_moe:
        n_exp = a_arch["n_experts"]
        exp_p = a_arch.get("expert_params") or 0     # per-expert params (whole expert)
        experts_total = min(n_exp * exp_p, total)
        pruned = prune_experts * experts_total       # whole experts removed
    else:
        experts_total = 0.0
        pruned = 0.0                                  # dense: no honest wafer lever

    res_dense = total * BYTES16
    res_pruned = (total - pruned) * BYTES16
    return {
        "is_moe": is_moe, "total_b": total / 1e9, "active_b": active / 1e9,
        "experts_b": experts_total / 1e9, "pruned_b": pruned / 1e9,
        "res_dense_gb": res_dense / 1e9, "res_pruned_gb": res_pruned / 1e9,
        "act_gb": active * BYTES16 / 1e9, "prune_experts": prune_experts,
    }


def forecast(p, chip):
    """Capacity is EXACT arithmetic (SRAM is a hard 44 GB/wafer, weights 16-bit).
    Throughput is deliberately NOT forecast as a single number: single-stream t/s on a
    wafer is layer-DEPTH bound, not active-bytes bound, so active-bytes only sets a
    bandwidth CEILING for concurrent throughput, not a single-stream rate."""
    sram = CHIPS[chip]["sram_gb"]
    wafers_d = math.ceil(p["res_dense_gb"] / sram)
    wafers_s = math.ceil(max(p["res_pruned_gb"], 1e-9) / sram)
    return {"chip": CHIPS[chip]["name"], "sram_gb": sram,
            "wafers_dense": wafers_d, "wafers_pruned": wafers_s,
            "wafers_saved": wafers_d - wafers_s,
            "resident_cut_pct": 100 * (1 - p["res_pruned_gb"] / p["res_dense_gb"]) if p["res_dense_gb"] else 0.0}


def emit_plan(arch, prune_experts, p, f, usage=None):
    """The actionable artifact a REAP/Cerebras pipeline can consume: per-MoE-layer,
    how many experts to drop and (if a router-usage profile is supplied) WHICH ones,
    ranked least-important-first. Without usage, the plan names the method to apply."""
    n_exp = arch.get("n_experts") or 0
    layers = arch.get("layers") or 0
    drop = round(prune_experts * n_exp) if n_exp else 0
    per_layer = []
    for L in range(layers):
        row = {"layer": L, "n_experts": n_exp, "drop": drop, "keep": n_exp - drop}
        if usage and str(L) in usage:                       # resolve specific expert ids
            ranked = sorted(range(n_exp), key=lambda e: usage[str(L)][e] if e < len(usage[str(L)]) else 0.0)
            row["drop_ids"] = ranked[:drop]
        per_layer.append(row)
    return {
        "tool": "pollard-pack", "target": f["chip"], "model": arch.get("kind"),
        "prune_experts": prune_experts,
        "ranking": ("router-activation-frequency from --expert-usage (least-used dropped)"
                    if usage else
                    "SUPPLY --expert-usage <router-usage.json> to resolve specific expert ids; "
                    "else drop the N least-activated experts per layer using your calibration set"),
        "forecast": {"wafers_dense": f["wafers_dense"], "wafers_pruned": f["wafers_pruned"],
                     "wafers_saved": f["wafers_saved"], "resident_dense_gb": round(p["res_dense_gb"], 1),
                     "resident_pruned_gb": round(p["res_pruned_gb"], 1),
                     "resident_cut_pct": round(f["resident_cut_pct"], 1)},
        "per_layer": per_layer,
        "scope": "Capacity forecast + prune plan. Cerebras inference is 16-bit/closed-stack; "
                 "apply the drop-list in your own compression/REAP pipeline. Not a benchmark.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True, help="source GGUF (any precision; only arch is read)")
    ap.add_argument("--target", default="wse3t", choices=list(CHIPS))
    ap.add_argument("--prune-experts", type=float, default=0.5,
                    help="fraction of experts to REAP-prune, sensitivity-ranked (0..1; ~0.5 = parity per Cerebras)")
    ap.add_argument("--sensitivity", help="pollard sensitivity.json — ranks WHICH experts to drop")
    ap.add_argument("--expert-usage", help="router-usage json {\"<layer>\":[per-expert score,...]} to resolve drop-ids")
    ap.add_argument("--emit-plan", help="write the actionable per-layer expert-prune plan to this path")
    ap.add_argument("--json", help="write the forecast to this path")
    a = ap.parse_args()

    meta = read_gguf_meta(a.gguf)
    arch = analyse(gguf_to_config(meta, a.gguf))
    p = plan(arch, a.prune_experts)
    f = forecast(p, a.target)

    print(f"== pollard-pack (wafer capacity planner) :: {a.gguf}")
    print(f"   {arch['kind']}  total {p['total_b']:.1f}B  active/token {p['active_b']:.1f}B"
          f"  ({'MoE' if p['is_moe'] else 'dense'})")
    print(f"   target {f['chip']}  ({f['sram_gb']:.0f} GB SRAM/wafer, weights 16-bit resident)")
    if not p["is_moe"]:
        print("   NO WAFER LEVER: dense model. On-wafer weights are 16-bit regardless of source")
        print("   quant, and there is no user low-bit / inference-sparsity path — so Pollard cannot")
        print(f"   reduce the footprint. Needs {f['wafers_dense']} wafer(s) either way. (The wafer win")
        print("   is MoE expert-pruning; a dense model has none.)")
    else:
        if a.sensitivity:
            print(f"   experts to drop ranked by {a.sensitivity}")
        print(f"   REAP-prune {p['prune_experts']*100:.0f}% of experts (experts = {p['experts_b']:.1f}B of "
              f"{p['total_b']:.1f}B); hot set (router/attn/embed/shared) kept.")
        print("   ---- FORECAST (capacity = exact arithmetic on 44 GB/16-bit; NOT a benchmark) ----")
        print(f"   resident:  {p['res_dense_gb']:.1f} GB -> {p['res_pruned_gb']:.1f} GB   (-{f['resident_cut_pct']:.0f}%)")
        print(f"   WAFERS:    {f['wafers_dense']} -> {f['wafers_pruned']}   (save {f['wafers_saved']})   <- the cost lever")
        print(f"   active/token: {p['act_gb']:.2f} GB (~unchanged by pruning; top-k still routes to k survivors)")
        print("   THROUGHPUT: not forecast here — single-stream t/s on a wafer is layer-DEPTH bound,")
        print("   not active-bytes bound; active-bytes only caps CONCURRENT throughput.")
    print("   SCOPE: forecast only. Cerebras inference is 16-bit, closed-stack, no user low-bit path;")
    print("   real deployment + the sparse layout require a Cerebras partnership.")
    if a.json:
        json.dump({"arch": arch["kind"], **p, **f}, open(a.json, "w"), indent=2)
        print(f"   wrote {a.json}")
    if a.emit_plan:
        if not p["is_moe"]:
            print("   --emit-plan skipped: dense model has no expert-prune plan (no wafer lever).")
        else:
            usage = json.load(open(a.expert_usage)) if a.expert_usage else None
            json.dump(emit_plan(arch, a.prune_experts, p, f, usage), open(a.emit_plan, "w"), indent=2)
            print(f"   wrote prune plan -> {a.emit_plan}"
                  f"{'  (specific expert ids resolved)' if a.expert_usage else '  (method only; add --expert-usage for ids)'}")


if __name__ == "__main__":
    main()
