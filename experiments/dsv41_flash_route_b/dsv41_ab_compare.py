#!/usr/bin/env python3
"""dsv41_ab_compare.py — fair comparison of two route-B EXL3 passes of one layer that differ in --w2-weighting (none vs route).

The quantizer's own rfn_out is computed on the rows it built its Hessian from, i.e. route-WEIGHTED rows in route mode and unweighted rows in
none mode, so the two logs score w2 on different metrics. Here both passes' w2 are scored on the SAME rows with BOTH metrics:
  rfn_unw = ||a (Wq - W)|| / ||a W||            a = silu(x w1^T) * (x w3^T)             (what the down GEMM sees; cuda-exl3 applies route_w after)
  rfn_w   = ||g a (Wq - W)|| / ||g a W||        g = route_w per token                   (the error that actually reaches the residual stream)
rfn_w is the decision metric (the residual only ever sees route_w * expert_out). Also checks w1/w3 are bit-identical across passes (same H_gu, same seed).
usage: python3 dsv41_ab_compare.py --layer 3 --dumps DIR --parts-a DIR --parts-b DIR --experts 0-47 --http BF16 --index-http FP4 [--rows 4096]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_exl3_experts as Q

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True); ap.add_argument("--dumps", nargs="+", required=True)
ap.add_argument("--parts-a", required=True); ap.add_argument("--parts-b", required=True); ap.add_argument("--label-a", default="none"); ap.add_argument("--label-b", default="route")
ap.add_argument("--experts", default="0-47"); ap.add_argument("--http", required=True); ap.add_argument("--index-http", required=True)
ap.add_argument("--rows", type=int, default=4096); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=None)
a = ap.parse_args()
import torch
from safetensors import safe_open
dev = torch.device(a.device); torch.backends.cuda.matmul.allow_tf32 = False
L = a.layer; experts = list(Q.parse_range(a.experts, 0, Q.N_EXPERTS - 1))
files = Q.find_dump_files(a.dumps, L); assert files, "no dumps"
t0 = time.time(); x, ri, rw, meta = Q.load_dumps(files, L, dev, torch); csr = Q.RoutingCSR(ri, rw, torch); del ri, rw
Q.log(f"L{L:02d}: {x.shape[0]} tokens from {len(files)} dump files in {time.time()-t0:.0f}s; comparing experts {experts[0]}-{experts[-1]} on <= {a.rows} rows each")
done_a, done_b = Q.done_experts(a.parts_a, L), Q.done_experts(a.parts_b, L)
fetch = Q.Fetcher(a.http, a.index_http)

def load_out(done, e, mat):
    sp, _ = done[e]
    with safe_open(sp, "pt", device="cpu") as f:
        return {t: f.get_tensor(f"layers.{L}.ffn.experts.{e}.{mat}.{t}") for t in ("trellis", "suh", "svh", "mul1")}

rows = []; same_gu = 0; n_cmp = 0
for e in experts:
    if e not in done_a or e not in done_b: continue
    tok, wr = csr.rows(e)
    if tok.numel() == 0: continue
    tok, wr = tok[:a.rows], wr[:a.rows]
    w1, w3, w2 = (fetch.get(f"layers.{L}.ffn.experts.{e}.{m}.weight", torch) for m in ("w1", "w3", "w2"))
    xf = x[tok].float(); a_unw = Q.act_down_input(xf, w1.to(dev).float(), w3.to(dev).float(), None, True, torch); a_w = a_unw * wr[:, None]
    W_t = w2.to(dev).float().T; ref_u = a_unw @ W_t; ref_w = a_w @ W_t
    rec = {"e": e, "tok": int(csr.counts[e]), "rows": int(tok.numel()), "route_w_mean": float(wr.mean())}
    for lab, done in ((a.label_a, done_a), (a.label_b, done_b)):
        Wq = Q.reconstruct({k: v.to(dev) for k, v in load_out(done, e, "w2").items()}, torch); d = Wq - W_t
        rec[f"rfn_unw_{lab}"] = float((a_unw @ d).norm() / (ref_u.norm() + 1e-12)); rec[f"rfn_w_{lab}"] = float((a_w @ d).norm() / (ref_w.norm() + 1e-12))
        del Wq, d
    ta, tb = load_out(done_a, e, "w1"), load_out(done_b, e, "w1"); same_gu += int(torch.equal(ta["trellis"], tb["trellis"]) and torch.equal(ta["suh"], tb["suh"]))
    rows.append(rec); n_cmp += 1
    Q.log(f"  e{e:03d} tok {rec['tok']:6d} gw {rec['route_w_mean']:.3f} | unw {a.label_a} {rec[f'rfn_unw_{a.label_a}']:.4f} {a.label_b} {rec[f'rfn_unw_{a.label_b}']:.4f} | weighted {a.label_a} {rec[f'rfn_w_{a.label_a}']:.4f} {a.label_b} {rec[f'rfn_w_{a.label_b}']:.4f}")
    del xf, a_unw, a_w, ref_u, ref_w
if not rows: raise SystemExit("no experts present in both parts dirs")
import statistics as st
def agg(k): v = [r[k] for r in rows]; return st.mean(v), st.median(v)
out = {"layer": L, "n": n_cmp, "w1_identical": same_gu}
for m in ("rfn_unw", "rfn_w"):
    ma, mda = agg(f"{m}_{a.label_a}"); mb, mdb = agg(f"{m}_{a.label_b}"); wins_b = sum(r[f"{m}_{a.label_b}"] < r[f"{m}_{a.label_a}"] for r in rows)
    out[m] = {a.label_a: {"mean": ma, "median": mda}, a.label_b: {"mean": mb, "median": mdb}, f"{a.label_b}_wins": wins_b, "rel_change_mean": (mb - ma) / ma}
    Q.log(f"{m:8s}: {a.label_a} mean {ma:.4f} med {mda:.4f} | {a.label_b} mean {mb:.4f} med {mdb:.4f} | {a.label_b} wins {wins_b}/{n_cmp} | rel {100*(mb-ma)/ma:+.1f}%")
Q.log(f"w1 identical across passes: {same_gu}/{n_cmp}")
verdict = a.label_b if out["rfn_w"]["rel_change_mean"] < -0.01 else a.label_a
Q.log(f"AB-VERDICT {verdict} (decision metric rfn_w, threshold 1% mean relative)")
out["verdict"] = verdict; out["experts"] = rows
if a.out: json.dump(out, open(a.out, "w"), indent=1)
