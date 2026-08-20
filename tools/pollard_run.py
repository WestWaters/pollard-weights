#!/usr/bin/env python3
"""pollard-run — measured expert placement for llama.cpp (RAM-streaming runtime).

The RAM-streaming design (all experts resident in system RAM, active experts
streamed to the GPU per token) is how big MoEs run on GPU boxes today —
llama.cpp ships the machinery as `--cpu-moe` (all experts) and `--n-cpu-moe N`
(the FIRST N layers, blind). pollard-run replaces the blind split with a
measured one: it reads your routing profile (experiments/e2), ranks layers by
observed reuse, fills your VRAM budget with the layers that earn it, and emits
the llama-server command with per-layer `--override-tensor` placement plus a
placement report.

Usage:
  pollard-run --gguf model.gguf --profile heat_profile.json --vram 24
  pollard-run --gguf model.gguf --profile heat_profile.json --vram 24 --launch
  pollard-run --gguf model.gguf --vram 24            # no profile: depth-order fallback

The output command runs stock llama-server — nothing here forks the runtime.
"""
import argparse
import json
import os
import subprocess
import sys

from pollard_calc import read_gguf_meta, gguf_to_config, analyse, find_llama_bin


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--profile", help="heat profile json from experiments/e2_analyse_routing.py, "
                                      "or a name from profiles/ (e.g. 'qwen3-30b-a3b')")
    ap.add_argument("--vram", required=True,
                    help="GPU memory budget in GB for weights (leave headroom for "
                         "KV/activations), or 'auto' to read free VRAM from nvidia-smi")
    ap.add_argument("--llama-server", default="llama-server")
    ap.add_argument("--rpc", help="RPC servers to pool, 'host:port[,host:port…]' (run "
                                  "ggml-rpc-server on each peer) — run a model too big for "
                                  "one box across several")
    ap.add_argument("--launch", action="store_true", help="exec the command instead of printing it")
    ap.add_argument("--extra", default="", help="extra llama-server args appended verbatim")
    a = ap.parse_args()

    if str(a.vram).lower() == "auto":
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            free_mb = total_mb = 0
            for line in out.strip().splitlines():
                nums = [int(x) for x in line.replace(",", " ").split() if x.strip().isdigit()]
                if len(nums) >= 2:
                    free_mb += nums[0]; total_mb += nums[1]
            if total_mb == 0:
                raise ValueError("nvidia-smi reported no GPU memory")
            # free VRAM reads ~0 when a model is already resident — but placement PLANS
            # for when pollard's own model loads (that resident copy is gone by then), so
            # fall back to TOTAL instead of erroring on a 0 budget (Frank's 09 failure).
            if free_mb < total_mb * 0.10:
                a.vram = total_mb / 1024 * 0.85
                print(f"[--vram auto] only {free_mb/1024:.1f} GB free of {total_mb/1024:.1f} "
                      f"GB — GPU is occupied; planning against TOTAL VRAM -> budget "
                      f"{a.vram:.1f} GB (15% reserved). Pass --vram N to pin it.")
            else:
                a.vram = free_mb / 1024 * 0.85          # keep 15% for KV/compute
                print(f"[--vram auto] nvidia-smi free VRAM: {free_mb/1024:.1f} GB "
                      f"-> budget {a.vram:.1f} GB (15% reserved)")
        except Exception:
            sys.exit("ERROR: --vram auto needs nvidia-smi (NVIDIA GPUs). On Apple "
                     "Silicon pass an explicit budget; the Metal wired ceiling is "
                     "roughly 75-80% of total RAM minus what's already resident.")
    else:
        a.vram = float(a.vram)
    meta = read_gguf_meta(a.gguf)
    cfg = gguf_to_config(meta, a.gguf)
    arch = analyse(cfg)
    if arch["kind"] != "moe":
        sys.exit("ERROR: not a MoE GGUF — measured expert placement needs experts. "
                 "For dense models just set -ngl to what fits.")

    layers = arch["layers"]
    file_gb = cfg["_gguf_file_bytes"] / 1e9
    bpw = cfg["_gguf_file_bytes"] * 8.0 / arch["total"]
    expert_gb_layer = arch["expert_params"] * arch["n_experts"] * bpw / 8 / 1e9
    other_gb = file_gb - expert_gb_layer * layers          # attn/embed/norms/etc

    heat = None
    if a.profile:
        ppath = a.profile
        if not os.path.exists(ppath):
            zoo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "profiles", a.profile + ".json")
            if os.path.exists(zoo):
                ppath = zoo
                print(f"[profile zoo] using {ppath}")
            else:
                sys.exit(f"ERROR: profile '{a.profile}' not found as a file or in profiles/")
        prof = json.load(open(ppath))
        rbl = prof.get("reuse_by_layer")
        if isinstance(rbl, list) and len(rbl) == layers:
            heat = [float(x) for x in rbl]
        elif isinstance(rbl, dict):
            heat = [float(rbl.get(str(i), 0.0)) for i in range(layers)]
    ranked = (sorted(range(layers), key=lambda i: -heat[i]) if heat
              else list(range(layers)))                     # fallback: shallow first

    budget = a.vram - other_gb                              # non-expert weights sit on GPU
    if budget < 0:
        sys.exit(f"ERROR: non-expert weights alone ({other_gb:.1f}GB) exceed the "
                 f"{a.vram:.0f}GB VRAM budget — lower the quant or raise --vram.")
    gpu_layers, used = [], 0.0
    for i in ranked:
        if used + expert_gb_layer <= budget:
            gpu_layers.append(i)
            used += expert_gb_layer
    cpu_layers = sorted(set(range(layers)) - set(gpu_layers))

    print(f"== pollard-run :: {os.path.basename(a.gguf)}")
    print(f"model               : {arch['total']/1e9:.1f}B MoE, {layers} layers, "
          f"{arch['n_experts']} experts/layer, {file_gb:.1f} GB @ {bpw:.2f} bpw")
    print(f"VRAM budget         : {a.vram:.0f} GB -> {other_gb:.1f} GB non-expert + "
          f"{used:.1f} GB hot experts")
    print(f"placement           : experts on GPU for {len(gpu_layers)} layers, "
          f"streamed from RAM for {len(cpu_layers)} layers")
    print(f"ranking signal      : "
          + ("measured per-layer reuse (profile)" if heat else
             "NONE — depth order fallback. Capture a profile with experiments/e2 "
             "for a measured split."))
    if cpu_layers:
        est = len(cpu_layers) / layers
        print(f"per-token stream    : ~{arch['top_k']+arch['shared']} experts on "
              f"{len(cpu_layers)} layers from system RAM ({est:.0%} of expert traffic)")

    cmd = [a.llama_server, "-m", a.gguf, "-ngl", "999"]
    if a.rpc:                                           # pool peers for a model too big
        cmd += ["--rpc", a.rpc]                         # for one box (run ggml-rpc-server there)
    if cpu_layers:
        pats = ",".join(f"blk\\.{i}\\.ffn_.*_exps\\.weight=CPU" for i in cpu_layers)
        # RAM-streaming means experts RESIDENT in RAM, not paged from disk —
        # llama.cpp itself recommends --no-mmap when overriding tensors to CPU.
        cmd += ["-ot", pats, "--no-mmap"]
    if a.extra:
        cmd += a.extra.split()
    print()
    if not a.launch:
        print("command:")
        print("  " + " ".join(cmd))
        return
    resolved = find_llama_bin(cmd[0])
    if resolved is None:
        sys.exit(f"ERROR: {cmd[0]} not found. install.sh builds it into "
                 f"runtime/llama.cpp/build/bin — re-run install.sh, or pass --llama-server.")
    cmd[0] = resolved
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
