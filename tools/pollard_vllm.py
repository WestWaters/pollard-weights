#!/usr/bin/env python3
"""pollard-vllm — will this build serve under vLLM, and at which tensor-parallel sizes?

A quantized model that loads fine on one GPU can fail to load at TP=4 for reasons that have nothing
to do with quality: vLLM shards tensors across ranks, so head counts and the intermediate dimension
have to divide by the TP degree, and for group-quantized weights each shard must still contain whole
groups. The failure arrives minutes into a load, as a shape error, on someone else's hardware.

This answers it before the download:

  pollard-vllm --model ./MyModel-Pollard-GPTQ            # which TP degrees work, and why
  pollard-vllm --model <hf-id> --tp 4                    # check one degree, exit 1 if it cannot
  pollard-vllm --model ./out --tp 4 --serve              # print the vllm serve command to run

It reads `config.json` and any quantization config; it does not load weights, so it works on a model
directory, a local shard set, or a bare HF id with only the config fetched. What it cannot tell you
is whether the machine has enough memory -- that is `pollard-calc`.

Lanes: GPTQ and MX builds serve under vLLM directly. GGUF is llama.cpp's format -- vLLM has only
partial GGUF support and llama.cpp splits across GPUs at RUNTIME (`-ts`, `--split-mode`), so a GGUF
build needs no TP-specific packaging at all. That distinction is worth knowing before anyone rebuilds
a model "for TP=4" that never needed it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

def tp_ceiling(cfg: dict) -> int:
    """The largest TP degree this model could ever use.

    Not an arbitrary cap: a rank has to receive at least one attention head, so the model's own head
    count is the hard ceiling. Everything below it is reported, because vLLM takes any positive
    degree and people run 3, 6 and 10 GPUs, not only powers of two.
    """
    text = cfg.get("text_config", cfg)
    return int(text.get("num_attention_heads") or 64)


def _load_config(model: str) -> dict:
    """Read config.json from a directory, or fetch it from the Hub for a bare id."""
    local = os.path.join(model, "config.json")
    if os.path.isfile(local):
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    if os.path.isdir(model):
        raise SystemExit(f"no config.json in {model} — is this a model directory?")
    url = f"https://huggingface.co/{model}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"could not read config for {model!r} ({e.code}). "
                         "Pass a local directory, or check the id.") from None
    except Exception as e:                        # offline, DNS, proxy
        raise SystemExit(f"could not reach the Hub for {model!r} ({e}). "
                         "Pass a local model directory instead.") from None


def _quant_group(cfg: dict) -> tuple[str | None, int | None]:
    """The quantization method and its group size, if the config declares one."""
    q = cfg.get("quantization_config") or {}
    method = q.get("quant_method") or q.get("method")
    group = q.get("group_size")
    if group in (-1, None) and method:            # -1 means per-channel: no group constraint
        group = None
    return method, group


def check(cfg: dict, tp: int) -> list[str]:
    """Reasons this model cannot shard at this TP degree. Empty list means it can."""
    text = cfg.get("text_config", cfg)
    bad: list[str] = []
    heads = text.get("num_attention_heads")
    kv = text.get("num_key_value_heads", heads)
    inter = text.get("intermediate_size")
    hidden = text.get("hidden_size")
    method, group = _quant_group(cfg)

    if heads and heads % tp:
        bad.append(f"attention heads {heads} not divisible by {tp}")
    # vLLM can replicate KV heads when there are fewer than ranks, but only by a whole factor
    if kv and kv % tp and tp % kv:
        bad.append(f"KV heads {kv} neither divisible by nor a divisor of {tp}")
    if inter and inter % tp:
        bad.append(f"intermediate size {inter} not divisible by {tp}")
    if hidden and hidden % tp:
        bad.append(f"hidden size {hidden} not divisible by {tp}")
    # group-quantized weights: each rank's shard must still hold whole groups
    if group and inter and (inter // max(tp, 1)) % group:
        bad.append(f"{method} group size {group} does not divide the per-rank "
                   f"intermediate slice {inter // tp}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--model", required=True, help="model directory or HF id")
    ap.add_argument("--tp", type=int, help="check one tensor-parallel degree (exit 1 if it fails)")
    ap.add_argument("--serve", action="store_true", help="print the vllm serve command")
    ap.add_argument("--max-len", type=int, default=0, help="--max-model-len for the serve command")
    a = ap.parse_args()

    cfg = _load_config(a.model)
    text = cfg.get("text_config", cfg)
    method, group = _quant_group(cfg)
    arch = (cfg.get("architectures") or ["?"])[0]

    print(f"\n  {a.model}")
    print(f"    architecture      {arch}")
    print(f"    hidden / inter    {text.get('hidden_size','?')} / {text.get('intermediate_size','?')}")
    print(f"    heads / kv-heads  {text.get('num_attention_heads','?')} / "
          f"{text.get('num_key_value_heads', text.get('num_attention_heads','?'))}")
    print(f"    quantization      {method or 'none (fp16/bf16)'}"
          f"{f', group {group}' if group else ''}")

    if method == "gguf" or a.model.endswith(".gguf"):
        print("\n    GGUF is llama.cpp's format. llama.cpp splits across GPUs at RUNTIME "
              "(`-ts 1,1,1,1`),\n    so there is no TP-specific build to make — the same file "
              "serves any number of GPUs.")
        return

    if a.tp:
        bad = check(cfg, a.tp)
        if bad:
            print(f"\n    TP={a.tp}: NO")
            for b in bad:
                print(f"      - {b}")
            sys.exit(1)
        print(f"\n    TP={a.tp}: yes")
    else:
        top = tp_ceiling(cfg)
        ok = [tp for tp in range(1, top + 1) if not check(cfg, tp)]
        print(f"\n    works at TP = {', '.join(map(str, ok)) if ok else 'nothing above 1'}")
        print(f"    (every degree up to {top}, this model's head count — vLLM takes any positive "
              "degree, not only powers of two)")
        shown = 0
        for tp in range(2, top + 1):
            bad = check(cfg, tp)
            if bad and shown < 4:            # say WHY for the first few that fail
                print(f"      TP={tp}: no — {bad[0]}")
                shown += 1

    if a.serve:
        # Default to one GPU: the largest degree that *divides* is not the degree anyone owns.
        tp = a.tp or 1
        cmd = ["vllm serve", a.model, f"--tensor-parallel-size {tp}"]
        if method in ("gptq", "gptq_marlin", "compressed-tensors", "fp8"):
            cmd.append(f"--quantization {method}")
        if a.max_len:
            cmd.append(f"--max-model-len {a.max_len}")
        print("\n   ", " \\\n      ".join(cmd))
        print("\n    then A/B it against the original with `pollard-serve-eval`.")


if __name__ == "__main__":
    main()
