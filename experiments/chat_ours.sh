#!/usr/bin/env bash
# Chat with OUR expert-only IQ3_XXS Laguna (12.40 GiB, coherence-verified).
# Needs:  sudo sysctl -w iogpu.wired_limit_mb=13824     (undo with =0)
# 8k context ~= 2.5 GB of KV, so start modest and raise if it holds.
exec ${LLAMA_DIR:-$HOME/llama.cpp}/build/bin/llama-cli \
  -m "${MODEL:?set MODEL to the path of an expert-only build}" \
  -ngl 99 -c "${CTX:-4096}" -b 256 -ub 256 --temp 0.6 -cnv
