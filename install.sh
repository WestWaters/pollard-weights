#!/usr/bin/env bash
# Pollard Weights — one-shot install: the tools + the llama.cpp runtime.
# After this runs you have: pollard-calc, pollard-fit, and a built llama.cpp
# (llama-quantize, llama-cli, llama-server) on PATH for this checkout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HERE/runtime/llama.cpp}"

echo "== Pollard Weights install =="

# 1. the python tools
python3 -m pip install --user "$HERE" 2>/dev/null || python3 -m pip install --break-system-packages --user "$HERE"
echo "tools installed: pollard-calc, pollard-fit"

# 2. the runtime (llama.cpp — Pollard builds are normal GGUFs; this is the engine)
if command -v llama-quantize >/dev/null 2>&1; then
  echo "runtime found: $(command -v llama-quantize) (using existing llama.cpp)"
else
  mkdir -p "$HERE/runtime"
  if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
    $( [ "$(uname -s)" = "Darwin" ] && echo -DGGML_METAL=ON )
  cmake --build "$LLAMA_DIR/build" -j --target llama-quantize llama-cli llama-server llama-imatrix
  echo "runtime built at $LLAMA_DIR/build/bin"
  echo "add to PATH:  export PATH=\"$LLAMA_DIR/build/bin:\$PATH\""
fi

echo
echo "quickstart:"
echo "  pollard-calc --model <hf-id> --ram 16          # what CAN this machine do"
echo "  pollard-fit  --gguf model-f16.gguf --ram 16    # build the Pollard Weights"
echo "  llama-cli    -m model-pollard.gguf             # run them"
