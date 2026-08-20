#!/usr/bin/env bash
# Pollard Weights — one-shot install: the tools + the llama.cpp runtime.
# After this runs you have: pollard-calc, pollard-fit, pollard-experts, and a
# built llama.cpp (llama-quantize, llama-cli, llama-server, ggml-rpc-server) on
# PATH for this checkout. The RPC backend is built in so a Pollard build can span
# multiple machines (pool their RAM) — see "across machines" below.
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
  # -DGGML_RPC=ON builds the RPC backend + ggml-rpc-server so one Pollard build can
  # span several machines (pipeline-parallel: each holds some layers, pooling
  # their RAM). This is how a GGUF too big for one box runs across many — the
  # llama.cpp answer to vLLM's clustering, no vLLM required.
  #
  # Each machine builds for ITS OWN accelerator; RPC then joins a MIXED cluster
  # (Apple Metal + NVIDIA CUDA + AMD ROCm + Intel SYCL + CPU boxes together).
  # Honor an explicit override, else auto-detect this machine's backend. Vulkan
  # is the cross-vendor catch-all — it runs on AMD/Intel/NVIDIA off the graphics
  # driver alone, no compute toolkit — so any GPU has a path.
  GPU_FLAGS=""
  if [ -n "${POLLARD_GPU:-}" ]; then
    GPU_FLAGS="$POLLARD_GPU"                    # override, e.g. POLLARD_GPU=-DGGML_VULKAN=ON
  elif [ "$(uname -s)" = "Darwin" ]; then
    GPU_FLAGS="-DGGML_METAL=ON"                 # Apple Silicon
  elif command -v nvcc >/dev/null 2>&1; then
    GPU_FLAGS="-DGGML_CUDA=ON"                  # NVIDIA: RTX, GB10 / DGX Spark
  elif command -v hipcc >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1; then
    GPU_FLAGS="-DGGML_HIP=ON"                   # AMD: Radeon / Instinct (ROCm)
  elif command -v icpx >/dev/null 2>&1; then
    GPU_FLAGS="-DGGML_SYCL=ON"                  # Intel: Arc / Data Center GPU (oneAPI)
  elif command -v vulkaninfo >/dev/null 2>&1; then
    GPU_FLAGS="-DGGML_VULKAN=ON"                # cross-vendor fallback (AMD/Intel/NVIDIA)
  else
    echo "note: no GPU toolkit detected — building CPU-only. Have a GPU? Install its"
    echo "      toolkit (CUDA / ROCm / oneAPI) or Vulkan drivers and re-run, or force it:"
    echo "      POLLARD_GPU=-DGGML_VULKAN=ON ./install.sh"
  fi
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_RPC=ON $GPU_FLAGS
  # NOTE: the RPC server target is `ggml-rpc-server` in current llama.cpp
  # (was `rpc-server` before it moved to tools/rpc/). Using the wrong name
  # silently builds nothing — verified against llama.cpp master.
  cmake --build "$LLAMA_DIR/build" -j \
    --target llama-quantize llama-cli llama-server llama-imatrix llama-perplexity ggml-rpc-server
  echo "runtime built at $LLAMA_DIR/build/bin ${GPU_FLAGS:+($GPU_FLAGS)}"
  echo "add to PATH:  export PATH=\"$LLAMA_DIR/build/bin:\$PATH\""
fi

echo
echo "quickstart:"
echo "  pollard-calc --model <hf-id> --ram 16          # what CAN this machine do"
echo "  pollard-fit  --gguf model-f16.gguf --ram 16    # build the Pollard Weights"
echo "  llama-cli    -m model-pollard.gguf             # run them"
echo
echo "across machines (pool RAM for a build bigger than one box):"
echo "  # on every OTHER machine:  ggml-rpc-server -H 0.0.0.0 -p 50052"
echo "  # on the main machine:     llama-cli -m model-pollard.gguf --rpc host2:50052,host3:50052"
