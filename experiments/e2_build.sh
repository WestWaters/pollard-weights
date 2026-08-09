#!/usr/bin/env bash
# Compile the E2 capture tool against an existing llama.cpp build.
#
# Deliberately NOT added to llama.cpp's own CMake tree: this experiment lives here, and dropping a
# target into the llama.cpp checkout would make a vendored clone dirty and lose the tool on the next
# upstream pull.
#
# Links the raw llama API only (no libcommon) so an upstream refactor of the example helpers can't
# break it.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_DIR/build}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -d "$BUILD_DIR/bin" ] || { echo "no llama.cpp build at $BUILD_DIR — run: cmake -B build -DGGML_METAL=ON && cmake --build build -j"; exit 1; }

# platform-aware: Metal frameworks on macOS, plain link on Linux/CUDA boxes
if [ "$(uname -s)" = "Darwin" ]; then
  EXTRA_LIBS="-framework Metal -framework Foundation -framework MetalKit"
else
  EXTRA_LIBS=""
fi
CXX="${CXX:-c++}"
$CXX -std=c++17 -O2 \
  -I"$LLAMA_DIR/include" \
  -I"$LLAMA_DIR/ggml/include" \
  "$HERE/e2_capture_routing.cpp" \
  -L"$BUILD_DIR/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$BUILD_DIR/bin" \
  -o "$HERE/e2_capture_routing"

echo "built: $HERE/e2_capture_routing"
