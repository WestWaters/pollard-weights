#!/usr/bin/env bash
# Wait for the imatrix, then: quantise experts-only -> bench -> coherence check.
# Coherence is checked against the SAME prompt the Q4 control answered, so the comparison is direct.
set -uo pipefail
cd "$(dirname "$0")"
B=${LLAMA_DIR:-$HOME/llama.cpp}/build/bin
M="$(cd "$(dirname "$0")/.." && pwd)/models"
OUT=$M/Laguna-XS-2.1-IQ3XXS-experts.gguf

while pgrep -q llama-imatrix; do sleep 30; done
[ -f imatrix_laguna_xs.dat ] || { echo "FAIL: no imatrix produced"; exit 1; }
echo "=== imatrix: $(ls -lh imatrix_laguna_xs.dat | awk '{print $5}') ==="

echo "=== quantising (experts only -> IQ3_XXS) ==="
./run_iq3.sh 2>&1 | tail -4 || { echo "FAIL: quantise"; exit 1; }
ls -lh "$OUT"

echo "=== bench (Metal) ==="
"$B/llama-bench" -m "$OUT" -ngl 99 -p 64 -n 64 -r 2 2>&1 | grep -E "^\||OutOfMemory|error" | head -8

echo "=== coherence (same prompt as the Q4 control) ==="
"$B/llama-cli" -m "$OUT" -ngl 99 -c 512 -b 64 -ub 64 -n 48 -st --no-warmup \
    -p "def fib(n):" --temp 0.1 </dev/null 2>&1 | sed -n '/def fib/,$p' | head -20
