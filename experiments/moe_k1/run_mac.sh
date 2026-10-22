#!/bin/bash
# run_mac.sh -- the small-MoE K1 / prune-then-quantize proof on a Mac (Metal), one heavy job at a time.
#
#   WORK=~/pollard_k1 HF=~/pollard_k1/hf PY=<python with numpy+gguf> TORCH_PY=<python with torch+transformers> \
#       experiments/moe_k1/run_mac.sh [--step all|arms|table] [extra moe_k1_arms.py flags]
#
# Takes the shared heavy-job lock (waits for it), releases it on exit, and refuses to start under the floors
# (10 GB disk, 2 GB RAM); moe_k1_arms.py re-checks the floors before every write.
set -u
LOCK=${LOCK:-/private/tmp/claude-501/mac-heavy.lock}
WORK=${WORK:-$HOME/pollard_k1}
HF=${HF:-$WORK/hf}
PY=${PY:-python3}
TORCH_PY=${TORCH_PY:-$PY}
HERE="$(cd "$(dirname "$0")" && pwd)"

until mkdir "$LOCK" 2>/dev/null; do sleep 20; done
echo "$$ k1 $(date)" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM

free_gb=$(df -g "$WORK" | awk 'NR==2{print $4}')
if [ "${free_gb:-0}" -lt 10 ]; then echo "STOP: ${free_gb} GB free < 10 GB floor"; exit 1; fi

"$PY" "$HERE/moe_k1_arms.py" --work "$WORK" --hf "$HF" --python "$PY" --torch-python "$TORCH_PY" "$@"
