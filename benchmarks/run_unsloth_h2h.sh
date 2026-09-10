#!/usr/bin/env bash
# Pollard vs Unsloth Dynamic 3.0 on Qwen3.8-27B, size-matched.
# Protocol and the reasoning behind every choice: unsloth-qwen38-27b-headtohead.md
#
#   BIN=/path/to/llama.cpp/build/bin EVAL=/path/to/wiki.test.raw ./run_unsloth_h2h.sh
#
# This is a GPU job and it is long. Run it when the GPU is free (60/40 rule), not alongside a game.
set -euo pipefail

BIN=${BIN:?set BIN to the llama.cpp bin dir}
EVAL=${EVAL:?set EVAL to the wikitext-2 raw test file}
WORK=${WORK:-$PWD/h2h}
NGL=${NGL:-99}
HERE=$(cd "$(dirname "$0")" && pwd)

POLLARD_REPO=PollardWeights/Qwen3.8-27B-Pollard
UNSLOTH_REPO=unsloth/Qwen3.8-27B-GGUF

# The headline pair: 12.08 vs 12.04 GB, 0.3% apart. Size is the variable that must be controlled --
# a 10% larger quant should win, and calling that a method win is how this comparison gets faked.
POLLARD_F=Qwen3.8-27B-Pollard-IQ3_S.gguf
UNSLOTH_F=Qwen3.8-27B-UD-IQ3_S.gguf
# Near-lossless KL host. NOT BF16: that is 54.7 GB over two shards and will not fit beside both
# candidates on a ~75 GB volume. KLD vs a Q8_0 host is what the 14B/30B cards already use -- but it
# measures agreement-with-Q8_0, so say that when reporting.
REF_F=Qwen3.8-27B-Q8_0.gguf

mkdir -p "$WORK"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1

fetch () {
  python3 - "$1" "$2" "$WORK" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))
PY
}

echo "== fetching (about 53 GB)"
P=$(fetch "$POLLARD_REPO" "$POLLARD_F" | tail -1)
U=$(fetch "$UNSLOTH_REPO" "$UNSLOTH_F" | tail -1)
R=$(fetch "$UNSLOTH_REPO" "$REF_F"     | tail -1)

sz () { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
PB=$(sz "$P"); UB=$(sz "$U")
echo "== sizes"
printf '   pollard %s bytes\n   unsloth %s bytes\n' "$PB" "$UB"
python3 -c "p=$PB;u=$UB;print(f'   size gap: {(p-u)/u*100:+.2f}%  (a win under +2% is not a clean win)')"

# pollard-bench was built for exactly this: one harness, matched size, PPL + mean/median KLD + top-1,
# and a Pareto verdict across both files. Using it instead of hand-rolling keeps our number and the
# rival's number on identical code paths, which is the whole point of a head-to-head.
echo "== head-to-head board"
python3 "$HERE/../tools/pollard_bench.py" \
    --gguf "$P" --vs "$U" --ref "$R" \
    --eval "$EVAL" --ngl "$NGL" \
    --llama-perplexity "$BIN/llama-perplexity" --llama-cli "$BIN/llama-cli" \
    --out "$WORK/results.json"

echo "== coherence gate on both (a looping build is not competitive at any perplexity)"
for f in "$P" "$U"; do
  echo "--- $(basename "$f")"
  python3 "$HERE/../tools/pollard_bench.py" --gguf "$f" --coherence --quick --ngl "$NGL" \
      --llama-cli "$BIN/llama-cli" --llama-perplexity "$BIN/llama-perplexity" || true
done

echo
echo "== done -> $WORK/results.json (feeds pollard-scorecard)"
echo "   Report the size gap beside every number, and publish the losses too."
