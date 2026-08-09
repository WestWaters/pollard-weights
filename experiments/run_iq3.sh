#!/usr/bin/env bash
# Expert-only IQ3 quantisation, mirroring poolside's own NVFP4 scheme for Laguna-S:
# quantise ONLY experts.*.{gate,up,down}; leave attention, the router (ffn_gate_inp), the shared
# expert, layer-0's dense MLP and the output head alone. Experts are ~98% of the bytes, so this
# keeps nearly all the savings while sparing the parts that steer the model — which blanket Q2_K
# crushed, producing empty output.
set -euo pipefail
Q=${LLAMA_DIR:-$HOME/llama.cpp}/build/bin/llama-quantize
M="$(cd "$(dirname "$0")/.." && pwd)/models"
"$Q" --allow-requantize \
     --imatrix "$(dirname "$0")/imatrix_laguna_xs.dat" \
     --tensor-type ffn_down_exps=iq3_xxs \
     --tensor-type ffn_gate_exps=iq3_xxs \
     --tensor-type ffn_up_exps=iq3_xxs \
     "$M/Laguna-XS-2.1-Q4_K_M.gguf" "$M/Laguna-XS-2.1-IQ3XXS-experts.gguf" Q4_K_M 6
