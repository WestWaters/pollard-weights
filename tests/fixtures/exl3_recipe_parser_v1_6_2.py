"""ExLlamaV3 v1.6.2's recipe parser, vendored VERBATIM for tests (pollard-exl3 --recipe pollard must load in it).

Source: github.com/turboderp-org/exllamav3, tag v1.6.2 --
  exllamav3/conversion/allocation.py  (HALF_RATES, _as_rate, rate_floor, create_q_strategy_from_recipe)
  exllamav3/conversion/convert_model.py (the -rcp YAML checks, wrapped below as parse_recipe_args)
Only change: the function-local `from ..modules.linear import Linear` is replaced by a module-level `Linear`
the test supplies (exllamav3 itself cannot import without its CUDA extension).

MIT License -- Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""
from __future__ import annotations
import fnmatch
import math
import os
import re

Linear = None          # set by the test: the class whose instances count as quantizable Linear modules
Model = Config = object

# Half-integer trellis rates with kernel instances (mul1 codebook only); above 3.5 bpw the steps are whole bits
HALF_RATES = (1.5, 2.5, 3.5)


def _as_rate(r: float):
    return int(r) if float(r).is_integer() else float(r)


def rate_floor(bpw: float, half_steps: bool):
    """Largest supported bitrate <= bpw: integers 1..8, plus 1.5 / 2.5 / 3.5 when half_steps"""
    r = math.floor(bpw * 2) / 2 if half_steps and bpw < 4 else math.floor(bpw)
    return _as_rate(max(1, min(8, r)))


def rate_next(bpw: float, half_steps: bool):
    """The next supported bitrate above bpw (8 stays 8)"""
    return _as_rate(min(8, bpw + (0.5 if half_steps and bpw < 4 else 1)))


def parse_recipe_args(recipe_path, bits=None, codebook="mul1"):
    """convert_model.py v1.6.2's -rcp checks (lines 203-229 + 293-305), returning (recipe_tensors, recipe_bits,
    recipe_head_bits) or raising ValueError with the message convert.py would print."""
    import yaml
    if not os.path.isfile(recipe_path):
        raise ValueError(f"Recipe file not found: {recipe_path}")
    try:
        with open(recipe_path, "r", encoding = "utf8") as f:
            recipe = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Recipe is not valid YAML (keys starting with '*' must be quoted):\n{e}")
    recipe_tensors = recipe.get("tensors") if isinstance(recipe, dict) else None
    if not isinstance(recipe_tensors, dict) or not recipe_tensors:
        raise ValueError("Recipe must contain a non-empty 'tensors' mapping")
    # Integer bitrates 1-8 (or 16 = unquantized), plus the half-integer trellis rates (mul1 codebook)
    def ok(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)): return False
        return (float(v).is_integer() and (1 <= v <= 8 or v == 16)) or float(v) in HALF_RATES
    bad = [k for k, v in recipe_tensors.items() if not ok(v)]
    if bad:
        raise ValueError(f"Recipe bitrates must be integers 1-8, 16, or one of {HALF_RATES}; bad keys e.g.: {bad[:5]}")
    recipe_bits = recipe.get("achieved_bpw") or recipe.get("target_bpw")
    if bits is None and recipe_bits is None:
        raise ValueError("Recipe has no target_bpw/achieved_bpw; pass --bits for reporting")
    recipe_head_bits = recipe.get("head_bits")
    half_ok = codebook == "mul1"
    if recipe_tensors is not None and not half_ok and any(v in HALF_RATES for v in recipe_tensors.values()):
        raise ValueError(f"Recipe uses half-integer bitrates, which need the mul1 codebook")
    return recipe_tensors, recipe_bits, recipe_head_bits


def create_q_strategy_from_recipe(
    model: Model,
    mtp_model: Model,
    config: Config,
    recipe_tensors: dict,
    head_bpw: float,
    mtp_bpw: float,
    vision_model: Model = None,
    vision_bpw: int = None,
    hq: bool = False,
    half_steps: bool = False,
) -> (dict, float):
    """
    Build the per-module quantization strategy from an explicit per-tensor recipe (e.g. produced
    by util/sc_optimize.py) instead of the budgeted allocation in create_q_strategy. recipe_tensors
    maps budgeted ("bits") Linear keys to bitrates (1-8, 1.5 / 2.5 / 3.5 with the mul1 codebook, or 16 to store
    unquantized). A key is either an exact tensor key or an fnmatch-style pattern ("*.linear_attn.*"), and entries
    resolve top to bottom: the last entry matching a tensor sets its bitrate, so later entries override earlier ones.
    Every budgeted tensor must be covered, and every entry must match at least one budgeted tensor. The recipe
    doesn't cover the MTP head, which takes mtp_bpw with the same --hq promotion as in create_q_strategy.
    """
    pass  # vendored: module-level Linear

    exact = {}
    patterns = []
    for i, (k, v) in enumerate(recipe_tensors.items()):
        if any(c in k for c in "*?["):
            patterns.append((i, k, re.compile(fnmatch.translate(k)), v))
        else:
            exact[k] = (i, k, v)
    used = set()

    def lookup(key):
        best = exact.get(key)
        for p in reversed(patterns):
            if best is not None and p[0] < best[0]:
                break
            if p[2].match(key):
                best = (p[0], p[1], p[3])
                break
        if best is None:
            return None
        used.add(best[1])
        return best[2]

    targets = {}
    budgeted = []
    sum_numel = 0
    sum_bits = 0
    missing = []

    def _add(module, fixed_bpw = None):
        nonlocal sum_numel, sum_bits
        if isinstance(module, Linear) and module.qmap is not None:
            if fixed_bpw is not None:
                targets[module.key] = fixed_bpw
            elif module.qbits_key == "bits":
                budgeted.append(module.key)
                bpw = lookup(module.key)
                if bpw is None:
                    missing.append(module.key)
                else:
                    targets[module.key] = bpw
                    if bpw <= 8:
                        sum_numel += module.weights_numel()
                        sum_bits += module.weights_numel() * bpw
            elif module.qbits_key == "head_bits":
                targets[module.key] = head_bpw
            elif module.qbits_key == "mtp_bits":
                if hq and mtp_bpw <= 8:
                    hs = half_steps and getattr(module, "q_half_bits", True)
                    targets[module.key] = max(mtp_bpw, rate_floor(mtp_bpw + module.select_hq_bits, hs))
                else:
                    targets[module.key] = mtp_bpw
            else:
                raise ValueError("Logic error in create_q_strategy_from_recipe")
        for sm in module.modules:
            _add(sm, fixed_bpw)

    modules = model.modules
    if mtp_model:
        modules = modules + mtp_model.modules
    for m in modules:
        _add(m)
    if vision_model:
        for m in vision_model.modules:
            _add(m, fixed_bpw = vision_bpw)

    if missing:
        layer_idx = re.compile(r'(?<=\.)\d+(?=\.)')
        groups = sorted({layer_idx.sub("*", k) for k in missing})
        raise ValueError(
            f"Recipe is missing bitrates for {len(missing)} quantizable tensor(s), "
            f"e.g.: {', '.join(missing[:5])}; uncovered key groups: {', '.join(groups[:10])}"
        )
    # Entries that never set a bitrate: an exact key or pattern matching no budgeted tensor is an error (likely a
    # typo, or a head/MTP/vision tensor, which take --head_bits etc.), one fully overridden by later entries is not
    budgeted_set = set(budgeted)
    pattern_rx = {k: rx for _, k, rx, _ in patterns}
    unknown = [
        k for k in recipe_tensors if k not in used and (
            k not in budgeted_set if k in exact else not any(pattern_rx[k].match(b) for b in budgeted)
        )
    ]
    if unknown:
        raise ValueError(
            f"Recipe contains {len(unknown)} key(s) not matching any budgeted tensor in the "
            f"model, e.g.: {', '.join(unknown[:5])}"
        )

    return targets, float(sum_bits) / sum_numel if sum_numel else 0


