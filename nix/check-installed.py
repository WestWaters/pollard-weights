"""Exercise store-installed commands outside the checkout, without network/GPU."""
import json
from pathlib import Path
import subprocess

COMMANDS = ("pollard", "pollard-calc", "pollard-fit", "pollard-experts",
            "pollard-runtime", "pollard-serve-eval", "pollard-ggufcheck", "pollard-node",
            "pollard-ngram", "pollard-forge", "pollard-thinklean", "pollard-kvbench")

for command in COMMANDS:
    result = subprocess.run([command, "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"{command}: {result.stderr}"
    assert "usage:" in result.stdout.lower(), f"{command}: no CLI help"

Path("config.json").write_text(json.dumps({
    "model_type": "llama", "hidden_size": 1024, "num_hidden_layers": 4,
    "num_attention_heads": 8, "num_key_value_heads": 8,
    "intermediate_size": 4096, "vocab_size": 32000,
}))
result = subprocess.run(["pollard-calc", "--config", "config.json", "--ram", "32", "--ctx", "4096"],
                        capture_output=True, text=True, timeout=30)
assert result.returncode == 0, result.stderr
assert "32 GB RAM" in result.stdout and "KV" in result.stdout, result.stdout
print(f"{len(COMMANDS)} installed commands and offline calculator passed")
