"""Exercise store-installed commands outside the checkout, without network/GPU."""
import json
from pathlib import Path
import struct
import subprocess

COMMANDS = ("pollard", "pollard-calc", "pollard-fit", "pollard-experts",
            "pollard-runtime", "pollard-serve-eval", "pollard-ggufcheck", "pollard-node",
            "pollard-ngram", "pollard-forge", "pollard-thinklean", "pollard-kvbench", "pollard-moefit")

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

# runtime-patches/ ships as package data: outside any checkout, the installed ggufcheck must still know clef-omni is
# an architecture only Pollard's runtime patch implements -- not "unknown"
arch, name = b"clef-omni", b"general.architecture"
with open("clef-omni.gguf", "wb") as fh:
    fh.write(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
    fh.write(struct.pack("<Q", len(name)) + name + struct.pack("<I", 8) + struct.pack("<Q", len(arch)) + arch)
    tensor = b"blk.0.weight"
    fh.write(struct.pack("<Q", len(tensor)) + tensor + struct.pack("<IQIQ", 1, 32, 12, 0))
result = subprocess.run(["pollard-ggufcheck", "--offline", "--json", "clef-omni.gguf"],
                        capture_output=True, text=True, timeout=30)
rows = json.loads(result.stdout)
assert rows and rows[0].get("runtime") == "pollard", result.stdout + result.stderr
assert rows[0]["reasons"].get("pollard_patch") == "llama.cpp-clef-omni", rows
print(f"{len(COMMANDS)} installed commands, offline calculator and installed runtime-patches passed")
