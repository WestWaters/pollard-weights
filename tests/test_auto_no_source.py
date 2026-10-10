"""`pollard` with neither --gguf nor --hf exits with a usage message, not a NameError (Joey's review)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_source_is_a_clean_usage_error():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "pollard_auto.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "NameError" not in r.stderr and "Traceback" not in r.stderr
    assert "--gguf" in r.stderr
