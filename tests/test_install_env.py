"""Exercise the installer shell branches with inert tools, without installing or building."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(os.name == "nt", reason="Unix shell installer")
@pytest.mark.parametrize("stock_exists", [False, True])
@pytest.mark.parametrize("skip_ik", [None, "", "1"])
def test_optional_installer_environment(tmp_path, stock_exists, skip_ik):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "install.sh"
    shutil.copy2(ROOT / "install.sh", script)
    bins = tmp_path / "bin"
    bins.mkdir()
    log = tmp_path / "commands.log"
    inert = {
        "python3": "exit 0",
        "cmake": 'printf "%s\\n" "$*" >> "$POLLARD_TEST_LOG"',
        "git": 'for arg do target="$arg"; done; mkdir -p "$target"',
    }
    if stock_exists:
        inert["llama-quantize"] = "exit 0"
    for name, body in inert.items():
        tool = bins / name
        tool.write_text("#!/bin/sh\n" + body + "\n")
        tool.chmod(0o755)
    env = dict(os.environ)
    env.pop("POLLARD_NO_IK", None)
    env.pop("GPU_FLAGS", None)
    env.update(PATH=f"{bins}:/usr/bin:/bin", POLLARD_GPU="-DGGML_METAL=ON",
               POLLARD_TEST_LOG=str(log), LLAMA_DIR=str(checkout / "llama"),
               IK_DIR=str(checkout / "ik"))
    if skip_ik is not None:
        env["POLLARD_NO_IK"] = skip_ik
    result = subprocess.run(["/bin/bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "quickstart:" in result.stdout
    commands = log.read_text().splitlines() if log.exists() else []
    configure = [line for line in commands if line.startswith("-S ")]
    expected = int(not stock_exists) + int(skip_ik != "1")
    assert len(configure) == expected
    assert all("-DGGML_METAL=ON" in line and "-DGGML_RPC=ON" in line for line in configure)
