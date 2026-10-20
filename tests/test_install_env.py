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
@pytest.mark.parametrize("ik_exists", [False, True])
@pytest.mark.parametrize("gpu", ["-DGGML_METAL=ON", None])
def test_optional_installer_environment(tmp_path, stock_exists, skip_ik, ik_exists, gpu):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    script = checkout / "install.sh"
    shutil.copy2(ROOT / "install.sh", script)
    bins = tmp_path / "bin"
    bins.mkdir()
    bash = shutil.which("bash")
    assert bash, "the Unix installer requires bash"
    for name in ("mkdir", "dirname"):
        target = shutil.which(name)
        assert target, f"the installer requires {name}"
        (bins / name).symlink_to(target)
    log = tmp_path / "commands.log"
    inert = {
        "python3": "exit 0",
        "cmake": 'printf "%s\\n" "$*" >> "$POLLARD_TEST_LOG"',
        "git": 'for arg do target="$arg"; done; mkdir -p "$target"',
        "uname": 'printf "Linux\\n"',
    }
    if stock_exists:
        inert["llama-quantize"] = "exit 0"
    for name, body in inert.items():
        tool = bins / name
        tool.write_text(f"#!{bash}\n" + body + "\n")
        tool.chmod(0o755)
    if ik_exists:
        quantize = checkout / "ik" / "build" / "bin" / "llama-quantize"
        quantize.parent.mkdir(parents=True)
        quantize.write_text(f"#!{bash}\nexit 0\n")
        quantize.chmod(0o755)
    env = dict(os.environ)
    env.pop("POLLARD_NO_IK", None)
    env.pop("GPU_FLAGS", None)
    env.pop("POLLARD_GPU", None)
    env.update(PATH=str(bins),
               POLLARD_TEST_LOG=str(log), LLAMA_DIR=str(checkout / "llama"),
               IK_DIR=str(checkout / "ik"))
    if gpu is not None:
        env["POLLARD_GPU"] = gpu
    if skip_ik is not None:
        env["POLLARD_NO_IK"] = skip_ik
    result = subprocess.run([bash, str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "quickstart:" in result.stdout
    commands = log.read_text().splitlines() if log.exists() else []
    configure = [line for line in commands if line.startswith("-S ")]
    expected = int(not stock_exists) + int(skip_ik != "1" and not ik_exists)
    assert len(configure) == expected
    assert all("-DGGML_RPC=ON" in line for line in configure)
    if gpu:
        assert all(gpu in line for line in configure)
    warning = "no GPU toolkit detected"
    assert (warning in result.stdout) == (gpu is None and expected > 0)
