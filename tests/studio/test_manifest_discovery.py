"""Studio reads the same tool declarations in a checkout and a flat pip install."""
import pytest

from pollard_studio import runner, validate


@pytest.fixture(params=["checkout", "installed", "tools_override", "repo_override"])
def tool_layout(request, tmp_path, monkeypatch):
    monkeypatch.delenv("POLLARD_TOOLS", raising=False)
    monkeypatch.delenv("POLLARD_REPO", raising=False)
    layout = request.param
    package_parent = tmp_path / ("tools" if layout == "checkout" else "site-packages")
    tools = package_parent
    if layout == "tools_override":
        tools = tmp_path / "custom-tools"
        monkeypatch.setenv("POLLARD_TOOLS", str(tools))
    elif layout == "repo_override":
        tools = tmp_path / "custom-repo" / "tools"
        monkeypatch.setenv("POLLARD_REPO", str(tools.parent))
    tools.mkdir(parents=True)
    if layout == "checkout":
        (tmp_path / "pyproject.toml").write_text("")
    (tools / "pollard_bench.py").write_text(
        '"""A small tool for manifest discovery."""\n'
        'import argparse\n'
        'parser = argparse.ArgumentParser()\n'
        'parser.add_argument("--rows", type=int)\n'
    )
    monkeypatch.setattr(runner, "__file__", str(package_parent / "pollard_studio" / "runner.py"))
    monkeypatch.setattr(runner, "REPO", runner.repo_root())


def test_app_discovers_tools_in_supported_layouts(tool_layout):
    pytest.importorskip("webview")
    from pollard_studio import app

    tools = app.load_manifest()
    assert [tool["module"] for tool in tools] == ["pollard_bench"]
    assert tools[0]["flags"][0]["names"] == ["--rows"]


def test_validator_discovers_tools_in_supported_layouts(tool_layout):
    assert validate.validate("pollard_bench", {"--rows": "12"}) == ["--rows", "12"]
    with pytest.raises(validate.Invalid, match="expects int"):
        validate.validate("pollard_bench", {"--rows": "invalid"})


@pytest.mark.parametrize("loader", ["app", "validate"])
def test_repo_override_matches_execution_when_installed_tools_also_exist(loader, tmp_path, monkeypatch):
    monkeypatch.delenv("POLLARD_TOOLS", raising=False)
    installed = tmp_path / "site-packages"
    override = tmp_path / "checkout" / "tools"
    for directory, flag in [(installed, "--installed-only"), (override, "--override-only")]:
        directory.mkdir(parents=True)
        (directory / "pollard_bench.py").write_text(
            'import argparse\nparser = argparse.ArgumentParser()\n'
            f'parser.add_argument("{flag}", type=int)\n'
        )
    monkeypatch.setattr(runner, "__file__", str(installed / "pollard_studio" / "runner.py"))
    monkeypatch.setenv("POLLARD_REPO", str(override.parent))
    monkeypatch.setattr(runner, "REPO", runner.repo_root())
    monkeypatch.setattr(runner.shutil, "which", lambda command: None)
    assert runner.resolve("pollard-bench")[-1] == str(override / "pollard_bench.py")
    if loader == "app":
        pytest.importorskip("webview")
        from pollard_studio import app
        flags = app.load_manifest()[0]["flags"]
        assert [f["names"] for f in flags] == [["--override-only"]]
    else:
        assert validate.validate("pollard_bench", {"--override-only": "12"}) == ["--override-only", "12"]
