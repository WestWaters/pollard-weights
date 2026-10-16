"""CUDA arch selection for engine updates (pollard-runtime --update).

The update carried CMakeCache's CMAKE_CUDA_ARCHITECTURES over verbatim. That held while the box never
changed, and breaks the day it does:
  * Rubin (compute capability 10.7, sm_107) needs nvcc >= 13.4; an older toolkit fails an hour into the
    build on 'compute_107'. It must be refused up front with the fix in the message.
  * CUDA 13 dropped sm_60/61/70 (everything below 7.5). ik_llama's non-native default list
    "60;61;70;75;80" -- or a list carried from a CUDA 12 build -- fails to configure on CUDA 13.
  * A list written for the old GPU carries no code for a new one.
The RTX 5070 Ti build box (cc 12.0, cache = native) must keep building exactly as before.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_runtime_update as U

NEW_CMAKE, OLD_CMAKE = (4, 4, 2), (3, 28, 3)


@pytest.fixture(autouse=True)
def _isolated_caps(tmp_path_factory, monkeypatch):
    """Never read or write the real runtime/cuda_caps.json; no POLLARD_CUDA_ARCH from the caller."""
    monkeypatch.setattr(U, "CUDA_CAPS_FILE", str(tmp_path_factory.mktemp("caps") / "cuda_caps.json"))
    monkeypatch.delenv("POLLARD_CUDA_ARCH", raising=False)


def test_parsers_read_real_tool_output():
    smi = "10.7\n10.7\n"                              # one line per GPU
    assert U.parse_compute_caps(smi) == ["10.7"]
    assert U.parse_compute_caps("12.0\n8.6\n") == ["8.6", "12.0"]
    assert U.parse_compute_caps("12.0\r\n12.0\r\n") == ["12.0"]          # Windows line ends
    assert U.parse_compute_caps("") == []
    assert U.parse_compute_caps("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.") == []
    nvcc = ("nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.8, V12.8.93\n"
            "Build cuda_12.8.r12.8/compiler.35583870_0\n")
    assert U.parse_nvcc_version(nvcc) == (12, 8)
    assert U.parse_nvcc_version("Cuda compilation tools, release 13.4, V13.4.48") == (13, 4)
    assert U.parse_nvcc_version("") is None
    assert U.parse_cmake_version("cmake version 4.4.2\n\nCMake suite maintained") == (4, 4, 2)
    assert U.parse_cmake_version("cmake version 3.31.8") == (3, 31, 8)


@pytest.mark.parametrize("ver,ok", [((3, 28, 3), False), ((3, 31, 7), False), ((3, 31, 8), True),
                                    ((4, 0, 0), False), ((4, 0, 1), False), ((4, 0, 2), True),
                                    ((4, 4, 2), True), (None, False)])
def test_f_suffix_needs_a_fixed_cmake(ver, ok):
    """`100f-virtual` is rejected by CMake's arch validator before 3.31.8 / 4.0.2."""
    assert U.cmake_parses_f_suffix(ver) is ok


def test_rubin_gets_sm107_plus_blackwell_family_ptx():
    v, note = U.select_cuda_arch(None, ["10.7"], (13, 4), NEW_CMAKE)
    assert v == "107-real;100f-virtual" and note


def test_rubin_on_an_old_cmake_avoids_the_f_suffix():
    v, _ = U.select_cuda_arch(None, ["10.7"], (13, 4), OLD_CMAKE)
    assert v == "107a-real;90-virtual"


@pytest.mark.parametrize("nvcc", [(13, 3), (12, 8), None])
def test_rubin_refuses_a_toolkit_that_cannot_target_it(nvcc):
    with pytest.raises(U.CudaArchError, match=r"13\.4"):
        U.select_cuda_arch("native", ["10.7"], nvcc, NEW_CMAKE)


def test_rubin_replaces_native_with_the_explicit_list():
    v, _ = U.select_cuda_arch("native", ["10.7"], (13, 4), NEW_CMAKE)
    assert v == "107-real;100f-virtual"


@pytest.mark.parametrize("nvcc", [(12, 8), (13, 3)])
def test_build_box_cc12_native_is_unchanged(nvcc):
    """The 5070 Ti box's cache says `native`; that keeps building exactly as before."""
    assert U.select_cuda_arch("native", ["12.0"], nvcc, NEW_CMAKE) == ("native", "")


def test_explicit_list_that_covers_the_gpu_is_kept():
    assert U.select_cuda_arch("120a-real", ["12.0"], (13, 3), NEW_CMAKE) == ("120a-real", "")


def test_cuda13_drops_pre_turing_archs_from_a_carried_list():
    """ik_llama's non-native default, carried into a CUDA 13 build: 60/61/70 cannot compile."""
    v, note = U.select_cuda_arch("60;61;70;75;80;120a-real", ["12.0"], (13, 3), NEW_CMAKE)
    assert v == "75;80;120a-real"
    assert "60" in note and "CUDA 13" in note
    for bad in ("60", "61", "70"):
        assert bad not in v.split(";")


def test_cuda12_keeps_pre_turing_archs():
    assert U.select_cuda_arch("60;61;70;75;80;120a-real", ["12.0"], (12, 8), NEW_CMAKE)[0] == "60;61;70;75;80;120a-real"


def test_stale_list_for_another_gpu_is_replaced():
    """Cache written on an 8.6 card, box now has a 12.0 card: the old list has no code for it."""
    v, note = U.select_cuda_arch("86-real", ["12.0"], (13, 3), NEW_CMAKE)
    assert v == "120a-real" and "12.0" in note


def test_virtual_only_entry_is_not_coverage():
    v, _ = U.select_cuda_arch("120a-virtual", ["12.0"], (13, 3), NEW_CMAKE)
    assert v == "120a-real"


def test_multi_gpu_box_gets_every_capability():
    v, _ = U.select_cuda_arch(None, U.parse_compute_caps("12.0\n10.7\n12.0\n"), (13, 4), NEW_CMAKE)
    assert v.split(";") == ["107-real", "100f-virtual", "120a-real"]


def test_no_gpu_on_cuda13_gets_a_portable_list_without_pre_turing():
    v, _ = U.select_cuda_arch(None, [], (13, 3), NEW_CMAKE)
    assert v == U.CUDA13_PORTABLE
    assert all(U._entry_num(e) >= 75 for e in v.split(";"))


def test_no_gpu_no_cuda13_leaves_the_engine_default():
    assert U.select_cuda_arch(None, [], (12, 8), NEW_CMAKE) == (None, "")


def test_override_wins_over_detection_and_the_rubin_check():
    assert U.select_cuda_arch("native", ["10.7"], (12, 8), OLD_CMAKE, override="90-real")[0] == "90-real"


def _tree(tmp_path, cache_lines):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "CMakeCache.txt").write_text("\n".join(cache_lines) + "\n")
    return str(tmp_path)


def test_cache_flags_replaces_ik_default_on_a_cuda13_box(tmp_path):
    """ik_llama's non-native list in the cache, CUDA 13.3 + an RTX 5070 Ti: neither 60/61/70 (cannot
    compile) nor 75/80 alone (no sm_120 code) is right -- the GPU's own arch is."""
    tree = _tree(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=60;61;70;75;80"])
    f = U.cache_flags(tree, detect=lambda: (["12.0"], (13, 3), NEW_CMAKE))
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "120a-real"


def test_cache_flags_keeps_the_build_box_native(tmp_path):
    tree = _tree(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native",
                            "GGML_NATIVE:BOOL=ON"])
    f = U.cache_flags(tree, detect=lambda: (["12.0"], (12, 8), NEW_CMAKE))
    assert f == {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "native", "GGML_NATIVE": "ON", "LLAMA_CURL": "OFF"}


def test_cache_flags_raises_for_rubin_on_old_nvcc(tmp_path):
    tree = _tree(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native"])
    with pytest.raises(U.CudaArchError):
        U.cache_flags(tree, detect=lambda: (["10.7"], (13, 3), NEW_CMAKE))


def test_cache_flags_override_skips_detection(tmp_path):
    tree = _tree(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native"])

    def boom():
        raise AssertionError("detection must not run when --cuda-arch is given")
    f = U.cache_flags(tree, cuda_arch="107a-real;90-virtual", detect=boom)
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "107a-real;90-virtual"


def test_cache_flags_never_probes_cuda_on_a_metal_build(tmp_path):
    tree = _tree(tmp_path, ["GGML_METAL:BOOL=ON"])

    def boom():
        raise AssertionError("no CUDA probe on a non-CUDA build")
    assert U.cache_flags(tree, detect=boom) == {"GGML_METAL": "ON", "LLAMA_CURL": "OFF"}


def test_update_refuses_before_cloning(monkeypatch, tmp_path):
    """The Rubin/toolkit check must stop the update before an upstream clone, not after a build."""
    tree = _tree(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native"])
    monkeypatch.setitem(U.ENGINES, "fake", {"dir": tree, "url": "https://example.invalid/x"})
    monkeypatch.setattr(U, "upstream_head", lambda url, ref=None: ("abc123456", None))
    monkeypatch.setattr(U, "detect_cuda", lambda: (["10.7"], (13, 3), NEW_CMAKE))
    cloned = []
    monkeypatch.setattr(U, "_run", lambda *a, **k: cloned.append(a) or (_ for _ in ()).throw(AssertionError(a)))
    assert U.update_engine("fake") is False
    assert not cloned


# ---- GPU hidden (CUDA_VISIBLE_DEVICES=-1): `native` would build for no GPU ----------------------------------
# The build box hides its GPU while it is in use. A `native` build configured then records
# CMAKE_CUDA_ARCHITECTURES_NATIVE "No CUDA devices found.-real" and ships with no Blackwell kernels.

HIDDEN_SMI = "No devices were found\n"          # what nvidia-smi prints with CUDA_VISIBLE_DEVICES=-1


def _detect(smi, nvcc=(12, 8)):
    def probe(cmd, timeout=20):
        if cmd[0] == "nvidia-smi":
            return smi
        return f"Cuda compilation tools, release {nvcc[0]}.{nvcc[1]}, V0" if "nvcc" in cmd[0] else "cmake version 4.4.2"
    return lambda: U.detect_cuda(probe=probe)


NATIVE = ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native", "GGML_NATIVE:BOOL=ON"]


def test_hidden_gpu_never_keeps_native_cuda12(tmp_path):
    with pytest.raises(U.CudaArchError, match="POLLARD_CUDA_ARCH"):
        U.cache_flags(_tree(tmp_path, NATIVE), detect=_detect(HIDDEN_SMI))


def test_hidden_gpu_cuda13_gets_the_portable_list(tmp_path):
    f = U.cache_flags(_tree(tmp_path, NATIVE), detect=_detect(HIDDEN_SMI, (13, 0)))
    assert f["CMAKE_CUDA_ARCHITECTURES"] == U.CUDA13_PORTABLE


def test_hidden_gpu_uses_the_arch_recorded_while_it_was_visible(tmp_path):
    seen = tmp_path / "seen"
    seen.mkdir()
    f = U.cache_flags(_tree(seen, NATIVE), detect=_detect("12.0\n"))
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "native"             # visible: unchanged
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    f = U.cache_flags(_tree(hidden, NATIVE), detect=_detect(HIDDEN_SMI))
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "120a-real"


def test_hidden_gpu_configured_arch_env_then_engines_json(tmp_path, monkeypatch):
    U._save_caps(["8.6"])                                         # env and engines.json win over the record
    f = U.cache_flags(_tree(tmp_path, NATIVE), detect=_detect(HIDDEN_SMI), configured="120a-real")
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "120a-real"
    monkeypatch.setenv("POLLARD_CUDA_ARCH", "89-real")
    f = U.cache_flags(str(tmp_path), detect=_detect(HIDDEN_SMI), configured="120a-real")
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "89-real"


def test_visible_gpu_still_wins_over_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLARD_CUDA_ARCH", "89-real")
    f = U.cache_flags(_tree(tmp_path, NATIVE), detect=_detect("12.0\n"))
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "native"


def test_hidden_gpu_explicit_cached_list_is_kept(tmp_path):
    f = U.cache_flags(_tree(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120a-real"]),
                      detect=_detect(HIDDEN_SMI))
    assert f["CMAKE_CUDA_ARCHITECTURES"] == "120a-real"


def _configured(tmp_path, cache_lines, native_value):
    tree = _tree(tmp_path, cache_lines)
    d = tmp_path / "build" / "CMakeFiles" / "4.4.2"
    d.mkdir(parents=True)
    (d / "CMakeCUDACompiler.cmake").write_text(f'set(CMAKE_CUDA_ARCHITECTURES_NATIVE "{native_value}")\n')
    return tree


def test_post_configure_check_catches_no_device_sentinel(tmp_path):
    tree = _configured(tmp_path, NATIVE, "No CUDA devices found.-real")
    assert "No CUDA devices" in U.configured_cuda_arch_problem(tree)


def test_post_configure_check_passes_explicit_arch_even_with_sentinel(tmp_path):
    # CMake probes the native arch whatever the list says; with an explicit list it is unused.
    tree = _configured(tmp_path, ["GGML_CUDA:BOOL=ON", "CMAKE_CUDA_ARCHITECTURES:STRING=120a-real"],
                       "No CUDA devices found.-real")
    assert U.configured_cuda_arch_problem(tree) == ""


def test_post_configure_check_passes_native_with_a_gpu(tmp_path):
    assert U.configured_cuda_arch_problem(_configured(tmp_path, NATIVE, "120-real")) == ""


def test_post_configure_check_unset_arch_with_ggml_native(tmp_path):
    tree = _configured(tmp_path, ["GGML_CUDA:BOOL=ON", "GGML_NATIVE:BOOL=ON"], "No CUDA devices found.-real")
    assert U.configured_cuda_arch_problem(tree)


def test_build_stops_after_configure_when_no_gpu_targeted(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, timeout=None, env=None):
        calls.append(cmd)
        _configured(tmp_path, NATIVE, "No CUDA devices found.-real")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(U, "_run", fake_run)
    ok, msg = U.build(str(tmp_path), {"GGML_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "native"}, 2)
    assert not ok and "No CUDA devices" in msg
    assert len(calls) == 1                                        # configure only, no compile
