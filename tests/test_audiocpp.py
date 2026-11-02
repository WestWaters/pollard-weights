"""pollard-audiocpp: the audio.cpp lane's header reader (logical names behind `_audiocpp.<i>` aliases), eligibility,
row-width fallback, classes, the scorers' parsing, and the measured plan -> exact rules. No audio.cpp binaries."""
from __future__ import annotations

import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_audiocpp as A  # noqa: E402


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, logical=None):
    """A header-only GGUF: tensors = [(physical name, ggml type id, dims)]; metadata carries a string, a big u8 array
    (an embedded sidecar blob) and, when given, audiocpp.tensor_names."""
    kv = [_s("general.architecture") + struct.pack("<I", 8) + _s("audiocpp"),
          _s("audiocpp.embedded_files.data") + struct.pack("<IIQ", 9, 0, 5) + b"\x01\x02\x03\x04\x05"]
    if logical:
        kv.append(_s("audiocpp.tensor_names") + struct.pack("<IIQ", 9, 8, len(logical)) + b"".join(map(_s, logical)))
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv)
    for name, t, dims in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", t, 0)
    open(path, "wb").write(out)


def test_header_reader_maps_aliases_to_logical_names(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("enc.0.w", 30, [768, 768]), ("_audiocpp.1", 30, [768, 2048]), ("enc.norm", 0, [768])],
               logical=["enc.0.w", "kokoro/bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight", "enc.norm"])
    t = A.tensor_types(str(p))
    assert t["enc.0.w"] == ("bf16", 768 * 768, 768)
    assert t["kokoro/bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"] == ("bf16", 768 * 2048, 768)
    assert t["enc.norm"] == ("f32", 768, 768)
    assert "_audiocpp.1" not in t


def test_header_reader_without_name_table(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, [("a.weight", 8, [256, 4])])
    assert A.tensor_types(str(p)) == {"a.weight": ("q8_0", 1024, 256)}


def test_eligible_is_q8_conversion_plus_reference():
    src = {"lin.weight": ("bf16", 100, 256), "norm.weight": ("f32", 10, 10), "lstm.weight_hh_l0": ("bf16", 50, 256),
           "codec.weight": ("bf16", 70, 256)}
    q8 = {"lin.weight": ("q8_0", 100, 256), "norm.weight": ("f16", 10, 10), "lstm.weight_hh_l0": ("bf16", 50, 256),
          "codec.weight": ("bf16", 70, 256)}
    ref = {"lstm.weight_hh_l0": ("q8_0", 50, 256), "codec.weight": ("bf16", 70, 256)}
    assert set(A.eligible_from(src, q8)) == {"lin.weight"}               # a float-width change is not eligible
    assert set(A.eligible_from(src, q8, ref)) == {"lin.weight", "lstm.weight_hh_l0"}   # the codec stays protected


def test_fit_type_falls_back_to_q8_on_narrow_rows():
    assert A.fit_type("q2_k", 768) == "q2_k"
    assert A.fit_type("q4_k", 128) == "q8_0"
    assert A.fit_type("q8_0", 128) == "q8_0"


def test_class_of():
    assert A.class_of("thinker.layers.12.mlp.experts.gate_up") == "thinker.layers.12"
    assert A.class_of("kokoro/decoder.decode.0.conv1.weight_v") == "kokoro/decoder.decode.0"
    assert A.class_of("kokoro/predictor.lstm.weight_hh_l0") == "kokoro/predictor.lstm"
    assert A.class_of("solo") == "solo"


def test_cer():
    assert A.cer("the quick brown fox", "The  quick brown fox") == 0.0
    assert A.cer("abcd", "abce") == 0.25
    assert A.cer("", "") == 0.0 and A.cer("", "x") == 1.0


def test_parse_cli_reads_text_and_duration():
    out = "\n".join(["task=tts", "request_index=0", "request_id=s0", "audio_out=o/s0.wav",
                     "metrics[s0].audio_duration_ms=3225", "request_id=a1",
                     "text_output=the quick brown fox", "metrics[a1].wall_ms=12.5"])
    got = A.parse_cli(out)
    assert got["s0"]["audio_duration_ms"] == 3225.0
    assert got["a1"]["text"] == "the quick brown fox"


def _classes():
    el = {"enc.0.w": {"numel": 768 * 768, "row": 768, "source": "bf16"},
          "enc.1.w": {"numel": 768 * 768, "row": 768, "source": "bf16"},
          "dec.fc.weight": {"numel": 128 * 512, "row": 128, "source": "bf16"}}
    return A.classes(el)


def test_plan_spends_bits_where_measured_cost_is():
    pytest.importorskip("numpy")
    cls = _classes()
    sens = {"enc.0": {"q2_k": 0.5, "q4_k": 0.01}, "enc.1": {"q2_k": 0.0, "q4_k": 0.0}, "dec.fc": {"q2_k": 0.0}}
    choice, res = A.plan(cls, sens, 5.0)
    assert choice["enc.1"] == "q2_k"                 # free to drop: takes the bottom
    assert A.BPW[choice["enc.0"]] > A.BPW["q2_k"]    # the sensitive one keeps bits
    rules = dict(A.rules_for(cls, choice))
    assert rules["dec.fc.weight"] == "q8_0"          # 128-wide rows can't take a k-quant
    assert res["size"] <= int(sum(c["numel"] for c in cls.values()) * 5.0 / 8)


def test_plan_refuses_unmeasured_classes_and_infeasible_budgets():
    pytest.importorskip("numpy")
    cls = _classes()
    with pytest.raises(SystemExit, match="no measurement"):
        A.plan(cls, {"enc.0": {"q2_k": 0.1}}, 6.0)
    with pytest.raises(SystemExit, match="floor"):
        A.plan(cls, {c: {"q2_k": 0.0} for c in cls}, 1.0)


def test_pollard_card_renders_an_audiocpp_ladder(tmp_path, monkeypatch):
    """The audio.cpp lane's card comes from pollard-card, the template every Pollard repo uses: its hero, its size
    table, its rung guide and files table -- with audio.cpp's numbers and run commands, not llama.cpp's."""
    import subprocess
    monkeypatch.setenv("POLLARD_HOME", str(tmp_path / "home"))
    import pollard_workspace as ws
    files = []
    for label, size in (("6.4bpw", 1391), ("8.0bpw", 1577)):
        f = tmp_path / f"M-tts-0.5B-Pollard-{label}.gguf"
        hdr = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)                 # a valid, empty GGUF header (audio.cpp's own)
        f.write_bytes(hdr + b"\0" * size)
        ws.record_build("org/M-tts-0.5B", "audiocpp", str(f), tag=label)
        files.append(f)
    rungs = [{"file": files[0].name, "delta": 0.0397}, {"file": files[1].name, "delta": 0.0248}]
    res = A.card_results(rungs, {"bytes": 1742, "delta": 0.0253}, 2998, "ming3_tts", "tts", 24, "citrinet")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(res))
    out = tmp_path / "README.md"
    tools = os.path.join(os.path.dirname(__file__), "..", "tools")
    subprocess.run([sys.executable, os.path.join(tools, "pollard_card.py"), "--model", "org/M-tts-0.5B",
                    "--lane", "audiocpp", "--results", str(rp), "--repo", "PollardWeights/M-tts-0.5B-Pollard",
                    "--out", str(out), "--params", "0.5B", "--license", "apache-2.0",
                    "--pipeline-tag", "text-to-speech"],
                   check=True, env=dict(os.environ, POLLARD_HOME=str(tmp_path / "home")), capture_output=True)
    md = out.read_text()
    assert "library_name: gguf" in md and "audio.cpp" in md and "Pollard shrank this model for audio.cpp" in md
    assert "## Which file should I choose?" in md and "## Available files" in md and "## How to run" in md
    assert "| file | size | error vs source | vs audio.cpp q8_0 |" in md
    assert "+0.0397" in md and "+0.0248" in md and "audiocpp_cli --task tts --family ming3_tts" in md
    assert "llama-server" not in md and "/Users/" not in md


def test_publish_goes_through_safeapi(tmp_path, monkeypatch):
    import types
    calls = []

    class Stub:
        def create_repo(self, repo, **k):
            calls.append(("create", repo, k.get("private")))

        def upload_file(self, **k):
            calls.append(("upload", k["path_in_repo"]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Stub))
    f = tmp_path / "M-Pollard-8.0.gguf"
    f.write_bytes(b"not a gguf header")
    readme = tmp_path / "README.md"
    readme.write_text("# card\n")
    url = A.publish("PollardWeights/M-Pollard", [str(f)], str(readme), log=lambda *a: None)
    assert url.endswith("/PollardWeights/M-Pollard")
    assert calls == [("create", "PollardWeights/M-Pollard", False), ("upload", "README.md"),
                     ("upload", "M-Pollard-8.0.gguf")]


def test_rung_names_carry_a_quant_type_the_hub_reads(tmp_path):
    """Ming went up as -6.4bpw.gguf and the Hub's GGUF widget could not read a type: names carry the main type."""
    p = tmp_path / "rung.gguf"
    # mostly q8_0 bytes, a q2_k slice, a big bf16 tensor that stays at source precision
    write_gguf(p, [("a", 8, [896, 4096]), ("b", 10, [256, 1024]), ("spk", 30, [896, 2048])])
    assert A.main_type(str(p)) == "Q8_0"
    assert A.rung_name("Ming-3.0-tts-0.5B", str(p), 6.4) == "Ming-3.0-tts-0.5B-Pollard-Q8_0-6.4bpw.gguf"
    for ok in ("M-Pollard-Q8_0-6.4bpw.gguf", "M-Pollard-IQ3_S.gguf", "M-Pollard-Q4_K_M.gguf", "M-bf16.gguf"):
        assert A.HUB_TYPE.search(ok), ok
    for bad in ("M-Pollard-6.4bpw.gguf", "M-Pollard-8.0.gguf", "M-Pollard-MIX-5bpw.gguf"):
        assert not A.HUB_TYPE.search(bad), bad


def test_publish_refuses_a_rung_name_without_a_type(tmp_path):
    f = tmp_path / "M-Pollard-6.4bpw.gguf"
    write_gguf(f, [("a", 8, [256, 4])])
    s = tmp_path / "s.json"
    s.write_text("{}")
    with pytest.raises(SystemExit, match="M-Pollard-6.4bpw.gguf -> M-Pollard-Q8_0-6.4bpw.gguf"):
        A.main(["publish", "--repo", "PollardWeights/M-Pollard", "--family", "x", "--base-model", "o/M",
                "--rung", f"{f}={s}", "--dry-run"])


def test_build_into_a_folder_names_the_rung(tmp_path, monkeypatch):
    def fake_convert(source, family, out, base, rules=(), extra=()):
        write_gguf(out, [("a", 8, [256, 1024]), ("b", 12, [256, 64])])
        return out
    monkeypatch.setattr(A, "convert", fake_convert)
    r = tmp_path / "rules.json"
    r.write_text(json.dumps({"bpw": 7.0, "rules": [["b", "q4_k"]]}))
    src = tmp_path / "ming-3.0-tts-0.5b-bf16.gguf"
    src.write_bytes(b"")
    out = tmp_path / "rungs"
    A.main(["build", "--source", str(src), "--family", "x", "--rules", str(r), "--out", str(out) + "/",
            "--name", "Ming-3.0-tts-0.5B"])
    assert os.listdir(out) == ["Ming-3.0-tts-0.5B-Pollard-Q8_0-7.0bpw.gguf"]
