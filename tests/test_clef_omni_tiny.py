"""clef-omni end to end on a tiny random checkpoint: generate it, convert it, serve one /v1/systemone request and
compare with the answers recorded when the runtime patch was rebased onto llama.cpp b76b133 (CPU, 2026-10-15).

Every step needs something CI does not have (torch + transformers, the cached Qwen3 tokenizer, a converter and a
llama-server built with the clef-omni patch), so each one skips cleanly when its piece is missing. Point
POLLARD_CONVERTER / POLLARD_LLAMA_BIN_DIR at a patched tree to run it anywhere.
"""
import glob, json, os, socket, subprocess, sys, time, urllib.request

import pytest

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(HERE, "fixtures"))

REQUEST = {
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": None, "shipping": "Parcels and delivery", "technical": None, "sales": None}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["can wait", "this week", "today", "right now"]},
    },
}
# Recorded on CPU; the old (pre-rebase) build agreed to 2.6e-9, Metal to 1.5e-5.
EXPECTED = {
    "route": {"billing": 0.20283965436170093, "sales": 0.26204547523755856,
              "shipping": 0.283206975655403, "technical": 0.25190789474533754},
    "angry": 0.49883521931470953,
    "urgency": {"0": 0.2197079674117298, "1": 0.2515022045208273, "2": 0.26299167600685935, "3": 0.2657981520605836},
}
TOL = 1e-4


def _tokenizer_dir():
    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/"))
    hits = [h for h in hits if os.path.exists(os.path.join(h, "tokenizer.json"))]
    return hits[0] if hits else None


def _patched(path, needle=b"clef-omni"):
    try:
        with open(path, "rb") as f:
            return needle in f.read()
    except OSError:
        return False


def _converter():
    for c in (os.environ.get("POLLARD_CONVERTER"),
              os.path.join(REPO, "runtime", "llama.cpp", "convert_hf_to_gguf.py")):
        if c and os.path.exists(c):
            gguf_py = os.path.join(os.path.dirname(c), "gguf-py", "gguf", "constants.py")
            if _patched(gguf_py):
                return c
    return None


def _server():
    for d in (os.environ.get("POLLARD_LLAMA_BIN_DIR"), os.path.join(REPO, "runtime", "llama.cpp", "build", "bin")):
        if not d:
            continue
        exe = os.path.join(d, "llama-server.exe" if os.name == "nt" else "llama-server")
        libs = glob.glob(os.path.join(d, "libllama*")) + glob.glob(os.path.join(d, "llama.dll"))
        if os.path.exists(exe) and any(_patched(p) for p in libs + [exe]):
            return exe
    return None


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3OmniMoeConfig"):
        pytest.skip("transformers has no Qwen3OmniMoeConfig")
    tok = _tokenizer_dir()
    if not tok:
        pytest.skip("Qwen/Qwen3-0.6B tokenizer not in the HF cache")
    conv = _converter()
    if not conv:
        pytest.skip("no convert_hf_to_gguf.py with the clef-omni patch (set POLLARD_CONVERTER)")
    import make_tiny_clef_omni
    d = tmp_path_factory.mktemp("clefomni")
    model = str(d / "hf")
    make_tiny_clef_omni.make(model, tok)
    out = str(d / "tiny-clef-omni.gguf")
    r = subprocess.run([sys.executable, conv, model, "--outtype", "f32", "--outfile", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return out


def test_tiny_checkpoint_converts_to_clef_omni(tiny_gguf):
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(tiny_gguf)
    arch = reader.fields["general.architecture"]
    assert bytes(arch.parts[arch.data[0]]).decode() == "clef-omni"
    names = {t.name for t in reader.tensors}
    assert any(n.startswith("blk.0.") for n in names)
    assert any("decision" in n or "joint" in n or n.startswith("dec.") for n in names), sorted(names)[:40]
    # llama-server refuses a clef-omni GGUF without its M-RoPE sections ("key not found")
    sec = reader.fields.get("clef-omni.rope.dimension_sections")
    assert sec is not None, "no clef-omni.rope.dimension_sections (mrope_section missing from the config)"
    assert [int(sec.parts[i][0]) for i in sec.data][:3] == [3, 3, 2]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_systemone_answers_match_the_recorded_run(tiny_gguf):
    exe = _server()
    if not exe:
        pytest.skip("no llama-server built with the clef-omni patch (set POLLARD_LLAMA_BIN_DIR)")
    port = _free_port()
    proc = subprocess.Popen([exe, "-m", tiny_gguf, "--port", str(port), "-ngl", "0", "-c", "1024", "-b", "1024",
                             "-ub", "1024", "-np", "1", "--threads", "2"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/health", timeout=1); break
            except Exception:
                time.sleep(0.25)
        req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(REQUEST).encode(),
                                     headers={"Content-Type": "application/json"})
        got = json.load(urllib.request.urlopen(req, timeout=60))["answers"]
    finally:
        proc.terminate(); proc.wait(timeout=10)
    for k, v in EXPECTED["route"].items():
        assert abs(got["route"]["probabilities"][k] - v) < TOL
    assert abs(got["angry"]["noul"] - EXPECTED["angry"]) < TOL
    for k, v in EXPECTED["urgency"].items():
        assert abs(got["urgency"]["probabilities"][k] - v) < TOL
