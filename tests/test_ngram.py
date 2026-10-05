"""pollard-ngram + pollard-fit --disk: lookup tables (n-gram / per-layer embeddings) live on the SSD."""
import os, struct, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pollard_ngram as N
import pollard_fit as F


def _gguf_header(path, arch, tensors):
    """A GGUF v3 header (KV + tensor infos, no data): enough for pollard_calc's metadata reader."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 1)
    out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    off = 0
    for name, dims in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<I", 1) + struct.pack("<Q", off)
        n = 1
        for d in dims:
            n *= d
        off += 2 * n
    open(path, "wb").write(out)


def test_find_tables_names():
    tp = {"token_embd.weight": 1, "per_layer_token_embd.weight": 2, "blk.3.engram.embed.weight": 3,
          "blk.0.ngram_embd.weight": 4, "blk.0.ffn_up.weight": 5, "output.weight": 6}
    got = set(N.find_tables(tp))
    assert got == {"per_layer_token_embd.weight", "blk.3.engram.embed.weight", "blk.0.ngram_embd.weight"}, got
    assert "token_embd.weight" not in got, "the normal token embedding is not a lazy table"


def test_disk_plan_auto_needs_lazy_runtime():
    tp = {"per_layer_token_embd.weight": 51_000_000_000, "blk.0.ffn_up.weight": 10}
    d = N.disk_plan({"general.architecture": "qwen4exp", "_tensor_params": tp}, "auto", "q8_0")
    assert d["params"] == 51_000_000_000 and d["names"] == ["per_layer_token_embd.weight"]
    assert abs(d["gb"] - 51e9 * 8.5 / 8 / 1e9) < 1e-6
    # an architecture this runtime does NOT lazy-read keeps its table in RAM, and says so
    n = N.disk_plan({"general.architecture": "made-up-arch", "_tensor_params": tp}, "auto", "q8_0")
    assert n["params"] == 0 and "does not lazy-read" in n["note"]
    assert N.disk_plan({"general.architecture": "qwen4exp", "_tensor_params": tp}, "off") is None
    forced = N.disk_plan({"general.architecture": "made-up-arch", "_tensor_params": tp}, r"per_layer_token_embd\.weight", "f16")
    assert forced["params"] == 51_000_000_000 and forced["type"] == "f16"


def test_fit_disk_tier_frees_the_budget():
    # a dense model whose 'other' group is dominated by a 20B lookup table, on a 24 GB machine
    h, L = 4096, 32
    ffn, attn, table = 3 * h * 14336, 4 * h * h, 20_000_000_000
    arch = dict(kind="dense", layers=L, hidden=h, total=L * (ffn + attn) + 2 * h * 150000 + table,
                expert_params=0, n_experts=0, dense_ffn_params=ffn, attn_params=attn)
    disk = {"params": table, "type": "q8_0", "gb": table * 8.5 / 8 / 1e9,
            "patterns": [r"per_layer_token_embd\.weight"], "names": ["per_layer_token_embd.weight"]}
    ov_d, emb_d, gb_d, base_d, (sum_d, _) = F.plan_allocation(arch, 24, 3, disk=disk)
    assert (r"per_layer_token_embd\.weight", "q8_0") in ov_d, "the table is pinned at the disk type"
    assert gb_d <= 24 * 0.85 - 3 + 1e-6, "the RAM projection excludes the table"
    assert "disk:" in sum_d
    ffn_type = lambda ov: {p.replace("\\", ""): t for p, t in ov}.get("blk.0.ffn_up.weight")
    # without the disk tier the same model must give up precision (or not fit at all)
    try:
        ov_r, *_ = F.plan_allocation(arch, 24, 3)
        assert F.BPW[ffn_type(ov_r)] < F.BPW[ffn_type(ov_d)], \
            f"charging the table to RAM must cost the transformer bits: {ffn_type(ov_r)} vs {ffn_type(ov_d)}"
    except SystemExit:
        pass                                                  # did not fit at all: the point exactly


def test_run_command_keeps_table_lazy():
    cmd = " ".join(N.run_command("m.gguf", ["per_layer_token_embd.weight"], ram=64, vram=12, ctx=4096, moe=True))
    assert "--load-mode mmap" in cmd and "--lazy-mode on" in cmd
    assert r"per_layer_token_embd\.weight=CPU" in cmd and "--n-cpu-moe" in cmd
    assert "--no-mmap" not in cmd and "mlock" not in cmd


def test_reader_sizes_every_tensor():
    from pollard_calc import read_gguf_meta
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf")
        _gguf_header(p, "qwen4exp", [("token_embd.weight", (64, 1000)), ("per_layer_token_embd.weight", (128, 50000)),
                                     ("blk.0.ffn_up.weight", (64, 256))])
        meta = read_gguf_meta(p)
        assert meta["_tensor_params"]["per_layer_token_embd.weight"] == 128 * 50000
        d_ = N.disk_plan(meta, "auto", "q8_0")
        assert d_["names"] == ["per_layer_token_embd.weight"]
