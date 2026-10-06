"""A pool can have enough total bytes while one worker still cannot fit."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import pollard_calc as C


def profile(count=2):
    return {"schema_version": 1, "nodes": [
        {"name": f"node-{i+1}", "available_memory_gb": 110, "runtime_reserve_gb": 10}
        for i in range(count)]}


def test_unified_memory_is_one_pool_not_ram_plus_vram():
    result = C.cluster_estimate(profile(8), 640, 5)
    assert result["capacity_gb"] == 800
    assert all(n["weights_gb"] == 80 and n["kv_gb"] == 5 for n in result["nodes"])


def test_replicated_kv_is_charged_on_every_node():
    result = C.cluster_estimate(profile(), 180, 15)
    assert not result["fits_all_nodes"]
    assert all(n["headroom_gb"] == -5 for n in result["nodes"])


def test_explicit_partitioned_kv_changes_the_budget():
    p = profile()
    p["kv_layout"] = "partitioned"
    for node in p["nodes"]:
        node["kv_fraction"] = .5
    assert C.cluster_estimate(p, 180, 15)["fits_all_nodes"]


def test_total_capacity_does_not_hide_a_small_worker():
    p = profile()
    p["nodes"][0]["available_memory_gb"] = 60
    result = C.cluster_estimate(p, 120, 0)
    assert result["capacity_gb"] > 120
    assert not result["fits_all_nodes"]


def test_explicit_unequal_weight_shards():
    p = profile()
    p["nodes"][0]["weight_fraction"] = .25
    p["nodes"][1]["weight_fraction"] = .75
    assert [n["weights_gb"] for n in C.cluster_estimate(p, 120, 0)["nodes"]] == [30, 90]


@pytest.mark.parametrize("bad", [True, -1, 0, float("nan"), float("inf"), "128"])
def test_bad_memory_values_are_rejected(bad):
    p = profile()
    p["nodes"][0]["available_memory_gb"] = bad
    with pytest.raises(ValueError):
        C.validate_cluster_profile(p)


@pytest.mark.parametrize("change", ["duplicate", "fraction", "partial", "kv", "address", "bandwidth"])
def test_invalid_layout_is_rejected(change):
    p = profile()
    if change == "duplicate": p["nodes"][1]["name"] = "node-1"
    if change == "fraction":
        for n in p["nodes"]: n["weight_fraction"] = .8
    if change == "partial": p["nodes"][0]["weight_fraction"] = .5
    if change == "kv": p["kv_layout"] = "partitioned"
    if change == "address": p["nodes"][0]["address"] = "192.0.2.1"
    if change == "bandwidth": p["nodes"][0]["memory_bandwidth_gbps"] = 273
    with pytest.raises(ValueError): C.validate_cluster_profile(p)


def test_report_does_not_claim_linear_network_scaling(capsys):
    p = profile()
    p["fabric"] = {"bandwidth_gbps": 20, "basis": "assumed"}
    C.cluster_report(p, 120, 0, 0)
    out = capsys.readouterr().out
    assert "not used as a throughput multiplier" in out
    assert "KV excluded" in out
    assert "No cluster speed prediction" in out


def test_cli_uses_cluster_budget_not_default_single_machine(tmp_path):
    cfg = {"model_type": "llama", "hidden_size": 1024, "num_hidden_layers": 4,
           "num_attention_heads": 8, "num_key_value_heads": 8, "intermediate_size": 4096,
           "vocab_size": 32000}
    config_path, profile_path = tmp_path / "config.json", tmp_path / "cluster.json"
    config_path.write_text(json.dumps(cfg))
    profile_path.write_text(json.dumps(profile()))
    result = subprocess.run([sys.executable, str(Path(C.__file__)), "--config", str(config_path),
                             "--cluster-profile", str(profile_path), "--ctx", "4096",
                             "--cluster-concurrency", "4"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "2-node profile" in result.stdout
    assert "4 sequences at 4096 tokens each" in result.stdout
    assert "16 GB RAM" not in result.stdout
    assert "RAM-bandwidth ceil" not in result.stdout
    # kv_cache_bytes returns bytes; profile budgets are decimal GB. Four
    # sequences here need 268,435,456 bytes, printed as 0.3 GB per replica.
    row = next(line for line in result.stdout.splitlines() if line.startswith("node-1 |"))
    assert float(row.split("|")[3]) == .3
    assert "Allocation fits the stated budgets" in result.stdout


def test_snapshot_is_not_modified_by_estimation():
    p = profile()
    original = copy.deepcopy(p)
    C.cluster_estimate(p, 120, 5)
    assert p == original


def test_boolean_schema_version_is_not_version_one():
    p = profile()
    p["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        C.validate_cluster_profile(p)


def test_report_labels_the_default_weight_split_as_an_assumption(capsys):
    C.cluster_report(profile(), 120, 5, 4096)
    assert "assumed equal shards" in capsys.readouterr().out
