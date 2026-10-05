# Per-node cluster budgets

`pollard-calc --cluster-profile profile.json --config config.json --ctx 32768`
checks each node, rather than treating summed RAM as one unrestricted allocation.
Add `--cluster-concurrency 4` to budget four sequences at that context length.

```json
{
  "schema_version": 1,
  "kv_layout": "replicated",
  "nodes": [
    {"name": "node-1", "available_memory_gb": 110, "runtime_reserve_gb": 12},
    {"name": "node-2", "available_memory_gb": 110, "runtime_reserve_gb": 12}
  ],
  "fabric": {"bandwidth_gbps": 20, "basis": "assumed"}
}
```

These are example numbers, not benchmark results. All GB values are decimal.
Use anonymous node labels. Do not put addresses, credentials or private paths in
a profile intended for publication.

`available_memory_gb` is a current allocatable-memory snapshot. On unified-memory
hardware, count the pool once, not host RAM plus GPU memory. MemAvailable is a
useful starting point, not a guarantee that CUDA can allocate that many bytes.
`runtime_reserve_gb` is additional headroom for activations, workspaces and other
non-KV allocations. Choose it from peak measurements when available; otherwise
document it as an assumption.

Weights default to equal fractions. To describe uneven placement, provide
`weight_fraction` on every node, summing to one. KV defaults to full replication
on every node. To use partitioned KV, set `kv_layout` to `partitioned` and provide
`kv_fraction` on every node, summing to one. These are explicit placement
assumptions, not automatic detection of tensor-parallel or RPC behavior. Runtime
KV replication, layer placement, quantization support and model divisibility
must be checked separately.

Config-based weight estimates also exclude format-specific scales, padding and
metadata. `--gguf` uses the actual file bytes instead, but on-disk size still
does not establish loaded GPU memory. Charge additional runtime allocations to
the reserve rather than treating a narrow arithmetic fit as an OOM guarantee.

Optional node `memory_bandwidth_gbps` needs a `bandwidth_basis` of `measured` or
`assumed`. The same basis is required for fabric bandwidth. They are recorded,
not converted into a linear throughput claim. Cross-node latency, collective
traffic, topology and storage access can dominate speed.

The verdict only means weights plus estimated KV fit the stated budgets on all
nodes. It is not a tested deployment, a scheduling reservation, a guarantee
against OOM, or a claim that one runtime supports an arbitrary number of nodes.
