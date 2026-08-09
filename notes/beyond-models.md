# Beyond models — the memory-tier argument

_2026-08-06, prompted by the TSMC/DRAM-shortage news ($1B of processors
waiting on DRAM to package): "I think my idea also solves their DRAM issue if they
use pollard weights for their systems instead. not just models."_

## The honest version of the claim

Pollard is a **memory-tier arbitrage method**: measure a workload's actual reuse,
hold the hot set in the scarce fast tier (DRAM), stream the cold set from the
abundant cheap tier (flash), and *compute* the boundary instead of guessing it.

That principle is not new — it is working-set theory (Denning, 1968), buffer
pools, page caches, CDNs. What IS new in Pollard is the instrument set for the
workload that's currently eating the world's DRAM: **model weights**, where the
"access pattern" has semantics no OS can see (expert routing, layer depth,
step redundancy, per-layer sensitivity). An LRU page cache can't know that
layer 40's experts are 6.9% active carrying 71% of energy. Pollard can.

## Why the DRAM connection is real (directionally)

The AI buildout prices DRAM as if every parameter must be resident. Every result
in this repo says that pricing is wrong by a measurable factor: residency ≠ file
size, and the gap between them is the routing-reuse number nobody measures. If
device makers sized DRAM to *measured hot sets* instead of total weights:

- an "AI PC" spec'd for a 20B-resident model needs the DRAM of a 4-6B hot set,
- the delta multiplied across hundreds of millions of devices is exactly the
  demand curve the shortage lives on.

Apple's own *LLM in a Flash* (2312.11514) is this thesis applied internally —
DRAM-constrained devices streaming weights from flash. Pollard is the
generalized, published, model-agnostic version with the measurement tooling.

## What NOT to claim

- Not "solves the DRAM shortage" — packaging supply chains have their own physics.
- Not a new idea in computing — a new *instrument* for a new *workload*.
- The claim that survives contact: **"the industry is provisioning DRAM for
  worst-case residency; Pollard measures the real requirement, and the real
  requirement is much smaller."** That's a procurement argument, not just an
  inference trick — and it's testable with the harnesses in this repo.
