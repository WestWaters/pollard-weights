#!/usr/bin/env python3
"""pollard-connectome -- get a real connectome to attach a brain to.

`pollard-flybrain` takes ANY connectome: a feather of body_pre / body_post / weight plus a per-neuron
+1/-1 sign file. What it does not do is find you one. This builds them.

  pollard-connectome --human --out h01_human          # human cortex (H01), ~13.5k neurons
  pollard-connectome --list                           # what is available and where it comes from

The human graph is H01 (Shapson-Coe et al. 2024), a cubic millimetre of human temporal cortex
reconstructed from electron microscopy -- the same KIND of data as the fly's MaleCNS, not an MRI
tractography estimate. Two things about it decide whether the result is biology or an artifact, and
both are handled here:

  * H01 ships 166 Avro shards, ~33 GB, holding ~166M synapses, and only ~0.3% of those join two cells
    whose soma is inside the volume -- the rest land on neurites cut off at the boundary. So this
    streams a shard, keeps the cell-to-cell pairs, deletes it, and moves on. Peak disk is one shard,
    about 200 MB, and it checkpoints after each so an interrupted run resumes.

  * The soma table covers all ~57k cells, most of which are GLIA. Left unfiltered, the single largest
    edge class is astrocyte->pyramidal, which is not a synapse -- astrocyte processes wrap real
    synapses and the detector picks them up. Keeping them would hand you a graph whose commonest
    "connection" is biologically impossible. Neurons only, by default.

Sign convention follows Dale's law from the cell type (pyramidal and spiny excitatory, interneuron
inhibitory) rather than the detector's own excitatory/inhibitory call, which agrees with Dale only
57.5% of the time on this data -- barely above chance, and not worth trusting.

Expect roughly 13,473 neurons and 75,452 edges. That is more cells than the fly's 8,552 but fewer
synapses, because a 1 mm^3 block cuts most connections: mean 1.51 synapses per edge against a fly
connectome that is whole. Worth saying plainly before anyone reads a fly-vs-human result: the fly
graph is an entire brain, the human graph is a fragment of one.

Data: H01 is CC-BY 4.0 (Harvard/Google). Cite Shapson-Coe et al., Science 384 (2024).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
import urllib.request

BUCKET   = "https://storage.googleapis.com/h01-release"
SYNAPSES = BUCKET + "/data/20210729/c3/synapses/exported/export%012d"
SOMAS    = BUCKET + "/data/20210601/c3/tables/somas.csv"
N_SHARDS = 166

# Dale's law from the H01 cell-type labels.
EXCITATORY = {"PYRAMIDAL", "SPINY_STELLATE", "SPINY_ATYPICAL", "UNCLASSIFIED_NEURON"}
INHIBITORY = {"INTERNEURON"}
NEURONS    = EXCITATORY | INHIBITORY


def _fetch(url: str, path: str, tries: int = 5) -> bool:
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except Exception as e:                       # a 33 GB pull over a home link WILL wobble
            print(f"      retry {i+1}/{tries}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    return False


def load_somas(path: str) -> dict:
    """segment id -> (cell type, cortical layer), for every soma inside the volume."""
    if not os.path.isfile(path) and not _fetch(SOMAS, path):
        raise SystemExit("could not download the H01 soma table")
    soma = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for key in ("c3_rep_strict", "c3_rep_manual"):
                v = r.get(key)
                if v and v != "NULL":
                    soma[int(v)] = (r.get("celltype") or "?", r.get("layer") or "?")
    return soma


def build_human(out: str, workdir: str, keep_glia: bool) -> None:
    try:
        import fastavro
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise SystemExit(f"needs fastavro, numpy and pandas ({e}). "
                         "pip install 'pollard-weights[flybrain]' fastavro") from None

    os.makedirs(workdir, exist_ok=True)
    somas = os.path.join(workdir, "h01_somas.csv")
    state = os.path.join(workdir, "h01_progress.json")
    shard = os.path.join(workdir, "h01_shard.tmp")

    soma = load_somas(somas)
    print(f"  somas: {len(soma):,} segment ids", flush=True)

    edges = collections.Counter()
    first = 0
    if os.path.isfile(state):
        st = json.load(open(state, encoding="utf-8"))
        first = st["next_shard"]
        for k, v in st["edges"].items():
            p, q = k.split(",")
            edges[(int(p), int(q))] = v
        print(f"  resuming at shard {first}/{N_SHARDS} with {len(edges):,} edges", flush=True)

    t0 = time.time()
    for i in range(first, N_SHARDS):
        if not _fetch(SYNAPSES % i, shard):
            raise SystemExit(f"shard {i} would not download; rerun to resume")
        with open(shard, "rb") as f:
            for rec in fastavro.reader(f):
                p = rec["pre_synaptic_site"]["neuron_id"]
                q = rec["post_synaptic_partner"]["neuron_id"]
                if p in soma and q in soma:
                    edges[(p, q)] += 1
        os.remove(shard)
        done, el = i - first + 1, time.time() - t0
        print(f"  shard {i+1:3d}/{N_SHARDS}  {len(edges):,} cell-to-cell edges"
              f"   [{el/60:.1f} min, eta {el/done*(N_SHARDS-i-1)/60:.0f} min]", flush=True)
        json.dump({"next_shard": i + 1,
                   "edges": {f"{p},{q}": v for (p, q), v in edges.items()}},
                  open(state, "w", encoding="utf-8"))

    keep = [(p, q, n) for (p, q), n in edges.items()
            if keep_glia or (soma[p][0] in NEURONS and soma[q][0] in NEURONS)]
    if not keep:
        raise SystemExit("no edges survived filtering -- that should not happen; keep the workdir")

    pre  = np.array([p for p, _, _ in keep], dtype=np.int64)
    post = np.array([q for _, q, _ in keep], dtype=np.int64)
    wts  = np.array([n for _, _, n in keep], dtype=np.float32)

    # The sign file is indexed by position in np.unique(concat(pre, post)) -- the same remap the
    # trainer applies. Build it the same way or every sign lands on the wrong neuron.
    nodes = np.unique(np.concatenate([pre, post]))
    signs = np.array([-1.0 if soma[int(n)][0] in INHIBITORY else 1.0 for n in nodes],
                     dtype=np.float32)

    pd.DataFrame({"body_pre": pre, "body_post": post, "weight": wts}).to_feather(out + ".feather")
    np.save(out + "_signs.npy", signs)
    inh = int((signs < 0).sum())
    print(f"\n  {out}.feather    {len(pre):,} edges, {int(wts.sum()):,} synapses "
          f"({wts.sum()/len(pre):.2f} per edge)")
    print(f"  {out}_signs.npy  {len(signs):,} neurons "
          f"({len(signs)-inh:,} excitatory / {inh:,} inhibitory, {100*inh/len(signs):.1f}% inhibitory)")
    print(f"\n  pollard-flybrain --train 900 --connectome {out}.feather --signs {out}_signs.npy \\")
    print(f"      --model <hf-id> --probes <text> --brain HumanBrain.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--human", action="store_true", help="build the H01 human cortex connectome")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--out", default="h01_human", help="output prefix (default h01_human)")
    ap.add_argument("--workdir", default="h01_work", help="scratch for shards and resume state")
    ap.add_argument("--keep-glia", action="store_true",
                    help="do NOT filter to neurons (astrocyte contacts included -- see the notes)")
    a = ap.parse_args()

    if a.list or not a.human:
        print(__doc__)
        print("  available:")
        print("    --human    H01 human temporal cortex, ~13.5k neurons / ~75k edges, CC-BY 4.0")
        print("    (fly)      MaleCNS v1.0 from FlyEM/Janelia -- see the pollard-flybrain docs")
        if not a.human:
            sys.exit(0 if a.list else 2)
    build_human(a.out, a.workdir, a.keep_glia)


if __name__ == "__main__":
    main()
