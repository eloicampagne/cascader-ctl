"""Random chain against the distance-ordered chain.

Question (reviewer). Does the chain gain from visiting each task after a close
one, or from the sequential accumulation of refinements alone? RandChain is a
Hamiltonian path in uniformly random order from the medoid root. It has the
chain's depth, its subtree sizes, its number of transfers and its SLE
allocation (edge lengths read from the same gradient distance), so only the
order of the tasks differs from Chain.

Protocol of the main tables, through the package in ../../src: per-method rate
selected on tuning seeds 500..504, evaluation on seeds 0..19, floor-one SLE
allocation, linear or MLP local model. WEAVE-UK selects its rate on the 48-row
validation slice, as in the paper. One random order per seed, shared by every
rate and budget of that seed.

    CASCADER_DATA_DIR=<data_links> python3.12 randchain.py --dataset Syn
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import cascader                                                   # noqa: E402
import cascader.methods as CM                                     # noqa: E402
import cascader.trees as CT                                       # noqa: E402
from cascader.data import load                                    # noqa: E402
from cascader.data.weave import weave_collection                  # noqa: E402
from cascader.experiments._common import (TUNE_OFFSET, etas_for,  # noqa: E402
                                          select_and_report, sweep)

assert Path(cascader.__file__).resolve().is_relative_to(HERE.parents[1] / "src"), cascader.__file__
torch.set_num_threads(1)


def build_randchain(nodes, D, root, rng=None, **_):
    """Path from ``root`` through the other tasks in uniformly random order."""
    nodes = list(nodes)
    rng = np.random.default_rng() if rng is None else rng
    rest = [v for v in nodes if v != root]
    order = [root] + [rest[i] for i in rng.permutation(len(rest))]
    return CT.root_tree(list(zip(order[:-1], order[1:])), root, nodes)


CT.CONSTRUCTIONS["RandChain"] = build_randchain
CM.SLE_METHODS = CM.SLE_METHODS | {"RandChain"}
METHODS = ["Individual", "Chain", "RandChain"]


def _fresh(c):
    # weave_collection returns one cached object for every seed, and
    # build_structure caches trees on it: drop that cache so each seed draws
    # its own random structure.
    c.__dict__.pop("_tree_cache", None)
    return c


def loader(ds, split):
    if ds == "WEAVE-UK":
        return (lambda s: _fresh(weave_collection(val_size=48, sample=False))) if split == "tune" \
            else (lambda s: _fresh(weave_collection(sample=False)))
    return lambda s: load(ds, seed=s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="linear", choices=("linear", "mlp"))
    p.add_argument("--tune", type=int, default=5)
    p.add_argument("--eval", type=int, default=20)
    p.add_argument("--out", default=str(HERE))
    a = p.parse_args()
    rows = []
    out = Path(a.out) / f"randchain_{a.dataset}_{a.model}_raw.csv"
    for split, seeds in (("tune", range(TUNE_OFFSET, TUNE_OFFSET + a.tune)), ("eval", range(a.eval))):
        sweep(a.dataset, seeds, split, rows, model=a.model, methods=METHODS, b_per_task=(1, 2, 4),
              etas=etas_for(a.dataset, a.model), loader=loader(a.dataset, split),
              on_seed=lambda c, sp, s, t: print(f"{c} {sp} seed={s} {t:.0f}s", flush=True))
        pd.DataFrame(rows).to_csv(out, index=False)
    summ = select_and_report(pd.DataFrame(rows), methods=METHODS)
    summ.to_csv(Path(a.out) / f"randchain_{a.dataset}_{a.model}_summary.csv", index=False)
    print(summ.pivot_table(index="b", columns="method", values="mean").to_string())


if __name__ == "__main__":
    main()
