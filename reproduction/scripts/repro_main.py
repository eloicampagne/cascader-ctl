"""Reproduction of the main tables (linear and MLP) with the current package.

The seven methods of cascader.methods.METHODS under the protocol of the
paper: per-method rate selected on tuning seeds 500..504, evaluation on seeds
0..19, floor-one SLE allocation. WEAVE-UK selects its rate on the 48-row
validation slice, which cascader.experiments.main_table does not do.

    CASCADER_DATA_DIR=<data_links> python3.12 repro_main.py --dataset Syn --model mlp
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


METHODS = list(CM.METHODS)


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
    out = Path(a.out) / f"repro_main_{a.dataset}_{a.model}_raw.csv"
    for split, seeds in (("tune", range(TUNE_OFFSET, TUNE_OFFSET + a.tune)), ("eval", range(a.eval))):
        sweep(a.dataset, seeds, split, rows, model=a.model, methods=METHODS, b_per_task=(1, 2, 4),
              etas=etas_for(a.dataset, a.model), loader=loader(a.dataset, split),
              on_seed=lambda c, sp, s, t: print(f"{c} {sp} seed={s} {t:.0f}s", flush=True))
        pd.DataFrame(rows).to_csv(out, index=False)
    summ = select_and_report(pd.DataFrame(rows), methods=METHODS)
    summ.to_csv(Path(a.out) / f"repro_main_{a.dataset}_{a.model}_summary.csv", index=False)
    print(summ.pivot_table(index="b", columns="method", values="mean").to_string())


if __name__ == "__main__":
    main()
