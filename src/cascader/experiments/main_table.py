"""The main table: seven methods, five collections, three per-task budgets.

Two tuning protocols, selected by ``--eta-mode``.

``per-method`` is the protocol of the paper's tables. Each
``(collection, budget, method)`` gets its own rate, selected on tuning seeds
disjoint from the evaluation seeds.

``shared`` applies one rate per collection, taken from :mod:`cascader.config`,
to every method. That rate was chosen by an Optuna search on a CTL-MSTc
cascade.

Run::

    cascader main-table --collections Syn-10 --eval-seeds 5
    cascader main-table --eta-mode per-method --model mlp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import get_config
from ..methods import METHODS
from ._common import (B_PER_TASK, HIGHER_IS_BETTER, Result, TUNE_OFFSET,
                      add_common_args, etas_for, progress, select_and_report,
                      sweep, write)

__all__ = ["run", "mean_ranks", "main"]


def run(collections: Sequence[str] = ("Syn-10", "WEAVE-UK", "FMNIST", "CIFAR"),
        model: str = "linear", eta_mode: str = "per-method",
        eval_seeds: int = 20, tune_seeds: int = 5,
        b_per_task: Sequence[int] = B_PER_TASK,
        out: str | Path = "results", stem: str = "main_table",
        regime: Optional[str] = None, quiet: bool = False) -> Result:
    """Run the comparison and return the paths it wrote."""
    rows: List[dict] = []
    cb = progress(quiet)
    for coll in collections:
        cfg = get_config(coll)
        if eta_mode == "shared":
            etas = (cfg.eta if model == "linear" else cfg.mlp_eta,)
        else:
            etas = etas_for(coll, model)
        if not quiet:
            print(f"=== {coll}: {len(etas)} rate(s), b={tuple(b_per_task)}, "
                  f"{tune_seeds} tuning + {eval_seeds} evaluation seeds ===",
                  flush=True)
        sweep(coll, range(TUNE_OFFSET, TUNE_OFFSET + tune_seeds), "tune", rows,
              model=model, b_per_task=b_per_task, etas=etas, regime=regime,
              on_seed=cb)
        sweep(coll, range(eval_seeds), "eval", rows, model=model,
              b_per_task=b_per_task, etas=etas, regime=regime, on_seed=cb)

    df = pd.DataFrame(rows)
    summary = select_and_report(df)
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(summary):
        print("\n=== mean score at the selected rate ===")
        print(summary.pivot_table(index=["collection", "b"], columns="method",
                                  values="mean").round(4).to_string())
        print("\n=== mean rank ===")
        print(mean_ranks(summary).round(2).to_string())
    return result


def mean_ranks(summary: pd.DataFrame,
               methods: Sequence[str] = METHODS) -> pd.Series:
    """Average rank of each method over the ``(collection, budget)`` cells.

    Rank 1 is best, and the sign of "best" follows the metric: accuracy is
    maximized on the image collections, error minimized elsewhere. A method
    absent from a cell is skipped rather than imputed.
    """
    ranks = {m: [] for m in methods}
    for (coll, b, _model), g in summary.groupby(["collection", "b", "model"]):
        s = g.set_index("method")["mean"]
        s = s.reindex([m for m in methods if m in s.index])
        if len(s) < 2:
            continue
        order = s.rank(ascending=coll not in HIGHER_IS_BETTER)
        for m, r in order.items():
            ranks[m].append(float(r))
    return pd.Series({m: float(np.mean(v)) for m, v in ranks.items() if v}
                     ).sort_values()


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(p)
    p.add_argument("--eta-mode", default="per-method",
                   choices=("per-method", "shared"))
    p.add_argument("--budgets", default=",".join(map(str, B_PER_TASK)),
                   help="comma-separated per-task budgets")
    p.add_argument("--stem", default="main_table")
    a = p.parse_args(argv)
    run(collections=a.collections.split(","), model=a.model, eta_mode=a.eta_mode,
        eval_seeds=a.eval_seeds, tune_seeds=a.tune_seeds,
        b_per_task=[int(x) for x in a.budgets.split(",")],
        out=a.out, stem=a.stem, regime=a.regime, quiet=a.quiet)


if __name__ == "__main__":
    main()
