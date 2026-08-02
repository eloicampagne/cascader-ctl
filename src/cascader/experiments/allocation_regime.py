"""The two allocation regimes, side by side.

``exact`` apportions ``B`` over the SLE weights and lets a light subtree round to
zero steps. ``floor1`` refines every task at least once and apportions the
surplus. Both spend exactly ``B``.

The choice is not cosmetic. Under ``exact`` the tasks the weights starve are the
leaves, and a leaf that receives nothing is an exact copy of its parent, which is
where a cascade is supposed to earn its keep. On WEAVE-UK that erases the tree
advantage; on the generated collections it matters less, since the weights are
flatter. ``floor1`` is what the reported tables use, and this experiment is the
evidence for that choice rather than an assertion of it.

Also reported is the realized spend per regime and method, which is the
regression guard for the allocation bug: any number other than ``B`` is a defect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..allocation import individual_alloc
from ..cascade import eval_all
from ..config import get_config
from ..data import load
from ..methods import METHODS, allocate, build_structure, run_method
from ..models import model_factory
from ._common import Result, write

__all__ = ["run", "main"]


def run(collections: Sequence[str] = ("WEAVE-UK", "Syn-10"),
        methods: Sequence[str] = METHODS, b_per_task: Sequence[int] = (1, 2, 4),
        reps: int = 5, model: str = "linear", out: str | Path = "results",
        stem: str = "allocation_regime", quiet: bool = False) -> Result:
    """Run every method under both regimes and record score and realized spend."""
    rows: List[dict] = []
    for coll in collections:
        cfg = get_config(coll)
        eta = cfg.eta if model == "linear" else cfg.mlp_eta
        for rep in range(reps):
            tasks = load(coll, seed=rep)
            factory = model_factory(model, tasks.d)
            for b in b_per_task:
                B = b * tasks.T
                for regime in ("floor1", "exact"):
                    for method in methods:
                        rng = np.random.default_rng(10_000 + rep)
                        if method == "Individual":
                            alloc = individual_alloc(tasks.nodes, B)
                            zeros = sum(1 for v in alloc.values() if v == 0)
                        else:
                            tree = build_structure(method, tasks, cfg,
                                                   np.random.default_rng(10_000 + rep))
                            alloc = allocate(method, tree, tasks, B, cfg,
                                             regime=regime)
                            zeros = sum(1 for v in alloc.values() if v == 0)
                        torch.manual_seed(rep)
                        try:
                            fitted = run_method(method, tasks, B, eta, cfg, factory,
                                                rng=rng, regime=regime)
                            value = eval_all(fitted, tasks.X_test, tasks.y_test,
                                             tasks.task, tasks.stats)
                        except RuntimeError:
                            value = np.nan
                        rows.append(dict(collection=coll, method=method,
                                         regime=regime, b=b, B=B, rep=rep,
                                         model=model, spent=int(sum(alloc.values())),
                                         zero_budget_tasks=zeros, value=value))
            if not quiet:
                print(f"  {coll} rep={rep} done", flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["collection", "method", "regime", "b"])
                 .agg(mean=("value", "mean"), spent=("spent", "max"),
                      B=("B", "max"), zero_budget_tasks=("zero_budget_tasks", "mean"),
                      n=("value", "count")).reset_index())
    summary["overspend"] = summary["spent"] - summary["B"]
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(summary):
        bad = summary[summary.overspend != 0]
        print(f"\n=== budget check: {len(bad)} cell(s) not spending exactly B ===")
        if len(bad):
            print(bad.to_string(index=False))
        print("\n=== mean score by regime ===")
        print(summary.pivot_table(index=["collection", "method", "b"],
                                  columns="regime", values="mean")
              .round(4).to_string())
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collections", default="WEAVE-UK,Syn-10")
    p.add_argument("--budgets", default="1,2,4")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--model", default="linear", choices=("linear", "mlp"))
    p.add_argument("--out", default="results")
    p.add_argument("--stem", default="allocation_regime")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    run(collections=a.collections.split(","),
        b_per_task=[int(x) for x in a.budgets.split(",")], reps=a.reps,
        model=a.model, out=a.out, stem=a.stem, quiet=a.quiet)


if __name__ == "__main__":
    main()
