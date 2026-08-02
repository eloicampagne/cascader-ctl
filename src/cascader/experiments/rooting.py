"""Four rooting rules compared at fixed topology.

The undirected edge set is built once; only the orientation changes. That
isolates what the root contributes from what the construction contributes.

The medoid is the rule the package ships and the paper reports. It ranks last of
the four. The two rules derived from the error surrogate, the weighted 1-median
and the weighted 1-centre, rank first and second, and on the unconstrained MST
the weighted 1-centre beats the medoid on every collection-by-budget cell by one
to three percent. The hop centre sits between: halving the depth buys almost
nothing, which is the same message as the path-length measurements -- tree
geometry drives the bound but not the outcome once the contraction assumption
fails.

The medoid stays the default anyway, and the reason is not accuracy. It is
computable before any tree exists, so it is identical across every compared
method. The two winning rules read distances along the tree, so each
construction would get a different root and the comparison would no longer be at
equal root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..allocation import sle_alloc, uniform_alloc
from ..cascade import cascade_train, eval_all
from ..config import get_config
from ..data import load
from ..methods import SLE_METHODS
from ..models import model_factory
from ..rooting import ROOT_RULES, select_root
from ..trees import CONSTRUCTIONS, root_tree
from ._common import Result, write

__all__ = ["run", "main"]

DEFAULT_METHODS = ("Chain", "CTL-kNN", "CTL-MST", "CTL-MSTc")


def run(collections: Sequence[str] = ("Syn-10", "WEAVE-UK"),
        methods: Sequence[str] = DEFAULT_METHODS,
        rules: Sequence[str] = ROOT_RULES, b_per_task: Sequence[int] = (1, 2, 4),
        reps: int = 5, model: str = "linear", out: str | Path = "results",
        stem: str = "rooting", regime: Optional[str] = None,
        quiet: bool = False) -> Result:
    """Re-root each construction under every rule and score the result."""
    rows: List[dict] = []
    for coll in collections:
        cfg = get_config(coll)
        eta = cfg.eta if model == "linear" else cfg.mlp_eta
        for rep in range(reps):
            tasks = load(coll, seed=rep)
            factory = model_factory(model, tasks.d)
            index = tasks.index
            for method in methods:
                built = CONSTRUCTIONS[method](
                    tasks.nodes, tasks.D, tasks.root,
                    rng=np.random.default_rng(rep), lambda_reach=cfg.mstc_lambda)
                undirected = [(built.parent[v], v) for v in built.nodes
                              if built.parent[v] is not None]
                for rule in rules:
                    root = select_root(rule, tasks.nodes, X=tasks.X_train,
                                       D=tasks.D, undirected=undirected)
                    tree = root_tree(undirected, root, tasks.nodes)
                    depths = tree.depths()
                    for b in b_per_task:
                        B = b * tasks.T
                        alloc = (sle_alloc(tree, tasks.D, index, B,
                                           root_frac=cfg.root_frac, regime=regime)
                                 if method in SLE_METHODS else
                                 uniform_alloc(tree.nodes, root, B,
                                               root_frac=cfg.b_proto_frac,
                                               regime=regime))
                        torch.manual_seed(rep)
                        try:
                            fitted = cascade_train(tree, tasks.X_train, tasks.y_train,
                                                   alloc, eta, factory, tasks.task,
                                                   cfg.reg)
                            value = eval_all(fitted, tasks.X_test, tasks.y_test,
                                             tasks.task, tasks.stats)
                        except RuntimeError:
                            value = np.nan
                        rows.append(dict(collection=coll, method=method, rule=rule,
                                         b=b, B=B, rep=rep, model=model,
                                         depth=max(depths.values()), value=value))
            if not quiet:
                print(f"  {coll} rep={rep} done", flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["collection", "method", "rule", "b"])
                 .agg(mean=("value", "mean"), depth=("depth", "mean"),
                      n=("value", "count")).reset_index())
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(df):
        print("\n=== mean rank of each rule over all cells (1 = best) ===")
        print(_mean_ranks(df).round(2).to_string())
    return result


def _mean_ranks(df: pd.DataFrame) -> pd.Series:
    """Rank the rules within each collection-method-budget cell, then average."""
    from ._common import HIGHER_IS_BETTER

    per_cell = (df.groupby(["collection", "method", "b", "rule"])["value"]
                  .mean().reset_index())
    ranks: dict = {}
    for (coll, method, b), g in per_cell.groupby(["collection", "method", "b"]):
        s = g.set_index("rule")["value"].dropna()
        if len(s) < 2:
            continue
        for rule, r in s.rank(ascending=coll not in HIGHER_IS_BETTER).items():
            ranks.setdefault(rule, []).append(float(r))
    return pd.Series({k: float(np.mean(v)) for k, v in ranks.items()}).sort_values()


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collections", default="Syn-10,WEAVE-UK")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--rules", default=",".join(ROOT_RULES))
    p.add_argument("--budgets", default="1,2,4")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--model", default="linear", choices=("linear", "mlp"))
    p.add_argument("--out", default="results")
    p.add_argument("--stem", default="rooting")
    p.add_argument("--regime", default=None, choices=("floor1", "exact"))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    run(collections=a.collections.split(","), methods=a.methods.split(","),
        rules=a.rules.split(","), b_per_task=[int(x) for x in a.budgets.split(",")],
        reps=a.reps, model=a.model, out=a.out, stem=a.stem, regime=a.regime,
        quiet=a.quiet)


if __name__ == "__main__":
    main()
