"""How much of the budget the root should get.

The root is the only task trained from scratch; everything else inherits. At a
tight budget it can end up with a single step, in which case the cascade
propagates a barely-trained model. The share reserved for it is a free
parameter, fixed at 0.10 for the SLE allocation and at ``b_proto_frac`` for the
uniform one, and neither value was ever swept.

This sweeps it, for every method at once. For each fraction ``f``, the root gets
``round(f * B)`` and the rest is split over the remaining tasks -- by the SLE
weights for the cascades, uniformly for Star. No floor is imposed: at large
``f`` some tasks receive zero steps and are exact copies of their parent. At
``f = 1`` the root consumes everything, and Star degenerates to "train one
source, copy it everywhere".

Sweeping every method rather than only Star matters. Comparing a tuned Star
against a cascade pinned at 0.10 would be the same bias in reverse.

What it shows: WEAVE-UK has interior optima, and the gap between Star at its
best fraction and the best cascade at its own best fraction survives. The
generated and image collections are optimal at ``f = 0`` or ``0.05``, and the
ranking does not move. Robustness to the root's share is a WEAVE-UK statement,
not a general one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..allocation import apportion, sle_weights
from ..cascade import cascade_train, eval_all
from ..config import get_config
from ..data import load
from ..methods import SLE_METHODS
from ..models import model_factory
from ..trees import CONSTRUCTIONS
from ._common import Result, etas_for, write

__all__ = ["FRACTIONS", "alloc_with_fraction", "run", "main"]

FRACTIONS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0)
DEFAULT_METHODS = ("Star", "Chain", "CTL-MST", "CTL-MSTc")


def alloc_with_fraction(tree, D, index, B: int, f: float,
                        weighted: bool = True) -> Dict:
    """Give the root ``round(f * B)``, split the rest, sum exactly ``B``.

    ``weighted=True`` splits by the SLE weights, ``False`` splits uniformly.
    """
    others = [v for v in tree.nodes if v != tree.root]
    if not others:
        return {tree.root: int(B)}
    if f >= 1.0:
        return {tree.root: int(B), **{v: 0 for v in others}}
    b_root = int(round(f * B))
    rest = B - b_root
    if weighted:
        w = sle_weights(tree, D, index)
        shares = apportion(np.array([w[v] for v in others]), rest)
    else:
        shares = apportion(np.ones(len(others)), rest)
    alloc = {v: int(shares[i]) for i, v in enumerate(others)}
    alloc[tree.root] = b_root
    return alloc


def run(collections: Sequence[str] = ("WEAVE-UK", "Syn-10"),
        methods: Sequence[str] = DEFAULT_METHODS,
        fractions: Sequence[float] = FRACTIONS,
        b_per_task: Sequence[int] = (1, 2, 4), eval_seeds: int = 20,
        tune_seeds: int = 5, model: str = "linear", out: str | Path = "results",
        stem: str = "root_fraction", quiet: bool = False) -> Result:
    """Sweep the root's share, selecting the learning rate for every cell."""
    rows: List[dict] = []
    for coll in collections:
        cfg = get_config(coll)
        etas = etas_for(coll, model)
        for split, seeds in (("tune", range(500, 500 + tune_seeds)),
                             ("eval", range(eval_seeds))):
            for seed in seeds:
                tasks = load(coll, seed=seed)
                factory = model_factory(model, tasks.d)
                index = tasks.index
                trees = {m: CONSTRUCTIONS[m](
                    tasks.nodes, tasks.D, tasks.root,
                    rng=np.random.default_rng(seed), lambda_reach=cfg.mstc_lambda)
                    for m in methods}
                for b in b_per_task:
                    B = b * tasks.T
                    for method in methods:
                        tree = trees[method]
                        weighted = method in SLE_METHODS
                        for f in fractions:
                            alloc = alloc_with_fraction(tree, tasks.D, index, B, f,
                                                        weighted)
                            assert sum(alloc.values()) == B
                            for eta in etas:
                                torch.manual_seed(seed)
                                try:
                                    fitted = cascade_train(
                                        tree, tasks.X_train, tasks.y_train, alloc,
                                        eta, factory, tasks.task, cfg.reg)
                                    value = eval_all(fitted, tasks.X_test,
                                                     tasks.y_test, tasks.task,
                                                     tasks.stats)
                                except RuntimeError:
                                    value = np.nan
                                rows.append(dict(collection=coll, method=method,
                                                 frac=f, b=b, B=B, seed=seed,
                                                 eta=eta, split=split, model=model,
                                                 value=value))
                if not quiet:
                    print(f"  {coll} {split} seed={seed} done", flush=True)

    df = pd.DataFrame(rows)
    summary = _select(df)
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(summary):
        print("\n=== best fraction per method and budget ===")
        print(summary.loc[summary.groupby(["collection", "method", "b"])["mean"]
                          .idxmin()][["collection", "method", "b", "frac", "mean"]]
              .to_string(index=False))
    return result


def _select(df: pd.DataFrame) -> pd.DataFrame:
    """Choose each cell's learning rate on the tuning split, score on evaluation."""
    from ._common import HIGHER_IS_BETTER

    tune, ev = df[df.split == "tune"], df[df.split == "eval"]
    out = []
    for (coll, method, b, f), g in ev.groupby(["collection", "method", "b", "frac"]):
        t = tune[(tune.collection == coll) & (tune.method == method)
                 & (tune.b == b) & (tune.frac == f)]
        scores = t.groupby("eta")["value"].mean().dropna()
        if scores.empty:
            continue
        eta = scores.idxmax() if coll in HIGHER_IS_BETTER else scores.idxmin()
        v = g[g.eta == eta]["value"].dropna().to_numpy()
        if v.size == 0:
            continue
        out.append(dict(collection=coll, method=method, b=b, frac=f, eta=eta,
                        mean=v.mean(),
                        se=v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0,
                        n=v.size))
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collections", default="WEAVE-UK,Syn-10")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    p.add_argument("--budgets", default="1,2,4")
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--tune-seeds", type=int, default=5)
    p.add_argument("--model", default="linear", choices=("linear", "mlp"))
    p.add_argument("--out", default="results")
    p.add_argument("--stem", default="root_fraction")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    run(collections=a.collections.split(","), methods=a.methods.split(","),
        fractions=[float(x) for x in a.fractions.split(",")],
        b_per_task=[int(x) for x in a.budgets.split(",")],
        eval_seeds=a.eval_seeds, tune_seeds=a.tune_seeds, model=a.model,
        out=a.out, stem=a.stem, quiet=a.quiet)


if __name__ == "__main__":
    main()
