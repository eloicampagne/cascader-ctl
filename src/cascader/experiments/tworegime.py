"""Where the tree overtakes the chain.

A chain is a degenerate spanning tree whose total edge length is at least the
MST's. Whether that excess costs anything depends on the ratio of budget to task
distances: a budget of ``b`` steps caps how much parent-child distance an edge
can absorb, so an informed tree pays off exactly when the budget cannot absorb
the chain's detour.

The experiment varies the two quantities independently. Scatter ``s`` is the
ratio of between-cluster to within-cluster spread: the ``K`` centres are drawn at
scale ``s`` around a common vector and each task at unit scale from its centre,
so clusters overlap at ``s = 0.5`` and separate at ``s = 20``. Reported is
``(CTL-MST - Chain) / Chain`` in percent under the SLE allocation, a negative
entry meaning the tree wins.

The sign changes along one monotone boundary, and the budget at which the chain
takes over rises with the scatter. That boundary is checkable before training,
by comparing the per-task budget with the median pairwise distance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..cascade import eval_all
from ..config import get_config
from ..data.synthetic import syn_collection
from ..methods import run_method
from ..models import model_factory
from ._common import Result, write

__all__ = ["SCATTERS", "BUDGETS", "run", "main"]

SCATTERS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
BUDGETS = (200, 500, 1000, 2000)
TAU_WITHIN = 1.0


def run(scatters: Sequence[float] = SCATTERS, budgets: Sequence[int] = BUDGETS,
        reps: int = 10, T: int = 200, d: int = 5, n_clusters: int = 5,
        methods: Sequence[str] = ("Chain", "CTL-MST"), eta: float = 0.1,
        out: str | Path = "results", stem: str = "tworegime",
        regime: Optional[str] = None, quiet: bool = False) -> Result:
    """Sweep scatter against budget and report the tree-versus-chain gap."""
    cfg = get_config("Syn-10")
    rows: List[dict] = []
    for scatter in scatters:
        for rep in range(reps):
            tasks = syn_collection(seed=rep, T=T, d=d, n_clusters=n_clusters,
                                   tau_within=TAU_WITHIN, tau_between=scatter,
                                   sigma=0.5)
            factory = model_factory("linear", tasks.d)
            median_d = float(np.median(tasks.D[np.triu_indices(tasks.T, k=1)]))
            for B in budgets:
                for method in methods:
                    torch.manual_seed(rep)
                    rng = np.random.default_rng(10_000 + rep)
                    try:
                        fitted = run_method(method, tasks, B, eta, cfg, factory,
                                            rng=rng, regime=regime)
                        value = eval_all(fitted, tasks.X_test, tasks.y_test,
                                         tasks.task, tasks.stats)
                    except RuntimeError:
                        value = np.nan
                    rows.append(dict(scatter=scatter, B=B, b=B / tasks.T, rep=rep,
                                     method=method, value=value,
                                     median_distance=median_d))
        if not quiet:
            print(f"  scatter={scatter} done", flush=True)

    df = pd.DataFrame(rows)
    summary = _gap(df)
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(summary):
        print("\n=== (CTL-MST - Chain) / Chain, percent; negative = tree wins ===")
        print(summary.pivot_table(index="scatter", columns="B",
                                  values="gap_pct").round(2).to_string())
    return result


def _gap(df: pd.DataFrame) -> pd.DataFrame:
    """Paired relative gap between the tree and the chain, per cell."""
    out = []
    for (scatter, B), g in df.groupby(["scatter", "B"]):
        p = g.pivot_table(index="rep", columns="method", values="value")
        if not {"Chain", "CTL-MST"} <= set(p.columns):
            continue
        p = p.dropna()
        gap = (p["CTL-MST"] - p["Chain"]) / p["Chain"] * 100
        out.append(dict(scatter=scatter, B=B, gap_pct=float(gap.mean()),
                        se=float(gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1
                        else 0.0,
                        median_distance=float(g["median_distance"].mean()),
                        n=len(gap)))
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scatters", default=",".join(map(str, SCATTERS)))
    p.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--out", default="results")
    p.add_argument("--stem", default="tworegime")
    p.add_argument("--regime", default=None, choices=("floor1", "exact"))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    run(scatters=[float(x) for x in a.scatters.split(",")],
        budgets=[int(x) for x in a.budgets.split(",")], reps=a.reps, T=a.T,
        out=a.out, stem=a.stem, regime=a.regime, quiet=a.quiet)


if __name__ == "__main__":
    main()
