"""Scaling in the number of tasks.

Two envelopes, and they answer different questions.

``fixed-b`` holds the per-task budget constant, so ``B = b * T`` grows with the
collection. Each task always gets the same refinement; what changes is how much
structure there is to exploit.

``fixed-B`` holds the total budget constant, so the per-task budget shrinks as
``1/T``. This is the regime the method is for -- more tasks, no more compute --
and it is where the gain against Individual grows with ``T``.

The construction cost is the thing to watch. CTL-MSTc's greedy coverage term
recomputes a ``|remaining| x |remaining|`` gain at every step, so it grows near
cubically in ``T`` and becomes the bottleneck well before training does. The
timings are recorded per cell so the trade-off is visible rather than asserted.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..cascade import cascade_train, eval_all, independent_train
from ..config import get_config
from ..data.synthetic import syn_collection
from ..methods import allocate, build_structure
from ..allocation import individual_alloc
from ..models import model_factory
from ._common import Result, write

__all__ = ["run", "main"]

T_LIST = (50, 100, 200, 500)
ARMS = ("Individual", "Star", "Chain", "CTL-MST", "CTL-MSTc")


def run(Ts: Sequence[int] = T_LIST, reps: int = 5, envelope: str = "fixed-b",
        b_per_task: int = 4, B_total: int = 2000, model: str = "linear",
        methods: Sequence[str] = ARMS, eta: float = 0.1,
        out: str | Path = "results", stem: str = "scaling",
        regime: Optional[str] = None, quiet: bool = False) -> Result:
    """Sweep the task count under a fixed per-task or fixed total budget."""
    cfg = get_config("Syn-10")
    rows: List[dict] = []
    for T in Ts:
        B = b_per_task * T if envelope == "fixed-b" else B_total
        for rep in range(reps):
            tasks = syn_collection(seed=rep, T=T)
            factory = model_factory(model, tasks.d)
            for method in methods:
                torch.manual_seed(rep)
                rng = np.random.default_rng(10_000 + rep)
                t0 = time.perf_counter()
                try:
                    if method == "Individual":
                        alloc = individual_alloc(tasks.nodes, B)
                        build_s = 0.0
                        fitted = independent_train(tasks.nodes, tasks.X_train,
                                                   tasks.y_train, alloc, eta,
                                                   factory, tasks.task, cfg.reg)
                    else:
                        tree = build_structure(method, tasks, cfg, rng)
                        build_s = time.perf_counter() - t0
                        alloc = allocate(method, tree, tasks, B, cfg, regime=regime)
                        fitted = cascade_train(tree, tasks.X_train, tasks.y_train,
                                               alloc, eta, factory, tasks.task,
                                               cfg.reg)
                    value = eval_all(fitted, tasks.X_test, tasks.y_test,
                                     tasks.task, tasks.stats)
                except RuntimeError:
                    build_s, value = float("nan"), float("nan")
                rows.append(dict(T=T, B=B, b=B / T, envelope=envelope, rep=rep,
                                 method=method, value=value,
                                 build_seconds=build_s,
                                 total_seconds=time.perf_counter() - t0))
        if not quiet:
            print(f"  T={T} done", flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["envelope", "T", "b", "method"])
                 .agg(mean=("value", "mean"), std=("value", "std"),
                      build_seconds=("build_seconds", "mean"),
                      n=("value", "count")).reset_index())
    summary["gain_vs_individual_pct"] = summary.apply(
        lambda r: _gain(summary, r), axis=1)
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(summary):
        print("\n=== mean MSE-equivalent score ===")
        print(summary.pivot_table(index="T", columns="method",
                                  values="mean").round(4).to_string())
        print("\n=== construction seconds ===")
        print(summary.pivot_table(index="T", columns="method",
                                  values="build_seconds").round(3).to_string())
    return result


def _gain(summary: pd.DataFrame, row) -> float:
    """Relative improvement over Individual in the same cell, in percent."""
    same_cell = ((summary["T"] == row["T"])
                 & (summary["envelope"] == row["envelope"])
                 & (summary["method"] == "Individual"))
    base = summary.loc[same_cell, "mean"]
    if base.empty or not np.isfinite(row["mean"]) or float(base.iloc[0]) == 0.0:
        return float("nan")
    return float((float(base.iloc[0]) - row["mean"]) / float(base.iloc[0]) * 100)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--Ts", default=",".join(map(str, T_LIST)))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--envelope", default="fixed-b", choices=("fixed-b", "fixed-B"))
    p.add_argument("--b-per-task", type=int, default=4)
    p.add_argument("--B-total", type=int, default=2000)
    p.add_argument("--model", default="linear", choices=("linear", "mlp"))
    p.add_argument("--out", default="results")
    p.add_argument("--stem", default="scaling")
    p.add_argument("--regime", default=None, choices=("floor1", "exact"))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    run(Ts=[int(x) for x in a.Ts.split(",")], reps=a.reps, envelope=a.envelope,
        b_per_task=a.b_per_task, B_total=a.B_total, model=a.model, out=a.out,
        stem=a.stem, regime=a.regime, quiet=a.quiet)


if __name__ == "__main__":
    main()
