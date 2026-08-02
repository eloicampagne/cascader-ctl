"""Shared machinery for the packaged reproductions.

Every experiment produces one tidy dataframe with one row per
``(collection, budget, method, seed, learning rate, split)`` and writes it
beside a summary. Keeping the raw rows means a report can be recomputed without
re-running anything, which is what turned the allocation bug into a two-minute
re-analysis rather than a two-day re-run.

Two conventions are load-bearing.

Budgets are expressed **per task**. ``b = B / T`` is comparable across
collections, absolute budgets are not, and on CelebA the realized task count
varies with the seed, so a fixed absolute ``B`` would silently vary the budget
each task receives.

Learning rates are selected on **disjoint tuning seeds**, offset by
:data:`TUNE_OFFSET` from the evaluation seeds, and never on the split a number
is reported from. On the generated collections a tuning seed is a fresh draw of
tasks. WEAVE-UK has no such freedom -- its panel is fixed -- so its rates are
chosen on a validation split held out of the training rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..cascade import eval_all
from ..config import get_config
from ..data import load
from ..methods import METHODS
from ..models import model_factory

__all__ = ["ETA_GRID", "B_PER_TASK", "TUNE_OFFSET", "HIGHER_IS_BETTER",
           "etas_for", "evaluate", "sweep", "select_and_report", "write",
           "add_common_args", "Result"]

#: Learning-rate grid, spanning four orders of magnitude. The published rate for
#: the collection is added to it, so the published setting is always reachable.
ETA_GRID = (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5)

#: Per-task budgets. One to four steps per task is the tight regime the method
#: is about; beyond it the constructions converge on each other.
B_PER_TASK = (1, 2, 4)

#: Evaluation seeds are 0.., tuning seeds are TUNE_OFFSET.. -- disjoint by
#: construction, so no task draw is ever shared between the two.
TUNE_OFFSET = 500

#: Collections scored by accuracy rather than by error.
HIGHER_IS_BETTER = frozenset({"FMNIST", "CIFAR", "CelebA"})


@dataclass
class Result:
    """Where an experiment wrote its rows."""

    raw: Path
    summary: Optional[Path] = None
    frame: Optional[pd.DataFrame] = None


def etas_for(collection: str, model: str = "linear") -> tuple:
    """The grid, augmented with the published rate for this collection and model."""
    cfg = get_config(collection)
    published = cfg.eta if model == "linear" else cfg.mlp_eta
    return tuple(sorted(set(ETA_GRID) | {round(float(published), 10)}))


def evaluate(models, tasks) -> float:
    """Mean test score of a fitted collection, in the units the tables report."""
    return eval_all(models, tasks.X_test, tasks.y_test, tasks.task, tasks.stats)


def sweep(collection: str, seeds: Iterable[int], split: str, rows: List[dict],
          *, model: str = "linear", methods: Sequence[str] = METHODS,
          b_per_task: Sequence[int] = B_PER_TASK,
          etas: Optional[Sequence[float]] = None,
          loader: Optional[Callable] = None,
          regime: Optional[str] = None,
          on_seed: Optional[Callable[[str, str, int, float], None]] = None) -> None:
    """Run every ``(budget, rate, method)`` cell on each seed, appending to ``rows``.

    The task draw, the tree and the distance matrix are computed once per seed
    and reused across budgets and rates, so the comparison at a given seed is
    free of draw-to-draw noise. ``torch.manual_seed`` is reset per cell, so two
    methods at the same seed start from the same initialization.
    """
    from ..methods import run_method

    cfg = get_config(collection)
    etas = etas_for(collection, model) if etas is None else tuple(etas)
    loader = loader or (lambda s: load(collection, seed=s))

    for seed in seeds:
        t0 = time.perf_counter()
        tasks = loader(seed)
        factory = model_factory(model, tasks.d)
        for b in b_per_task:
            B = b * tasks.T          # per task, so CelebA's variable T is handled
            for eta in etas:
                rng = np.random.default_rng(10_000 + seed)
                for method in methods:
                    torch.manual_seed(seed)
                    try:
                        fitted = run_method(method, tasks, B, eta, cfg, factory,
                                            rng=rng, regime=regime)
                        value = evaluate(fitted, tasks)
                        if not np.isfinite(value):
                            value = np.nan
                    except RuntimeError:      # diverged at this rate
                        value = np.nan
                    rows.append(dict(collection=collection, model=model, T=tasks.T,
                                     B=B, b=b, seed=seed, eta=eta, method=method,
                                     split=split, value=value))
        if on_seed is not None:
            on_seed(collection, split, seed, time.perf_counter() - t0)


def select_and_report(df: pd.DataFrame,
                      methods: Sequence[str] = METHODS) -> pd.DataFrame:
    """Pick each method's rate on the tuning split, then score it on evaluation.

    Grouping is on the per-task budget ``b`` rather than on ``B``: on CelebA the
    task count varies by seed, so two seeds at the same ``b`` have different
    ``B``. Returns mean, standard error and the selected rate per cell.
    """
    tune = df[df.split == "tune"]
    ev = df[df.split == "eval"]
    out = []
    for (coll, b, model), g in ev.groupby(["collection", "b", "model"]):
        higher = coll in HIGHER_IS_BETTER
        for method in methods:
            t = tune[(tune.collection == coll) & (tune.b == b)
                     & (tune.model == model) & (tune.method == method)]
            scores = t.groupby("eta")["value"].mean().dropna()
            if scores.empty:
                continue
            eta = scores.idxmax() if higher else scores.idxmin()
            v = g[(g.method == method) & (g.eta == eta)].sort_values("seed")["value"]
            v = v.dropna().to_numpy()
            if v.size == 0:
                continue
            out.append(dict(collection=coll, model=model, b=b, method=method,
                            eta=eta, mean=v.mean(),
                            se=v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0,
                            n=v.size))
    return pd.DataFrame(out)


def write(rows: List[dict], out_dir: Path, stem: str,
          summary: Optional[pd.DataFrame] = None) -> Result:
    """Write the raw rows, and the summary when one was computed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{stem}_raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    summary_path = None
    if summary is not None and len(summary):
        summary_path = out_dir / f"{stem}_summary.csv"
        summary.to_csv(summary_path, index=False)
    return Result(raw=raw, summary=summary_path, frame=summary)


def add_common_args(parser) -> None:
    """Arguments every packaged experiment accepts."""
    parser.add_argument("--collections", default="Syn-10,WEAVE-UK,FMNIST,CIFAR",
                        help="comma-separated collection names")
    parser.add_argument("--model", default="linear", choices=("linear", "mlp"),
                        help="local model class")
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--tune-seeds", type=int, default=5)
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--regime", default=None, choices=("floor1", "exact"),
                        help="allocation regime; defaults to CASCADER_ALLOC")
    parser.add_argument("--quiet", action="store_true")


def progress(quiet: bool) -> Callable:
    """A per-seed progress callback, or a no-op."""
    if quiet:
        return lambda *a: None

    def report(collection: str, split: str, seed: int, secs: float) -> None:
        print(f"  {collection} {split} seed={seed} in {secs:.0f}s", flush=True)

    return report
