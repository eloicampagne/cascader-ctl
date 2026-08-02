"""Tasks arriving one at a time.

The offline method assumes the collection is known: the distance matrix is
``O(T^2)``, the tree is built globally, and SLE splits a total budget. In
deployment none of the three is available.

The online variant drops all of them. When task ``v`` arrives it is attached to
one already-trained task, copies its parameters, and takes ``b = B / T`` steps.
No coverage weight, no allocation, no knowledge of what comes next, and the
distance matrix is never formed -- only ``v``'s column against the tasks already
seen.

This is not Prim. Prim chooses both which task to attach and to whom; here the
order is imposed and only the parent is chosen. The gap between the two measures
what ignorance of the future costs.

Four arms, by decreasing knowledge:

``Offline-MSTc``
    the global tree with SLE allocation, everything known in advance;
``Online-NN``
    imposed order, parent = the nearest already-trained task under the proxy;
``Online-grad``
    imposed order, parent = the task whose model gives the smallest gradient
    norm at ``v``, normalized by ``v``'s cold-start gradient. No proxy distance
    is formed at all: the score is read at the candidate parent's actual point;
``Online-prev``
    parent = the previous task in arrival order, distance ignored;
``Individual``
    cold start, ``b`` steps per task.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..cascade import eval_all, train_steps
from ..config import get_config
from ..data import load
from ..methods import run_method
from ..models import model_factory
from ._common import Result, write

__all__ = ["ARMS", "gradient_norm", "online_cascade", "run", "main"]

ARMS = ("Offline-MSTc", "Online-grad", "Online-NN", "Online-prev", "Individual")


def gradient_norm(model: nn.Module, X, y, task: str) -> float:
    """Norm of the loss gradient of task ``(X, y)`` at ``model``'s parameters.

    How badly a candidate parent fits the arriving task, read at the parent's
    actual point rather than through a data proxy.
    """
    crit = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    Xt = torch.from_numpy(np.asarray(X)).float()
    yt = torch.from_numpy(np.asarray(y)).float().reshape(-1)
    model.zero_grad(set_to_none=True)
    crit(model(Xt), yt).backward()
    total = sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                if p.grad is not None)
    return float(np.sqrt(total))


def online_cascade(order, tasks, b: int, eta: float, factory, rule: str = "nn",
                   cold: Optional[Dict] = None, l2: float = 0.0):
    """Attach each arriving task to one already-trained task, then refine.

    ``rule`` is ``'prev'``, ``'nn'`` or ``'grad'``. Returns the fitted models and
    the edges the arrivals induced, which is the tree the process built without
    ever planning it.
    """
    index = tasks.index
    models: Dict = {}
    edges: List = []
    seen: List = []
    for v in order:
        if not seen:
            models[v] = train_steps(factory(), tasks.X_train[v], tasks.y_train[v],
                                    b, eta, tasks.task, l2)
        else:
            if rule == "prev":
                u = seen[-1]
            elif rule == "grad":
                scores = [gradient_norm(models[w], tasks.X_train[v],
                                        tasks.y_train[v], tasks.task) / cold[v]
                          for w in seen]
                u = seen[int(np.argmin(scores))]
            else:
                col = tasks.D[index[v], [index[w] for w in seen]]
                u = seen[int(np.argmin(col))]
            edges.append((u, v))
            models[v] = train_steps(copy.deepcopy(models[u]), tasks.X_train[v],
                                    tasks.y_train[v], b, eta, tasks.task, l2)
        seen.append(v)
    return models, edges


def run(collections: Sequence[str] = ("Syn-10", "CIFAR"), model: str = "linear",
        reps: int = 5, b_per_task: Sequence[int] = (1, 2, 4),
        out: str | Path = "results", stem: str = "online",
        regime: Optional[str] = None, quiet: bool = False) -> Result:
    """Compare the online arms against the offline cascade at equal total budget."""
    rows: List[dict] = []
    for coll in collections:
        cfg = get_config(coll)
        eta = cfg.eta if model == "linear" else cfg.mlp_eta
        for rep in range(reps):
            tasks = load(coll, seed=rep)
            factory = model_factory(model, tasks.d)
            arrival = list(np.random.default_rng(1000 + rep).permutation(tasks.T))
            order = [tasks.nodes[i] for i in arrival]
            cold = {v: gradient_norm(factory(), tasks.X_train[v], tasks.y_train[v],
                                     tasks.task) + 1e-12 for v in tasks.nodes}
            for b in b_per_task:
                B = b * tasks.T
                for arm in ARMS:
                    torch.manual_seed(rep)
                    try:
                        if arm in ("Offline-MSTc", "Individual"):
                            name = "CTL-MSTc" if arm == "Offline-MSTc" else "Individual"
                            fitted = run_method(name, tasks, B, eta, cfg, factory,
                                                rng=np.random.default_rng(rep),
                                                regime=regime)
                        else:
                            rule = {"Online-grad": "grad", "Online-NN": "nn",
                                    "Online-prev": "prev"}[arm]
                            fitted, _ = online_cascade(order, tasks, b, eta, factory,
                                                       rule, cold, cfg.reg)
                        value = eval_all(fitted, tasks.X_test, tasks.y_test,
                                         tasks.task, tasks.stats)
                    except RuntimeError:
                        value = np.nan
                    rows.append(dict(collection=coll, model=model, b=b, B=B,
                                     rep=rep, arm=arm, value=value))
            if not quiet:
                print(f"  {coll} rep={rep} done", flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["collection", "model", "b", "arm"])["value"]
                 .agg(["mean", "std", "count"]).reset_index())
    result = write(rows, Path(out), stem, summary)
    if not quiet and len(summary):
        print("\n=== mean score by arm ===")
        print(summary.pivot_table(index=["collection", "b"], columns="arm",
                                  values="mean").round(4).to_string())
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--collections", default="Syn-10,CIFAR")
    p.add_argument("--model", default="linear", choices=("linear", "mlp"))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--budgets", default="1,2,4")
    p.add_argument("--out", default="results")
    p.add_argument("--stem", default="online")
    p.add_argument("--regime", default=None, choices=("floor1", "exact"))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    run(collections=a.collections.split(","), model=a.model, reps=a.reps,
        b_per_task=[int(x) for x in a.budgets.split(",")], out=a.out,
        stem=a.stem, regime=a.regime, quiet=a.quiet)


if __name__ == "__main__":
    main()
