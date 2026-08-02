"""The cascade itself: Algorithm 1, plus the training and evaluation it needs.

:func:`cascade_train` is the whole method. Train the root from a fresh
initialization for its allotted steps, then walk the tree from the root: each
task copies its parent's converged parameters and takes its own allotted steps
from there. Nothing else is shared, no task is revisited, and the total number
of gradient steps equals the sum of the allocation, which by construction is
exactly :math:`B`.

A task allotted zero steps is an exact copy of its parent. That is a legitimate
outcome under the ``exact`` regime, not a failure.
"""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Callable, Dict, Hashable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .trees import RootedTree

Array = np.ndarray
Node = Hashable

__all__ = ["train_steps", "cascade_train", "independent_train", "eval_task", "eval_all"]


def train_steps(model: nn.Module, X: Array, y: Array, steps: int, eta: float,
                task: str = "regression", l2: float = 0.0) -> nn.Module:
    """Full-batch gradient descent for ``steps`` steps, in place.

    One step is one gradient evaluation on the task's whole training set, which
    is the unit the budget counts. Squared error for regression, logistic loss
    for binary classification. Gradients are clipped to unit norm, and the
    ``l2`` term is added to the gradient rather than to the loss.

    Raises ``RuntimeError`` on a non-finite loss, so a diverging learning rate
    is reported rather than silently producing NaN predictions.
    """
    if steps <= 0:
        return model
    model.train()
    X_t = torch.from_numpy(np.asarray(X)).float()
    y_t = torch.from_numpy(np.asarray(y)).float().reshape(-1)
    crit = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    for _ in range(int(steps)):
        loss = crit(model(X_t), y_t)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss; the learning rate is too large")
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad if l2 == 0.0 else p.grad + l2 * p
                    p.add_(g, alpha=-eta)
    return model


def cascade_train(tree: RootedTree, X: Dict[Node, Array], y: Dict[Node, Array],
                  alloc: Mapping[Node, int], eta: float,
                  factory: Callable[[], nn.Module],
                  task: str = "regression", l2: float = 0.0) -> Dict[Node, nn.Module]:
    """Run the cascade and return one trained model per task.

    ``tree`` fixes who inherits from whom, ``alloc`` fixes how many steps each
    task gets. The traversal is breadth-first from the root; any order in which
    a parent precedes its children gives the same models, since a task's outcome
    depends only on its ancestors.
    """
    children: Dict[Node, list] = defaultdict(list)
    for u, v in tree.edges:
        children[u].append(v)

    root = tree.root
    models: Dict[Node, nn.Module] = {
        root: train_steps(factory(), X[root], y[root],
                          max(0, int(alloc.get(root, 0))), eta, task, l2)
    }
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            models[v] = train_steps(copy.deepcopy(models[u]), X[v], y[v],
                                    max(0, int(alloc.get(v, 0))), eta, task, l2)
            q.append(v)
    return models


def independent_train(nodes, X: Dict[Node, Array], y: Dict[Node, Array],
                      alloc: Mapping[Node, int], eta: float,
                      factory: Callable[[], nn.Module],
                      task: str = "regression", l2: float = 0.0) -> Dict[Node, nn.Module]:
    """The Individual baseline: every task from its own fresh initialization."""
    return {v: train_steps(factory(), X[v], y[v], max(0, int(alloc.get(v, 0))),
                           eta, task, l2)
            for v in nodes}


def eval_task(model: nn.Module, X: Array, y: Array, task: str = "regression",
              stats: Optional[Mapping[str, Array]] = None) -> float:
    """Score one task: RMSE for regression, accuracy for classification.

    When ``stats`` is given, regression predictions are mapped back to the
    original units before scoring, so the reported RMSE is in the units of the
    data and not of the normalized target.
    """
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.asarray(X)).float()).numpy().reshape(-1)
    y = np.asarray(y).reshape(-1)
    if task == "classification":
        return float(np.mean((out > 0) == (y > 0.5)))
    if stats is not None:
        out = out * stats["y_std"] + stats["y_mean"]
    return float(np.sqrt(np.mean((y - out.reshape(-1)) ** 2)))


def eval_all(models: Mapping[Node, nn.Module], X: Dict[Node, Array],
             y: Dict[Node, Array], task: str = "regression",
             stats: Optional[Mapping[Node, Mapping[str, Array]]] = None) -> float:
    """Mean per-task score over the collection, unweighted.

    Every task counts equally regardless of its sample count, matching how the
    tables are computed.
    """
    return float(np.mean([
        eval_task(models[v], X[v], y[v], task, stats[v] if stats else None)
        for v in models
    ]))
