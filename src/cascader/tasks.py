"""The task collection every method consumes.

A collection is a set of tasks over a shared feature space, each with its own
training and test split, plus the pairwise distance matrix and the root the
cascade starts from. Data loaders return one of these; methods take one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Mapping, Optional

import numpy as np

Array = np.ndarray
Node = Hashable

__all__ = ["TaskCollection"]


@dataclass
class TaskCollection:
    """Tasks, splits, geometry and root.

    Attributes
    ----------
    nodes
        Ordered task identifiers. The order indexes ``D`` and fixes the
        remainder order of :func:`cascader.allocation.individual_alloc`, so it
        must stay stable within a run.
    X_train, y_train, X_test, y_test
        Per-task arrays. Training features are what the distance and the tree
        are computed from; nothing reads the test split before evaluation.
    D
        Symmetric ``(T, T)`` distance matrix aligned with ``nodes``.
    root
        The task the cascade starts from.
    task
        ``'regression'`` or ``'classification'``.
    stats
        Per-task normalization statistics, when the targets were normalized.
        Evaluation maps predictions back through these, so regression scores are
        in the units of the data.
    name
        Collection label, used in reports.
    """

    nodes: List[Node]
    X_train: Dict[Node, Array]
    y_train: Dict[Node, Array]
    X_test: Dict[Node, Array]
    y_test: Dict[Node, Array]
    D: Array
    root: Node
    task: str = "regression"
    stats: Optional[Dict[Node, Mapping[str, Array]]] = None
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        """Number of tasks."""
        return len(self.nodes)

    @property
    def d(self) -> int:
        """Feature dimension."""
        return int(np.asarray(self.X_train[self.nodes[0]]).shape[1])

    @property
    def index(self) -> Dict[Node, int]:
        """Task identifier to row/column index in ``D``."""
        return {v: i for i, v in enumerate(self.nodes)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"TaskCollection(name={self.name!r}, T={self.T}, d={self.d}, "
                f"task={self.task!r}, root={self.root!r})")
