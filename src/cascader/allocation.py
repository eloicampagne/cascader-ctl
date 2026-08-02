"""Splitting the global budget across tasks.

The budget :math:`B` counts refinement steps in total, not per task. Every
allocator in this module returns a dict whose values are nonnegative integers
summing to **exactly** :math:`B`. That invariant is the contract between the
budget the paper reports and the compute a method actually consumes, and
:mod:`tests.test_allocation` checks it on every allocator, every regime and
every budget.

Two regimes, both exact:

``exact``
    Pure largest-remainder apportionment of :math:`B` over the SLE weights. A
    task whose share rounds to zero receives no refinement and inherits its
    parent unchanged. Faithful to the weights, harsh on light subtrees.

``floor1`` (default)
    Every task is refined at least once, and the surplus :math:`B - T` is
    apportioned over the weights. Requires :math:`B \\ge T` and falls back to
    ``exact`` below that.

The distinction is not cosmetic: under ``exact`` the tree advantage on WEAVE-UK
disappears, because the tasks the SLE weights starve are exactly the leaves the
cascade relies on. ``floor1`` is what the reported tables use.

A historical note kept deliberately. An earlier implementation applied the
``max(., 1)`` floor *after* apportionment, which overspent the budget by the
number of zero-weight tasks -- 74% of :math:`B` for the MST on WEAVE-UK at
:math:`B = T`. Any method whose weights concentrate was silently handed more
compute than the budget it was credited with. The regression test for this is
:func:`tests.test_allocation.test_floor_is_applied_before_apportionment`.
"""

from __future__ import annotations

import os
from typing import Dict, Hashable, Mapping, Optional, Sequence

import numpy as np

from .trees import RootedTree

Array = np.ndarray
Node = Hashable

__all__ = [
    "apportion",
    "sle_weights",
    "alloc_from_weights",
    "sle_alloc",
    "uniform_alloc",
    "individual_alloc",
    "default_regime",
    "REGIMES",
]

REGIMES = ("floor1", "exact")


def default_regime() -> str:
    """Allocation regime, overridable by the ``CASCADER_ALLOC`` variable."""
    regime = os.environ.get("CASCADER_ALLOC", "floor1")
    if regime not in REGIMES:
        raise ValueError(f"CASCADER_ALLOC must be one of {REGIMES}, got {regime!r}")
    return regime


def apportion(weights: Array, total: int) -> Array:
    """Largest-remainder apportionment of ``total`` over ``weights``.

    Returns nonnegative integers summing to exactly ``total``. Each entry gets
    the floor of its exact share, then the ``total - sum(floors)`` largest
    fractional remainders each get one more. Uniform weights are used when the
    given ones are all zero.
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if total <= 0:
        return np.zeros(len(w), dtype=int)
    if w.sum() < 1e-12:
        w = np.ones(len(w), dtype=float)
    exact = w / w.sum() * total
    base = np.floor(exact).astype(int)
    k = int(total - base.sum())
    if k > 0:
        for i in np.argsort(-(exact - np.floor(exact)))[:k]:
            base[i] += 1
    return base


def sle_weights(tree: RootedTree, D: Array, index: Mapping[Node, int]) -> Dict[Node, float]:
    r"""Subtree-weighted log-edge weights :math:`w_v = |s_v|\log(1 + d(\mathrm{pa}(v), v))`.

    ``|s_v|`` is the number of tasks in ``v``'s subtree, so a node that many
    descendants inherit from is refined more; the log-edge factor adds budget
    where the parent is far, since that is the gap refinement has to close. The
    root carries no weight, its budget being set separately.
    """
    sz = tree.subtree_sizes()
    return {
        v: float(sz[v] * np.log1p(D[index[tree.parent[v]], index[v]]))
        for v in tree.nodes if v != tree.root
    }


def alloc_from_weights(
    nodes: Sequence[Node],
    root: Node,
    weights: Mapping[Node, float],
    B: int,
    root_frac: float = 0.10,
    regime: Optional[str] = None,
) -> Dict[Node, int]:
    """Turn per-task weights into an integer allocation summing to exactly ``B``.

    ``root_frac`` is the share of the freely allocatable budget reserved for the
    root before the rest is apportioned. The root is trained from scratch while
    every other task starts from a parent, so it needs a head start; a sweep of
    this fraction is reported in the appendix, with interior optima on WEAVE-UK
    and an optimum at zero elsewhere.
    """
    regime = default_regime() if regime is None else regime
    nodes = list(nodes)
    others = [v for v in nodes if v != root]
    if not others:
        return {root: int(B)}
    w = np.array([max(0.0, float(weights.get(v, 0.0))) for v in others])

    if regime == "floor1" and B >= len(nodes):
        surplus = B - len(nodes)
        b_root = min(max(0, int(root_frac * surplus)), surplus)
        shares = apportion(w, surplus - b_root)
        alloc = {v: int(shares[i]) + 1 for i, v in enumerate(others)}
        alloc[root] = b_root + 1
        return alloc

    b_root = min(max(1, int(root_frac * B)), B)
    shares = apportion(w, B - b_root)
    alloc = {v: int(shares[i]) for i, v in enumerate(others)}
    alloc[root] = b_root
    return alloc


def sle_alloc(tree: RootedTree, D: Array, index: Mapping[Node, int], B: int,
              root_frac: float = 0.10, regime: Optional[str] = None) -> Dict[Node, int]:
    """SLE allocation for a rooted cascade. Used by Chain, CTL-kNN, CTL-MST, CTL-MSTc."""
    return alloc_from_weights(tree.nodes, tree.root, sle_weights(tree, D, index),
                              B, root_frac=root_frac, regime=regime)


def uniform_alloc(nodes: Sequence[Node], root: Node, B: int,
                  root_frac: float = 0.10,
                  regime: Optional[str] = None) -> Dict[Node, int]:
    """Equal split over the non-root tasks, after reserving ``root_frac`` for the root.

    The allocation the paper gives Star and RandTree, neither of which carries a
    geometry the SLE weights could read.
    """
    regime = default_regime() if regime is None else regime
    nodes = list(nodes)
    others = [v for v in nodes if v != root]
    if not others:
        return {root: int(B)}

    if regime == "floor1" and B >= len(nodes):
        surplus = B - len(nodes)
        b_root = min(max(0, int(root_frac * surplus)), surplus)
        base, extra = divmod(surplus - b_root, len(others))
        alloc = {v: 1 + base + (1 if i < extra else 0) for i, v in enumerate(others)}
        alloc[root] = b_root + 1
        return alloc

    b_root = min(max(1, int(root_frac * B)), B)
    base, extra = divmod(max(0, B - b_root), len(others))
    alloc = {v: base + (1 if i < extra else 0) for i, v in enumerate(others)}
    alloc[root] = b_root
    return alloc


def individual_alloc(nodes: Sequence[Node], B: int) -> Dict[Node, int]:
    """``B // T`` steps per task, the first ``B % T`` tasks getting one more.

    The no-transfer baseline. Splitting the remainder rather than dropping it
    matters at the budgets the paper studies, where ``B % T`` can be a
    substantial share of the total.
    """
    nodes = list(nodes)
    base, extra = divmod(int(B), len(nodes))
    return {v: base + (1 if i < extra else 0) for i, v in enumerate(nodes)}
