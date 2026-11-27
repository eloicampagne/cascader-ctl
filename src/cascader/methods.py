"""The seven methods of Table 1, each a construction paired with an allocation.

======================  ==================================  ==================
method                  structure                           allocation
======================  ==================================  ==================
``Individual``          none, no transfer                   ``B // T`` each
``Star``                depth-1 hub                         uniform
``RandTree``            random recursive tree               uniform
``Chain``               greedy nearest-neighbour path       SLE
``Chain-2opt``          greedy path shortened by 2-opt      SLE
``CTL-MST``             minimum spanning tree               SLE
``CTL-Forest``          MST cut at long edges, chains inside SLE
``kNN-BFS``             BFS tree over the k-NN graph        SLE   (ablation)
``CTL-MSTc``            budget-aware greedy tree            SLE   (ablation)
======================  ==================================  ==================

Star and RandTree get the uniform allocation because neither carries a geometry
the SLE weights could read: Star has one depth level, RandTree's edges are not
informative. Giving them SLE anyway is available through ``allocation='sle'``
and is reported in the appendix; it does not change their ranking.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Optional

import numpy as np
import torch.nn as nn

from .allocation import individual_alloc, sle_alloc, uniform_alloc
from .neighbors import EdgeLengths, build_knn_structure, knn_lists
from .cascade import cascade_train, independent_train
from .config import DatasetConfig
from .tasks import TaskCollection
from .trees import CONSTRUCTIONS, RootedTree

Node = Hashable

__all__ = ["METHODS", "ABLATIONS", "SLE_METHODS", "build_structure", "allocate", "run_method"]

#: The seven methods of the paper's main tables. kNN-BFS and CTL-MSTc remain
#: available to build_structure as ablations.
METHODS = ["Individual", "Star", "RandTree", "Chain", "Chain-2opt", "CTL-MST", "CTL-Forest"]
ABLATIONS = ["kNN-BFS", "CTL-MSTc"]

#: Methods whose structure carries geometry, hence receive the SLE allocation.
SLE_METHODS = frozenset({"Chain", "Chain-2opt", "CTL-MST", "CTL-Forest", "kNN-BFS", "CTL-kNN", "CTL-MSTc"})


def build_structure(method: str, tasks: TaskCollection, cfg: DatasetConfig,
                    rng: Optional[np.random.Generator] = None) -> RootedTree:
    """Build the rooted cascade a method runs on.

    ``CTL-MSTc`` reads ``cfg.mstc_lambda``; ``RandTree`` reads ``rng`` and is the
    only construction that is not a deterministic function of the distances.
    """
    if method == "Individual":
        raise ValueError("Individual has no cascade structure")
    if tasks.D is None:
        return _build_from_neighbours(method, tasks)
    try:
        builder = CONSTRUCTIONS[method]
    except KeyError:
        raise KeyError(f"unknown method {method!r}; choose from {METHODS + ABLATIONS}") from None
    if method == "RandTree":
        return builder(tasks.nodes, tasks.D, tasks.root,
                       rng=rng, lambda_reach=cfg.mstc_lambda)
    # Every other construction is a deterministic function of the distances, the
    # root and (for CTL-MSTc) the coverage weight, and does not depend on the rate
    # or the budget, so a sweep over those reuses one tree per collection. The
    # cache lives on the collection; the key includes the identity of ``D`` and
    # the root, so replacing either rebuilds the tree.
    cache = tasks.__dict__.setdefault("_tree_cache", {})
    key = (method, tasks.root, id(tasks.D),
           float(cfg.mstc_lambda) if method == "CTL-MSTc" else None)
    tree = cache.get(key)
    if tree is None:
        tree = builder(tasks.nodes, tasks.D, tasks.root, rng=rng, lambda_reach=cfg.mstc_lambda)
        cache[key] = tree
    return tree


def _build_from_neighbours(method: str, tasks: TaskCollection) -> RootedTree:
    """Tree of a collection without dense distances, from neighbour lists (cached)."""
    if tasks.proxies is None:
        raise ValueError("a collection without D must carry its gradient proxies")
    cache = tasks.__dict__.setdefault("_tree_cache", {})
    key = (method, tasks.root, "knn")
    tree = cache.get(key)
    if tree is None:
        lab = tasks.__dict__.get("_knn_lists")
        if lab is None and method != "Star":
            lab = knn_lists(tasks.proxies, method=tasks.meta.get("neighbours", "exact"))
            tasks.__dict__["_knn_lists"] = lab
        tree = build_knn_structure(method, tasks.nodes, tasks.proxies, tasks.root, lab=lab)
        cache[key] = tree
    return tree


def _edge_lengths(tasks: TaskCollection):
    """``D`` when the collection has it, otherwise lengths computed from the proxies."""
    return tasks.D if tasks.D is not None else EdgeLengths(tasks.proxies)


def allocate(method: str, tree: RootedTree, tasks: TaskCollection, B: int,
             cfg: DatasetConfig, regime: Optional[str] = None,
             allocation: Optional[str] = None) -> Dict[Node, int]:
    """Split ``B`` over the tasks according to the method's allocation rule.

    ``allocation`` overrides the default pairing, which is what the appendix
    experiment that gives Star the SLE allocation does.
    """
    rule = allocation or ("sle" if method in SLE_METHODS else "uniform")
    if rule == "sle":
        return sle_alloc(tree, _edge_lengths(tasks), tasks.index, B,
                         root_frac=cfg.root_frac, regime=regime)
    if rule == "uniform":
        return uniform_alloc(tree.nodes, tree.root, B,
                             root_frac=cfg.b_proto_frac, regime=regime)
    raise ValueError(f"unknown allocation {rule!r}; use 'sle' or 'uniform'")


def run_method(method: str, tasks: TaskCollection, B: int, eta: float,
               cfg: DatasetConfig, factory: Callable[[], nn.Module],
               rng: Optional[np.random.Generator] = None,
               regime: Optional[str] = None,
               allocation: Optional[str] = None) -> Dict[Node, nn.Module]:
    """Build the structure, split the budget, run the cascade.

    Returns one trained model per task. The total number of gradient steps taken
    is exactly ``B`` for every method, which is what makes the comparison a
    comparison.
    """
    if method == "Individual":
        alloc = individual_alloc(tasks.nodes, B)
        return independent_train(tasks.nodes, tasks.X_train, tasks.y_train, alloc,
                                 eta, factory, tasks.task, cfg.reg)

    tree = build_structure(method, tasks, cfg, rng)
    alloc = allocate(method, tree, tasks, B, cfg, regime=regime, allocation=allocation)
    return cascade_train(tree, tasks.X_train, tasks.y_train, alloc, eta, factory,
                         tasks.task, cfg.reg)
