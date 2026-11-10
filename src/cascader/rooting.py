"""Root selection.

The undirected edge set of an MST does not depend on the root; the root fixes
the traversal order and therefore the depth at which each task sits. Under a
finite budget that ordering matters.

The paper ships :func:`medoid_root`, computable from the data before any tree
exists and hence identical across every compared method. That uniformity is its
justification. The two rules derived from the error surrogate,
:func:`weighted_1_median` and :func:`weighted_1_centre`, read distances along
the tree, so each construction gets its own root.

Where the surrogate comes from. Unrolling the edge recursion along a root-to-leaf
path, edge :math:`j` enters the bound as
:math:`\\delta_j\\exp(-\\lambda\\sum_{i\\ge j} b_i)`: an edge is corrected only by
the budget spent downstream of it, a geometric discount of ratio
:math:`q = \\rho^{B/T}` per hop counted from the leaf. A root should therefore
minimize :math:`\\Phi(r,v) = \\sum_j \\delta_j q^{\\mathrm{hops}(j \\to v)}`.
As :math:`q \\to 1`, minimizing :math:`\\max_v \\Phi` gives the weighted 1-centre
and minimizing the mean gives the weighted 1-median. Neither is the medoid,
which ignores the topology, and neither is tree height, which counts hops
without weighting them by edge length.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Hashable, Iterable, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

Array = np.ndarray
Node = Hashable
Edge = Tuple[Node, Node]

__all__ = [
    "medoid_root",
    "distance_medoid",
    "hop_centre",
    "weighted_1_centre",
    "weighted_1_median",
    "select_root",
    "ROOT_RULES",
]


def medoid_root(nodes: Sequence[Node], X: Dict[Node, Array]) -> Node:
    """The paper's default: medoid of the per-task mean feature vectors.

    Each task is summarized by ``X[v].mean(axis=0)`` and the task minimizing the
    total Euclidean distance to the others is returned. Note this lives in the
    raw feature space, which is not the space the tree is built in when the
    construction distance is the gradient one -- see :func:`distance_medoid` for
    the variant that stays in the construction geometry.
    """
    nodes = list(nodes)
    means = np.stack([np.asarray(X[v], dtype=float).mean(axis=0) for v in nodes])
    D = cdist(means, means)
    return nodes[int(np.argmin(D.sum(axis=1)))]


def distance_medoid(nodes: Sequence[Node], D: Array) -> Node:
    """Medoid of the construction distance matrix: ``argmin_v sum_u D[u, v]``.

    Topology-agnostic like :func:`medoid_root`, and therefore equally uniform
    across methods, but measured in the geometry the tree is actually built in.
    """
    nodes = list(nodes)
    return nodes[int(np.argmin(np.asarray(D, dtype=float).sum(axis=1)))]


def hop_centre(nodes: Sequence[Node], undirected: Iterable[Edge]) -> Node:
    """Tree centre by leaf peeling: minimizes eccentricity in hops.

    Peeling leaves one or two nodes, the classical characterisation of the tree
    centre; a two-node centre is resolved by taking the smaller identifier so
    the rule stays deterministic. Halves the depth of a path, and buys almost
    nothing empirically -- depth is not what binds at these scales.
    """
    nodes = list(nodes)
    adj: Dict[Node, set] = defaultdict(set)
    for u, v in undirected:
        adj[u].add(v)
        adj[v].add(u)
    deg = {v: len(adj[v]) for v in nodes}
    remaining = set(nodes)
    leaves = [v for v in nodes if deg[v] <= 1]
    while len(remaining) > 2 and leaves:
        nxt = []
        for v in leaves:
            remaining.discard(v)
            for w in adj[v]:
                if w in remaining:
                    deg[w] -= 1
                    if deg[w] == 1:
                        nxt.append(w)
        leaves = nxt
    return sorted(remaining)[0]


def _tree_distances(nodes: Sequence[Node], undirected: Iterable[Edge],
                    D: Array) -> Dict[Node, Dict[Node, float]]:
    """All-pairs path lengths along the tree, each edge weighted by ``D``."""
    nodes = list(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    adj: Dict[Node, list] = defaultdict(list)
    for u, v in undirected:
        w = float(D[idx[u], idx[v]])
        adj[u].append((v, w))
        adj[v].append((u, w))
    out: Dict[Node, Dict[Node, float]] = {}
    for s in nodes:
        dist = {s: 0.0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    q.append(v)
        out[s] = dist
    return out


def weighted_1_centre(nodes: Sequence[Node], undirected: Iterable[Edge],
                      D: Array) -> Node:
    """The task whose farthest task is closest along weighted tree paths.

    The :math:`q \\to 1` limit of minimizing :math:`\\max_v \\Phi(r, v)`.
    """
    td = _tree_distances(nodes, undirected, D)
    return min(nodes, key=lambda s: max(td[s].values()))


def weighted_1_median(nodes: Sequence[Node], undirected: Iterable[Edge],
                      D: Array) -> Node:
    """The task minimizing total weighted tree distance to all others.

    The :math:`q \\to 1` limit of minimizing the mean of :math:`\\Phi(r, v)`.
    """
    td = _tree_distances(nodes, undirected, D)
    return min(nodes, key=lambda s: sum(td[s].values()))


#: Rules that need only the data or the distance matrix, hence apply before the
#: tree exists and stay identical across methods.
_PRE_TREE = {"medoid", "distance-medoid"}

ROOT_RULES = ("medoid", "distance-medoid", "hop-centre", "w-centre", "w-median")


def select_root(rule: str, nodes: Sequence[Node], *, X: Dict[Node, Array] | None = None,
                D: Array | None = None, undirected: Iterable[Edge] | None = None) -> Node:
    """Dispatch to a named rooting rule.

    ``medoid`` needs ``X``; ``distance-medoid`` needs ``D``; the three
    topological rules need ``undirected`` and, except for ``hop-centre``, ``D``.
    """
    if rule == "medoid":
        if X is None:
            raise ValueError("the 'medoid' rule needs the per-task features X")
        return medoid_root(nodes, X)
    if rule == "distance-medoid":
        if D is None:
            raise ValueError("the 'distance-medoid' rule needs a distance matrix D")
        return distance_medoid(nodes, D)
    if undirected is None:
        raise ValueError(f"the {rule!r} rule needs the undirected tree")
    edges = list(undirected)
    if rule == "hop-centre":
        return hop_centre(nodes, edges)
    if D is None:
        raise ValueError(f"the {rule!r} rule needs a distance matrix D")
    if rule == "w-centre":
        return weighted_1_centre(nodes, edges, D)
    if rule == "w-median":
        return weighted_1_median(nodes, edges, D)
    raise ValueError(f"unknown rooting rule {rule!r}; choose from {ROOT_RULES}")
