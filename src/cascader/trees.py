"""Cascade structures: the rooted trees along which parameters travel.

A construction takes an ordered node list and a distance matrix and returns a
:class:`RootedTree`. The six constructions compared in the paper are here, plus
:func:`root_tree`, which orients an undirected edge set from a chosen root.

Orientation and topology are separate concerns. The undirected MST edge set does
not depend on the root; the root fixes the traversal order and the depth of each
task, which is what a finite budget sees.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
Node = Hashable
Edge = Tuple[Node, Node]

__all__ = [
    "RootedTree",
    "root_tree",
    "mst_edges",
    "build_star",
    "build_chain",
    "build_knn",
    "build_mst",
    "build_mstc",
    "build_random",
    "CONSTRUCTIONS",
]


# --------------------------------------------------------------------------
# The rooted-tree container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RootedTree:
    """A cascade: a root, its directed edges, and the induced parent map.

    ``edges`` is ordered so that a parent always appears before its children,
    which is the order :func:`cascader.cascade.cascade_train` consumes.
    """

    root: Node
    edges: List[Edge]
    parent: Dict[Node, Optional[Node]]
    nodes: List[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.nodes:
            object.__setattr__(self, "nodes", list(self.parent.keys()))

    @property
    def children(self) -> Dict[Node, List[Node]]:
        ch: Dict[Node, List[Node]] = defaultdict(list)
        for u, v in self.edges:
            ch[u].append(v)
        return ch

    def depths(self) -> Dict[Node, int]:
        """Number of edges from the root to each node."""
        d = {self.root: 0}
        for u, v in self.edges:
            d[v] = d[u] + 1
        return d

    def subtree_sizes(self) -> Dict[Node, int]:
        """Number of nodes in each rooted subtree, the node itself included."""
        sz = {v: 1 for v in self.nodes}
        for u, v in reversed(self.edges):
            sz[u] += sz[v]
        return sz

    def edge_length(self, D: Array, index: Dict[Node, int]) -> Dict[Node, float]:
        """Distance from each non-root node to its parent."""
        return {v: float(D[index[self.parent[v]], index[v]])
                for v in self.nodes if v != self.root}

    def total_length(self, D: Array, index: Dict[Node, int]) -> float:
        return float(sum(self.edge_length(D, index).values()))


def root_tree(undirected: Iterable[Edge], root: Node,
              nodes: Optional[Sequence[Node]] = None) -> RootedTree:
    """Orient an undirected tree away from ``root`` by breadth-first search.

    The returned edge order is the BFS order, so every parent precedes its
    children. When ``nodes`` is given, a node unreachable from ``root`` raises
    ``ValueError``: a cascade that silently dropped tasks would also be scored
    on fewer tasks.

    ``nodes`` fixes the order of :attr:`RootedTree.nodes`. Pass the collection's
    own order: allocators hand out the apportionment remainder in that order, so
    a tree that reordered its nodes would hand the leftover steps to different
    tasks than the collection intends. Defaults to discovery order.
    """
    adj: Dict[Node, List[Node]] = defaultdict(list)
    for u, v in undirected:
        adj[u].append(v)
        adj[v].append(u)
    parent: Dict[Node, Optional[Node]] = {root: None}
    edges: List[Edge] = []
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                edges.append((u, v))
                q.append(v)
    if nodes is not None:
        missing = [v for v in nodes if v not in parent]
        if missing:
            raise ValueError(f"{len(missing)} task(s) unreachable from the root; "
                             "the edge set does not span the collection")
        order = list(nodes)
    else:
        order = list(parent)
    return RootedTree(root=root, edges=edges, parent=parent, nodes=order)


# --------------------------------------------------------------------------
# Constructions
# --------------------------------------------------------------------------

def mst_edges(nodes: Sequence[Node], D: Array) -> List[Edge]:
    """Minimum spanning tree of the complete graph weighted by ``D`` (Prim).

    Dense Prim in ``O(T^2)`` time and ``O(T)`` extra memory: one vectorized
    argmin per added task over the current attachment costs. Returns
    ``len(nodes) - 1`` undirected edges of minimum total weight.
    """
    n = len(nodes)
    if n <= 1:
        return []
    D = np.asarray(D, dtype=float)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    best = D[0].copy()
    best[0] = np.inf
    via = np.zeros(n, dtype=int)
    edges: List[Edge] = []
    for _ in range(n - 1):
        v = int(np.argmin(np.where(visited, np.inf, best)))
        visited[v] = True
        edges.append((nodes[via[v]], nodes[v]))
        closer = ~visited & (D[v] < best)
        best[closer] = D[v][closer]
        via[closer] = v
    return edges


def build_mst(nodes: Sequence[Node], D: Array, root: Node, **_: object) -> RootedTree:
    """CTL-MST: the minimum spanning tree, rooted at ``root``.

    Minimizing total edge weight enforces transfer locality: every task is
    initialized from a near neighbour under the chosen distance.
    """
    nodes = list(nodes)
    return root_tree(mst_edges(nodes, D), root, nodes)


def build_star(nodes: Sequence[Node], D: Array, root: Node, **_: object) -> RootedTree:
    """Star: every task is a child of the root, depth one everywhere.

    The classical transfer baseline. One source is trained, then copied into
    every task and refined once.
    """
    nodes = list(nodes)
    edges = [(root, v) for v in nodes if v != root]
    parent: Dict[Node, Optional[Node]] = {root: None}
    parent.update({v: root for v in nodes if v != root})
    return RootedTree(root=root, edges=edges, parent=parent, nodes=nodes)


def build_chain(nodes: Sequence[Node], D: Array, root: Node, **_: object) -> RootedTree:
    """Chain: the greedy nearest-neighbour Hamiltonian path from ``root``.

    A degenerate spanning tree of depth ``T - 1``. Its total edge length is at
    least the MST's, and once the budget is large enough to absorb that excess
    the chain matches any tree with two fewer hyperparameters.
    """
    nodes = list(nodes)
    D = np.asarray(D, dtype=float)
    remaining = np.ones(len(nodes), dtype=bool)
    cur = nodes.index(root)
    remaining[cur] = False
    edges: List[Edge] = []
    for _ in range(len(nodes) - 1):
        nxt = int(np.argmin(np.where(remaining, D[cur], np.inf)))
        edges.append((nodes[cur], nodes[nxt]))
        remaining[nxt] = False
        cur = nxt
    return root_tree(edges, root, nodes)


def build_knn(nodes: Sequence[Node], D: Array, root: Node, k: int = 20,
              **_: object) -> RootedTree:
    """CTL-kNN: breadth-first spanning tree of the symmetrised k-nearest-neighbour graph.

    Cheaper than an MST and bounded in depth, at the cost of ignoring edge
    weights beyond the neighbour lists. When the k-NN graph is disconnected the
    remaining components are attached by their shortest crossing edge.
    """
    nodes = list(nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    k = min(k, n - 1)
    nbrs: Dict[Node, set] = {v: set() for v in nodes}
    for v in nodes:
        d = D[idx[v]].copy()
        d[idx[v]] = np.inf
        for j in np.argsort(d)[:k]:
            nbrs[v].add(nodes[j])
            nbrs[nodes[j]].add(v)
    edges: List[Edge] = []
    visited = {root}
    q = deque([root])
    while len(visited) < n:
        while q:
            u = q.popleft()
            for w in sorted(nbrs[u], key=lambda x: D[idx[u], idx[x]]):
                if w not in visited:
                    visited.add(w)
                    edges.append((u, w))
                    q.append(w)
        if len(visited) < n:
            best = None
            for v in set(nodes) - visited:
                for u in visited:
                    dd = float(D[idx[u], idx[v]])
                    if best is None or dd < best[0]:
                        best = (dd, u, v)
            _, u, v = best  # type: ignore[misc]
            visited.add(v)
            edges.append((u, v))
            q.append(v)
    return root_tree(edges, root, nodes)


def build_random(nodes: Sequence[Node], D: Array, root: Node,
                 rng: Optional[np.random.Generator] = None, **_: object) -> RootedTree:
    """RandTree: a random recursive tree, re-rooted at ``root``.

    Task ``i`` attaches to a uniformly chosen earlier task. Geometry-free by
    construction, so it isolates what the distance matrix contributes: any gap
    between an informed tree and this one is the geometry's doing.
    """
    nodes = list(nodes)
    rng = np.random.default_rng() if rng is None else rng
    undirected = [(nodes[int(rng.integers(0, i))], nodes[i]) for i in range(1, len(nodes))]
    return root_tree(undirected, root, nodes)


def softplus_penalty(D: Array, tau: float) -> Array:
    r"""Locality penalty :math:`\phi_\tau(d) = \tau\log(1 + e^{(d-\tau)/\tau})`.

    Softplus centered at ``tau``, the median pairwise distance by default. Near
    field (:math:`d \ll \tau`) is nearly free, far field is charged linearly.
    """
    tau = max(float(tau), 1e-12)
    return tau * np.log1p(np.exp(np.clip((D - tau) / tau, -100.0, 100.0)))


def build_mstc(nodes: Sequence[Node], D: Array, root: Node,
               lambda_reach: float = 0.6, tau: Optional[float] = None,
               **_: object) -> RootedTree:
    r"""CTL-MSTc: the budget-aware greedy construction of Section 3.3.

    Grows a spanning tree one edge at a time. Among candidate attachments
    :math:`(u \to v)` with ``u`` already in the tree, it maximizes

    .. math:: \mathrm{score}(u,v) = -\psi(u)\,\phi_\tau(d(u,v)) + \lambda\, G(v)

    where :math:`\psi(u)` is the current subtree size of ``u``, so attaching
    below a heavy subtree is charged more; :math:`\phi_\tau` is the softplus
    locality penalty; and the coverage gain

    .. math:: G(v) = \sum_{w \notin \mathcal{T}} \max(0,\ c(w) - \phi_\tau(d(v,w)))

    rewards a task that brings the rest of the collection closer, ``c(w)`` being
    ``w``'s current cheapest attachment cost. Setting ``lambda_reach=0`` drops
    the coverage term and leaves a locality-penalised greedy tree.

    ``tau`` defaults to the median finite off-diagonal distance.
    """
    nodes = list(nodes)
    n = len(nodes)
    if n <= 1:
        return RootedTree(root=root, edges=[], parent={root: None}, nodes=nodes)

    D = np.asarray(D, dtype=float)
    if tau is None:
        tri = D[np.triu_indices(n, k=1)]
        tri = tri[np.isfinite(tri)]
        tau = float(np.median(tri)) if tri.size else 1.0
    phi = softplus_penalty(D, tau)

    root_idx = nodes.index(root)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[root_idx] = True
    parent_arr = np.full(n, -1, dtype=int)
    subtree = np.ones(n, dtype=int)
    min_cost = phi[root_idx].copy()
    min_cost[root_idx] = 0.0
    edges: List[Edge] = []
    # gains[x] = sum over remaining w of max(0, min_cost[w] - phi[x, w]), kept up to
    # date incrementally: adding a task only changes the terms of the tasks whose
    # cheapest attachment it improves, so each step costs O(T * changed) instead
    # of the O(T^2) of a full recomputation.
    gains = (np.maximum(0.0, min_cost[None, :] - phi)[:, ~in_tree].sum(axis=1)
             if lambda_reach != 0.0 else None)

    while in_tree.sum() < n:
        remaining = np.flatnonzero(~in_tree)
        tree_nodes = np.flatnonzero(in_tree)
        psi = subtree[tree_nodes].astype(float)
        penalties = phi[np.ix_(tree_nodes, remaining)] * psi[:, None]
        best_u_pos = np.argmin(penalties, axis=0)
        best_pen = penalties[best_u_pos, np.arange(len(remaining))]
        if gains is not None:
            scores = -best_pen + lambda_reach * gains[remaining]
        else:
            scores = -best_pen
        pick = int(np.argmax(scores))
        v = int(remaining[pick])
        u = int(tree_nodes[best_u_pos[pick]])
        parent_arr[v] = u
        in_tree[v] = True
        cur = u
        while cur != -1:
            subtree[cur] += 1
            cur = parent_arr[cur]
        new_cost = np.minimum(min_cost, phi[v])
        new_cost[v] = 0.0
        if gains is not None:
            gains -= np.maximum(0.0, min_cost[v] - phi[:, v])
            changed = np.flatnonzero((new_cost < min_cost) & ~in_tree)
            if changed.size:
                gains += (np.maximum(0.0, new_cost[changed][None, :] - phi[:, changed])
                          - np.maximum(0.0, min_cost[changed][None, :] - phi[:, changed])
                          ).sum(axis=1)
        min_cost = new_cost
        edges.append((nodes[u], nodes[v]))

    parent: Dict[Node, Optional[Node]] = {root: None}
    for i in range(n):
        if i != root_idx:
            parent[nodes[i]] = nodes[parent_arr[i]]
    return RootedTree(root=root, edges=edges, parent=parent, nodes=nodes)


#: Construction name -> builder. Keys match the method names of Table 1.
CONSTRUCTIONS = {
    "Star": build_star,
    "RandTree": build_random,
    "Chain": build_chain,
    "CTL-kNN": build_knn,
    "CTL-MST": build_mst,
    "CTL-MSTc": build_mstc,
}
