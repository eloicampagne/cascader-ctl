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
    "build_chain_2opt",
    "build_forest",
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


def build_chain_2opt(nodes: Sequence[Node], D: Array, root: Node, **_: object) -> RootedTree:
    """Chain-2opt: the greedy chain shortened by 2-opt, root kept at the start.

    Reversing a segment ``p[i..j]`` of the path replaces the edges
    ``(p[i-1], p[i])`` and ``(p[j], p[j+1])`` by ``(p[i-1], p[j])`` and
    ``(p[i], p[j+1])`` and leaves every other edge unchanged. Each pass applies,
    for every ``i``, the reversal of largest decrease, until none shortens the path.
    """
    nodes = list(nodes)
    D = np.asarray(D, dtype=float)
    chain = build_chain(nodes, D, root)
    ix = {v: i for i, v in enumerate(nodes)}
    p = [ix[root]] + [ix[v] for _, v in chain.edges]
    p = np.array(p); n = len(p)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a, b = p[i - 1], p[i]
            j = np.arange(i + 1, n); c = p[j]
            nxt = p[np.minimum(j + 1, n - 1)]; last = j + 1 >= n
            old = D[c, nxt]; new = D[b, nxt]
            old[last] = 0.0; new[last] = 0.0
            delta = D[a, c] + new - D[a, b] - old
            k = int(np.argmin(delta))
            if delta[k] < -1e-12:
                jj = j[k]; p[i:jj + 1] = p[i:jj + 1][::-1]; improved = True
    edges = [(nodes[p[k]], nodes[p[k + 1]]) for k in range(n - 1)]
    return root_tree(edges, root, nodes)


def build_forest(nodes: Sequence[Node], D: Array, root: Node, z: float = 2.0,
                 **_: object) -> RootedTree:
    """CTL-Forest: branch between groups of tasks, chain within each group.

    MST edges longer than ``mean + z * std`` of the MST edge lengths (Zahn's
    criterion) are cut. The remaining components are linked by the cut edges,
    which are the shortest links between them, and each component is rebuilt as
    a greedy chain from the task through which the cascade enters it. When no
    edge exceeds the threshold the forest is a single greedy chain.
    """
    nodes = list(nodes)
    D = np.asarray(D, dtype=float)
    idx = {v: i for i, v in enumerate(nodes)}
    edges = mst_edges(nodes, D)
    if not edges:
        return root_tree([], root, nodes)
    lens = np.array([D[idx[u], idx[v]] for u, v in edges])
    thr = lens.mean() + z * lens.std()
    keep = [e for e, l in zip(edges, lens) if l <= thr]
    bridges = [e for e, l in zip(edges, lens) if l > thr]

    adj: Dict[Node, List[Node]] = defaultdict(list)
    for u, v in keep:
        adj[u].append(v)
        adj[v].append(u)
    comp_of: Dict[Node, int] = {}
    comps: List[List[Node]] = []
    for s in nodes:
        if s in comp_of:
            continue
        comp_of[s] = len(comps)
        comp, q = [], deque([s])
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj[u]:
                if w not in comp_of:
                    comp_of[w] = len(comps)
                    q.append(w)
        comps.append(comp)

    # orient the bridges away from the root's component to find each entry task
    badj: Dict[int, List[Tuple[int, Node, Node]]] = defaultdict(list)
    for u, v in bridges:
        badj[comp_of[u]].append((comp_of[v], u, v))
        badj[comp_of[v]].append((comp_of[u], v, u))
    entry = {comp_of[root]: root}
    q = deque([comp_of[root]])
    while q:
        a = q.popleft()
        for b, _u, v in badj[a]:
            if b not in entry:
                entry[b] = v
                q.append(b)

    undirected: List[Edge] = list(bridges)
    for i, comp in enumerate(comps):
        if len(comp) > 1:
            sub = [entry[i]] + [v for v in comp if v != entry[i]]
            sD = D[np.ix_([idx[v] for v in sub], [idx[v] for v in sub])]
            undirected += build_chain(sub, sD, entry[i]).edges
    return root_tree(undirected, root, nodes)


def build_knn(nodes: Sequence[Node], D: Array, root: Node, k: int = 20,
              **_: object) -> RootedTree:
    """kNN-BFS: breadth-first spanning tree of the symmetrised k-nearest-neighbour graph.

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

    The cheapest attachment of a remaining task ``w`` costs at least its
    cheapest unweighted cost :math:`c(w)`, since :math:`\psi \ge 1`, so
    :math:`-c(w) + \lambda G(w)` bounds its score from above. Each step scores
    exactly only the tasks whose bound exceeds the best exact score found so
    far, in decreasing order of the bound, in batches. The tree is the one the
    exhaustive search returns, ties included: the smallest task index wins, and
    within a task the smallest tree-member index.

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
    lam = float(lambda_reach)

    root_idx = nodes.index(root)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[root_idx] = True
    parent_arr = np.full(n, -1, dtype=int)
    subtree = np.ones(n, dtype=float)
    min_cost = phi[root_idx].copy()
    min_cost[root_idx] = 0.0
    edges: List[Edge] = []
    # gains[x] = sum over remaining w of max(0, min_cost[w] - phi[x, w]), kept up to
    # date incrementally: adding a task only changes the terms of the tasks whose
    # cheapest attachment it improves.
    gains = (np.maximum(0.0, min_cost[None, :] - phi)[:, ~in_tree].sum(axis=1)
             if lam != 0.0 else np.zeros(n))

    while in_tree.sum() < n:
        remaining = np.flatnonzero(~in_tree)
        tree_nodes = np.flatnonzero(in_tree)
        psi = subtree[tree_nodes]
        bound = -min_cost[remaining] + lam * gains[remaining]
        order = np.argsort(-bound, kind="stable")
        cand, bound_sorted = remaining[order], bound[order]
        best, v, u = -np.inf, -1, -1
        start, size = 0, 32
        while start < len(cand) and bound_sorted[start] >= best:
            ws = cand[start:start + size]
            pen = phi[np.ix_(tree_nodes, ws)] * psi[:, None]
            js = np.argmin(pen, axis=0)
            sc = -pen[js, np.arange(len(ws))] + lam * gains[ws]
            for k in np.flatnonzero(sc >= best):
                if sc[k] > best or (sc[k] == best and ws[k] < v):
                    best, v, u = float(sc[k]), int(ws[k]), int(tree_nodes[js[k]])
            start += size
            size *= 2
        parent_arr[v] = u
        in_tree[v] = True
        cur = u
        while cur != -1:
            subtree[cur] += 1
            cur = parent_arr[cur]
        new_cost = np.minimum(min_cost, phi[v])
        new_cost[v] = 0.0
        if lam != 0.0:
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
    "Chain-2opt": build_chain_2opt,
    "CTL-Forest": build_forest,
    "kNN-BFS": build_knn,
    "CTL-kNN": build_knn,          # former name of kNN-BFS
    "CTL-MST": build_mst,
    "CTL-MSTc": build_mstc,
}
