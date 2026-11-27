"""Cascade constructions from nearest-neighbour lists, for large collections.

Above :data:`cascader.distances.DENSE_MAX` tasks the dense ``(T, T)`` distance
matrix no longer fits comfortably in memory. The constructions below read only
the gradient proxies ``G`` (one row per task) and, for each task, the list of its
``k`` nearest neighbours under the gradient distance. Every distance they use is
the exact ``||g_u - g_v||``, computed when needed.

* **Chain** visits the nearest remaining task among the current task's
  neighbours, and searches all remaining tasks only when the list is exhausted.
  With exact lists it is the dense Chain.
* **CTL-MST** is the minimum spanning tree of the symmetrized neighbour graph,
  with components joined through their nearest outside task. It is the dense MST
  whenever every MST edge lies in the graph, which ``k = 20`` gave on Synth and
  on four of five CIFAR-10 seeds checked (one edge differed on the fifth).
* **CTL-Forest** cuts that MST with the dense rule and rebuilds each component
  as a neighbour-list Chain.
* **Chain-2opt** applies 2-opt with neighbour lists: a reversal is tried only
  when its first new edge joins a task to one of its neighbours. It reaches a
  different local optimum than the dense 2-opt, of similar length, and its error
  can differ by a few percent (the dense 2-opt itself varies as much with the
  order in which it visits the tasks).

By default the lists come from an exact search by blocks of rows, which costs
``O(T^2 p)`` time, like the medoid root, but only ``O(T k)`` memory. HNSW
(``hnswlib``, optional dependency ``cascader-ctl[ann]``) is faster and keeps the
MST (99.4 % of edges shared at ``T = 50000`` on the generator of Synth), but a
single missed neighbour diverts the rest of a greedy Chain: with a recall above
99 %, the HNSW Chain shared 58 % of its edges with the exact one at that size.
Select it with ``method='hnsw'``, or ``tasks.meta['neighbours'] = 'hnsw'``.
"""

from __future__ import annotations

import warnings
from collections import defaultdict, deque
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from .trees import RootedTree, build_star, root_tree

Array = np.ndarray
Node = Hashable

__all__ = [
    "K_DEFAULT",
    "EdgeLengths",
    "knn_lists",
    "knn_chain_order",
    "knn_mst_edges",
    "build_knn_structure",
    "SPARSE_CONSTRUCTIONS",
]

#: Neighbours per task. Ten sufficed except on CIFAR-10, whose groups of about
#: ten tasks sharing an optimum kept the lists inside each group; twenty
#: recovered the dense MST there on four of five seeds checked.
K_DEFAULT = 20

_BLOCK_ENTRIES = 8_000_000


class EdgeLengths:
    """Read-only stand-in for ``D`` that computes ``D[i, j] = ||g_i - g_j||`` on demand.

    The SLE allocation reads ``D`` only along tree edges, so this is all it needs.
    """

    def __init__(self, G: Array):
        self.G = np.asarray(G, dtype=np.float64)

    def __getitem__(self, ij: Tuple[int, int]) -> float:
        i, j = ij
        return float(np.linalg.norm(self.G[i] - self.G[j]))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.G.shape[0], self.G.shape[0])


def _blocked_knn(G: Array, k: int) -> Array:
    T = G.shape[0]
    sq = np.einsum("ij,ij->i", G, G)
    bs = max(1, _BLOCK_ENTRIES // T)
    lab = np.empty((T, k), dtype=np.int64)
    for s in range(0, T, bs):
        r = np.arange(s, min(T, s + bs))
        D2 = sq[r, None] + sq[None, :] - 2.0 * (G[r] @ G.T)
        D2[np.arange(len(r)), r] = np.inf
        part = np.argpartition(D2, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(D2, part, axis=1), axis=1)
        lab[s:s + len(r)] = np.take_along_axis(part, order, axis=1)
    return lab


def _hnsw_knn(G: Array, k: int) -> Array:
    import hnswlib

    T, p = G.shape
    # Higher recall settings in high dimension, where the defaults miss neighbours.
    M, efc, ef = (16, 100, 64) if p <= 128 else (32, 200, 200)
    Gf = np.ascontiguousarray(G, dtype=np.float32)
    index = hnswlib.Index(space="l2", dim=p)
    index.init_index(max_elements=T, M=M, ef_construction=efc, random_seed=0)
    index.add_items(Gf, np.arange(T))
    index.set_ef(max(ef, k + 1))
    lab, _ = index.knn_query(Gf, k=k + 1)
    out = np.empty((T, k), dtype=np.int64)
    for i, row in enumerate(lab):
        nb = [int(j) for j in row if j != i][:k]
        out[i, :len(nb)] = nb
        if len(nb) < k:                       # duplicate rows can push i out of its own list
            out[i, len(nb):] = nb[-1] if nb else (i + 1) % T
    return out


def knn_lists(G: Array, k: int = K_DEFAULT, method: str = "exact") -> Array:
    """``(T, k)`` indices of the ``k`` nearest neighbours of each row of ``G``.

    ``method`` is ``'exact'`` (blocks of rows, the default), ``'hnsw'``, or
    ``'auto'``, which uses HNSW when ``hnswlib`` is installed and the exact
    search otherwise.
    """
    G = np.asarray(G, dtype=np.float64)
    k = int(min(k, G.shape[0] - 1))
    if method not in ("auto", "hnsw", "exact"):
        raise ValueError("method must be 'auto', 'hnsw' or 'exact'")
    if method in ("auto", "hnsw"):
        try:
            return _hnsw_knn(G, k)
        except ImportError:
            if method == "hnsw":
                raise
            warnings.warn("hnswlib is not installed: falling back to the exact neighbour "
                          "search, quadratic in time. Install cascader-ctl[ann] for HNSW.",
                          RuntimeWarning, stacklevel=2)
    return _blocked_knn(G, k)


def _dist_to(G: Array, i: int, J: Array) -> Array:
    return np.linalg.norm(G[J] - G[i], axis=1)


def knn_chain_order(G: Array, lab: Array, start: int,
                    members: Optional[Sequence[int]] = None) -> List[int]:
    """Greedy nearest-neighbour path from ``start`` over ``members`` (positions)."""
    T = G.shape[0]
    members = np.arange(T) if members is None else np.asarray(members)
    remaining = np.zeros(T, dtype=bool)
    remaining[members] = True
    remaining[start] = False
    cur, path = start, [start]
    for _ in range(len(members) - 1):
        cand = np.array([j for j in lab[cur] if remaining[j]], dtype=np.int64)
        if cand.size == 0:
            cand = np.flatnonzero(remaining)
        nxt = int(cand[np.argmin(_dist_to(G, cur, cand))])
        remaining[nxt] = False
        path.append(nxt)
        cur = nxt
    return path


def _two_opt(G: Array, lab: Array, path: List[int]) -> List[int]:
    p = np.array(path)
    n = len(p)

    def d(a: int, b: int) -> float:
        return float(np.linalg.norm(G[a] - G[b]))

    improved = True
    while improved:
        improved = False
        pos = np.empty(G.shape[0], dtype=np.int64)
        pos[p] = np.arange(n)
        for i in range(1, n - 1):
            a, b = p[i - 1], p[i]
            best, bj = -1e-12, None
            for c in lab[a]:
                j = pos[c]
                if j <= i:
                    continue
                old = d(a, b) + (d(c, p[j + 1]) if j + 1 < n else 0.0)
                new = d(a, c) + (d(b, p[j + 1]) if j + 1 < n else 0.0)
                if new - old < best:
                    best, bj = new - old, j
            if bj is not None:
                p[i:bj + 1] = p[i:bj + 1][::-1]
                pos[p] = np.arange(n)
                improved = True
    return [int(x) for x in p]


def knn_mst_edges(G: Array, lab: Array) -> List[Tuple[int, int]]:
    """MST of the symmetrized neighbour graph, edges weighted by exact distances.

    Components of the graph, if any, are joined through the nearest outside task
    of each component (exact search by blocks), until the graph is connected.
    """
    T = G.shape[0]
    rows = np.repeat(np.arange(T), lab.shape[1])
    cols = lab.ravel()
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    w = np.empty(len(rows))
    for s in range(0, len(rows), 1_000_000):
        sl = slice(s, s + 1_000_000)
        w[sl] = np.linalg.norm(G[rows[sl]] - G[cols[sl]], axis=1)
    w += 1e-12                                  # keep zero-length edges in the sparse graph
    A = coo_matrix((w, (rows, cols)), shape=(T, T)).tocsr()
    A = A.maximum(A.T)
    ncomp, comp = connected_components(A, directed=False)
    sq = np.einsum("ij,ij->i", G, G)
    while ncomp > 1:
        add_r, add_c, add_w = [], [], []
        for c in range(ncomp):
            inside = np.flatnonzero(comp == c)
            outside = np.flatnonzero(comp != c)
            best = (np.inf, -1, -1)
            bs = max(1, _BLOCK_ENTRIES // len(outside))
            for s in range(0, len(inside), bs):
                r = inside[s:s + bs]
                D2 = sq[r, None] + sq[None, outside] - 2.0 * (G[r] @ G[outside].T)
                i, j = np.unravel_index(int(np.argmin(D2)), D2.shape)
                if D2[i, j] < best[0]:
                    best = (D2[i, j], int(r[i]), int(outside[j]))
            add_r.append(best[1]); add_c.append(best[2])
            add_w.append(float(np.linalg.norm(G[best[1]] - G[best[2]])) + 1e-12)
        A = A.maximum(coo_matrix((add_w, (add_r, add_c)), shape=(T, T)).tocsr())
        A = A.maximum(A.T)
        ncomp, comp = connected_components(A, directed=False)
    M = minimum_spanning_tree(A).tocoo()
    return list(zip(M.row.tolist(), M.col.tolist()))


def _forest_edges(G: Array, lab: Array, root: int, z: float) -> List[Tuple[int, int]]:
    edges = knn_mst_edges(G, lab)
    lens = np.array([np.linalg.norm(G[u] - G[v]) for u, v in edges])
    thr = lens.mean() + z * lens.std()
    keep = [e for e, l in zip(edges, lens) if l <= thr]
    bridges = [e for e, l in zip(edges, lens) if l > thr]
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in keep:
        adj[u].append(v)
        adj[v].append(u)
    T = G.shape[0]
    comp_of = -np.ones(T, dtype=np.int64)
    comps: List[List[int]] = []
    for s in range(T):
        if comp_of[s] >= 0:
            continue
        comp_of[s] = len(comps)
        comp, q = [s], deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if comp_of[w] < 0:
                    comp_of[w] = len(comps)
                    comp.append(w)
                    q.append(w)
        comps.append(comp)
    badj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for u, v in bridges:
        badj[comp_of[u]].append((comp_of[v], v))
        badj[comp_of[v]].append((comp_of[u], u))
    entry = {int(comp_of[root]): root}
    q = deque([int(comp_of[root])])
    while q:
        a = q.popleft()
        for b, v in badj[a]:
            if b not in entry:
                entry[b] = v
                q.append(b)
    und = list(bridges)
    for i, comp in enumerate(comps):
        if len(comp) > 1:
            path = knn_chain_order(G, lab, entry[i], comp)
            und += list(zip(path[:-1], path[1:]))
    return und


SPARSE_CONSTRUCTIONS = ("Star", "Chain", "Chain-2opt", "CTL-MST", "CTL-Forest")


def build_knn_structure(method: str, nodes: Sequence[Node], G: Array, root: Node,
                        lab: Optional[Array] = None, k: int = K_DEFAULT,
                        z: float = 2.0) -> RootedTree:
    """Build ``method`` from neighbour lists over the proxies ``G`` (rows aligned with ``nodes``)."""
    nodes = list(nodes)
    if method == "Star":
        return build_star(nodes, None, root)
    if method not in SPARSE_CONSTRUCTIONS:
        raise ValueError(f"{method!r} needs the dense distance matrix; "
                         f"neighbour-list constructions: {SPARSE_CONSTRUCTIONS}")
    G = np.asarray(G, dtype=np.float64)
    if lab is None:
        lab = knn_lists(G, k)
    r = nodes.index(root)
    if method in ("Chain", "Chain-2opt"):
        path = knn_chain_order(G, lab, r)
        if method == "Chain-2opt":
            path = _two_opt(G, lab, path)
        edges = list(zip(path[:-1], path[1:]))
    elif method == "CTL-MST":
        edges = knn_mst_edges(G, lab)
    else:
        edges = _forest_edges(G, lab, r, z)
    return root_tree([(nodes[u], nodes[v]) for u, v in edges], root, nodes)
