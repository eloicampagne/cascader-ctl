"""The constructions are what the paper says they are.

Each construction is checked against its definition rather than against a stored
output: the MST against an independent minimum-spanning-tree implementation, the
chain against the greedy nearest-neighbour rule, the star against depth one,
CTL-MSTc against the score it claims to maximize.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascader.trees import (CONSTRUCTIONS, build_chain, build_knn, build_mst,
                            build_mstc, build_random, build_star, mst_edges,
                            root_tree, softplus_penalty)

ALL = list(CONSTRUCTIONS)


# --------------------------------------------------------------------------
# Every construction is a spanning tree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_is_a_spanning_tree_rooted_where_asked(small, name):
    tree = CONSTRUCTIONS[name](small.nodes, small.D, small.root,
                               rng=np.random.default_rng(0), lambda_reach=0.5)
    assert tree.root == small.root
    assert len(tree.edges) == small.T - 1
    assert set(tree.nodes) == set(small.nodes)
    assert tree.parent[tree.root] is None
    assert all(tree.parent[v] is not None for v in small.nodes if v != small.root)

    # acyclic and connected: every node reaches the root by following parents
    for v in small.nodes:
        seen, cur = set(), v
        while cur is not None:
            assert cur not in seen, "cycle in the parent map"
            seen.add(cur)
            cur = tree.parent[cur]
        assert tree.root in seen


@pytest.mark.parametrize("name", ALL)
def test_node_order_is_the_collection_order(small, name):
    """Allocators hand out the remainder in node order, so it must be preserved."""
    tree = CONSTRUCTIONS[name](small.nodes, small.D, small.root,
                               rng=np.random.default_rng(0), lambda_reach=0.5)
    assert tree.nodes == list(small.nodes)


@pytest.mark.parametrize("name", ALL)
def test_parents_precede_children_in_the_edge_list(small, name):
    tree = CONSTRUCTIONS[name](small.nodes, small.D, small.root,
                               rng=np.random.default_rng(0), lambda_reach=0.5)
    placed = {tree.root}
    for u, v in tree.edges:
        assert u in placed, "a child appears before its parent"
        placed.add(v)


# --------------------------------------------------------------------------
# The MST is a minimum spanning tree
# --------------------------------------------------------------------------

def test_mst_total_weight_matches_an_independent_implementation(small):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from scipy.sparse.csgraph import minimum_spanning_tree

    ours = mst_edges(small.nodes, small.D)
    idx = small.index
    ours_weight = sum(float(small.D[idx[u], idx[v]]) for u, v in ours)
    reference = float(minimum_spanning_tree(scipy_sparse.csr_matrix(small.D)).sum())
    assert ours_weight == pytest.approx(reference, rel=1e-9)


def test_mst_is_no_longer_than_the_chain(small):
    """A chain is a spanning tree, so the MST is at most as long. Section 3.1."""
    idx = small.index
    mst = build_mst(small.nodes, small.D, small.root)
    chain = build_chain(small.nodes, small.D, small.root)
    assert mst.total_length(small.D, idx) <= chain.total_length(small.D, idx) + 1e-9


def test_undirected_mst_edges_do_not_depend_on_the_root(small):
    """Rooting fixes the traversal, not the topology."""
    base = {frozenset(e) for e in mst_edges(small.nodes, small.D)}
    for root in small.nodes[:5]:
        tree = build_mst(small.nodes, small.D, root)
        rooted = {frozenset((tree.parent[v], v)) for v in small.nodes if v != root}
        assert rooted == base


# --------------------------------------------------------------------------
# Star and chain
# --------------------------------------------------------------------------

def test_star_is_depth_one(small):
    tree = build_star(small.nodes, small.D, small.root)
    depths = tree.depths()
    assert max(depths.values()) == 1
    assert all(tree.parent[v] == small.root for v in small.nodes if v != small.root)


def test_chain_is_a_path_and_is_greedy(small):
    tree = build_chain(small.nodes, small.D, small.root)
    depths = tree.depths()
    assert max(depths.values()) == small.T - 1
    assert sorted(depths.values()) == list(range(small.T))

    # each step goes to the nearest task not yet visited
    idx = small.index
    order = sorted(small.nodes, key=lambda v: depths[v])
    visited = {order[0]}
    for cur, nxt in zip(order, order[1:]):
        candidates = [v for v in small.nodes if v not in visited]
        best = min(candidates, key=lambda v: small.D[idx[cur], idx[v]])
        assert small.D[idx[cur], idx[nxt]] == pytest.approx(
            small.D[idx[cur], idx[best]])
        visited.add(nxt)


def test_knn_is_shallower_than_the_chain(small):
    knn = build_knn(small.nodes, small.D, small.root, k=4)
    chain = build_chain(small.nodes, small.D, small.root)
    assert max(knn.depths().values()) < max(chain.depths().values())


def test_knn_connects_even_when_the_neighbour_graph_is_not(small):
    """k = 1 usually leaves several components; the tree must still span."""
    tree = build_knn(small.nodes, small.D, small.root, k=1)
    assert len(tree.edges) == small.T - 1
    assert set(tree.nodes) == set(small.nodes)


# --------------------------------------------------------------------------
# CTL-MSTc
# --------------------------------------------------------------------------

def test_mstc_without_coverage_only_pays_the_locality_penalty(tiny):
    """lambda = 0 leaves a greedy tree penalised by subtree size times phi_tau."""
    D, nodes, root = tiny.D, tiny.nodes, tiny.root
    tree = build_mstc(nodes, D, root, lambda_reach=0.0)
    tau = float(np.median(D[np.triu_indices(len(nodes), k=1)]))
    phi = softplus_penalty(D, tau)
    idx = tiny.index

    # replay the greedy rule and check every pick
    in_tree = {root}
    subtree = {v: 1 for v in nodes}
    for u, v in tree.edges:
        best = None
        for w in in_tree:
            for c in set(nodes) - in_tree:
                score = subtree[w] * phi[idx[w], idx[c]]
                if best is None or score < best[0] - 1e-12:
                    best = (score, w, c)
        assert best is not None
        assert subtree[u] * phi[idx[u], idx[v]] == pytest.approx(best[0])
        in_tree.add(v)
        cur = u
        while cur is not None:
            subtree[cur] += 1
            cur = tree.parent[cur]


def test_coverage_term_changes_the_tree(small):
    a = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.0)
    b = build_mstc(small.nodes, small.D, small.root, lambda_reach=5.0)
    assert a.edges != b.edges


def test_softplus_penalty_is_the_published_phi():
    r"""phi_tau(d) = tau log(1 + exp((d - tau)/tau)), softplus centered at tau."""
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    tau = 2.0
    got = softplus_penalty(D, tau)
    expected = tau * np.log1p(np.exp((D - tau) / tau))
    np.testing.assert_allclose(got, expected)
    # monotone increasing, and nearly free in the near field
    grid = np.linspace(0, 10, 50).reshape(1, -1)
    vals = softplus_penalty(grid, 2.0).ravel()
    assert np.all(np.diff(vals) > 0)
    assert vals[0] < 0.1 * vals[-1]


def test_mstc_is_deterministic(small):
    a = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.6)
    b = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.6)
    assert a.edges == b.edges


# --------------------------------------------------------------------------
# RandTree
# --------------------------------------------------------------------------

def test_randtree_ignores_the_distances(small):
    """Geometry-free by construction: the same seed gives the same tree on any D."""
    a = build_random(small.nodes, small.D, small.root,
                     rng=np.random.default_rng(3))
    b = build_random(small.nodes, small.D * 100 + 1, small.root,
                     rng=np.random.default_rng(3))
    assert a.edges == b.edges


def test_randtree_varies_with_the_stream(small):
    a = build_random(small.nodes, small.D, small.root, rng=np.random.default_rng(1))
    b = build_random(small.nodes, small.D, small.root, rng=np.random.default_rng(2))
    assert a.edges != b.edges


# --------------------------------------------------------------------------
# root_tree
# --------------------------------------------------------------------------

def test_root_tree_reorients_without_changing_topology(small):
    undirected = mst_edges(small.nodes, small.D)
    base = {frozenset(e) for e in undirected}
    for root in small.nodes[:4]:
        tree = root_tree(undirected, root, small.nodes)
        assert tree.root == root
        assert {frozenset(e) for e in tree.edges} == base
        assert tree.nodes == list(small.nodes)


def test_depths_and_subtree_sizes_agree_on_a_path(tiny):
    chain = build_chain(tiny.nodes, tiny.D, tiny.root)
    depths = chain.depths()
    sizes = chain.subtree_sizes()
    for v in tiny.nodes:
        assert sizes[v] == tiny.T - depths[v]
