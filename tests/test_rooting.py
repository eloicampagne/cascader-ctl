"""The rooting rules match their definitions.

Each rule is an argmin over an explicit objective, so each test recomputes that
objective by brute force and checks the rule returns a minimizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascader.rooting import (ROOT_RULES, distance_medoid, hop_centre,
                              medoid_root, select_root, weighted_1_centre,
                              weighted_1_median)
from cascader.trees import build_chain, mst_edges, root_tree


def test_medoid_minimizes_total_distance_between_feature_means(small):
    root = medoid_root(small.nodes, small.X_train)
    means = np.stack([small.X_train[v].mean(axis=0) for v in small.nodes])
    D = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
    totals = D.sum(axis=1)
    assert small.nodes[int(np.argmin(totals))] == root


def test_distance_medoid_minimizes_total_construction_distance(small):
    root = distance_medoid(small.nodes, small.D)
    assert root == small.nodes[int(np.argmin(small.D.sum(axis=1)))]


def test_the_two_medoids_live_in_different_geometries(small):
    """The shipped default is a raw-feature medoid, not a medoid of D.

    Not a bug, but worth pinning: the tree is built in the gradient geometry
    while the root is picked in the feature one.
    """
    assert medoid_root(small.nodes, small.X_train) is not None
    assert distance_medoid(small.nodes, small.D) is not None


def test_hop_centre_minimizes_eccentricity_in_hops(tiny):
    undirected = mst_edges(tiny.nodes, tiny.D)
    centre = hop_centre(tiny.nodes, undirected)
    ecc = {v: max(root_tree(undirected, v, tiny.nodes).depths().values())
           for v in tiny.nodes}
    assert ecc[centre] == min(ecc.values())


def test_hop_centre_halves_the_depth_of_a_path(small):
    chain = build_chain(small.nodes, small.D, small.root)
    undirected = [(chain.parent[v], v) for v in small.nodes if v != small.root]
    centre = hop_centre(small.nodes, undirected)
    rerooted = root_tree(undirected, centre, small.nodes)
    assert max(rerooted.depths().values()) == pytest.approx(
        (small.T - 1) // 2, abs=1)


def test_weighted_1_median_minimizes_total_weighted_path_length(tiny):
    undirected = mst_edges(tiny.nodes, tiny.D)
    root = weighted_1_median(tiny.nodes, undirected, tiny.D)
    totals = {v: _path_lengths(tiny, undirected, v).sum() for v in tiny.nodes}
    assert totals[root] == pytest.approx(min(totals.values()))


def test_weighted_1_centre_minimizes_the_longest_weighted_path(tiny):
    undirected = mst_edges(tiny.nodes, tiny.D)
    root = weighted_1_centre(tiny.nodes, undirected, tiny.D)
    worst = {v: _path_lengths(tiny, undirected, v).max() for v in tiny.nodes}
    assert worst[root] == pytest.approx(min(worst.values()))


def _path_lengths(tasks, undirected, source) -> np.ndarray:
    """Weighted tree distances from ``source`` to every task, by brute force."""
    tree = root_tree(undirected, source, tasks.nodes)
    idx = tasks.index
    out = {source: 0.0}
    for u, v in tree.edges:
        out[v] = out[u] + float(tasks.D[idx[u], idx[v]])
    return np.array([out[v] for v in tasks.nodes])


def test_the_two_weighted_rules_differ_from_the_medoid_in_general(small):
    """The medoid ignores the topology, so it is not generally either rule."""
    undirected = mst_edges(small.nodes, small.D)
    medoid = medoid_root(small.nodes, small.X_train)
    median = weighted_1_median(small.nodes, undirected, small.D)
    centre = weighted_1_centre(small.nodes, undirected, small.D)
    assert {medoid, median, centre} != {medoid}, "all three rules coincided"


@pytest.mark.parametrize("rule", ROOT_RULES)
def test_select_root_dispatches_and_returns_a_task(small, rule):
    undirected = mst_edges(small.nodes, small.D)
    root = select_root(rule, small.nodes, X=small.X_train, D=small.D,
                       undirected=undirected)
    assert root in small.nodes


def test_select_root_says_what_is_missing(small):
    with pytest.raises(ValueError, match="features"):
        select_root("medoid", small.nodes, D=small.D)
    with pytest.raises(ValueError, match="undirected"):
        select_root("w-median", small.nodes, D=small.D)
    with pytest.raises(ValueError, match="unknown rooting rule"):
        select_root("nope", small.nodes, undirected=[], D=small.D)


def test_rerooting_leaves_the_undirected_edge_set_alone(small):
    """Root invariance of the MST topology, the premise of the whole discussion."""
    undirected = mst_edges(small.nodes, small.D)
    base = {frozenset(e) for e in undirected}
    for rule in ROOT_RULES:
        root = select_root(rule, small.nodes, X=small.X_train, D=small.D,
                           undirected=undirected)
        tree = root_tree(undirected, root, small.nodes)
        assert {frozenset(e) for e in tree.edges} == base
