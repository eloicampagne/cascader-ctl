"""Gram-matrix distances, the blocked medoid and the neighbour-list constructions."""

import dataclasses

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from cascader.config import get_config
from cascader.data.synthetic import subspace_collection
from cascader.distances import gradient_geometry, gram_euclidean
from cascader.methods import allocate, build_structure, run_method
from cascader.models import model_factory
from cascader.neighbors import EdgeLengths, build_knn_structure, knn_lists
from cascader.rooting import medoid_root
from cascader.trees import CONSTRUCTIONS


def _edges(tree):
    return {frozenset(e) for e in tree.edges}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_gram_distances_match_cdist(dtype):
    E = np.random.default_rng(0).standard_normal((300, 50)).astype(dtype)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    D = gram_euclidean(E)
    ref = cdist(E.astype(np.float64), E.astype(np.float64))
    assert np.allclose(D, ref, atol=1e-12, rtol=0)
    assert np.array_equal(D, D.T)
    assert np.all(np.diag(D) == 0.0)


def test_medoid_matches_dense_computation():
    rng = np.random.default_rng(1)
    for _ in range(5):
        X = {v: rng.standard_normal((20, 8)) + rng.standard_normal(8) for v in range(120)}
        means = np.stack([X[v].mean(axis=0) for v in range(120)])
        assert medoid_root(list(range(120)), X) == int(np.argmin(cdist(means, means).sum(axis=1)))


def test_exact_lists_match_brute_force():
    G = np.random.default_rng(2).standard_normal((400, 30))
    lab = knn_lists(G, k=10, method="exact")
    D = cdist(G, G)
    np.fill_diagonal(D, np.inf)
    assert np.array_equal(lab, np.argsort(D, axis=1)[:, :10])


def test_hnsw_lists_have_high_recall():
    pytest.importorskip("hnswlib")
    G = np.random.default_rng(3).standard_normal((2000, 20))
    lab = knn_lists(G, k=10, method="hnsw")
    ref = knn_lists(G, k=10, method="exact")
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(lab, ref)])
    assert recall > 0.95


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_lists_reproduce_dense_trees_on_syn(seed):
    tasks = subspace_collection(seed=seed)
    lab = knn_lists(tasks.proxies, k=20, method="exact")
    for method in ("Chain", "CTL-MST", "CTL-Forest"):
        dense = CONSTRUCTIONS[method](tasks.nodes, tasks.D, tasks.root)
        sparse = build_knn_structure(method, tasks.nodes, tasks.proxies, tasks.root, lab=lab)
        assert _edges(sparse) == _edges(dense), method


def test_loader_keeps_proxies_consistent_with_D():
    tasks = subspace_collection(seed=0, T=80)
    assert tasks.proxies.shape == (80, tasks.d)
    assert np.allclose(gram_euclidean(tasks.proxies), tasks.D, atol=1e-12)
    D, G = gradient_geometry(tasks.nodes, tasks.X_train, tasks.y_train, dense_max=50)
    assert D is None and np.array_equal(G, tasks.proxies)


def test_collection_without_dense_distances_runs_every_sparse_method():
    dense = subspace_collection(seed=0, T=120)
    tasks = dataclasses.replace(dense, D=None)
    cfg = get_config("Syn")
    B = 3 * tasks.T
    lengths = EdgeLengths(tasks.proxies)
    assert lengths[0, 5] == pytest.approx(dense.D[0, 5], abs=1e-12)
    for method in ("Star", "Chain", "Chain-2opt", "CTL-MST", "CTL-Forest"):
        tree = build_structure(method, tasks, cfg)
        assert sorted(tree.nodes) == sorted(tasks.nodes) and len(tree.edges) == tasks.T - 1
        assert sum(allocate(method, tree, tasks, B, cfg).values()) == B
        models = run_method(method, tasks, B, cfg.eta, cfg, model_factory("linear", tasks.d))
        assert len(models) == tasks.T
    with pytest.raises(ValueError):
        build_structure("kNN-BFS", tasks, cfg)
