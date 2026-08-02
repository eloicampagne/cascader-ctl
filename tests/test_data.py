"""Collections load, and carry the invariants the rest of the code assumes."""

from __future__ import annotations

import numpy as np
import pytest

from cascader.data import COLLECTIONS, MissingDataError, load
from cascader.data.synthetic import generate_clustered_tasks, syn_collection
from cascader.data.transforms import denormalize_y, normalize_xy
from cascader.data.weave import weave_collection


def test_collection_invariants(small):
    assert len(small.nodes) == small.T == len(set(small.nodes))
    assert small.D.shape == (small.T, small.T)
    np.testing.assert_allclose(small.D, small.D.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(small.D), 0.0, atol=1e-10)
    assert small.root in small.nodes
    assert small.task in ("regression", "classification")
    for v in small.nodes:
        assert small.X_train[v].shape[1] == small.d == small.X_test[v].shape[1]
        assert len(small.X_train[v]) == len(small.y_train[v])
        assert len(small.X_test[v]) == len(small.y_test[v])


def test_index_aligns_with_the_distance_matrix(small):
    idx = small.index
    assert [small.nodes[idx[v]] for v in small.nodes] == list(small.nodes)


def test_synthetic_draws_are_reproducible():
    a = syn_collection(seed=3, T=10, d=2)
    b = syn_collection(seed=3, T=10, d=2)
    np.testing.assert_allclose(a.D, b.D)
    assert a.root == b.root


def test_synthetic_seeds_differ():
    a = syn_collection(seed=1, T=10, d=2)
    b = syn_collection(seed=2, T=10, d=2)
    assert not np.allclose(a.D, b.D)


def test_scatter_controls_the_cluster_separation():
    """Between-cluster spread is what tau_between sets; the distances follow."""
    tight = syn_collection(seed=0, T=60, d=4, n_clusters=4, tau_within=1.0,
                           tau_between=0.5)
    wide = syn_collection(seed=0, T=60, d=4, n_clusters=4, tau_within=1.0,
                          tau_between=20.0)
    off = np.triu_indices(60, k=1)
    assert np.std(wide.D[off]) > np.std(tight.D[off])


def test_generated_targets_follow_the_task_parameters():
    thetas, _, _, (X_tr, y_tr), _, _ = generate_clustered_tasks(
        T=5, d=3, n_train=200, n_test=10, n_clusters=2, sigma=0.0,
        rng=np.random.default_rng(0))
    for t in range(5):
        np.testing.assert_allclose(y_tr[t], X_tr[t] @ thetas[t], atol=1e-9)


def test_normalization_round_trips():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 3)) * 3 + 7, rng.normal(size=50) * 5 - 2
    Xn, yn, stats = normalize_xy(X, y)
    np.testing.assert_allclose(Xn.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(yn.mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(denormalize_y(yn, stats), y, atol=1e-6)


def test_per_task_normalization_is_per_task(small):
    """Each task standardizes its own split, so its training target is centered."""
    for v in small.nodes[:5]:
        assert abs(float(np.mean(small.y_train[v]))) < 1e-6


def test_test_features_use_the_training_statistics(small):
    """A test split centered on its own mean would leak; it must not be."""
    offsets = [abs(float(np.mean(small.X_test[v]))) for v in small.nodes]
    assert max(offsets) > 1e-6


def test_bundled_weave_sample_loads_offline():
    tasks = weave_collection(sample=True)
    assert tasks.name == "WEAVE-UK-sample"
    assert tasks.T == 50
    assert tasks.task == "regression"
    assert tasks.d == 54          # 48 lags + 6 calendar terms
    assert tasks.stats is not None


def test_weave_validation_split_holds_out_training_rows():
    full = weave_collection(sample=True)
    held = weave_collection(sample=True, val_size=24)
    v = held.nodes[0]
    assert len(held.X_test[v]) == 24
    assert len(held.X_train[v]) == len(full.X_train[v]) - 24


def test_a_missing_collection_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADER_DATA_DIR", str(tmp_path))
    with pytest.raises(MissingDataError) as excinfo:
        weave_collection(sample=False)
    assert "CASCADER_DATA_DIR" in str(excinfo.value)


def test_unknown_collection_names_are_rejected():
    with pytest.raises(KeyError):
        load("not-a-collection")


def test_the_catalogue_matches_the_configured_collections():
    from cascader.config import CONFIGS

    assert set(COLLECTIONS) == set(CONFIGS)
