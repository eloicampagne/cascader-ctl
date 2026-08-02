"""Each distance matches the formula printed in the appendix.

Every check re-derives the value from the formula on a two- or three-task
example, so a silent change of definition fails here rather than showing up as a
slightly different table three months later.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascader.distances import (DISTANCES, compute_cka_distance,
                                compute_feature_distance, compute_gradient_distance,
                                compute_js_distance, compute_kl_distance,
                                compute_mean_cov_distance, compute_model_distance,
                                compute_target_distance,
                                compute_wasserstein_distance, distance_matrix,
                                symmetrize)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    nodes = ["a", "b", "c"]
    X = {v: rng.normal(size=(20, 4)) for v in nodes}
    y = {v: rng.normal(size=20) for v in nodes}
    return nodes, X, y


# --------------------------------------------------------------------------
# Shared shape contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DISTANCES))
def test_every_distance_is_symmetric_with_a_zero_diagonal(toy, name):
    nodes, X, y = toy
    D = distance_matrix(name, nodes, X, y)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(D, D.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
    assert np.all(D >= -1e-12), "negative edge weights are not usable as weights"


def test_unknown_distance_names_are_rejected(toy):
    nodes, X, y = toy
    with pytest.raises(KeyError, match="gradient"):
        distance_matrix("does-not-exist", nodes, X, y)


def test_symmetrize_zeroes_the_diagonal():
    D = np.array([[3.0, 1.0], [5.0, 2.0]])
    S = symmetrize(D)
    assert S[0, 1] == S[1, 0] == 3.0
    np.testing.assert_allclose(np.diag(S), 0.0)


# --------------------------------------------------------------------------
# Optimization geometry
# --------------------------------------------------------------------------

def test_gradient_distance_is_the_published_formula(toy):
    r"""g_v = X_v^T y_v, l2-normalized, then Euclidean distance between the g's."""
    nodes, X, y = toy
    D = compute_gradient_distance(nodes, X, y, normalize=True)
    g = {v: X[v].T @ y[v] for v in nodes}
    g = {v: g[v] / (np.linalg.norm(g[v]) + 1e-8) for v in nodes}
    assert D[0, 1] == pytest.approx(np.linalg.norm(g["a"] - g["b"]))
    assert D[0, 2] == pytest.approx(np.linalg.norm(g["a"] - g["c"]))


def test_normalized_gradient_distance_ignores_the_target_scale(toy):
    """Rescaling a task's targets rescales g, and the normalized distance is blind."""
    nodes, X, y = toy
    scaled = dict(y)
    scaled["b"] = y["b"] * 17.0
    a = compute_gradient_distance(nodes, X, y, normalize=True)
    b = compute_gradient_distance(nodes, X, scaled, normalize=True)
    np.testing.assert_allclose(a, b, atol=1e-6)

    unnormalized = compute_gradient_distance(nodes, X, scaled, normalize=False)
    assert not np.allclose(unnormalized, compute_gradient_distance(
        nodes, X, y, normalize=False))


def test_model_distance_is_the_ridge_solution_gap(toy):
    nodes, X, y = toy
    lam = 1e-2
    D = compute_model_distance(nodes, X, y, regularization=lam)
    theta = {v: np.linalg.solve(X[v].T @ X[v] + lam * np.eye(4), X[v].T @ y[v])
             for v in nodes}
    assert D[0, 1] == pytest.approx(np.linalg.norm(theta["a"] - theta["b"]))


def test_model_distance_recovers_the_true_parameter_gap():
    """With ample data and no noise, ridge solutions sit on the true thetas."""
    rng = np.random.default_rng(0)
    theta = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 3.0])}
    X = {v: rng.normal(size=(4000, 2)) for v in theta}
    y = {v: X[v] @ theta[v] for v in theta}
    D = compute_model_distance(["a", "b"], X, y, regularization=1e-8)
    assert D[0, 1] == pytest.approx(np.linalg.norm(theta["a"] - theta["b"]), rel=1e-3)


# --------------------------------------------------------------------------
# Feature based
# --------------------------------------------------------------------------

def test_feature_distance_is_normalized_by_the_dimension(toy):
    nodes, X, y = toy
    D = compute_feature_distance(nodes, X, y)
    flat = {v: X[v].ravel() for v in nodes}
    expected = np.linalg.norm(flat["a"] - flat["b"]) / np.sqrt(flat["a"].size)
    assert D[0, 1] == pytest.approx(expected)


def test_feature_distance_falls_back_when_shapes_differ():
    rng = np.random.default_rng(0)
    nodes = ["a", "b"]
    X = {"a": rng.normal(size=(10, 3)), "b": rng.normal(size=(25, 3))}
    D = compute_feature_distance(nodes, X, None)
    assert D.shape == (2, 2) and D[0, 1] > 0


def test_mean_cov_distance_adds_the_two_terms():
    rng = np.random.default_rng(0)
    nodes = ["a", "b"]
    X = {"a": rng.normal(size=(200, 3)), "b": rng.normal(size=(200, 3)) + 5.0}
    D = compute_mean_cov_distance(nodes, X, None)
    mu = np.linalg.norm(X["a"].mean(0) - X["b"].mean(0))
    cov = np.linalg.norm(np.cov(X["a"], rowvar=False) - np.cov(X["b"], rowvar=False))
    assert D[0, 1] == pytest.approx(mu + cov)


def test_cka_is_zero_for_a_rotation_and_positive_otherwise():
    """CKA compares kernels, so an orthogonal transform leaves it unchanged."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(30, 4))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    D = compute_cka_distance(["a", "b", "c"], {"a": Z, "b": Z @ Q,
                                               "c": rng.normal(size=(30, 4))}, None)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert D[0, 2] > 0.1


def test_cka_requires_a_common_sample_count():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="identical sample count"):
        compute_cka_distance(["a", "b"], {"a": rng.normal(size=(10, 2)),
                                          "b": rng.normal(size=(11, 2))}, None)


# --------------------------------------------------------------------------
# Target based
# --------------------------------------------------------------------------

def test_target_distance_is_the_euclidean_gap(toy):
    nodes, X, y = toy
    D = compute_target_distance(nodes, X, y)
    assert D[0, 1] == pytest.approx(np.linalg.norm(y["a"] - y["b"]))


def test_kl_is_symmetric_and_js_is_bounded(toy):
    nodes, X, y = toy
    kl = compute_kl_distance(nodes, X, y)
    js = compute_js_distance(nodes, X, y)
    np.testing.assert_allclose(kl, kl.T)
    assert np.all(js <= np.sqrt(np.log(2)) + 1e-9), "JS distance exceeds its bound"
    assert np.all(js >= 0)


def test_identical_targets_have_zero_divergence():
    y = {"a": np.linspace(0, 1, 100), "b": np.linspace(0, 1, 100)}
    X = {"a": np.zeros((100, 1)), "b": np.zeros((100, 1))}
    assert compute_kl_distance(["a", "b"], X, y)[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert compute_js_distance(["a", "b"], X, y)[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_matches_a_shift():
    """W1 between a sample and the same sample shifted by c equals c."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=5000)
    D = compute_wasserstein_distance(["a", "b"], None, {"a": v, "b": v + 2.0})
    assert D[0, 1] == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------
# The claim that justifies the default
# --------------------------------------------------------------------------

def test_gradient_distance_tracks_the_model_distance():
    """The appendix picks the gradient proxy for ranking closest to the oracle.

    Built here on tasks whose true parameters are known, so the ordering the
    proxy has to recover is not in doubt.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    d, T = 5, 30
    thetas = rng.normal(size=(T, d)) * 2.0
    nodes = list(range(T))
    X = {v: rng.normal(size=(200, d)) for v in nodes}
    y = {v: X[v] @ thetas[v] + rng.normal(scale=0.1, size=200) for v in nodes}

    oracle = np.linalg.norm(thetas[:, None, :] - thetas[None, :, :], axis=-1)
    grad = compute_gradient_distance(nodes, X, y)
    off = np.triu_indices(T, k=1)
    rho = spearmanr(grad[off], oracle[off]).statistic
    assert rho > 0.6, f"gradient proxy correlates only {rho:.2f} with the oracle"
