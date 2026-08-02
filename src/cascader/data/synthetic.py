"""Syn-10: clustered linear regression tasks with controlled geometry.

Each task ``t`` has its own parameter vector drawn around one of ``K`` cluster
centres, which are themselves drawn around a global vector::

    theta_c   ~ N(theta_0, tau_between^2 I)      c = 1..K
    theta_t   ~ N(theta_{c(t)}, tau_within^2 I)
    y         = X theta_t + eps,    eps ~ N(0, sigma^2)

The ratio ``tau_between / tau_within`` is the scatter that decides which
construction wins: clustered geometry at tight budget favours the tree, and the
chain takes over as either the budget grows or the clusters merge. Sweeping it
is :mod:`cascader.experiments.tworegime`.

The published Syn-10 configuration is the default of :func:`syn_collection`:
``T=200, d=3, K=5, tau_within=tau_between=10, sigma=5, n_train=64, n_test=128``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..distances import compute_gradient_distance
from ..rooting import medoid_root
from ..tasks import TaskCollection
from .transforms import normalize_xy

__all__ = ["generate_clustered_tasks", "syn_collection"]

Array = np.ndarray


def generate_clustered_tasks(T: int = 200, d: int = 3, n_train: int = 64,
                             n_test: int = 128, n_clusters: int = 5,
                             tau_within: float = 10.0, tau_between: float = 10.0,
                             sigma: float = 5.0,
                             rng: Optional[np.random.Generator] = None):
    """Draw ``T`` clustered linear tasks.

    Returns ``(thetas, theta_global, cluster_centers, (X_tr, y_tr),
    (X_te, y_te), cluster_ids)`` with per-task arrays stacked along axis 0.
    """
    rng = np.random.default_rng(rng)
    theta_global = rng.normal(size=(d,))
    cluster_centers = theta_global + rng.normal(scale=tau_between, size=(n_clusters, d))
    cluster_ids = rng.integers(0, n_clusters, size=T)
    thetas = np.stack([
        cluster_centers[cluster_ids[t]] + rng.normal(scale=tau_within, size=(d,))
        for t in range(T)
    ])

    def split(n: int) -> Tuple[Array, Array]:
        X = rng.normal(size=(T, n, d))
        y = np.einsum("tnd,td->tn", X, thetas) + rng.normal(scale=sigma, size=(T, n))
        return X, y

    return (thetas, theta_global, cluster_centers, split(n_train), split(n_test),
            cluster_ids)


def syn_collection(seed: int = 0, T: int = 200, d: int = 3, n_clusters: int = 5,
                   tau_within: float = 10.0, tau_between: float = 10.0,
                   sigma: float = 5.0, n_train: int = 64,
                   n_test: int = 128) -> TaskCollection:
    """The published Syn-10 collection at a given seed.

    Features and targets are standardized per task, the test features being
    mapped through the training statistics; the reported metric is then
    denormalized back to the original scale. The distance is the gradient one
    on normalized data, the root the raw-feature medoid.
    """
    rng = np.random.default_rng(seed)
    _, _, _, (Xt, yt), (Xe, ye), clusters = generate_clustered_tasks(
        T=T, d=d, n_train=n_train, n_test=n_test, n_clusters=n_clusters,
        tau_within=tau_within, tau_between=tau_between, sigma=sigma, rng=rng)

    nodes = list(range(T))
    X_raw = {v: Xt[v] for v in nodes}
    X_tr, y_tr, X_te, stats = {}, {}, {}, {}
    for v in nodes:
        X_tr[v], y_tr[v], st = normalize_xy(Xt[v], yt[v])
        stats[v] = st
        X_te[v] = (Xe[v] - st["X_mean"]) / st["X_std"]

    D = compute_gradient_distance(nodes, X_tr, y_tr)
    return TaskCollection(
        nodes=nodes, X_train=X_tr, y_train=y_tr, X_test=X_te,
        y_test={v: ye[v] for v in nodes}, D=D,
        root=medoid_root(nodes, X_raw), task="regression", stats=stats,
        name="Syn-10", meta={"cluster_ids": clusters, "seed": seed})
