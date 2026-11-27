"""Pairwise task-distance proxies.

Every distance defined in the paper's appendix "Distance metrics for MST-based
cascades" is implemented here, with the same formula and the same name. Each
function maps an ordered list of task identifiers and the per-task data to a
symmetric ``(T, T)`` matrix with a zero diagonal, usable directly as edge
weights for a spanning-tree construction.

Some of these are divergences rather than metrics. They are used as nonnegative
edge weights, which is all a spanning tree requires.

``gradient`` is the default of the paper: it scores highest against the model
distance in the appendix audit and is the one every reported cascade uses.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Sequence

import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    "compute_gradient_distance",
    "gradient_proxies",
    "gradient_geometry",
    "gram_euclidean",
    "DENSE_MAX",
    "compute_feature_distance",
    "compute_target_distance",
    "compute_model_distance",
    "compute_mean_cov_distance",
    "compute_cka_distance",
    "compute_kl_distance",
    "compute_js_distance",
    "compute_wasserstein_distance",
    "symmetrize",
    "distance_matrix",
    "DISTANCES",
]

Array = np.ndarray
NodeDict = Dict[Hashable, Array]


def symmetrize(D: Array) -> Array:
    """Return ``(D + D.T) / 2`` with an exactly zero diagonal."""
    D = np.asarray(D, dtype=float)
    S = 0.5 * (D + D.T)
    np.fill_diagonal(S, 0.0)
    return S


def _pairwise_euclidean(E: Array) -> Array:
    """Euclidean distances between the rows of ``E``.

    Computed by :func:`scipy.spatial.distance.cdist`, which evaluates the norm of
    each difference (same values as broadcasting) in ``O(T^2)`` memory instead of
    the ``O(T^2 p)`` a broadcast ``E[:, None] - E[None]`` would allocate.
    """
    D = cdist(np.asarray(E, dtype=float), np.asarray(E, dtype=float))
    np.fill_diagonal(D, 0.0)
    return D


_BLOCK_ENTRIES = 8_000_000


def gram_euclidean(E: Array) -> Array:
    r"""Euclidean distances between the rows of ``E`` through the Gram matrix.

    Uses :math:`\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\top b`,
    so the work is one matrix product, which BLAS runs far faster than the
    pairwise loop of :func:`scipy.spatial.distance.cdist`. The rows are cast to
    float64 first: in float32 the cancellation for nearby rows reaches about
    :math:`10^{-5}` and changes some trees, whereas in float64 the result matches
    ``cdist`` to about :math:`10^{-14}` and gives the same trees on every
    collection of the paper. The output is exactly symmetric with a zero diagonal.

    The result is assembled in place over blocks of rows, so the peak memory is
    one ``(T, T)`` array plus one block.
    """
    E = np.asarray(E, dtype=np.float64)
    T = E.shape[0]
    sq = np.einsum("ij,ij->i", E, E)
    D = E @ E.T
    bs = max(1, _BLOCK_ENTRIES // max(1, T))
    for s in range(0, T, bs):
        blk = D[s:s + bs]
        blk *= 2.0
        np.subtract(sq[s:s + bs, None] + sq[None, :], blk, out=blk)
        np.maximum(blk, 0.0, out=blk)
        np.sqrt(blk, out=blk)
    # Mirror the upper triangle so that symmetry is exact whatever the BLAS.
    for s in range(0, T, bs):
        D[s:s + bs, :s] = D[:s, s:s + bs].T
    np.fill_diagonal(D, 0.0)
    return D


#: Largest number of tasks for which the loaders store the dense ``(T, T)``
#: distance matrix (about 3.2 GB in float64 at this size). Above it, a
#: collection keeps only the gradient proxies and the cascade is built from
#: nearest-neighbour lists (:mod:`cascader.neighbors`).
DENSE_MAX = 20000


# --------------------------------------------------------------------------
# Optimization-geometry distances
# --------------------------------------------------------------------------

def gradient_proxies(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict,
    normalize: bool = True,
) -> Array:
    r"""Gradient proxies :math:`g_v = X_v^\top y_v`, one row per task, as float64.

    With ``normalize=True`` each row is scaled to unit :math:`\ell_2` norm. The
    product and the normalization run in the dtype of the data, as they always
    have, and only the result is cast, so the distances are unchanged.
    """
    G = np.stack([np.asarray(X[v]).T @ np.asarray(y[v]).reshape(-1) for v in nodes])
    if normalize:
        G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return np.asarray(G, dtype=np.float64)


def compute_gradient_distance(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict,
    normalize: bool = True,
) -> Array:
    r"""Gradient distance :math:`d(u,v) = \lVert g_u - g_v \rVert`.

    The gradient of the squared loss at :math:`\theta = 0` is
    :math:`g_v = X_v^\top y_v`, up to a factor absorbed by the normalization.
    With ``normalize=True`` (the paper's default and the setting behind every
    reported result) each :math:`g_v` is scaled to unit :math:`\ell_2` norm
    first, so the distance compares gradient *directions* and is invariant to
    the per-task target scale. Distances are computed by :func:`gram_euclidean`.
    """
    return gram_euclidean(gradient_proxies(nodes, X, y, normalize))


def gradient_geometry(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict,
    dense_max: int | None = None,
):
    """Return ``(D, G)``: the dense gradient distance and the proxies.

    ``D`` is ``None`` when the collection has more than ``dense_max`` tasks
    (default :data:`DENSE_MAX`); the cascade is then built from neighbour lists
    over ``G``, without ever forming the ``(T, T)`` matrix.
    """
    G = gradient_proxies(nodes, X, y)
    limit = DENSE_MAX if dense_max is None else dense_max
    D = gram_euclidean(G) if len(G) <= limit else None
    return D, G


def compute_model_distance(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict,
    regularization: float = 1e-2,
) -> Array:
    r"""Model distance :math:`\lVert \hat T_u - \hat T_v \rVert` on ridge solutions.

    :math:`\hat T_v = (X_v^\top X_v + \lambda I)^{-1} X_v^\top y_v`. In the
    linear setting this is the analytical per-task optimum, so the model
    distance is the oracle the appendix audit ranks the cheap proxies against.
    It is not usable as a construction distance at the budgets the paper
    studies, since computing it already solves every task.
    """
    thetas = []
    for v in nodes:
        Xv = np.asarray(X[v], dtype=float)
        yv = np.asarray(y[v], dtype=float).reshape(-1)
        d = Xv.shape[1]
        A = Xv.T @ Xv + regularization * np.eye(d)
        thetas.append(np.linalg.solve(A, Xv.T @ yv))
    return _pairwise_euclidean(np.stack(thetas))


# --------------------------------------------------------------------------
# Feature-based distances
# --------------------------------------------------------------------------

def compute_feature_distance(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict | None = None,
) -> Array:
    r"""Normalized Euclidean distance between flattened feature matrices.

    :math:`d(u,v) = \sqrt{\lVert \mathrm{vec}(X_u) - \mathrm{vec}(X_v)\rVert^2 / D}`
    when all tasks share a sample count, with ``D`` the number of overlapping
    dimensions. When the shapes differ, each task falls back to the
    mean-and-variance embedding described in the same appendix entry.
    """
    shapes = {np.asarray(X[v]).shape for v in nodes}
    if len(shapes) == 1:
        E = np.stack([np.asarray(X[v], dtype=float).ravel() for v in nodes])
        return _pairwise_euclidean(E) / np.sqrt(E.shape[1])
    E = np.stack([
        np.concatenate([np.asarray(X[v], dtype=float).mean(axis=0),
                        np.asarray(X[v], dtype=float).var(axis=0)])
        for v in nodes
    ])
    return _pairwise_euclidean(E) / np.sqrt(E.shape[1])


def compute_mean_cov_distance(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict | None = None,
) -> Array:
    r"""Gaussian mean--covariance distance.

    :math:`d(u,v) = \lVert \mu_u - \mu_v \rVert + \lVert S_u - S_v \rVert_F`,
    approximating each task's feature distribution by
    :math:`\mathcal{N}(\mu_v, S_v)`.
    """
    mus, covs = [], []
    for v in nodes:
        Xv = np.asarray(X[v], dtype=float)
        mus.append(Xv.mean(axis=0))
        covs.append(np.cov(Xv, rowvar=False))
    mus = np.stack(mus)
    covs = np.stack([np.atleast_2d(c) for c in covs])
    d_mu = _pairwise_euclidean(mus)
    flat = covs.reshape(len(nodes), -1)
    d_cov = _pairwise_euclidean(flat)
    return d_mu + d_cov


def compute_cka_distance(
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict | None = None,
) -> Array:
    r"""Centered-kernel-alignment dissimilarity :math:`1 - \mathrm{CKA}(u,v)`.

    Linear kernels :math:`K_v = Z_v Z_v^\top` on centered representations,
    centered again by :math:`H = I - \tfrac1n \mathbf{1}\mathbf{1}^\top`, with
    :math:`\mathrm{HSIC}(K_u, K_v) = \mathrm{tr}(\tilde K_u \tilde K_v)` and
    :math:`\mathrm{CKA} = \mathrm{HSIC}(K_u,K_v) /
    \sqrt{\mathrm{HSIC}(K_u,K_u)\mathrm{HSIC}(K_v,K_v)}`.

    All tasks must share a sample count, since CKA compares kernels over the
    same index set.
    """
    Ks = []
    n_ref = None
    for v in nodes:
        Z = np.asarray(X[v], dtype=float)
        n = Z.shape[0]
        if n_ref is None:
            n_ref = n
        elif n != n_ref:
            raise ValueError("CKA requires an identical sample count across tasks")
        Z = Z - Z.mean(axis=0, keepdims=True)
        K = Z @ Z.T
        H = np.eye(n) - np.ones((n, n)) / n
        Ks.append(H @ K @ H)
    T = len(nodes)
    self_hsic = np.array([float(np.sum(K * K)) for K in Ks])
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            denom = np.sqrt(self_hsic[i] * self_hsic[j]) + 1e-12
            cka = float(np.sum(Ks[i] * Ks[j])) / denom
            D[i, j] = D[j, i] = 1.0 - cka
    return D


# --------------------------------------------------------------------------
# Target-based distances
# --------------------------------------------------------------------------

def compute_target_distance(
    nodes: Sequence[Hashable],
    X: NodeDict | None,
    y: NodeDict,
) -> Array:
    r"""Target distance :math:`\lVert y_u - y_v \rVert`.

    Defined only when the tasks share a sample count; otherwise the targets are
    compared through their sorted quantiles, which reduces to the same value
    under a common ordering.
    """
    lens = {np.asarray(y[v]).reshape(-1).size for v in nodes}
    if len(lens) == 1:
        E = np.stack([np.asarray(y[v], dtype=float).reshape(-1) for v in nodes])
    else:
        q = np.linspace(0.0, 1.0, 64)
        E = np.stack([np.quantile(np.asarray(y[v], dtype=float).reshape(-1), q)
                      for v in nodes])
    return _pairwise_euclidean(E)


def _histograms(nodes, y, bins: int = 50, eps: float = 1e-10) -> Array:
    vals = [np.asarray(y[v], dtype=float).reshape(-1) for v in nodes]
    lo = min(float(v.min()) for v in vals)
    hi = max(float(v.max()) for v in vals)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    P = np.stack([np.histogram(v, bins=edges)[0].astype(float) for v in vals])
    P += eps
    return P / P.sum(axis=1, keepdims=True)


def compute_kl_distance(
    nodes: Sequence[Hashable],
    X: NodeDict | None,
    y: NodeDict,
    bins: int = 50,
) -> Array:
    r"""Symmetric KL divergence between smoothed target histograms.

    :math:`d(u,v) = \tfrac12(\mathrm{KL}(p_u \| p_v) + \mathrm{KL}(p_v \| p_u))`.
    """
    P = _histograms(nodes, y, bins)
    T = len(nodes)
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            kl_ij = float(np.sum(P[i] * np.log(P[i] / P[j])))
            kl_ji = float(np.sum(P[j] * np.log(P[j] / P[i])))
            D[i, j] = D[j, i] = 0.5 * (kl_ij + kl_ji)
    return D


def compute_js_distance(
    nodes: Sequence[Hashable],
    X: NodeDict | None,
    y: NodeDict,
    bins: int = 50,
) -> Array:
    r"""Jensen--Shannon distance between smoothed target histograms.

    :math:`d(u,v) = \sqrt{\tfrac12\mathrm{KL}(p_u\|m) + \tfrac12\mathrm{KL}(p_v\|m)}`
    with :math:`m = \tfrac12(p_u + p_v)`.
    """
    P = _histograms(nodes, y, bins)
    T = len(nodes)
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            m = 0.5 * (P[i] + P[j])
            js = 0.5 * float(np.sum(P[i] * np.log(P[i] / m))) \
                + 0.5 * float(np.sum(P[j] * np.log(P[j] / m)))
            D[i, j] = D[j, i] = float(np.sqrt(max(js, 0.0)))
    return D


def compute_wasserstein_distance(
    nodes: Sequence[Hashable],
    X: NodeDict | None,
    y: NodeDict,
) -> Array:
    r"""1-Wasserstein distance between empirical target distributions."""
    from scipy.stats import wasserstein_distance

    vals = [np.asarray(y[v], dtype=float).reshape(-1) for v in nodes]
    T = len(nodes)
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            D[i, j] = D[j, i] = float(wasserstein_distance(vals[i], vals[j]))
    return D


#: Name -> callable(nodes, X, y) -> (T, T) distance matrix.
DISTANCES: Dict[str, Callable[..., Array]] = {
    "gradient": compute_gradient_distance,
    "model": compute_model_distance,
    "feature": compute_feature_distance,
    "mean_cov": compute_mean_cov_distance,
    "cka": compute_cka_distance,
    "target": compute_target_distance,
    "kl": compute_kl_distance,
    "js": compute_js_distance,
    "wasserstein": compute_wasserstein_distance,
}


def distance_matrix(
    kind: str,
    nodes: Sequence[Hashable],
    X: NodeDict,
    y: NodeDict,
    **kwargs: Any,
) -> Array:
    """Dispatch to a named distance and return a symmetric zero-diagonal matrix.

    Raises ``KeyError`` with the list of valid names when ``kind`` is unknown.
    """
    try:
        fn = DISTANCES[kind]
    except KeyError:
        raise KeyError(
            f"unknown distance {kind!r}; choose from {sorted(DISTANCES)}"
        ) from None
    return symmetrize(fn(list(nodes), X, y, **kwargs))
