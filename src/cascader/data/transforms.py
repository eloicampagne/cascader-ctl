"""Per-task normalization, shared by every loader.

Normalization is per task, not global: each task standardizes its own features
and target, and the test split is mapped through the training statistics. The
statistics travel with the collection so regression scores can be reported in
the units of the data.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Array = np.ndarray

__all__ = ["normalize_xy", "denormalize_y", "standardize"]


def normalize_xy(X: Array, y: Array, eps: float = 1e-8) -> Tuple[Array, Array, Dict]:
    """Standardize features column-wise and the target globally.

    Returns ``(X_norm, y_norm, stats)`` where ``stats`` holds ``X_mean``,
    ``X_std``, ``y_mean`` and ``y_std``. ``eps`` guards constant columns.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + eps
    # kept as numpy scalars on purpose: denormalization multiplies a float32
    # prediction by y_std, and a Python float would leave the product in
    # float32 while a numpy float64 promotes it. The choice is worth 1e-8 on a
    # reported RMSE, which is enough to break an equality against the reference.
    y_mean = y.mean()
    y_std = y.std() + eps
    stats = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}
    return (X - X_mean) / X_std, (y - y_mean) / y_std, stats


def denormalize_y(y_norm: Array, stats: Dict) -> Array:
    """Map normalized predictions back to the original target scale."""
    return np.asarray(y_norm) * stats["y_std"] + stats["y_mean"]


def standardize(X_train: Array, X_test: Array) -> Tuple[Array, Array]:
    """Standardize a train/test pair using the training statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std
