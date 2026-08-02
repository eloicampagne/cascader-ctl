"""WEAVE-UK: half-hourly electricity demand at 500 UK substations.

One task per substation, each an autoregressive forecast from 48 consumption
lags plus calendar features. The tasks are fixed, so repetitions vary only the
initialization; the panel itself does not resample.

The full collection is 135 MB of CSV and is not distributed with the package.
A 50-substation sample is bundled so the tests and the quick start run offline
-- use ``sample=True``, or leave it to the automatic fallback when no data
directory is configured. Results on the sample are not the paper's numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..distances import compute_gradient_distance
from ..rooting import medoid_root
from ..tasks import TaskCollection
from ._paths import MissingDataError, bundled_dir, data_dir, require
from .transforms import normalize_xy

__all__ = ["load_weave_csv", "weave_collection"]

Array = np.ndarray

_HOW = (
    "WEAVE-UK is not redistributed with the package. Obtain train_weave_500.csv "
    "and test_weave_500.csv from the authors, or pass sample=True to use the "
    "bundled 50-substation extract."
)


def load_weave_csv(csv_path: str | Path, node_col: str = "id_unique",
                   y_col: str = "consumption",
                   feature_prefix: str = "consumption_l"
                   ) -> Tuple[Dict, Dict]:
    """Read one WEAVE CSV into per-substation ``(X, y)`` dicts.

    Features are the lag columns plus four calendar terms derived from the
    timestamp: hour and day-of-week, each as a sine/cosine pair, followed by the
    raw hour and day-of-week. Rows with a non-finite entry are dropped.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ("Unnamed: 0",) if c in df.columns], errors="ignore")
    if node_col not in df.columns:
        raise ValueError(f"column {node_col!r} missing; found {df.columns.tolist()}")

    time_col = next((c for c in ("date", "timestamp", "time") if c in df.columns), None)
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"no feature column starts with {feature_prefix!r}")

    if time_col is not None:
        df = df.sort_values([node_col, time_col])
        ts = pd.to_datetime(df[time_col])
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        feature_cols += ["hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        df = df.sort_values([node_col])

    X_dict: Dict = {}
    y_dict: Dict = {}
    for v, g in df.groupby(node_col):
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[y_col].to_numpy(dtype=np.float32)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if mask.sum() == 0:
            continue
        X_dict[v], y_dict[v] = X[mask], y[mask]
    return X_dict, y_dict


_CACHE: Dict = {}


def weave_collection(path: Optional[str | Path] = None, sample: Optional[bool] = None,
                     val_size: int = 0) -> TaskCollection:
    """Load WEAVE-UK.

    Parameters
    ----------
    path
        Directory holding ``train_weave_500.csv`` and ``test_weave_500.csv``.
        Defaults to :func:`cascader.data._paths.data_dir`.
    sample
        ``True`` forces the bundled 50-substation extract, ``False`` forbids it.
        ``None`` uses the full collection when present and falls back to the
        sample otherwise.
    val_size
        When positive, hold out the first ``val_size`` training rows of each
        substation as a validation split and train on the rest. Used for
        learning-rate selection, where scoring on the test split would leak: the
        WEAVE tasks are fixed, so a rate chosen on test is a rate chosen on the
        reported number.
    """
    key = (str(path), sample, val_size)
    if key in _CACHE:
        return _CACHE[key]

    use_sample = sample
    if use_sample is None:
        try:
            require(data_dir(path) / "train_weave_500.csv", "WEAVE-UK", _HOW)
            use_sample = False
        except MissingDataError:
            use_sample = True

    if use_sample:
        base = bundled_dir()
        train_csv = base / "weave_sample_train.csv.gz"
        test_csv = base / "weave_sample_test.csv.gz"
        name = "WEAVE-UK-sample"
    else:
        base = data_dir(path)
        train_csv = require(base / "train_weave_500.csv", "WEAVE-UK train", _HOW)
        test_csv = require(base / "test_weave_500.csv", "WEAVE-UK test", _HOW)
        name = "WEAVE-UK"

    Xtr_raw, ytr_raw = load_weave_csv(train_csv)
    Xte_raw, yte_raw = load_weave_csv(test_csv)
    nodes = sorted(set(Xtr_raw) & set(Xte_raw))

    if val_size > 0:
        Xte_raw = {v: Xtr_raw[v][:val_size] for v in nodes}
        yte_raw = {v: ytr_raw[v][:val_size] for v in nodes}
        Xtr_raw = {v: Xtr_raw[v][val_size:] for v in nodes}
        ytr_raw = {v: ytr_raw[v][val_size:] for v in nodes}

    X_tr, y_tr, X_te, stats = {}, {}, {}, {}
    for v in nodes:
        X_tr[v], y_tr[v], st = normalize_xy(Xtr_raw[v], ytr_raw[v])
        stats[v] = st
        X_te[v] = (Xte_raw[v] - st["X_mean"]) / st["X_std"]

    coll = TaskCollection(
        nodes=nodes, X_train=X_tr, y_train=y_tr, X_test=X_te,
        y_test={v: yte_raw[v] for v in nodes},
        D=compute_gradient_distance(nodes, X_tr, y_tr),
        root=medoid_root(nodes, {v: Xtr_raw[v] for v in nodes}),
        task="regression", stats=stats, name=name,
        meta={"sample": use_sample, "val_size": val_size})
    _CACHE[key] = coll
    return coll
