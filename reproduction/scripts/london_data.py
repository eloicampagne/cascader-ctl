"""
London daily household electricity as a many-task collection with a genuinely
non-linear target (rebuttal follow-up on the data-linearity objection).

WEAVE-UK forecasts half-hourly consumption from 48 recent lags, which is close to
a first-order expansion and therefore nearly linear. The London dataset
(data/london/daily_dataset.csv, 5566 households, 2011-11 to 2014-02) is daily,
so the one-step horizon is a full day: lag structure is far weaker, and the
signal is dominated by seasonality and household occupancy patterns, which enter
non-linearly.

Tasks are households. Target is daily energy_sum. Features are recent lags, weekly
and monthly rolling statistics, and calendar variables. Chronological split, so
the test period follows the training period.

Cached to london_cache/ on first use.
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
PROJECT = HERE.parent.parent.parent
RAW = PROJECT / "data" / "london" / "daily_dataset.csv"
CACHE = HERE / "london_cache"


def _build(n_tasks=500, min_days=700, n_test_days=120):
    df = pd.read_csv(RAW, usecols=["LCLid", "day", "energy_sum"])
    df["day"] = pd.to_datetime(df["day"])
    df = df.dropna().sort_values(["LCLid", "day"])
    counts = df.groupby("LCLid")["day"].nunique()
    keep = counts[counts >= min_days].index[:n_tasks]
    df = df[df.LCLid.isin(keep)]

    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    for tid, (hid, g) in enumerate(df.groupby("LCLid")):
        g = g.sort_values("day").reset_index(drop=True)
        s = g["energy_sum"].astype(np.float64)
        feats = pd.DataFrame(index=g.index)
        for L in (1, 2, 3, 4, 5, 6, 7, 14, 21, 28):
            feats[f"lag{L}"] = s.shift(L)
        feats["roll7"] = s.shift(1).rolling(7).mean()
        feats["roll28"] = s.shift(1).rolling(28).mean()
        feats["std7"] = s.shift(1).rolling(7).std()
        feats["min7"] = s.shift(1).rolling(7).min()
        feats["max7"] = s.shift(1).rolling(7).max()
        dow = g["day"].dt.dayofweek
        doy = g["day"].dt.dayofyear
        for k in range(7):
            feats[f"dow{k}"] = (dow == k).astype(float)
        feats["doy"] = doy / 366.0
        feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        feats["month"] = g["day"].dt.month / 12.0
        ok = feats.notna().all(axis=1)
        F = feats[ok].to_numpy(np.float32)
        t = s[ok].to_numpy(np.float32)
        if len(t) < min_days - 40:
            continue
        cut = len(t) - n_test_days
        X_tr[tid], y_tr[tid] = F[:cut], t[:cut]
        X_te[tid], y_te[tid] = F[cut:], t[cut:]
    return X_tr, y_tr, X_te, y_te


def load_london(n_tasks=500, normalize=True):
    """Returns nodes, X_tr, y_tr (normalized if asked), X_te, y_te_raw, stats."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"london_{n_tasks}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        X_tr, y_tr, X_te, y_te = (z[k].item() for k in ("X_tr", "y_tr", "X_te", "y_te"))
    else:
        X_tr, y_tr, X_te, y_te = _build(n_tasks=n_tasks)
        np.savez_compressed(f, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
    nodes = sorted(X_tr)
    if not normalize:
        return nodes, X_tr, y_tr, X_te, y_te, None
    Xn_tr, yn_tr, Xn_te, stats = {}, {}, {}, {}
    for v in nodes:
        mu, sd = X_tr[v].mean(0), X_tr[v].std(0) + 1e-8
        ym, ys = float(y_tr[v].mean()), float(y_tr[v].std() + 1e-8)
        Xn_tr[v] = ((X_tr[v] - mu) / sd).astype(np.float32)
        Xn_te[v] = ((X_te[v] - mu) / sd).astype(np.float32)
        yn_tr[v] = ((y_tr[v] - ym) / ys).astype(np.float32)
        stats[v] = {"X_mean": mu, "X_std": sd, "y_mean": ym, "y_std": ys}
    return nodes, Xn_tr, yn_tr, Xn_te, y_te, stats


if __name__ == "__main__":
    nodes, Xtr, ytr, Xte, yte, stats = load_london()
    print(f"tasks {len(nodes)}, d={Xtr[nodes[0]].shape[1]}, "
          f"train {Xtr[nodes[0]].shape[0]}, test {Xte[nodes[0]].shape[0]}")
