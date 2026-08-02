"""Aggregate-as-root gap-filling on WEAVE, against the paper's CTL variants.

Scenario: the total (aggregate) load is metered completely, the individual
substations are observed only partially, and the goal is to fill the per-node
gaps. WEAVE is autoregressive (48 consumption lags + calendar) and lagging is
linear, so the aggregate is a WEAVE task at aggregate scale: same features, same
model class, homogeneous with the nodes. That makes the fully-observed aggregate a
natural CTL root.

We compare, all trained on each node's observed points and evaluated on its gaps:
  Individual            random init, B/T steps
  Star / Chain / CTL-MST / CTL-MSTc   the paper's constructions, node-rooted, SLE
  Agg-direct            the aggregate model applied as-is (0 node steps)
  Agg-Star              each node warm-started from the aggregate, then refined
  Agg-MST / Agg-MSTc    the paper's cascade, but the root inherits the aggregate

Report RMSE in consumption units (denormalised, averaged over nodes), so the
scale matches the paper's WEAVE tables.

    python3.12 weave_aggregate_root.py --f 0.3 --B 10000 --seeds 6
"""

import argparse
import copy
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

from table1_nonconvex import (make_model, train_steps, tree_mst, tree_chain,
                              tree_mstc, sle_alloc, uniform_alloc)
from graph import compute_gradient_distance

HERE = Path(__file__).parent.resolve()
CSV = HERE.parent.parent.parent / "CTL" / "data" / "train_weave_500.csv"
LAGS = [f"consumption_l{i}" for i in range(1, 49)]
CFG = dict(reg=0.0, b_proto_frac=0.1526, mstc_lambda=0.6)


def load():
    df = pd.read_csv(CSV)
    dt = pd.to_datetime(df["date"])
    df["hour_sin"] = np.sin(2 * np.pi * df["instant"] / 48)
    df["hour_cos"] = np.cos(2 * np.pi * df["instant"] / 48)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    cal = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    feat = LAGS + cal
    df = df.sort_values(["id_unique", "date"])
    nodes = sorted(df["id_unique"].unique())
    X = {v: g[feat].to_numpy(np.float32) for v, g in df.groupby("id_unique")}
    y = {v: g["consumption"].to_numpy(np.float32) for v, g in df.groupby("id_unique")}
    agg = df.groupby("date")
    Xa = np.concatenate([agg[LAGS].sum().to_numpy(np.float32),
                         agg[cal].first().to_numpy(np.float32)], axis=1)
    ya = agg["consumption"].sum().to_numpy(np.float32)
    return nodes, X, y, Xa, ya


def zfit(X, y):
    return X.mean(0), X.std(0) + 1e-8, y.mean(), y.std() + 1e-8


def zap(X, y, s):
    xm, xs, ym, ys = s
    return (X - xm) / xs, (y - ym) / ys


def train_lin(model, X, y, steps, eta):
    return train_steps(model, X, y, steps, eta, "regression", 0.0)


def rmse_cons(model, Xz, yz, ys):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xz).float()).numpy().reshape(-1)
    return float(np.sqrt(np.mean((pred - yz) ** 2))) * ys   # z-RMSE * y_std


def run(f, B, seeds, eta_grid):
    nodes, X, y, Xa, ya = load()
    T = len(nodes)
    d = X[nodes[0]].shape[1]
    sa = zfit(Xa, ya)
    Xaz, yaz = zap(Xa, ya, sa)
    torch.manual_seed(0)
    agg_model = train_lin(make_model("linear", d), Xaz, yaz, 400, 0.05)
    agg_state = agg_model.state_dict()

    def split(seed):
        rng = np.random.default_rng(seed)
        obs, gap, ysd = {}, {}, {}
        for v in nodes:
            n = len(y[v])
            k = max(d + 2, int(f * n))
            idx = rng.permutation(n)
            oi, gi = np.sort(idx[:k]), np.sort(idx[k:])
            s = zfit(X[v][oi], y[v][oi])
            ysd[v] = s[3]
            obs[v] = zap(X[v][oi], y[v][oi], s)
            gap[v] = zap(X[v][gi], y[v][gi], s)
        D = compute_gradient_distance(nodes, {v: obs[v][0] for v in nodes},
                                      {v: obs[v][1] for v in nodes})
        root = nodes[int(np.argmin(D.sum(1)))]
        return obs, gap, ysd, D, root

    def fit(init, v, steps, obs, seed):
        m = make_model("linear", d)
        if init is not None:
            m.load_state_dict(copy.deepcopy(init))
        else:
            torch.manual_seed(seed * 131 + (hash(str(v)) % 100000))
        return train_lin(m, obs[v][0], obs[v][1], int(max(0, steps)), eta_run[0])

    def cascade(dir_edges, root, alloc, root_init, obs, seed):
        ch = defaultdict(list)
        for u, w in dir_edges:
            ch[u].append(w)
        models = {root: fit(root_init, root, alloc.get(root, 0), obs, seed)}
        q = deque([root])
        while q:
            u = q.popleft()
            for w in ch[u]:
                models[w] = fit(models[u].state_dict(), w, alloc.get(w, 0), obs, seed)
                q.append(w)
        return models

    def evaluate(models, gap, ysd):
        return float(np.mean([rmse_cons(models[v], gap[v][0], gap[v][1], ysd[v])
                              for v in nodes]))

    def one(seed, methods):
        obs, gap, ysd, D, root = split(seed)
        b_ind = max(1, B // T)
        out = {}
        for m in methods:
            if m == "Individual":
                mdl = {v: fit(None, v, b_ind, obs, seed) for v in nodes}
            elif m == "Agg-direct":
                mdl = {v: agg_model for v in nodes}
            elif m == "Agg-Star":
                mdl = {v: fit(agg_state, v, b_ind, obs, seed) for v in nodes}
            elif m == "Star":
                de = [(root, v) for v in nodes if v != root]
                mdl = cascade(de, root, uniform_alloc(nodes, root, de, B,
                              CFG["b_proto_frac"]), None, obs, seed)
            elif m in ("Chain", "CTL-MST", "CTL-MSTc", "Agg-MST", "Agg-MSTc"):
                if m in ("CTL-MST", "Agg-MST"):
                    r, de, _ = tree_mst(nodes, D, root)
                elif m in ("CTL-MSTc", "Agg-MSTc"):
                    r, de, _ = tree_mstc(nodes, D, root, {v: obs[v][0] for v in nodes},
                                         {v: obs[v][1] for v in nodes}, CFG["mstc_lambda"])
                else:
                    r, de, _ = tree_chain(nodes, D, root)
                alloc = sle_alloc(nodes, r, de, D, B)
                init = agg_state if m.startswith("Agg") else None
                mdl = cascade(de, r, alloc, init, obs, seed)
            out[m] = evaluate(mdl, gap, ysd)
        return out

    methods = ["Individual", "Star", "Chain", "CTL-MST", "CTL-MSTc",
               "Agg-direct", "Agg-Star", "Agg-MST", "Agg-MSTc"]
    # calibrate eta on Individual, seed 0
    eta_run = [0.05]
    cal = {}
    for e in eta_grid:
        eta_run[0] = e
        cal[e] = one(0, ["Individual"])["Individual"]
    eta_run[0] = min(cal, key=cal.get)
    print("eta cal (Individual RMSE, seed 0):",
          {k: round(v) for k, v in cal.items()}, "-> eta", eta_run[0], flush=True)

    rows = []
    for seed in range(seeds):
        r = one(seed, methods)
        for m, v in r.items():
            rows.append(dict(seed=seed, method=m, rmse=v))
        print(f"seed {seed}: " + " ".join(f"{m}={r[m]:.0f}" for m in methods), flush=True)

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="seed", columns="method", values="rmse")
    print(f"\n=== gap-fill RMSE (consumption units, lower better), f={f}, B={B}, {seeds} seeds ===")
    print(df.groupby("method")["rmse"].agg(["mean", "std"])
          .reindex(methods).to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n=== paired t vs Individual ===")
    for m in methods:
        if m == "Individual":
            continue
        p = sstats.ttest_rel(piv["Individual"], piv[m]).pvalue
        print(f"  {m:11s} {piv[m].mean():6.1f} vs Individual {piv['Individual'].mean():.1f}"
              f"  gain {piv['Individual'].mean()-piv[m].mean():+6.1f}  p={p:.3g}")
    for a, b in [("Agg-MST", "CTL-MST"), ("Agg-MSTc", "CTL-MSTc"), ("CTL-MSTc", "CTL-MST")]:
        p = sstats.ttest_rel(piv[b], piv[a]).pvalue
        print(f"  {a} {piv[a].mean():.1f} vs {b} {piv[b].mean():.1f}"
              f"  gain {piv[b].mean()-piv[a].mean():+.1f}  p={p:.3g}")
    df.to_csv(HERE / f"weave_aggregate_root_f{f}_raw.csv", index=False)
    print(f"\nwrote weave_aggregate_root_f{f}_raw.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.3)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    run(a.f, a.B, a.seeds, (0.005, 0.02, 0.05, 0.1, 0.2))
