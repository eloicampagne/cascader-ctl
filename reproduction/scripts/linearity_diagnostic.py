"""
Is the data actually non-linear?

A non-convex local MODEL does not guarantee that the
DATA is essentially linear, so the MLP has no non-linear structure to exploit and
the non-convex evidence is weaker than it looks. On WEAVE-UK in particular,
half-hourly forecasting from 48 recent lags is close to a first-order expansion,
which is linear by construction.

This measures the non-linearity headroom of each dataset directly, independently
of any budget or cascade: per task, fit the exact linear optimum (closed-form
ridge, no learning-rate confound) and an MLP trained to convergence with Adam,
then compare held-out RMSE. If the MLP cannot beat the linear optimum, the
conditional mean is linear and the dataset carries no non-linear signal.

Datasets: WEAVE-UK (half-hourly, the suspect case), the clustered two-layer-ReLU
teacher tasks (a positive control, non-linear by construction), and Syn-10
(linear generator, a negative control).

Run: python3.12 linearity_diagnostic.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import weave_tasks, syn_tasks, MLPHead, denormalize_y

N_TASKS_SHOW = 100          # subsample tasks for the diagnostic
MLP_EPOCHS = 3000
HIDDEN = 32


def ridge_closed_form(X, y, lam=1e-6):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def fit_mlp_converged(X, y, hidden=HIDDEN, epochs=MLP_EPOCHS, lr=1e-2):
    torch.manual_seed(0)
    m = MLPHead(X.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().reshape(-1)
    crit = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(m(Xt), yt)
        loss.backward()
        opt.step()
    m.eval()
    return m


def evaluate(name, nodes, X_tr, y_tr, X_te, y_te, stats=None, n_show=N_TASKS_SHOW):
    rng = np.random.default_rng(0)
    sub = list(nodes)[:n_show] if len(nodes) > n_show else list(nodes)
    rows = []
    for i, v in enumerate(sub):
        Xtr, ytr = X_tr[v], np.asarray(y_tr[v]).reshape(-1)
        Xte, yte = X_te[v], np.asarray(y_te[v]).reshape(-1)
        w = ridge_closed_form(Xtr, ytr)
        pred_lin = Xte @ w
        m = fit_mlp_converged(Xtr, ytr)
        with torch.no_grad():
            pred_mlp = m(torch.from_numpy(Xte).float()).numpy().reshape(-1)
        if stats is not None:
            st = stats[v]
            pred_lin = denormalize_y(pred_lin, st)
            pred_mlp = denormalize_y(pred_mlp, st)
        rmse_lin = float(np.sqrt(np.mean((yte - pred_lin) ** 2)))
        rmse_mlp = float(np.sqrt(np.mean((yte - pred_mlp) ** 2)))
        rows.append(dict(dataset=name, task=str(v), rmse_linear=rmse_lin, rmse_mlp=rmse_mlp))
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i+1}/{len(sub)} tasks", flush=True)
    df = pd.DataFrame(rows)
    lin, mlp = df.rmse_linear.mean(), df.rmse_mlp.mean()
    gain = 100.0 * (lin - mlp) / lin
    wins = float((df.rmse_mlp < df.rmse_linear).mean())
    print(f"\n{name}: linear-optimum RMSE {lin:.4g} | converged-MLP RMSE {mlp:.4g} "
          f"| MLP gain {gain:+.1f}% | MLP wins {wins:.0%} of tasks", flush=True)
    return df, dict(dataset=name, rmse_linear=lin, rmse_mlp=mlp, mlp_gain_pct=gain, mlp_win_frac=wins)


def nonlinear_teacher_tasks(T=100, d=5, n_tr=64, n_te=128, hidden=8, K=5, seed=0):
    """Clustered two-layer-ReLU teachers: y = W2 relu(W1 x), non-linear by construction."""
    rng = np.random.default_rng(seed)
    centers = [(rng.normal(size=(hidden, d)), rng.normal(size=hidden)) for _ in range(K)]
    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    for v in range(T):
        W1c, W2c = centers[v % K]
        W1 = W1c + 0.3 * rng.normal(size=W1c.shape)
        W2 = W2c + 0.3 * rng.normal(size=W2c.shape)
        f = lambda Z: np.maximum(Z @ W1.T, 0) @ W2
        Xa = rng.normal(size=(n_tr, d)); Xb = rng.normal(size=(n_te, d))
        X_tr[v], y_tr[v] = Xa, f(Xa) + 0.1 * rng.normal(size=n_tr)
        X_te[v], y_te[v] = Xb, f(Xb)
    return list(range(T)), X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
    summaries, raws = [], []

    print("=== WEAVE-UK (half-hourly, 48 lags) ===", flush=True)
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, seed = weave_tasks()
    df, s = evaluate("WEAVE-UK", nodes, Xn_tr, yn_tr, Xn_te, y_te, stats)
    raws.append(df); summaries.append(s)

    print("\n=== Syn-10 (linear generator, negative control) ===", flush=True)
    nodes, Xn_tr, yn_tr, Xn_te, y_te2, stats2, D, seed = syn_tasks(0)
    df, s = evaluate("Syn-10", nodes, Xn_tr, yn_tr, Xn_te, y_te2, stats2)
    raws.append(df); summaries.append(s)

    print("\n=== ReLU-teacher (non-linear, positive control) ===", flush=True)
    nodes, Xa, ya, Xb, yb = nonlinear_teacher_tasks()
    df, s = evaluate("ReLU-teacher", nodes, Xa, ya, Xb, yb, None)
    raws.append(df); summaries.append(s)

    pd.concat(raws).to_csv(HERE / "linearity_diagnostic_raw.csv", index=False)
    out = pd.DataFrame(summaries)
    out.to_csv(HERE / "linearity_diagnostic_summary.csv", index=False)
    print("\n=== summary: non-linearity headroom ===", flush=True)
    print(out.to_string(index=False), flush=True)
