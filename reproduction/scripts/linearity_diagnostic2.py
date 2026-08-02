"""
Non-linearity headroom, part 2: London daily and the two image collections.

Same protocol as linearity_diagnostic.py. Per task, fit the exact linear optimum
and an MLP trained to convergence, then compare held-out performance. Regression
uses closed-form ridge and RMSE, classification uses a converged logistic
regression and accuracy.

Run: python3.12 linearity_diagnostic2.py
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import MLPHead, image_tasks, denormalize_y
from london_data import load_london

MLP_EPOCHS = 3000
HIDDEN = 32
N_SHOW = 100


def fit_mlp(X, y, epochs=MLP_EPOCHS, lr=1e-2, classification=False):
    torch.manual_seed(0)
    m = MLPHead(X.shape[1], HIDDEN)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(np.asarray(y)).float().reshape(-1)
    crit = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        crit(m(Xt), yt).backward()
        opt.step()
    m.eval()
    return m


def run_regression(name, nodes, X_tr, y_tr, X_te, y_te, stats):
    rows = []
    for i, v in enumerate(list(nodes)[:N_SHOW]):
        Xtr, ytr = X_tr[v], np.asarray(y_tr[v]).reshape(-1)
        Xte, yte = X_te[v], np.asarray(y_te[v]).reshape(-1)
        w = np.linalg.solve(Xtr.T @ Xtr + 1e-6 * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        pl = Xte @ w
        m = fit_mlp(Xtr, ytr)
        with torch.no_grad():
            pm = m(torch.from_numpy(Xte).float()).numpy().reshape(-1)
        if stats is not None:
            pl, pm = denormalize_y(pl, stats[v]), denormalize_y(pm, stats[v])
        rows.append(dict(dataset=name,
                         lin=float(np.sqrt(np.mean((yte - pl) ** 2))),
                         mlp=float(np.sqrt(np.mean((yte - pm) ** 2)))))
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i+1} tasks", flush=True)
    return pd.DataFrame(rows), True


def run_classification(name, nodes, X_tr, y_tr, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    rows = []
    for i, v in enumerate(list(nodes)[:N_SHOW]):
        Xtr, ytr = X_tr[v], np.asarray(y_tr[v]).reshape(-1)
        Xte, yte = X_te[v], np.asarray(y_te[v]).reshape(-1)
        lr = LogisticRegression(max_iter=5000, C=1e3).fit(Xtr, ytr)
        acc_lin = float((lr.predict(Xte) == yte).mean())
        m = fit_mlp(Xtr, ytr, classification=True)
        with torch.no_grad():
            out = m(torch.from_numpy(Xte).float()).numpy().reshape(-1)
        acc_mlp = float(((out > 0) == (yte > 0.5)).mean())
        rows.append(dict(dataset=name, lin=acc_lin, mlp=acc_mlp))
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i+1} tasks", flush=True)
    return pd.DataFrame(rows), False


if __name__ == "__main__":
    out, raws = [], []

    print("=== London daily (T=500 households) ===", flush=True)
    nodes, Xtr, ytr, Xte, yte, stats = load_london()
    df, lower_better = run_regression("London-daily", nodes, Xtr, ytr, Xte, yte, stats)
    raws.append(df)
    gain = 100 * (df.lin.mean() - df.mlp.mean()) / df.lin.mean()
    out.append(dict(dataset="London-daily", metric="RMSE", linear=df.lin.mean(),
                    mlp=df.mlp.mean(), mlp_gain_pct=gain,
                    mlp_win_frac=float((df.mlp < df.lin).mean())))
    print(f"London-daily: linear {df.lin.mean():.4g} | MLP {df.mlp.mean():.4g} "
          f"| gain {gain:+.1f}% | wins {(df.mlp < df.lin).mean():.0%}", flush=True)

    for ds in ("FMNIST", "CIFAR"):
        print(f"\n=== {ds} (clustered binary, PCA-100) ===", flush=True)
        nodes, Xa, ya, Xb, yb = image_tasks(ds, 0)
        df, _ = run_classification(ds, nodes, Xa, ya, Xb, yb)
        raws.append(df)
        gain = 100 * (df.mlp.mean() - df.lin.mean()) / df.lin.mean()
        out.append(dict(dataset=ds, metric="accuracy", linear=df.lin.mean(),
                        mlp=df.mlp.mean(), mlp_gain_pct=gain,
                        mlp_win_frac=float((df.mlp > df.lin).mean())))
        print(f"{ds}: linear {df.lin.mean():.4f} | MLP {df.mlp.mean():.4f} "
              f"| gain {gain:+.1f}% | wins {(df.mlp > df.lin).mean():.0%}", flush=True)

    pd.concat(raws).to_csv(HERE / "linearity_diagnostic2_raw.csv", index=False)
    res = pd.DataFrame(out)
    res.to_csv(HERE / "linearity_diagnostic2_summary.csv", index=False)
    print("\n=== summary ===", flush=True)
    print(res.to_string(index=False), flush=True)
