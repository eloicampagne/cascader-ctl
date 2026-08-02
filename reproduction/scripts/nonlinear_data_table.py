"""
Full Table-1-style comparison on data that provably requires non-linearity.

The linearity diagnostic shows WEAVE-UK and Syn-10 carry no non-linear signal: a
converged MLP loses to the exact linear optimum on every task. So the MLP results
on those datasets test CTL under a non-convex OPTIMIZATION landscape, not under a
non-linear TARGET. This script supplies the missing case, running the same
faithful protocol on clustered two-layer-ReLU teacher tasks, where the converged
MLP beats the linear optimum by 86% and the linear model is structurally unable
to fit the target.

Tasks: T=200, clustered ReLU teachers (K=5 clusters, weights perturbed within
cluster), so the task geometry is the distance between teacher weights.
Methods: Individual, Star, RandTree, Chain, CTL-MST, CTL-MSTc, plus a linear
Individual reference showing the linear model's ceiling on this data.
Budgets 500 and 2000, 10 repetitions, gradient distance, SLE allocation.

Run: python3.12 nonlinear_data_table.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    make_model, eval_all, run_method, compute_gradient_distance, select_seed_node,
)

T = 200
D_IN = 5
HIDDEN_TEACHER = 8
K_CLUSTERS = 5
N_TR, N_TE = 64, 128
Bs = (500, 2000)
N_REPS = 10
ETA_MLP = 0.05
METHODS = ["Individual", "Star", "RandTree", "Chain", "CTL-MST", "CTL-MSTc"]

CFG = dict(task="regression", reg=0.0, b_proto_frac=0.10, mstc_lambda=0.6)


def relu_teacher_tasks(seed, T=T, d=D_IN, hidden=HIDDEN_TEACHER, K=K_CLUSTERS,
                       n_tr=N_TR, n_te=N_TE, noise=0.1, spread=0.3):
    """y = W2 relu(W1 x), teacher weights clustered so tasks form K groups."""
    rng = np.random.default_rng(seed)
    centers = [(rng.normal(size=(hidden, d)), rng.normal(size=hidden)) for _ in range(K)]
    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    for v in range(T):
        W1c, W2c = centers[v % K]
        W1 = W1c + spread * rng.normal(size=W1c.shape)
        W2 = W2c + spread * rng.normal(size=W2c.shape)
        f = lambda Z: np.maximum(Z @ W1.T, 0) @ W2
        Xa = rng.normal(size=(n_tr, d)).astype(np.float32)
        Xb = rng.normal(size=(n_te, d)).astype(np.float32)
        X_tr[v] = Xa
        y_tr[v] = (f(Xa) + noise * rng.normal(size=n_tr)).astype(np.float32)
        X_te[v] = Xb
        y_te[v] = f(Xb).astype(np.float32)
    return list(range(T)), X_tr, y_tr, X_te, y_te


def main():
    rows = []
    for rep in range(N_REPS):
        nodes, X_tr, y_tr, X_te, y_te = relu_teacher_tasks(rep)
        d = X_tr[nodes[0]].shape[1]
        D = compute_gradient_distance(nodes, X_tr, y_tr)
        root = select_seed_node(X_tr, y_tr)
        rng = np.random.default_rng(80_000 + rep)
        for B in Bs:
            # linear reference: what the best linear local model can do under budget
            torch.manual_seed(rep)
            lin = run_method("Individual", nodes, X_tr, y_tr, X_tr, y_tr, D, root,
                             B, 0.05, CFG, lambda: make_model("linear", d), rng)
            rows.append(dict(rep=rep, B=B, method="Individual-Linear",
                             value=eval_all(lin, X_te, y_te, "regression")))
            for m in METHODS:
                torch.manual_seed(rep)
                models = run_method(m, nodes, X_tr, y_tr, X_tr, y_tr, D, root,
                                    B, ETA_MLP, CFG, lambda: make_model("mlp", d), rng)
                rows.append(dict(rep=rep, B=B, method=m,
                                 value=eval_all(models, X_te, y_te, "regression")))
        done = [r for r in rows if r["rep"] == rep]
        print(f"rep={rep}: " + " ".join(f"{r['method']}@{r['B']}={r['value']:.2f}"
                                        for r in done), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "nonlinear_data_table_raw.csv", index=False)
    summ = df.groupby(["B", "method"])["value"].agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(HERE / "nonlinear_data_table_summary.csv", index=False)
    order = ["Individual-Linear", "Individual", "Star", "RandTree", "Chain", "CTL-MST", "CTL-MSTc"]
    print("\n=== ReLU-teacher tasks (data provably non-linear), mean RMSE ===", flush=True)
    print(summ.pivot_table(index="B", columns="method", values="mean")[order].round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
