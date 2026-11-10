"""
Lambda sensitivity of CTL-MSTc on WEAVE-UK at the PAPER's learning rate.

The earlier lambda sweep used eta=0.01, which put its absolute RMSE level far
below the main tables and made WEAVE-UK linear appear at three different levels
across experiments. This rerun uses the published eta (Optuna, 0.0015882), the
same one behind every other linear WEAVE number, so the sweep's level coincides
with the main tables and only lambda varies.

Faithful Table 1 protocol otherwise: gradient distance, raw-feature medoid root,
SLE allocation, linear local model, T=500.

Run: python3.12 weave_lambda_papereta.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    CONFIG, make_model, eval_all, cascade_train, tree_mstc, sle_alloc, weave_tasks,
)

LAMBDAS = (0.0, 0.1, 1.0, 5.0, 10.0)
Bs = (500, 1000, 2000)
N_REPS = 5


def main():
    cfg = CONFIG["WEAVE-UK"]
    eta = cfg["eta"]                      # the paper's Optuna rate
    rows = []
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, seed = weave_tasks()
    d = Xn_tr[nodes[0]].shape[1]
    print(f"eta = {eta}  (paper's rate), T = {len(nodes)}", flush=True)
    for rep in range(N_REPS):
        for lam in LAMBDAS:
            root, dir_edges, _ = tree_mstc(nodes, D, seed, Xn_tr, yn_tr, lam)
            for B in Bs:
                torch.manual_seed(1000 * rep + int(lam * 10))
                alloc = sle_alloc(nodes, root, dir_edges, D, B)
                models = cascade_train(dir_edges, root, Xn_tr, yn_tr, alloc, eta,
                                       cfg["task"], cfg["reg"],
                                       lambda: make_model("linear", d))
                rows.append(dict(rep=rep, lam=lam, B=B,
                                 value=eval_all(models, Xn_te, y_te, "regression", stats)))
        print(f"  rep={rep} done: " + " ".join(
            f"l={r['lam']}@{r['B']}={r['value']:.0f}"
            for r in rows if r["rep"] == rep and r["B"] == 500), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "weave_lambda_papereta_raw.csv", index=False)
    piv = df.groupby(["B", "lam"])["value"].agg(["mean", "std"]).reset_index()
    piv.to_csv(HERE / "weave_lambda_papereta_summary.csv", index=False)
    print("\n=== lambda sweep at the paper's eta (mean RMSE, 5 reps) ===", flush=True)
    print(df.pivot_table(index="B", columns="lam", values="value").round(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
