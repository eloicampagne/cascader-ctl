"""Lambda sensitivity of CTL-MSTc on the two image collections.

The WEAVE-UK sweep shows a plateau on one dataset. The objection concerns
variation across benchmarks, so the plateau has to be exhibited on the two
datasets whose learning rates differ most, Fashion-MNIST at 0.296 and CIFAR-10
at 0.067.

Faithful Table 1 protocol otherwise: gradient distance, medoid root, SLE
allocation, logistic head on PCA features, T=200.

    python3.12 image_lambda.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, make_model, eval_all, cascade_train,
                              tree_mstc, sle_alloc, image_tasks)
from graph import compute_gradient_distance, select_seed_node

LAMBDAS = (0.0, 0.1, 0.75, 5.0, 10.0)
Bs = (500, 2000)
N_REPS = 5


def main():
    rows = []
    for ds in ("FMNIST", "CIFAR"):
        cfg = CONFIG[ds]
        eta = cfg["eta"]
        for rep in range(N_REPS):
            nodes, X_tr, y_tr, X_te, y_te = image_tasks(ds, seed=rep)
            D = compute_gradient_distance(nodes, X_tr, y_tr)
            root0 = select_seed_node(X_tr, y_tr)
            d = X_tr[nodes[0]].shape[1]
            for lam in LAMBDAS:
                root, dir_edges, _ = tree_mstc(nodes, D, root0, X_tr, y_tr, lam)
                for B in Bs:
                    torch.manual_seed(1000 * rep + int(lam * 100))
                    alloc = sle_alloc(nodes, root, dir_edges, D, B)
                    models = cascade_train(dir_edges, root, X_tr, y_tr, alloc,
                                           eta, cfg["task"], cfg["reg"],
                                           lambda: make_model("linear", d))
                    rows.append(dict(dataset=ds, rep=rep, lam=lam, B=B,
                                     value=eval_all(models, X_te, y_te,
                                                    "classification")))
            print(f"{ds} rep={rep}: " + " ".join(
                f"l={r['lam']}@500={r['value']:.3f}"
                for r in rows if r["dataset"] == ds and r["rep"] == rep
                and r["B"] == 500), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "image_lambda_raw.csv", index=False)
    piv = df.groupby(["dataset", "B", "lam"])["value"].agg(["mean", "std"]).reset_index()
    piv.to_csv(HERE / "image_lambda_summary.csv", index=False)
    print("\n=== accuracy, mean over 5 reps ===", flush=True)
    print(df.pivot_table(index=["dataset", "B"], columns="lam",
                         values="value").round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
