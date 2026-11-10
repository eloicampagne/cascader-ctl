"""
Single consolidated WEAVE-UK run so every WEAVE number traces to
ONE execution with ONE seeding. Computes, per (regime, B, rep):

  Individual, Star, RIT (reweighted independent training, SLE budget but no
  cascade), Geo-MST (geography tree), CTL-MST, CTL-MSTc.

Faithful Table 1 protocol of table1_nonconvex.py: gradient distance, raw-feature
medoid root for the gradient methods, SLE allocation on the tree methods, uniform
b_seed for Star, B//T for Individual. Geo-MST uses the substation coordinates and
the geographic medoid. Linear and MLP, budgets 500/1000/2000, 20 repetitions.
Raw per-cell values are written so effect sizes and capped p-values can be
computed downstream.

Run: python3.12 weave_all.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    CONFIG, MLP_ETA, make_model, eval_all, run_method, cascade_train,
    tree_mst, sle_alloc, train_steps, weave_tasks, N_REPS_WEAVE,
)
from weave_geography import geo_distance_matrix, run_geo_mst

Bs = (500, 1000, 2000)
N_REPS = N_REPS_WEAVE


def run_rit(nodes, X_tr, y_tr, D_grad, root, B, eta, cfg, factory):
    """Reweighted Independent Training: the SLE budget of the gradient MST, but
    each task trained independently from scratch with no parent warm-start."""
    _, dir_edges, _ = tree_mst(nodes, D_grad, root)
    alloc = sle_alloc(nodes, root, dir_edges, D_grad, B)
    task, l2 = cfg["task"], cfg["reg"]
    return {v: train_steps(factory(), X_tr[v], y_tr[v],
                           int(max(0, alloc.get(v, 0))), eta, task, l2)
            for v in nodes}


def main():
    cfg = CONFIG["WEAVE-UK"]
    rows = []
    for rep in range(N_REPS):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, seed = weave_tasks()
        d = Xn_tr[nodes[0]].shape[1]
        D_geo, geo_root, kept = geo_distance_matrix(nodes)
        assert len(kept) == len(nodes), "some WEAVE nodes lack coordinates"
        for mc in ("linear", "mlp"):
            eta = cfg["eta"] if mc == "linear" else MLP_ETA["WEAVE-UK"]
            factory = lambda: make_model(mc, d)
            rng = np.random.default_rng(70_000 + rep)
            for B in Bs:
                def rec(method, models):
                    rows.append(dict(model=mc, B=B, seed=rep, method=method,
                                     value=eval_all(models, Xn_te, y_te, "regression", stats)))
                torch.manual_seed(rep)
                for m in ("Individual", "Star", "CTL-MST", "CTL-MSTc"):
                    rec(m, run_method(m, nodes, Xn_tr, yn_tr, Xn_tr, yn_tr,
                                      D_grad, seed, B, eta, cfg, factory, rng))
                rec("RIT", run_rit(nodes, Xn_tr, yn_tr, D_grad, seed, B, eta, cfg, factory))
                rec("Geo-MST", run_geo_mst(nodes, Xn_tr, yn_tr, D_geo, geo_root, B, eta, cfg, factory))
            done = [r for r in rows if r["seed"] == rep and r["model"] == mc]
            print(f"  rep={rep} {mc}: " + " ".join(
                f"{r['method']}@{r['B']}={r['value']:.0f}" for r in done), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "weave_all_raw.csv", index=False)
    summ = (df.groupby(["model", "B", "method"])["value"]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(HERE / "weave_all_summary.csv", index=False)
    order = ["Individual", "Star", "RIT", "Geo-MST", "CTL-MST", "CTL-MSTc"]
    print("\n=== consolidated WEAVE-UK (mean RMSE, 20 reps) ===", flush=True)
    print(summ.pivot_table(index=["model", "B"], columns="method", values="mean")[order].round(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
