"""CTL-MSTq contre Chain, CTL-MST et CTL-MSTc sur les collections réelles.

La construction dérivée du surrogate remplace les deux hyperparamètres libres de
CTL-MSTc par un seul scalaire déterminé par le problème, q = rho^{B/T}. Comme
rho n'est pas observable pour un modèle non convexe, q est balayé, ce qui donne
aussi la courbe d'interpolation entre Prim, atteint en q=0, et une construction
équilibrante quand q tend vers un.

    python3.12 mstq_eval.py --datasets Syn-10,WEAVE-UK,CIFAR --reps 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all,
                              cascade_train, train_steps, sle_alloc,
                              uniform_alloc, tree_chain, tree_mst, tree_mstc)
from graph import compute_gradient_distance, select_seed_node, root_tree
from mechanism_edge_vs_depth import load
from mstq import mstq_edges

QS = (0.0, 0.5, 0.9, 0.99)


def methods_for(qs):
    return ["Individual", "Chain", "CTL-MST", "CTL-MSTc"] + [f"MSTq-{q}" for q in qs]


def build(method, nodes, D, root, X, y, lam):
    if method == "Chain":
        return tree_chain(nodes, D, root)
    if method == "CTL-MST":
        return tree_mst(nodes, D, root)
    if method == "CTL-MSTc":
        return tree_mstc(nodes, D, root, X, y, lam)
    q = float(method.split("-")[1])
    edges = mstq_edges(nodes, D, root, q=q)
    return root, edges, {v: u for u, v in edges}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--alloc", default="both", choices=["sle", "uniform", "both"])
    a = ap.parse_args()
    allocs = ("sle", "uniform") if a.alloc == "both" else (a.alloc,)
    methods = methods_for(QS)

    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        eta = MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, root, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for B in Bs:
                for alloc_kind in allocs:
                    for m in methods:
                        torch.manual_seed(rep)
                        f = lambda: make_model("mlp", d)
                        if m == "Individual":
                            b = max(1, B // len(nodes))
                            models = {v: train_steps(f(), Xtr[v], ytr[v], b, eta,
                                                     cfg["task"], cfg["reg"])
                                      for v in nodes}
                        else:
                            r, edges, _ = build(m, nodes, D, root, Xtr, ytr,
                                                cfg["mstc_lambda"])
                            alloc = (sle_alloc(nodes, r, edges, D, B)
                                     if alloc_kind == "sle" else
                                     uniform_alloc(nodes, r, edges, B,
                                                   cfg["b_proto_frac"]))
                            models = cascade_train(edges, r, Xtr, ytr, alloc,
                                                   eta, cfg["task"], cfg["reg"], f)
                        rows.append(dict(
                            dataset=ds, rep=rep, B=B, alloc=alloc_kind, method=m,
                            value=eval_all(models, Xte, yte, cfg["task"], st)))
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "mstq_eval_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "mstq_eval_raw.csv", index=False)
    higher = {"CIFAR", "FMNIST", "CelebA"}
    from scipy import stats as st_
    for alloc_kind in allocs:
        g0 = df[df.alloc == alloc_kind]
        print(f"\n=== allocation {alloc_kind}, valeur moyenne ===", flush=True)
        print(g0.pivot_table(index=["dataset", "B"], columns="method",
                             values="value")[methods].round(4).to_string(),
              flush=True)
        print(f"\n=== {alloc_kind}, écart à Chain, apparié ===", flush=True)
        for (ds, B), g in g0.groupby(["dataset", "B"]):
            base = g[g.method == "Chain"].sort_values("rep")["value"].values
            out = []
            for m in methods[2:]:
                cur = g[g.method == m].sort_values("rep")["value"].values
                if len(cur) != len(base):
                    continue
                diff = cur - base
                p = st_.ttest_rel(cur, base).pvalue
                better = (diff.mean() > 0) if ds in higher else (diff.mean() < 0)
                out.append(f"{m} {diff.mean():+.4f}{'*' if p < 0.05 and better else ''}")
            print(f"{ds:9s} B={B:5d}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
