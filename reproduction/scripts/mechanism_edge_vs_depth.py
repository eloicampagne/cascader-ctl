"""Off convexity, does per-task error follow edge length or path depth.

Reviewer fdRt proposes that the non-convex benefit of a cascade is basin
selection from a nearby initialisation, which would make the parent-child
distance the governing quantity rather than the depth of the path. The
prediction is testable at the level of individual tasks, since within one tree
each task carries its own parent distance, its own depth and its own error.

The script records that triple for every task, then reports standardised
regression coefficients of the task error on the two candidates. It also reports
the parent-distance distributions per construction, which is where the
tree-versus-chain comparison is decided if the proposal is right.

    python3.12 mechanism_edge_vs_depth.py --datasets Syn-10,WEAVE-UK,CIFAR
"""

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_task,
                              cascade_train, sle_alloc, tree_chain, tree_mst,
                              tree_mstc, weave_tasks, image_tasks, syn_tasks)
from graph import compute_gradient_distance, select_seed_node

METHODS = ("Chain", "CTL-MST", "CTL-MSTc")
Bs_REG = (500, 2000)
Bs_CLS = (500, 2000)


def node_geometry(nodes, root, dir_edges, D):
    """Parent distance, depth and cumulative path length for every task."""
    idx = {v: i for i, v in enumerate(nodes)}
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    delta = {root: 0.0}
    depth = {root: 0}
    cum = {root: 0.0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            delta[v] = D[idx[u], idx[v]]
            depth[v] = depth[u] + 1
            cum[v] = cum[u] + delta[v]
            q.append(v)
    return delta, depth, cum


def build(method, nodes, D, seed, X, y, lam):
    if method == "Chain":
        return tree_chain(nodes, D, seed)
    if method == "CTL-MST":
        return tree_mst(nodes, D, seed)
    return tree_mstc(nodes, D, seed, X, y, lam)


def load(ds, rep):
    if ds == "WEAVE-UK":
        nodes, Xtr, ytr, Xte, yte, st, D, seed = weave_tasks()
        return nodes, Xtr, ytr, Xte, yte, st, D, seed, Bs_REG
    if ds == "Syn-10":
        nodes, Xtr, ytr, Xte, yte, st, D, seed = syn_tasks(rep)
        return nodes, Xtr, ytr, Xte, yte, st, D, seed, Bs_REG
    nodes, Xtr, ytr, Xte, yte = image_tasks(ds, seed=rep)
    D = compute_gradient_distance(nodes, Xtr, ytr)
    return nodes, Xtr, ytr, Xte, yte, None, D, select_seed_node(Xtr, ytr), Bs_CLS


def zscore(a):
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        eta = MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, seed, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for m in METHODS:
                root, dir_edges, _ = build(m, nodes, D, seed, Xtr, ytr,
                                           cfg["mstc_lambda"])
                delta, depth, cum = node_geometry(nodes, root, dir_edges, D)
                for B in Bs:
                    torch.manual_seed(rep)
                    alloc = sle_alloc(nodes, root, dir_edges, D, B)
                    models = cascade_train(dir_edges, root, Xtr, ytr, alloc, eta,
                                           cfg["task"], cfg["reg"],
                                           lambda: make_model("mlp", d))
                    for v in nodes:
                        if v == root:
                            continue
                        rows.append(dict(
                            dataset=ds, rep=rep, method=m, B=B, node=str(v),
                            err=eval_task(models[v], Xte[v], yte[v], cfg["task"],
                                          st[v] if st else None),
                            delta=delta[v], depth=depth[v], cum=cum[v],
                            budget=alloc.get(v, 0)))
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "mechanism_edge_vs_depth_raw.csv",
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "mechanism_edge_vs_depth_raw.csv", index=False)
    higher = {"CIFAR", "FMNIST", "CelebA"}

    print("\n=== distance au parent par construction ===", flush=True)
    print(df.groupby(["dataset", "method"])["delta"].agg(
        moy="mean", med="median", p90=lambda s: s.quantile(0.90),
        max="max").round(4).to_string(), flush=True)

    print("\n=== profondeur par construction ===", flush=True)
    print(df.groupby(["dataset", "method"])["depth"].agg(
        moy="mean", max="max").round(1).to_string(), flush=True)

    print("\n=== régression standardisée de l'erreur par tâche ===", flush=True)
    print("coefficients sur (delta, profondeur, budget), au sein d'un arbre",
          flush=True)
    out = []
    for (ds, m, B), g in df.groupby(["dataset", "method", "B"]):
        co = []
        for rep, gr in g.groupby("rep"):
            if len(gr) < 20:
                continue
            X = np.column_stack([zscore(gr.delta), zscore(gr.depth),
                                 zscore(np.log1p(gr.budget)), np.ones(len(gr))])
            yv = zscore(gr.err) * (-1 if ds in higher else 1)
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            co.append(beta[:3])
        if not co:
            continue
        co = np.array(co)
        mu, se = co.mean(0), co.std(0, ddof=1) / np.sqrt(len(co))
        out.append(dict(dataset=ds, method=m, B=B,
                        b_delta=mu[0], se_delta=se[0],
                        b_depth=mu[1], se_depth=se[1],
                        b_budget=mu[2], se_budget=se[2], n=len(co)))
    print(pd.DataFrame(out).round(3).to_string(index=False), flush=True)
    pd.DataFrame(out).to_csv(HERE / "mechanism_edge_vs_depth_coef.csv",
                             index=False)


if __name__ == "__main__":
    main()
