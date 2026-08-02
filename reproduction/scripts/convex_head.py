"""Rendre le raffinement contractif plutôt que chercher un espace convexe.

La distance qui construit l'arbre est deja une quantite convexe, le gradient en
theta=0 du moindre carre, g = X^T y. Ce qui ne l'est pas est l'operateur de
raffinement, une descente sur un reseau. L'ecart entre la Section 3.3 et le
tableau non convexe vient donc de la, la planification et le transfert ne vivant
pas dans la meme geometrie.

On teste la correction la moins chere. La racine entraine tout le reseau, puis
chaque descendant hérite du corps de son parent, le gele, et ne raffine que la
couche de sortie. Chaque tache resout alors un moindre carre, ou une regression
logistique, convexe et demarre au point du parent, si bien que l'hypothese de
contraction est vraie par construction. C'est aussi le regime pratique du
sondage lineaire sur un corps pre-entraine.

Prediction a falsifier: sous ce regime l'ordre du tableau lineaire revient, les
constructions par arbre repassant devant Chain sur les trois collections.

    python3.12 convex_head.py --datasets Syn-10,WEAVE-UK,CIFAR --reps 10
"""

import argparse
import copy
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all,
                              train_steps, sle_alloc, uniform_alloc,
                              tree_chain, tree_mst, tree_mstc)
from mechanism_edge_vs_depth import load
from mstq import mstq_edges

Q = 0.9
METHODS = ["Chain", "CTL-MST", "CTL-MSTc", "MSTq-l0.0", "MSTq-l1.5"]


def freeze_body(model):
    """Ne laisse entrainable que la derniere couche lineaire."""
    lins = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lins[-1].parameters():
        p.requires_grad_(True)
    return model


def cascade(dir_edges, root, X, y, alloc, eta, task, l2, factory, head_only):
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    models = {root: train_steps(factory(), X[root], y[root],
                                int(max(0, alloc.get(root, 0))), eta, task, l2)}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            m = copy.deepcopy(models[u])
            if head_only:
                freeze_body(m)
            models[v] = train_steps(m, X[v], y[v],
                                    int(max(0, alloc.get(v, 0))), eta, task, l2)
            q.append(v)
    return models


def build(method, nodes, D, root, X, y, lam_mstc):
    if method == "Chain":
        return tree_chain(nodes, D, root)
    if method == "CTL-MST":
        return tree_mst(nodes, D, root)
    if method == "CTL-MSTc":
        return tree_mstc(nodes, D, root, X, y, lam_mstc)
    lam = float(method.split("-l")[1])
    e = mstq_edges(nodes, D, root, q=Q, lam=lam)
    return root, e, {v: u for u, v in e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--alloc", default="sle", choices=["sle", "uniform"])
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        cfg, eta = CONFIG[ds], MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, root, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            f = lambda: make_model("mlp", d)
            for B in Bs:
                # reference: chaque tache seule, entrainement complet
                torch.manual_seed(rep)
                b = max(1, B // len(nodes))
                indiv = {v: train_steps(f(), Xtr[v], ytr[v], b, eta,
                                        cfg["task"], cfg["reg"]) for v in nodes}
                rows.append(dict(dataset=ds, rep=rep, B=B, mode="full",
                                 method="Individual",
                                 value=eval_all(indiv, Xte, yte, cfg["task"], st)))
                for mode in ("full", "head"):
                    for m in METHODS:
                        torch.manual_seed(rep)
                        r, edges, _ = build(m, nodes, D, root, Xtr, ytr,
                                            cfg["mstc_lambda"])
                        alloc = (sle_alloc(nodes, r, edges, D, B)
                                 if a.alloc == "sle" else
                                 uniform_alloc(nodes, r, edges, B,
                                               cfg["b_proto_frac"]))
                        models = cascade(edges, r, Xtr, ytr, alloc, eta,
                                         cfg["task"], cfg["reg"], f,
                                         mode == "head")
                        rows.append(dict(dataset=ds, rep=rep, B=B, mode=mode,
                                         method=m,
                                         value=eval_all(models, Xte, yte,
                                                        cfg["task"], st)))
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "convex_head_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "convex_head_raw.csv", index=False)
    higher = {"CIFAR", "FMNIST", "CelebA"}
    from scipy import stats as st_

    for mode in ("full", "head"):
        g0 = df[df["mode"] == mode]
        print(f"\n=== corps {mode}, valeur moyenne ===", flush=True)
        cols = [c for c in (["Individual"] + METHODS)
                if c in set(g0.method)]
        print(g0.pivot_table(index=["dataset", "B"], columns="method",
                             values="value")[cols].round(4).to_string(),
              flush=True)

    print("\n=== ecart a Chain, apparie, dans chaque mode, * = mieux et p<0.05 ===",
          flush=True)
    for (ds, B, mode), g in df.groupby(["dataset", "B", "mode"]):
        base = g[g.method == "Chain"].sort_values("rep")["value"].values
        if len(base) == 0:
            continue
        out = []
        for m in METHODS[1:]:
            cur = g[g.method == m].sort_values("rep")["value"].values
            if len(cur) != len(base):
                continue
            diff = cur - base
            p = st_.ttest_rel(cur, base).pvalue
            better = (diff.mean() > 0) if ds in higher else (diff.mean() < 0)
            out.append(f"{m} {100*diff.mean()/abs(base.mean()):+.1f}%"
                       f"{'*' if p < 0.05 and better else ''}")
        print(f"{ds:9s} B={B:5d} {mode:5s}: " + " | ".join(out), flush=True)

    print("\n=== effet du gel, head moins full, meme methode ===", flush=True)
    for (ds, B), g in df[df["mode"].isin(["full", "head"])].groupby(["dataset", "B"]):
        out = []
        for m in METHODS:
            fu = g[(g.method == m) & (g["mode"] == "full")].sort_values("rep")["value"].values
            he = g[(g.method == m) & (g["mode"] == "head")].sort_values("rep")["value"].values
            if len(fu) != len(he) or len(fu) == 0:
                continue
            out.append(f"{m} {100*(he-fu).mean()/abs(fu.mean()):+.1f}%")
        print(f"{ds:9s} B={B:5d}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
