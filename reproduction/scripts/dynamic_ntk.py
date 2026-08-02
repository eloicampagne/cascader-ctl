"""Distances NTK et arbre dynamique, à budget strictement égal.

La distance du papier est ||X^T y_u - X^T y_v|| sur des vecteurs normalises, soit
le gradient de la perte en theta=0 pour un modele lineaire. Hors du lineaire elle
n'est plus le bon proxy de ||theta*_u - theta*_v||. Le remplacant naturel est le
gradient de la perte du reseau en un point partage theta_0, qui redonne la
formule du papier quand le modele est lineaire et qui est la quantite correcte
dans le regime linearise, ou theta* - theta_0 est l'image du gradient par
l'inverse du noyau tangent.

Quatre regimes, tous au meme budget total B.

    static     la distance du papier, arbre fige avant tout entrainement
    ntk-init   gradients du reseau en une initialisation partagee
    ntk-root   gradients du reseau au point de la racine entrainee, gratuit
               puisque la cascade entraine la racine de toute facon
    dynamic    l'arbre croit pendant la cascade, chaque parent fraichement
               entraine scorant les taches restantes par
               ||grad_v(theta_u)|| / ||grad_v(theta_0)||, une difficulte de
               transfert dirigee et sans echelle

Le budget de la racine vaut b_proto_frac * B dans tous les regimes et le reste
est partage uniformement, faute de quoi ntk-root et dynamic auraient besoin de
connaitre l'arbre avant de le construire. Toutes les methodes subissent la meme
allocation, donc la comparaison reste appariee.

    python3.12 dynamic_ntk.py --datasets Syn-10,WEAVE-UK,CIFAR --reps 5
"""

import argparse
import copy
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all,
                              train_steps, tree_chain, tree_mst, tree_mstc)
from mechanism_edge_vs_depth import load
from mstq import mstq_edges

Q = 0.9
TREES = ["Chain", "CTL-MST", "CTL-MSTc", "MSTq"]


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def _crit(task):
    return nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()


def grad_vec(model, X, y, task):
    """Gradient de la perte de la tache au point courant du modele, aplati."""
    crit = _crit(task)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().reshape(-1)
    model.zero_grad(set_to_none=True)
    crit(model(Xt), yt).backward()
    g = torch.cat([(p.grad if p.grad is not None
                    else torch.zeros_like(p)).reshape(-1)
                   for p in model.parameters()])
    return g.detach().numpy()


def grad_norm(model, X, y, task):
    crit = _crit(task)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().reshape(-1)
    model.zero_grad(set_to_none=True)
    crit(model(Xt), yt).backward()
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.pow(2).sum())
    return float(np.sqrt(s))


def ntk_distance(nodes, X, y, model, task):
    G = np.stack([grad_vec(model, X[v], y[v], task) for v in nodes])
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return np.linalg.norm(G[:, None, :] - G[None, :, :], axis=-1)


# ---------------------------------------------------------------------------
# Allocation commune, racine fixee puis partage plat
# ---------------------------------------------------------------------------

def flat_alloc(nodes, root, B, frac):
    seed = max(1, int(frac * B))
    rest = max(0, B - seed)
    per = max(1, rest // max(1, len(nodes) - 1)) if rest else 0
    a = {v: per for v in nodes}
    a[root] = seed
    return a


def build(name, nodes, D, root, X, y, lam_mstc):
    if name == "Chain":
        return tree_chain(nodes, D, root)[1]
    if name == "CTL-MST":
        return tree_mst(nodes, D, root)[1]
    if name == "CTL-MSTc":
        return tree_mstc(nodes, D, root, X, y, lam_mstc)[1]
    return mstq_edges(nodes, D, root, q=Q, lam=0.0)


def cascade_from(edges, root, root_model, X, y, alloc, eta, task, l2):
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
    models = {root: root_model}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            models[v] = train_steps(copy.deepcopy(models[u]), X[v], y[v],
                                    int(max(0, alloc.get(v, 0))), eta, task, l2)
            q.append(v)
    return models


# ---------------------------------------------------------------------------
# Arbre dynamique: la construction et l'entrainement avancent ensemble
# ---------------------------------------------------------------------------

def dynamic_cascade(nodes, X, y, root, root_model, alloc, eta, task, l2, cold):
    """Prim glouton sur ||grad_v(theta_u)|| / ||grad_v(theta_0)||.

    A chaque etape, le noeud tout juste entraine score les taches encore
    detachees, la meilleure est attachee et entrainee, et ainsi de suite. Aucun
    pas de gradient n'est depense en dehors de la cascade elle-meme, donc le
    budget est identique aux regimes statiques.
    """
    models = {root: root_model}
    edges = []
    rem = set(nodes) - {root}
    best = {v: (np.inf, root) for v in rem}
    fresh = [root]
    while rem:
        for u in fresh:
            mu = models[u]
            for v in rem:
                c = grad_norm(mu, X[v], y[v], task) / cold[v]
                if c < best[v][0]:
                    best[v] = (c, u)
        v = min(rem, key=lambda w: best[w][0])
        u = best[v][1]
        edges.append((u, v))
        models[v] = train_steps(copy.deepcopy(models[u]), X[v], y[v],
                                int(max(0, alloc.get(v, 0))), eta, task, l2)
        rem.discard(v)
        best.pop(v, None)
        fresh = [v]
    return models, edges


def depth_of(edges, root):
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    d, q, mx = {root: 0}, deque([root]), 0
    while q:
        u = q.popleft()
        for v in ch[u]:
            d[v] = d[u] + 1
            mx = max(mx, d[v])
            q.append(v)
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--regimes", default="static,ntk-init,ntk-root,dynamic")
    a = ap.parse_args()
    regimes = a.regimes.split(",")

    rows = []
    for ds in a.datasets.split(","):
        cfg, eta = CONFIG[ds], MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D0, root, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for B in Bs:
                alloc = flat_alloc(nodes, root, B, cfg["b_proto_frac"])
                torch.manual_seed(rep)
                init = make_model("mlp", d)
                rootm = train_steps(copy.deepcopy(init), Xtr[root], ytr[root],
                                    alloc[root], eta, cfg["task"], cfg["reg"])
                mats = {}
                if "static" in regimes:
                    mats["static"] = D0
                if "ntk-init" in regimes:
                    mats["ntk-init"] = ntk_distance(nodes, Xtr, ytr,
                                                    copy.deepcopy(init), cfg["task"])
                if "ntk-root" in regimes:
                    mats["ntk-root"] = ntk_distance(nodes, Xtr, ytr,
                                                    copy.deepcopy(rootm), cfg["task"])
                for reg_name, D in mats.items():
                    for t in TREES:
                        edges = build(t, nodes, D, root, Xtr, ytr,
                                      cfg["mstc_lambda"])
                        models = cascade_from(edges, root,
                                              copy.deepcopy(rootm), Xtr, ytr,
                                              alloc, eta, cfg["task"], cfg["reg"])
                        rows.append(dict(dataset=ds, rep=rep, B=B,
                                         regime=reg_name, tree=t,
                                         depth=depth_of(edges, root),
                                         value=eval_all(models, Xte, yte,
                                                        cfg["task"], st)))
                if "dynamic" in regimes:
                    t0 = time.time()
                    cold = {v: max(1e-8, grad_norm(copy.deepcopy(init), Xtr[v],
                                                   ytr[v], cfg["task"]))
                            for v in nodes}
                    models, edges = dynamic_cascade(
                        nodes, Xtr, ytr, root, copy.deepcopy(rootm), alloc,
                        eta, cfg["task"], cfg["reg"], cold)
                    rows.append(dict(dataset=ds, rep=rep, B=B,
                                     regime="dynamic", tree="greedy",
                                     depth=depth_of(edges, root),
                                     value=eval_all(models, Xte, yte,
                                                    cfg["task"], st)))
                    print(f"    dynamic {ds} B={B} en {time.time()-t0:.0f}s",
                          flush=True)
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "dynamic_ntk_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "dynamic_ntk_raw.csv", index=False)
    print("\n=== valeur moyenne, allocation plate partout ===", flush=True)
    print(df.pivot_table(index=["dataset", "B"], columns=["regime", "tree"],
                         values="value").round(4).to_string(), flush=True)
    print("\n=== profondeur moyenne ===", flush=True)
    print(df.pivot_table(index=["dataset"], columns=["regime", "tree"],
                         values="depth").round(1).to_string(), flush=True)

    higher = {"CIFAR", "FMNIST", "CelebA"}
    from scipy import stats as st_
    print("\n=== ecart a (static, Chain), apparie, * = mieux et p<0.05 ===",
          flush=True)
    for (ds, B), g in df.groupby(["dataset", "B"]):
        base = g[(g.regime == "static") & (g.tree == "Chain")] \
            .sort_values("rep")["value"].values
        out = []
        for (reg_name, t), h in g.groupby(["regime", "tree"]):
            cur = h.sort_values("rep")["value"].values
            if len(cur) != len(base) or (reg_name, t) == ("static", "Chain"):
                continue
            diff = cur - base
            p = st_.ttest_rel(cur, base).pvalue
            better = (diff.mean() > 0) if ds in higher else (diff.mean() < 0)
            out.append(f"{reg_name}/{t} {100*diff.mean()/abs(base.mean()):+.1f}%"
                       f"{'*' if p < 0.05 and better else ''}")
        print(f"{ds:9s} B={B:5d}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
