"""Où l'arbre reprend l'avantage sur la chaîne, expérience contrôlée.

La géométrie des tâches est fixée, seul l'écartement entre grappes varie, et le
budget varie indépendamment. Le point de croisement entre CTL-MST et Chain, s'il
existe, doit apparaître quand les tâches deviennent trop éloignées pour qu'un
chemin les relie de proche en proche dans le budget imparti.

Trois bras séparent deux choses que la table principale confond, la non-convexité
de l'optimiseur et la non-linéarité de la cible.

    lin/lin   générateur linéaire, modèle local linéaire, le régime du papier
    lin/mlp   générateur linéaire, modèle local non convexe
    relu/mlp  enseignants ReLU, cible non linéaire et modèle non convexe

Les deux allocations sont balayées, puisque sur une chaîne l'allocation par
sous-arbre concentre le budget côté racine alors que sur un arbre équilibré elle
le répartit, ce qui confond topologie et partage du budget si on ne la fixe pas.

    python3.12 scatter_sweep.py --arms lin/lin,lin/mlp,relu/mlp --reps 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (make_model, eval_all, cascade_train, train_steps,
                              sle_alloc, uniform_alloc, tree_chain, tree_mst,
                              tree_mstc, normalize_xy)
from data import generate_clustered_tasks
from graph import compute_gradient_distance, select_seed_node

METHODS = ("Individual", "Star", "Chain", "CTL-MST", "CTL-MSTc")
SLE_METHODS = {"Chain", "CTL-MST", "CTL-MSTc"}
SCATTERS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
BUDGETS = (200, 500, 1000, 2000)
TAU_WITHIN = 1.0
MSTC_LAMBDA = 0.75


def relu_teacher_tasks(T, d, n_tr, n_te, hidden, K, tau_within, tau_between, rng):
    """Enseignants ReLU en grappes, avec écartement réglable.

    Les centres de grappes sont tirés à l'échelle `tau_between` et les tâches
    s'en écartent de `tau_within`, exactement la paramétrisation du générateur
    linéaire, de sorte que le balayage d'écartement soit le même objet dans les
    deux régimes.
    """
    centers = [(rng.normal(scale=tau_between, size=(hidden, d)),
                rng.normal(scale=tau_between, size=hidden)) for _ in range(K)]
    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    for v in range(T):
        W1c, W2c = centers[v % K]
        W1 = W1c + tau_within * rng.normal(size=W1c.shape)
        W2 = W2c + tau_within * rng.normal(size=W2c.shape)
        f = lambda Z: np.maximum(Z @ W1.T, 0) @ W2
        Xa = rng.normal(size=(n_tr, d))
        Xb = rng.normal(size=(n_te, d))
        X_tr[v], y_tr[v] = Xa, f(Xa) + 0.1 * rng.normal(size=n_tr)
        X_te[v], y_te[v] = Xb, f(Xb)
    return list(range(T)), X_tr, y_tr, X_te, y_te


def build_tasks(gen, scatter, rep, T=200, d=5, n_tr=64, n_te=128, K=5):
    rng = np.random.default_rng(1000 * rep + 7)
    if gen == "lin":
        _, _, _, (Xt, yt), (Xe, ye), _ = generate_clustered_tasks(
            T=T, d=d, n_train=n_tr, n_test=n_te, n_clusters=K,
            tau_within=TAU_WITHIN, tau_between=scatter, sigma=0.5, rng=rng)
        nodes = list(range(T))
        Xr = {v: Xt[v] for v in nodes}
        yr = {v: yt[v] for v in nodes}
        Xer = {v: Xe[v] for v in nodes}
        yer = {v: ye[v] for v in nodes}
    else:
        nodes, Xr, yr, Xer, yer = relu_teacher_tasks(
            T, d, n_tr, n_te, 8, K, TAU_WITHIN, scatter, rng)
    Xn, yn, Xn_te, stats = {}, {}, {}, {}
    for v in nodes:
        Xn[v], yn[v], st = normalize_xy(Xr[v], yr[v])
        stats[v] = st
        Xn_te[v] = (Xer[v] - st["X_mean"]) / st["X_std"]
    D = compute_gradient_distance(nodes, Xn, yn)
    root = select_seed_node(Xr, yr)
    return nodes, Xn, yn, Xn_te, yer, stats, D, root


def run_method(method, nodes, X, y, D, root, B, eta, factory, alloc_kind):
    if method == "Individual":
        b = max(1, B // len(nodes))
        return {v: train_steps(factory(), X[v], y[v], b, eta, "regression", 0.0)
                for v in nodes}
    if method == "Star":
        r, edges = root, [(root, v) for v in nodes if v != root]
    elif method == "Chain":
        r, edges, _ = tree_chain(nodes, D, root)
    elif method == "CTL-MST":
        r, edges, _ = tree_mst(nodes, D, root)
    else:
        r, edges, _ = tree_mstc(nodes, D, root, X, y, MSTC_LAMBDA)
    if method in SLE_METHODS and alloc_kind == "sle":
        alloc = sle_alloc(nodes, r, edges, D, B)
    else:
        alloc = uniform_alloc(nodes, r, edges, B, 0.10)
    return cascade_train(edges, r, X, y, alloc, eta, "regression", 0.0, factory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="lin/lin,lin/mlp,relu/mlp")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--scatters", default=None)
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--out", default="scatter_sweep")
    a = ap.parse_args()
    scatters = ([float(s) for s in a.scatters.split(",")] if a.scatters
                else list(SCATTERS))
    budgets = ([int(b) for b in a.budgets.split(",")] if a.budgets
               else list(BUDGETS))

    rows = []
    for arm in a.arms.split(","):
        gen, mc = arm.split("/")
        mc = "linear" if mc == "lin" else mc
        for s in scatters:
            for rep in range(a.reps):
                nodes, X, y, Xte, yte, stats, D, root = build_tasks(gen, s, rep)
                d = X[nodes[0]].shape[1]
                for B in budgets:
                    for alloc in ("sle", "uniform"):
                        for m in METHODS:
                            torch.manual_seed(rep)
                            models = run_method(m, nodes, X, y, D, root, B,
                                                a.eta,
                                                lambda: make_model(mc, d), alloc)
                            rows.append(dict(
                                arm=arm, scatter=s, rep=rep, B=B, alloc=alloc,
                                method=m, mean_delta=float(np.mean(
                                    D[np.triu_indices(len(nodes), 1)])),
                                value=eval_all(models, Xte, yte, "regression",
                                               stats)))
                print(f"  {arm} scatter={s} rep={rep} fait", flush=True)
            pd.DataFrame(rows).to_csv(HERE / f"{a.out}_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{a.out}_raw.csv", index=False)
    for (arm, alloc), g in df.groupby(["arm", "alloc"]):
        p = g.pivot_table(index=["scatter", "B"], columns="method",
                          values="value")
        p["MST-Chain"] = p["CTL-MST"] - p["Chain"]
        p["MSTc-Chain"] = p["CTL-MSTc"] - p["Chain"]
        print(f"\n=== {arm}, allocation {alloc}, RMSE moyenne ===", flush=True)
        print(p.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
