"""CTL quand les taches arrivent une a une, sans rien connaitre de la suite.

En deploiement, les taches peuvent arriver en continu, de sorte que
ni eta ni lambda ne peuvent etre choisis hors ligne. L'objection porte au-dela des
hyperparametres, puisque l'arbre lui-meme suppose la collection connue: la matrice
de distances est O(T^2), le MST se construit globalement et SLE repartit un budget
total.

La variante en ligne supprime les trois. A l'arrivee de v, on l'attache a la tache
deja entrainee la plus proche, on copie ses parametres et on raffine b = B/T pas.
Aucun lambda, aucune allocation, aucune connaissance des taches futures, et la
matrice de distances n'est jamais formee, seule la colonne de v contre les taches
deja vues etant calculee, soit O(T^2/2) evaluations mais aucune anticipation.

Ce n'est pas le MST. Prim choisit a la fois quelle tache attacher et a qui, alors
qu'ici l'ordre est impose et seul le parent se choisit. L'ecart entre les deux
mesure ce que coute l'ignorance du futur.

Six bras au meme budget total, par connaissance decroissante du futur:

    Offline-MSTc  arbre global et allocation SLE, tout est connu d'avance
    Dynamic       aucun arbre prealable, la construction avance avec la cascade,
                  mais l'ordre des taches reste choisi, donc la collection est connue
    Online-dyn    ordre impose, parent = tache deja entrainee dont le modele donne
                  le plus petit ||grad_v(theta_u)|| / ||grad_v(theta_0)||
    Online-NN     ordre impose, parent = tache deja entrainee la plus proche au proxy
    Online-prev   parent = tache precedente dans l'ordre d'arrivee, distance ignoree
    Individual    depart a froid, b pas par tache

Online-dyn est la version en ligne de l'arbre dynamique: le score se lit au point
reel du parent candidat, aucun proxy de distance n'est forme, et rien du futur
n'est utilise.

    python3.12 online_arrival.py --datasets Syn-10,WEAVE-UK,CIFAR --reps 5
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all, train_steps,
                              cascade_train, sle_alloc, tree_mstc)
from mechanism_edge_vs_depth import load
from dynamic_ntk import grad_norm, dynamic_cascade


def online_run(order, D, nidx, X, y, b, eta, task, reg, factory, rule, cold=None):
    """Cascade en ligne.

    rule = 'prev' parent = tache precedente
           'nn'   parent = plus proche au proxy de distance
           'dyn'  parent = celui dont le modele minimise le gradient normalise
    """
    models, edges = {}, []
    seen = []
    for v in order:
        if not seen:
            m = train_steps(factory(), X[v], y[v], b, eta, task, reg)
        else:
            if rule == "prev":
                u = seen[-1]
            elif rule == "dyn":
                sc = [grad_norm(models[w], X[v], y[v], task) / cold[v] for w in seen]
                u = seen[int(np.argmin(sc))]
            else:
                col = D[nidx[v], [nidx[w] for w in seen]]
                u = seen[int(np.argmin(col))]
            edges.append((u, v))
            m = train_steps(copy.deepcopy(models[u]), X[v], y[v], b, eta, task, reg)
        models[v] = m
        seen.append(v)
    return models, edges


def mean_edge(edges, D, nidx):
    if not edges:
        return float("nan")
    return float(np.mean([D[nidx[u], nidx[v]] for u, v in edges]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--etas", default="",
                    help="grille; vide = pas publie de la collection")
    ap.add_argument("--tune-reps", type=int, default=3)
    ap.add_argument("--methods", default="prev,dyn")
    a = ap.parse_args()

    grid = [float(x) for x in a.etas.split(",")] if a.etas else None
    rows = []
    for ds in a.datasets.split(","):
        cfg, eta0 = CONFIG[ds], MLP_ETA[ds]
        etas = grid or [eta0]
        # graines de reglage decalees, puis graines rapportees
        for split, reps in (("tune", range(500, 500 + a.tune_reps)),
                            ("eval", range(a.reps))):
         for eta in etas:
          for rep in reps:
                nodes, Xtr, ytr, Xte, yte, st, D, root, Bs = load(ds, rep)
                d = Xtr[nodes[0]].shape[1]
                nidx = {v: i for i, v in enumerate(nodes)}
                D = np.asarray(D, dtype=float)
                rng = np.random.default_rng(rep)
                order = list(rng.permutation(nodes))
                for B in Bs:
                    b = max(1, B // len(nodes))
                    f = lambda: make_model("mlp", d)

                    torch.manual_seed(rep)
                    indiv = {v: train_steps(f(), Xtr[v], ytr[v], b, eta,
                                            cfg["task"], cfg["reg"]) for v in nodes}
                    rows.append(dict(dataset=ds, rep=rep, B=B, eta=eta, split=split, method="Individual",
                                     value=eval_all(indiv, Xte, yte, cfg["task"], st),
                                     mean_edge=float("nan")))

                    torch.manual_seed(rep)
                    cold = {v: grad_norm(f(), Xtr[v], ytr[v], cfg["task"]) + 1e-12
                            for v in nodes}

                    sel = {"prev": "Online-prev", "nn": "Online-NN",
                           "dyn": "Online-dyn"}
                    for rule in a.methods.split(","):
                        name = sel[rule]
                        torch.manual_seed(rep)
                        models, edges = online_run(order, D, nidx, Xtr, ytr, b, eta,
                                                   cfg["task"], cfg["reg"], f, rule, cold)
                        rows.append(dict(dataset=ds, rep=rep, B=B, eta=eta, split=split, method=name,
                                         value=eval_all(models, Xte, yte, cfg["task"], st),
                                         mean_edge=mean_edge(edges, D, nidx)))

                    torch.manual_seed(rep)
                    r, edges, _ = tree_mstc(nodes, D, root, Xtr, ytr, cfg["mstc_lambda"])
                    alloc = sle_alloc(nodes, r, edges, D, B)
                    models = cascade_train(edges, r, Xtr, ytr, alloc, eta,
                                           cfg["task"], cfg["reg"], f)
                    rows.append(dict(dataset=ds, rep=rep, B=B, eta=eta, split=split, method="Offline-MSTc",
                                     value=eval_all(models, Xte, yte, cfg["task"], st),
                                     mean_edge=mean_edge(edges, D, nidx)))
                print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "online_arrival_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "online_arrival_raw.csv", index=False)

    higher = {"CIFAR", "FMNIST", "CelebA"}
    from scipy import stats as S
    tune, ev = df[df.split == "tune"], df[df.split == "eval"]
    order_m = ["Individual", "Online-prev", "Online-NN", "Online-dyn",
               "Dynamic", "Offline-MSTc"]
    print("\n=== pas retenu par bras, sur graines disjointes, "
          "et resultat sur les graines rapportees ===", flush=True)
    for (ds, B), g in ev.groupby(["dataset", "B"]):
        hi = ds in higher
        vals, picked = {}, {}
        for m in order_m:
            t = tune[(tune.dataset == ds) & (tune.B == B) & (tune.method == m)]
            mean = t.groupby("eta")["value"].mean().dropna()
            if mean.empty:
                continue
            e = mean.idxmax() if hi else mean.idxmin()
            picked[m] = e
            vals[m] = g[(g.method == m) & (g.eta == e)].sort_values("rep")["value"].values
        if "Individual" not in vals:
            continue
        base = vals["Individual"]
        out = []
        for m in order_m[1:]:
            if m not in vals:
                continue
            cur = vals[m]
            pv = S.ttest_rel(cur, base).pvalue if len(cur) > 1 else float("nan")
            out.append(f"{m} {np.mean(cur):.4f} (eta {picked[m]:g}) "
                       f"{100*(np.mean(cur)-np.mean(base))/abs(np.mean(base)):+.1f}% "
                       f"(p={pv:.3f})")
        print(f"{ds:9s} B={B:5d}: Individual {np.mean(base):.4f} "
              f"(eta {picked['Individual']:g}) | " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
