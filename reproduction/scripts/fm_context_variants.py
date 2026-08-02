"""Trois facons de cascader le contexte d'un modele de fondation tabulaire.

Suite de `fm_context_cascade.py`, dont le resultat est que l'heritage non informe
capte presque toute l'efficacite en donnees mais efface la geometrie: en tirant
uniformement dans le contexte du parent, le contexte d'un noeud profond devient
un melange aleatoire de toute la collection, et Star, RandTree et CTL-MST
deviennent indiscernables. Les deux variantes ci-dessous rendent l'arbre
determinant, chacune a sa maniere.

    uniform     contexte de taille n fixe, les n-b lignes heritees sont tirees
                uniformement dans le contexte du parent. Le controle.

    grow        contexte = contexte du parent plus b lignes propres, sans
                troncature. La taille vaut n0 + b * profondeur(v), donc le cout
                d'inference O(n^2 + n m^2) croit le long de la cascade. La
                profondeur devient un cout et non plus seulement un terme
                d'erreur, ce qui donne a l'arbre un avantage que la chaine ne
                peut pas avoir: a b=8 et T=100 une feuille de chaine porte
                n0 + 99b lignes.

    proximity   contexte de taille n fixe, mais chaque ligne porte l'etiquette
                du noeud dont elle vient. L'enfant garde ses b lignes propres
                puis les n-b lignes du contexte du parent dont le noeud source
                est le plus proche de lui au sens de la distance du papier. Les
                pas de gradient deviennent des poids 0/1 poses par la distance.
                L'information ne circule que le long des aretes.

    python3.12 fm_context_variants.py --variant grow --T 100 --b 8 --reps 3
"""

import argparse
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import weave_tasks, tree_mst, tree_random, tree_chain
from data import denormalize_y

torch.set_num_threads(max(1, torch.get_num_threads()))
METHODS = ("Star", "RandTree", "CTL-MST", "Chain")


BACKEND = "tabpfn"


def fit_predict(ctxX, ctxy, Xte, n_est):
    if BACKEND == "tabicl":
        from tabicl import TabICLRegressor as R
    else:
        from tabpfn import TabPFNRegressor as R
    r = R(device="cpu", n_estimators=n_est)
    r.fit(np.asarray(ctxX, dtype=np.float64), np.asarray(ctxy, dtype=np.float64))
    return r.predict(np.asarray(Xte, dtype=np.float64))


def rmse(pred, yte, st):
    p = denormalize_y(np.asarray(pred).reshape(-1), st)
    return float(np.sqrt(np.mean((np.asarray(yte).reshape(-1) - p) ** 2)))


def own_rows(X, y, v, k, rng):
    idx = rng.choice(len(X[v]), size=min(k, len(X[v])), replace=False)
    return X[v][idx], np.asarray(y[v]).reshape(-1)[idx], np.full(len(idx), v)


def build_contexts(variant, edges, root, X, y, b, n, D, nidx, rng, cap):
    """Contexte de chaque tache. Chaque ligne porte le noeud dont elle vient."""
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    ctx = {root: own_rows(X, y, root, n, rng)}
    q = deque([root])
    while q:
        u = q.popleft()
        pX, py, ps = ctx[u]
        for v in ch[u]:
            oX, oy, os_ = own_rows(X, y, v, b, rng)
            if variant == "grow":
                cX = np.vstack([oX, pX]) if len(pX) else oX
                cy = np.concatenate([oy, py])
                cs = np.concatenate([os_, ps])
                if cap and len(cX) > cap:            # garde-fou, jamais atteint
                    cX, cy, cs = cX[:cap], cy[:cap], cs[:cap]
            else:
                k = max(0, n - len(oX))
                if k and len(pX):
                    if variant == "proximity":
                        # les n-b lignes dont le noeud source est le plus proche
                        dv = D[nidx[v], [nidx[s] for s in ps]]
                        j = np.argsort(dv, kind="stable")[:k]
                    else:
                        j = rng.choice(len(pX), size=min(k, len(pX)),
                                       replace=False)
                    cX = np.vstack([oX, pX[j]])
                    cy = np.concatenate([oy, py[j]])
                    cs = np.concatenate([os_, ps[j]])
                else:
                    cX, cy, cs = oX, oy, os_
            ctx[v] = (cX, cy, cs)
            q.append(v)
    return ctx


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
    ap.add_argument("--variant", default="grow",
                    choices=["uniform", "grow", "proximity"])
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-est", type=int, default=1)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--backend", default="tabpfn", choices=["tabpfn", "tabicl"])
    a = ap.parse_args()
    global BACKEND
    BACKEND = a.backend
    methods = a.methods.split(",")

    allnodes, Xtr, ytr, Xte, yte, st, Dfull, _ = weave_tasks()
    gidx = {v: i for i, v in enumerate(allnodes)}

    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        nodes = sorted(rng.choice(allnodes, size=a.T, replace=False).tolist())
        sub = [gidx[v] for v in nodes]
        D = Dfull[np.ix_(sub, sub)]
        nidx = {v: i for i, v in enumerate(nodes)}
        root = nodes[int(np.argmin(D.sum(axis=1)))]

        trees = {"Star": (root, [(root, v) for v in nodes if v != root]),
                 "RandTree": tree_random(nodes, D, root,
                                         np.random.default_rng(1000 + rep))[:2],
                 "CTL-MST": tree_mst(nodes, D, root)[:2],
                 "Chain": tree_chain(nodes, D, root)[:2]}

        t0 = time.perf_counter()
        pr = {v: fit_predict(*own_rows(Xtr, ytr, v, a.n, rng)[:2], Xte[v], a.n_est)
              for v in nodes}
        rows.append(dict(rep=rep, method="Individual-n", value=float(
            np.mean([rmse(pr[v], yte[v], st[v]) for v in nodes])),
            mean_ctx=a.n, max_ctx=a.n, depth=0, secs=time.perf_counter() - t0))
        t0 = time.perf_counter()
        pr = {v: fit_predict(*own_rows(Xtr, ytr, v, a.b, rng)[:2], Xte[v], a.n_est)
              for v in nodes}
        rows.append(dict(rep=rep, method="Individual-b", value=float(
            np.mean([rmse(pr[v], yte[v], st[v]) for v in nodes])),
            mean_ctx=a.b, max_ctx=a.b, depth=0, secs=time.perf_counter() - t0))

        for m in methods:
            r_, e_ = trees[m]
            ctx = build_contexts(a.variant, e_, r_, Xtr, ytr, a.b, a.n, D, nidx,
                                 np.random.default_rng(rep), a.cap)
            sizes = [len(ctx[v][0]) for v in nodes]
            t0 = time.perf_counter()
            pr = {v: fit_predict(ctx[v][0], ctx[v][1], Xte[v], a.n_est)
                  for v in nodes}
            secs = time.perf_counter() - t0
            rows.append(dict(rep=rep, method=m, value=float(
                np.mean([rmse(pr[v], yte[v], st[v]) for v in nodes])),
                mean_ctx=float(np.mean(sizes)), max_ctx=int(np.max(sizes)),
                depth=depth_of(e_, r_), secs=secs))
            print(f"  rep={rep} {m:9s} ctx moy {np.mean(sizes):6.0f} max "
                  f"{np.max(sizes):5d} prof {depth_of(e_, r_):4d} "
                  f"{secs:6.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(
            HERE / f"fm_variant_{a.variant}_{a.backend}_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"fm_variant_{a.variant}_{a.backend}_raw.csv", index=False)
    print(f"\n=== variante {a.variant}, T={a.T}, n={a.n}, b={a.b} ===", flush=True)
    print(df.groupby("method")[["value", "mean_ctx", "max_ctx", "depth", "secs"]]
          .mean().round(1).to_string(), flush=True)

    from scipy import stats as S
    ref = df[df.method == "Individual-n"].sort_values("rep")["value"].values
    print(f"\n=== ecart au plafond Individual-n ({ref.mean():.0f}) ===", flush=True)
    for m in ["Individual-b"] + methods:
        cur = df[df.method == m].sort_values("rep")["value"].values
        p = S.ttest_rel(cur, ref).pvalue if len(cur) > 1 else float("nan")
        print(f"  {m:13s} {cur.mean():7.0f} {100*(cur-ref).mean()/ref.mean():+6.1f}%"
              f" (p={p:.3f})", flush=True)


if __name__ == "__main__":
    main()
