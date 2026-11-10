"""CTL sans gradient: cascader le contexte d'un modele de fondation tabulaire.

Dans les deploiements a grande echelle, le budget d'entrainement n'est pas
toujours le goulot. CTL ne depend pas du tout de l'entrainement par gradient. Avec un modele de fondation
tabulaire gele, TabPFN ici, il n'y a aucun parametre a copier: l'objet transfere
est le contexte, et le budget devient le nombre de lignes etiquetees que chaque
tache doit fournir.

C'est le regime ou le cout mord vraiment. L'inference en contexte coute
O(n^2 + n m^2) pour n lignes de contexte et m variables, donc reduire n d'un
facteur k divise le terme dominant par k^2, pour chacune des T series.

Protocole. WEAVE-UK, T sous-stations, regression sur 54 variables. Toutes les
taches ont un contexte de taille exactement n, donc un cout d'inference
identique. Ce qui varie est d'ou viennent les lignes.

    Individual-n   n lignes propres. Plafond de reference, et le seul bras qui
                   consomme n lignes etiquetees par tache.
    Individual-b   b lignes propres, contexte plus court, donc moins cher aussi.
    Star-b         b lignes propres plus n-b tirees du contexte de la racine.
    RandTree-b     b lignes propres plus n-b tirees d'un parent aleatoire.
    CTL-MST-b      b lignes propres plus n-b tirees du contexte du parent MST.

Les bras en cascade ne consomment que b lignes propres par tache, contre n pour
Individual-n, a cout d'inference egal. La question posee est donc: de combien
peut-on reduire les donnees propres de chaque tache en heritant du contexte d'une
tache voisine.

L'arbre, la distance X^T y et la racine au medoide sont ceux du papier. Les
lignes heritees viennent d'un contexte normalise par les statistiques du parent,
alors que la tache est normalisee par les siennes; c'est la meme hypothese que
fait CTL en copiant des parametres entre problemes normalises par tache.

    python3.12 fm_context_cascade.py --T 100 --n 64 --bs 4,8,16 --reps 3
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

from table1_nonconvex import weave_tasks, tree_mst, tree_random, CTL_DIR
from data import denormalize_y

torch.set_num_threads(max(1, torch.get_num_threads()))


def predictor(n_est):
    from tabpfn import TabPFNRegressor
    return lambda: TabPFNRegressor(device="cpu", n_estimators=n_est)


def rmse(v, pred, yte, st):
    p = denormalize_y(np.asarray(pred).reshape(-1), st)
    return float(np.sqrt(np.mean((np.asarray(yte).reshape(-1) - p) ** 2)))


def fit_predict(make, ctxX, ctxy, Xte):
    r = make()
    r.fit(np.asarray(ctxX, dtype=np.float64), np.asarray(ctxy, dtype=np.float64))
    return r.predict(np.asarray(Xte, dtype=np.float64))


def own_rows(X, y, v, k, rng):
    idx = rng.choice(len(X[v]), size=min(k, len(X[v])), replace=False)
    return X[v][idx], np.asarray(y[v]).reshape(-1)[idx]


def cascade_contexts(edges, root, X, y, b, n, rng):
    """Contexte de chaque tache: b lignes propres plus n-b heritees du parent.

    Parcours en largeur, donc le contexte du parent existe avant celui de
    l'enfant. La racine recoit n lignes propres, comme le prototype du papier.
    """
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    ctx = {root: own_rows(X, y, root, n, rng)}
    q = deque([root])
    while q:
        u = q.popleft()
        pX, py = ctx[u]
        for v in ch[u]:
            oX, oy = own_rows(X, y, v, b, rng)
            k = max(0, n - len(oX))
            if k and len(pX):
                j = rng.choice(len(pX), size=min(k, len(pX)), replace=False)
                cX = np.vstack([oX, pX[j]])
                cy = np.concatenate([oy, py[j]])
            else:
                cX, cy = oX, oy
            ctx[v] = (cX, cy)
            q.append(v)
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bs", default="4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-est", type=int, default=1)
    a = ap.parse_args()
    bs = [int(x) for x in a.bs.split(",")]
    make = predictor(a.n_est)

    allnodes, Xtr, ytr, Xte, yte, st, Dfull, root_full = weave_tasks()
    idx = {v: i for i, v in enumerate(allnodes)}

    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        nodes = sorted(rng.choice(allnodes, size=a.T, replace=False).tolist())
        sub = [idx[v] for v in nodes]
        D = Dfull[np.ix_(sub, sub)]
        root = nodes[int(np.argmin(D.sum(axis=1)))]      # medoide du sous-ensemble
        r_mst, e_mst = tree_mst(nodes, D, root)[:2]
        r_rnd, e_rnd = tree_random(nodes, D, root,
                                   np.random.default_rng(1000 + rep))[:2]
        e_star = [(root, v) for v in nodes if v != root]

        t0 = time.perf_counter()
        base_n = {v: fit_predict(make, *own_rows(Xtr, ytr, v, a.n, rng), Xte[v])
                  for v in nodes}
        rows.append(dict(rep=rep, b=a.n, method="Individual-n",
                         value=float(np.mean([rmse(v, base_n[v], yte[v], st[v])
                                              for v in nodes]))))
        print(f"  rep={rep} Individual-n en {time.perf_counter()-t0:.0f}s",
              flush=True)

        for b in bs:
            t0 = time.perf_counter()
            pr = {v: fit_predict(make, *own_rows(Xtr, ytr, v, b, rng), Xte[v])
                  for v in nodes}
            rows.append(dict(rep=rep, b=b, method="Individual-b",
                             value=float(np.mean([rmse(v, pr[v], yte[v], st[v])
                                                  for v in nodes]))))
            for name, (r_, e_) in (("Star", (root, e_star)),
                                   ("RandTree", (r_rnd, e_rnd)),
                                   ("CTL-MST", (r_mst, e_mst))):
                ctx = cascade_contexts(e_, r_, Xtr, ytr, b, a.n,
                                       np.random.default_rng(rep))
                pr = {v: fit_predict(make, ctx[v][0], ctx[v][1], Xte[v])
                      for v in nodes}
                rows.append(dict(rep=rep, b=b, method=name,
                                 value=float(np.mean([rmse(v, pr[v], yte[v], st[v])
                                                      for v in nodes]))))
            print(f"  rep={rep} b={b} en {time.perf_counter()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "fm_context_cascade_raw.csv",
                                      index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "fm_context_cascade_raw.csv", index=False)
    print(f"\n=== RMSE moyenne, T={a.T}, contexte n={a.n} ===", flush=True)
    print(df.pivot_table(index="b", columns="method",
                         values="value").round(1).to_string(), flush=True)

    from scipy import stats as S
    ref = df[df.method == "Individual-n"].sort_values("rep")["value"].values
    print(f"\n=== ecart au plafond Individual-n ({ref.mean():.1f}) ===", flush=True)
    for b in bs:
        g = df[df.b == b]
        out = []
        for m in ("Individual-b", "Star", "RandTree", "CTL-MST"):
            cur = g[g.method == m].sort_values("rep")["value"].values
            p = S.ttest_rel(cur, ref).pvalue if len(cur) > 1 else float("nan")
            out.append(f"{m} {cur.mean():7.1f} {100*(cur-ref).mean()/ref.mean():+6.1f}%"
                       f" (p={p:.3f})")
        print(f"b={b:3d} ({a.n//b}x moins de lignes propres): " + " | ".join(out),
              flush=True)


if __name__ == "__main__":
    main()
