"""CTL transfere une allocation de contexte, pas des parametres.

Modele de fondation tabulaire gele, TabICLv2 ici, N noeuds, aucun gradient. Le
cout d'inference en contexte est O(n^2 + n m^2) pour n lignes et m variables, donc
la longueur de contexte est la ressource a budgeter.

On fixe cette longueur a n pour chaque noeud, sans jamais donner a un noeud tout
son jeu de donnees. Ce qui varie est la composition du contexte: une distribution
w_v sur les noeuds sources, le noeud u fournissant round(n * w_{v,u}) lignes. Le
poids reflete la proximite des noeuds au sens du proxy de gradient du papier.

L'objet cascade est w_v. Chaque noeud herite de l'allocation de son parent, la
repondere par sa propre proximite, puis y reinjecte une part alpha de ses lignes
propres:

    w_v = normalise( (1 - alpha) * (w_pa * kappa_v) + alpha * delta_v ),
    kappa_v(u) = exp(-d(v,u) / tau).

Deroulee le long d'une branche, cette recursion donne un melange geometrique sur
les ancetres, exactement la structure du surrogate du papier. Et surtout, son
support est contenu dans celui du parent augmente de v, donc calculer w_v ne
demande que |support(w_pa)| evaluations de proximite, de l'ordre de la
profondeur, la ou une allocation globale en demande N par noeud, soit N^2 en tout.

Bras, tous a longueur de contexte n donc a cout d'inference identique:

    Individual-n    w = delta_v, n lignes propres. Plafond, et le seul bras qui
                    consomme n lignes etiquetees par noeud.
    Individual-b    b = alpha * n lignes propres, contexte plus court.
    Global          w propto exp(-d(v,u)/tau) sur les N noeuds. Reference haute,
                    coute N evaluations de proximite par noeud.
    CTL-<arbre>     la recursion ci-dessus, support borne a K sources.
    CTL-flat        la meme sans la reponderation kappa, pour separer ce
                    qu'apporte le transfert de l'allocation de ce qu'apporte la
                    reponderation par proximite.

    python3.12 fm_context_weights.py --T 100 --n 64 --alpha 0.125 --reps 3
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
TREES = ("Star", "RandTree", "CTL-MST", "Chain")


def regressor(backend, n_est):
    if backend == "tabicl":
        from tabicl import TabICLRegressor
        return lambda: TabICLRegressor(device="cpu", n_estimators=n_est)
    from tabpfn import TabPFNRegressor
    return lambda: TabPFNRegressor(device="cpu", n_estimators=n_est)


def fit_predict(make, cX, cy, Xte):
    r = make()
    r.fit(np.asarray(cX, dtype=np.float64), np.asarray(cy, dtype=np.float64))
    return r.predict(np.asarray(Xte, dtype=np.float64))


def rmse(pred, yte, st):
    p = denormalize_y(np.asarray(pred).reshape(-1), st)
    return float(np.sqrt(np.mean((np.asarray(yte).reshape(-1) - p) ** 2)))


def draw(X, y, w, n, rng):
    """Contexte de n lignes tire selon l'allocation w, un dict noeud -> poids."""
    src = sorted(w, key=lambda u: -w[u])
    tot = sum(w[u] for u in src)
    cX, cy, want = [], [], n
    for i, u in enumerate(src):
        k = want if i == len(src) - 1 else int(round(n * w[u] / tot))
        k = min(k, want, len(X[u]))
        if k <= 0:
            continue
        j = rng.choice(len(X[u]), size=k, replace=False)
        cX.append(X[u][j])
        cy.append(np.asarray(y[u]).reshape(-1)[j])
        want -= k
        if want <= 0:
            break
    return np.vstack(cX), np.concatenate(cy)


def ctl_weights(edges, root, nodes, D, nidx, alpha, tau, K, reweight):
    """Allocation de contexte cascadee le long de l'arbre.

    Renvoie aussi le nombre d'evaluations de proximite faites par noeud, qui est
    la taille du support du parent: c'est le cout que l'allocation globale paie
    en N.
    """
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    w = {root: {root: 1.0}}
    cost = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            wp = w[u]
            cost[v] = len(wp)
            nw = defaultdict(float)
            for s, ws in wp.items():
                k = (np.exp(-D[nidx[v], nidx[s]] / tau) if reweight else 1.0)
                nw[s] += (1.0 - alpha) * ws * k
            nw[v] += alpha
            tot = sum(nw.values())
            nw = {s: x / tot for s, x in nw.items()}
            if len(nw) > K:
                keep = sorted(nw, key=lambda s: -nw[s])[:K]
                tot = sum(nw[s] for s in keep)
                nw = {s: nw[s] / tot for s in keep}
            w[v] = nw
            q.append(v)
    return w, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tabicl", choices=["tabicl", "tabpfn"])
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.125)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-est", type=int, default=1)
    a = ap.parse_args()
    make = regressor(a.backend, a.n_est)
    b = max(1, int(round(a.alpha * a.n)))

    allnodes, Xtr, ytr, Xte, yte, st, Dfull, _ = weave_tasks()
    gidx = {v: i for i, v in enumerate(allnodes)}

    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        nodes = sorted(rng.choice(allnodes, size=a.T, replace=False).tolist())
        sub = [gidx[v] for v in nodes]
        D = Dfull[np.ix_(sub, sub)]
        nidx = {v: i for i, v in enumerate(nodes)}
        tau = float(np.median(D[np.triu_indices(a.T, k=1)]))
        root = nodes[int(np.argmin(D.sum(axis=1)))]
        trees = {"Star": (root, [(root, v) for v in nodes if v != root]),
                 "RandTree": tree_random(nodes, D, root,
                                         np.random.default_rng(1000 + rep))[:2],
                 "CTL-MST": tree_mst(nodes, D, root)[:2],
                 "Chain": tree_chain(nodes, D, root)[:2]}

        def run(name, W, cost):
            t0 = time.perf_counter()
            g = np.random.default_rng(rep)
            errs = []
            for v in nodes:
                cX, cy = draw(Xtr, ytr, W[v], a.n, g)
                errs.append(rmse(fit_predict(make, cX, cy, Xte[v]), yte[v], st[v]))
            rows.append(dict(rep=rep, method=name, value=float(np.mean(errs)),
                             prox_evals=float(np.mean([cost[v] for v in nodes])),
                             secs=time.perf_counter() - t0))
            print(f"  rep={rep} {name:16s} {np.mean(errs):7.0f} "
                  f"prox/noeud {np.mean([cost[v] for v in nodes]):5.1f} "
                  f"{time.perf_counter()-t0:5.0f}s", flush=True)

        run("Individual-n", {v: {v: 1.0} for v in nodes}, {v: 0 for v in nodes})
        t0 = time.perf_counter()
        g = np.random.default_rng(rep)
        errs = [rmse(fit_predict(make, *draw(Xtr, ytr, {v: 1.0}, b, g), Xte[v]),
                     yte[v], st[v]) for v in nodes]
        rows.append(dict(rep=rep, method="Individual-b", value=float(np.mean(errs)),
                         prox_evals=0.0, secs=time.perf_counter() - t0))
        print(f"  rep={rep} {'Individual-b':16s} {np.mean(errs):7.0f}", flush=True)

        Wg = {}
        for v in nodes:
            k = np.exp(-D[nidx[v]] / tau)
            k[nidx[v]] = 0.0
            top = np.argsort(-k)[:a.K - 1]
            d = {nodes[j]: float(k[j]) for j in top}
            s = sum(d.values()) or 1.0
            d = {u: (1 - a.alpha) * x / s for u, x in d.items()}
            d[v] = a.alpha
            Wg[v] = d
        run("Global", Wg, {v: a.T for v in nodes})

        for t in TREES:
            r_, e_ = trees[t]
            W, c = ctl_weights(e_, r_, nodes, D, nidx, a.alpha, tau, a.K, True)
            run(t if t.startswith("CTL") else f"CTL-{t}", W, c)
        r_, e_ = trees["CTL-MST"]
        W, c = ctl_weights(e_, r_, nodes, D, nidx, a.alpha, tau, a.K, False)
        run("CTL-MST-noreweight", W, c)
        pd.DataFrame(rows).to_csv(HERE / "fm_context_weights_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "fm_context_weights_raw.csv", index=False)
    print(f"\n=== {a.backend}, T={a.T}, n={a.n}, alpha={a.alpha} (b={b}) ===",
          flush=True)
    print(df.groupby("method")[["value", "prox_evals", "secs"]].mean()
          .round(1).sort_values("value").to_string(), flush=True)

    from scipy import stats as S
    ref = df[df.method == "Individual-n"].sort_values("rep")["value"].values
    print(f"\n=== ecart au plafond Individual-n ({ref.mean():.0f}) ===", flush=True)
    for m in df.method.unique():
        if m == "Individual-n":
            continue
        cur = df[df.method == m].sort_values("rep")["value"].values
        p = S.ttest_rel(cur, ref).pvalue if len(cur) > 1 else float("nan")
        print(f"  {m:16s} {cur.mean():7.0f} {100*(cur-ref).mean()/ref.mean():+6.1f}%"
              f" (p={p:.3f})", flush=True)


if __name__ == "__main__":
    main()
