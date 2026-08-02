"""Temps d'horloge reel de la cascade, a P travailleurs, sur taches synthetiques.

Le comptage de `critical_path.py` est structurel. On le confronte ici a une mesure.
La cascade s'execute par niveaux de profondeur, toutes les taches d'un meme niveau
etant lancees ensemble puisqu'elles ne dependent que de parents deja entraines, avec
une barriere entre niveaux. La chaine n'a que des niveaux de largeur un, donc P ne
lui sert a rien; l'arbre doit approcher son facteur structurel.

Deux precautions. On utilise des threads et non des processus, de sorte que les
modeles restent en memoire partagee et qu'aucun pickling ne pollue la mesure, et on
force torch.set_num_threads(1) pour que le parallelisme de taches ne concurrence pas
celui des BLAS. Les taches sont dimensionnees pour qu'un pas de gradient coute assez
cher pour dominer l'overhead, faute de quoi on mesurerait Python.

Borne attendue a P travailleurs: max(chemin critique, travail total / P).

    python3.12 parallel_wallclock.py
"""

import copy
import random
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
CTL_DIR = HERE.parent.parent.parent / "CTL"
sys.path.insert(0, str(CTL_DIR))

torch.set_num_threads(1)

from data import generate_clustered_tasks, normalize_xy
from graph import compute_gradient_distance, select_seed_node
from table1_nonconvex import sle_alloc, uniform_alloc, tree_chain, tree_mst

T, K, D_IN, HIDDEN, N_TR = 200, 5, 100, 256, 2048
BUDGETS = (2000,)
WORKERS = (1, 2, 4, 8)
REPS = 8
ETA = 0.05
PROTO_FRAC = 0.10


def make_model():
    return nn.Sequential(nn.Linear(D_IN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))


def train_steps(model, X, y, steps):
    if steps <= 0:
        return model
    crit = nn.MSELoss()
    for _ in range(steps):
        loss = crit(model(X).reshape(-1), y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-ETA)
    return model


def tasks(rep):
    rng = np.random.default_rng(1000 * rep + 3)
    _, _, _, (Xt, yt), _, _ = generate_clustered_tasks(
        T=T, d=D_IN, n_train=N_TR, n_test=8, n_clusters=K,
        tau_within=1.0, tau_between=5.0, sigma=0.5, rng=rng)
    nodes = list(range(T))
    Xn, yn = {}, {}
    for v in nodes:
        a, b, _ = normalize_xy(Xt[v], yt[v])
        Xn[v] = torch.from_numpy(a).float()
        yn[v] = torch.from_numpy(b).float().reshape(-1)
    Dm = compute_gradient_distance(nodes, {v: Xt[v] for v in nodes},
                                   {v: yt[v] for v in nodes})
    root = select_seed_node({v: Xt[v] for v in nodes}, {v: yt[v] for v in nodes})
    return nodes, Xn, yn, Dm, root


def levels_of(edges, root):
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    lv, dep, q = defaultdict(list), {root: 0}, deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            dep[v] = dep[u] + 1
            lv[dep[v]].append((u, v))
            q.append(v)
    return [lv[d] for d in sorted(lv)]


def critical_path(edges, root, alloc):
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    cum, q = {root: alloc.get(root, 0)}, deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            cum[v] = cum[u] + alloc.get(v, 0)
            q.append(v)
    return max(cum.values())


def run(edges, root, X, y, alloc, workers):
    models = {root: train_steps(make_model(), X[root], y[root],
                                int(alloc.get(root, 0)))}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for level in levels_of(edges, root):
            futs = {v: ex.submit(train_steps, copy.deepcopy(models[u]),
                                 X[v], y[v], int(alloc.get(v, 0)))
                    for u, v in level}
            for v, f in futs.items():
                models[v] = f.result()
    return models


rows = []
for rep in range(REPS):
    nodes, X, y, Dm, root = tasks(rep)
    trees = {}
    for name, builder in (("Chain", tree_chain), ("CTL-MST", tree_mst)):
        r, edges, _ = builder(nodes, Dm, root)
        trees[name] = (r, edges)
    # chauffe, pour que la premiere configuration mesuree ne paie pas la mise en
    # cache ni la montee en frequence
    r0, e0 = trees["Chain"]
    run(e0, r0, X, y, {v: 2 for v in nodes}, 4)

    cfgs = [(n, k, P) for n in trees for k in ("flat", "sle") for P in WORKERS]
    random.Random(rep).shuffle(cfgs)   # l'ordre ne doit pas correler avec la methode
    for name, kind, P in cfgs:
        r, edges = trees[name]
        B = BUDGETS[0]
        alloc = (sle_alloc(nodes, r, edges, Dm, B) if kind == "sle"
                 else uniform_alloc(nodes, r, edges, B, PROTO_FRAC))
        cp = critical_path(edges, r, alloc)
        total = sum(alloc.values())
        t0 = time.perf_counter()
        run(edges, r, X, y, alloc, P)
        dt = time.perf_counter() - t0
        rows.append(dict(rep=rep, B=B, method=name, alloc=kind, workers=P,
                         seconds=round(dt, 3), crit=cp, total=total,
                         steps_per_s=round(total / dt, 1),
                         bound=max(cp, total / P)))
    print(f"  rep{rep} fait", flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "parallel_wallclock.csv", index=False)
print("\n=== pas de gradient totaux, controle ===")
print(df.groupby(["method", "alloc"])["total"].first().to_string())
print("\n=== secondes, mediane sur 8 repetitions, ecart interquartile ===")
for (m, k, P), g in df.groupby(["method", "alloc", "workers"]):
    q1, q3 = g.seconds.quantile([0.25, 0.75])
    print(f"{m:8s} {k:4s} P={P}: {g.seconds.median():5.2f}s  [{q1:.2f}, {q3:.2f}]")
print("\n=== acceleration mesuree contre P=1, et plafond structurel ===")
for (m, k), g in df.groupby(["method", "alloc"]):
    base = g[g.workers == 1]["seconds"].median()
    cp, tot = g.crit.iloc[0], g.total.iloc[0]
    out = [f"P={P} x{base / g[g.workers == P]['seconds'].median():.2f}"
           for P in WORKERS[1:]]
    print(f"{m:8s} {k:4s} cp={cp:5d} total={tot:5d} plafond x{tot/cp:.2f}: "
          + " | ".join(out))
print("\n=== a P=1, arbre contre chaine, meme budget ===")
for k in ("flat", "sle"):
    g = df[(df.alloc == k) & (df.workers == 1)]
    c = g[g.method == "Chain"]["seconds"]
    t = g[g.method == "CTL-MST"]["seconds"]
    from scipy import stats as S
    print(f"{k:4s}: Chain {c.median():.2f}s  CTL-MST {t.median():.2f}s  "
          f"ratio {t.median()/c.median():.2f}  p={S.mannwhitneyu(t, c).pvalue:.3f}")
