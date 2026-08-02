"""Cascades : balayage de la fraction du budget total reservee a la racine.

Pendant du balayage fait sur Star dans `star_root_fraction.py`. La fraction
donnee a la racine y valait 0.10, codee en dur dans `_alloc_from_weights`, alors
que Star recevait `b_proto_frac` (0.14 a 0.15). Comparer Star a sa meilleure
fraction contre une cascade a fraction fixe serait le meme biais a l'envers.

Pour chaque fraction f : la racine recoit round(f*B), le reste est reparti sur
les T-1 autres taches proportionnellement aux poids SLE |s_v| log(1 + d(pa,v)),
par plus grands restes, somme exacte a B. Aucun plancher : a f grand, une partie
des taches recoit zero pas et herite du parent sans raffinement.

Le pas d'apprentissage est selectionne pour chaque (collection, budget, methode,
f) sur les graines de reglage.

    python3.12 cascade_root_fraction.py
    python3.12 cascade_root_fraction.py --datasets WEAVE-UK --methods CTL-MSTc
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import (CONFIG, make_model, eval_all, cascade_train,
                              tree_chain, tree_knn, tree_mst, tree_mstc)
from table2_permethod_lowbudget import etas_for, load, B_PER_TASK, TUNE_OFFSET
from graph import _compute_subtree_sizes

HERE = Path(__file__).resolve().parent
FRACS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0)


def build(method, nodes, D, seed_node, X, y, lam):
    if method == "Chain":
        return tree_chain(nodes, D, seed_node)[:2]
    if method == "CTL-kNN":
        return tree_knn(nodes, D, seed_node)[:2]
    if method == "CTL-MST":
        return tree_mst(nodes, D, seed_node)[:2]
    if method == "CTL-MSTc":
        return tree_mstc(nodes, D, seed_node, X, y, lam)[:2]
    raise ValueError(method)


def sle_weights(nodes, root, dir_edges, D):
    nidx = {v: i for i, v in enumerate(nodes)}
    pmap = {v: u for u, v in dir_edges}
    sz = _compute_subtree_sizes(pmap, root, nodes)
    return np.array([sz[v] * np.log1p(D[nidx[pmap[v]], nidx[v]])
                     for v in nodes if v != root], dtype=float)


def alloc_with_frac(nodes, root, dir_edges, D, B, f):
    others = [v for v in nodes if v != root]
    if f >= 1.0:
        return {root: B, **{v: 0 for v in others}}
    b_root = int(round(f * B))
    rest = B - b_root
    w = sle_weights(nodes, root, dir_edges, D)
    if w.sum() < 1e-12:
        w = np.ones(len(others))
    t = w / w.sum() * rest
    b = np.floor(t).astype(int)
    k = int(rest - b.sum())
    if k > 0:
        for i in np.argsort(-(t - np.floor(t)))[:k]:
            b[i] += 1
    alloc = {v: int(b[i]) for i, v in enumerate(others)}
    alloc[root] = b_root
    return alloc


def sweep(dataset, methods, seeds, tag, rows, stem):
    cfg = CONFIG[dataset]
    for seed in seeds:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, D, root0, task = load(dataset, seed)
        d = Xtr[n[0]].shape[1]
        factory = lambda: make_model("linear", d)
        trees = {m: build(m, n, D, root0, Xtr, ytr, cfg["mstc_lambda"])
                 for m in methods}
        for B in (b * len(n) for b in B_PER_TASK):
            for m in methods:
                root, dir_edges = trees[m]
                for f in FRACS:
                    alloc = alloc_with_frac(n, root, dir_edges, D, B, f)
                    assert sum(alloc.values()) == B, (dataset, m, B, f)
                    for eta in etas_for(dataset):
                        torch.manual_seed(seed)
                        try:
                            models = cascade_train(dir_edges, root, Xtr, ytr,
                                                   alloc, eta, cfg["task"],
                                                   cfg["reg"], factory)
                            v = eval_all(models, Xte, yte, task, st)
                            if not np.isfinite(v):
                                v = np.nan
                        except RuntimeError:
                            v = np.nan
                        rows.append(dict(dataset=dataset, T=len(n), B=B,
                                         b=B // len(n), method=m, frac=f,
                                         seed=seed, eta=eta, split=tag, value=v))
        print(f"  {dataset} {tag} seed={seed} en {time.perf_counter()-t0:.0f}s",
              flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{stem}_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="WEAVE-UK,CIFAR,Syn-10,FMNIST,CelebA")
    ap.add_argument("--methods", default="Chain,CTL-MST,CTL-MSTc")
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--stem", default="casc_frac")
    a = ap.parse_args()
    methods = a.methods.split(",")

    rows = []
    for ds in a.datasets.split(","):
        print(f"=== {ds} : {methods} x fractions {FRACS} ===", flush=True)
        sweep(ds, methods, range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds),
              "tune", rows, a.stem)
        sweep(ds, methods, range(a.eval_seeds), "eval", rows, a.stem)
    pd.DataFrame(rows).to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
