"""Star sous l'allocation uniforme (celle du papier) et sous SLE (celle des cascades).

Star recoit dans le papier une allocation uniforme avec une fraction b_proto_frac
reservee a la racine, alors que Chain et les arbres CTL recoivent SLE. La
comparaison n'est donc pas a allocation egale. Sous SLE, un noeud feuille v de
l'etoile pese log(1 + d(racine, v)), donc les taches les plus eloignees de la
racine recoivent davantage de raffinement.

A b = 1 le surplus B - T est nul, donc les deux allocations donnent un pas a
chaque tache et coincident : l'ecart ne peut apparaitre qu'a b >= 2.

    python3.12 star_alloc_test.py
    python3.12 star_alloc_test.py --datasets WEAVE-UK --eval-seeds 3 --tune-seeds 1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import (CONFIG, make_model, eval_all, cascade_train,
                              sle_alloc, uniform_alloc)
from table2_permethod_lowbudget import (etas_for, load, B_PER_TASK, TUNE_OFFSET,
                                        HIGHER)

HERE = Path(__file__).resolve().parent
VARIANTS = ("Star-uniform", "Star-SLE")


def run_star(variant, nodes, X_tr, y_tr, D, root, B, eta, cfg, factory):
    dir_edges = [(root, v) for v in nodes if v != root]
    if variant == "Star-SLE":
        alloc = sle_alloc(nodes, root, dir_edges, D, B)
    else:
        alloc = uniform_alloc(nodes, root, dir_edges, B, cfg["b_proto_frac"])
    return cascade_train(dir_edges, root, X_tr, y_tr, alloc, eta,
                         cfg["task"], cfg["reg"], factory)


def sweep(dataset, seeds, tag, rows, stem):
    cfg = CONFIG[dataset]
    for seed in seeds:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, D, root, task = load(dataset, seed)
        d = Xtr[n[0]].shape[1]
        factory = lambda: make_model("linear", d)
        for B in (b * len(n) for b in B_PER_TASK):
            for eta in etas_for(dataset):
                for variant in VARIANTS:
                    torch.manual_seed(seed)
                    try:
                        models = run_star(variant, n, Xtr, ytr, D, root, B, eta,
                                          cfg, factory)
                        v = eval_all(models, Xte, yte, task, st)
                        if not np.isfinite(v):
                            v = np.nan
                    except RuntimeError:
                        v = np.nan
                    rows.append(dict(dataset=dataset, T=len(n), B=B,
                                     b=B // len(n), seed=seed, eta=eta,
                                     method=variant, split=tag, value=v))
        print(f"  {dataset} {tag} seed={seed} en {time.perf_counter()-t0:.0f}s",
              flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{stem}_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,FMNIST,CIFAR,CelebA")
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--stem", default="star_alloc")
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        print(f"=== {ds} ===", flush=True)
        sweep(ds, range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds), "tune", rows, a.stem)
        sweep(ds, range(a.eval_seeds), "eval", rows, a.stem)
    pd.DataFrame(rows).to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
