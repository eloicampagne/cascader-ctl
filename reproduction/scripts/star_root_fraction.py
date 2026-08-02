"""Star : balayage de la fraction du budget total reservee a la racine.

La baseline Star du papier donne b_proto_frac du budget a la racine (0.14 a 0.15
selon la collection) et repartit le reste uniformement. Rien ne garantit que ce
soit le bon arbitrage, et a budget serre la racine peut se retrouver a un pas,
donc Star propage un modele non entraine.

On balaye f, la part du budget donnee a la racine, le reste etant reparti
uniformement sur les T-1 autres taches par plus grands restes (somme exacte a B).
Aucun plancher n'est impose : quand f est grand, certaines taches recoivent zero
pas et sont des copies exactes de la racine. A f = 1 la racine consomme tout le
budget et Star devient << entrainer une seule source, la copier partout >>.

Le pas d'apprentissage est selectionne pour chaque (collection, budget, f) sur
les graines de reglage, donc chaque variante est jugee a son propre optimum.

    python3.12 star_root_fraction.py
    python3.12 star_root_fraction.py --datasets WEAVE-UK --eval-seeds 3 --tune-seeds 1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import CONFIG, make_model, eval_all, cascade_train
from table2_permethod_lowbudget import etas_for, load, B_PER_TASK, TUNE_OFFSET

HERE = Path(__file__).resolve().parent
FRACS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0)


def star_alloc(nodes, root, B, f):
    """Racine = round(f*B), reste uniforme sur les autres, somme exacte a B."""
    others = [v for v in nodes if v != root]
    if f >= 1.0:
        return {root: B, **{v: 0 for v in others}}
    b_root = int(round(f * B))
    rest = B - b_root
    base, extra = divmod(rest, len(others))
    alloc = {v: base + (1 if i < extra else 0) for i, v in enumerate(others)}
    alloc[root] = b_root
    return alloc


def sweep(dataset, seeds, tag, rows, stem):
    cfg = CONFIG[dataset]
    for seed in seeds:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, D, root, task = load(dataset, seed)
        d = Xtr[n[0]].shape[1]
        factory = lambda: make_model("linear", d)
        dir_edges = [(root, v) for v in n if v != root]
        for B in (b * len(n) for b in B_PER_TASK):
            for f in FRACS:
                alloc = star_alloc(n, root, B, f)
                assert sum(alloc.values()) == B, (dataset, B, f)
                for eta in etas_for(dataset):
                    torch.manual_seed(seed)
                    try:
                        models = cascade_train(dir_edges, root, Xtr, ytr, alloc,
                                               eta, cfg["task"], cfg["reg"], factory)
                        v = eval_all(models, Xte, yte, task, st)
                        if not np.isfinite(v):
                            v = np.nan
                    except RuntimeError:
                        v = np.nan
                    rows.append(dict(dataset=dataset, T=len(n), B=B, b=B // len(n),
                                     frac=f, seed=seed, eta=eta, split=tag, value=v))
        print(f"  {dataset} {tag} seed={seed} en {time.perf_counter()-t0:.0f}s",
              flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{stem}_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="WEAVE-UK,CIFAR,Syn-10,FMNIST,CelebA")
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--stem", default="star_frac")
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        print(f"=== {ds} : fractions {FRACS} ===", flush=True)
        sweep(ds, range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds), "tune", rows, a.stem)
        sweep(ds, range(a.eval_seeds), "eval", rows, a.stem)
    pd.DataFrame(rows).to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
