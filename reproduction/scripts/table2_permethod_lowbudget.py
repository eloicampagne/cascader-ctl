"""Table 1 du papier refaite : un pas d'apprentissage propre a chaque methode,
et une grille de budgets exprimee en budget MOYEN par tache.

Motivation. Le pas publie est choisi par une recherche Optuna dont l'objectif est
la perte d'une cascade CTL-MSTc (`CTL/optim.py`), puis applique tel quel a
Individual, Star, RandTree et Chain. La table principale compare donc des
baselines a un pas regle sur la cascade. Ici chaque methode recoit son propre pas,
selectionne sur des graines disjointes de celles d'evaluation.

Budgets. L'allocation reste SLE pour les arbres informes, donc B/T est un budget
moyen par tache. On balaye b = 1, 2, 4 pas par tache, le regime serre que la
methode revendique, au lieu de budgets absolus qui donnent un b different d'une
collection a l'autre.

    python3.12 table2_permethod_lowbudget.py
    python3.12 table2_permethod_lowbudget.py --datasets Syn-10 --eval-seeds 2 --tune-seeds 1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import (CONFIG, MLP_ETA, METHODS, make_model, eval_all,
                              run_method, syn_tasks, weave_tasks, image_tasks,
                              celeba_tasks)
from graph import compute_gradient_distance, select_seed_node

HERE = Path(__file__).resolve().parent
GRID = (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
B_PER_TASK = (1, 2, 4)
T_OF = {"Syn-10": 200, "WEAVE-UK": 500, "FMNIST": 200, "CIFAR": 200, "CelebA": 400}
N_EVAL = 20
TUNE_OFFSET = 500
HIGHER = {"FMNIST", "CIFAR", "CelebA"}


def etas_for(dataset, model="linear"):
    """La grille, augmentee du pas publie pour ce jeu et cette classe de modele."""
    pub = CONFIG[dataset]["eta"] if model == "linear" else MLP_ETA[dataset]
    return tuple(sorted(set(GRID) | {round(float(pub), 8)}))


def budgets_for(dataset):
    return tuple(b * T_OF[dataset] for b in B_PER_TASK)


def load(dataset, seed):
    """(nodes, X_tr, y_tr, X_te, y_te, stats, D, root, task)."""
    if dataset == "WEAVE-UK":
        n, Xtr, ytr, Xte, yte, st, D, r = weave_tasks()
        return n, Xtr, ytr, Xte, yte, st, D, r, "regression"
    if dataset == "Syn-10":
        n, Xtr, ytr, Xte, yte, st, D, r = syn_tasks(seed)
        return n, Xtr, ytr, Xte, yte, st, D, r, "regression"
    if dataset == "CelebA":
        n, Xtr, ytr, Xte, yte = celeba_tasks(seed)
    else:
        n, Xtr, ytr, Xte, yte = image_tasks(dataset, seed)
    D = compute_gradient_distance(n, Xtr, ytr)
    return (n, Xtr, ytr, Xte, yte, None, D, select_seed_node(Xtr, ytr),
            "classification")


def sweep(dataset, seeds, tag, rows, stem, model="linear"):
    cfg = CONFIG[dataset]
    for seed in seeds:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, D, root, task = load(dataset, seed)
        d = Xtr[n[0]].shape[1]
        factory = lambda: make_model(model, d)
        # CelebA drops attribute pairs with too few samples, so the number of
        # tasks varies by seed (240 to 400). The budget is per task, so it has to
        # follow the actual count, otherwise the effective b varies with the seed.
        seed_budgets = tuple(b * len(n) for b in B_PER_TASK)
        for B in seed_budgets:
            for eta in etas_for(dataset, model):
                rng = np.random.default_rng(10_000 + seed)
                for m in METHODS:
                    torch.manual_seed(seed)
                    try:
                        models = run_method(m, n, Xtr, ytr, Xtr, ytr, D, root,
                                            B, eta, cfg, factory, rng)
                        v = eval_all(models, Xte, yte, task, st)
                        if not np.isfinite(v):
                            v = np.nan
                    except RuntimeError:
                        v = np.nan
                    rows.append(dict(dataset=dataset, model=model, T=len(n), B=B,
                                     b=B // len(n), seed=seed, eta=eta,
                                     method=m, split=tag, value=v))
        print(f"  {dataset} {tag} seed={seed} T={len(n)} en "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{stem}_raw.csv", index=False)


def report(df, stem):
    from scipy import stats as S
    out = []
    tune, ev = df[df.split == "tune"], df[df.split == "eval"]
    # grouped on b, not B: on CelebA the task count varies by seed, so B does too
    for (ds, b), g in ev.groupby(["dataset", "b"]):
        hi = ds in HIGHER
        picked = {}
        for m in METHODS:
            t = tune[(tune.dataset == ds) & (tune.b == b) & (tune.method == m)]
            s = t.groupby("eta")["value"].mean()
            picked[m] = s.idxmax() if hi else s.idxmin()
        for m in METHODS:
            v = g[(g.method == m) & (g.eta == picked[m])].sort_values("seed")["value"].values
            out.append(dict(dataset=ds, b=b, method=m,
                            eta=picked[m], mean=v.mean(),
                            se=v.std(ddof=1) / np.sqrt(len(v)), n=len(v)))
    o = pd.DataFrame(out)
    o.to_csv(HERE / f"{stem}_summary.csv", index=False)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,FMNIST,CIFAR,CelebA")
    ap.add_argument("--eval-seeds", type=int, default=N_EVAL)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--model", default="linear", choices=("linear", "mlp"))
    ap.add_argument("--stem", default="t2pm")
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        print(f"=== {ds}: T={T_OF[ds]}, B={budgets_for(ds)} (b={B_PER_TASK}), "
              f"modele={a.model}, {len(etas_for(ds, a.model))} pas, "
              f"{a.tune_seeds}+{a.eval_seeds} graines ===",
              flush=True)
        sweep(ds, range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds), "tune", rows,
              a.stem, a.model)
        sweep(ds, range(a.eval_seeds), "eval", rows, a.stem, a.model)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    o = report(df, a.stem)
    print("\n=== moyennes au pas retenu ===", flush=True)
    print(o.pivot_table(index=["dataset", "b"], columns="method",
                        values="mean").round(4).to_string(), flush=True)
    print("\n=== pas retenu ===", flush=True)
    print(o.pivot_table(index=["dataset", "b"], columns="method",
                        values="eta").to_string(), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
