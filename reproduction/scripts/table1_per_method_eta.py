"""Table 1 du papier, refaite avec un pas d'apprentissage propre a chaque methode.

Le reviewer Lufy demande si eta a ete regle sur CTL puis applique aux baselines.
`CTL/optim.py` confirme que oui: l'objectif Optuna entraine une cascade et
minimise sa perte. La reponse promet de remplacer ce protocole par un reglage par
methode dans tout le papier. Ce fichier le fait.

Tout le reste est celui de `table1_nonconvex.py`: memes chargeurs, meme
`run_method`, memes allocations (SLE pour Chain/kNN/MST/MSTc, uniforme pour Star
et RandTree, B//T pour Individual), memes budgets, meme nombre de graines, les
deux classes de modele local.

Le pas n'est pas choisi sur les graines rapportees. Une grille log est evaluee sur
un jeu de graines disjoint (indices decales de 500, donc tirages de taches
differents pour Syn-10 et pour les images), le meilleur pas est retenu par
(jeu, modele, budget, methode), et seules les graines d'evaluation sont
rapportees. Reserve: sur WEAVE-UK les taches sont fixees, donc les graines ne
different que par l'initialisation et le jeu de test est partage entre reglage et
evaluation.

La grille contient les valeurs publiees par jeu et par classe de modele, de sorte
que la selection ne peut pas faire pire que la configuration du papier.

    python3.12 table1_per_method_eta.py --datasets Syn-10,WEAVE-UK,FMNIST,CIFAR
    python3.12 table1_per_method_eta.py --datasets CIFAR --eval-seeds 3 --tune-seeds 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, METHODS, make_model, eval_all,
                              run_method, syn_tasks, weave_tasks, image_tasks,
                              N_SEEDS_SYN, N_SEEDS_IMG, N_REPS_WEAVE)
from graph import compute_gradient_distance, select_seed_node

GRID = (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
BUDGETS = {"Syn-10": (500, 1000, 2000), "WEAVE-UK": (500, 1000, 2000),
           "FMNIST": (500, 2000), "CIFAR": (500, 2000)}
N_SEEDS = {"Syn-10": N_SEEDS_SYN, "WEAVE-UK": N_REPS_WEAVE,
           "FMNIST": N_SEEDS_IMG, "CIFAR": N_SEEDS_IMG}
HIGHER = {"FMNIST", "CIFAR", "CelebA"}
TUNE_OFFSET = 500


def etas_for(dataset, mc):
    """La grille, augmentee du pas publie pour ce jeu et cette classe."""
    pub = CONFIG[dataset]["eta"] if mc == "linear" else MLP_ETA[dataset]
    return tuple(sorted(set(GRID) | {round(float(pub), 8)}))


def load(dataset, seed):
    """Renvoie (nodes, X_tr, y_tr, X_te, y_te, stats, D, root, task)."""
    if dataset == "WEAVE-UK":
        n, Xtr, ytr, Xte, yte, st, D, r = weave_tasks()
        return n, Xtr, ytr, Xte, yte, st, D, r, "regression"
    if dataset == "Syn-10":
        n, Xtr, ytr, Xte, yte, st, D, r = syn_tasks(seed)
        return n, Xtr, ytr, Xte, yte, st, D, r, "regression"
    n, Xtr, ytr, Xte, yte = image_tasks(dataset, seed)
    D = compute_gradient_distance(n, Xtr, ytr)
    return (n, Xtr, ytr, Xte, yte, None, D, select_seed_node(Xtr, ytr),
            "classification")


def sweep(dataset, seeds, tag, rows, stem):
    cfg = CONFIG[dataset]
    for seed in seeds:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, D, root, task = load(dataset, seed)
        d = Xtr[n[0]].shape[1]
        for mc in ("linear", "mlp"):
            factory = lambda: make_model(mc, d)
            for B in BUDGETS[dataset]:
                for eta in etas_for(dataset, mc):
                    rng = np.random.default_rng(10_000 + seed)
                    for m in METHODS:
                        torch.manual_seed(seed)
                        try:
                            models = run_method(m, n, Xtr, ytr, Xtr, ytr, D,
                                                root, B, eta, cfg, factory, rng)
                            v = eval_all(models, Xte, yte, task, st)
                            if not np.isfinite(v):
                                v = np.nan
                        except RuntimeError:
                            v = np.nan              # divergence
                        rows.append(dict(dataset=dataset, model=mc, B=B,
                                         seed=seed, eta=eta, method=m,
                                         split=tag, value=v))
        print(f"  {dataset} {tag} seed={seed} en {time.perf_counter()-t0:.0f}s",
              flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{stem}_raw.csv", index=False)


def report(df, stem):
    from scipy import stats as S
    out = []
    tune = df[df.split == "tune"]
    ev = df[df.split == "eval"]
    for (ds, mc, B), g in ev.groupby(["dataset", "model", "B"]):
        hi = ds in HIGHER
        picked, vals = {}, {}
        for m in METHODS:
            t = tune[(tune.dataset == ds) & (tune.model == mc) &
                     (tune.B == B) & (tune.method == m)]
            mean = t.groupby("eta")["value"].mean().dropna()
            if mean.empty:
                continue
            e = mean.idxmax() if hi else mean.idxmin()
            picked[m] = e
            s_ = g[(g.method == m) & (g.eta == e)].sort_values("seed")
            vals[m] = s_["value"].values
        if "Individual" not in vals:
            continue
        base = vals["Individual"]
        casc = {m: v for m, v in vals.items() if m not in ("Individual", "Star")}
        best = (max(casc, key=lambda m: np.nanmean(casc[m])) if hi else
                min(casc, key=lambda m: np.nanmean(casc[m])))
        p = S.ttest_rel(casc[best], base, nan_policy="omit").pvalue
        for m, v in vals.items():
            out.append(dict(dataset=ds, model=mc, B=B, method=m,
                            eta=picked[m], mean=np.nanmean(v),
                            std=np.nanstd(v), n=int(np.isfinite(v).sum())))
        gap = (np.nanmean(casc[best]) - np.nanmean(base)) / abs(np.nanmean(base))
        print(f"{ds:9s} {mc:6s} B={B:5d}: Individual {np.nanmean(base):9.4f} "
              f"(eta {picked['Individual']:g}) | meilleure cascade {best} "
              f"{np.nanmean(casc[best]):9.4f} (eta {picked[best]:g}) | "
              f"{100*gap:+6.1f}% (p={p:.4f})", flush=True)
    o = pd.DataFrame(out)
    o.to_csv(HERE / f"{stem}_summary.csv", index=False)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,FMNIST,CIFAR")
    ap.add_argument("--eval-seeds", type=int, default=0, help="0 = celui du papier")
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--stem", default="t1_permethod")
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        ne = a.eval_seeds or N_SEEDS[ds]
        print(f"=== {ds}: {ne} graines d'evaluation, {a.tune_seeds} de reglage, "
              f"{len(etas_for(ds, 'mlp'))} pas ===", flush=True)
        sweep(ds, range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds), "tune",
              rows, a.stem)
        sweep(ds, range(ne), "eval", rows, a.stem)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    print("\n=== pas retenu par methode, et cascade contre Individual ===",
          flush=True)
    o = report(df, a.stem)
    print("\n=== Table 1, chaque methode a son pas ===", flush=True)
    for mc in ("linear", "mlp"):
        s_ = o[o.model == mc]
        if s_.empty:
            continue
        print(f"\n-- modele local {mc} --", flush=True)
        print(s_.pivot_table(index=["dataset", "B"], columns="method",
                             values="mean").round(4).to_string(), flush=True)
        print(s_.pivot_table(index=["dataset", "B"], columns="method",
                             values="eta").to_string(), flush=True)


if __name__ == "__main__":
    main()
