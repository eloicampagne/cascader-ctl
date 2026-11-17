"""Ablation du proxy de distance de CTL-MST, au protocole des tables principales.

L'ancienne table tab:all_experiments venait d'un run anterieur (pas partage,
budgets absolus, Syn-2/Syn-5 non documentes). On la refait au protocole de
tab:simplified-opt :

  allocation floor1 (CTL_ALLOC=floor1), budget par tache b in {1, 2, 4}
  chaque proxy recoit son propre pas, choisi sur 5 graines de reglage
    (offset 500) disjointes des 20 graines d'evaluation
  WEAVE-UK regle sur la validation des 48 premieres lignes d'entrainement, avec
    les distances recalculees sur l'entrainement reduit
  meme racine (medoide des traits bruts) pour tous les proxys ; la matrice du
    proxy sert a construire le MST et aux poids SLE

Le proxy gradient doit reproduire la colonne CTL-MST de tab:simplified-opt.

    CTL_ALLOC=floor1 python3.12 proxy_ablation_floor1.py --dataset Syn-10
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import table1_nonconvex as T1
from table1_nonconvex import (CONFIG, make_model, eval_all, run_method,
                              syn_tasks, weave_tasks, image_tasks, celeba_tasks)
from graph import (compute_gradient_distance, compute_feature_distance,
                   compute_mean_cov_feature_distance, compute_cka_distance,
                   compute_target_distance, compute_kl_distance,
                   compute_wasserstein_distance, compute_js_target_distance,
                   compute_model_distance, select_seed_node)
from table2_permethod_lowbudget import etas_for, B_PER_TASK, TUNE_OFFSET, N_EVAL
from weave_tune_on_validation import split_train_val

HERE = Path(__file__).resolve().parent
PROXIES = {
    "Gradient": lambda n, X, y: compute_gradient_distance(n, X, y),
    "Feature": lambda n, X, y: compute_feature_distance(n, X, y),
    "MeanCov": lambda n, X, y: compute_mean_cov_feature_distance(n, X, y),
    "CKA": lambda n, X, y: compute_cka_distance(n, X, y),
    "Target": lambda n, X, y: compute_target_distance(n, y),
    "KL": lambda n, X, y: compute_kl_distance(n, y),
    "Wasserstein": lambda n, X, y: compute_wasserstein_distance(n, y),
    "JS": lambda n, X, y: compute_js_target_distance(n, X, y),
    "Model": lambda n, X, y: compute_model_distance(n, X, y),
}


def matrices(nodes, X, y):
    out = {}
    for k, f in PROXIES.items():
        D = np.asarray(f(nodes, X, y), dtype=float)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        out[k] = np.maximum(D, 0.0)
    return out


def load(ds, seed):
    """(nodes, X_tr, y_tr, X_ev, y_ev, stats, root, task) pour une graine."""
    if ds == "Syn-10":
        n, Xtr, ytr, Xte, yte, st, _D, r = syn_tasks(seed)
        return n, Xtr, ytr, Xte, yte, st, r, "regression"
    if ds == "CelebA":
        n, Xtr, ytr, Xte, yte = celeba_tasks(seed)
    else:
        n, Xtr, ytr, Xte, yte = image_tasks(ds, seed)
    return n, Xtr, ytr, Xte, yte, None, select_seed_node(Xtr, ytr), "classification"


def run_cell(ds, tag, seed, nodes, Xtr, ytr, Xev, yev, st, root, task, rows, Ds=None):
    cfg = CONFIG[ds]
    d = Xtr[nodes[0]].shape[1]
    factory = lambda: make_model("linear", d)
    Ds = Ds if Ds is not None else matrices(nodes, Xtr, ytr)
    for b in B_PER_TASK:
        B = b * len(nodes)
        for eta in etas_for(ds, "linear"):
            for name, D in Ds.items():
                torch.manual_seed(seed)
                rng = np.random.default_rng(10_000 + seed)
                try:
                    models = run_method("CTL-MST", nodes, Xtr, ytr, Xtr, ytr, D,
                                        root, B, eta, cfg, factory, rng)
                    v = eval_all(models, Xev, yev, task, st)
                    v = v if np.isfinite(v) else np.nan
                except RuntimeError:
                    v = np.nan
                rows.append(dict(dataset=ds, T=len(nodes), b=b, B=B, seed=seed,
                                 eta=eta, proxy=name, split=tag, value=v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=N_EVAL)
    a = ap.parse_args()
    ds = a.dataset
    stem = HERE / f"proxy_ablation_floor1_{ds}_raw.csv"
    print(f"=== {ds}, allocation={T1.ALLOC_MODE}, b={B_PER_TASK}, "
          f"{len(etas_for(ds, 'linear'))} pas, {a.tune_seeds}+{a.eval_seeds} graines",
          flush=True)
    rows = []
    if ds == "WEAVE-UK":
        n, Xtr, ytr, Xval, yval, st, _D, r = split_train_val()
        Dt = matrices(n, Xtr, ytr)          # taches fixes : une fois par phase
        for seed in range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds):
            t0 = time.perf_counter()
            run_cell(ds, "tune", seed, n, Xtr, ytr, Xval, yval, st, r, "regression", rows, Dt)
            print(f"  tune seed={seed} en {time.perf_counter()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(stem, index=False)
        n, Xtr, ytr, Xte, yte, st, _D, r = weave_tasks()
        De = matrices(n, Xtr, ytr)
        for seed in range(a.eval_seeds):
            t0 = time.perf_counter()
            run_cell(ds, "eval", seed, n, Xtr, ytr, Xte, yte, st, r, "regression", rows, De)
            print(f"  eval seed={seed} en {time.perf_counter()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(stem, index=False)
    else:
        seeds = ([("tune", s) for s in range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds)]
                 + [("eval", s) for s in range(a.eval_seeds)])
        for tag, seed in seeds:
            t0 = time.perf_counter()
            n, Xtr, ytr, Xev, yev, st, r, task = load(ds, seed)
            run_cell(ds, tag, seed, n, Xtr, ytr, Xev, yev, st, r, task, rows)
            print(f"  {tag} seed={seed} T={len(n)} en {time.perf_counter()-t0:.0f}s",
                  flush=True)
            pd.DataFrame(rows).to_csv(stem, index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
