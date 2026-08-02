"""Syn-10 avec la geometrie du papier (tau_within=5), pas propre a chaque methode.

`table1_nonconvex.syn_tasks` construit la collection avec tau_within=tau_between=10,
donc des grappes qui se recouvrent, alors que la Table 1 du papier utilise
tau_within=5, tau_between=10. Les valeurs Syn-10 des tables non convexe et
par-methode ne sont donc pas comparables a la Table 1. Ce script rejoue le
protocole de `table1_per_method_eta.py` sur Syn-10 seul, avec tau_within=5, et
rejoue aussi la ligne au pas publie pour la table non convexe.

    python3.12 syn_papergeom_permethod.py                 # tout
    python3.12 syn_papergeom_permethod.py --tune-seeds 1 --eval-seeds 1   # test
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import table1_nonconvex as T1
from table1_nonconvex import CONFIG, MLP_ETA, METHODS, make_model, eval_all, run_method
import table1_per_method_eta as PM

HERE = Path(__file__).resolve().parent

# `table1_nonconvex.syn_tasks` prend d=3 et tau_within=10. Le papier utilise d=10
# (d'ou le nom Syn-10, et le defaut de generate_clustered_tasks) et tau_within=5.
# Preuve pour la dimension : sous d=3 un modele qui predit zero atteint MSE 390,
# or la Table 1 rapporte 513 a 949, donc elle ne peut pas etre en dimension 3.
D_PAPER = 10
TAU_WITHIN_PAPER = 5.0


def syn_tasks_paper(seed):
    """La collection Syn-10 telle que definie dans le papier."""
    return T1.syn_tasks(seed, T=200, d=D_PAPER, K=5, tau=TAU_WITHIN_PAPER,
                        tau_b=10.0, sigma=5.0, n_tr=64, n_te=128)


def sweep(seeds, tag, rows, stem):
    cfg = CONFIG["Syn-10"]
    for seed in seeds:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, D, root = syn_tasks_paper(seed)
        d = Xtr[n[0]].shape[1]
        for mc in ("linear", "mlp"):
            factory = lambda: make_model(mc, d)
            for B in PM.BUDGETS["Syn-10"]:
                for eta in PM.etas_for("Syn-10", mc):
                    rng = np.random.default_rng(10_000 + seed)
                    for m in METHODS:
                        torch.manual_seed(seed)
                        try:
                            models = run_method(m, n, Xtr, ytr, Xtr, ytr, D,
                                                root, B, eta, cfg, factory, rng)
                            v = eval_all(models, Xte, yte, "regression", st)
                            if not np.isfinite(v):
                                v = np.nan
                        except RuntimeError:
                            v = np.nan
                        rows.append(dict(dataset="Syn-10", model=mc, B=B,
                                         seed=seed, eta=eta, method=m,
                                         split=tag, value=v))
        print(f"  {tag} seed={seed} en {time.perf_counter()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{stem}_raw.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--stem", default="syn_papergeom")
    a = ap.parse_args()

    print(f"Syn-10, tau_within={TAU_WITHIN_PAPER} (geometrie du papier), "
          f"{a.eval_seeds} graines d'evaluation, {a.tune_seeds} de reglage, "
          f"{len(PM.etas_for('Syn-10', 'mlp'))} pas MLP / "
          f"{len(PM.etas_for('Syn-10', 'linear'))} lineaires", flush=True)

    rows = []
    sweep(range(PM.TUNE_OFFSET, PM.TUNE_OFFSET + a.tune_seeds), "tune", rows, a.stem)
    sweep(range(a.eval_seeds), "eval", rows, a.stem)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    PM.report(df, a.stem)
    print("done", flush=True)


if __name__ == "__main__":
    main()
