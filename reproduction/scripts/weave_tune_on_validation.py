"""Phase de reglage de WEAVE-UK sur une validation, pas sur le test.

Les 500 postes de WEAVE-UK sont fixes : les graines ne changent que
l'initialisation, donc selectionner le pas sur le test revient a le selectionner
sur le jeu d'evaluation. Les notebooks du papier evitent cela en decoupant une
validation dans l'entrainement (`VAL_SIZE = 48`, expe-alignment-weave*.ipynb) ;
on reprend ce protocole.

Pendant le reglage : entrainement sur les lignes [48:], score sur les lignes
[:48] ramenees a l'echelle brute, et matrice de distances recalculee sur
l'entrainement reduit pour que la validation n'informe pas le graphe.

Ne recalcule que la phase de reglage. Les lignes d'evaluation, qui couvrent deja
tous les pas de la grille, sont reprises du sweep principal.

    python3.12 weave_tune_on_validation.py --stem t2pm
    CTL_ALLOC=floor1 python3.12 weave_tune_on_validation.py --stem t2pm_floor1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import table1_nonconvex as T1
from table1_nonconvex import (CONFIG, METHODS, make_model, eval_all, run_method,
                              weave_tasks, denormalize_y)
from graph import compute_gradient_distance
from table2_permethod_lowbudget import etas_for, budgets_for, TUNE_OFFSET

HERE = Path(__file__).resolve().parent
VAL_SIZE = 48                      # comme expe-alignment-weave*.ipynb


def split_train_val():
    """(nodes, X_tr, y_tr, X_val, y_val_raw, stats, D) sur l'entrainement reduit."""
    nodes, Xn, yn, _Xte, _yte, stats, _D, root = weave_tasks()
    X_tr = {v: Xn[v][VAL_SIZE:] for v in nodes}
    y_tr = {v: yn[v][VAL_SIZE:] for v in nodes}
    X_val = {v: Xn[v][:VAL_SIZE] for v in nodes}
    y_val = {v: denormalize_y(yn[v][:VAL_SIZE].reshape(-1), stats[v])
             for v in nodes}
    D = compute_gradient_distance(nodes, X_tr, y_tr)   # graphe sans la validation
    return nodes, X_tr, y_tr, X_val, y_val, stats, D, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="t2pm")
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--model", default="linear", choices=("linear", "mlp"))
    a = ap.parse_args()

    print(f"WEAVE-UK, reglage sur validation ({VAL_SIZE} lignes), "
          f"modele={a.model}, allocation={T1.ALLOC_MODE}, "
          f"sortie {a.stem}_weaveval_raw.csv", flush=True)
    nodes, X_tr, y_tr, X_val, y_val, stats, D, root = split_train_val()
    print(f"  {len(nodes)} taches, entrainement {X_tr[nodes[0]].shape[0]} lignes, "
          f"validation {X_val[nodes[0]].shape[0]} lignes", flush=True)

    cfg = CONFIG["WEAVE-UK"]
    d = X_tr[nodes[0]].shape[1]
    factory = lambda: make_model(a.model, d)
    rows = []
    for seed in range(TUNE_OFFSET, TUNE_OFFSET + a.tune_seeds):
        t0 = time.perf_counter()
        for B in budgets_for("WEAVE-UK"):
            for eta in etas_for("WEAVE-UK", a.model):
                rng = np.random.default_rng(10_000 + seed)
                for m in METHODS:
                    torch.manual_seed(seed)
                    try:
                        models = run_method(m, nodes, X_tr, y_tr, X_tr, y_tr, D,
                                            root, B, eta, cfg, factory, rng)
                        v = eval_all(models, X_val, y_val, "regression", stats)
                        if not np.isfinite(v):
                            v = np.nan
                    except RuntimeError:
                        v = np.nan
                    rows.append(dict(dataset="WEAVE-UK", model=a.model,
                                     T=len(nodes), B=B, b=B // 500, seed=seed,
                                     eta=eta, method=m, split="tune", value=v))
        print(f"  graine {seed} en {time.perf_counter()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{a.stem}_weaveval_raw.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
