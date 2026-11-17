"""Grille dispersion x budget, version retenue pour le corps du texte.

Deux changements par rapport a tworegime_configs_eta.

L'axe des budgets est recentre sur la frontiere. L'ancienne grille allait de
b = 1 a b = 10, soit trois colonnes sous le croisement et six au-dessus, ce qui
sur-echantillonne le cote de la chaine sans raison. La nouvelle s'arrete a b = 6
et resserre le pas autour de 1.75. On ne descend pas sous b = 1 : floor1 exige
B >= T et bascule sinon sur l'allocation exacte, ce qui changerait le sens de la
table.

CTL-Forest entre dans la comparaison. C'est la construction que le diagnostic
d'accumulation designe : brancher entre grappes, chainer dedans.

    CTL_ALLOC=floor1 python3.12 tworegime_final.py --eval-reps 20 --tune-reps 5
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import (CONFIG, cascade_train, eval_all, make_model,
                              sle_alloc, tree_chain, tree_forest, tree_mst)
from tworegime_configs import SCATTERS, T, set_alloc_mode, tasks

HERE = Path(__file__).parent.resolve()
B_PER_TASK = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6)
METHODS = ("Chain", "CTL-MST", "CTL-Forest")
ETAS = (0.1, 0.2, 0.3, 0.5, 0.7)
TUNE_OFFSET = 500
BUILD = {"Chain": lambda n, D, r: tree_chain(n, D, r),
         "CTL-MST": lambda n, D, r: tree_mst(n, D, r),
         "CTL-Forest": lambda n, D, r: tree_forest(n, D, r)}


def sweep(scatter, reps, split, etas, rows):
    cfg = CONFIG["Syn-10"]
    for rep in reps:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, Dm, roots = tasks(scatter, rep)
        d = Xtr[n[0]].shape[1]
        trees = {m: BUILD[m](n, Dm, roots["medoid"]) for m in METHODS}
        ncomp = 1 + sum(1 for u, v in trees["CTL-Forest"][1]
                        if Dm[n.index(u), n.index(v)] >
                        np.mean([Dm[n.index(a), n.index(b)]
                                 for a, b in trees["CTL-MST"][1]]))
        for b, eta in itertools.product(B_PER_TASK, etas):
            B = int(round(b * T))
            for m in METHODS:
                root, dir_edges, _ = trees[m]
                alloc = sle_alloc(n, root, dir_edges, Dm, B)
                torch.manual_seed(rep)
                try:
                    models = cascade_train(dir_edges, root, Xtr, ytr, alloc, eta,
                                           cfg["task"], cfg["reg"],
                                           lambda: make_model("linear", d))
                    v = eval_all(models, Xte, yte, "regression", st)
                    if not np.isfinite(v):
                        v = np.nan
                except RuntimeError:
                    v = np.nan
                rows.append(dict(scatter=scatter, b=b, B=B, rep=rep, eta=eta,
                                 method=m, split=split, value=v, ncomp=ncomp))
        print(f"  s={scatter} {split} rep={rep} en {time.perf_counter()-t0:.0f}s",
              flush=True)


def report(df, out_path):
    rows = []
    for (s, b), g in df.groupby(["scatter", "b"]):
        series, chosen = {}, {}
        for m in METHODS:
            t = g[(g.split == "tune") & (g.method == m)]
            best = t.groupby("eta")["value"].mean().dropna()
            if best.empty:
                continue
            chosen[m] = best.idxmin()
            series[m] = (g[(g.split == "eval") & (g.method == m)
                           & (g.eta == chosen[m])].set_index("rep")["value"])
        p = pd.concat(series, axis=1).dropna()
        for m in METHODS:
            if m == "Chain":
                continue
            gap = (p[m] - p["Chain"]) / p["Chain"] * 100
            rows.append(dict(scatter=s, b=b, method=m, gap_pct=gap.mean(),
                             se=gap.std(ddof=1) / np.sqrt(len(gap)),
                             eta=chosen[m], eta_chain=chosen["Chain"], n=len(gap)))
    o = pd.DataFrame(rows)
    o.to_csv(out_path, index=False)
    for m in o.method.unique():
        sub = o[o.method == m]
        print(f"\n=== ({m} - Chain)/Chain en %, negatif = mieux que la chaine ===")
        print(sub.pivot_table(index="scatter", columns="b", values="gap_pct")
                 .round(1).to_string())
        print("--- erreur type")
        print(sub.pivot_table(index="scatter", columns="b", values="se")
                 .round(1).to_string())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-reps", type=int, default=20)
    ap.add_argument("--tune-reps", type=int, default=5)
    ap.add_argument("--etas", default=",".join(str(e) for e in ETAS))
    ap.add_argument("--scatters", default=",".join(str(s) for s in SCATTERS))
    ap.add_argument("--stem", default="tworegime_final")
    a = ap.parse_args()
    set_alloc_mode("floor1")
    etas = [float(x) for x in a.etas.split(",")]

    rows = []
    for scatter in [float(s) for s in a.scatters.split(",")]:
        sweep(scatter, range(TUNE_OFFSET, TUNE_OFFSET + a.tune_reps), "tune",
              etas, rows)
        sweep(scatter, range(a.eval_reps), "eval", etas, rows)
        pd.DataFrame(rows).to_csv(HERE / f"{a.stem}_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    report(df, HERE / f"{a.stem}_summary.csv")
    print("done", flush=True)


if __name__ == "__main__":
    main()
