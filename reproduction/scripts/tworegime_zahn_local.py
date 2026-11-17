"""Variante locale de Zahn contre la coupe globale de CTL-Forest, sur la grille
de tab:tworegime.

Le texte affirme que le critere local de Zahn (1971), qui compare une arete aux
aretes situees a <= 2 sauts de chacune de ses extremites, se comporte comme la
coupe globale mu + 2 sigma. Aucun resultat n'etait conserve pour cette
affirmation ; ce script la mesure au protocole exact de tworegime_final.py
(floor1, taux choisi par methode sur cinq repetitions de reglage disjointes,
vingt repetitions d'evaluation).

    CTL_ALLOC=floor1 python3.12 tworegime_zahn_local.py
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import (CONFIG, cascade_train, eval_all, make_model,
                              sle_alloc, tree_chain, tree_forest, mst_from_distance_matrix)
from tworegime_configs import SCATTERS, T, set_alloc_mode, tasks
from tworegime_final import B_PER_TASK, ETAS, TUNE_OFFSET
from zahn_local import inconsistent_local

HERE = Path(__file__).parent.resolve()
METHODS = ("Chain", "CTL-Forest", "CTL-Forest-local")


def build(n, D, r):
    edges = mst_from_distance_matrix(n, D)
    cut = inconsistent_local(n, edges, D, f=2.0, depth=2)
    return {"Chain": tree_chain(n, D, r),
            "CTL-Forest": tree_forest(n, D, r),
            "CTL-Forest-local": tree_forest(n, D, r, cut=cut)}, len(cut)


def sweep(scatter, reps, split, rows):
    cfg = CONFIG["Syn-10"]
    for rep in reps:
        t0 = time.perf_counter()
        n, Xtr, ytr, Xte, yte, st, Dm, roots = tasks(scatter, rep)
        d = Xtr[n[0]].shape[1]
        trees, ncut_local = build(n, Dm, roots["medoid"])
        for b, eta in itertools.product(B_PER_TASK, ETAS):
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
                    v = v if np.isfinite(v) else np.nan
                except RuntimeError:
                    v = np.nan
                rows.append(dict(scatter=scatter, b=b, B=B, rep=rep, eta=eta,
                                 method=m, split=split, value=v,
                                 ncut_local=ncut_local))
        print(f"  s={scatter} {split} rep={rep} en {time.perf_counter()-t0:.0f}s",
              flush=True)


def main():
    set_alloc_mode("floor1")
    rows = []
    out = HERE / "tworegime_zahn_local_raw.csv"
    for s in SCATTERS:
        sweep(s, range(TUNE_OFFSET, TUNE_OFFSET + 5), "tune", rows)
        sweep(s, range(20), "eval", rows)
        pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    res = []
    for (s, b), g in df.groupby(["scatter", "b"]):
        ser = {}
        for m in METHODS:
            t = g[(g.split == "tune") & (g.method == m)]
            eta = t.groupby("eta")["value"].mean().dropna().idxmin()
            ser[m] = g[(g.split == "eval") & (g.method == m) & (g.eta == eta)].set_index("rep")["value"]
        p = pd.concat(ser, axis=1).dropna()
        for m in METHODS[1:]:
            gap = (p[m] - p["Chain"]) / p["Chain"] * 100
            res.append(dict(scatter=s, b=b, method=m, gap_pct=gap.mean(),
                            se=gap.std(ddof=1) / np.sqrt(len(gap)), n=len(gap)))
    o = pd.DataFrame(res)
    o.to_csv(HERE / "tworegime_zahn_local_summary.csv", index=False)
    print(o.pivot_table(index=["method", "scatter"], columns="b", values="gap_pct").round(1).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
