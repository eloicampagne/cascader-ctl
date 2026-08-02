"""Ce que devient l'avantage de la cascade quand le nombre de taches croit.

Le reviewer Lufy objecte que nos experiences ne sont pas a grande echelle. L'axe
sur lequel CTL passe a l'echelle n'est pas la taille du modele local mais le
nombre de taches T, puisque le budget est partage entre elles. Aucune de nos
experiences ne fait varier T, ce que `scatter_sweep.py` signalait deja comme une
reserve. On le fait ici, T allant de 50 a 1000, sur la famille synthetique a
grappes pour que le cout reste tenable.

Deux regimes de budget, qui repondent a deux questions differentes:

    fixe-total     B constant quand T croit, donc b = B/T decroit. C'est le
                   scenario reel invoque par le reviewer, davantage de taches
                   sous une meme enveloppe de calcul.
    fixe-par-tache B = b0 * T, donc chaque tache garde le meme nombre de pas.
                   Isole l'effet du nombre de taches de celui du budget par tache.

On mesure aussi le temps de construction de l'arbre, seule partie de CTL dont le
cout croit plus vite que lineairement en T.

    python3.12 scale_tasks.py --Ts 50,100,200,500,1000 --reps 5
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

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all, train_steps,
                              cascade_train, sle_alloc, uniform_alloc, syn_tasks,
                              tree_mst)

CFG = CONFIG["Syn-10"]
ETA = MLP_ETA["Syn-10"]
# CTL-MSTc est ecarte: sa construction coute 390 s a T=5000 et croit en T^2.9,
# ce qui rend le balayage impraticable sans rien changer au message.
METHODS = ("Star", "CTL-MST")
SLE = {"CTL-MST"}          # Star recoit l'allocation uniforme, comme dans le papier


def build(m, nodes, D, root, X, y, lam):
    if m == "Star":
        return root, [(root, v) for v in nodes if v != root]
    return tree_mst(nodes, D, root)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", default="50,100,200,500,1000")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--B-total", type=int, default=2000)
    ap.add_argument("--b-per-task", type=int, default=4)
    a = ap.parse_args()

    Ts = [int(t) for t in a.Ts.split(",")]
    rows = []
    for T in Ts:
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, root = syn_tasks(rep, T=T)
            d = Xtr[nodes[0]].shape[1]
            f = lambda: make_model("mlp", d)
            # L'arbre ne depend pas du budget, on le construit une fois par (T, rep).
            trees = {}
            for m in METHODS:
                t0 = time.perf_counter()
                trees[m] = build(m, nodes, D, root, Xtr, ytr, CFG["mstc_lambda"])
                trees[m] = trees[m] + (time.perf_counter() - t0,)
                print(f"    {m} bati en {trees[m][2]:.1f}s", flush=True)
            for regime, B in (("fixe-total", a.B_total),
                              ("fixe-par-tache", a.b_per_task * T)):
                b = max(1, B // T)
                torch.manual_seed(rep)
                models = {v: train_steps(f(), Xtr[v], ytr[v], b, ETA,
                                         CFG["task"], CFG["reg"]) for v in nodes}
                rows.append(dict(T=T, rep=rep, regime=regime, B=B, b=b,
                                 method="Individual",
                                 value=eval_all(models, Xte, yte, CFG["task"], st),
                                 build_s=0.0))
                for m in METHODS:
                    r, edges, bt = trees[m]
                    alloc = (sle_alloc(nodes, r, edges, D, B) if m in SLE else
                             uniform_alloc(nodes, r, edges, B,
                                           CFG["b_proto_frac"]))
                    torch.manual_seed(rep)
                    models = cascade_train(edges, r, Xtr, ytr, alloc, ETA,
                                           CFG["task"], CFG["reg"], f)
                    rows.append(dict(T=T, rep=rep, regime=regime, B=B, b=b,
                                     method=m,
                                     value=eval_all(models, Xte, yte,
                                                    CFG["task"], st),
                                     build_s=bt))
            print(f"  T={T} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "scale_tasks_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "scale_tasks_raw.csv", index=False)

    from scipy import stats as S
    for regime, g0 in df.groupby("regime"):
        print(f"\n=== {regime} ===", flush=True)
        print(g0.pivot_table(index="T", columns="method",
                             values="value").round(3).to_string(), flush=True)
        print("\n  ecart a Individual, appariee", flush=True)
        for T, g in g0.groupby("T"):
            base = g[g.method == "Individual"].sort_values("rep")["value"].values
            out = []
            for m in METHODS:
                cur = g[g.method == m].sort_values("rep")["value"].values
                p = S.ttest_rel(cur, base).pvalue
                out.append(f"{m} {cur.mean():7.3f} "
                           f"{100*(cur-base).mean()/base.mean():+6.1f}% (p={p:.4f})")
            print(f"  T={T:5d} b={g.b.iloc[0]:4d}: Individual {base.mean():8.3f} | "
                  + " | ".join(out), flush=True)

    print("\n=== temps de construction de l'arbre, secondes ===", flush=True)
    print(df[df.method != "Individual"].pivot_table(
        index="T", columns="method", values="build_s").round(3).to_string(),
        flush=True)


if __name__ == "__main__":
    main()
