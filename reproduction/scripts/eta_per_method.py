"""Chaque methode recoit son propre pas d'apprentissage.

Le reviewer Lufy demande si eta a ete regle sur CTL puis applique aux baselines.
La lecture de `CTL/optim.py` confirme que l'objectif Optuna entraine une cascade et
minimise sa perte, donc oui. La seule reponse convaincante est de refaire la
comparaison en accordant a chaque methode son meilleur pas, ce qui retire tout
avantage de reglage a CTL et le met en position defavorable, puisque les baselines
gagnent un degre de liberte qu'elles n'avaient pas.

Balayage log de six valeurs, meilleur eta choisi par methode, par jeu et par budget
sur la moyenne des graines, puis comparaison a ce point.

    python3.12 eta_per_method.py --datasets Syn-10,WEAVE-UK,CIFAR --reps 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, make_model, eval_all, cascade_train,
                              train_steps, sle_alloc, uniform_alloc,
                              tree_chain, tree_mst, tree_mstc)
from mechanism_edge_vs_depth import load

ETAS = (0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
METHODS = ["Individual", "Star", "Chain", "CTL-MST", "CTL-MSTc"]


def build(m, nodes, D, root, X, y, lam):
    if m == "Star":
        return root, [(root, v) for v in nodes if v != root]
    if m == "Chain":
        r, e, _ = tree_chain(nodes, D, root)
        return r, e
    if m == "CTL-MST":
        r, e, _ = tree_mst(nodes, D, root)
        return r, e
    r, e, _ = tree_mstc(nodes, D, root, X, y, lam)
    return r, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--alloc", default="sle", choices=["sle", "uniform"])
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, root, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for B in Bs:
                for eta in ETAS:
                    for m in METHODS:
                        torch.manual_seed(rep)
                        f = lambda: make_model("mlp", d)
                        try:
                            if m == "Individual":
                                b = max(1, B // len(nodes))
                                models = {v: train_steps(f(), Xtr[v], ytr[v], b,
                                                         eta, cfg["task"], cfg["reg"])
                                          for v in nodes}
                            else:
                                r, edges = build(m, nodes, D, root, Xtr, ytr,
                                                 cfg["mstc_lambda"])
                                alloc = (sle_alloc(nodes, r, edges, D, B)
                                         if a.alloc == "sle" else
                                         uniform_alloc(nodes, r, edges, B,
                                                       cfg["b_proto_frac"]))
                                models = cascade_train(edges, r, Xtr, ytr, alloc,
                                                       eta, cfg["task"], cfg["reg"], f)
                            v = eval_all(models, Xte, yte, cfg["task"], st)
                        except RuntimeError:
                            v = np.nan       # divergence, eta trop grand
                        rows.append(dict(dataset=ds, rep=rep, B=B, eta=eta,
                                         method=m, value=v))
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "eta_per_method_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "eta_per_method_raw.csv", index=False)
    higher = {"CIFAR", "FMNIST", "CelebA"}

    print("\n=== meilleur eta par methode, et valeur a ce point ===", flush=True)
    best = []
    for (ds, B, m), g in df.groupby(["dataset", "B", "method"]):
        mean = g.groupby("eta")["value"].mean().dropna()
        if mean.empty:
            continue
        e = mean.idxmax() if ds in higher else mean.idxmin()
        best.append(dict(dataset=ds, B=B, method=m, eta=e, value=mean[e]))
    bdf = pd.DataFrame(best)
    print(bdf.pivot_table(index=["dataset", "B"], columns="method",
                          values="eta").to_string(), flush=True)
    print("\n=== valeur a l'eta propre de chaque methode ===", flush=True)
    print(bdf.pivot_table(index=["dataset", "B"], columns="method",
                          values="value").round(4).to_string(), flush=True)

    print("\n=== cascade contre entrainement independant, chacun a son propre eta ===",
          flush=True)
    for (ds, B), g in bdf.groupby(["dataset", "B"]):
        s_ = g.set_index("method")["value"]
        casc = s_.drop(["Individual", "Star"])
        cb = casc.idxmax() if ds in higher else casc.idxmin()
        gap = casc[cb] - s_["Individual"]
        print(f"{ds:9s} B={B:5d}: Individual {s_['Individual']:.4f} (eta "
              f"{g[g.method=='Individual'].eta.iloc[0]}) | meilleure cascade {cb} "
              f"{casc[cb]:.4f} | {100*gap/abs(s_['Individual']):+.1f}%", flush=True)

    print("\n=== CTL-MSTc contre le meilleur autre, chacun a son propre eta ===",
          flush=True)
    for (ds, B), g in bdf.groupby(["dataset", "B"]):
        s_ = g.set_index("method")["value"]
        base = s_.drop("CTL-MSTc")
        bb = base.idxmax() if ds in higher else base.idxmin()
        gap = s_["CTL-MSTc"] - base[bb]
        win = (gap > 0) if ds in higher else (gap < 0)
        print(f"{ds:9s} B={B:5d}: CTL-MSTc {s_['CTL-MSTc']:.4f} | meilleur autre "
              f"{bb} {base[bb]:.4f} | {100*gap/abs(base[bb]):+.1f}% "
              f"{'CTL gagne' if win else 'CTL perd'}", flush=True)
    bdf.to_csv(HERE / "eta_per_method_best.csv", index=False)


if __name__ == "__main__":
    main()
