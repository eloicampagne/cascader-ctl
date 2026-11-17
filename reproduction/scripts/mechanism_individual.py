"""Erreur d'Individual par tache, pour controler la difficulte propre de chaque tache
dans la regression de mechanism_edge_vs_depth(_2opt).py.

Meme chargement, memes graines, meme pas MLP et memes budgets que ces scripts.
Sortie : mechanism_individual_raw.csv (dataset, rep, B, node, err_ind).

    python3.12 mechanism_individual.py --datasets WEAVE-UK,CIFAR
"""
import argparse

import numpy as np
import pandas as pd
import torch

from mechanism_edge_vs_depth import HERE, load
from table1_nonconvex import CONFIG, MLP_ETA, eval_task, make_model, run_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="WEAVE-UK,CIFAR")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    rows = []
    for ds in a.datasets.split(","):
        cfg, eta = CONFIG[ds], MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, seed, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for B in Bs:
                torch.manual_seed(rep)
                models = run_method("Individual", nodes, Xtr, ytr, Xtr, ytr, D, seed, B, eta, cfg,
                                    lambda: make_model("mlp", d), np.random.default_rng(rep))
                for v in nodes:
                    rows.append(dict(dataset=ds, rep=rep, B=B, node=str(v),
                                     err_ind=eval_task(models[v], Xte[v], yte[v], cfg["task"],
                                                       st[v] if st else None)))
            print(f"  {ds} rep={rep} fait", flush=True)
            pd.DataFrame(rows).to_csv(HERE / "mechanism_individual_raw.csv", index=False)


if __name__ == "__main__":
    main()
