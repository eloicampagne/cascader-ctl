"""La mesure de Gan et Isola, faite chez nous, autour de deux points de depart.

Neural Thickets (arXiv 2603.12228) mesure la densite d'experts de tache autour
des poids pre-entraines,

    delta(m) = P[ s(theta + eps) >= s(theta) + m ],  eps gaussien d'echelle sigma,

et montre qu'elle croit fortement avec la taille du modele. Si elle est elevee
autour d'un unique theta partage, alors un seul point de depart suffit a toutes
les taches: c'est Star, et l'arbre ne sert a rien.

C'est exactement testable chez nous, et sans aucun pas de gradient. Pour chaque
tache v on tire N perturbations gaussiennes et on compte la fraction qui ameliore
la perte de v, autour de deux points:

    racine     theta_root, le point que Star donne a tout le monde
    parent     theta_pa(v), le point que l'arbre MST donne a v

Si delta_parent est nettement superieure a delta_racine, le choix du parent porte
quelque chose que la densite seule n'explique pas, et l'objection de Gan et Isola
ne s'applique pas a notre regime. Si les deux se rejoignent, Star suffit et il
faut le dire.

Cout: N passes avant par tache et par point de reference, aucune retropropagation.
C'est l'experience la moins chere qui reponde a la citation du reviewer.

    python3.12 expert_density.py --datasets CIFAR,WEAVE-UK --N 64 --reps 3
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, train_steps,
                              sle_alloc, cascade_train, tree_mst)
from mechanism_edge_vs_depth import load

SIGMAS = (0.005, 0.02, 0.05)
MARGINS = (0.0, 0.02, 0.05)          # amelioration relative exigee


@torch.no_grad()
def loss_of(model, X, y, task):
    xb = torch.from_numpy(X).float()
    yb = torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1)
    out = model(xb).reshape(-1)
    if task == "classification":
        return float(torch.nn.functional.binary_cross_entropy_with_logits(out, yb))
    return float(torch.nn.functional.mse_loss(out, yb))


@torch.no_grad()
def density(model, X, y, task, sigma, N, gen):
    """Fraction des N perturbations qui ameliorent la perte, par marge."""
    base = loss_of(model, X, y, task)
    if not np.isfinite(base) or base <= 0:
        return None, base
    scale = float(torch.cat([p.reshape(-1) for p in model.parameters()]).std())
    hits = np.zeros(len(MARGINS))
    probe = copy.deepcopy(model)
    for _ in range(N):
        for q, p in zip(probe.parameters(), model.parameters()):
            q.copy_(p + sigma * scale * torch.randn(p.shape, generator=gen))
        l = loss_of(probe, X, y, task)
        for i, m in enumerate(MARGINS):
            hits[i] += float(l <= base * (1.0 - m))
    return hits / N, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="CIFAR,WEAVE-UK")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--max-tasks", type=int, default=100)
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        cfg, eta = CONFIG[ds], MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, root, _ = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            f = lambda: make_model("mlp", d)
            r, edges = tree_mst(nodes, D, root)[:2]
            alloc = sle_alloc(nodes, r, edges, D, a.B)
            torch.manual_seed(rep)
            models = cascade_train(edges, r, Xtr, ytr, alloc, eta,
                                   cfg["task"], cfg["reg"], f)
            parent = {v: u for u, v in edges}

            sel = nodes[:a.max_tasks]
            gen = torch.Generator().manual_seed(rep)
            for sigma in SIGMAS:
                for v in sel:
                    if v == r:
                        continue
                    for name, ref in (("racine", models[r]),
                                      ("parent", models[parent[v]])):
                        h, base = density(ref, Xtr[v], ytr[v], cfg["task"],
                                          sigma, a.N, gen)
                        if h is None:
                            continue
                        rows.append(dict(dataset=ds, rep=rep, sigma=sigma,
                                         point=name, task=v, base=base,
                                         **{f"d{m}": h[i]
                                            for i, m in enumerate(MARGINS)}))
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "expert_density_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "expert_density_raw.csv", index=False)
    cols = [f"d{m}" for m in MARGINS]
    print("\n=== fraction des perturbations qui ameliorent, par marge ===",
          flush=True)
    print(df.pivot_table(index=["dataset", "sigma"], columns="point",
                         values=cols).round(3).to_string(), flush=True)

    from scipy import stats as S
    print("\n=== parent contre racine, apparie par tache ===", flush=True)
    for (ds, sg), g in df.groupby(["dataset", "sigma"]):
        w = g.pivot_table(index=["rep", "task"], columns="point", values="d0.0")
        w = w.dropna()
        if w.empty:
            continue
        p = S.ttest_rel(w["parent"], w["racine"]).pvalue
        print(f"{ds:9s} sigma={sg:<6}: parent {w['parent'].mean():.3f} | "
              f"racine {w['racine'].mean():.3f} | "
              f"delta {w['parent'].mean()-w['racine'].mean():+.3f} (p={p:.4f})",
              flush=True)


if __name__ == "__main__":
    main()
