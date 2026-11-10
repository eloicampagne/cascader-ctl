"""
Depth sweep of the local model.

Question: how does the CTL advantage evolve as the local model moves from a
plain linear map (0 hidden layers, the published Table 1 regime) to a deeper,
strictly non-convex MLP (1, 2, 3 hidden layers)?

Everything except the local model class is held at the faithful Table 1 protocol
of table1_nonconvex.py (gradient distance, SLE allocation on the tree methods,
uniform b_seed for Star, medoid rooting, per-dataset config). The single moving
part is network depth. The learning rate is RE-CALIBRATED per (dataset, depth)
on the Individual baseline, over a small grid, so a deeper network is never
handicapped by a learning rate tuned for a shallower one. This isolates the
effect of depth on the CTL-vs-baseline comparison.

Datasets: WEAVE-UK and Syn-10, the two strongest end-to-end regression settings.
Methods : Individual, Star, CTL-MST, CTL-MSTc.
Depths  : 0 (linear), 1, 2, 3 hidden layers of width 32.

Run: python3.12 mlp_depth.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    CONFIG, train_steps, eval_all, run_method, METHODS,
    weave_tasks, syn_tasks,
)

DEPTHS = [0, 1, 2, 3]           # number of hidden layers (0 == linear)
WIDTH = 32
Bs = [500, 2000]
DEPTH_METHODS = ["Individual", "Star", "CTL-MST", "CTL-MSTc"]
ETA_GRID = [0.05, 0.1, 0.2, 0.5, 1.0]
N_REPS = 10


class VarMLP(nn.Module):
    """d -> (Linear(width) -> ReLU) * n_hidden -> Linear(1). n_hidden=0 is linear."""
    def __init__(self, d, width=WIDTH, n_hidden=1):
        super().__init__()
        layers, prev = [], d
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).reshape(-1)


def factory_for(d, depth):
    return lambda: VarMLP(d, WIDTH, depth)


def calibrate(dataset, loader):
    """Best Individual-baseline eta per depth, at B=1000, single seed."""
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, seed = loader(0)
    d = Xn_tr[nodes[0]].shape[1]
    B = 1000
    best = {}
    for depth in DEPTHS:
        b = max(1, B // len(nodes))
        scores = {}
        for eta in ETA_GRID:
            torch.manual_seed(0)
            models = {v: train_steps(VarMLP(d, WIDTH, depth), Xn_tr[v], yn_tr[v],
                                     b, eta, "regression", 0.0) for v in nodes}
            scores[eta] = eval_all(models, Xn_te, y_te, "regression", stats)
        best[depth] = min(scores, key=scores.get)
        print(f"  {dataset} depth={depth}: eta*={best[depth]} "
              f"({', '.join(f'{e}:{s:.0f}' for e, s in scores.items())})", flush=True)
    return best


def run_dataset(dataset, loader, n_reps=N_REPS):
    print(f"=== {dataset}: calibrating eta per depth ===", flush=True)
    eta_star = calibrate(dataset, loader)
    cfg = CONFIG[dataset]
    rows = []
    for rep in range(n_reps):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, seed = loader(rep)
        d = Xn_tr[nodes[0]].shape[1]
        rng = np.random.default_rng(30_000 + rep)
        for depth in DEPTHS:
            eta = eta_star[depth]
            factory = factory_for(d, depth)
            for B in Bs:
                for method in DEPTH_METHODS:
                    torch.manual_seed(1000 * rep + depth)
                    models = run_method(method, nodes, Xn_tr, yn_tr, Xn_tr, yn_tr,
                                        D, seed, B, eta, cfg, factory, rng)
                    val = eval_all(models, Xn_te, y_te, "regression", stats)
                    rows.append(dict(dataset=dataset, depth=depth, eta=eta,
                                     B=B, seed=rep, method=method, value=val))
            print(f"  {dataset} rep={rep} depth={depth}: " + " ".join(
                f"{r['method']}@{r['B']}={r['value']:.0f}"
                for r in rows if r["seed"] == rep and r["depth"] == depth), flush=True)
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "mlp_depth_raw.csv", index=False)
    summ = (df.groupby(["dataset", "depth", "B", "method"])["value"]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(HERE / "mlp_depth_summary.csv", index=False)

    # CTL-MSTc gain over Individual, per (dataset, depth, B)
    piv = summ.pivot_table(index=["dataset", "depth", "B"], columns="method",
                           values="mean")
    gains = []
    for (ds, depth, B), r in piv.iterrows():
        indiv, mstc, mst = r["Individual"], r["CTL-MSTc"], r["CTL-MST"]
        gains.append(dict(dataset=ds, depth=depth, B=B,
                          Individual=indiv, Star=r["Star"], CTL_MST=mst,
                          CTL_MSTc=mstc,
                          gain_mstc_pct=100.0 * (indiv - mstc) / indiv))
    gdf = pd.DataFrame(gains)
    gdf.to_csv(HERE / "mlp_depth_gain.csv", index=False)
    print("\n=== CTL-MSTc gain over Individual by depth ===", flush=True)
    print(gdf.to_string(index=False), flush=True)


if __name__ == "__main__":
    all_rows = []
    all_rows += run_dataset("WEAVE-UK", lambda r: weave_tasks())
    all_rows += run_dataset("Syn-10", lambda r: syn_tasks(r))
    summarize(all_rows)
    print("=== Done ===", flush=True)
