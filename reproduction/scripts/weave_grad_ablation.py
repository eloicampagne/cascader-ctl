"""
Disentangle learned geometry from SLE allocation on WEAVE-UK, on the PAPER's own
gradient proxy (the core Table 1 claim).

The published Table 1 compares CTL-MST (SLE allocation) against RandTree (uniform
allocation), which confounds tree structure with budget allocation. This runs the
clean 2x2 on the gradient proxy:

    tree structure  in {gradient MST, random spanning tree}
    budget alloc.   in {SLE, uniform}

gradMST+SLE is exactly the published CTL-MST, so it should reproduce the Table 1
number (WEAVE-UK B=2000 linear ~1210), validating the setup. The key cell is
rand+SLE: if the gradient MST beats a random tree under the same SLE allocation,
the learned geometry adds value beyond allocation. Both trees rooted at the same
raw-feature medoid (the Table 1 rooting). Linear and MLP, 20 repetitions.

Run: python3.12 weave_grad_ablation.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    CONFIG, MLP_ETA, make_model, eval_all, cascade_train,
    tree_mst, tree_random, sle_alloc, uniform_alloc, weave_tasks, N_REPS_WEAVE,
)

Bs = (500, 1000, 2000)
N_REPS = N_REPS_WEAVE
VARIANTS = [("gradmst", "sle"), ("gradmst", "unif"), ("rand", "sle"), ("rand", "unif")]


def run_variant(structure, alloc_kind, nodes, X_tr, y_tr, D, root, B, eta, cfg, factory, rng):
    if structure == "gradmst":
        r, dir_edges, _ = tree_mst(nodes, D, root)
    else:
        r, dir_edges, _ = tree_random(nodes, D, root, rng)
    if alloc_kind == "sle":
        alloc = sle_alloc(nodes, r, dir_edges, D, B)
    else:
        alloc = uniform_alloc(nodes, r, dir_edges, B, cfg["b_proto_frac"])
    return cascade_train(dir_edges, r, X_tr, y_tr, alloc, eta, cfg["task"], cfg["reg"], factory)


def main():
    cfg = CONFIG["WEAVE-UK"]
    rows = []
    for rep in range(N_REPS):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, seed = weave_tasks()
        d = Xn_tr[nodes[0]].shape[1]
        for mc in ("linear", "mlp"):
            eta = cfg["eta"] if mc == "linear" else MLP_ETA["WEAVE-UK"]
            factory = lambda: make_model(mc, d)
            for B in Bs:
                for vi, (struct, alloc_kind) in enumerate(VARIANTS):
                    torch.manual_seed(1000 * rep + vi)
                    rng = np.random.default_rng(60_000 + rep)   # same random tree per (rep,B)
                    models = run_variant(struct, alloc_kind, nodes, Xn_tr, yn_tr,
                                         D_grad, seed, B, eta, cfg, factory, rng)
                    rows.append(dict(model=mc, B=B, seed=rep,
                                     variant=f"{struct}+{alloc_kind}",
                                     value=eval_all(models, Xn_te, y_te, "regression", stats)))
            done = [r for r in rows if r["seed"] == rep and r["model"] == mc]
            print(f"  rep={rep} {mc}: " + " ".join(
                f"{r['variant']}@{r['B']}={r['value']:.0f}" for r in done), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "weave_grad_ablation_raw.csv", index=False)
    summ = (df.groupby(["model", "B", "variant"])["value"]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(HERE / "weave_grad_ablation_summary.csv", index=False)
    print("\n=== 2x2 gradient proxy: structure (gradmst/rand) x allocation (sle/unif) ===", flush=True)
    print(summ.pivot_table(index=["model", "B"], columns="variant", values="mean").round(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
