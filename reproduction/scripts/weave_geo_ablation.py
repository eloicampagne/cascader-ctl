"""
Disentangle geographic structure from SLE allocation on WEAVE-UK (Lufy W4).

Is the geography tree strong because physical proximity is an informative task
prior, or merely because it inherits the SLE budget allocation that any tree gets?
A 2x2 design separates the two factors:

    tree structure  in {geographic MST, random spanning tree}
    budget alloc.   in {SLE, uniform}

If performance follows the structure row (geographic << random), the spatial prior
is the driver. If it follows the allocation column (SLE << uniform), the allocation
is. Both trees are rooted at the same geographic medoid, and the SLE variants use
the same geographic edge distances, so the only thing that changes across the row
is whether the tree edges track geography.

Everything else matches the faithful Table 1 protocol of table1_nonconvex.py.
Run: python3.12 weave_geo_ablation.py
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
from weave_geography import geo_distance_matrix

Bs = (500, 1000, 2000)
N_REPS = N_REPS_WEAVE


def run_variant(structure, alloc_kind, nodes, X_tr, y_tr, D_geo, geo_root,
                B, eta, cfg, factory, rng):
    if structure == "geo":
        root, dir_edges, _ = tree_mst(nodes, D_geo, geo_root)
    else:                                        # random spanning tree
        root, dir_edges, _ = tree_random(nodes, D_geo, geo_root, rng)
    if alloc_kind == "sle":
        alloc = sle_alloc(nodes, root, dir_edges, D_geo, B)
    else:                                        # uniform b_seed
        alloc = uniform_alloc(nodes, root, dir_edges, B, cfg["b_proto_frac"])
    return cascade_train(dir_edges, root, X_tr, y_tr, alloc, eta,
                         cfg["task"], cfg["reg"], factory)


VARIANTS = [("geo", "sle"), ("geo", "unif"), ("rand", "sle"), ("rand", "unif")]


def main():
    cfg = CONFIG["WEAVE-UK"]
    rows = []
    for rep in range(N_REPS):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, seed = weave_tasks()
        d = Xn_tr[nodes[0]].shape[1]
        D_geo, geo_root, kept = geo_distance_matrix(nodes)
        sub = lambda dct: {v: dct[v] for v in kept}
        Xk_tr, yk_tr, Xk_te, yk_te, statsk = sub(Xn_tr), sub(yn_tr), sub(Xn_te), sub(y_te), sub(stats)
        # align D_geo to kept order (geo_distance_matrix already returns kept-aligned D)
        if rep == 0:
            print(f"nodes with coords: {len(kept)}/{len(nodes)}", flush=True)
        for mc in ("linear", "mlp"):
            eta = cfg["eta"] if mc == "linear" else MLP_ETA["WEAVE-UK"]
            factory = lambda: make_model(mc, d)
            for B in Bs:
                for vi, (struct, alloc_kind) in enumerate(VARIANTS):
                    torch.manual_seed(1000 * rep + vi)
                    rng = np.random.default_rng(50_000 + rep)   # same random tree per (rep,B)
                    models = run_variant(struct, alloc_kind, kept, Xk_tr, yk_tr,
                                         D_geo, geo_root, B, eta, cfg, factory, rng)
                    rows.append(dict(model=mc, B=B, seed=rep,
                                     variant=f"{struct}+{alloc_kind}",
                                     value=eval_all(models, Xk_te, yk_te, "regression", statsk)))
            done = [r for r in rows if r["seed"] == rep and r["model"] == mc]
            print(f"  rep={rep} {mc}: " + " ".join(
                f"{r['variant']}@{r['B']}={r['value']:.0f}" for r in done), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "weave_geo_ablation_raw.csv", index=False)
    summ = (df.groupby(["model", "B", "variant"])["value"]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(HERE / "weave_geo_ablation_summary.csv", index=False)
    print("\n=== 2x2: structure (geo/rand) x allocation (sle/unif), mean RMSE ===", flush=True)
    print(summ.pivot_table(index=["model", "B"], columns="variant", values="mean").round(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
