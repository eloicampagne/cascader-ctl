"""Does CTL-MST's bounded depth buy anything as T grows?

A greedy chain often beats the tree constructions. In the convex
regime the paper's depth bound predicts shallow trees should win; the non-convex
significance test shows Chain winning at T<=500. This script asks what happens
as the number of tasks grows, on the paper's own synthetic generator (Syn-10:
d=3, 5 clusters), holding the per-task budget fixed (B = per_task * T).

For each T it records, per method, in the MLP (non-convex) regime:
  - test RMSE (mean over seeds)
  - tree depth (max / median root-to-leaf hop count)  -> the number of SEQUENTIAL
    refinement stages a cascade needs; a chain needs T-1, a bushy tree far fewer.

The accuracy question (does the Chain-MST gap narrow?) and the parallelism
question (how does depth scale?) are answered together.

    python3.12 large_t_chain_vs_mst.py --Ts 200,500,1000,2000 --seeds 8 --pertask 4
"""

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from table1_nonconvex import (syn_tasks, make_model, run_method, CONFIG,
                              eval_all, tree_chain, tree_mst, tree_mstc)

HERE = Path(__file__).parent.resolve()
METHODS = ["Individual", "Chain", "CTL-MST", "CTL-MSTc"]
MLP_ETA_SYN = 0.2   # same value table1_nonconvex uses for Syn-10 MLP, all methods


def depth_stats(dir_edges, root, nodes):
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)
    leaves = [v for v in nodes if not children[v]]
    dl = np.array([depth[v] for v in leaves])
    return int(dl.max()), float(np.median(dl))


def tree_for(method, nodes, D, seed, X, y, lam):
    if method == "Chain":
        return tree_chain(nodes, D, seed)
    if method == "CTL-MST":
        return tree_mst(nodes, D, seed)
    if method == "CTL-MSTc":
        return tree_mstc(nodes, D, seed, X, y, lam)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", type=str, default="200,500,1000,2000")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pertask", type=int, default=4,
                    help="B = pertask * T (holds per-task budget fixed as T grows)")
    args = ap.parse_args()
    Ts = [int(x) for x in args.Ts.split(",")]
    cfg = CONFIG["Syn-10"]
    lam = cfg["mstc_lambda"]

    rows, depth_rows = [], []
    for T in Ts:
        B = args.pertask * T
        t0 = time.time()
        for seed in range(args.seeds):
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats_d, D, root = syn_tasks(seed, T=T)
            d = Xn_tr[nodes[0]].shape[1]
            torch.manual_seed(seed)
            factory = lambda: make_model("mlp", d)
            rng = np.random.default_rng(30_000 + seed)
            for method in METHODS:
                models = run_method(method, nodes, Xn_tr, yn_tr, Xn_tr, yn_tr,
                                    D, root, B, MLP_ETA_SYN, cfg, factory, rng)
                rmse = eval_all(models, Xn_te, y_te, "regression", stats_d)
                rows.append(dict(T=T, B=B, seed=seed, method=method, rmse=rmse))
                if method in ("Chain", "CTL-MST", "CTL-MSTc"):
                    r, edges, _ = tree_for(method, nodes, D, root,
                                           Xn_tr, yn_tr, lam)
                    dmax, dmed = depth_stats(edges, r, nodes)
                    depth_rows.append(dict(T=T, seed=seed, method=method,
                                           depth_max=dmax, depth_med=dmed))
        print(f"T={T} B={B}: {args.seeds} seeds in {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    dd = pd.DataFrame(depth_rows)

    acc = (df.groupby(["T", "method"])["rmse"].agg(["mean", "std"]).reset_index())
    dep = (dd.groupby(["T", "method"])[["depth_max", "depth_med"]].mean()
           .reset_index())
    print("\n=== RMSE (MLP, lower better) ===")
    print(acc.pivot(index="T", columns="method", values="mean")
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n=== tree depth (sequential refinement stages) ===")
    print(dep.pivot(index="T", columns="method", values="depth_max")
          .to_string(float_format=lambda v: f"{v:.0f}"))

    # Chain vs MST gap and its significance per T
    print("\n=== Chain vs CTL-MST: paired gap (RMSE_MST - RMSE_Chain), + = Chain better ===")
    for T in Ts:
        piv = df[df["T"] == T].pivot_table(index="seed", columns="method",
                                           values="rmse")
        if "Chain" in piv and "CTL-MST" in piv:
            sub = piv[["Chain", "CTL-MST"]].dropna()
            gap = (sub["CTL-MST"] - sub["Chain"]).values
            p = stats.ttest_rel(sub["CTL-MST"], sub["Chain"]).pvalue
            print(f"  T={T:5d}: mean gap {gap.mean():+.3f}  "
                  f"(Chain better if +)  p={p:.3g}  n={len(sub)}")

    df.to_csv(HERE / "large_t_chain_vs_mst_raw.csv", index=False)
    acc.to_csv(HERE / "large_t_chain_vs_mst_rmse.csv", index=False)
    dep.to_csv(HERE / "large_t_chain_vs_mst_depth.csv", index=False)
    print("\nwrote large_t_chain_vs_mst_{raw,rmse,depth}.csv")


if __name__ == "__main__":
    main()
