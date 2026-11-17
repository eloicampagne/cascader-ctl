"""Depth diagnostics: Tables 10-11 and Figures 9-12.

The published versions came from CTL/run_depth_vs_gain.py, CTL/run_path_corr_extended.py
and CTL/run_celeba_depth_expes.py. Those three scripts call cascade_train_tree, which
splits the budget flat after a 10% seed share, and hold a single learning rate
(eta=0.01) for both CTL and Individual. They also run a synthetic geometry that is
not the one the main table reports: d=5 and tau_between=5, against d=3 and
tau_within=tau_between=10 for Syn-10. All three predate the floor-one allocation fix.

This script rebuilds the same quantities under the protocol of the main table:
Syn-10 geometry, SLE allocation in its floor-one regime, and a learning rate
selected per (collection, method) on tuning seeds disjoint from the evaluation
runs. The measured quantity is unchanged: per-task depth in the rooted MST, and
per-task test quality under CTL-MST and under Individual at b = 5.

    CTL_ALLOC=floor1 python3.12 fig_depth_diagnostics.py --stage tune
    CTL_ALLOC=floor1 python3.12 fig_depth_diagnostics.py --stage run
    python3.12 fig_depth_diagnostics.py --stage plot
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import (CONFIG, make_model, eval_task, run_method,
                              syn_tasks, celeba_tasks, tree_mst)
from table2_permethod_lowbudget import etas_for, TUNE_OFFSET
from graph import compute_gradient_distance, select_seed_node

HERE = Path(__file__).resolve().parent
STEM = "fig_depth"

# The published diagnostics ran at B = 5T only. On Syn-10 that budget is past the
# point where it binds (Individual already reaches the least-squares optimum at
# b = 4), so the CTL-versus-Individual comparison is empty there; b = 1 is the
# regime the method targets. Both are measured and reported separately.
B_PER_TASK_LIST = (1, 5)
N_RUNS = 10                          # matches the caption
N_TUNE = 3
B_PLOT = 1                           # budget the figures draw
T_LIST = {"Synth": (300, 500, 1000, 2000, 3000, 4000, 5000),
          "CelebA": (100, 200, 300, 400, 1000)}
T_REF = {"Synth": 1000, "CelebA": 400}          # where the rate is selected
TASK = {"Synth": "regression", "CelebA": "classification"}
CFG_OF = {"Synth": "Syn-10", "CelebA": "CelebA"}
METHODS = ["CTL-MST", "Individual"]
# Each bucket: (min_depth, max_depth, label), as in run_depth_vs_gain.py
BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10**9, "6+")]


def bucket_of(depth):
    for lo, hi, lab in BUCKETS:
        if lo <= depth <= hi:
            return lab
    return "6+"


def build(dataset, T, seed):
    """(nodes, X_tr, y_tr, X_te, y_te, stats, D, root, depths)."""
    if dataset == "Synth":
        nodes, Xtr, ytr, Xte, yte, st, D, root = syn_tasks(seed, T=T)
    else:
        nodes, Xtr, ytr, Xte, yte = celeba_tasks(seed, n_tasks=T)
        st = None
        D = compute_gradient_distance(nodes, Xtr, ytr)
        root = select_seed_node(Xtr, ytr)
    r, dir_edges, _ = tree_mst(nodes, D, root)
    parent = {v: u for u, v in dir_edges}
    depth = {r: 0}
    for v in nodes:                      # walk to the root, memoised
        chain = []
        u = v
        while u not in depth:
            chain.append(u)
            u = parent[u]
        for w in reversed(chain):
            depth[w] = depth[parent[w]] + 1
    return nodes, Xtr, ytr, Xte, yte, st, D, root, depth


def per_node(dataset, T, seed, etas, b_per_task):
    """One run: per-task depth and test quality under CTL-MST and Individual."""
    nodes, Xtr, ytr, Xte, yte, st, D, root, depth = build(dataset, T, seed)
    cfg = CONFIG[CFG_OF[dataset]]
    task = TASK[dataset]
    d = Xtr[nodes[0]].shape[1]
    factory = lambda: make_model("linear", d)
    B = b_per_task * len(nodes)
    out = {}
    for m in METHODS:
        rng = np.random.default_rng(10_000 + seed)
        torch.manual_seed(seed)
        models = run_method(m, nodes, Xtr, ytr, Xtr, ytr, D, root, B,
                            etas[m], cfg, factory, rng)
        out[m] = {v: eval_task(models[v], Xte[v], yte[v], task,
                               st[v] if st else None) for v in nodes}
    rows = []
    for v in nodes:
        q_ctl, q_ind = out["CTL-MST"][v], out["Individual"][v]
        # gain oriented so that positive always means CTL wins
        gain = (q_ind - q_ctl) if task == "regression" else (q_ctl - q_ind)
        rows.append(dict(dataset=dataset, b=b_per_task, T_req=T, T=len(nodes),
                         run=seed, node=v,
                         depth=int(depth[v]), bucket=bucket_of(int(depth[v])),
                         q_ctl=q_ctl, q_indiv=q_ind, gain=gain))
    return rows


def tune(datasets):
    """Rate per (dataset, method), selected at the reference T on tuning seeds."""
    rows = []
    for ds in datasets:
        cfg_name = CFG_OF[ds]
        T = T_REF[ds]
        for b_per_task, eta in itertools.product(B_PER_TASK_LIST,
                                                 etas_for(cfg_name, "linear")):
            for seed in range(TUNE_OFFSET, TUNE_OFFSET + N_TUNE):
                nodes, Xtr, ytr, Xte, yte, st, D, root, _ = build(ds, T, seed)
                cfg = CONFIG[cfg_name]
                d = Xtr[nodes[0]].shape[1]
                factory = lambda: make_model("linear", d)
                B = b_per_task * len(nodes)
                for m in METHODS:
                    rng = np.random.default_rng(10_000 + seed)
                    torch.manual_seed(seed)
                    try:
                        models = run_method(m, nodes, Xtr, ytr, Xtr, ytr, D, root,
                                            B, eta, cfg, factory, rng)
                        v = float(np.mean([eval_task(models[u], Xte[u], yte[u],
                                                     TASK[ds], st[u] if st else None)
                                           for u in nodes]))
                    except RuntimeError:
                        v = np.nan
                    rows.append(dict(dataset=ds, b=b_per_task, eta=float(eta),
                                     seed=seed, method=m, value=v))
            print(f"  {ds} tune b={b_per_task} eta={eta}", flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{STEM}_tune_raw.csv", index=False)


def picked_etas():
    t = pd.read_csv(HERE / f"{STEM}_tune_raw.csv")
    out = {}
    for (ds, b, m), g in t.groupby(["dataset", "b", "method"]):
        s = g.groupby("eta")["value"].mean().dropna()
        out[(ds, int(b), m)] = s.idxmax() if TASK[ds] == "classification" else s.idxmin()
    return out


def run(datasets):
    p = HERE / f"{STEM}_raw.csv"
    prev = pd.read_csv(p) if p.exists() else pd.DataFrame()
    done = (set(zip(prev.dataset, prev.b, prev.T_req, prev.run))
            if len(prev) else set())
    pk = picked_etas()
    rows = []
    for ds in datasets:
        for b_per_task in B_PER_TASK_LIST:
            etas = {m: pk[(ds, b_per_task, m)] for m in METHODS}
            print(f"{ds} b={b_per_task}: eta {etas}", flush=True)
            for T in T_LIST[ds]:
                for seed in range(N_RUNS):
                    if (ds, b_per_task, T, seed) in done:
                        continue
                    t0 = time.perf_counter()
                    rows.extend(per_node(ds, T, seed, etas, b_per_task))
                    pd.concat([prev, pd.DataFrame(rows)],
                              ignore_index=True).to_csv(p, index=False)
                    print(f"  {ds} b={b_per_task} T={T:5d} run={seed} "
                          f"{time.perf_counter()-t0:6.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

def corr_table(df):
    # df is already restricted to one budget
    from scipy.stats import spearmanr
    out = []
    for (ds, T), g in df.groupby(["dataset", "T_req"]):
        rhos, mx, mn = [], [], []
        for run, h in g.groupby("run"):
            if h.depth.nunique() > 1:
                rhos.append(spearmanr(h.depth, h.q_ctl).statistic)
            mx.append(h.depth.max())
            mn.append(h.depth.mean())
        out.append(dict(dataset=ds, T=T,
                        rho=np.mean(rhos), rho_sd=np.std(rhos, ddof=1),
                        max_depth=np.mean(mx), max_depth_sd=np.std(mx, ddof=1),
                        mean_depth=np.mean(mn), mean_depth_sd=np.std(mn, ddof=1)))
    return pd.DataFrame(out).sort_values(["dataset", "T"])


def bucket_table(df):
    out = []
    for (ds, b), g in df.groupby(["dataset", "bucket"]):
        out.append(dict(dataset=ds, bucket=b, n=len(g),
                        gain_mean=g.gain.mean(), gain_sd=g.gain.std(ddof=1),
                        q_ctl=g.q_ctl.mean(), q_indiv=g.q_indiv.mean()))
    order = [b[2] for b in BUCKETS]
    o = pd.DataFrame(out)
    o["k"] = o.bucket.map({b: i for i, b in enumerate(order)})
    return o.sort_values(["dataset", "k"]).drop(columns="k")


def plot(outdir, scatter=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = pd.read_csv(HERE / f"{STEM}_raw.csv")
    # eval_task returns RMSE for regression; Syn-10 is reported in MSE
    # everywhere else in the paper, so convert before tabulating.
    syn = full.dataset == "Synth"
    full.loc[syn, "q_ctl"] = full.loc[syn, "q_ctl"] ** 2
    full.loc[syn, "q_indiv"] = full.loc[syn, "q_indiv"] ** 2
    full.loc[syn, "gain"] = full.loc[syn, "q_indiv"] - full.loc[syn, "q_ctl"]
    outdir = Path(outdir)
    # The figures report the binding regime, b = 1; the tables report both.
    df = full[full.b == B_PLOT]
    ct = pd.concat([corr_table(full[full.b == b]).assign(b=b)
                    for b in sorted(full.b.unique())], ignore_index=True)
    bt = pd.concat([bucket_table(full[full.b == b]).assign(b=b)
                    for b in sorted(full.b.unique())], ignore_index=True)
    ct.to_csv(HERE / f"{STEM}_corr_table.csv", index=False)
    bt.to_csv(HERE / f"{STEM}_bucket_table.csv", index=False)
    ct = ct[ct.b == B_PLOT]

    # --- Figure 9: rho and tree depth against T -----------------------------
    for ds, stem in [("Synth", "correlation_evolution_vs_T_extended"),
                     ("CelebA", "celeba_correlation_vs_T")]:
        g = ct[ct.dataset == ds]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.errorbar(g["T"], g.rho, yerr=g.rho_sd, color="crimson", marker="o",
                    capsize=3, linewidth=1.6, label=r"Spearman $\rho$(depth, $Q$)")
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"Number of tasks $T$")
        ax.set_ylabel(r"$\rho$(depth, $Q$)", color="crimson")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.errorbar(g["T"], g.max_depth, yerr=g.max_depth_sd, color="steelblue",
                     marker="s", linestyle="--", capsize=3, linewidth=1.4,
                     label="max depth")
        ax2.errorbar(g["T"], g.mean_depth, yerr=g.mean_depth_sd, color="forestgreen",
                     marker="^", linestyle=":", capsize=3, linewidth=1.4,
                     label="mean depth")
        ax2.set_ylabel("Tree depth")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.85)
        ax.set_title(f"{ds}: depth-error correlation against scale", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", outdir / f"{stem}.pdf")

    order = [b[2] for b in BUCKETS]
    # --- Figure 10: gain distribution per depth bucket ----------------------
    for ds, stem in [("Synth", "depth_gain_violin"),
                     ("CelebA", "celeba_depth_gain_violin")]:
        g = df[df.dataset == ds]
        if g.empty:
            continue
        labs = [b for b in order if (g.bucket == b).any()]
        # A raw difference has tails two orders of magnitude wider than its
        # body, and a ratio to the independent error diverges on the tasks
        # independent training already solves. This normalisation is bounded in
        # [-1, 1], keeps the sign, and leaves every bucket on one scale.
        g = g.assign(shown=(g.q_indiv - g.q_ctl) / (g.q_indiv + g.q_ctl))
        if ds != "Synth":
            g = g.assign(shown=(g.q_ctl - g.q_indiv) / (g.q_ctl + g.q_indiv))
        ylab = "Normalised CTL gain over Independent"
        data = [g[g.bucket == b].shown.dropna().values for b in labs]
        lo, hi = -1.0, 1.0
        pad = 0.04
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("tab:blue")
            pc.set_alpha(0.55)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylim(min(lo - pad, -pad), hi + pad)
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels(labs)
        ax.set_xlabel("Tree depth bucket")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, axis="y")
        ax.set_title(f"{ds}: CTL gain per depth level", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", outdir / f"{stem}.pdf")

    # --- Figure 11: quality against depth, CTL and Independent --------------
    for ds, stem, ylab in [("Synth", "depth_rmse_vs_depth", "MSE"),
                           ("CelebA", "celeba_depth_acc_vs_depth", "Accuracy")]:
        g = df[df.dataset == ds]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for col, lab, color in [("q_ctl", "CTL-MST", "tab:blue"),
                                ("q_indiv", "Independent", "tab:gray")]:
            s = g.groupby("bucket")[col].agg(["mean", "sem"]).reindex(order).dropna()
            ax.errorbar(range(len(s)), s["mean"], yerr=s["sem"], marker="o",
                        capsize=3, color=color, linewidth=1.6, label=lab)
            ax.set_xticks(range(len(s)))
            ax.set_xticklabels(s.index)
        ax.set_xlabel("Tree depth bucket")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_title(f"{ds}: quality against depth", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", outdir / f"{stem}.pdf")

    # --- Figure 12: per-task scatter, one panel per T -----------------------
    # Dropped from the papers; pass --scatter to regenerate it.
    for ds, stem in ([("Synth", "depth_scatter_by_T"),
                      ("CelebA", "celeba_depth_scatter_by_T")] if scatter else []):
        g = df[df.dataset == ds]
        if g.empty:
            continue
        Ts = sorted(g["T"].unique())
        ncol = min(4, len(Ts))
        nrow = int(np.ceil(len(Ts) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow),
                                 squeeze=False)
        for k, T in enumerate(Ts):
            ax = axes[k // ncol][k % ncol]
            h = g[g["T"] == T]
            ax.scatter(h.depth, h.q_ctl, s=4, alpha=0.25, color="tab:blue",
                       label="CTL-MST")
            ax.scatter(h.depth, h.q_indiv, s=4, alpha=0.25, color="tab:gray",
                       label="Independent")
            ax.set_title(f"$T={T}$", fontsize=9)
            ax.set_xlabel("depth", fontsize=8)
            ax.set_ylabel("MSE" if ds == "Synth" else "accuracy", fontsize=8)
            ax.grid(alpha=0.2)
            if k == 0:
                ax.legend(fontsize=7, markerscale=2.5)
        for k in range(len(Ts), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", outdir / f"{stem}.pdf")

    print(f"\n=== figures drawn at b = {B_PLOT} ===")
    print("\n=== Table 10 (depth-error correlation) ===")
    print(ct.to_string(index=False))
    print("\n=== Table 11 (gain per depth bucket) ===")
    print(bt.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("tune", "run", "plot"))
    ap.add_argument("--datasets", default="Synth,CelebA")
    ap.add_argument("--outdir", default=str(HERE.parent.parent / "figures"))
    ap.add_argument("--scatter", action="store_true",
                    help="also draw the per-task scatter (large file)")
    a = ap.parse_args()
    ds = a.datasets.split(",")
    if a.stage == "tune":
        tune(ds)
    elif a.stage == "run":
        run(ds)
    else:
        plot(a.outdir, a.scatter)
