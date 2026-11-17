"""Figure 13: CTL gain over Independent across a grid of task heterogeneity.

The published version came from CTL/run_robustness_sweep.py, which allocates the
budget flat after a 10% seed share, holds eta=0.01 for both methods, runs d=5
rather than the d=3 of Syn-10, and averages 5 runs where the caption announces 10.

Here the grid is unchanged, tau_within in {2,5,10,20} against tau_between in
{2,5,10} at T=1000 and b=5, and the rest follows the main table: Syn-10 geometry
(d=3, sigma=5), SLE allocation in its floor-one regime, a rate selected per
(cell, method) on tuning seeds disjoint from the evaluation runs, and 10 runs.

    CTL_ALLOC=floor1 python3.12 fig_robustness_sweep.py --stage run
    python3.12 fig_robustness_sweep.py --stage plot
"""
import argparse
import os
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table1_nonconvex import CONFIG, make_model, eval_all, run_method, syn_tasks
from table2_permethod_lowbudget import etas_for, TUNE_OFFSET

HERE = Path(__file__).resolve().parent
STEM = "fig_robust"

T = 1000
# The published sweep ran at B = 5T. On the Syn-10 generator that budget is past
# the point where it binds, Individual reaching the least-squares optimum, so the
# comparison is empty there; b = 1 is the regime the method targets. Both are run,
# and the figure draws the binding one.
B_PER_TASK_LIST = (1, 5)
B_PLOT = 1
N_RUNS = 10                       # matches the caption
N_TUNE = 3
TAU_WITHIN = (2.0, 5.0, 10.0, 20.0)
TAU_BETWEEN = (2.0, 5.0, 10.0)
METHODS = ["CTL-MST", "Individual"]
D_SYN, SIGMA, K = 3, 5.0, 5       # the Syn-10 generator
if os.environ.get("CTL_SYN") == "subspace":   # the Syn generator, (3, 10) being Syn itself
    TAU_WITHIN = (1.0, 3.0, 5.0, 10.0)
    TAU_BETWEEN = (3.0, 10.0, 30.0)
    D_SYN = 40


def one(tw, tb, seed, etas, b_per_task):
    nodes, Xtr, ytr, Xte, yte, st, D, root = syn_tasks(
        seed, T=T, d=D_SYN, K=K, tau=tw, tau_b=tb, sigma=SIGMA)
    cfg = CONFIG["Syn-10"]
    d = Xtr[nodes[0]].shape[1]
    factory = lambda: make_model("linear", d)
    B = b_per_task * len(nodes)
    out = {}
    for m in METHODS:
        rng = np.random.default_rng(10_000 + seed)
        torch.manual_seed(seed)
        models = run_method(m, nodes, Xtr, ytr, Xtr, ytr, D, root, B,
                            etas[m], cfg, factory, rng)
        out[m] = eval_all(models, Xte, yte, "regression", st) ** 2   # MSE
    return out


def run(stage):
    p = HERE / f"{STEM}_{stage}_raw.csv"
    prev = pd.read_csv(p) if p.exists() else pd.DataFrame()
    done = (set(zip(prev.b, prev.tau_within, prev.tau_between, prev.run,
                    prev.eta, prev.method)) if len(prev) else set())
    rows = []
    grid = etas_for("Syn-10", "linear")
    seeds = (range(TUNE_OFFSET, TUNE_OFFSET + N_TUNE) if stage == "tune"
             else range(N_RUNS))
    picked = picked_etas() if stage == "eval" else None
    for b_per_task, (tw, tb) in itertools.product(
            B_PER_TASK_LIST, itertools.product(TAU_WITHIN, TAU_BETWEEN)):
        t0 = time.perf_counter()
        for seed in seeds:
            if stage == "tune":
                for eta in grid:
                    if all((b_per_task, tw, tb, seed, float(eta), m) in done
                           for m in METHODS):
                        continue
                    v = one(tw, tb, seed, {m: eta for m in METHODS}, b_per_task)
                    for m in METHODS:
                        rows.append(dict(b=b_per_task, tau_within=tw, tau_between=tb,
                                         run=seed, eta=float(eta), method=m, mse=v[m]))
            else:
                etas = {m: picked[(b_per_task, tw, tb, m)] for m in METHODS}
                if all((b_per_task, tw, tb, seed, float(etas[m]), m) in done
                       for m in METHODS):
                    continue
                v = one(tw, tb, seed, etas, b_per_task)
                for m in METHODS:
                    rows.append(dict(b=b_per_task, tau_within=tw, tau_between=tb,
                                     run=seed, eta=float(etas[m]), method=m, mse=v[m]))
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
        print(f"  b={b_per_task} tau_w={tw:5.1f} tau_b={tb:5.1f} {stage} "
              f"{time.perf_counter()-t0:6.0f}s", flush=True)


def picked_etas():
    t = pd.read_csv(HERE / f"{STEM}_tune_raw.csv")
    out = {}
    for (b, tw, tb, m), g in t.groupby(["b", "tau_within", "tau_between", "method"]):
        out[(int(b), tw, tb, m)] = g.groupby("eta")["mse"].mean().idxmin()
    return out


def summarise():
    ev = pd.read_csv(HERE / f"{STEM}_eval_raw.csv")
    ev = ev[ev.b == B_PLOT]
    w = ev.pivot_table(index=["tau_within", "tau_between", "run"],
                       columns="method", values="mse").reset_index()
    # the paper reports task-averaged RMSE: score each run on its RMSE
    for m in ("Individual", "CTL-MST"):
        w[m] = np.sqrt(w[m])
    # gain as a percentage of the Independent error, positive = CTL wins
    w["gain"] = 100.0 * (w["Individual"] - w["CTL-MST"]) / w["Individual"]
    g = w.groupby(["tau_within", "tau_between"])["gain"]
    s = g.agg(["mean", "sem", "count"]).reset_index()
    s.to_csv(HERE / f"{STEM}_summary.csv", index=False)
    return s


def plot(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summarise()
    M = s.pivot(index="tau_within", columns="tau_between", values="mean") \
         .reindex(index=TAU_WITHIN, columns=TAU_BETWEEN)
    E = s.pivot(index="tau_within", columns="tau_between", values="sem") \
         .reindex(index=TAU_WITHIN, columns=TAU_BETWEEN)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # A symmetric diverging scale wastes half its range when every cell is a
    # gain, and the greens then differ too little to read. Anchor at zero and go
    # sequential while nothing is negative; keep the diverging scale otherwise.
    lo, hi = np.nanmin(M.values), np.nanmax(M.values)
    if lo >= 0:
        im = ax1.imshow(M.values, cmap="Greens", origin="lower", vmin=0, vmax=hi)
    else:
        m = max(abs(lo), abs(hi))
        im = ax1.imshow(M.values, cmap="RdYlGn", origin="lower", vmin=-m, vmax=m)
    ax1.set_xticks(range(len(TAU_BETWEEN)))
    ax1.set_xticklabels([f"{v:.0f}" for v in TAU_BETWEEN])
    ax1.set_yticks(range(len(TAU_WITHIN)))
    ax1.set_yticklabels([f"{v:.0f}" for v in TAU_WITHIN])
    ax1.set_xlabel(r"$\tau_{\mathrm{between}}$")
    ax1.set_ylabel(r"$\tau_{\mathrm{within}}$")
    ax1.set_title("CTL gain over Independent (%)", fontsize=11)
    for i, j in itertools.product(range(len(TAU_WITHIN)), range(len(TAU_BETWEEN))):
        v, e = M.values[i, j], E.values[i, j]
        if np.isfinite(v):
            shade = (v - min(0.0, lo)) / max(1e-9, hi - min(0.0, lo))
            ax1.text(j, i, f"{v:.1f}\n$\\pm${e:.1f}", ha="center", va="center",
                     fontsize=8, color="white" if shade > 0.6 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    for tb in TAU_BETWEEN:
        g = s[s.tau_between == tb].sort_values("tau_within")
        ax2.errorbar(g.tau_within, g["mean"], yerr=g["sem"], marker="o",
                     capsize=3, linewidth=1.6, label=rf"$\tau_{{between}}={tb:.0f}$")
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.set_xlabel(r"$\tau_{\mathrm{within}}$")
    ax2.set_ylabel("CTL gain over Independent (%)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9)
    ax2.set_title("Gain against within-cluster spread", fontsize=11)
    fig.tight_layout()
    out = Path(outdir) / "robustness_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print(s.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("tune", "eval", "plot"))
    ap.add_argument("--outdir", default=str(HERE.parent.parent / "figures"))
    a = ap.parse_args()
    if a.stage == "plot":
        plot(a.outdir)
    else:
        run(a.stage)
