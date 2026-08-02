"""L'hybride, sur les collections réelles et sur la grille d'écartement.

CTL-MSTq applique les trois corrections que le surrogate impose et supprime le
gain de portée, qu'il ne contient pas. Il répare la défaillance du synthétique et
perd sur WEAVE-UK, où ce terme absent de la théorie gagne pourtant sa place.
L'hybride garde les trois corrections et réintroduit le gain,

    J(u,v) = delta(u,v) (1 + q n_v) - lam * gain(v),

la question étant si un seul score peut tenir les deux familles à la fois.

lam est balayé sur une grille large, le gain valant trois à cinq fois le terme de
localité en médiane, si bien que lam autour de 0.25 égalise les deux et que la
grille doit s'étendre de part et d'autre. Une grille trop étroite ferait choisir
une borne, erreur déjà rencontrée ailleurs dans ce dossier.

    python3.12 hybrid_eval.py real
    python3.12 hybrid_eval.py scatter
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all,
                              cascade_train, train_steps, sle_alloc,
                              uniform_alloc, tree_chain, tree_mst, tree_mstc)
from mechanism_edge_vs_depth import load as load_real
from mstq import mstq_edges
import scatter_sweep as SS

Q = 0.9
LAMS = (0.0, 0.15, 0.5, 1.5, 5.0, 15.0, 50.0)
BASE = ["Chain", "CTL-MST", "CTL-MSTc"]
METHODS = BASE + [f"MSTq{Q}-l{l}" for l in LAMS]


def build(method, nodes, D, root, X, y, lam_mstc):
    if method == "Chain":
        return tree_chain(nodes, D, root)
    if method == "CTL-MST":
        return tree_mst(nodes, D, root)
    if method == "CTL-MSTc":
        return tree_mstc(nodes, D, root, X, y, lam_mstc)
    lam = float(method.split("-l")[1])
    e = mstq_edges(nodes, D, root, q=Q, lam=lam)
    return root, e, {v: u for u, v in e}


def one(method, nodes, X, y, D, root, B, eta, factory, task, reg, lam_mstc,
        frac, alloc_kind):
    r, edges, _ = build(method, nodes, D, root, X, y, lam_mstc)
    alloc = (sle_alloc(nodes, r, edges, D, B) if alloc_kind == "sle"
             else uniform_alloc(nodes, r, edges, B, frac))
    return cascade_train(edges, r, X, y, alloc, eta, task, reg, factory)


def run_real(a):
    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        eta = MLP_ETA[ds]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, root, Bs = load_real(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for B in Bs:
                for m in METHODS:
                    torch.manual_seed(rep)
                    models = one(m, nodes, Xtr, ytr, D, root, B, eta,
                                 lambda: make_model("mlp", d), cfg["task"],
                                 cfg["reg"], cfg["mstc_lambda"],
                                 cfg["b_proto_frac"], a.alloc)
                    rows.append(dict(dataset=ds, rep=rep, B=B, method=m,
                                     value=eval_all(models, Xte, yte,
                                                    cfg["task"], st)))
            print(f"  {ds} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "hybrid_real_raw.csv", index=False)
    return pd.DataFrame(rows), ["dataset", "B"]


def run_scatter(a):
    rows = []
    scatters = [float(s) for s in a.scatters.split(",")]
    budgets = [int(b) for b in a.budgets.split(",")]
    for arm in a.arms.split(","):
        gen, mc = arm.split("/")
        mc = "linear" if mc == "lin" else mc
        for s in scatters:
            for rep in range(a.reps):
                nodes, X, y, Xte, yte, st, D, root = SS.build_tasks(gen, s, rep)
                d = X[nodes[0]].shape[1]
                for B in budgets:
                    for m in METHODS:
                        torch.manual_seed(rep)
                        models = one(m, nodes, X, y, D, root, B, a.eta,
                                     lambda: make_model(mc, d), "regression",
                                     0.0, SS.MSTC_LAMBDA, 0.10, a.alloc)
                        rows.append(dict(dataset=f"{arm}|s={s}", rep=rep, B=B,
                                         method=m,
                                         value=eval_all(models, Xte, yte,
                                                        "regression", st)))
                print(f"  {arm} s={s} rep={rep} fait", flush=True)
        pd.DataFrame(rows).to_csv(HERE / "hybrid_scatter_raw.csv", index=False)
    return pd.DataFrame(rows), ["dataset", "B"]


def report(df, keys, tag):
    higher = {"CIFAR", "FMNIST", "CelebA"}
    from scipy import stats as st_
    print(f"\n=== {tag}, valeur moyenne ===", flush=True)
    print(df.pivot_table(index=keys, columns="method",
                         values="value")[METHODS].round(4).to_string(), flush=True)
    print(f"\n=== {tag}, écart à Chain, apparié, * = mieux et p<0.05 ===",
          flush=True)
    for key, g in df.groupby(keys):
        ds = key[0] if isinstance(key, tuple) else key
        base = g[g.method == "Chain"].sort_values("rep")["value"].values
        out = []
        for m in METHODS[1:]:
            cur = g[g.method == m].sort_values("rep")["value"].values
            if len(cur) != len(base):
                continue
            diff = cur - base
            p = st_.ttest_rel(cur, base).pvalue
            better = (diff.mean() > 0) if ds in higher else (diff.mean() < 0)
            out.append(f"{m} {100*diff.mean()/abs(base.mean()):+.1f}%"
                       f"{'*' if p < 0.05 and better else ''}")
        print(f"{str(key):26s}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "scatter"])
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,CIFAR")
    ap.add_argument("--arms", default="lin/lin,lin/mlp,relu/mlp")
    ap.add_argument("--scatters", default="1.0,5.0,20.0")
    ap.add_argument("--budgets", default="200,500,2000")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--alloc", default="sle", choices=["sle", "uniform"])
    a = ap.parse_args()
    df, keys = (run_real(a) if a.mode == "real" else run_scatter(a))
    df.to_csv(HERE / f"hybrid_{a.mode}_raw.csv", index=False)
    report(df, keys, a.mode)
