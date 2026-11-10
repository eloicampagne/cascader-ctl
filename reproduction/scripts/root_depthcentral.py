"""The non-convex table again, rooted at the depth-central node.

The published protocol roots every cascade at the medoid, the task minimising
total distance to the others, which is computable before the tree exists but
ignores the topology. The depth bound makes the root-to-leaf path the governing
quantity, so the natural alternative is the tree centre, the node minimising
eccentricity and therefore the depth the bound penalises. This is the two-pass
rule: build the undirected tree once with the medoid as
construction seed, then re-root at its centre by relabelling, with no rebuild.

The rule leaves Star untouched, its centre being the hub, and halves the depth
of Chain, whose centre is the midpoint of the path. Individual has no root. Both
rootings run inside the same loop on the same seeds and the same task draws, so
the comparison carries no cross-run drift.

    python3.12 root_depthcentral.py                 # all five, MLP only
    python3.12 root_depthcentral.py --datasets WEAVE-UK --reps 5
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, METHODS, SLE_METHODS,
                              make_model, eval_all, cascade_train, train_steps,
                              sle_alloc, uniform_alloc, tree_random, tree_chain,
                              tree_knn, tree_mst, tree_mstc,
                              weave_tasks, image_tasks, syn_tasks,
                              celeba_tasks)
from graph import compute_gradient_distance, root_tree, select_seed_node

REG_SETS = {"Syn-10", "WEAVE-UK"}
Bs_REG = (500, 1000, 2000)
Bs_CLS = (500, 2000)
Bs_CELEBA = (2000, 4000)


def tree_center(nodes, undirected, D):
    """Centre of a tree by leaf peeling, ties broken by total distance.

    The peeling leaves one or two nodes, the classical characterisation of the
    tree centre, and a two-node centre is resolved by the medoid criterion so the
    rule stays deterministic.
    """
    if len(nodes) <= 2:
        return nodes[0]
    adj = defaultdict(set)
    for u, v in undirected:
        adj[u].add(v)
        adj[v].add(u)
    deg = {v: len(adj[v]) for v in nodes}
    remaining = set(nodes)
    leaves = [v for v in nodes if deg[v] <= 1]
    while len(remaining) > 2:
        nxt = []
        for v in leaves:
            remaining.discard(v)
            for w in adj[v]:
                if w in remaining:
                    deg[w] -= 1
                    if deg[w] == 1:
                        nxt.append(w)
        leaves = nxt
        if not leaves:
            break
    cand = sorted(remaining)
    if len(cand) == 1:
        return cand[0]
    nidx = {v: i for i, v in enumerate(nodes)}
    return min(cand, key=lambda v: D[nidx[v]].sum())


def build(method, nodes, D, seed, X_prox, y_prox, lam, rng):
    if method == "Star":
        return seed, [(seed, v) for v in nodes if v != seed]
    if method == "RandTree":
        r, e, _ = tree_random(nodes, D, seed, rng)
    elif method == "Chain":
        r, e, _ = tree_chain(nodes, D, seed)
    elif method == "CTL-kNN":
        r, e, _ = tree_knn(nodes, D, seed)
    elif method == "CTL-MST":
        r, e, _ = tree_mst(nodes, D, seed)
    elif method == "CTL-MSTc":
        r, e, _ = tree_mstc(nodes, D, seed, X_prox, y_prox, lam)
    else:
        raise ValueError(method)
    return r, e


def depth_of(nodes, root, dir_edges):
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    dep, stack, best = {root: 0}, [root], 0
    while stack:
        u = stack.pop()
        for v in children[u]:
            dep[v] = dep[u] + 1
            best = max(best, dep[v])
            stack.append(v)
    return best


def run_one(method, nodes, X_tr, y_tr, X_prox, y_prox, D, seed, B, eta, cfg,
            factory, rng, root_mode):
    T = len(nodes)
    if method == "Individual":
        b = max(1, B // T)
        return {v: train_steps(factory(), X_tr[v], y_tr[v], b, eta,
                               cfg["task"], cfg["reg"]) for v in nodes}, 0
    root, dir_edges = build(method, nodes, D, seed, X_prox, y_prox,
                            cfg["mstc_lambda"], rng)
    if root_mode == "depth-central":
        c = tree_center(nodes, dir_edges, D)
        root, dir_edges, _ = root_tree([(u, v) for u, v in dir_edges], c)
    if method in SLE_METHODS:
        alloc = sle_alloc(nodes, root, dir_edges, D, B)
    else:
        alloc = uniform_alloc(nodes, root, dir_edges, B, cfg["b_proto_frac"])
    models = cascade_train(dir_edges, root, X_tr, y_tr, alloc, eta,
                           cfg["task"], cfg["reg"], factory)
    return models, depth_of(nodes, root, dir_edges)


def dataset_iter(ds, rep):
    """Return nodes, train, test, stats, D, medoid seed, budgets, task type."""
    if ds == "WEAVE-UK":
        nodes, Xtr, ytr, Xte, yte, stats, D, seed = weave_tasks()
        return nodes, Xtr, ytr, Xte, yte, stats, D, seed, Bs_REG
    if ds == "Syn-10":
        nodes, Xtr, ytr, Xte, yte, stats, D, seed = syn_tasks(rep)
        return nodes, Xtr, ytr, Xte, yte, stats, D, seed, Bs_REG
    if ds == "CelebA":
        nodes, Xtr, ytr, Xte, yte = celeba_tasks(rep)
        Bs = Bs_CELEBA
    else:
        nodes, Xtr, ytr, Xte, yte = image_tasks(ds, seed=rep)
        Bs = Bs_CLS
    D = compute_gradient_distance(nodes, Xtr, ytr)
    seed = select_seed_node(Xtr, ytr)
    return nodes, Xtr, ytr, Xte, yte, None, D, seed, Bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Syn-10,WEAVE-UK,FMNIST,CIFAR,CelebA")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--model", default="mlp", choices=["mlp", "linear"])
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        task = cfg["task"]
        eta = MLP_ETA[ds] if a.model == "mlp" else cfg["eta"]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, stats, D, seed, Bs = dataset_iter(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            for B in Bs:
                for mode in ("medoid", "depth-central"):
                    rng = np.random.default_rng(20_000 + rep)
                    for method in METHODS:
                        torch.manual_seed(rep)
                        factory = lambda: make_model(a.model, d)
                        models, dep = run_one(method, nodes, Xtr, ytr, Xtr, ytr,
                                              D, seed, B, eta, cfg, factory,
                                              rng, mode)
                        val = eval_all(models, Xte, yte, task, stats)
                        rows.append(dict(dataset=ds, model=a.model, B=B,
                                         rep=rep, root=mode, method=method,
                                         value=val, depth=dep))
                sel = [r for r in rows if r["rep"] == rep and r["B"] == B]
                print(f"  {ds} rep={rep} B={B}: " + " ".join(
                    f"{r['method'][:8]}/{r['root'][:3]}="
                    f"{r['value']:.3f}" for r in sel), flush=True)
            pd.DataFrame(rows).to_csv(HERE / "root_depthcentral_raw.csv",
                                      index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "root_depthcentral_raw.csv", index=False)
    piv = df.groupby(["dataset", "B", "method", "root"])["value"].mean().unstack()
    piv.to_csv(HERE / "root_depthcentral_summary.csv")
    print("\n=== mean value by rooting ===", flush=True)
    print(piv.round(4).to_string(), flush=True)
    dep = df.groupby(["dataset", "method", "root"])["depth"].mean().unstack()
    print("\n=== mean tree depth ===", flush=True)
    print(dep.round(1).to_string(), flush=True)

    from scipy import stats as st
    print("\n=== paired over reps, depth-central vs medoid ===", flush=True)
    for (ds, B), g in df.groupby(["dataset", "B"]):
        out = []
        for m in METHODS[1:]:
            a1 = g[(g.method == m) & (g.root == "medoid")].sort_values("rep")["value"].values
            a2 = g[(g.method == m) & (g.root == "depth-central")].sort_values("rep")["value"].values
            if len(a1) < 3 or np.allclose(a1, a2):
                out.append(f"{m} identical")
                continue
            t = st.wilcoxon(a2, a1)
            better = 100 * (a2.mean() - a1.mean()) / a1.mean()
            out.append(f"{m} {better:+.1f}% p={t.pvalue:.3f}")
        print(f"{ds} B={B}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
