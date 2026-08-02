"""Is there an optimal depth, and an optimal root, for a CTL cascade.

Unrolling the edge-wise recursion $E_v \\le \\rho^{b_v}(E_{pa(v)} + \\delta_v)$
along a root-to-leaf path gives

    E_{v_k} <= sum_j delta_j * exp(-lambda * sum_{i=j..k} b_i),

so an edge is corrected only by the budget spent downstream of it. Under a
uniform allocation this is a geometric discount of ratio q = rho^{B/T} per hop,
counted backwards from the leaf, and the quantity that governs the bound is the
discounted path length

    Phi(r, v) = sum_{j in path(r,v)} delta_j q^{hops from j to v}.

Two predictions follow and both are tested here.

The optimal root minimises Phi, not the number of hops. As q -> 1 the objective
degenerates to the weighted 1-centre (worst case) or the weighted 1-median
(average case) of the tree under delta, neither of which is the medoid, which
ignores the topology, nor the unweighted centre, which counts hops without
weighing them.

The optimal depth is a compromise. Splitting one long jump d into l collinear
hops gives Phi(l) = (d/l) q (1 - q^l)/(1 - q), decreasing in l, so depth helps as
long as the intermediates lie near the geodesic and hurts as soon as they
detour. Two constrained Prim families sweep the depth axis while staying
distance-informed, a depth cap taking Star to the MST and a child-count cap
taking the MST to a binary tree. The cap counts children, so cap 2 gives a binary
tree and not a path, and the realised depth tops out near 50 on T=500. The
depth-499 end of the axis is the greedy nearest-neighbour chain of the main
table, which is not reachable inside this family.

    python3.12 depth_sweep.py --datasets WEAVE-UK,Syn-10 --reps 10
"""

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, MLP_ETA, make_model, eval_all,
                              cascade_train, train_steps, sle_alloc,
                              weave_tasks, image_tasks, syn_tasks)
from graph import compute_gradient_distance, root_tree, select_seed_node

DEPTH_CAPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 0)     # 0 = pas de contrainte
DEGREE_CAPS = (2, 3, 4, 6, 0)
ROOTS = ("medoid", "hop-centre", "w-centre", "w-median")
Bs_REG = (500, 1000, 2000)
Bs_CLS = (500, 2000)


# --- constructions ----------------------------------------------------------

def prim_capped(nodes, D, seed, depth_cap=0, degree_cap=0):
    """Prim depuis `seed`, sous contrainte de profondeur ou de degré.

    depth_cap=1 rend l'étoile, degree_cap=2 un chemin glouton, et les deux à
    zéro l'arbre couvrant minimal ordinaire. La contrainte ne porte que sur les
    parents admissibles, la règle de sélection restant celle de Prim, donc la
    famille reste informée par la distance sur toute son étendue.
    """
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    in_tree = {seed}
    depth = {seed: 0}
    deg = defaultdict(int)
    edges = []
    while len(in_tree) < n:
        elig = [u for u in in_tree
                if (depth_cap == 0 or depth[u] < depth_cap)
                and (degree_cap == 0 or deg[u] < degree_cap)]
        if not elig:                       # contrainte infaisable, on la relâche
            elig = list(in_tree)
        best = None
        for u in elig:
            du = D[idx[u]]
            for v in nodes:
                if v in in_tree:
                    continue
                w = du[idx[v]]
                if best is None or w < best[0]:
                    best = (w, u, v)
        _, u, v = best
        edges.append((u, v))
        in_tree.add(v)
        depth[v] = depth[u] + 1
        deg[u] += 1
    return edges


# --- enracinements ----------------------------------------------------------

def _tree_dists(nodes, undirected, D):
    """Distances le long de l'arbre, pondérées par delta, entre toutes paires."""
    idx = {v: i for i, v in enumerate(nodes)}
    adj = defaultdict(list)
    for u, v in undirected:
        w = D[idx[u], idx[v]]
        adj[u].append((v, w))
        adj[v].append((u, w))
    out = {}
    for s in nodes:
        dist = {s: 0.0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    q.append(v)
        out[s] = dist
    return out


def hop_centre(nodes, undirected):
    adj = defaultdict(set)
    for u, v in undirected:
        adj[u].add(v)
        adj[v].add(u)
    deg = {v: len(adj[v]) for v in nodes}
    rem = set(nodes)
    leaves = [v for v in nodes if deg[v] <= 1]
    while len(rem) > 2 and leaves:
        nxt = []
        for v in leaves:
            rem.discard(v)
            for w in adj[v]:
                if w in rem:
                    deg[w] -= 1
                    if deg[w] == 1:
                        nxt.append(w)
        leaves = nxt
    return sorted(rem)[0]


def pick_root(kind, nodes, undirected, D, medoid_node):
    if kind == "medoid":
        return medoid_node
    if kind == "hop-centre":
        return hop_centre(nodes, undirected)
    td = _tree_dists(nodes, undirected, D)
    if kind == "w-centre":
        return min(nodes, key=lambda s: max(td[s].values()))
    if kind == "w-median":
        return min(nodes, key=lambda s: sum(td[s].values()))
    raise ValueError(kind)


def stats_of(nodes, root, dir_edges, D):
    idx = {v: i for i, v in enumerate(nodes)}
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    dep, cum = {root: 0}, {root: 0.0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            dep[v] = dep[u] + 1
            cum[v] = cum[u] + D[idx[u], idx[v]]
            q.append(v)
    leaves = [v for v in nodes if not children[v]]
    return max(dep.values()), float(np.mean([cum[v] for v in leaves]))


# --- boucle -----------------------------------------------------------------

def load(ds, rep):
    if ds == "WEAVE-UK":
        nodes, Xtr, ytr, Xte, yte, st, D, seed = weave_tasks()
        return nodes, Xtr, ytr, Xte, yte, st, D, seed, Bs_REG
    if ds == "Syn-10":
        nodes, Xtr, ytr, Xte, yte, st, D, seed = syn_tasks(rep)
        return nodes, Xtr, ytr, Xte, yte, st, D, seed, Bs_REG
    nodes, Xtr, ytr, Xte, yte = image_tasks(ds, seed=rep)
    D = compute_gradient_distance(nodes, Xtr, ytr)
    return nodes, Xtr, ytr, Xte, yte, None, D, select_seed_node(Xtr, ytr), Bs_CLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="WEAVE-UK,Syn-10")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--model", default="mlp", choices=["mlp", "linear"])
    a = ap.parse_args()

    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        eta = MLP_ETA[ds] if a.model == "mlp" else cfg["eta"]
        for rep in range(a.reps):
            nodes, Xtr, ytr, Xte, yte, st, D, med, Bs = load(ds, rep)
            d = Xtr[nodes[0]].shape[1]
            fams = ([("depth", c) for c in DEPTH_CAPS]
                    + [("degree", c) for c in DEGREE_CAPS if c])
            for fam, cap in fams:
                und = prim_capped(nodes, D, med,
                                  depth_cap=cap if fam == "depth" else 0,
                                  degree_cap=cap if fam == "degree" else 0)
                for rk in ROOTS:
                    r = pick_root(rk, nodes, und, D, med)
                    root, dir_edges, _ = root_tree(und, r)
                    dep, cum = stats_of(nodes, root, dir_edges, D)
                    for B in Bs:
                        torch.manual_seed(rep)
                        alloc = sle_alloc(nodes, root, dir_edges, D, B)
                        models = cascade_train(dir_edges, root, Xtr, ytr, alloc,
                                               eta, cfg["task"], cfg["reg"],
                                               lambda: make_model(a.model, d))
                        rows.append(dict(
                            dataset=ds, rep=rep, family=fam, cap=cap, root=rk,
                            B=B, depth=dep, cum_len=cum,
                            value=eval_all(models, Xte, yte, cfg["task"], st)))
                print(f"  {ds} rep={rep} {fam}={cap}: " + " ".join(
                    f"{r['root'][:8]}/d{r['depth']}@{r['B']}={r['value']:.3f}"
                    for r in rows[-len(ROOTS) * len(Bs):]), flush=True)
            pd.DataFrame(rows).to_csv(HERE / "depth_sweep_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "depth_sweep_raw.csv", index=False)
    for ds, g in df.groupby("dataset"):
        for B, gb in g.groupby("B"):
            t = gb.groupby(["family", "cap"]).agg(
                depth=("depth", "mean"), cum=("cum_len", "mean")).reset_index()
            v = gb.pivot_table(index=["family", "cap"], columns="root",
                               values="value")
            t = t.merge(v.reset_index(), on=["family", "cap"])
            t = t.sort_values("depth")
            print(f"\n=== {ds}, B={B}, valeur par profondeur réalisée ===",
                  flush=True)
            print(t.to_string(index=False,
                              float_format=lambda x: f"{x:.3f}"), flush=True)
    df.groupby(["dataset", "family", "cap", "B", "root"]).agg(
        depth=("depth", "mean"), cum=("cum_len", "mean"),
        value=("value", "mean")).to_csv(HERE / "depth_sweep_summary.csv")


if __name__ == "__main__":
    main()
