"""Why the deepest tree is not the one with the longest paths.

Proposition 1 attenuates a root-to-leaf path by the product of its edge terms, so
what governs the accumulated error is the cumulative transfer distance along the
path, not the number of hops on its own. A greedy nearest-neighbour chain has the
worst possible depth, T-1, yet strings together the shortest available edges, so
its cumulative path length can stay comparable to a balanced tree's. A random
chain has the same depth for a far longer cumulative length, which is why it
destroys the gain.

This script measures the three quantities on the actual constructed trees:
depth, cumulative root-to-leaf distance, and mean edge length.

    python3.12 path_geometry.py
"""

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (weave_tasks, image_tasks, syn_tasks,
                              tree_mst, tree_chain, tree_random, tree_mstc)
from graph import compute_gradient_distance, select_seed_node


def path_stats(nodes, dir_edges, root, D):
    """Depth and cumulative distance of every root-to-leaf path."""
    idx = {v: i for i, v in enumerate(nodes)}
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    depth = {root: 0}
    length = {root: 0.0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            length[v] = length[u] + D[idx[u], idx[v]]
            q.append(v)
    leaves = [v for v in nodes if not children[v]]
    return (np.array([depth[v] for v in leaves]),
            np.array([length[v] for v in leaves]),
            np.array([D[idx[u], idx[v]] for u, v in dir_edges]))


def shuffled_chain(nodes, seed, rng):
    """A chain in random order, the control the paper's ablation already runs."""
    order = [seed] + list(rng.permutation([v for v in nodes if v != seed]))
    return seed, [(order[i], order[i + 1]) for i in range(len(order) - 1)], None


def report(name, nodes, D, X, y, seed):
    rng = np.random.default_rng(0)
    rows = []
    trees = {
        "Star": (seed, [(seed, v) for v in nodes if v != seed], None),
        "RandTree": tree_random(nodes, D, seed, np.random.default_rng(0)),
        "Chain (greedy NN)": tree_chain(nodes, D, seed),
        "Chain (shuffled)": shuffled_chain(nodes, seed, rng),
        "CTL-MST": tree_mst(nodes, D, seed),
        "CTL-MSTc": tree_mstc(nodes, D, seed, X, y, 0.75),
    }
    for label, (r, edges, _) in trees.items():
        dep, cum, ew = path_stats(nodes, edges, r, D)
        rows.append(dict(dataset=name, tree=label, T=len(nodes),
                         depth_med=float(np.median(dep)),
                         depth_max=int(dep.max()),
                         cum_med=float(np.median(cum)),
                         cum_max=float(cum.max()),
                         edge_mean=float(ew.mean())))
    df = pd.DataFrame(rows)
    print(f"\n=== {name}, T={len(nodes)} ===", flush=True)
    print(df[["tree", "depth_med", "depth_max", "cum_med", "cum_max",
              "edge_mean"]].to_string(index=False,
                                      float_format=lambda x: f"{x:.3f}"), flush=True)
    return rows


def main():
    rows = []
    nodes, Xn_tr, yn_tr, _, _, _, D, seed = weave_tasks()
    rows += report("WEAVE-UK", nodes, D, Xn_tr, yn_tr, seed)

    for ds in ("FMNIST", "CIFAR"):
        nodes, Xtr, ytr, _, _ = image_tasks(ds, seed=0)
        D = compute_gradient_distance(nodes, Xtr, ytr)
        rows += report(ds, nodes, D, Xtr, ytr, select_seed_node(Xtr, ytr))

    syn = syn_tasks(0)
    nodes, Xtr, ytr = syn[0], syn[1], syn[2]
    D = compute_gradient_distance(nodes, Xtr, ytr)
    rows += report("Syn-5", nodes, D, Xtr, ytr, select_seed_node(Xtr, ytr))

    pd.DataFrame(rows).to_csv(HERE / "path_geometry.csv", index=False)
    print("\nécrit path_geometry.csv", flush=True)


if __name__ == "__main__":
    main()
