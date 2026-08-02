"""Chemin critique de la cascade, en pas de gradient et en rondes parallèles.

Une cascade ne se parallelise qu'aux branchements. Toutes les taches d'une meme
profondeur peuvent etre raffinees simultanement, puisqu'elles ne dependent que de
parents deja entraines, si bien que le temps d'horloge sur une machine a T
travailleurs vaut le maximum, sur les feuilles, de la somme des budgets le long du
chemin racine-feuille. Une chaine n'a aucun branchement, son chemin critique vaut
donc le budget total. Un arbre a branches le divise.

    python3.12 critical_path.py
"""

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, sle_alloc, uniform_alloc, tree_chain,
                              tree_mst, tree_mstc)
from mechanism_edge_vs_depth import load
from mstq import mstq_edges

METHODS = ["Chain", "CTL-MST", "CTL-MSTc"]


def build(m, nodes, D, root, X, y, lam):
    if m == "Chain":
        return tree_chain(nodes, D, root)
    if m == "CTL-MST":
        return tree_mst(nodes, D, root)
    if m == "CTL-MSTc":
        return tree_mstc(nodes, D, root, X, y, lam)
    e = mstq_edges(nodes, D, root, q=0.9)
    return root, e, {v: u for u, v in e}


def stats(edges, root, alloc):
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    cum = {root: alloc.get(root, 0)}
    dep = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            cum[v] = cum[u] + alloc.get(v, 0)
            dep[v] = dep[u] + 1
            q.append(v)
    width = defaultdict(int)
    for v, d in dep.items():
        width[d] += 1
    return max(cum.values()), max(dep.values()) + 1, max(width.values())


rows = []
for ds in ["Syn-10", "WEAVE-UK", "CIFAR"]:
    cfg = CONFIG[ds]
    nodes, Xtr, ytr, _, _, _, D, root, Bs = load(ds, 0)
    for B in Bs:
        for m in METHODS:
            r, edges, _ = build(m, nodes, D, root, Xtr, ytr, cfg["mstc_lambda"])
            for kind in ("sle", "flat"):
                alloc = (sle_alloc(nodes, r, edges, D, B) if kind == "sle"
                         else uniform_alloc(nodes, r, edges, B,
                                            cfg["b_proto_frac"]))
                cp, rounds, width = stats(edges, r, alloc)
                rows.append(dict(dataset=ds, n_tasks=len(nodes), B=B, alloc=kind,
                                 method=m, critical_path=cp, rounds=rounds,
                                 max_width=width))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\n=== facteur de reduction du chemin critique, Chain / methode ===")
for (ds, B, kind), g in df.groupby(["dataset", "B", "alloc"]):
    base = g[g.method == "Chain"].iloc[0]
    out = []
    for m in METHODS[1:]:
        h = g[g.method == m].iloc[0]
        out.append(f"{m} x{base.critical_path / max(1, h.critical_path):.2f} "
                   f"(rondes {int(base.rounds)} -> {int(h.rounds)})")
    print(f"{ds:9s} T={int(base.n_tasks):4d} B={B:5d} {kind:4s}: " + " | ".join(out))
df.to_csv(HERE / "critical_path.csv", index=False)
