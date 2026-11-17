"""Critere d'arete inconsistante de Zahn, version locale, contre notre variante globale.

Zahn (1971) ne compare pas une arete a la distribution de tout l'arbre : pour
u-v il regarde les aretes situees a d sauts de u et celles a d sauts de v, et
declare u-v inconsistante si elle depasse mu + f*sigma des DEUX voisinages. La
coupe du papier utilise la distribution globale, ce qui est plus simple et sans
profondeur de voisinage a choisir. On mesure ce que la difference change.
"""
import numpy as np
from collections import defaultdict, deque


def _adj(nodes, edges):
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v); g[v].append(u)
    return g


def _nbh_edges(g, start, skip, depth):
    """Aretes a <= depth sauts de start, en excluant l'arete skip."""
    seen, out, q = {start}, [], deque([(start, 0)])
    while q:
        x, dd = q.popleft()
        if dd == depth:
            continue
        for y in g[x]:
            if {x, y} == set(skip):
                continue
            out.append((x, y))
            if y not in seen:
                seen.add(y); q.append((y, dd + 1))
    return out


def inconsistent_local(nodes, edges, D, f=2.0, depth=2):
    idx = {v: i for i, v in enumerate(nodes)}
    ln = lambda e: D[idx[e[0]], idx[e[1]]]
    g = _adj(nodes, edges)
    cut = []
    for e in edges:
        ok = True
        for end in e:
            nb = _nbh_edges(g, end, e, depth)
            if len(nb) < 2:
                ok = False; break
            L = np.array([ln(x) for x in nb])
            if ln(e) <= L.mean() + f * L.std():
                ok = False; break
        if ok:
            cut.append(e)
    return cut


def inconsistent_global(nodes, edges, D, f=2.0):
    idx = {v: i for i, v in enumerate(nodes)}
    L = np.array([D[idx[u], idx[v]] for u, v in edges])
    thr = L.mean() + f * L.std()
    return [e for e, l in zip(edges, L) if l > thr]
