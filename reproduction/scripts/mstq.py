"""CTL-MSTq, la construction que le surrogate donne directement.

En réorganisant la somme des longueurs de chemin escomptées par arête plutôt que
par tâche,

    sum_v Phi(r,v) = sum_{e=(u,v)} delta_e * M_q(v),
    M_q(v) = sum_{w dans sous-arbre(v)} q^{1 + prof(w)},

le coût d'une arête est sa longueur multipliée par la masse escomptée du
sous-arbre qu'elle portera, avec q = rho^{B/T}. La construction gloutonne
associée attache donc, à chaque étape, l'arête minimisant delta(u,v) * M_q(v),
la masse future de v étant estimée par son bassin, c'est-à-dire le nombre de
tâches encore hors de l'arbre qui sont plus proches de v que de l'arbre courant.

Trois différences avec CTL-MSTc, toutes issues de la dérivation. Pas de softplus,
la borne étant linéaire en delta. La pénalité de masse porte sur l'enfant et non
sur le parent, et elle est escomptée par q. Pas de terme de portée, qui n'a aucun
correspondant dans Phi et qui est le terme allongeant les arêtes.

Aux deux limites la construction fait ce qu'il faut. Quand q tend vers zéro, à
budget large, elle redonne Prim sur les distances brutes. Quand q tend vers un, à
budget serré, elle pénalise les longues arêtes portant beaucoup de descendants.
"""

import numpy as np


def mstq_edges(nodes, D, root, q=0.9, cap=None, lam=0.0):
    """Prim glouton sur delta(u,v) * M_q_estime(v).

    En développant M_q(v) = q (1 + q n_v + ...), le facteur q de tête est commun
    à tous les candidats et ne change pas l'argmin, si bien que le score se
    ramène à delta(u,v) * (1 + q n_v), où n_v compte les tâches non encore
    attachées plus proches de v que de l'arbre courant, tronqué à l'horizon
    1/(1-q) au-delà duquel l'escompte géométrique rend les descendants
    invisibles. Avec q=0 le score vaut delta et la construction est exactement
    celle de Prim.
    """
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    D = np.asarray(D, dtype=float)
    horizon = cap if cap is not None else (1.0 / max(1e-9, 1.0 - q))

    r = idx[root]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[r] = True
    min_cost = D[r].copy()
    min_cost[r] = 0.0
    parent = np.full(n, -1, dtype=int)
    parent[np.flatnonzero(~in_tree)] = r
    edges = []

    while in_tree.sum() < n:
        rem = np.flatnonzero(~in_tree)
        # bassin de chaque candidat parmi les tâches encore hors de l'arbre
        sub = D[np.ix_(rem, rem)]
        catch = (sub < min_cost[rem][None, :]).sum(axis=1)      # v en ligne
        score = min_cost[rem] * (1.0 + q * np.minimum(catch, horizon))
        if lam:
            # Variante hybride: on garde le gain de portee de CTL-MSTc, absent du
            # surrogate mais empiriquement utile sur WEAVE et CIFAR, tout en
            # conservant les trois corrections que la derivation impose.
            gain = np.maximum(0.0, min_cost[rem][None, :] - sub).sum(axis=1)
            score = score - lam * gain
        pick = int(np.argmin(score))
        v = int(rem[pick])
        u = int(parent[v])
        edges.append((nodes[u], nodes[v]))
        in_tree[v] = True
        better = D[v] < min_cost
        parent[better & ~in_tree] = v
        min_cost = np.minimum(min_cost, D[v])
        min_cost[v] = 0.0
    return edges
