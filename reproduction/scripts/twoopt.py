"""2-opt improvement of a rooted Hamiltonian path (root fixed at the start).

Reversing the segment p[i..j] replaces the edges (p[i-1], p[i]) and (p[j], p[j+1])
by (p[i-1], p[j]) and (p[i], p[j+1]); every other edge keeps its length, since the
distance is symmetric. A pass takes, for each i, the reversal of largest decrease,
and passes repeat until no reversal shortens the path."""
import numpy as np


def two_opt_order(p, D, tol=1e-12):
    p = np.array(p); n = len(p); D = np.asarray(D, float)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a, b = p[i - 1], p[i]
            j = np.arange(i + 1, n); c = p[j]
            nxt = p[np.minimum(j + 1, n - 1)]; last = j + 1 >= n
            old = D[c, nxt]; new = D[b, nxt]
            old[last] = 0.0; new[last] = 0.0
            delta = D[a, c] + new - D[a, b] - old
            k = int(np.argmin(delta))
            if delta[k] < -tol:
                jj = j[k]; p[i:jj + 1] = p[i:jj + 1][::-1]; improved = True
    return p
