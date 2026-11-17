"""Frontiere CTL / transfert direct du Theoreme (ctl-vs-star), mesuree exactement.

Remplace expB_ctl_vs_direct (supplementary_experiments.py), dont la mesure melait
des configurations hors hypotheses (rho >= 1), comptait des egalites comme des
victoires et ne situait pas la frontiere.

Cadre du theoreme : source exacte, chemin s -> v_1 -> ... -> v_m = v, meme
budget b a chaque noeud, erreur mesuree a la cible. Pas de bruit : objectifs
quadratiques L_v(th) = 1/2 (th - th_v*)' H_v (th - th_v*), b pas de gradient
de pas eta, donc th - th_v* <- (I - eta H_v)^b (th - th_v*) exactement.

Geometrie (dimension D=20, d(s,v)=1 sans perte de generalite, la dynamique est
lineaire et sans bruit) : k intermediaires aux abscisses j/(k+1) sur le segment
s -> v, deplaces orthogonalement
    arc       de xi * sin(pi j/(k+1)) le long d'une direction w fixe
              (detour regulier, aretes de longueurs voisines)
    zigzag    de xi le long de directions w_j orthogonales a u, tirees au hasard
              (le plan de l'ancienne mesure)
delta_max / d(s,v) est calcule exactement ; les configurations d'un chemin de
MST verifient delta_max <= d(s,v) (propriete minimax du MST).

Familles :
    A  isotrope, H_v = lam I, rho = 1 - eta lam exact, q = rho^b ; le facteur
       rho_v^b de la cible est commun aux deux erreurs, la comparaison ne depend
       que de (q, geometrie, k) ; deterministe pour l'arc.
    B  anisotrope, H_v = R_v diag(spectre) R_v' avec R_v tiree au hasard,
       spectre log-uniforme sur [L/c, L], eta = 1/L, donc rho_v = 1 - 1/c < 1 ;
       homogene (c identique partout) ou heterogene (intermediaires c_int,
       cible c_tgt < c_int, rho_v << rho_max). 20 graines.

Sorties : ratio err_CTL / err_direct, ratio theorique
    r = delta_max (1 - q^m) / (d(s,v) (1 - q)),  q = rho_max^b,
et frontiere empirique (ratio = 1) contre frontiere theorique (r = 1).

    python3.12 boundary_phase.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIM = 20
KS = (1, 7, 31)
XIS = np.round(np.linspace(0.0, 2.0, 41), 3)          # zigzag et famille B
def xis_arc(k):
    """Amplitudes de l'arc : delta_max/d atteint environ 2 pour chaque k."""
    return np.round(np.linspace(0.0, max(2.1, 0.7 * (k + 1)), 161), 4)


def path_optima(k, xi, geom, rng):
    u = np.zeros(DIM)
    u[0] = 1.0
    pts = [np.zeros(DIM)]
    w_fixed = np.zeros(DIM)
    w_fixed[1] = 1.0
    for j in range(1, k + 1):
        base = (j / (k + 1)) * u
        if geom == "arc":
            pts.append(base + xi * np.sin(np.pi * j / (k + 1)) * w_fixed)
        else:
            w = rng.normal(size=DIM)
            w -= (w @ u) * u
            w /= np.linalg.norm(w)
            pts.append(base + xi * w)
    pts.append(u.copy())
    return pts


def deltas(pts):
    return [float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts))]


def is_mst_path(pts):
    """Vrai si le chemin s -> v_1 -> ... -> v est l'arbre couvrant minimal de ses points (distance d)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    P = np.array(pts)
    W = np.linalg.norm(P[:, None] - P[None], axis=-1)
    T = minimum_spanning_tree(W).toarray()
    E = {tuple(sorted(e)) for e in zip(*np.nonzero(T))}
    return E == {(i, i + 1) for i in range(len(pts) - 1)}


# ---------------------------------------------------------------- famille A

def family_A():
    rows = []
    qs = np.round(np.concatenate([np.linspace(0.02, 0.9, 45), np.linspace(0.91, 0.995, 18)]), 4)
    for geom, seeds in (("arc", [0]), ("zigzag", range(20))):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for k in KS:
                for xi in (xis_arc(k) if geom == "arc" else XIS):
                    pts = path_optima(k, xi, geom, rng)
                    dl = deltas(pts)
                    mstf = is_mst_path(pts)
                    edges = [pts[i] - pts[i - 1] for i in range(1, len(pts))]
                    m = k + 1
                    for q in qs:
                        # erreur CTL (sans le facteur commun rho_v^b) :
                        # e_m / rho_v^b = sum_i q^{m-i} * (th_{i-1} - th_i)
                        e = sum(q ** (m - i) * (-edges[i - 1]) for i in range(1, m + 1))
                        err_ctl = float(np.linalg.norm(e))
                        err_dir = 1.0                       # d(s,v) = 1
                        r = max(dl) * (1 - q ** m) / (1 - q)
                        rows.append(dict(family="A", geom=geom, seed=seed, k=k, xi=xi, q=q, mst=mstf,
                                         dmax_over_d=max(dl), r=r, ratio=err_ctl / err_dir,
                                         bound_ratio=sum(q ** (m - i) * dl[i - 1] for i in range(1, m + 1))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- famille B

def rand_hessian(rng, c, L=1.0):
    Q = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]
    spec = np.exp(rng.uniform(np.log(L / c), np.log(L), size=DIM))
    spec[0], spec[-1] = L, L / c                      # bornes atteintes : rho = 1 - 1/c
    return Q @ np.diag(spec) @ Q.T


def family_B():
    rows = []
    settings = [("hom", 10, 10), ("hom", 100, 100), ("het", 100, 4), ("het", 1000, 4)]
    bs = (1, 2, 4, 8, 16, 32, 64)
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        for name, c_int, c_tgt in settings:
            for k in KS:
                Hs = [rand_hessian(rng, c_int) for _ in range(k)] + [rand_hessian(rng, c_tgt)]
                rho_int = 1 - 1 / c_int
                rho_tgt = 1 - 1 / c_tgt
                rho_max = max(rho_int, rho_tgt)
                for xi in xis_arc(k)[::16]:
                    pts = path_optima(k, xi, "arc", np.random.default_rng(seed))
                    dl = deltas(pts)
                    mstf = is_mst_path(pts)
                    m = k + 1
                    for b in bs:
                        Ms = [np.linalg.matrix_power(np.eye(DIM) - H, b) for H in Hs]   # eta = 1/L = 1
                        th = pts[0].copy()
                        for i in range(1, m + 1):
                            th = pts[i] + Ms[i - 1] @ (th - pts[i])
                        err_ctl = float(np.linalg.norm(th - pts[-1]))
                        err_dir = float(np.linalg.norm(Ms[-1] @ (pts[0] - pts[-1])))
                        q = rho_max ** b
                        r = max(dl) * (1 - q ** m) / (1 - q)
                        cond = sum((rho_int ** b) ** (m - i) * dl[i - 1] for i in range(1, m + 1))
                        rows.append(dict(family="B", setting=f"{name}-c{c_int}/{c_tgt}", seed=seed, k=k,
                                         cond_exact=cond, mst=mstf,
                                         xi=xi, b=b, q=q, rho_tgt=rho_tgt, rho_max=rho_max,
                                         dmax_over_d=max(dl), r=r, ratio=err_ctl / err_dir,
                                         rel_gap=(err_ctl - err_dir) / err_dir))
    return pd.DataFrame(rows)


def summarize(A, B):
    """Imprime chaque chiffre cite dans l'annexe app:boundary. Victoire : erreur strictement plus basse."""
    A = A.assign(win=A.ratio < 1, cond=A.bound_ratio < 1)
    B = B.assign(win=B.ratio < 1, cond=B.cond_exact < 1)
    print(f"A : {len(A)} configurations ; B : {len(B)} configurations")
    print("\n[A] condition vraie -> CTL gagne (consequence de l'inegalite triangulaire) :",
          A[A.cond].groupby("geom").win.mean().round(4).to_dict())
    print("\n[A, arc] premier delta_max/d ou CTL perd, rapporte au seuil de la condition :")
    a = A[A.geom == "arc"]
    for k in KS:
        line = []
        for q in (0.3, 0.5, 0.7, 0.9, 0.99):
            qq = a.q.iloc[(a.q - q).abs().argmin()]
            h = a[(a.k == k) & (a.q == qq)]
            lose = h[~h.win].dmax_over_d.min()
            thr = h[h.cond].dmax_over_d.max()
            line.append(f"q={qq:.2f}: {thr:.2f}->{lose:.2f} (x{lose / thr:.2f})")
        print(f"  k={k}: " + " ; ".join(line))
    print("\n[A] gain sur la geodesique (1 - rapport), k=7 :",
          {q: round(1 - a[(a.k == 7) & (a.xi == 0) & (np.isclose(a.q, a.q.iloc[(a.q - q).abs().argmin()]))].ratio.iloc[0], 4)
           for q in (0.5, 0.9, 0.99)})
    print("\n[A] chemins qui sont le MST de leurs points : part des configurations, part des victoires")
    for geom, g in A.groupby("geom"):
        print(f"  {geom}: MST {g.mst.mean():.3f} ; CTL gagne {g[g.mst].win.mean():.3f} sur MST,"
              f" {g[~g.mst].win.mean():.3f} hors MST ; pertes MST par k : {g[g.mst & ~g.win].groupby('k').size().to_dict()}")
    print("\n[B] condition vraie -> CTL gagne, par reglage (toutes valeurs de b) et minimum par cellule (reglage, b) :")
    c = B[B.cond]
    print("  ", c.groupby("setting").win.mean().round(4).to_dict())
    print("   min par cellule :", round(c.groupby(["setting", "b"]).win.mean().min(), 4),
          " rapport le plus defavorable :", round(c.ratio.max(), 2))
    print("\n[B] chemins MST : part des victoires par reglage, min et max sur b :")
    m = B[B.mst].groupby(["setting", "b"]).win.mean().groupby("setting").agg(["min", "max"]).round(4)
    print(m.to_string())


def main():
    A, B = family_A(), family_B()
    A.to_csv(HERE / "boundary_phase_A.csv", index=False)
    B.to_csv(HERE / "boundary_phase_B.csv", index=False)
    summarize(A, B)


if __name__ == "__main__":
    main()
