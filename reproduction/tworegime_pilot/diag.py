"""Diagnostic geometrique des deux generateurs de la grille arbre/chaine (sans entrainement)."""
import sys
import numpy as np
from scipy.stats import spearmanr
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT + "/cascader/reproduction/scripts")
sys.path.insert(0, ROOT + "/cascader/src")
from table1_nonconvex import tree_chain, tree_mst
from data import generate_clustered_tasks, normalize_xy
from graph import compute_gradient_distance, select_seed_node
from cascader.data.synthetic import generate_subspace_tasks

T, K = 200, 5

def gen(kind, w, rep):
    if kind == "old":
        th, _, _, (Xt, yt), _, cid = generate_clustered_tasks(T=T, d=5, n_train=64, n_test=128, n_clusters=K,
            tau_within=1.0, tau_between=w, sigma=0.5, rng=np.random.default_rng(7000 + rep))
    else:
        th, _, (Xt, yt), _, cid = generate_subspace_tasks(T=T, D=40, r=3, n_train=512 if kind == "syn_n512" else 64, n_test=128, n_clusters=K,
            tau_within=3.0, tau_between=3.0 * w, sigma=0.5 if kind == "syn_s05" else 5.0, rng=np.random.default_rng(rep))
    return th, Xt, yt, cid

def depth(root, edges):
    ch = {}
    for u, v in edges: ch.setdefault(u, []).append(v)
    d, st, mx = {root: 0}, [root], 0
    while st:
        u = st.pop()
        for v in ch.get(u, []):
            d[v] = d[u] + 1; mx = max(mx, d[v]); st.append(v)
    return mx

rows = []
for kind in ("old", "syn", "syn_s05", "syn_n512"):
    for w in (0.5, 2, 5, 20):
        acc = []
        for rep in range(20):
            th, Xt, yt, cid = gen(kind, w, rep)
            nodes = list(range(T))
            Xn, yn, ols, pop = {}, {}, [], []
            for v in nodes:
                Xn[v], yn[v], s = normalize_xy(Xt[v], yt[v])
                A = np.c_[Xn[v]]
                ols.append(np.linalg.lstsq(A, yn[v], rcond=None)[0])      # cible effective de la descente
                pop.append(th[v] * s["X_std"].ravel() / s["y_std"])     # parametre vrai dans l'espace normalise
            ols, pop = np.array(ols), np.array(pop)
            Dm = compute_gradient_distance(nodes, Xn, yn)
            iu = np.triu_indices(T, 1)
            Dpop = np.linalg.norm(pop[:, None] - pop[None], axis=-1)
            same = cid[:, None] == cid[None]
            within = np.median(Dpop[iu][same[iu]]); between = np.median(Dpop[iu][~same[iu]])
            noise = np.median(np.linalg.norm(ols - pop, axis=1))
            # separation du proxy : AUC d'une paire intra contre une paire inter
            dw, db = Dm[iu][same[iu]], Dm[iu][~same[iu]]
            auc = (np.searchsorted(np.sort(db), dw, side="right")).sum()
            auc = 1 - auc / (len(dw) * len(db))
            auc = 1 - auc
            root = select_seed_node({v: Xt[v] for v in nodes}, {v: yt[v] for v in nodes})
            r_mst, e_mst, _ = tree_mst(nodes, Dm, root)
            r_ch, e_ch, _ = tree_chain(nodes, Dm, root)
            inter = lambda E: sum(cid[u] != cid[v] for u, v in E)
            H = [np.linalg.cond(Xn[v].T @ Xn[v] / len(Xn[v])) for v in nodes[:50]]
            acc.append(dict(within=within, between=between, noise=noise,
                            rho=spearmanr(Dm[iu], Dpop[iu])[0], auc=auc,
                            inter_mst=inter(e_mst), inter_chain=inter(e_ch),
                            depth_mst=depth(r_mst, e_mst), cond=np.median(H),
                            ynorm=np.median(np.linalg.norm(pop, axis=1))))
        m = {k: np.mean([a[k] for a in acc]) for k in acc[0]}
        rows.append(dict(regime=kind, scatter=w, **m))
        print(f"{kind} w={w:<4} intra={m['within']:.3f} inter={m['between']:.3f} bruitOLS={m['noise']:.3f} "
              f"bruit/intra={m['noise']/m['within']:.2f} |theta|={m['ynorm']:.2f} rhoProxy={m['rho']:.2f} "
              f"P(intra<inter)={m['auc']:.2f} aretesInterMST={m['inter_mst']:.1f} aretesInterChain={m['inter_chain']:.1f} "
              f"profMST={m['depth_mst']:.0f} kappa={m['cond']:.0f}", flush=True)

import pandas as pd
pd.DataFrame(rows).to_csv(__import__("pathlib").Path(__file__).with_name("diag20.csv"), index=False)
