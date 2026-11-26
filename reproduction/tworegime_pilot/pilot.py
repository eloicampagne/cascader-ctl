"""Pilote : grille arbre/chaine de la table 4 sous plusieurs generateurs, code actuel du package.
Usage : python3.12 pilot.py REGIME EVAL_REPS TUNE_REPS
REGIME : old | syn | syn_s05 | syn_n512"""
import sys, os, time
import numpy as np, pandas as pd
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT + "/cascader/reproduction/scripts"); sys.path.insert(0, ROOT + "/cascader/src")
import tworegime_configs as TC
import tworegime_final as TF
from data import generate_clustered_tasks, normalize_xy
from graph import compute_gradient_distance
from cascader.data.synthetic import generate_subspace_tasks

regime, ne, nt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = os.path.dirname(os.path.abspath(__file__))

def tasks(scatter, rep):
    T = TC.T
    if regime == "old":
        _, _, _, (Xt, yt), (Xe, ye), cid = generate_clustered_tasks(T=T, d=5, n_train=64, n_test=128, n_clusters=5,
            tau_within=1.0, tau_between=scatter, sigma=0.5, rng=np.random.default_rng(7000 + rep))
    else:
        sigma = 0.5 if regime == "syn_s05" else 5.0
        ntr = 512 if regime == "syn_n512" else 64
        _, _, (Xt, yt), (Xe, ye), cid = generate_subspace_tasks(T=T, D=40, r=3, n_train=ntr, n_test=128, n_clusters=5,
            tau_within=3.0, tau_between=3.0 * scatter, sigma=sigma, rng=np.random.default_rng(rep))
    nodes = list(range(T)); Xn, yn, Xte, st = {}, {}, {}, {}
    for v in nodes:
        Xn[v], yn[v], s = normalize_xy(Xt[v], yt[v]); st[v] = s
        Xte[v] = (Xe[v] - s["X_mean"]) / s["X_std"]
    Dm = compute_gradient_distance(nodes, Xn, yn)
    roots = TC.roots_for(nodes, {v: Xt[v] for v in nodes}, {v: yt[v] for v in nodes}, Dm)
    return nodes, Xn, yn, Xte, {v: ye[v] for v in nodes}, st, Dm, roots

TF.tasks = tasks
TC.set_alloc_mode("floor1")
rows = []
t0 = time.time()
for s in TC.SCATTERS:
    TF.sweep(s, range(TF.TUNE_OFFSET, TF.TUNE_OFFSET + nt), "tune", TF.ETAS, rows)
    TF.sweep(s, range(ne), "eval", TF.ETAS, rows)
    pd.DataFrame(rows).to_csv(f"{OUT}/pilot_{regime}_raw.csv", index=False)
df = pd.DataFrame(rows)
TF.report(df, f"{OUT}/pilot_{regime}_summary.csv")
print("temps", round(time.time() - t0), "s")
