"""Grille dispersion x budget, sous toutes les allocations et tous les enracinements.

Sous l'allocation floor1 et l'enracinement medoid, la frontiere entre chaine et
arbre n'est plus un seuil mais une fenetre : l'arbre perd a b = 1, gagne entre
1,5 et 3, reperd ensuite. Deux ingredients peuvent produire cet effet de bord
bas, et ce script les separe.

  Allocation. floor1 donne un pas a chaque tache avant de repartir le surplus,
  donc a b = 1 toutes les allocations coincident et seule la topologie joue.
  exact repartit strictement au prorata des poids, en laissant certaines taches
  a zero pas : c'est le regime ou l'allocation SLE peut concentrer le budget sur
  les aretes longues, ce que la chaine ne peut pas exploiter.

  Enracinement. Le medoid minimise la somme des distances sur les moyennes de
  features, avant construction de l'arbre. La derivation de Phi donne deux
  autres candidats sur la meme matrice de distances que celle qui construit
  l'arbre : la 1-mediane (somme) et le 1-centre (max).

Generateur identique a tworegime_fine : T = 200, K = 5, d = 5, tau_within = 1,
sigma = 0.5, centres tires a l'echelle s, modele local lineaire.

    python3.12 tworegime_configs.py --reps 10
    python3.12 tworegime_configs.py --reps 20 --allocs sle-exact --roots 1median
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import table1_nonconvex as T1
from table1_nonconvex import (CONFIG, cascade_train, eval_all, make_model,
                              sle_alloc, tree_chain, tree_mst, uniform_alloc)
from data import generate_clustered_tasks, normalize_xy
from graph import compute_gradient_distance, select_seed_node

HERE = Path(__file__).parent.resolve()
SCATTERS = (0.5, 2.0, 5.0, 20.0)
B_PER_TASK = (1, 1.25, 1.5, 2, 3, 4, 5, 7, 10)
METHODS = ("Chain", "CTL-MST")
ALLOCS = ("sle-floor1", "sle-exact", "uniform-floor1", "uniform-exact")
ROOTS = ("medoid", "1median", "1center")
T, D_DIM, K, TAU_WITHIN, SIGMA = 200, 5, 5, 1.0, 0.5


def set_alloc_mode(mode):
    """`_alloc_from_weights` lit l'environnement a l'appel, `uniform_alloc` lit
    une constante de module fixee a l'import : les deux doivent suivre."""
    os.environ["CTL_ALLOC"] = mode
    T1.ALLOC_MODE = mode


def roots_for(nodes, X_raw, y_raw, D):
    """Les trois regles, sur les donnees que chacune est censee lire."""
    nidx = {v: i for i, v in enumerate(nodes)}
    order = np.array([nidx[v] for v in nodes])
    sub = D[np.ix_(order, order)]
    return {"medoid": select_seed_node(X_raw, y_raw),
            "1median": nodes[int(np.argmin(sub.sum(axis=1)))],
            "1center": nodes[int(np.argmin(sub.max(axis=1)))]}


def tasks(scatter, rep):
    if os.environ.get("CTL_SYN") == "subspace":
        # generateur de Syn, memes graines que syn_suite.part_tworegime (table 4) :
        # tau_within = 3, tau_between = 3 * scatter, D = 40, sigma = 5
        sys.path.insert(0, str(HERE.parents[2] / "cascader" / "src"))
        from cascader.data.synthetic import generate_subspace_tasks
        _, _, (Xt, yt), (Xe, ye), _ = generate_subspace_tasks(
            T=T, D=40, r=3, n_train=64, n_test=128, n_clusters=K, tau_within=3.0,
            tau_between=3.0 * scatter, sigma=5.0, rng=np.random.default_rng(rep))
    else:
        rng = np.random.default_rng(7000 + rep)
        _, _, _, (Xt, yt), (Xe, ye), _ = generate_clustered_tasks(
            T=T, d=D_DIM, n_train=64, n_test=128, n_clusters=K,
            tau_within=TAU_WITHIN, tau_between=scatter, sigma=SIGMA, rng=rng)
    nodes = list(range(T))
    X_raw = {v: Xt[v] for v in nodes}
    Xn, yn, Xte, st = {}, {}, {}, {}
    for v in nodes:
        Xn[v], yn[v], s = normalize_xy(Xt[v], yt[v])
        st[v] = s
        Xte[v] = (Xe[v] - s["X_mean"]) / s["X_std"]
    Dm = compute_gradient_distance(nodes, Xn, yn)
    roots = roots_for(nodes, X_raw, {v: yt[v] for v in nodes}, Dm)
    return nodes, Xn, yn, Xte, {v: ye[v] for v in nodes}, st, Dm, roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--allocs", default=",".join(ALLOCS))
    ap.add_argument("--roots", default=",".join(ROOTS))
    ap.add_argument("--scatters", default=",".join(str(s) for s in SCATTERS))
    ap.add_argument("--stem", default="tworegime_configs")
    a = ap.parse_args()

    allocs = a.allocs.split(",")
    roots_wanted = a.roots.split(",")
    scatters = [float(s) for s in a.scatters.split(",")]
    cfg = CONFIG["Syn-10"]
    frac = cfg["b_proto_frac"]
    rows = []

    for scatter in scatters:
        t0 = time.perf_counter()
        for rep in range(a.reps):
            n, Xtr, ytr, Xte, yte, st, Dm, roots = tasks(scatter, rep)
            d = Xtr[n[0]].shape[1]
            median_d = float(np.median(Dm[np.triu_indices(T, k=1)]))
            # l'arbre ne depend que de la racine et de la methode, pas du budget
            trees = {(rn, m): (tree_chain if m == "Chain" else tree_mst)(n, Dm, roots[rn])
                     for rn in roots_wanted for m in METHODS}
            for rn, ac, b in itertools.product(roots_wanted, allocs, B_PER_TASK):
                kind, mode = ac.split("-")
                set_alloc_mode(mode)
                B = int(round(b * T))
                for m in METHODS:
                    root, dir_edges, _ = trees[(rn, m)]
                    alloc = (sle_alloc(n, root, dir_edges, Dm, B) if kind == "sle"
                             else uniform_alloc(n, root, dir_edges, B, frac))
                    assert sum(alloc.values()) == B, (ac, b, m)
                    torch.manual_seed(rep)
                    try:
                        models = cascade_train(dir_edges, root, Xtr, ytr, alloc,
                                               a.eta, cfg["task"], cfg["reg"],
                                               lambda: make_model("linear", d))
                        v = eval_all(models, Xte, yte, "regression", st)
                        if not np.isfinite(v):
                            v = np.nan
                    except RuntimeError:
                        v = np.nan
                    rows.append(dict(scatter=scatter, alloc=ac, root=rn, b=b, B=B,
                                     rep=rep, method=m, value=v,
                                     n_zero=sum(1 for x in alloc.values() if x == 0),
                                     median_distance=median_d))
        print(f"  s={scatter} en {time.perf_counter()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"{a.stem}_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{a.stem}_raw.csv", index=False)
    report(df, HERE / f"{a.stem}_summary.csv")


def report(df, out_path):
    out = []
    for (ac, rn, s, b), g in df.groupby(["alloc", "root", "scatter", "b"]):
        p = g.pivot_table(index="rep", columns="method", values="value").dropna()
        if p.empty:
            continue
        gap = (p["CTL-MST"] - p["Chain"]) / p["Chain"] * 100
        out.append(dict(alloc=ac, root=rn, scatter=s, b=b, gap_pct=gap.mean(),
                        se=gap.std(ddof=1) / np.sqrt(len(gap)),
                        mst=p["CTL-MST"].mean(), chain=p["Chain"].mean(),
                        n_zero=g[g.method == "CTL-MST"]["n_zero"].mean(), n=len(gap)))
    o = pd.DataFrame(out)
    o.to_csv(out_path, index=False)

    for ac in o.alloc.unique():
        for rn in o.root.unique():
            sub = o[(o.alloc == ac) & (o.root == rn)]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="scatter", columns="b", values="gap_pct")
            print(f"\n=== {ac} / racine {rn} : (CTL-MST - Chain)/Chain en % ===")
            print(piv.round(1).to_string())
            print(diagnose(piv))

    print("\n=== classement des configurations ===")
    print(rank(o).to_string(index=False))
    print("done", flush=True)


def diagnose(piv):
    """Une frontiere : gain a bas budget, perte a haut budget, un seul changement
    de signe. Une fenetre : perte aux deux bouts, donc deux changements."""
    lines = []
    for s, row in piv.iterrows():
        v = row.dropna()
        sgn = np.sign(v.to_numpy())
        crossings = int((sgn[1:] * sgn[:-1] < 0).sum())
        neg = v[v < 0]
        span = f"{neg.index.min():g}-{neg.index.max():g}" if len(neg) else "aucune"
        shape = ("frontiere" if crossings == 1 and sgn[0] < 0 else
                 "fenetre" if crossings == 2 else
                 "jamais" if not len(neg) else "autre")
        lines.append(f"  s={s:<5g} {shape:<10s} b<0 sur {span:<9s} "
                     f"gain max {v.min():+.1f}% a b={v.idxmin():g}")
    return "\n".join(lines)


def rank(o):
    """Compte les frontieres et resume le gain sur la moitie basse de la grille."""
    rows = []
    for (ac, rn), g in o.groupby(["alloc", "root"]):
        piv = g.pivot_table(index="scatter", columns="b", values="gap_pct")
        nfront = nwin = 0
        for _, row in piv.iterrows():
            v = row.dropna().to_numpy()
            sgn = np.sign(v)
            c = int((sgn[1:] * sgn[:-1] < 0).sum())
            nfront += int(c == 1 and sgn[0] < 0)
            nwin += int(sgn[0] < 0)
        low = g[g.b <= 2]
        rows.append(dict(alloc=ac, root=rn, frontieres=nfront, gain_a_b1=nwin,
                         gap_b1=g[g.b == 1].gap_pct.mean(),
                         gap_bas=low.gap_pct.mean(),
                         mst_moy=g.mst.mean(), chain_moy=g.chain.mean()))
    return pd.DataFrame(rows).sort_values(["frontieres", "gap_bas"],
                                          ascending=[False, True]).round(2)


if __name__ == "__main__":
    main()
