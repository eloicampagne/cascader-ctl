"""CTL sur M5: 30490 series de vente reelles sous une seule enveloppe de calcul.

Les collections du papier restent petites et homogenes. WEAVE-UK,
la plus grande collection reelle du papier, compte 500 taches. M5 (Makridakis,
Spiliotis et Assimakopoulos, 2022) en offre 30490, soit 3049 produits dans 10
magasins, sur 1941 jours, avec 68 % de zeros. C'est le cas typique: agregat bien mesure, series individuelles partielles et
bruitees.

Chaque serie est une tache. Le modele local est le MLP du papier, la distance est
le proxy X^T y, la racine est le medoide, l'allocation est SLE pour l'arbre et
uniforme pour Star, exactement comme ailleurs. Rien n'est adapte a M5.

Prevision a un pas: les variables du jour t utilisent les ventes reelles jusqu'a
t-1. Ce n'est pas la tache a 28 jours recursifs de la competition, et la metrique
n'est pas la WRMSSE. On le dit plutot que de l'habiller.

Cible en log1p par defaut, et donc RMSLE. C'est la convention du domaine sur des
comptages intermittents, la meme que la NWRMSLE de Favorita, et sans elle la
moyenne des RMSE est dominee par les quelques series a fort volume, dont
l'ecart-type monte a 53 contre une mediane a 0.93. `--target raw` retablit la
RMSE brute.

A T=30490 la matrice de distances ferait 3.7 Go, donc Prim la calcule par
morceaux sans jamais la former.

    python3.12 m5_tasks.py --T 500 --reps 3          # pilote
    python3.12 m5_tasks.py --T 0   --reps 3          # les 30490 series
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (make_model, train_steps, cascade_train,
                              sle_alloc, uniform_alloc)

# Folder holding sales_train_evaluation.csv, sales_test_evaluation.csv and calendar.csv.
M5DIR = Path(__import__("os").environ.get("M5_DATA_DIR", "."))
LAGS = (1, 7, 14, 28)
ROLLS = (7, 28)
N_TEST = 28
B_PROTO_FRAC = 0.15
METHODS = ("Star", "CTL-MST", "CTL-MST-flat")
SLE = {"CTL-MST"}


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

def load_m5(train_days, target="log"):
    """Renvoie (X, y, meta) avec X en (T, W + 28, F) et y en (T, W + 28)."""
    tr = pd.read_csv(M5DIR / "sales_train_evaluation.csv")
    te = pd.read_csv(M5DIR / "sales_test_evaluation.csv")
    cal = pd.read_csv(M5DIR / "calendar.csv")

    meta = tr[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    S = np.concatenate([tr.iloc[:, 5:].values, te.iloc[:, 5:].values],
                       axis=1).astype(np.float32)          # (T, 1969)
    T, D = S.shape
    need = max(max(LAGS), max(ROLLS))
    start = D - N_TEST - train_days                        # premier jour predit
    assert start > need, "fenetre d'entrainement trop longue"

    idx = np.arange(start, D)                              # jours predits
    feats = [S[:, idx - L] for L in LAGS]
    for R in ROLLS:
        c = np.cumsum(np.concatenate([np.zeros((T, 1), np.float32), S], 1), 1)
        feats.append((c[:, idx] - c[:, idx - R]) / R)
        feats.append(((S[:, idx - 1] > 0).astype(np.float32) if R == ROLLS[0]
                      else (c[:, idx] - c[:, idx - R] > 0).astype(np.float32)))

    cal = cal.iloc[:D]
    wday = cal["wday"].values[idx]
    for k in range(1, 8):
        feats.append(np.broadcast_to((wday == k).astype(np.float32), (T, len(idx))))
    feats.append(np.broadcast_to(
        (cal["event_name_1"].notna().values[idx]).astype(np.float32),
        (T, len(idx))))
    snap = {s: cal[f"snap_{s}"].values[idx].astype(np.float32)
            for s in ("CA", "TX", "WI")}
    feats.append(np.stack([snap[s] for s in meta["state_id"].values]))

    X = np.stack(feats, axis=-1).astype(np.float32)        # (T, W+28, F)
    y = S[:, idx].astype(np.float32)
    if target == "log":
        X[..., :len(LAGS) + len(ROLLS)] = np.log1p(
            np.maximum(X[..., :len(LAGS) + len(ROLLS)], 0.0))
        y = np.log1p(y)
    return X, y, meta


def per_task_normalise(X, y, n_tr, tol=1e-4):
    """Normalisation par tache sur la fenetre d'entrainement, comme WEAVE.

    Une variable constante sur la fenetre d'entrainement ne porte aucun signal
    exploitable, mais elle varie parfois dans la fenetre de test. La diviser par
    un plancher minuscule enverrait les entrees a 1e6. On la met a zero.
    """
    xm = X[:, :n_tr].mean(1, keepdims=True)
    xsd = X[:, :n_tr].std(1, keepdims=True)
    dead = xsd < tol
    xs = np.where(dead, 1.0, xsd).astype(np.float32)
    Xn = (X - xm) / xs
    Xn[np.broadcast_to(dead, Xn.shape)] = 0.0

    ym = y[:, :n_tr].mean(1, keepdims=True)
    ysd = y[:, :n_tr].std(1, keepdims=True)
    ys = np.where(ysd < tol, 1.0, ysd).astype(np.float32)
    return Xn, (y - ym) / ys, ym.ravel(), ys.ravel()


def gradient_distance_matrix_rows(Xn, yn, n_tr):
    """g_v = X_v^T y_v normalise, la ligne v du proxy du papier."""
    g = np.einsum("tnf,tn->tf", Xn[:, :n_tr], yn[:, :n_tr])
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)


def prim_chunked(G, root_i):
    """MST euclidien sur les lignes de G, sans former la matrice T x T."""
    T = len(G)
    best = np.full(T, np.inf, dtype=np.float32)
    par = np.full(T, -1, dtype=np.int64)
    used = np.zeros(T, dtype=bool)
    cur = root_i
    used[cur] = True
    edges = []
    for _ in range(T - 1):
        d = np.linalg.norm(G - G[cur], axis=1)
        upd = (~used) & (d < best)
        best[upd] = d[upd]
        par[upd] = cur
        best[used] = np.inf
        cur = int(np.argmin(best))
        used[cur] = True
        edges.append((int(par[cur]), cur))
        best[cur] = np.inf
    return edges


# ---------------------------------------------------------------------------

def evaluate(models, Xn, y_raw, ym, ys, n_tr, nodes):
    errs = []
    with torch.no_grad():
        for v in nodes:
            xb = torch.from_numpy(np.ascontiguousarray(Xn[v, n_tr:])).float()
            p = models[v](xb).numpy().reshape(-1) * ys[v] + ym[v]
            errs.append(np.sqrt(np.mean((y_raw[v, n_tr:] - p) ** 2)))
    return float(np.mean(errs))


def main():
    global METHODS
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=500, help="0 = toutes les series")
    ap.add_argument("--train-days", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--etas", default="0.03,0.1,0.3")
    ap.add_argument("--budgets", default="")
    ap.add_argument("--target", default="log", choices=["log", "raw"])
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()

    t0 = time.perf_counter()
    X, y_raw, meta = load_m5(a.train_days, a.target)
    Tall = len(X)
    print(f"M5 charge: {Tall} series, {X.shape[1]} jours, {X.shape[2]} variables, "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    if a.T and a.T < Tall:
        sel = np.random.default_rng(0).choice(Tall, size=a.T, replace=False)
        sel.sort()
        X, y_raw = X[sel], y_raw[sel]
    T = len(X)
    n_tr = X.shape[1] - N_TEST
    Xn, yn, ym, ys = per_task_normalise(X, y_raw, n_tr)
    nodes = list(range(T))
    d = X.shape[2]

    t0 = time.perf_counter()
    G = gradient_distance_matrix_rows(Xn, yn, n_tr)
    root = int(np.argmin(np.linalg.norm(G - G.mean(0), axis=1)))   # medoide
    mst_edges = prim_chunked(G, root)
    print(f"racine {root}, MST bati en {time.perf_counter()-t0:.0f}s", flush=True)
    trees = {"Star": (root, [(root, v) for v in nodes if v != root]),
             "CTL-MST": (root, mst_edges),
             "CTL-MST-flat": (root, mst_edges)}   # meme arbre, allocation plate

    METHODS = tuple(a.methods.split(","))
    Bs = ([int(b) for b in a.budgets.split(",")] if a.budgets
          else [2 * T, 10 * T])
    Xtr = {v: np.ascontiguousarray(Xn[v, :n_tr]) for v in nodes}
    ytr = {v: np.ascontiguousarray(yn[v, :n_tr]) for v in nodes}

    # Prim renvoie des indices, la distance SLE a besoin d'une fonction d(u,v)
    class LazyD:
        def __getitem__(self, k):
            i, j = k
            return float(np.linalg.norm(G[i] - G[j]))
    D = LazyD()

    ref = float(np.sqrt(((y_raw[:, n_tr:] -
                          y_raw[:, :n_tr].mean(1, keepdims=True)) ** 2)
                        .mean(1)).mean())
    print(f"reference, predicteur constant par serie: RMSE {ref:.4f}", flush=True)

    etas = [float(e) for e in a.etas.split(",")]
    rows = []
    for rep in range(a.reps):
        f = lambda: make_model("mlp", d)
        for B in Bs:
            b = max(1, B // T)
            for eta in etas:
                t0 = time.perf_counter()
                torch.manual_seed(rep)
                models = {v: train_steps(f(), Xtr[v], ytr[v], b, eta,
                                         "regression", 0.0) for v in nodes}
                rows.append(dict(rep=rep, B=B, b=b, eta=eta, method="Individual",
                                 value=evaluate(models, Xn, y_raw, ym, ys,
                                                n_tr, nodes)))
                print(f"  rep={rep} B={B} eta={eta} Individual "
                      f"{rows[-1]['value']:.4f} ({time.perf_counter()-t0:.0f}s)",
                      flush=True)
                for m in METHODS:
                    t0 = time.perf_counter()
                    r_, e_ = trees[m]
                    alloc = (sle_alloc(nodes, r_, e_, D, B) if m in SLE else
                             uniform_alloc(nodes, r_, e_, B, B_PROTO_FRAC))
                    torch.manual_seed(rep)
                    models = cascade_train(e_, r_, Xtr, ytr, alloc, eta,
                                           "regression", 0.0, f)
                    rows.append(dict(rep=rep, B=B, b=b, eta=eta, method=m,
                                     value=evaluate(models, Xn, y_raw, ym, ys,
                                                    n_tr, nodes)))
                    print(f"  rep={rep} B={B} eta={eta} {m:9s} "
                          f"{rows[-1]['value']:.4f} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(HERE / f"m5_{a.target}_T{T}_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"m5_{a.target}_T{T}_raw.csv", index=False)
    print(f"\n=== M5, T={T} series, {'RMSLE' if a.target == 'log' else 'RMSE'} par serie, chaque methode a son eta ===",
          flush=True)
    print(f"reference constante {ref:.4f}", flush=True)
    from scipy import stats as S
    best = df.groupby(["B", "method", "eta"])["value"].mean().reset_index()
    for B, g in best.groupby("B"):
        pick = g.loc[g.groupby("method")["value"].idxmin()].set_index("method")
        b = int(df[df.B == B].b.iloc[0])
        base_eta = pick.loc["Individual", "eta"]
        base = df[(df.B == B) & (df.method == "Individual") &
                  (df.eta == base_eta)].sort_values("rep")["value"].values
        out = []
        for m in METHODS:
            e = pick.loc[m, "eta"]
            cur = df[(df.B == B) & (df.method == m) &
                     (df.eta == e)].sort_values("rep")["value"].values
            p = S.ttest_rel(cur, base).pvalue if len(cur) > 1 else float("nan")
            out.append(f"{m} {cur.mean():.4f} (eta {e}) "
                       f"{100*(cur-base).mean()/base.mean():+.1f}% (p={p:.4f})")
        print(f"B={B:7d} (b={b}): Individual {base.mean():.4f} "
              f"(eta {base_eta}) | " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
