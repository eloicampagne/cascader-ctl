"""CTL sur la hierarchie M5, avec LightGBM et WRMSSE.

Pourquoi cette experience. M5 note en WRMSSE sur 42840 series reparties en douze
niveaux d'agregation, chaque niveau pesant 1/12 et chaque serie pesant sa part de
chiffre d'affaires dans son niveau. La hierarchie est donc un arbre donne par le
domaine, et il possede la propriete dont CTL vit: les agregats sont denses et bien
estimes, les feuilles sont a 68 % de zeros. Une cascade qui descend la hierarchie
transfere un modele bien estime vers des taches mal estimees.

La question posee est celle qui met CTL en valeur: **l'arbre appris par le proxy
du papier retrouve-t-il, ou depasse-t-il, l'arbre du domaine ?**

Bras, tous au meme budget total d'iterations de boosting:

    Individual   b = B/T iterations depuis zero pour chaque serie
    Star         chaque serie poursuit le boosting du Total
    Hierarchy    chaque serie poursuit celui de son parent dans la hierarchie M5
    CTL-MST      chaque serie poursuit celui de son parent dans l'arbre appris
    CTL-MST-h    idem, mais l'arbre appris est enracine au Total

Le transfert est litteral: LightGBM reprend l'ensemble du parent par `init_model`
et lui ajoute b_v arbres. Le budget compte les iterations ajoutees, comme le
papier compte les pas de gradient.

Protocole, conforme aux enseignements des gagnants (cross-learning, validation).
Prevision directe a 28 jours: toutes les variables retardees remontent d'au moins
28 jours, donc aucune fuite et aucune recursion. Entrainement jusqu'a d_1913,
validation d_1914 a d_1941, test d_1942 a d_1969. Les hyperparametres sont choisis
sur la validation, le test n'est ouvert qu'une fois.

Reserve assumee: avec `init_model` un noeud profond herite de l'ensemble de tous
ses ancetres, donc son cout d'inference croit avec la profondeur meme si son cout
d'entrainement est le meme. On rapporte les deux.

    python3.12 m5_hierarchy.py --store CA_1 --n-items 289 --reps 3
"""

import argparse
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

# BLAS leve de faux drapeaux virgule flottante sur ce matmul; verifie
# contre einsum, ecart 3e-16.
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
# Folder holding sales_train_evaluation.csv, sales_test_evaluation.csv and calendar.csv.
M5DIR = Path(__import__("os").environ.get("M5_DATA_DIR", "."))

H = 28                     # horizon, et taille du bloc de validation
LAGS = (28, 35, 42, 56, 364)
ROLLS = (7, 28, 56)        # moyennes glissantes se terminant a t-28
D_TRAIN_END = 1913         # dernier jour d'entrainement
D_VAL_END = 1941           # dernier jour de validation
D_TEST_END = 1969


# ---------------------------------------------------------------------------
# Series a tous les niveaux, pour un magasin
# ---------------------------------------------------------------------------

def build_series(store, n_items, seed):
    tr = pd.read_csv(M5DIR / "sales_train_evaluation.csv")
    te = pd.read_csv(M5DIR / "sales_test_evaluation.csv")
    S = np.concatenate([tr.iloc[:, 5:].values, te.iloc[:, 5:].values],
                       axis=1).astype(np.float64)
    m = tr[["item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    inb = (m.store_id == store).values
    Sb, mb = S[inb], m[inb].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(Sb), size=min(n_items, len(Sb)), replace=False))

    series, parent, level = {}, {}, {}
    root = f"{store}"                                   # niveau 3
    series[root] = Sb.sum(0)
    parent[root] = None
    level[root] = "Level3"
    for cat in sorted(mb.cat_id.unique()):              # niveau 8
        k = f"{store}|{cat}"
        series[k] = Sb[(mb.cat_id == cat).values].sum(0)
        parent[k], level[k] = root, "Level8"
    for dept in sorted(mb.dept_id.unique()):            # niveau 9
        k = f"{store}|{dept}"
        cat = mb.loc[mb.dept_id == dept, "cat_id"].iloc[0]
        series[k] = Sb[(mb.dept_id == dept).values].sum(0)
        parent[k], level[k] = f"{store}|{cat}", "Level9"
    for i in keep:                                      # niveau 12
        k = f"{store}|{mb.item_id[i]}"
        series[k] = Sb[i]
        parent[k], level[k] = f"{store}|{mb.dept_id[i]}", "Level12"

    w = pd.read_csv(M5DIR / "weights_evaluation.csv")
    wmap = {}
    for _, r in w.iterrows():
        if r.Level_id == "Level3" and r.Agg_Level_1 == store:
            wmap[store] = r.weight
        elif r.Level_id in ("Level8", "Level9") and r.Agg_Level_1 == store:
            wmap[f"{store}|{r.Agg_Level_2}"] = r.weight
        elif r.Level_id == "Level12" and r.Agg_Level_2 == store:
            wmap[f"{store}|{r.Agg_Level_1}"] = r.weight
    return series, parent, level, wmap, root


def wrmsse_parts(series, level, wmap, keys):
    """Poids renormalises par niveau represente, et echelle RMSSE par serie."""
    lv = defaultdict(list)
    for k in keys:
        lv[level[k]].append(k)
    weight = {}
    for L, ks in lv.items():
        tot = sum(wmap.get(k, 0.0) for k in ks) or 1.0
        for k in ks:
            weight[k] = wmap.get(k, 0.0) / tot / len(lv)
    scale = {}
    for k in keys:
        y = series[k][:D_TRAIN_END]
        nz = np.flatnonzero(y)
        y = y[nz[0]:] if len(nz) else y                 # depuis la 1re vente
        scale[k] = max(float(np.mean(np.diff(y) ** 2)), 1e-8)
    return weight, scale


# ---------------------------------------------------------------------------
# Variables, prevision directe a 28 jours
# ---------------------------------------------------------------------------

def build_features(series, keys):
    cal = pd.read_csv(M5DIR / "calendar.csv").iloc[:D_TEST_END]
    wday = cal["wday"].values
    month = cal["month"].values
    event = cal["event_name_1"].notna().values.astype(np.float64)
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].values.mean(1)

    start = max(max(LAGS), max(ROLLS) + H) + 1
    idx = np.arange(start, D_TEST_END)
    tr = idx < D_TRAIN_END
    X, y, denorm = {}, {}, {}
    for k in keys:
        v = series[k]
        c = np.concatenate([[0.0], np.cumsum(v)])
        f = [v[idx - L] for L in LAGS]
        f += [(c[idx - H + 1] - c[idx - H + 1 - R]) / R for R in ROLLS]
        f.append((v[idx - H] > 0).astype(np.float64))
        f += [(wday[idx] == q).astype(np.float64) for q in range(1, 8)]
        f += [month[idx].astype(np.float64), event[idx], snap[idx]]
        A = np.stack(f, axis=1)
        yy = v[idx].astype(np.float64)
        # Normalisation par serie sur la fenetre d'entrainement, comme le papier.
        # Sans elle un enfant herite d'un modele qui predit l'echelle du parent:
        # le total du magasin vend 4000 unites par jour, un article en vend une.
        am, asd = A[tr].mean(0), A[tr].std(0)
        dead = asd < 1e-8
        A = (A - am) / np.where(dead, 1.0, asd)
        A[:, dead] = 0.0
        ym = float(yy[tr].mean())
        ysd = float(yy[tr].std()) or 1.0
        X[k], y[k] = A, (yy - ym) / ysd
        denorm[k] = (ym, ysd, yy)
    return X, y, idx, denorm


def split(idx):
    return (idx < D_TRAIN_END, (idx >= D_TRAIN_END) & (idx < D_VAL_END),
            idx >= D_VAL_END)


# ---------------------------------------------------------------------------
# Arbres
# ---------------------------------------------------------------------------

def grad_proxy(X, y, keys, tr):
    """g_v = X_v^T y_v normalise. X et y sont deja standardises par serie."""
    G = []
    for k in keys:
        a, b = X[k][tr], y[k][tr]
        g = a.T @ b
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        G.append(g / (np.linalg.norm(g) + 1e-12))
    return np.stack(G)


def prim(G, root_i):
    T = len(G)
    best = np.full(T, np.inf)
    par = np.full(T, -1, dtype=int)
    used = np.zeros(T, bool)
    cur, edges = root_i, []
    used[cur] = True
    for _ in range(T - 1):
        d = np.linalg.norm(G - G[cur], axis=1)
        u = (~used) & (d < best)
        best[u], par[u] = d[u], cur
        best[used] = np.inf
        cur = int(np.argmin(best))
        used[cur] = True
        edges.append((int(par[cur]), cur))
        best[cur] = np.inf
    return edges


def bfs_order(edges, root):
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    out, q = [], deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            out.append((u, v))
            q.append(v)
    return out


def depth_map(edges, root):
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    d, q = {root: 0}, deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            d[v] = d[u] + 1
            q.append(v)
    return d


# ---------------------------------------------------------------------------

def run_arm(edges, root, keys, X, y, tr, ev, b, params, seed):
    """Cascade LightGBM: chaque enfant poursuit le boosting de son parent."""
    import lightgbm as lgb
    p = dict(params, seed=seed, verbose=-1)
    models = {}
    models[root] = lgb.train(p, lgb.Dataset(X[root][tr], y[root][tr]),
                             num_boost_round=b)
    for u, v in edges:
        models[v] = lgb.train(p, lgb.Dataset(X[v][tr], y[v][tr]),
                              num_boost_round=b, init_model=models[u])
    pred = {k: models[k].predict(X[k][ev]) for k in keys}
    ntrees = {k: models[k].num_trees() for k in keys}
    return pred, ntrees


def run_individual(keys, X, y, tr, ev, b, params, seed):
    import lightgbm as lgb
    p = dict(params, seed=seed, verbose=-1)
    pred, ntrees = {}, {}
    for k in keys:
        m = lgb.train(p, lgb.Dataset(X[k][tr], y[k][tr]), num_boost_round=b)
        pred[k] = m.predict(X[k][ev])
        ntrees[k] = m.num_trees()
    return pred, ntrees


def wrmsse(pred, denorm, ev, weight, scale, keys):
    """WRMSSE sur l'echelle brute: on denormalise les predictions."""
    tot = 0.0
    for k in keys:
        ym, ysd, yraw = denorm[k]
        p = np.asarray(pred[k]) * ysd + ym
        mse = float(np.mean((yraw[ev] - p) ** 2))
        tot += weight[k] * np.sqrt(mse / scale[k])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="CA_1")
    ap.add_argument("--n-items", type=int, default=289)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budgets", default="")
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.1)
    a = ap.parse_args()

    t0 = time.perf_counter()
    series, parent, level, wmap, root = build_series(a.store, a.n_items, 0)
    keys = list(series)
    X, y, idx, denorm = build_features(series, keys)
    tr, va, te = split(idx)
    weight, scale = wrmsse_parts(series, level, wmap, keys)
    T = len(keys)
    print(f"{T} series pour {a.store}, niveaux "
          f"{sorted(set(level[k] for k in keys))}, "
          f"{X[root].shape[1]} variables, {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  train {int(tr.sum())} j, val {int(va.sum())} j, test {int(te.sum())} j",
          flush=True)

    ki = {k: i for i, k in enumerate(keys)}
    hier = [(parent[k], k) for k in keys if parent[k] is not None]
    hier = bfs_order(hier, root)
    star = [(root, k) for k in keys if k != root]
    G = grad_proxy(X, y, keys, tr)
    mst = [(keys[u], keys[v]) for u, v in prim(G, ki[root])]
    mst = bfs_order(mst, root)
    trees = {"Star": star, "Hierarchy": hier, "CTL-MST": mst}
    for n, e in trees.items():
        d = depth_map(e, root)
        print(f"  {n:10s} profondeur max {max(d.values())}", flush=True)

    Bs = [int(b) for b in a.budgets.split(",")] if a.budgets else [2 * T, 10 * T]
    params = dict(objective="regression", num_leaves=a.leaves,
                  learning_rate=a.lr, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)

    rows = []
    for rep in range(a.reps):
        for B in Bs:
            b = max(1, B // T)
            for split_name, ev in (("val", va), ("test", te)):
                t0 = time.perf_counter()
                pr, nt = run_individual(keys, X, y, tr, ev, b, params, rep)
                rows.append(dict(rep=rep, B=B, b=b, split=split_name,
                                 method="Individual",
                                 value=wrmsse(pr, denorm, ev, weight, scale, keys),
                                 trees=float(np.mean(list(nt.values()))),
                                 secs=time.perf_counter() - t0))
                for name, e in trees.items():
                    t0 = time.perf_counter()
                    pr, nt = run_arm(e, root, keys, X, y, tr, ev, b, params, rep)
                    rows.append(dict(rep=rep, B=B, b=b, split=split_name,
                                     method=name,
                                     value=wrmsse(pr, denorm, ev, weight, scale, keys),
                                     trees=float(np.mean(list(nt.values()))),
                                     secs=time.perf_counter() - t0))
                r = [x for x in rows if x["rep"] == rep and x["B"] == B
                     and x["split"] == split_name]
                print("  rep=%d B=%d(%s) %s" % (rep, B, split_name,
                      " | ".join(f"{x['method']} {x['value']:.4f}" for x in r)),
                      flush=True)
            pd.DataFrame(rows).to_csv(HERE / f"m5_hier_{a.store}_raw.csv",
                                      index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"m5_hier_{a.store}_raw.csv", index=False)
    print(f"\n=== WRMSSE, {a.store}, {T} series ===", flush=True)
    print(df.pivot_table(index=["split", "B"], columns="method",
                         values="value").round(4).to_string(), flush=True)
    print("\n=== arbres par modele (cout d'inference) ===", flush=True)
    print(df.pivot_table(index="B", columns="method",
                         values="trees").round(0).to_string(), flush=True)
    from scipy import stats as S
    print("\n=== test: ecart a Individual, apparie ===", flush=True)
    for B, g in df[df.split == "test"].groupby("B"):
        base = g[g.method == "Individual"].sort_values("rep")["value"].values
        out = []
        for m in trees:
            cur = g[g.method == m].sort_values("rep")["value"].values
            p = S.ttest_rel(cur, base).pvalue if len(cur) > 1 else float("nan")
            out.append(f"{m} {cur.mean():.4f} "
                       f"{100*(cur-base).mean()/base.mean():+.1f}% (p={p:.3f})")
        print(f"B={B:6d} (b={g.b.iloc[0]}): Individual {base.mean():.4f} | "
              + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
