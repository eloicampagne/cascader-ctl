"""
Non-convex replication of the paper's main table.

Re-runs the FULL published pipeline with a non-convex local model (MLP),
faithfully matching the exact protocol reconstructed from the experiment
notebooks (expe-alignment-weave-2, expe-alignment, expe-alignment-classif):

  Allocation : SLE (subtree-log-edge) for the geometry-aware trees
               (MST, MSTc, Chain, kNN); uniform b_seed = b_proto_frac * B for
               Star and RandTree; B // T for Individual.  This is exactly what
               run_experiment_grid / run_sweep_experiment_with_mst_variants do
               with allocation="sle".
  Rooting    : medoid via select_seed_node on the (pre-per-task-normalisation)
               feature means, shared across all methods.
  Distance   : gradient  g_v = X_v^T y_v, for every tree method.
  Budget     : B counts total refinement steps.
  Datasets / per-dataset HPO hyperparameters (from the notebooks):
    Syn-10   T=200  clustered d=3 sigma=5 tau_within=tau_between=10 nc=5
             eta=0.001271  b_proto_frac=0.144  mstc_lambda=0.672
    WEAVE-UK T=500  eta=0.00159  b_proto_frac=0.153  mstc_lambda=0.6
    FMNIST   T=200  clustered binary tasks nc=20, PCA-100
             eta=0.2955 reg=7.9e-3 b_proto_frac=0.10 mstc_lambda=0.0059
    CIFAR-10 T=200  clustered binary tasks nc=20, PCA-100
             eta=0.0671 reg=4.4e-4 b_proto_frac=0.101 mstc_lambda=0.259

  Models     : 'linear' (reproduces the published Table 1 as a reference) and
               'mlp' (d->32->ReLU->1, the non-convex model). CNN is intentionally
               excluded (too slow). CelebA is excluded (no local attribute file;
               frozen-ResNet + linear head stays in the convex-adjacent regime).
               For MLP, eta is RE-CALIBRATED per dataset on the Individual
               baseline (calibrate_mlp()); everything else stays at the paper's
               values, so the CTL-vs-baseline comparison is internal to one
               faithful protocol.

Run:  python3.12 table1_nonconvex.py --exp all
      python3.12 table1_nonconvex.py --exp calibrate      # MLP eta grid
      python3.12 table1_nonconvex.py --exp validate       # linear vs published
"""

import os
import sys
import copy
import warnings
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent.resolve()
CTL_DIR = HERE.parent.parent.parent / "CTL"
sys.path.insert(0, str(CTL_DIR))

import torch
import torch.nn as nn
from data import (generate_clustered_tasks, generate_clustered_binary_tasks,
                  load_weave_data, normalize_xy, denormalize_y,
                  load_fashion_mnist, load_cifar10, standardize)
from graph import (compute_gradient_distance, mst_from_distance_matrix,
                   root_tree, alloc_subtree_log_edge,
                   build_mstc_tree_with_distance, select_seed_node)

DEVICE = torch.device("cpu")
WEAVE_TRAIN = str(CTL_DIR / "data" / "train_weave_500.csv")
WEAVE_TEST = str(CTL_DIR / "data" / "test_weave_500.csv")

# --- per-dataset published config (linear/logistic reference) ---------------
CONFIG = {
    "Syn-10":   dict(eta=0.0012710047170539321, reg=0.0,
                     b_proto_frac=0.14414738903213942, mstc_lambda=0.6720291677813087,
                     task="regression"),
    "WEAVE-UK": dict(eta=0.0015881946767357276, reg=0.0,
                     b_proto_frac=0.1526099724429147, mstc_lambda=0.6,
                     task="regression"),
    "FMNIST":   dict(eta=0.2955513127628392, reg=0.00790085761024867,
                     b_proto_frac=0.10, mstc_lambda=0.005891330018108831,
                     task="classification"),
    "CIFAR":    dict(eta=0.0670885289345722, reg=0.00044369124423313027,
                     b_proto_frac=0.10113798086175203, mstc_lambda=0.2588665932490024,
                     task="classification"),
    # CelebA: frozen ResNet-18 features + head. lr/reg came from an Optuna study
    # (not stored as literals in expe-celeba.ipynb); the linear eta is calibrated
    # on Individual like the MLP one. mstc_lambda=0.196 and SLE are from the notebook.
    "CelebA":   dict(eta=0.2, reg=1e-3,
                     b_proto_frac=0.10, mstc_lambda=0.196,
                     task="classification"),
}
# MLP learning rate, re-calibrated per dataset on Individual (see calibrate_mlp):
# best Individual score at B=1000 (reg) / B=2000 (cls) on the grid {.01,.05,.1,.2,.5}
MLP_ETA = {"Syn-10": 0.2, "WEAVE-UK": 0.2, "FMNIST": 0.2, "CIFAR": 0.1, "CelebA": 0.05}

METHODS = ["Individual", "Star", "RandTree", "Chain",
           "CTL-kNN", "CTL-MST", "CTL-MSTc"]
SLE_METHODS = {"Chain", "CTL-kNN", "CTL-MST", "CTL-MSTc"}   # get SLE allocation

N_SEEDS_IMG = 20      # matches the published n_repeats
N_SEEDS_SYN = 20
N_REPS_WEAVE = 20


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x).reshape(-1)


class MLPHead(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


def make_model(model_class, d):
    if model_class == "linear":
        return LinearHead(d)
    if model_class == "mlp":
        return MLPHead(d, hidden=32)
    raise ValueError(model_class)


# ---------------------------------------------------------------------------
# Training / evaluation (torch, full batch; handles regression + classification)
# ---------------------------------------------------------------------------

def train_steps(model, X, y, steps, eta, task, l2=0.0):
    if steps <= 0:
        return model
    model.train()
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float().reshape(-1)
    crit = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    for _ in range(steps):
        pred = model(X_t)
        loss = crit(pred, y_t)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad if l2 == 0.0 else p.grad + l2 * p
                    p.add_(g, alpha=-eta)
    return model


def eval_task(model, X, y, task, stats=None):
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X).float()).numpy().reshape(-1)
    if task == "classification":
        return float(np.mean((out > 0) == (y.reshape(-1) > 0.5)))
    pred = denormalize_y(out, stats) if stats is not None else out
    return float(np.sqrt(np.mean((y.reshape(-1) - pred.reshape(-1)) ** 2)))


def eval_all(models, X_te, y_te, task, stats=None):
    vals = [eval_task(models[v], X_te[v], y_te[v], task,
                      stats[v] if stats else None) for v in models]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Tree builders (gradient distance D given, rooted at the shared seed)
# ---------------------------------------------------------------------------

def tree_mst(nodes, D, seed):
    return root_tree(mst_from_distance_matrix(nodes, D), seed)


def tree_random(nodes, D, seed, rng):
    """Random recursive tree (parent = uniform earlier index), re-rooted at the
    seed by BFS. Identical to run_experiment_grid's RandTree construction."""
    undirected = [(nodes[rng.integers(0, i)], nodes[i]) for i in range(1, len(nodes))]
    return root_tree(undirected, seed)


def tree_chain(nodes, D, seed):
    """Greedy nearest-neighbour Hamiltonian path from the seed (knn_chain_edges)."""
    idx = {v: i for i, v in enumerate(nodes)}
    cur, visited, edges = seed, {seed}, []
    remaining = set(nodes) - {seed}
    while remaining:
        nxt = min(remaining, key=lambda v: D[idx[cur], idx[v]])
        edges.append((cur, nxt))
        remaining.discard(nxt)
        visited.add(nxt)
        cur = nxt
    return root_tree(edges, seed)


def tree_knn(nodes, D, seed, k=20):
    """BFS spanning tree over the symmetrised k-NN graph from the seed
    (build_knn_tree uses k=20 in the notebooks)."""
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    k = min(k, n - 1)
    nbrs = {v: set() for v in nodes}
    for v in nodes:
        d = D[idx[v]].copy(); d[idx[v]] = np.inf
        for j in np.argsort(d)[:k]:
            nbrs[v].add(nodes[j]); nbrs[nodes[j]].add(v)
    edges, visited, q = [], {seed}, deque([seed])
    while len(visited) < n:
        while q:
            u = q.popleft()
            for w in sorted(nbrs[u], key=lambda x: D[idx[u], idx[x]]):
                if w not in visited:
                    visited.add(w); edges.append((u, w)); q.append(w)
        if len(visited) < n:
            best = None
            for v in set(nodes) - visited:
                for u in visited:
                    dd = D[idx[u], idx[v]]
                    if best is None or dd < best[0]:
                        best = (dd, u, v)
            _, u, v = best
            visited.add(v); edges.append((u, v)); q.append(v)
    return root_tree(edges, seed)


def tree_mstc(nodes, D, seed, X, y, lam):
    T = len(nodes)
    tau = float(np.median(D[np.triu_indices(T, k=1)]))
    root, parent_map, dir_edges, _ = build_mstc_tree_with_distance(
        V=nodes, y=y, X=X, distance_type="gradient",
        seed=seed, lambda_reach=lam, tau=tau)
    return root, dir_edges, parent_map


# ---------------------------------------------------------------------------
# Cascade training under an explicit per-node budget allocation
# ---------------------------------------------------------------------------

def cascade_train(dir_edges, root, X_tr, y_tr, alloc, eta, task, l2, factory):
    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    m0 = train_steps(factory(), X_tr[root], y_tr[root],
                     int(max(0, alloc.get(root, 0))), eta, task, l2)
    models = {root: m0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            m = train_steps(copy.deepcopy(models[u]), X_tr[v], y_tr[v],
                            int(max(0, alloc.get(v, 0))), eta, task, l2)
            models[v] = m
            q.append(v)
    return models


def uniform_alloc(nodes, root, dir_edges, B, b_proto_frac):
    """Uniform split over non-root nodes, summing to exactly B.

    The previous version floored the per-child budget and then clamped it up to
    one, which both overspent B at tight budget and underspent it otherwise.
    """
    others = [v for v in nodes if v != root]
    if not others:
        return {root: B}
    T = len(nodes)
    if ALLOC_MODE == "floor1" and B >= T:
        # same floor as sle_alloc: one step everywhere, then split the surplus
        surplus = B - T
        seed_steps = min(max(0, int(b_proto_frac * surplus)), surplus)
        base, extra = divmod(surplus - seed_steps, len(others))
        alloc = {v: 1 + base + (1 if i < extra else 0)
                 for i, v in enumerate(others)}
        alloc[root] = 1 + seed_steps
        return alloc
    seed_steps = min(max(1, int(b_proto_frac * B)), B)
    remaining = max(0, B - seed_steps)
    base, extra = divmod(remaining, len(others))
    alloc = {v: base + (1 if i < extra else 0) for i, v in enumerate(others)}
    alloc[root] = seed_steps
    return alloc


#: allocation regime, selected by the CTL_ALLOC environment variable.
#:   "exact"  (default, scheme A) apportion b_v proportional to the SLE weights,
#:            summing to exactly B. A node whose share rounds to zero receives no
#:            refinement and inherits its parent unchanged.
#:   "floor1" (scheme B) every task is refined at least once; the surplus B - T
#:            is apportioned by the SLE weights. Requires B >= T, otherwise falls
#:            back to "exact". Both sum to exactly B.
ALLOC_MODE = os.environ.get("CTL_ALLOC", "floor1")


def sle_alloc(nodes, root, dir_edges, D, B):
    """SLE allocation; the floor1/exact regime is handled in graph._alloc_from_weights."""
    nidx = {v: i for i, v in enumerate(nodes)}
    pmap = {v: u for u, v in dir_edges}
    return alloc_subtree_log_edge(nodes, root, pmap, D, nidx, B)


def run_method(method, nodes, X_tr, y_tr, X_prox, y_prox, D, seed, B, eta,
               cfg, factory, rng):
    task, l2, frac, lam = cfg["task"], cfg["reg"], cfg["b_proto_frac"], cfg["mstc_lambda"]
    T = len(nodes)

    if method == "Individual":
        # exactly B steps in total: B//T each, the first B%T tasks get one more
        base, extra = divmod(B, T)
        return {v: train_steps(factory(), X_tr[v], y_tr[v],
                               base + (1 if i < extra else 0), eta, task, l2)
                for i, v in enumerate(nodes)}

    if method == "Star":
        dir_edges = [(seed, v) for v in nodes if v != seed]
        root = seed
    elif method == "RandTree":
        root, dir_edges, _ = tree_random(nodes, D, seed, rng)
    elif method == "Chain":
        root, dir_edges, _ = tree_chain(nodes, D, seed)
    elif method == "CTL-kNN":
        root, dir_edges, _ = tree_knn(nodes, D, seed)
    elif method == "CTL-MST":
        root, dir_edges, _ = tree_mst(nodes, D, seed)
    elif method == "CTL-MSTc":
        root, dir_edges, _ = tree_mstc(nodes, D, seed, X_prox, y_prox, lam)
    else:
        raise ValueError(method)

    if method in SLE_METHODS:
        alloc = sle_alloc(nodes, root, dir_edges, D, B)
    else:                                   # Star, RandTree -> uniform b_seed
        alloc = uniform_alloc(nodes, root, dir_edges, B, frac)
    return cascade_train(dir_edges, root, X_tr, y_tr, alloc, eta, task, l2, factory)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_IMG_CACHE = {}

def _image_pca(dataset):
    """standardise -> PCA-100 basis on full training set (cached)."""
    if dataset in _IMG_CACHE:
        return _IMG_CACHE[dataset]
    if dataset == "FMNIST":
        Xtr, ytr, Xte, yte = load_fashion_mnist(str(CTL_DIR / "data" / "fashion_mnist"))
    else:
        Xtr, ytr, Xte, yte = load_cifar10(str(CTL_DIR / "data" / "cifar10"))
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32)
    Xte = Xte.reshape(len(Xte), -1).astype(np.float32)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, svd_solver="randomized", random_state=0)
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    Xte_p = pca.transform(Xte_s).astype(np.float32)
    _IMG_CACHE[dataset] = (Xtr_p, ytr, Xte_p, yte)
    return _IMG_CACHE[dataset]


def image_tasks(dataset, seed, T=200, n_clusters=20, n_tr=64, n_te=128):
    """Clustered binary tasks on PCA features (matches the notebooks)."""
    Xtr_p, ytr, Xte_p, yte = _image_pca(dataset)
    rng = np.random.default_rng(seed)
    Xt, yt, Xe, ye, _, _ = generate_clustered_binary_tasks(
        Xtr_p, ytr, Xte_p, yte, n_tasks=T, n_clusters=n_clusters,
        n_train_per_task=n_tr, n_test_per_task=n_te, rng=rng)
    nodes = list(range(T))
    X_tr = {v: Xt[v] for v in nodes}
    y_tr = {v: yt[v] for v in nodes}
    X_te = {v: Xe[v] for v in nodes}
    y_te = {v: ye[v] for v in nodes}
    return nodes, X_tr, y_tr, X_te, y_te


_CELEBA = {}

def _celeba_cache():
    if "c" not in _CELEBA:
        c = HERE / "celeba_cache"
        _CELEBA["c"] = (np.load(c / "X_train.npy"), np.load(c / "y_train.npy"),
                        np.load(c / "X_test.npy"), np.load(c / "y_test.npy"))
    return _CELEBA["c"]


def celeba_tasks(seed, n_tasks=400, n_clusters=5, n_tr=64, n_te=128):
    """Clustered binary attribute-pair tasks on frozen ResNet-18 features
    (ports generate_clustered_binary_attribute_tasks from expe-celeba.ipynb:
    each task discriminates attr_i=1 vs attr_j=1, clusters share a base pair)."""
    Xtr, ytr, Xte, yte = _celeba_cache()
    rng = np.random.default_rng(seed)
    n_attrs = ytr.shape[1]
    all_pairs = [(i, j) for i in range(n_attrs) for j in range(i + 1, n_attrs)]
    cidx = rng.choice(len(all_pairs), size=n_clusters, replace=False)
    cluster_pairs = [all_pairs[i] for i in cidx]
    per = n_tasks // n_clusters
    cluster_ids = np.repeat(np.arange(n_clusters), per)
    if len(cluster_ids) < n_tasks:
        cluster_ids = np.concatenate([cluster_ids,
                                      rng.choice(n_clusters, size=n_tasks - len(cluster_ids))])
    rng.shuffle(cluster_ids)

    def sample(Xs, ys, ai, aj, npc):
        mi, mj = ys[:, ai] == 1, ys[:, aj] == 1
        idx_i = np.where(mi & ~(mi & mj))[0]
        idx_j = np.where(mj & ~(mi & mj))[0]
        if len(idx_i) < npc or len(idx_j) < npc:
            return None
        si = rng.choice(idx_i, size=npc, replace=False)
        sj = rng.choice(idx_j, size=npc, replace=False)
        X = np.vstack([Xs[si], Xs[sj]]).astype(np.float32)
        y = np.hstack([np.ones(npc), np.zeros(npc)]).astype(np.float32)
        return X, y

    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    v = 0
    for cid in cluster_ids:
        ai, aj = cluster_pairs[cid]
        tr = sample(Xtr, ytr, ai, aj, n_tr // 2)
        te = sample(Xte, yte, ai, aj, n_te // 2)
        if tr is None or te is None:
            continue
        X_tr[v], y_tr[v] = tr
        X_te[v], y_te[v] = te
        v += 1
    nodes = list(X_tr.keys())
    return nodes, X_tr, y_tr, X_te, y_te


_WEAVE = {}

def weave_tasks():
    if "w" not in _WEAVE:
        Xtr_raw, ytr_raw = load_weave_data(WEAVE_TRAIN)
        Xte_raw, yte_raw = load_weave_data(WEAVE_TEST)
        nodes = sorted(set(Xtr_raw) & set(Xte_raw))
        Xn_tr, yn_tr, Xn_te, stats = {}, {}, {}, {}
        for v in nodes:
            Xn_tr[v], yn_tr[v], st = normalize_xy(Xtr_raw[v], ytr_raw[v])
            stats[v] = st
            Xn_te[v] = (Xte_raw[v] - st["X_mean"]) / st["X_std"]
        D = compute_gradient_distance(nodes, Xn_tr, yn_tr)
        seed = select_seed_node({v: Xtr_raw[v] for v in nodes},
                                {v: ytr_raw[v] for v in nodes})   # raw-feature medoid
        _WEAVE["w"] = (nodes, Xn_tr, yn_tr, Xn_te,
                       {v: yte_raw[v] for v in nodes}, stats, D, seed)
    return _WEAVE["w"]


def syn_tasks(seed, T=200, d=3, K=5, tau=10.0, tau_b=10.0, sigma=5.0,
              n_tr=64, n_te=128):
    rng = np.random.default_rng(seed)
    _, _, _, (Xt, yt), (Xe, ye), _ = generate_clustered_tasks(
        T=T, d=d, n_train=n_tr, n_test=n_te, n_clusters=K,
        tau_within=tau, tau_between=tau_b, sigma=sigma, rng=rng)
    nodes = list(range(T))
    X_raw = {v: Xt[v] for v in nodes}
    Xn_tr, yn_tr, Xn_te, stats = {}, {}, {}, {}
    for v in nodes:
        Xn_tr[v], yn_tr[v], st = normalize_xy(Xt[v], yt[v])
        stats[v] = st
        Xn_te[v] = (Xe[v] - st["X_mean"]) / st["X_std"]
    D = compute_gradient_distance(nodes, Xn_tr, yn_tr)
    root = select_seed_node(X_raw, {v: yt[v] for v in nodes})
    return nodes, Xn_tr, yn_tr, Xn_te, {v: ye[v] for v in nodes}, stats, D, root


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _save(rows, stem):
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"{stem}_raw.csv", index=False)
    (df.groupby(["dataset", "model", "B", "method"])["value"]
       .agg(["mean", "std", "count"]).reset_index()
       .to_csv(HERE / f"{stem}_summary.csv", index=False))


def run_regression(dataset, loader, model_classes=("linear", "mlp"),
                   Bs=(500, 1000, 2000), n_reps=5, stem=None):
    cfg = CONFIG[dataset]
    rows = []
    for rep in range(n_reps):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, seed = loader(rep)
        d = Xn_tr[nodes[0]].shape[1]
        rng = np.random.default_rng(20_000 + rep)
        for mc in model_classes:
            torch.manual_seed(rep)
            eta = cfg["eta"] if mc == "linear" else MLP_ETA[dataset]
            factory = lambda: make_model(mc, d)
            for B in Bs:
                for method in METHODS:
                    models = run_method(method, nodes, Xn_tr, yn_tr, Xn_tr, yn_tr,
                                        D, seed, B, eta, cfg, factory, rng)
                    val = eval_all(models, Xn_te, y_te, "regression", stats)
                    rows.append(dict(dataset=dataset, model=mc, B=B, seed=rep,
                                     method=method, value=val))
                print(f"  {dataset} {mc} rep={rep} B={B}: " + " ".join(
                    f"{r['method']}={r['value']:.0f}" for r in rows[-7:]), flush=True)
        _save(rows, stem or f"t1nc_{dataset.lower().replace('-','')}")
    return rows


def run_classification(dataset, model_classes=("linear", "mlp"),
                       Bs=(500, 2000), n_seeds=N_SEEDS_IMG, task_loader=None):
    cfg = CONFIG[dataset]
    rows = []
    for seed in range(n_seeds):
        if task_loader is not None:
            nodes, X_tr, y_tr, X_te, y_te = task_loader(seed)
        else:
            nodes, X_tr, y_tr, X_te, y_te = image_tasks(dataset, seed)
        D = compute_gradient_distance(nodes, X_tr, y_tr)
        root = select_seed_node(X_tr, y_tr)
        rng = np.random.default_rng(10_000 + seed)
        d = X_tr[nodes[0]].shape[1]
        for mc in model_classes:
            torch.manual_seed(seed)
            eta = cfg["eta"] if mc == "linear" else MLP_ETA[dataset]
            factory = lambda: make_model(mc, d)
            for B in Bs:
                for method in METHODS:
                    models = run_method(method, nodes, X_tr, y_tr, X_tr, y_tr,
                                        D, root, B, eta, cfg, factory, rng)
                    acc = eval_all(models, X_te, y_te, "classification")
                    rows.append(dict(dataset=dataset, model=mc, B=B, seed=seed,
                                     method=method, value=acc))
            done = [r for r in rows if r["seed"] == seed and r["model"] == mc]
            print(f"  {dataset} {mc} seed={seed}: " + " ".join(
                f"{r['method']}@{r['B']}={r['value']:.3f}" for r in done[-7:]),
                flush=True)
        _save(rows, f"t1nc_{dataset.lower()}")
    return rows


# ---------------------------------------------------------------------------
# Validation against published Table 1 (linear only) + MLP eta calibration
# ---------------------------------------------------------------------------

def validate():
    print("=== VALIDATION: linear reference vs published Table 1 ===")
    print("WEAVE published B=500/1000/2000: "
          "Chain 1474/1306/1203  MST 1450/1299/1210  MSTc 1346/1232/1168")
    run_regression("WEAVE-UK", lambda r: weave_tasks(),
                   model_classes=("linear",), Bs=(500, 1000, 2000), n_reps=1,
                   stem="t1nc_validate_weave")
    print("\nFMNIST published B=500/2000 (acc): Chain .931/.957 MST .931/.951 MSTc .933/.960")
    run_classification("FMNIST", model_classes=("linear",), Bs=(500, 2000), n_seeds=2)
    print("\nCIFAR published B=500/2000 (acc): Chain .634/.669 MST .657/.691 MSTc .638/.673")
    run_classification("CIFAR", model_classes=("linear",), Bs=(500, 2000), n_seeds=2)


def calibrate_mlp(etas=(0.01, 0.05, 0.1, 0.2, 0.5)):
    print("=== MLP eta calibration on Individual ===")
    # regression
    for dataset, loader in (("WEAVE-UK", lambda: weave_tasks()),
                            ("Syn-10", lambda: syn_tasks(0))):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, seed = loader()
        d = Xn_tr[nodes[0]].shape[1]
        cfg = CONFIG[dataset]; B = 1000
        for eta in etas:
            torch.manual_seed(0)
            b = max(1, B // len(nodes))
            models = {v: train_steps(make_model("mlp", d), Xn_tr[v], yn_tr[v],
                                     b, eta, "regression", 0.0) for v in nodes}
            print(f"  {dataset} eta={eta}: Individual RMSE={eval_all(models, Xn_te, y_te, 'regression', stats):.0f}")
    # classification
    for dataset in ("FMNIST", "CIFAR"):
        nodes, X_tr, y_tr, X_te, y_te = image_tasks(dataset, 0)
        d = X_tr[nodes[0]].shape[1]; cfg = CONFIG[dataset]; B = 2000
        for eta in etas:
            torch.manual_seed(0)
            b = max(1, B // len(nodes))
            models = {v: train_steps(make_model("mlp", d), X_tr[v], y_tr[v],
                                     b, eta, "classification", cfg["reg"]) for v in nodes}
            print(f"  {dataset} eta={eta}: Individual acc={eval_all(models, X_te, y_te, 'classification'):.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all")
    args = parser.parse_args()
    exps = set(args.exp.split(",")) if args.exp != "all" else \
        {"syn", "weave", "fmnist", "cifar", "celeba"}

    if "calibrate" in exps:
        calibrate_mlp()
    if "validate" in exps:
        validate()
    if "syn" in exps:
        print("=== Syn-10 ===")
        run_regression("Syn-10", lambda r: syn_tasks(r),
                       Bs=(500, 1000, 2000), n_reps=N_SEEDS_SYN, stem="t1nc_syn10")
    if "weave" in exps:
        print("=== WEAVE-UK ===")
        run_regression("WEAVE-UK", lambda r: weave_tasks(),
                       Bs=(500, 1000, 2000), n_reps=N_REPS_WEAVE, stem="t1nc_weave")
    if "fmnist" in exps:
        print("=== Fashion-MNIST ===")
        run_classification("FMNIST")
    if "cifar" in exps:
        print("=== CIFAR-10 ===")
        run_classification("CIFAR")
    if "celeba" in exps:
        print("=== CelebA ===")
        run_classification("CelebA", Bs=(2000, 4000), n_seeds=N_SEEDS_IMG,
                           task_loader=celeba_tasks)
    print("=== Done ===")
