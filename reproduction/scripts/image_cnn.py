"""
Convolutional local model on the image collections.

The MLP head on PCA-100 features is a generic non-convex model, not one adapted
to images: PCA destroys the spatial layout a convolution exploits. This script
replaces the local model by a small CNN acting on the raw image, keeping every
other element of the protocol identical (same clustered binary tasks, same
gradient proxy computed on PCA features, same SLE allocation, same budget in
gradient steps, same eta calibration on the Individual baseline).

Two modes.

  diag   non-linearity headroom with an adapted model. Per task, the exact
         linear reference (logistic regression, on PCA-100 and on raw pixels,
         best of the two) against a CNN trained to convergence. Answers whether
         the image data carry non-linear structure that the MLP simply could
         not see.

  table  full method comparison (Individual, Star, RandTree, Chain, CTL-MST,
         CTL-MSTc) with the CNN as local model, budgets 500 and 2000.

Run: python3.12 image_cnn.py diag | python3.12 image_cnn.py table [--reps N]
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    CTL_DIR, CONFIG, make_model, run_method, eval_all, train_steps,
    compute_gradient_distance, select_seed_node, _image_pca,
)
from data import (generate_clustered_binary_tasks, load_fashion_mnist,
                  load_cifar10, standardize)

DATASETS = ("FMNIST", "CIFAR", "CelebA")
SHAPE = {"FMNIST": (1, 28, 28), "CIFAR": (3, 32, 32), "CelebA": (3, 64, 64)}
BUDGETS = {"FMNIST": (500, 2000), "CIFAR": (500, 2000), "CelebA": (2000, 4000)}
ETA_GRID = (0.01, 0.05, 0.1, 0.2, 0.5)
METHODS = ["Individual", "Star", "RandTree", "Chain", "CTL-MST", "CTL-MSTc"]
CHANNELS = (8, 16, 32)


# ---------------------------------------------------------------------------
# Model: conv blocks down to an 8x8 map, then a linear read-out. Two blocks on
# 28x28 and 32x32 (2.0k and 2.4k parameters, against 3.3k for the MLP head),
# three on 64x64 (8.1k, against 16.4k for the MLP head on ResNet features).
# The convolutional model is the smaller one in every case, so the comparison
# is about inductive bias rather than capacity.
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, shape):
        super().__init__()
        c, h, w = shape
        n_blocks = 3 if h > 40 else 2
        layers, cin = [], c
        for i in range(n_blocks):
            layers += [nn.Conv2d(cin, CHANNELS[i], 3, padding=1), nn.ReLU(),
                       nn.MaxPool2d(2)]
            cin = CHANNELS[i]
        self.features = nn.Sequential(*layers)
        red = 2 ** n_blocks
        self.head = nn.Linear(cin * (h // red) * (w // red), 1)

    def forward(self, x):
        z = self.features(x)
        return self.head(z.flatten(1)).reshape(-1)


# ---------------------------------------------------------------------------
# Raw-pixel tasks, identical in composition to the PCA ones
# ---------------------------------------------------------------------------

_RAW = {}

def _image_raw(dataset):
    """standardise the flattened pixels, keep the image shape."""
    if dataset in _RAW:
        return _RAW[dataset]
    if dataset == "FMNIST":
        Xtr, ytr, Xte, yte = load_fashion_mnist(str(CTL_DIR / "data" / "fashion_mnist"))
    else:
        Xtr, ytr, Xte, yte = load_cifar10(str(CTL_DIR / "data" / "cifar10"))
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32)
    Xte = Xte.reshape(len(Xte), -1).astype(np.float32)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    _RAW[dataset] = (Xtr_s.astype(np.float32), ytr, Xte_s.astype(np.float32), yte)
    return _RAW[dataset]


_CELEBA_RAW = {}

def _celeba_raw():
    """standardise the 64x64 images pixelwise on the training split."""
    if "c" not in _CELEBA_RAW:
        c = HERE / "celeba_cache"
        Xtr = np.load(c / "Ximg_train.npy").astype(np.float32) / 255.0
        Xte = np.load(c / "Ximg_test.npy").astype(np.float32) / 255.0
        mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
        _CELEBA_RAW["c"] = ((Xtr - mean) / std, np.load(c / "y_train.npy"),
                            (Xte - mean) / std, np.load(c / "y_test.npy"))
    return _CELEBA_RAW["c"]


def celeba_tasks_raw(seed):
    """Same attribute-pair tasks as celeba_tasks, on the images rather than the
    frozen ResNet-18 features. The sampler draws indices from the attribute
    labels and the rng only, so swapping the cached array behind it keeps the
    task composition identical."""
    import table1_nonconvex as t1
    from table1_nonconvex import celeba_tasks
    Xtr, ytr, Xte, yte = _celeba_raw()
    saved = t1._CELEBA.get("c")
    t1._CELEBA["c"] = (Xtr, ytr, Xte, yte)
    try:
        return celeba_tasks(seed)
    finally:
        if saved is None:
            t1._CELEBA.pop("c", None)
        else:
            t1._CELEBA["c"] = saved


def image_tasks_raw(dataset, seed, T=200, n_clusters=20, n_tr=64, n_te=128):
    """Same tasks as image_tasks: the sampler draws indices from the labels and
    the rng only, so a given seed selects the same images whatever the feature
    representation. Returned as (n, C, H, W)."""
    if dataset == "CelebA":
        return celeba_tasks_raw(seed)
    Xtr_s, ytr, Xte_s, yte = _image_raw(dataset)
    rng = np.random.default_rng(seed)
    Xt, yt, Xe, ye, _, _ = generate_clustered_binary_tasks(
        Xtr_s, ytr, Xte_s, yte, n_tasks=T, n_clusters=n_clusters,
        n_train_per_task=n_tr, n_test_per_task=n_te, rng=rng)
    s = SHAPE[dataset]
    nodes = list(range(T))
    return (nodes,
            {v: Xt[v].reshape(-1, *s) for v in nodes}, {v: yt[v] for v in nodes},
            {v: Xe[v].reshape(-1, *s) for v in nodes}, {v: ye[v] for v in nodes})


# ---------------------------------------------------------------------------
# diag: is there non-linear headroom once the model can see the image?
# ---------------------------------------------------------------------------

def fit_to_convergence(X, y, shape, epochs=1500, lr=1e-2):
    torch.manual_seed(0)
    m = SmallCNN(shape)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(np.asarray(y)).float().reshape(-1)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        crit(m(Xt), yt).backward()
        opt.step()
    m.eval()
    return m


def run_diag(n_tasks, epochs, datasets, tag):
    from sklearn.linear_model import LogisticRegression
    rows = []
    for ds in datasets:
        print(f"=== {ds}: converged CNN vs converged linear ===", flush=True)
        _, Xp_tr, yp_tr, Xp_te, yp_te = _pca_tasks(ds, 0)
        _, Xr_tr, yr_tr, Xr_te, yr_te = image_tasks_raw(ds, 0)
        for i in range(n_tasks):
            ytr = np.asarray(yp_tr[i]).reshape(-1)
            yte = np.asarray(yp_te[i]).reshape(-1)
            lin_pca = LogisticRegression(max_iter=5000, C=1e3).fit(Xp_tr[i], ytr)
            acc_pca = float((lin_pca.predict(Xp_te[i]) == yte).mean())
            flat_tr = Xr_tr[i].reshape(len(Xr_tr[i]), -1)
            flat_te = Xr_te[i].reshape(len(Xr_te[i]), -1)
            lin_raw = LogisticRegression(max_iter=5000, C=1e3).fit(flat_tr, ytr)
            acc_raw = float((lin_raw.predict(flat_te) == yte).mean())
            m = fit_to_convergence(Xr_tr[i], ytr, SHAPE[ds], epochs=epochs)
            with torch.no_grad():
                out = m(torch.from_numpy(Xr_te[i]).float()).numpy().reshape(-1)
            acc_cnn = float(((out > 0) == (yte > 0.5)).mean())
            rows.append(dict(dataset=ds, task=i, lin_pca=acc_pca, lin_raw=acc_raw,
                             lin_best=max(acc_pca, acc_raw), cnn=acc_cnn))
            if (i + 1) % 10 == 0:
                d = pd.DataFrame(rows).query("dataset == @ds")
                print(f"  {i+1} tasks: linear {d.lin_best.mean():.4f} "
                      f"CNN {d.cnn.mean():.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"image_cnn_diag{tag}_raw.csv", index=False)
    summ = df.groupby("dataset").agg(
        lin_pca=("lin_pca", "mean"), lin_raw=("lin_raw", "mean"),
        lin_best=("lin_best", "mean"), cnn=("cnn", "mean"),
        cnn_win=("cnn", "count")).reset_index()
    summ["cnn_gain_pct"] = 100 * (summ.cnn - summ.lin_best) / summ.lin_best
    summ["cnn_win_frac"] = [float((df[df.dataset == d].cnn
                                   > df[df.dataset == d].lin_best).mean())
                            for d in summ.dataset]
    summ.drop(columns=["cnn_win"]).to_csv(HERE / f"image_cnn_diag{tag}_summary.csv", index=False)
    print("\n=== summary ===", flush=True)
    print(summ.drop(columns=["cnn_win"]).round(4).to_string(index=False), flush=True)


def _pca_tasks(ds, seed):
    """The feature-space view of the same tasks: PCA-100 for the two small
    collections, frozen ResNet-18 for CelebA. Used for the linear reference in
    diag and, in table, for the distance proxy and the root."""
    from table1_nonconvex import image_tasks, celeba_tasks
    return celeba_tasks(seed) if ds == "CelebA" else image_tasks(ds, seed)


# ---------------------------------------------------------------------------
# table: the full method comparison with the CNN as local model
# ---------------------------------------------------------------------------

def calibrate(ds, shape, X_tr, y_tr, X_te, y_te, nodes, D, root, cfg):
    """Best eta for the Individual baseline at B=2000, as for the MLP table."""
    best, best_eta = -np.inf, None
    for eta in ETA_GRID:
        torch.manual_seed(0)
        b = max(1, 2000 // len(nodes))
        models = {v: train_steps(SmallCNN(shape), X_tr[v], y_tr[v], b, eta,
                                 "classification", cfg["reg"]) for v in nodes}
        acc = eval_all(models, X_te, y_te, "classification")
        print(f"  {ds} eta={eta}: Individual acc={acc:.4f}", flush=True)
        if acc > best:
            best, best_eta = acc, eta
    print(f"  {ds}: eta={best_eta}", flush=True)
    return best_eta


def run_table(n_reps, datasets, tag, rep0=0, fixed_eta=None):
    rows = []
    for ds in datasets:
        cfg = dict(CONFIG[ds])
        shape = SHAPE[ds]
        if fixed_eta is None:
            nodes, Xr_tr, y_tr, Xr_te, y_te = image_tasks_raw(ds, 0)
            _, Xp_tr, _, _, _ = _pca_tasks(ds, 0)
            D = compute_gradient_distance(nodes, Xp_tr, y_tr)
            root = select_seed_node(Xp_tr, y_tr)
            print(f"=== {ds}: calibrating eta on Individual ===", flush=True)
            eta = calibrate(ds, shape, Xr_tr, y_tr, Xr_te, y_te, nodes, D, root, cfg)
        else:
            eta = fixed_eta
            print(f"=== {ds}: eta={eta} supplied ===", flush=True)
        for rep in range(rep0, rep0 + n_reps):
            nodes, Xr_tr, y_tr, Xr_te, y_te = image_tasks_raw(ds, rep)
            pnodes, Xp_tr, yp_tr, _, _ = _pca_tasks(ds, rep)
            assert list(pnodes) == list(nodes) and all(
                np.array_equal(np.asarray(y_tr[v]).reshape(-1),
                               np.asarray(yp_tr[v]).reshape(-1)) for v in nodes), \
                "raw and feature views describe different tasks"
            D = compute_gradient_distance(nodes, Xp_tr, y_tr)
            root = select_seed_node(Xp_tr, y_tr)
            rng = np.random.default_rng(90_000 + rep)
            for B in BUDGETS[ds]:
                for m in METHODS:
                    torch.manual_seed(rep)
                    models = run_method(m, nodes, Xr_tr, y_tr, Xp_tr, y_tr, D, root,
                                        B, eta, cfg, lambda: SmallCNN(shape), rng)
                    val = eval_all(models, Xr_te, y_te, "classification")
                    rows.append(dict(dataset=ds, rep=rep, B=B, method=m,
                                     eta=eta, value=val))
                    print(f"  {ds} rep={rep} B={B} {m}: {val:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"image_cnn_table{tag}_raw.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby(["dataset", "B", "method"])["value"].agg(
        ["mean", "std", "count"]).reset_index()
    summ.to_csv(HERE / f"image_cnn_table{tag}_summary.csv", index=False)
    print("\n=== CNN local model, mean accuracy ===", flush=True)
    print(summ.pivot_table(index=["dataset", "B"], columns="method",
                           values="mean")[METHODS].round(4).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["diag", "table"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    ap.add_argument("--rep0", type=int, default=0)
    ap.add_argument("--eta", type=float, default=None,
                    help="skip calibration, e.g. to shard reps across processes")
    a = ap.parse_args()
    dss = DATASETS if a.dataset == "all" else (a.dataset,)
    tag = "" if a.dataset == "all" else "_" + a.dataset.lower()
    if a.mode == "diag":
        run_diag(a.tasks, a.epochs, dss, tag)
    else:
        if a.rep0:
            tag += f"_r{a.rep0}"
        run_table(a.reps, dss, tag, rep0=a.rep0, fixed_eta=a.eta)
