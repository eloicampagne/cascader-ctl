"""Image collections: Fashion-MNIST, CIFAR-10, CelebA.

Each task is a binary discrimination between two classes, and tasks are grouped
into clusters that share a class pair, so the collection has the same clustered
geometry as Syn-10 on real features. Fashion-MNIST and CIFAR-10 use 100 PCA
components fitted once on the full training set; CelebA uses frozen ResNet-18
features and discriminates attribute pairs.

Fashion-MNIST and CIFAR-10 download themselves on first use into the data
directory. CelebA does not: its 1.7 GB of images are the user's to obtain, and
the loader expects the extracted feature arrays.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..distances import compute_gradient_distance
from ..rooting import medoid_root
from ..tasks import TaskCollection
from ._paths import data_dir, require
from .transforms import standardize

__all__ = ["load_fashion_mnist", "load_cifar10", "image_collection",
           "celeba_collection", "generate_clustered_binary_tasks"]

Array = np.ndarray

_FMNIST_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_fashion_mnist(path: Optional[str | Path] = None
                       ) -> Tuple[Array, Array, Array, Array]:
    """Fashion-MNIST as flat float32 arrays in ``[0, 1]``, downloaded on demand."""
    import gzip

    root = data_dir(path) / "fashion_mnist"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    for name in files.values():
        _download(_FMNIST_URL + name, root / name)

    def images(p: Path) -> Array:
        with gzip.open(p, "rb") as f:
            f.read(16)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(
                np.float32) / 255.0

    def labels(p: Path) -> Array:
        with gzip.open(p, "rb") as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    return (images(root / files["train_images"]), labels(root / files["train_labels"]),
            images(root / files["test_images"]), labels(root / files["test_labels"]))


def load_cifar10(path: Optional[str | Path] = None
                 ) -> Tuple[Array, Array, Array, Array]:
    """CIFAR-10 as flat float32 arrays in ``[0, 1]``, downloaded on demand."""
    import pickle
    import tarfile

    root = data_dir(path) / "cifar10"
    extracted = root / "cifar-10-batches-py"
    if not extracted.exists():
        archive = _download(_CIFAR_URL, root / "cifar-10-python.tar.gz")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root)

    def batch(p: Path) -> Tuple[Array, Array]:
        with open(p, "rb") as f:
            b = pickle.load(f, encoding="bytes")
        return b[b"data"].astype(np.float32) / 255.0, np.array(b[b"labels"])

    Xs, ys = zip(*[batch(extracted / f"data_batch_{i}") for i in range(1, 6)])
    X_test, y_test = batch(extracted / "test_batch")
    return np.vstack(Xs), np.concatenate(ys), X_test, y_test


def generate_clustered_binary_tasks(X_train: Array, y_train: Array, X_test: Array,
                                    y_test: Array, n_tasks: int, n_clusters: int = 20,
                                    n_train_per_task: int = 64,
                                    n_test_per_task: int = 128,
                                    rng: Optional[np.random.Generator] = None):
    """Binary tasks grouped into clusters that share a class pair.

    ``n_clusters`` class pairs are drawn without replacement, tasks are assigned
    to clusters in equal parts and shuffled, and each task samples a balanced
    train and test set for its pair. Tasks in one cluster differ only by which
    examples they drew, so within-cluster distance is sampling noise and
    between-cluster distance is the class pair.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]
    all_pairs = list(combinations(range(n_classes), 2))
    cluster_pairs = [all_pairs[i] for i in
                     rng.choice(len(all_pairs), size=n_clusters, replace=False)]

    cluster_ids = np.repeat(np.arange(n_clusters), n_tasks // n_clusters)
    if len(cluster_ids) < n_tasks:
        cluster_ids = np.concatenate(
            [cluster_ids, rng.choice(n_clusters, size=n_tasks - len(cluster_ids))])
    rng.shuffle(cluster_ids)

    X_tr = np.zeros((n_tasks, n_train_per_task, d), dtype=np.float32)
    y_tr = np.zeros((n_tasks, n_train_per_task), dtype=np.float32)
    X_te = np.zeros((n_tasks, n_test_per_task, d), dtype=np.float32)
    y_te = np.zeros((n_tasks, n_test_per_task), dtype=np.float32)

    for t in range(n_tasks):
        a, b = cluster_pairs[cluster_ids[t]]
        for X_src, y_src, X_dst, y_dst, n in (
                (X_train, y_train, X_tr, y_tr, n_train_per_task),
                (X_test, y_test, X_te, y_te, n_test_per_task)):
            half = n // 2
            ia = rng.choice(np.where(y_src == a)[0], size=half, replace=False)
            ib = rng.choice(np.where(y_src == b)[0], size=half, replace=False)
            X_dst[t, :half], X_dst[t, half:] = X_src[ia], X_src[ib]
            y_dst[t, :half], y_dst[t, half:] = 0.0, 1.0
            perm = rng.permutation(n)
            X_dst[t], y_dst[t] = X_dst[t, perm], y_dst[t, perm]

    return X_tr, y_tr, X_te, y_te, cluster_ids


_PCA_CACHE: Dict = {}


def _pca_features(dataset: str, path, n_components: int = 100):
    """Standardize then project on ``n_components`` PCA directions, cached."""
    key = (dataset, str(path), n_components)
    if key in _PCA_CACHE:
        return _PCA_CACHE[key]
    from sklearn.decomposition import PCA

    loader = load_fashion_mnist if dataset == "FMNIST" else load_cifar10
    Xtr, ytr, Xte, yte = loader(path)
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32)
    Xte = Xte.reshape(len(Xte), -1).astype(np.float32)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    out = (pca.fit_transform(Xtr_s).astype(np.float32), ytr,
           pca.transform(Xte_s).astype(np.float32), yte)
    _PCA_CACHE[key] = out
    return out


def image_collection(dataset: str, seed: int = 0, T: int = 200, n_clusters: int = 20,
                     n_train: int = 64, n_test: int = 128,
                     path: Optional[str | Path] = None) -> TaskCollection:
    """FMNIST or CIFAR as ``T`` clustered binary tasks on PCA-100 features."""
    if dataset not in ("FMNIST", "CIFAR"):
        raise ValueError("dataset must be 'FMNIST' or 'CIFAR'")
    Xtr, ytr, Xte, yte = _pca_features(dataset, path)
    rng = np.random.default_rng(seed)
    Xt, yt, Xe, ye, clusters = generate_clustered_binary_tasks(
        Xtr, ytr, Xte, yte, n_tasks=T, n_clusters=n_clusters,
        n_train_per_task=n_train, n_test_per_task=n_test, rng=rng)

    nodes = list(range(T))
    X_tr = {v: Xt[v] for v in nodes}
    y_tr = {v: yt[v] for v in nodes}
    return TaskCollection(
        nodes=nodes, X_train=X_tr, y_train=y_tr,
        X_test={v: Xe[v] for v in nodes}, y_test={v: ye[v] for v in nodes},
        D=compute_gradient_distance(nodes, X_tr, y_tr),
        root=medoid_root(nodes, X_tr), task="classification",
        name=dataset, meta={"cluster_ids": clusters, "seed": seed})


_CELEBA_HOW = (
    "CelebA features are not distributed with the package. Extract frozen "
    "ResNet-18 penultimate features into X_train.npy, y_train.npy, X_test.npy "
    "and y_test.npy under <data_dir>/celeba_features/."
)


def celeba_collection(seed: int = 0, n_tasks: int = 400, n_clusters: int = 5,
                      n_train: int = 64, n_test: int = 128,
                      path: Optional[str | Path] = None) -> TaskCollection:
    """CelebA attribute-pair tasks on frozen ResNet-18 features.

    Each task discriminates ``attr_i = 1`` against ``attr_j = 1`` on the examples
    carrying exactly one of the two, and clusters share a base pair. Pairs with
    too few examples are skipped, so the realized task count varies with the
    seed between roughly 240 and ``n_tasks`` -- express budgets per task rather
    than absolutely, or the effective budget drifts across seeds.
    """
    root_dir = data_dir(path) / "celeba_features"
    require(root_dir / "X_train.npy", "CelebA features", _CELEBA_HOW)
    Xtr = np.load(root_dir / "X_train.npy")
    ytr = np.load(root_dir / "y_train.npy")
    Xte = np.load(root_dir / "X_test.npy")
    yte = np.load(root_dir / "y_test.npy")

    rng = np.random.default_rng(seed)
    n_attrs = ytr.shape[1]
    all_pairs = [(i, j) for i in range(n_attrs) for j in range(i + 1, n_attrs)]
    cluster_pairs = [all_pairs[i] for i in
                     rng.choice(len(all_pairs), size=n_clusters, replace=False)]
    cluster_ids = np.repeat(np.arange(n_clusters), n_tasks // n_clusters)
    if len(cluster_ids) < n_tasks:
        cluster_ids = np.concatenate(
            [cluster_ids, rng.choice(n_clusters, size=n_tasks - len(cluster_ids))])
    rng.shuffle(cluster_ids)

    def sample(X, Y, ai, aj, per_class):
        mi, mj = Y[:, ai] == 1, Y[:, aj] == 1
        idx_i = np.where(mi & ~(mi & mj))[0]
        idx_j = np.where(mj & ~(mi & mj))[0]
        if len(idx_i) < per_class or len(idx_j) < per_class:
            return None
        si = rng.choice(idx_i, size=per_class, replace=False)
        sj = rng.choice(idx_j, size=per_class, replace=False)
        return (np.vstack([X[si], X[sj]]).astype(np.float32),
                np.hstack([np.ones(per_class), np.zeros(per_class)]).astype(np.float32))

    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    v = 0
    for cid in cluster_ids:
        ai, aj = cluster_pairs[cid]
        tr = sample(Xtr, ytr, ai, aj, n_train // 2)
        te = sample(Xte, yte, ai, aj, n_test // 2)
        if tr is None or te is None:
            continue
        X_tr[v], y_tr[v] = tr
        X_te[v], y_te[v] = te
        v += 1

    nodes = list(X_tr)
    return TaskCollection(
        nodes=nodes, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        D=compute_gradient_distance(nodes, X_tr, y_tr),
        root=medoid_root(nodes, X_tr), task="classification",
        name="CelebA", meta={"seed": seed, "requested_tasks": n_tasks})
