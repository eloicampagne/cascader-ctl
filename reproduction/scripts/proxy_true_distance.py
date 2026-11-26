"""Gradient proxy against the distances the bound uses, on Syn.

For each evaluation seed of Syn, compare the gradient distance of the paper,
hat d(u,v) = || g_u/||g_u|| - g_v/||g_v|| ||, g_v = X_v' y_v, with

    d_opt(u,v) = || theta*_u - theta*_v ||, theta*_v the least-squares solution
                 (with intercept) on the normalized training data, i.e. the
                 point the gradient refinements contract to;
    d_gen(u,v) = || theta_u - theta_v ||, the generator's task parameters.

Reports, per seed: Spearman and Pearson correlations over all pairs, the mean
overlap of the 5 nearest neighbours, and the true length (under d_opt) of the
greedy chain built on hat d against the greedy chain built on d_opt.

    python3.12 proxy_true_distance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import cascader                                                   # noqa: E402
from cascader.data import load                                    # noqa: E402
from cascader.data.synthetic import generate_subspace_tasks       # noqa: E402
from cascader.distances import _pairwise_euclidean                # noqa: E402
from cascader.trees import build_chain                            # noqa: E402

assert Path(cascader.__file__).resolve().is_relative_to(HERE.parents[1] / "src"), cascader.__file__
K = 5


def knn_overlap(A, B, k=K):
    n = len(A)
    A, B = A + np.diag(np.full(n, np.inf)), B + np.diag(np.full(n, np.inf))
    na, nb = np.argsort(A, axis=1)[:, :k], np.argsort(B, axis=1)[:, :k]
    return float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(n)]))


def chain_length(tree, Dtrue):
    return float(sum(Dtrue[u, v] for u, v in tree.edges))


rows = []
for seed in range(20):
    c = load("Syn", seed=seed)
    # Same rng and defaults as subspace_collection, so the draw is identical.
    thetas, _, (Xt, yt), _, _ = generate_subspace_tasks(
        T=200, D=40, r=3, n_train=64, n_test=128, n_clusters=5, tau_within=3.0,
        tau_between=10.0, sigma=5.0, rng=np.random.default_rng(seed))
    assert np.allclose((Xt[0] - c.stats[0]["X_mean"]) / c.stats[0]["X_std"], c.X_train[0])
    opt = []
    for v in c.nodes:
        Xa = np.hstack([c.X_train[v], np.ones((len(c.X_train[v]), 1))])
        opt.append(np.linalg.lstsq(Xa, c.y_train[v], rcond=None)[0])
    D_opt = _pairwise_euclidean(np.stack(opt))
    D_gen = _pairwise_euclidean(np.asarray(thetas))
    iu = np.triu_indices(c.T, 1)
    r = dict(seed=seed)
    for name, Dt in (("opt", D_opt), ("gen", D_gen)):
        r[f"spearman_{name}"] = spearmanr(c.D[iu], Dt[iu]).statistic
        r[f"pearson_{name}"] = pearsonr(c.D[iu], Dt[iu]).statistic
        r[f"knn{K}_{name}"] = knn_overlap(c.D, Dt)
    r["chain_len_proxy"] = chain_length(build_chain(c.nodes, c.D, c.root), D_opt)
    r["chain_len_oracle"] = chain_length(build_chain(c.nodes, D_opt, c.root), D_opt)
    rows.append(r)
    print(seed, {k: round(v, 3) for k, v in r.items() if k != "seed"}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(Path(sys.argv[1] if len(sys.argv) > 1 else HERE) / "proxy_true_distance_Syn.csv", index=False)
df["chain_excess"] = df.chain_len_proxy / df.chain_len_oracle - 1
print(df.drop(columns="seed").agg(["mean", "std", "min", "max"]).T.round(3).to_string())
