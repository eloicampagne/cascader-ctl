"""Every timing reported in the paper, measured with the packaged implementation.

Three measurements on the clustered synthetic generator (Syn-10 geometry):

  construction  graph time (gradient proxy, pairwise distances, medoid root,
                MST) against training time of the linear cascade at B=2000.
  scaling       construction time of the MST, the Chain and CTL-MSTc as T grows,
                with the fitted log-log slope over the largest sizes.
  parallel      wall-clock time of CTL-MST at P workers, one thread per task of a
                depth level and a barrier between levels, under the floor-one
                allocation of the main tables.

    python3.12 timing.py --part all --out timing_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cascader.allocation import sle_alloc, uniform_alloc  # noqa: E402
from cascader.cascade import cascade_train  # noqa: E402
from cascader.config import get_config  # noqa: E402
from cascader.data.synthetic import generate_clustered_tasks  # noqa: E402
from cascader.data.transforms import normalize_xy  # noqa: E402
from cascader.distances import compute_gradient_distance  # noqa: E402
from cascader.models import model_factory  # noqa: E402
from cascader.rooting import medoid_root  # noqa: E402
from cascader.trees import build_chain, build_mst, build_mstc  # noqa: E402


def tasks(T, seed, d=3, n_train=64):
    rng = np.random.default_rng(seed)
    _, _, _, (Xt, yt), _, _ = generate_clustered_tasks(T=T, d=d, n_train=n_train,
                                                         n_test=1, rng=rng)
    nodes = list(range(T))
    X, y = {}, {}
    for v in nodes:
        X[v], y[v], _ = normalize_xy(Xt[v], yt[v])
    return nodes, {v: Xt[v] for v in nodes}, X, y


def part_construction(Ts, seeds, B=2000):
    cfg = get_config("Syn-10")
    out = []
    for T in Ts:
        g, tr = [], []
        for s in range(seeds):
            nodes, Xraw, X, y = tasks(T, s)
            t0 = time.perf_counter()
            D = compute_gradient_distance(nodes, X, y)
            root = medoid_root(nodes, Xraw)
            tree = build_mst(nodes, D, root)
            g.append(time.perf_counter() - t0)
            alloc = sle_alloc(tree, D, {v: i for i, v in enumerate(nodes)}, B,
                              root_frac=cfg.root_frac)
            torch.manual_seed(s)
            t0 = time.perf_counter()
            cascade_train(tree, X, y, alloc, cfg.eta, model_factory("linear", 3))
            tr.append(time.perf_counter() - t0)
        row = dict(T=T, graph_mean=float(np.mean(g)), graph_se=float(np.std(g, ddof=1) / np.sqrt(len(g))),
                   train_mean=float(np.mean(tr)), train_se=float(np.std(tr, ddof=1) / np.sqrt(len(tr))))
        row["ratio_pct"] = 100 * row["graph_mean"] / row["train_mean"]
        print(f"[construction] T={T}: graph {row['graph_mean']:.4f}s train {row['train_mean']:.3f}s "
              f"ratio {row['ratio_pct']:.2f}%", flush=True)
        out.append(row)
    return out


def part_scaling(Ts, seeds_for, mstc_max=5000):
    cfg = get_config("Syn-10")
    out = []
    for T in Ts:
        rows = defaultdict(list)
        for s in range(seeds_for(T)):
            nodes, Xraw, X, y = tasks(T, s)
            t0 = time.perf_counter(); D = compute_gradient_distance(nodes, X, y)
            rows["distance"].append(time.perf_counter() - t0)
            root = medoid_root(nodes, Xraw)
            t0 = time.perf_counter(); build_mst(nodes, D, root)
            rows["mst"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); build_chain(nodes, D, root)
            rows["chain"].append(time.perf_counter() - t0)
            if T <= mstc_max and s == 0:
                t0 = time.perf_counter(); build_mstc(nodes, D, root, lambda_reach=cfg.mstc_lambda)
                rows["mstc"].append(time.perf_counter() - t0)
        row = {"T": T, **{k: float(np.median(v)) for k, v in rows.items()}}
        print(f"[scaling] {row}", flush=True)
        out.append(row)
    return out


def part_parallel(reps=8, T=200, d=100, hidden=256, n_tr=2048, B=2000, eta=0.05,
                  workers=(1, 2, 4, 8)):
    torch.set_num_threads(1)
    cfg = get_config("Syn-10")
    out = []

    def model():
        return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def steps(m, X, y, k):
        crit = nn.MSELoss()
        for _ in range(k):
            loss = crit(m(X).reshape(-1), y)
            m.zero_grad(set_to_none=True); loss.backward()
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(p.grad, alpha=-eta)
        return m

    for rep in range(reps):
        nodes, Xraw, X, y = tasks(T, 1000 + rep, d=d, n_train=n_tr)
        Xt = {v: torch.from_numpy(X[v]).float() for v in nodes}
        yt = {v: torch.from_numpy(y[v]).float() for v in nodes}
        D = compute_gradient_distance(nodes, X, y)
        root = medoid_root(nodes, Xraw)
        tree = build_mst(nodes, D, root)
        index = {v: i for i, v in enumerate(nodes)}
        for kind in ("sle", "flat"):
            alloc = (sle_alloc(tree, D, index, B, root_frac=cfg.root_frac) if kind == "sle"
                     else uniform_alloc(tree.nodes, tree.root, B, root_frac=cfg.root_frac))
            levels = defaultdict(list)
            for v, dep in tree.depths().items():
                levels[dep].append(v)
            ch = tree.children
            cum = {root: alloc[root]}
            for u, v in tree.edges:
                cum[v] = cum[u] + alloc[v]
            cp = max(cum.values())
            order = list(workers)
            np.random.default_rng(rep).shuffle(order)
            for P in order:
                torch.manual_seed(rep)
                models = {}
                pool = ThreadPoolExecutor(P)
                t0 = time.perf_counter()
                models[root] = steps(model(), Xt[root], yt[root], alloc[root])
                for dep in sorted(levels)[1:]:
                    vs = levels[dep]
                    res = pool.map(lambda v: (v, steps(copy.deepcopy(models[tree.parent[v]]),
                                                       Xt[v], yt[v], alloc[v])), vs)
                    models.update(dict(res))
                wall = time.perf_counter() - t0
                pool.shutdown()
                out.append(dict(rep=rep, alloc=kind, workers=P, wall=wall,
                                critical_path=cp, total=sum(alloc.values())))
                print(f"[parallel] rep={rep} {kind} P={P} wall={wall:.2f}s cp={cp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("all", "construction", "scaling", "parallel"))
    ap.add_argument("--out", default="timing_results.json")
    a = ap.parse_args()
    res = {}
    if a.part in ("all", "construction"):
        res["construction"] = part_construction((100, 200, 500, 1000, 2000), seeds=50)
    if a.part in ("all", "scaling"):
        res["scaling"] = part_scaling((200, 500, 1000, 2000, 5000, 10000),
                                      seeds_for=lambda T: 5 if T <= 2000 else 3 if T <= 5000 else 2)
    if a.part in ("all", "parallel"):
        res["parallel"] = part_parallel()
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
