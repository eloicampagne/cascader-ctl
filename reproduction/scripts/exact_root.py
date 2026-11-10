"""Exact-root ablation: does the Chain's tight-budget lead over the MST and Star come from the root residual?

Proposition 1 bounds a task's error by P_{1:m}||theta_root - theta_root*|| plus discounted edge terms.
With a random root, the first term decays with depth and favors deep trees. Initializing the root at its
own least-squares solution (the theorem's exact-root assumption) removes it. Same trees, allocation, clipping and
per-method rate selection as the main table; only the root's starting point changes.

    CASCADER_DATA_DIR=<dir with train_weave_500.csv> python3 exact_root.py out.json
"""
import sys, copy, json
from collections import defaultdict, deque
import numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cascader.config import get_config
from cascader.data.weave import weave_collection
from cascader.methods import build_structure, allocate
from cascader.cascade import train_steps
from cascader.models import model_factory
from cascader.experiments._common import etas_for, evaluate, TUNE_OFFSET

METHODS = ["Star", "Chain", "CTL-MST", "CTL-MSTc"]
BS = [1, 2, 4]

def ols_root(model, X, y):
    X1 = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(X1, y, rcond=None)[0]
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor(w[:-1], dtype=model.fc.weight.dtype)[None, :])
        model.fc.bias.copy_(torch.tensor([w[-1]], dtype=model.fc.bias.dtype))
    return model

def run(tasks, method, b, eta, root_mode, seed, cfg):
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 + seed)
    tree = build_structure(method, tasks, cfg, rng)
    alloc = allocate(method, tree, tasks, b * tasks.T, cfg)
    fac = model_factory("linear", tasks.d)
    ch = defaultdict(list)
    for u, v in tree.edges: ch[u].append(v)
    r = tree.root
    m0 = fac()
    if root_mode == "exact": m0 = ols_root(m0, tasks.X_train[r], tasks.y_train[r])
    models = {r: train_steps(m0, tasks.X_train[r], tasks.y_train[r], int(alloc.get(r, 0)), eta, tasks.task, cfg.reg)}
    q = deque([r])
    while q:
        u = q.popleft()
        for v in ch[u]:
            models[v] = train_steps(copy.deepcopy(models[u]), tasks.X_train[v], tasks.y_train[v], int(alloc.get(v, 0)), eta, tasks.task, cfg.reg)
            q.append(v)
    return evaluate(models, tasks)

def main():
    cfg = get_config("WEAVE-UK"); etas = etas_for("WEAVE-UK", "linear")
    val = weave_collection(val_size=48); test = weave_collection()
    rows = []
    for root_mode, n_tune, n_eval in (("exact", 2, 2), ("random", 5, 20)):
        for b in BS:
            for method in METHODS:
                scores = {}
                for eta in etas:
                    vals = []
                    for s in range(TUNE_OFFSET, TUNE_OFFSET + n_tune):
                        try: vals.append(run(val, method, b, eta, root_mode, s, cfg))
                        except RuntimeError: vals.append(np.nan)
                    scores[eta] = np.mean(vals)
                eta_star = min((e for e in scores if np.isfinite(scores[e])), key=scores.get)
                ev = [run(test, method, b, eta_star, root_mode, s, cfg) for s in range(n_eval)]
                rows.append(dict(root=root_mode, b=b, method=method, eta=eta_star, mean=float(np.mean(ev)),
                                 se=float(np.std(ev, ddof=1) / np.sqrt(len(ev))), n=len(ev), vals=ev))
                print(f"{root_mode:6s} b={b} {method:9s} eta={eta_star:<8.4g} RMSE={np.mean(ev):8.1f} ± {rows[-1]['se']:.1f}", flush=True)
    pd.DataFrame(rows).to_json(sys.argv[1], orient="records", indent=1)

if __name__ == "__main__":
    main()
