"""Stronger baselines and variants of CTL, one ingredient changed at a time.

Table "Stronger baselines" of the appendix uses base, exactroot and momentum with
--tune 5 --eval 20, and the water-filling paragraph uses kkt.

Variants (each applied to every method it can apply to, Star and Individual included):
  base        the published pipeline (medoid root, SLE floor-one, GD with clipping)
  exactroot   root initialized at its fitted optimum (least squares, or logistic fit)
  invL        per-task step eta_v = c / L_v, L_v the smoothness constant of task v
  momentum    heavy ball (beta) with the velocity inherited from the parent
  momentum0   heavy ball with the velocity reset at every task
  avgparents  initialization from the distance-weighted mean of the k nearest
              already-refined tasks (cascades only)
  kkt         KKT water-filling allocation of the surplus (cascades only)
  w1c, w1m    weighted 1-centre / 1-median root of the MST (cascades only)

Per (variant, method, b): rate (or c) selected on tuning seeds, evaluated on eval seeds.
    python3.12 stronger_baselines.py --dataset WEAVE-UK --variants base,exactroot,momentum,kkt \
        --tune 5 --eval 20 --out res.json
"""
import argparse, copy, json, sys, time
from pathlib import Path
from collections import defaultdict, deque
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cascader import data
from cascader.config import get_config
from cascader.data.weave import weave_collection
from cascader.methods import build_structure, allocate
from cascader.allocation import individual_alloc, apportion
from cascader.models import model_factory
from cascader.cascade import eval_all
from cascader.experiments._common import etas_for, TUNE_OFFSET
from cascader.trees import root_tree, mst_edges
from cascader.rooting import weighted_1_centre, weighted_1_median

ETA_GRID = None
C_GRID = (0.25, 0.5, 1.0, 1.5)
torch.set_num_threads(1)


# ----------------------------------------------------------------------------- training
def smoothness(X, task):
    lam = np.linalg.eigvalsh(X.T @ X / len(X))[-1] + 1.0  # +1: bias column, standardized X
    return 2 * lam if task == "regression" else lam / 4


def make_stepper(task, l2):
    crit = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    def run(model, X, y, steps, eta, beta=0.0, vel=None):
        if steps <= 0:
            return model, vel
        Xt = torch.from_numpy(np.asarray(X)).float(); yt = torch.from_numpy(np.asarray(y)).float().reshape(-1)
        params = list(model.parameters())
        if beta > 0 and vel is None:
            vel = [torch.zeros_like(p) for p in params]
        for _ in range(int(steps)):
            loss = crit(model(Xt), yt)
            if not torch.isfinite(loss):
                raise RuntimeError("diverged")
            model.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            with torch.no_grad():
                for i, p in enumerate(params):
                    g = p.grad if l2 == 0.0 else p.grad + l2 * p
                    if beta > 0:
                        vel[i].mul_(beta).add_(g); g = vel[i]
                    p.add_(g, alpha=-eta)
        return model, vel
    return run


def fit_root(model, X, y, task, l2):
    """Fitted optimum of the root task: least squares, or 500 L-BFGS iterations."""
    if task == "regression" and hasattr(model, "fc"):
        X1 = np.c_[X, np.ones(len(X))]; w = np.linalg.lstsq(X1, y, rcond=None)[0]
        with torch.no_grad():
            model.fc.weight.copy_(torch.tensor(w[:-1], dtype=torch.float32)[None]); model.fc.bias.fill_(float(w[-1]))
        return model
    crit = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    Xt = torch.from_numpy(np.asarray(X)).float(); yt = torch.from_numpy(np.asarray(y)).float().reshape(-1)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = crit(model(Xt), yt)
        if l2: loss = loss + 0.5 * l2 * sum((p ** 2).sum() for p in model.parameters())
        loss.backward(); return loss
    opt.step(closure)
    return model


# ----------------------------------------------------------------------------- allocation
def kkt_alloc(tree, D, index, B, kappa, root_frac):
    nodes = tree.nodes; others = [v for v in nodes if v != tree.root]
    if B < len(nodes):
        raise ValueError
    surplus = B - len(nodes); b_root = int(root_frac * surplus); rest = surplus - b_root
    sz = tree.subtree_sizes()
    w = np.array([sz[v] * max(D[index[tree.parent[v]], index[v]], 1e-12) for v in others])
    active = np.ones(len(others), bool); x = np.zeros(len(others))
    for _ in range(len(others)):
        lw = np.log(w[active]); x_act = rest / active.sum() + (lw - lw.mean()) / kappa
        if (x_act >= 0).all():
            x[:] = 0; x[active] = x_act; break
        idx = np.flatnonzero(active); active[idx[x_act < 0]] = False
    shares = apportion(np.maximum(x, 0), rest) if rest > 0 else np.zeros(len(others), int)
    a = {v: 1 + int(shares[i]) for i, v in enumerate(others)}; a[tree.root] = 1 + b_root
    return a


# ----------------------------------------------------------------------------- one run
def run_once(tc, method, b, eta, variant, seed, cfg, model_kind, opts):
    torch.manual_seed(seed)
    task, l2 = tc.task, cfg.reg
    step = make_stepper(task, l2)
    fac = model_factory(model_kind, tc.d)
    B = b * tc.T
    beta = opts.get("beta", 0.0) if variant in ("momentum", "momentum0") else 0.0
    eta_of = (lambda v: eta / tc.meta["_L"][v]) if variant == "invL" else (lambda v: eta)
    if method == "Individual":
        alloc = individual_alloc(tc.nodes, B)
        models = {v: step(fac(), tc.X_train[v], tc.y_train[v], alloc[v], eta_of(v), beta)[0] for v in tc.nodes}
        return eval_all(models, tc.X_test, tc.y_test, task, tc.stats)
    rng = np.random.default_rng(10_000 + seed)
    tree = build_structure(method, tc, cfg, rng)
    if variant in ("w1c", "w1m") and method != "Star":
        und = [(u, v) for u, v in tree.edges]
        r = (weighted_1_centre if variant == "w1c" else weighted_1_median)(tc.nodes, und, tc.D)
        tree = root_tree(und, r, tc.nodes)
    if variant == "kkt" and method != "Star":
        alloc = kkt_alloc(tree, tc.D, tc.index, B, opts.get("kappa", 0.2), cfg.root_frac)
    else:
        alloc = allocate(method, tree, tc, B, cfg)
    root = tree.root
    m0 = fac()
    if variant == "exactroot":
        m0 = fit_root(m0, tc.X_train[root], tc.y_train[root], task, l2)
    models, vels = {}, {}
    models[root], vels[root] = step(m0, tc.X_train[root], tc.y_train[root], alloc[root], eta_of(root), beta)
    children = defaultdict(list)
    for u, v in tree.edges:
        children[u].append(v)
    order, q = [], deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            order.append(v); q.append(v)
    done = [root]
    for v in order:
        if variant == "avgparents" and method != "Star":
            k = opts.get("k", 3)
            cand = np.array([tc.index[u] for u in done])
            dist = tc.D[tc.index[v], cand]
            near = np.argsort(dist)[:k]
            wts = 1.0 / (dist[near] + 1e-6); wts /= wts.sum()
            init = copy.deepcopy(models[tree.parent[v]])
            with torch.no_grad():
                for p_init, *ps in zip(init.parameters(), *[models[done[j]].parameters() for j in near]):
                    p_init.copy_(sum(w * p for w, p in zip(wts, ps)))
            vel0 = None
        else:
            init = copy.deepcopy(models[tree.parent[v]])
            vel0 = [t.clone() for t in vels[tree.parent[v]]] if (beta > 0 and vels.get(tree.parent[v]) is not None and opts.get("transfer", True)) else None
        models[v], vels[v] = step(init, tc.X_train[v], tc.y_train[v], alloc[v], eta_of(v), beta, vel0)
        done.append(v)
    return eval_all(models, tc.X_test, tc.y_test, task, tc.stats)


# ----------------------------------------------------------------------------- protocol
_CACHE = {}
def loader(dataset, seed, split):
    key = (dataset, seed, split)
    if key not in _CACHE:
        if dataset == "WEAVE-UK":
            tc = weave_collection(val_size=48) if split == "tune" else weave_collection()
        else:
            tc = data.load(dataset, seed=seed)
        tc.meta["_L"] = {v: smoothness(np.asarray(tc.X_train[v], float), tc.task) for v in tc.nodes}
        _CACHE[key] = tc
    return _CACHE[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--model", default="linear")
    ap.add_argument("--variants", default="base"); ap.add_argument("--methods", default="Individual,Star,Chain,CTL-MST,CTL-MSTc")
    ap.add_argument("--budgets", default="1,2,4"); ap.add_argument("--tune", type=int, default=3); ap.add_argument("--eval", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = get_config(a.dataset); higher = a.dataset in ("FMNIST", "CIFAR", "CelebA")
    rows = []
    for variant in a.variants.split(","):
        opts_list = {"momentum": [{"beta": 0.5}, {"beta": 0.9}], "momentum0": [{"beta": 0.5, "transfer": False}, {"beta": 0.9, "transfer": False}], "avgparents": [{"k": 2}, {"k": 4}],
                     "kkt": [{"kappa": 0.05}, {"kappa": 0.3}, {"kappa": 2.0}]}.get(variant, [{}])
        grid = C_GRID if variant == "invL" else etas_for(a.dataset, a.model)
        for b in map(int, a.budgets.split(",")):
            for method in a.methods.split(","):
                if method == "Individual" and variant not in ("base", "invL", "momentum"):
                    continue
                if method == "Star" and variant in ("avgparents", "kkt", "w1c", "w1m"):
                    continue
                best = None
                for opts in opts_list:
                    for eta in grid:
                        vals = []
                        for s in range(TUNE_OFFSET, TUNE_OFFSET + a.tune):
                            try:
                                vals.append(run_once(loader(a.dataset, s, "tune"), method, b, eta, variant, s, cfg, a.model, opts))
                            except RuntimeError:
                                vals.append(np.nan)
                        sc = np.mean(vals)
                        if not np.isfinite(sc): continue
                        key = -sc if higher else sc
                        if best is None or key < best[0]:
                            best = (key, eta, opts)
                _, eta, opts = best
                ev = []
                for s in range(a.eval):
                    try:
                        ev.append(run_once(loader(a.dataset, s, "eval"), method, b, eta, variant, s, cfg, a.model, opts))
                    except RuntimeError:
                        ev.append(np.nan)
                ev = np.array(ev, float)
                rows.append(dict(dataset=a.dataset, model=a.model, variant=variant, b=b, method=method, eta=eta, opts=opts,
                                 mean=float(np.nanmean(ev)), se=float(np.nanstd(ev, ddof=1) / np.sqrt(np.isfinite(ev).sum())), vals=ev.tolist()))
                print(f"{a.dataset} {variant:10s} b={b} {method:10s} eta={eta:<7.3g} {opts} -> {rows[-1]['mean']:.4f} ± {rows[-1]['se']:.4f}", flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
