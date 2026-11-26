"""Empirical contraction of the refinements, linear and MLP.

Question. Proposition 2.3 needs ||G_v^b(theta) - theta*_v|| <= rho_v^b ||theta - theta*_v||
with rho_v < 1. For the MLP no such theta*_v is given. We measure, on the
Chain cascade of the main tables (SLE floor-one allocation, Chain-selected rate
for each budget), at the parameters theta_in that a task actually inherits from
its parent:

    hat theta_v   the point full-batch GD (package ``train_steps``, same rate,
                  clipping and l2) reaches from theta_in, run until the
                  unclipped gradient norm falls below GTOL or NMAX steps;
    r_k           ||G_v^k(theta_in) - hat theta_v|| / ||theta_in - hat theta_v||,
                  k = 1..8, the realized contraction toward that point;
    rho_loc       max |eig(I - eta H)| at hat theta_v, H the Hessian of the
                  regularized loss (power iteration on Hessian-vector products),
                  the local analogue of rho_v = ||M_v||_2;
    basin         (b = 1 only) ||hat theta_v - hat theta_v^fresh|| / ||theta_in - hat theta_v||,
                  hat theta_v^fresh reached from a fresh initialization: zero when
                  the minimizer does not depend on the start.

    CASCADER_DATA_DIR=<data_links> python3.12 contraction.py --dataset Syn --model mlp --summaries <dir> --seed0 0 --seeds 1
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import cascader                                                   # noqa: E402
from cascader.cascade import cascade_train, train_steps          # noqa: E402
from cascader.config import get_config                           # noqa: E402
from cascader.data import load                                    # noqa: E402
from cascader.data.weave import weave_collection                  # noqa: E402
from cascader.methods import allocate, build_structure            # noqa: E402
from cascader.models import model_factory                         # noqa: E402

assert Path(cascader.__file__).resolve().is_relative_to(HERE.parents[1] / "src"), cascader.__file__
torch.set_num_threads(1)

KMAX, GTOL, NMAX, CHUNK = 8, 1e-5, 100_000, 5_000


def flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()


def loss_fn(m, X, y, task, l2):
    crit = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    loss = crit(m(X), y)
    if l2:
        loss = loss + 0.5 * l2 * sum((p ** 2).sum() for p in m.parameters())
    return loss


def grad_norm(m, X, y, task, l2):
    m.zero_grad(set_to_none=True)
    loss_fn(m, X, y, task, l2).backward()
    return float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))


def converge(m, Xn, yn, X, y, eta, task, l2):
    """Run package GD until the gradient vanishes; return steps and final norm."""
    steps, g = 0, grad_norm(m, X, y, task, l2)
    while g > GTOL and steps < NMAX:
        train_steps(m, Xn, yn, CHUNK, eta, task, l2)
        steps += CHUNK
        g = grad_norm(m, X, y, task, l2)
    return steps, g


def rho_local(m, X, y, eta, task, l2, iters=200):
    """Spectral radius of I - eta H at the current parameters (H symmetric)."""
    params = list(m.parameters())
    grads = torch.autograd.grad(loss_fn(m, X, y, task, l2), params, create_graph=True)
    gvec = torch.cat([g.reshape(-1) for g in grads])
    v = torch.randn_like(gvec, generator=torch.Generator().manual_seed(0))
    v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        hv = torch.cat([h.reshape(-1) for h in torch.autograd.grad(gvec, params, v, retain_graph=True)])
        w = v - eta * hv
        lam = float(w.norm())
        v = w / lam
    return lam


def loader(ds, split):
    if ds == "WEAVE-UK":
        return (lambda s: weave_collection(val_size=48, sample=False)) if split == "tune" \
            else (lambda s: weave_collection(sample=False))
    return lambda s: load(ds, seed=s)


def chain_etas(ds, model):
    summ = pd.read_csv(Path(ARGS.summaries) / f"randchain_{ds}_{model}_summary.csv")
    summ = summ[summ.method == "Chain"]
    return dict(zip(summ.b, summ.eta))


def main():
    cfg = get_config(ARGS.dataset)
    etas = chain_etas(ARGS.dataset, ARGS.model)
    load_eval = loader(ARGS.dataset, "eval")
    rows, out = [], Path(ARGS.out) / f"contraction_{ARGS.dataset}_{ARGS.model}_s{ARGS.seed0}.csv"
    for seed in range(ARGS.seed0, ARGS.seed0 + ARGS.seeds):
        tasks = load_eval(seed)
        factory = model_factory(ARGS.model, tasks.d)
        tree = build_structure("Chain", tasks, cfg)
        parent = {v: u for u, v in tree.edges}
        sample = np.random.default_rng(seed).choice(sorted(parent, key=str), ARGS.tasks, replace=False)
        for b in (1, 2, 4):
            t0 = time.perf_counter()
            eta = float(etas[b])
            alloc = allocate("Chain", tree, tasks, b * tasks.T, cfg)
            torch.manual_seed(seed)
            fitted = cascade_train(tree, tasks.X_train, tasks.y_train, alloc, eta, factory,
                                   tasks.task, cfg.reg)
            for v in sample:
                Xn, yn = tasks.X_train[v], tasks.y_train[v]
                X = torch.from_numpy(np.asarray(Xn)).float()
                y = torch.from_numpy(np.asarray(yn)).float().reshape(-1)
                m_in = fitted[parent[v]]
                th_in = flat(m_in)
                traj, m = [], copy.deepcopy(m_in)
                for _ in range(KMAX):
                    train_steps(m, Xn, yn, 1, eta, tasks.task, cfg.reg)
                    traj.append(flat(m))
                m_hat = copy.deepcopy(m)
                steps, g = converge(m_hat, Xn, yn, X, y, eta, tasks.task, cfg.reg)
                th_hat = flat(m_hat)
                d0 = float((th_in - th_hat).norm())
                r = dict(dataset=ARGS.dataset, model=ARGS.model, seed=seed, b=b, eta=eta,
                         task=str(v), steps_v=int(alloc.get(v, 0)), dist0=d0,
                         conv_steps=steps + KMAX, grad_final=g,
                         loss_in=float(loss_fn(m_in, X, y, tasks.task, cfg.reg)),
                         loss_hat=float(loss_fn(m_hat, X, y, tasks.task, cfg.reg)),
                         rho_loc=rho_local(m_hat, X, y, eta, tasks.task, cfg.reg))
                for k, th in enumerate(traj, 1):
                    r[f"r{k}"] = float((th - th_hat).norm()) / d0
                if b == 1:
                    torch.manual_seed(10_000 + seed)
                    m_f = factory()
                    train_steps(m_f, Xn, yn, KMAX, eta, tasks.task, cfg.reg)
                    steps_f, g_f = converge(m_f, Xn, yn, X, y, eta, tasks.task, cfg.reg)
                    r.update(basin=float((flat(m_f) - th_hat).norm()) / d0, grad_final_fresh=g_f,
                             loss_hat_fresh=float(loss_fn(m_f, X, y, tasks.task, cfg.reg)))
                rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{ARGS.dataset} {ARGS.model} seed={seed} b={b} eta={eta} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="mlp", choices=("linear", "mlp"))
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--tasks", type=int, default=25)
    p.add_argument("--summaries", required=True, help="directory of randchain_*_summary.csv")
    p.add_argument("--out", default=str(HERE))
    ARGS = p.parse_args()
    main()
