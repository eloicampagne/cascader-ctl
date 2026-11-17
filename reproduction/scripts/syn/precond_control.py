"""Experience de controle : raffinement par gradient simple contre gradient preconditionne.

Question. La Proposition de tightness montre que l'anisotropie de la perte peut
tourner l'erreur vers l'arete suivante. Un preconditionneur proche de l'inverse
de la hessienne rend chaque raffinement isotrope. On mesure, a cout compte en
equivalents-pas, ce que le preconditionnement change pour chaque methode, et on
rattache chaque jeu de donnees a un regime par le conditionnement de ses taches.

Modele local lineaire (regression) ou logistique (classification), protocole des
tables principales : arbre, racine medoide et allocation SLE du package, pas
choisi par (jeu, methode, variante, budget) sur les graines de reglage.

Variante "precond" : g <- P_v g avec P_v = (c X~'X~/n + lam I)^{-1}, X~ = [X, 1],
c = 2 pour l'erreur quadratique (hessienne exacte), c = 1/4 pour la perte
logistique (majorant de Gauss-Newton), lam = 1e-3 tr/D~. Pas d'ecretage du
gradient pour cette variante (le pas de Newton a sa propre echelle, eta <= 1).

Cout (equivalents-pas, un pas = 4 n D~ operations) : preparation de P_v
(n D~^2 + D~^3/3) / (4 n D~), puis chaque pas coute 1 + D~/(2n).

    python3.12 precond_control.py --dataset Syn
    python3.12 precond_control.py --diag      # conditionnement des taches, par jeu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

import cascader.cascade as CC                                    # noqa: E402
from cascader.config import get_config                           # noqa: E402
from cascader.data import load                                   # noqa: E402
from cascader.data.weave import weave_collection                 # noqa: E402
from cascader.experiments._common import TUNE_OFFSET, etas_for   # noqa: E402
from cascader.methods import run_method                          # noqa: E402
from cascader.models import model_factory                        # noqa: E402

torch.set_num_threads(1)
METHODS = ("Individual", "Star", "Chain-2opt", "CTL-MST")
B_PLAIN = (1, 2, 4, 8, 16)
B_PREC = (1, 2, 4)
ETAS_PREC = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
CLS = {"FMNIST", "CIFAR", "CelebA"}
_PLAIN_TRAIN = CC.train_steps
_P = {}


def _precond(X, task):
    key = id(X)
    if key not in _P:
        Xa = np.hstack([np.asarray(X, float), np.ones((len(X), 1))])
        H = (2.0 if task == "regression" else 0.25) * Xa.T @ Xa / len(Xa)
        lam = 1e-3 * np.trace(H) / H.shape[0]
        _P[key] = torch.from_numpy(np.linalg.inv(H + lam * np.eye(H.shape[0]))).float()
    return _P[key]


def precond_train_steps(model, X, y, steps, eta, task="regression", l2=0.0):
    if steps <= 0:
        return model
    P = _precond(X, task)
    X_t = torch.from_numpy(np.asarray(X)).float()
    y_t = torch.from_numpy(np.asarray(y)).float().reshape(-1)
    crit = torch.nn.BCEWithLogitsLoss() if task == "classification" else torch.nn.MSELoss()
    w, b = model.fc.weight, model.fc.bias
    for _ in range(int(steps)):
        loss = crit(model(X_t), y_t)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            g = torch.cat([w.grad.reshape(-1), b.grad.reshape(-1)])
            if l2:
                g = g + l2 * torch.cat([w.reshape(-1), b.reshape(-1)])
            d = P @ g
            w.add_(d[:-1].reshape(w.shape), alpha=-eta)
            b.add_(d[-1:].reshape(b.shape), alpha=-eta)
    return model


def collection(ds, seed, split):
    if ds == "WEAVE-UK":
        return weave_collection(val_size=48) if split == "tune" else load(ds, seed=seed)
    return load(ds, seed=seed)


def costs(c):
    v = c.nodes[0]
    n, Dt = len(c.X_train[v]), c.X_train[v].shape[1] + 1
    return dict(setup=(n * Dt ** 2 + Dt ** 3 / 3) / (4 * n * Dt), per_step=1 + Dt / (2 * n), n=n, D=Dt - 1)


def run(ds, n_eval, n_tune):
    cfg = get_config(ds)
    task = "classification" if ds in CLS else "regression"
    rows, out = [], HERE / f"precond_{ds}.csv"
    for split, seeds in (("tune", range(TUNE_OFFSET, TUNE_OFFSET + n_tune)), ("eval", range(n_eval))):
        for s in seeds:
            t0 = time.perf_counter()
            c = collection(ds, s, split)
            fac = model_factory("linear", c.d)
            cst = costs(c)
            for variant, bs, etas in (("plain", B_PLAIN, tuple(sorted(set(etas_for(ds, "linear")) | {0.7, 1.0, 2.0}))), ("precond", B_PREC, ETAS_PREC)):
                CC.train_steps = _PLAIN_TRAIN if variant == "plain" else precond_train_steps
                _P.clear()
                for b in bs:
                    B = b * c.T
                    for eta in etas:
                        for m in METHODS:
                            torch.manual_seed(s)
                            try:
                                mo = run_method(m, c, B, eta, cfg, fac, rng=np.random.default_rng(10_000 + s))
                                v = CC.eval_all(mo, c.X_test, c.y_test, task, getattr(c, "stats", None))
                                v = v if np.isfinite(v) else np.nan
                            except RuntimeError:
                                v = np.nan
                            rows.append(dict(dataset=ds, split=split, seed=s, variant=variant, b=b,
                                             eta=eta, method=m, value=v, **cst))
            CC.train_steps = _PLAIN_TRAIN
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{ds} {split} {s} {time.perf_counter() - t0:.0f}s", flush=True)


def diag():
    rows = []
    for ds in ("Syn", "WEAVE-UK", "FMNIST", "CIFAR", "CelebA"):
        c = load(ds, seed=0)
        for v in c.nodes:
            X = np.hstack([np.asarray(c.X_train[v], float), np.ones((len(c.X_train[v]), 1))])
            ev = np.linalg.eigvalsh(X.T @ X / len(X))
            pos = ev[ev > 1e-8 * ev.max()]
            rows.append(dict(dataset=ds, task=v, n=len(X), D=X.shape[1] - 1, rank_def=len(pos) < len(ev),
                             kappa=pos.max() / pos.min(), top_over_mean=ev.max() / ev.mean()))
        print(ds, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "precond_diag.csv", index=False)
    print(d.groupby("dataset").agg(n=("n", "first"), D=("D", "first"), rank_def=("rank_def", "mean"),
                                   kappa_med=("kappa", "median"), top_over_mean=("top_over_mean", "median")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()
    diag() if a.diag else run(a.dataset, a.eval_seeds, a.tune_seeds)
