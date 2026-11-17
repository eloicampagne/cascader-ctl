"""Experiences du papier refaites sur Syn, par le package cascader (branche paper-sync).

Syn remplace Syn-10 : cascader.data.load("Syn") = subspace_collection, T=200,
D=40, n_train=64, parametres de tache dans un sous-espace de dimension 3, K=5
grappes (tau_within=3, tau_between=10), sigma=5. Protocole des tables : pas
choisi par (methode, budget) sur les graines de reglage 500.., evaluation sur
0..19, allocation SLE a plancher d'un pas pour les cascades.

Parties (chacune ecrit ses lignes brutes, l'analyse est dans syn_report.py) :
    main     Tables 2 et 3 (lineaire, MLP), sept methodes + kNN-BFS, b=1,2,4
    closed   Table 17 : OLS, ridge par tache (LOO), ridge poolee, MTL-EP,
             MTL laplacien sur le MST, sur graines de reglage et d'evaluation
    global   Table 16 : g pas sur les taches concatenees puis b pas locaux
    largeT   Table 14 : T in {200,500,1000,2000}, b=4, MLP, profondeurs
    scaleT   Table scaleT : T in {100..5000}, MLP, B=1e4 fixe et b=4 fixe
    timing   Table timing : construction contre entrainement, CTL-MST lineaire
    mech     regression de l'erreur par tache sur distance au parent, profondeur
             (--alloc uniform : meme budget a chaque tache, sans compensation SLE)

    python3.12 syn_suite.py main --model linear --chunk 0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from cascader.cascade import cascade_train, eval_all, eval_task, train_steps  # noqa: E402
from cascader.config import get_config                                         # noqa: E402
from cascader.data import load, subspace_collection                            # noqa: E402
from cascader.distances import compute_gradient_distance                        # noqa: E402
from cascader.experiments._common import TUNE_OFFSET, etas_for, sweep          # noqa: E402
from cascader.methods import METHODS, allocate, build_structure, run_method    # noqa: E402
from cascader.models import model_factory                                      # noqa: E402
from cascader.rooting import medoid_root                                       # noqa: E402

torch.set_num_threads(1)
TUNE = list(range(TUNE_OFFSET, TUNE_OFFSET + 5))
EVAL = list(range(20))
CHUNKS3 = {0: [("tune", TUNE), ("eval", EVAL[:7])], 1: [("eval", EVAL[7:14])], 2: [("eval", EVAL[14:])]}
CFG = get_config("Syn")


def syn(seed, **kw):
    return load("Syn", seed=seed, **kw)


def out(name):
    return HERE / f"{name}.csv"


# ---------------------------------------------------------------- main (Tables 2, 3, kNN)

def part_main(model, chunk):
    rows = []
    for split, seeds in CHUNKS3[chunk]:
        sweep("Syn", seeds, split, rows, model=model, methods=METHODS + ["kNN-BFS"],
              b_per_task=(1, 2, 4), etas=etas_for("Syn", model), loader=syn,
              on_seed=lambda c, sp, s, t: print(f"main {model} {sp} {s} {t:.0f}s", flush=True))
        pd.DataFrame(rows).to_csv(out(f"main_{model}_c{chunk}"), index=False)


# ---------------------------------------------------------------- closed forms (Table 17)

def part_closed():
    import ols_baseline as OB
    import p_sweep as PS
    rows = []
    for split, seeds in (("tune", TUNE), ("eval", EVAL)):
        for s in seeds:
            c = syn(s)
            for name, h, v in PS.closed_forms(c):
                rows.append(dict(method=name, h=h, seed=s, split=split, value=v))
            rows.append(dict(method="Ridge", h=np.nan, seed=s, split=split, value=OB.fit_ridge(c)[0]))
            print(f"closed {split} {s}", flush=True)
    pd.DataFrame(rows).to_csv(out("closed"), index=False)


# ---------------------------------------------------------------- pooled pretraining (Table 16)

G_GRID = (0, 1, 2, 5, 10, 25, 50, 100, 250, 500)


def part_global(model, etas=None, suffix=""):
    rows = []
    for split, seeds in (("tune", TUNE), ("eval", EVAL)):
        for s in seeds:
            c = syn(s)
            fac = model_factory(model, c.d)
            Xp = np.concatenate([c.X_train[v] for v in c.nodes])
            yp = np.concatenate([np.asarray(c.y_train[v]).reshape(-1) for v in c.nodes])
            for eta in (etas or etas_for("Syn", model)):
                torch.manual_seed(s)
                pooled = fac()
                done = 0
                for g in G_GRID:
                    try:
                        train_steps(pooled, Xp, yp, g - done, eta, "regression")
                    except RuntimeError:
                        break
                    done = g
                    state = {k: t.clone() for k, t in pooled.state_dict().items()}
                    for b in (1, 2, 4):
                        models = {}
                        try:
                            for v in c.nodes:
                                m = fac()
                                m.load_state_dict(state)
                                models[v] = train_steps(m, c.X_train[v], c.y_train[v], b, eta, "regression")
                            val = eval_all(models, c.X_test, c.y_test, "regression", c.stats)
                        except RuntimeError:
                            val = np.nan
                        rows.append(dict(model=model, g=g, b=b, eta=eta, seed=s, split=split, value=val))
            print(f"global {model} {split} {s}", flush=True)
            pd.DataFrame(rows).to_csv(out(f"global_{model}{suffix}"), index=False)


# ---------------------------------------------------------------- large T (Table 14) and scaleT

def _depth_rows(c, methods, T, seed):
    rows = []
    for m in methods:
        if m in ("Individual", "Star"):
            continue
        tree = build_structure(m, c, CFG, np.random.default_rng(seed))
        d = tree.depths()
        rows.append(dict(T=T, seed=seed, method=m, depth_max=max(d.values())))
    return rows


def part_largeT(T):
    methods = ["Individual", "Chain", "Chain-2opt", "CTL-MST"]
    rows, drows = [], []
    loader = lambda s: syn(s, T=T)
    for split, seeds in (("tune", TUNE[:3]), ("eval", EVAL)):
        sweep("Syn", seeds, split, rows, model="mlp", methods=methods, b_per_task=(4,),
              etas=etas_for("Syn", "mlp"), loader=loader,
              on_seed=lambda c, sp, s, t: print(f"largeT {T} {sp} {s} {t:.0f}s", flush=True))
        if split == "eval":
            for s in seeds:
                drows += _depth_rows(loader(s), methods, T, s)
        pd.DataFrame(rows).assign(Tn=T).to_csv(out(f"largeT_{T}"), index=False)
    pd.DataFrame(drows).to_csv(out(f"largeT_{T}_depth"), index=False)


def part_scaleT(T):
    methods = ["Individual", "Star", "CTL-MST"]
    rows = []
    loader = lambda s: syn(s, T=T)
    bs = sorted({10_000 // T, 4})
    for split, seeds in (("tune", TUNE[:3]), ("eval", EVAL[:5])):
        sweep("Syn", seeds, split, rows, model="mlp", methods=methods, b_per_task=tuple(bs),
              etas=etas_for("Syn", "mlp"), loader=loader,
              on_seed=lambda c, sp, s, t: print(f"scaleT {T} {sp} {s} {t:.0f}s", flush=True))
        pd.DataFrame(rows).assign(Tn=T).to_csv(out(f"scaleT_{T}"), index=False)


# ---------------------------------------------------------------- timing

def part_timing():
    from cascader.data.synthetic import generate_subspace_tasks
    from cascader.data.transforms import normalize_xy
    from cascader.trees import build_mst
    from cascader.allocation import sle_alloc
    rows = []
    for T in (100, 200, 500, 1000, 2000):
        for s in range(50):
            rng = np.random.default_rng(s)
            _, _, (Xt, yt), _, _ = generate_subspace_tasks(T=T, rng=rng)
            nodes = list(range(T))
            X, y = {}, {}
            for v in nodes:
                X[v], y[v], _ = normalize_xy(Xt[v], yt[v])
            t0 = time.perf_counter()
            D = compute_gradient_distance(nodes, X, y)
            root = medoid_root(nodes, {v: Xt[v] for v in nodes})
            tree = build_mst(nodes, D, root)
            t_graph = time.perf_counter() - t0
            alloc = sle_alloc(tree, D, {v: i for i, v in enumerate(nodes)}, 2000)
            torch.manual_seed(s)
            t0 = time.perf_counter()
            cascade_train(tree, X, y, alloc, 0.3, model_factory("linear", Xt.shape[2]), "regression")
            rows.append(dict(T=T, seed=s, graph=t_graph, train=time.perf_counter() - t0))
        print(f"timing {T}", flush=True)
    pd.DataFrame(rows).to_csv(out("timing"), index=False)


# ---------------------------------------------------------------- mechanism (error vs parent distance)

def part_mech(alloc_rule=None):
    rows = []
    for rep in range(10):
        c = syn(rep)
        idx = {v: i for i, v in enumerate(c.nodes)}
        fac = model_factory("mlp", c.d)
        ind = {}
        for B in (500, 2000):                     # difficulte propre a chaque tache
            torch.manual_seed(rep)
            mi = run_method("Individual", c, B, CFG.mlp_eta, CFG, fac, rng=np.random.default_rng(rep))
            ind[B] = {v: eval_task(mi[v], c.X_test[v], c.y_test[v], "regression", c.stats[v]) for v in c.nodes}
        for m in ("Chain", "Chain-2opt", "CTL-MST"):
            tree = build_structure(m, c, CFG, np.random.default_rng(rep))
            depth = tree.depths()
            for B in (500, 2000):
                alloc = allocate(m, tree, c, B, CFG, allocation=alloc_rule)
                torch.manual_seed(rep)
                models = cascade_train(tree, c.X_train, c.y_train, alloc, CFG.mlp_eta, fac, "regression")
                for u, v in tree.edges:
                    rows.append(dict(rep=rep, method=m, B=B, node=v,
                                     err=eval_task(models[v], c.X_test[v], c.y_test[v], "regression", c.stats[v]),
                                     delta=c.D[idx[u], idx[v]], depth=depth[v], budget=alloc[v],
                                     err_ind=ind[B][v]))
        print(f"mech {rep}", flush=True)
    pd.DataFrame(rows).to_csv(out("mech_raw" + (f"_{alloc_rule}" if alloc_rule else "")), index=False)


# ---------------------------------------------------------------- tree against chain (Table 4)

TR_SCATTERS = (0.5, 2.0, 5.0, 20.0)
TR_BUDGETS = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6)
TR_ETAS = (0.1, 0.2, 0.3, 0.5, 0.7)


def part_tworegime(omega):
    """tau_within = 3 (Syn), tau_between = 3 omega : le rapport des dispersions vaut omega."""
    rows = []
    for split, seeds in (("tune", TUNE), ("eval", EVAL)):
        for s in seeds:
            c = subspace_collection(seed=s, tau_within=3.0, tau_between=3.0 * omega)
            fac = model_factory("linear", c.d)
            for b in TR_BUDGETS:
                B = int(round(b * c.T))
                for eta in TR_ETAS:
                    for m in ("Chain", "CTL-MST", "CTL-Forest"):
                        torch.manual_seed(s)
                        try:
                            models = run_method(m, c, B, eta, CFG, fac, rng=np.random.default_rng(s))
                            v = eval_all(models, c.X_test, c.y_test, "regression", c.stats)
                        except RuntimeError:
                            v = np.nan
                        rows.append(dict(omega=omega, b=b, eta=eta, method=m, seed=s, split=split, value=v))
            print(f"tworegime {omega} {split} {s}", flush=True)
            pd.DataFrame(rows).to_csv(out(f"tworegime_{omega}"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--model", default="linear")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--omega", type=float, default=2.0)
    ap.add_argument("--alloc", default=None)
    a = ap.parse_args()
    {"main": lambda: part_main(a.model, a.chunk), "closed": part_closed,
     "global": lambda: part_global(a.model), "largeT": lambda: part_largeT(a.T),
     "scaleT": lambda: part_scaleT(a.T), "timing": part_timing, "mech": lambda: part_mech(a.alloc),
     "tworegime": lambda: part_tworegime(a.omega)}[a.part]()


if __name__ == "__main__":
    main()
