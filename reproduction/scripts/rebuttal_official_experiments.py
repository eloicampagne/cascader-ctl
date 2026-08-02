"""
Official NeurIPS 2026 rebuttal experiments for the CTL paper (Submission 6227).

Addresses the official reviews (CTL_reviews_neurips.pdf) following
`revision_roadmap_official.md`:

  EXP-A1  Non-convex CTL on WEAVE-UK (MLP end-to-end)      item 1  (n7Am W2/W5/Q2, fdRt W1/Q1/L1, Lufy W2)
  EXP-A2  Non-convex CTL on nonlinear synthetic teachers    item 1 + item 10 (proxy off-linearity)
  EXP-B   CTL-vs-direct-TL boundary, bound looseness        item 3  (n7Am W4/Q4, Lufy W1)
  EXP-C   eta / tau sensitivity                             item 2  (n7Am W6/Q3, Lufy W3)
  EXP-D   Root sensitivity + free re-rooting                item 6  (fdRt W3/Q2/L2)
  EXP-E   Source-solution perturbation                      item 13 (Lufy W5)
  EXP-F   Heterogeneity probe (cluster decoupling)          item 4  (n7Am W1/Q1)

Run:  python3.12 rebuttal_official_experiments.py --exp all
All results saved as CSV in this directory. CPU is forced (12x faster than MPS
for these model sizes).
"""

import sys
import copy
import json
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
from data import generate_clustered_tasks, load_weave_data, normalize_xy, denormalize_y
from graph import (
    compute_gradient_distance,
    mst_from_distance_matrix,
    root_tree,
    alloc_subtree_log_edge,
    cascade_train_tree_with_allocation,
    build_mstc_tree_with_distance,
)
from trainer import train_model_sgd, clone_model
from models import LinearModel, SimpleMLP

DEVICE = torch.device("cpu")  # 12x faster than MPS at this model scale

WEAVE_TRAIN = str(CTL_DIR / "data" / "train_weave_500.csv")
WEAVE_TEST = str(CTL_DIR / "data" / "test_weave_500.csv")

# ---------------------------------------------------------------------------
# Shared helpers (aligned with neurips_rebuttal_experiments.py, May version)
# ---------------------------------------------------------------------------

def _normalize(nodes, X_tr, y_tr, X_te):
    Xn_tr, yn_tr, Xn_te, stats = {}, {}, {}, {}
    for v in nodes:
        Xn_tr[v], yn_tr[v], st = normalize_xy(X_tr[v], y_tr[v])
        stats[v] = st
        Xn_te[v] = (X_te[v] - st["X_mean"]) / st["X_std"]
    return Xn_tr, yn_tr, Xn_te, stats


def _eval_rmse(models, Xn_te, y_te, stats, nodes):
    rmses = []
    for v in nodes:
        m = models[v]
        m.eval()
        with torch.no_grad():
            pred_n = m(torch.from_numpy(Xn_te[v]).float().to(DEVICE)).cpu().numpy().reshape(-1)
        pred = denormalize_y(pred_n, stats[v])
        rmses.append(float(np.sqrt(np.mean((y_te[v].reshape(-1) - pred.reshape(-1)) ** 2))))
    return float(np.mean(rmses)), float(np.std(rmses))


def _make_model(model_class, d, hidden=32):
    if model_class == "linear":
        return LinearModel(d)
    if model_class == "mlp":
        return SimpleMLP(d, hidden=hidden)
    raise ValueError(model_class)


def _build_mst(nodes, D):
    edges = mst_from_distance_matrix(nodes, D)
    root = nodes[int(np.argmin(D.sum(axis=1)))]  # medoid
    root, dir_edges, parent_map = root_tree(edges, root)
    return root, dir_edges, parent_map, edges


def _run_individual(nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, B, eta, model_class="linear"):
    b = max(1, B // len(nodes))
    d = Xn_tr[nodes[0]].shape[1]
    models = {}
    for v in nodes:
        m = _make_model(model_class, d)
        models[v] = train_model_sgd(m, Xn_tr[v], yn_tr[v], steps=b, eta=eta, device=DEVICE)
    return _eval_rmse(models, Xn_te, y_te, stats, nodes)


def _run_star(nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B, eta, model_class="linear"):
    b = max(1, B // len(nodes))
    src = nodes[int(np.argmin(D.sum(axis=1)))]
    d = Xn_tr[nodes[0]].shape[1]
    src_m = _make_model(model_class, d)
    src_m = train_model_sgd(src_m, Xn_tr[src], yn_tr[src], steps=b, eta=eta, device=DEVICE)
    models = {}
    for v in nodes:
        m = copy.deepcopy(src_m)
        models[v] = train_model_sgd(m, Xn_tr[v], yn_tr[v], steps=b, eta=eta, device=DEVICE)
    return _eval_rmse(models, Xn_te, y_te, stats, nodes)


def _run_ctl_tree(nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B, eta,
                  model_class="linear", root=None, dir_edges=None, parent_map=None):
    """CTL cascade with SLE allocation on the given (or default medoid-MST) tree."""
    node_idx = {v: i for i, v in enumerate(nodes)}
    if root is None:
        root, dir_edges, parent_map, _ = _build_mst(nodes, D)
    alloc = alloc_subtree_log_edge(nodes, root, parent_map, D, node_idx, B)
    models, _ = cascade_train_tree_with_allocation(
        dir_edges, root, Xn_tr, yn_tr, alloc, eta=eta, device=DEVICE,
        model_class=model_class)
    return _eval_rmse(models, Xn_te, y_te, stats, nodes)


def _load_weave():
    X_tr_raw, y_tr_raw = load_weave_data(WEAVE_TRAIN)
    X_te_raw, y_te_raw = load_weave_data(WEAVE_TEST)
    nodes = sorted(set(X_tr_raw.keys()) & set(X_te_raw.keys()))
    return nodes, X_tr_raw, y_tr_raw, X_te_raw, y_te_raw


_WEAVE_CACHE = {}

def _weave_prepared():
    if "prep" not in _WEAVE_CACHE:
        nodes, X_tr, y_tr, X_te, y_te = _load_weave()
        Xn_tr, yn_tr, Xn_te, stats = _normalize(nodes, X_tr, y_tr, X_te)
        D = compute_gradient_distance(nodes, Xn_tr, yn_tr)
        _WEAVE_CACHE["prep"] = (nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D)
    return _WEAVE_CACHE["prep"]


def _synth_dataset(T, d, n_clusters, tau_within, tau_between, sigma, seed,
                   n_train=32, n_test=128):
    rng = np.random.default_rng(seed)
    thetas, _, _, (X_tr, y_tr), (X_te, y_te), cluster_ids = generate_clustered_tasks(
        T=T, d=d, n_train=n_train, n_test=n_test,
        n_clusters=min(n_clusters, T // 4),
        tau_within=tau_within, tau_between=tau_between,
        sigma=sigma, rng=rng,
    )
    nodes = list(range(T))
    return (nodes,
            {v: X_tr[v] for v in nodes}, {v: y_tr[v] for v in nodes},
            {v: X_te[v] for v in nodes}, {v: y_te[v] for v in nodes},
            thetas, cluster_ids)


# ===========================================================================
# EXP-A1 — Non-convex CTL on WEAVE-UK (item 1)
# ===========================================================================

def expA1_nonconvex_weave(Bs=(500, 1000, 2000), etas=(0.01, 0.05, 0.2), n_reps=3):
    """End-to-end SimpleMLP (d->32->1, ReLU) refinement at every node of the
    cascade. Same protocol as Table 6 / EXP6: gradient-distance MST, SLE
    allocation, per-node full-batch SGD. Nothing is convex: every local
    refinement trains all MLP weights."""
    print("\n=== EXP-A1: Non-convex CTL on WEAVE-UK (MLP end-to-end) ===")
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D = _weave_prepared()
    print(f"  WEAVE: T={len(nodes)}")

    rows = []
    for B in Bs:
        for eta in etas:
            for rep in range(n_reps):
                r = {"B": B, "eta": eta, "rep": rep}
                r["Individual-MLP"], _ = _run_individual(
                    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, B, eta, "mlp")
                r["Star-MLP"], _ = _run_star(
                    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B, eta, "mlp")
                r["CTL-MST-MLP"], _ = _run_ctl_tree(
                    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B, eta, "mlp")
                r["CTL-MST-Linear"], _ = _run_ctl_tree(
                    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B, eta, "linear")
                rows.append(r)
                print(f"  B={B} eta={eta} rep={rep}: "
                      f"Indiv={r['Individual-MLP']:.1f} Star={r['Star-MLP']:.1f} "
                      f"CTL-MLP={r['CTL-MST-MLP']:.1f} CTL-Lin={r['CTL-MST-Linear']:.1f}",
                      flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby(["B", "eta"]).agg(["mean", "std"])
    df.to_csv(HERE / "expA1_nonconvex_weave_raw.csv", index=False)
    summary.to_csv(HERE / "expA1_nonconvex_weave_summary.csv")
    print(summary.to_string())
    return df


# ===========================================================================
# EXP-A2 — Non-convex CTL on nonlinear synthetic teachers (items 1 + 10)
# ===========================================================================

def _gen_teacher_tasks(T, d, hidden_t, n_train, n_test, s_between, s_within,
                       sigma, n_clusters, seed):
    """Clustered 2-layer ReLU teacher networks; data is genuinely nonlinear."""
    rng = np.random.default_rng(seed)

    def flat_dim():
        return hidden_t * d + hidden_t + hidden_t + 1  # W1, b1, w2, b2

    def unpack(w):
        i = 0
        W1 = w[i:i + hidden_t * d].reshape(hidden_t, d); i += hidden_t * d
        b1 = w[i:i + hidden_t]; i += hidden_t
        w2 = w[i:i + hidden_t]; i += hidden_t
        b2 = w[i]
        return W1, b1, w2, b2

    def teacher_forward(w, X):
        W1, b1, w2, b2 = unpack(w)
        H = np.maximum(X @ W1.T + b1, 0.0)
        return H @ w2 + b2

    p = flat_dim()
    w_global = rng.normal(size=p)
    centers = w_global + rng.normal(scale=s_between, size=(n_clusters, p))
    cluster_ids = rng.integers(0, n_clusters, size=T)
    teachers = np.stack([centers[cluster_ids[t]] + rng.normal(scale=s_within, size=p)
                         for t in range(T)])

    nodes = list(range(T))
    X_tr, y_tr, X_te, y_te = {}, {}, {}, {}
    for v in nodes:
        Xa = rng.normal(size=(n_train, d))
        Xb = rng.normal(size=(n_test, d))
        X_tr[v] = Xa
        X_te[v] = Xb
        y_tr[v] = teacher_forward(teachers[v], Xa) + rng.normal(scale=sigma, size=n_train)
        y_te[v] = teacher_forward(teachers[v], Xb) + rng.normal(scale=sigma, size=n_test)
    return nodes, X_tr, y_tr, X_te, y_te, teachers, cluster_ids


def _random_tree(nodes, rng):
    order = list(nodes)
    rng.shuffle(order)
    root = order[0]
    parent_map, dir_edges = {}, []
    for i, v in enumerate(order[1:], start=1):
        p = order[rng.integers(0, i)]
        parent_map[v] = p
        dir_edges.append((p, v))
    return root, dir_edges, parent_map


def expA2_nonconvex_synthetic(n_reps=10, T=100, d=5, hidden_t=16,
                              B_mult=5, eta=0.05, sigma=0.1):
    """Nonlinear clustered teacher tasks; every method trains a SimpleMLP
    end-to-end. Also compares distance proxies off-linearity (item 10):
    gradient proxy vs oracle teacher-weight distance vs random tree."""
    print("\n=== EXP-A2: Non-convex CTL on nonlinear synthetic teachers ===")
    B = B_mult * T
    rows = []
    for seed in range(n_reps):
        nodes, X_tr, y_tr, X_te, y_te, teachers, _ = _gen_teacher_tasks(
            T, d, hidden_t, n_train=32, n_test=128,
            s_between=1.0, s_within=0.1, sigma=sigma, n_clusters=5, seed=seed)
        Xn_tr, yn_tr, Xn_te, stats = _normalize(nodes, X_tr, y_tr, X_te)

        D_grad = compute_gradient_distance(nodes, Xn_tr, yn_tr)
        D_orac = np.linalg.norm(teachers[:, None, :] - teachers[None, :, :], axis=-1)

        r = {"seed": seed}
        r["Individual-Linear"], _ = _run_individual(
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, B, eta, "linear")
        r["Individual-MLP"], _ = _run_individual(
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, B, eta, "mlp")
        r["Star-MLP"], _ = _run_star(
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, B, eta, "mlp")
        r["CTL-MST-MLP (gradient)"], _ = _run_ctl_tree(
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, B, eta, "mlp")
        r["CTL-MST-MLP (oracle)"], _ = _run_ctl_tree(
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_orac, B, eta, "mlp")
        rng = np.random.default_rng(1000 + seed)
        root_r, edges_r, pm_r = _random_tree(nodes, rng)
        r["CTL-RandomTree-MLP"], _ = _run_ctl_tree(
            nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, B, eta, "mlp",
            root=root_r, dir_edges=edges_r, parent_map=pm_r)
        rows.append(r)
        print("  seed %d: " % seed + "  ".join(
            f"{k}={v:.4f}" for k, v in r.items() if k != "seed"), flush=True)

    df = pd.DataFrame(rows)
    summary = df.drop(columns="seed").agg(["mean", "std"]).T
    summary.columns = ["mean_rmse", "std_rmse"]
    df.to_csv(HERE / "expA2_nonconvex_synth_raw.csv", index=False)
    summary.to_csv(HERE / "expA2_nonconvex_synth_summary.csv")
    print(summary.to_string())
    return df


# ===========================================================================
# EXP-B — CTL-vs-direct-TL boundary and bound looseness (item 3)
# ===========================================================================

def _gd_linear(theta, X, y, steps, eta):
    n = len(y)
    for _ in range(steps):
        theta = theta - (eta / n) * X.T @ (X @ theta - y)
    return theta


def expB_ctl_vs_direct(n_seeds=20, d=20, n=32, sigma=0.1, eta=0.5,
                       bs=(5, 20, 80),
                       rhos=(0.5, 1.0, 2.0, 4.0, 8.0),
                       gammas=(0.0, 0.25, 0.5, 1.0, 2.0),
                       ks=(1, 3, 7),
                       cond_hets=(False, True)):
    """Controlled path geometry matching Thm 2.3's accounting exactly: in the
    many-task setting every node (intermediates included) receives the same
    refinement budget b, because intermediates are themselves target tasks.
    Direct TL trains the target for b steps from theta_s*; CTL trains it for
    the same b steps but warm-starts from the cascaded parent.

    Geometry: source optimum at 0, target optimum at distance rho along unit
    direction u; k intermediates evenly spaced on the segment, displaced
    orthogonally by gamma*rho (gamma=0: aligned relays; gamma large: weakly
    related intermediates — Lufy's third failure regime).

    cond_het=True gives intermediates ill-conditioned data (n_i = d + 4) and
    the target well-conditioned data (n_t = 4d), producing rho_v << rho_max —
    Lufy's first failure regime and the looseness source named in Remark 2.4.

    Also computes the Thm 2.3 relaxed bounds with exact per-task contraction
    rates rho_v = ||I - (eta/n) X_v^T X_v||_2 to quantify bound looseness
    (item 3 / n7Am W4 / Lufy W1)."""
    print("\n=== EXP-B: CTL vs direct TL boundary ===")
    rows = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        u = rng.normal(size=d); u /= np.linalg.norm(u)
        for cond_het in cond_hets:
            for rho in rhos:
                for gamma in gammas:
                    for k in ks:
                        theta_s = np.zeros(d)
                        theta_t = rho * u
                        inter = []
                        for j in range(1, k + 1):
                            w = rng.normal(size=d)
                            w -= (w @ u) * u
                            w /= np.linalg.norm(w)
                            inter.append((j / (k + 1)) * rho * u + gamma * rho * w)
                        path_optima = [theta_s] + inter + [theta_t]

                        n_inter = d + 4 if cond_het else n
                        n_target = 4 * d if cond_het else n
                        sizes = [n] + [n_inter] * k + [n_target]

                        Xs, ys, rhos_v = [], [], []
                        for th, n_v in zip(path_optima, sizes):
                            X = rng.normal(size=(n_v, d))
                            y = X @ th + rng.normal(scale=sigma, size=n_v)
                            M = np.eye(d) - (eta / n_v) * X.T @ X
                            Xs.append(X); ys.append(y)
                            rhos_v.append(float(np.linalg.norm(M, 2)))

                        for b in bs:
                            # source assumed exact (tilde-theta_s = theta_s*)
                            th_direct = _gd_linear(theta_s.copy(), Xs[-1], ys[-1], b, eta)
                            th = theta_s.copy()
                            for i in range(1, k + 2):
                                th = _gd_linear(th, Xs[i], ys[i], b, eta)

                            err_direct = float(np.linalg.norm(th_direct - theta_t))
                            err_ctl = float(np.linalg.norm(th - theta_t))

                            deltas = [float(np.linalg.norm(path_optima[i] - path_optima[i - 1]))
                                      for i in range(1, k + 2)]
                            delta_max = max(deltas)
                            rho_max = max(rhos_v[1:])
                            rho_t = rhos_v[-1]
                            m = k + 1
                            bound_ctl = delta_max * (rho_max ** b) * \
                                (1 - rho_max ** (m * b)) / max(1e-12, (1 - rho_max ** b))
                            bound_direct = (rho_t ** b) * rho

                            rows.append({
                                "seed": seed, "cond_het": cond_het, "rho": rho,
                                "gamma": gamma, "k": k, "b": b,
                                "err_direct": err_direct, "err_ctl": err_ctl,
                                "ctl_minus_direct": err_ctl - err_direct,
                                "bound_ctl": bound_ctl, "bound_direct": bound_direct,
                                "bound_predicts_ctl": bound_ctl < bound_direct,
                                "loose_ctl": bound_ctl / max(err_ctl, 1e-12),
                                "loose_direct": bound_direct / max(err_direct, 1e-12),
                                "rho_max": rho_max, "rho_target": rho_t,
                                "delta_max": delta_max,
                            })
        print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["cond_het", "b", "rho", "gamma", "k"])
               [["err_direct", "err_ctl", "ctl_minus_direct",
                 "loose_ctl", "loose_direct"]]
               .mean().reset_index())
    summary["ctl_wins"] = summary["ctl_minus_direct"] < 0
    # agreement between bound prediction and empirical winner
    agree = (df.groupby(["cond_het", "b"])
             .apply(lambda g: float(np.mean(g["bound_predicts_ctl"]
                                            == (g["ctl_minus_direct"] < 0))))
             .rename("bound_agreement").reset_index())
    df.to_csv(HERE / "expB_boundary_raw.csv", index=False)
    summary.to_csv(HERE / "expB_boundary_summary.csv", index=False)
    agree.to_csv(HERE / "expB_bound_agreement.csv", index=False)
    print(agree.to_string(index=False))
    print(summary.head(40).to_string(index=False))
    return summary


# ===========================================================================
# EXP-C — eta and tau sensitivity (item 2)
# ===========================================================================

def expC_eta_tau_sensitivity(etas=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
                             tau_mults=(0.25, 0.5, 1.0, 2.0, 4.0),
                             B=1000, n_reps=3, n_seeds=10):
    print("\n=== EXP-C: eta / tau sensitivity ===")
    rows_eta = []

    # --- WEAVE eta sweep ---
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D = _weave_prepared()
    for eta in etas:
        for rep in range(n_reps):
            r = {"dataset": "WEAVE-UK", "eta": eta, "rep": rep}
            r["Individual"], _ = _run_individual(
                nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, B, eta, "linear")
            r["CTL-MST"], _ = _run_ctl_tree(
                nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B, eta, "linear")
            rows_eta.append(r)
        m = np.mean([x["CTL-MST"] for x in rows_eta if x["eta"] == eta
                     and x["dataset"] == "WEAVE-UK"])
        print(f"  WEAVE eta={eta}: CTL-MST={m:.1f}", flush=True)

    # --- synthetic eta sweep ---
    for seed in range(n_seeds):
        s_nodes, X_tr, y_tr, X_te, y_te_s, _, _ = _synth_dataset(
            100, 5, 5, 2.0, 8.0, 0.1, seed)
        Xn, yn, Xte_n, st = _normalize(s_nodes, X_tr, y_tr, X_te)
        Ds = compute_gradient_distance(s_nodes, Xn, yn)
        for eta in etas:
            r = {"dataset": "Synthetic", "eta": eta, "rep": seed}
            r["Individual"], _ = _run_individual(
                s_nodes, Xn, yn, Xte_n, y_te_s, st, 500, eta, "linear")
            r["CTL-MST"], _ = _run_ctl_tree(
                s_nodes, Xn, yn, Xte_n, y_te_s, st, Ds, 500, eta, "linear")
            rows_eta.append(r)
        print(f"  synthetic seed {seed} done", flush=True)

    df_eta = pd.DataFrame(rows_eta)
    sum_eta = (df_eta.groupby(["dataset", "eta"])[["Individual", "CTL-MST"]]
               .agg(["mean", "std"]))
    df_eta.to_csv(HERE / "expC_eta_raw.csv", index=False)
    sum_eta.to_csv(HERE / "expC_eta_summary.csv")
    print(sum_eta.to_string())

    # --- WEAVE tau sweep for CTL-MSTc (lambda fixed at paper default 0.75) ---
    rows_tau = []
    T = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}
    tau_med = float(np.median(D[np.triu_indices(T, k=1)]))
    for c in tau_mults:
        for rep in range(n_reps):
            try:
                root, parent_map, dir_edges, _ = build_mstc_tree_with_distance(
                    V=nodes, y=yn_tr, X=Xn_tr, distance_type="gradient",
                    lambda_reach=0.75, tau=c * tau_med)
                alloc = alloc_subtree_log_edge(nodes, root, parent_map, D, node_idx, B)
                models, _ = cascade_train_tree_with_allocation(
                    dir_edges, root, Xn_tr, yn_tr, alloc, eta=0.01, device=DEVICE)
                rmse, _ = _eval_rmse(models, Xn_te, y_te, stats, nodes)
            except Exception as e:
                print(f"  tau={c}x rep={rep} error: {e}")
                rmse = float("nan")
            rows_tau.append({"tau_mult": c, "tau": c * tau_med, "rep": rep,
                             "mean_rmse": rmse})
        m = np.nanmean([x["mean_rmse"] for x in rows_tau if x["tau_mult"] == c])
        print(f"  WEAVE tau={c}x median: CTL-MSTc={m:.1f}", flush=True)

    df_tau = pd.DataFrame(rows_tau)
    sum_tau = df_tau.groupby("tau_mult")["mean_rmse"].agg(["mean", "std"]).reset_index()
    df_tau.to_csv(HERE / "expC_tau_raw.csv", index=False)
    sum_tau.to_csv(HERE / "expC_tau_summary.csv", index=False)
    print(sum_tau.to_string(index=False))
    return df_eta, df_tau


# ===========================================================================
# EXP-D — Root sensitivity on the fixed MST (item 6)
# ===========================================================================

def _tree_adjacency(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _eccentricities(nodes, edges):
    adj = _tree_adjacency(edges)
    ecc = {}
    for s in nodes:
        depth = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in depth:
                    depth[w] = depth[u] + 1
                    q.append(w)
        ecc[s] = max(depth.values())
    return ecc


def expD_root_sensitivity(B=1000, n_reps=3, n_random=5, n_seeds=10):
    """The undirected MST does not depend on the root; re-rooting only changes
    the cascade orientation and the SLE allocation. Depth-central rooting is
    therefore obtainable at zero extra graph-construction cost (one O(T^2)
    BFS pass), which is the 'approximate iterative scheme' fdRt asks about,
    collapsed to a single exact step."""
    print("\n=== EXP-D: Root sensitivity ===")
    rows = []

    def run_all_roots(tag, nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B_run, reps, seed=0):
        undirected = mst_from_distance_matrix(nodes, D)
        ecc = _eccentricities(nodes, undirected)
        medoid = nodes[int(np.argmin(D.sum(axis=1)))]
        depth_central = min(nodes, key=lambda v: ecc[v])
        worst = max(nodes, key=lambda v: ecc[v])
        rng = np.random.default_rng(seed)
        choices = [("medoid", medoid), ("depth-central", depth_central),
                   ("worst-leaf", worst)]
        choices += [(f"random-{i}", nodes[rng.integers(0, len(nodes))])
                    for i in range(n_random)]
        for name, root0 in choices:
            root, dir_edges, parent_map = root_tree(undirected, root0)
            for rep in range(reps):
                rmse, _ = _run_ctl_tree(
                    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D, B_run, 0.01,
                    "linear", root=root, dir_edges=dir_edges, parent_map=parent_map)
                rows.append({"dataset": tag, "root": name, "ecc": ecc[root0],
                             "rep": rep, "seed": seed, "mean_rmse": rmse})
        return ecc[medoid], ecc[depth_central]

    # WEAVE
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D = _weave_prepared()
    e_med, e_dc = run_all_roots("WEAVE-UK", nodes, Xn_tr, yn_tr, Xn_te, y_te,
                                stats, D, B, n_reps)
    print(f"  WEAVE eccentricity: medoid={e_med}, depth-central={e_dc}")

    # Synthetic
    for seed in range(n_seeds):
        s_nodes, X_tr, y_tr, X_te, y_te_s, _, _ = _synth_dataset(
            100, 5, 5, 2.0, 8.0, 0.1, seed)
        Xn, yn, Xte_n, st = _normalize(s_nodes, X_tr, y_tr, X_te)
        Ds = compute_gradient_distance(s_nodes, Xn, yn)
        run_all_roots("Synthetic", s_nodes, Xn, yn, Xte_n, y_te_s, st, Ds,
                      500, 1, seed=seed)
        print(f"  synthetic seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df["root_kind"] = df["root"].str.replace(r"random-\d+", "random", regex=True)
    summary = (df.groupby(["dataset", "root_kind"])["mean_rmse"]
               .agg(["mean", "std", "count"]).reset_index())
    df.to_csv(HERE / "expD_root_raw.csv", index=False)
    summary.to_csv(HERE / "expD_root_summary.csv", index=False)
    print(summary.to_string(index=False))
    return df


# ===========================================================================
# EXP-E — Source-solution perturbation (item 13)
# ===========================================================================

def expE_source_perturbation(sigmas=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
                             n_seeds=10, T=100, B_mult=5, eta=0.01):
    """Perturb the root solution after its refinement (relative Gaussian noise
    of magnitude sigma_p * ||theta_root||), then cascade normally. Tests the
    theta_s-hat ~ theta_s-star assumption (Lufy W5 / Q)."""
    print("\n=== EXP-E: Source perturbation ===")
    B = B_mult * T
    rows = []
    for seed in range(n_seeds):
        nodes, X_tr, y_tr, X_te, y_te, _, _ = _synth_dataset(
            T, 5, 5, 2.0, 8.0, 0.1, seed)
        Xn_tr, yn_tr, Xn_te, stats = _normalize(nodes, X_tr, y_tr, X_te)
        D = compute_gradient_distance(nodes, Xn_tr, yn_tr)
        node_idx = {v: i for i, v in enumerate(nodes)}
        root, dir_edges, parent_map, _ = _build_mst(nodes, D)
        alloc = alloc_subtree_log_edge(nodes, root, parent_map, D, node_idx, B)
        children = defaultdict(list)
        for u, v in dir_edges:
            children[u].append(v)
        d = Xn_tr[nodes[0]].shape[1]
        rng = np.random.default_rng(2000 + seed)

        # train the root once per seed, then branch per sigma
        root_model = LinearModel(d)
        root_model = train_model_sgd(root_model, Xn_tr[root], yn_tr[root],
                                     steps=max(1, alloc.get(root, 1)),
                                     eta=eta, device=DEVICE)
        for sig in sigmas:
            m0 = clone_model(root_model)
            if sig > 0:
                with torch.no_grad():
                    flat = torch.cat([p.reshape(-1) for p in m0.parameters()])
                    scale = sig * float(flat.norm())
                    for p in m0.parameters():
                        noise = torch.from_numpy(
                            rng.normal(size=p.shape).astype(np.float32))
                        p.add_(noise * scale / max(1e-12, float(
                            np.sqrt(sum(q.numel() for q in m0.parameters())))))
            models = {root: m0}
            q = deque([root])
            while q:
                u = q.popleft()
                for v in children[u]:
                    cm = clone_model(models[u])
                    steps = max(0, alloc.get(v, 0))
                    if steps > 0:
                        cm = train_model_sgd(cm, Xn_tr[v], yn_tr[v],
                                             steps=steps, eta=eta, device=DEVICE)
                    models[v] = cm
                    q.append(v)
            rmse, _ = _eval_rmse(models, Xn_te, y_te, stats, nodes)
            rows.append({"seed": seed, "sigma_p": sig, "mean_rmse": rmse})
        print(f"  seed {seed}: " + "  ".join(
            f"s={r['sigma_p']}:{r['mean_rmse']:.3f}"
            for r in rows[-len(sigmas):]), flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby("sigma_p")["mean_rmse"].agg(["mean", "std"]).reset_index()
    df.to_csv(HERE / "expE_perturbation_raw.csv", index=False)
    summary.to_csv(HERE / "expE_perturbation_summary.csv", index=False)
    print(summary.to_string(index=False))
    return df


# ===========================================================================
# EXP-F — Heterogeneity probe: cluster decoupling (item 4)
# ===========================================================================

def expF_heterogeneity(tau_betweens=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
                       n_seeds=10, T=100, B_mult=5, eta=0.01):
    """Intentionally heterogeneous task collections: 5 clusters whose centers
    are progressively decoupled (tau_between up to 32x tau_within). Measures
    whether CTL degrades gracefully (gain -> 0) or suffers negative transfer,
    and how many MST edges cross cluster boundaries."""
    print("\n=== EXP-F: Heterogeneity probe ===")
    rows = []
    for tb in tau_betweens:
        for seed in range(n_seeds):
            nodes, X_tr, y_tr, X_te, y_te, thetas, cids = _synth_dataset(
                T, 5, 5, 2.0, tb, 0.1, seed)
            Xn_tr, yn_tr, Xn_te, stats = _normalize(nodes, X_tr, y_tr, X_te)
            D = compute_gradient_distance(nodes, Xn_tr, yn_tr)
            B = B_mult * T
            indiv, _ = _run_individual(nodes, Xn_tr, yn_tr, Xn_te, y_te,
                                       stats, B, eta, "linear")
            ctl, _ = _run_ctl_tree(nodes, Xn_tr, yn_tr, Xn_te, y_te,
                                   stats, D, B, eta, "linear")
            undirected = mst_from_distance_matrix(nodes, D)
            cross = sum(1 for u, v in undirected if cids[u] != cids[v])
            rows.append({"tau_between": tb, "seed": seed,
                         "Individual": indiv, "CTL-MST": ctl,
                         "gain_pct": 100.0 * (indiv - ctl) / indiv,
                         "cross_cluster_edges": cross,
                         "cross_edge_frac": cross / (T - 1)})
        sub = [r for r in rows if r["tau_between"] == tb]
        print(f"  tau_between={tb}: gain={np.mean([r['gain_pct'] for r in sub]):.1f}% "
              f"cross-edges={np.mean([r['cross_edge_frac'] for r in sub]):.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    summary = (df.groupby("tau_between")
               [["Individual", "CTL-MST", "gain_pct", "cross_edge_frac"]]
               .agg(["mean", "std"]))
    df.to_csv(HERE / "expF_heterogeneity_raw.csv", index=False)
    summary.to_csv(HERE / "expF_heterogeneity_summary.csv")
    print(summary.to_string())
    return df


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        help="A1,A2,B,C,D,E,F or 'all'")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    exps = (set(args.exp.split(",")) if args.exp != "all"
            else {"A1", "A2", "B", "C", "D", "E", "F"})

    torch.manual_seed(0)

    if args.fast:
        if "A1" in exps: expA1_nonconvex_weave(Bs=(1000,), etas=(0.05,), n_reps=1)
        if "A2" in exps: expA2_nonconvex_synthetic(n_reps=2, T=30)
        if "B" in exps: expB_ctl_vs_direct(n_seeds=2)
        if "C" in exps: expC_eta_tau_sensitivity(n_reps=1, n_seeds=2)
        if "D" in exps: expD_root_sensitivity(n_reps=1, n_random=2, n_seeds=2)
        if "E" in exps: expE_source_perturbation(n_seeds=2)
        if "F" in exps: expF_heterogeneity(n_seeds=2)
    else:
        if "A1" in exps: expA1_nonconvex_weave()
        if "A2" in exps: expA2_nonconvex_synthetic()
        if "B" in exps: expB_ctl_vs_direct()
        if "C" in exps: expC_eta_tau_sensitivity()
        if "D" in exps: expD_root_sensitivity()
        if "E" in exps: expE_source_perturbation()
        if "F" in exps: expF_heterogeneity()

    print("\n=== Done ===")
    print(f"Results in {HERE}")
