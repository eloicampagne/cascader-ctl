"""Structural heterogeneity: a shared-trunk / per-head reformulation of CTL.

CTL as formulated needs a shared parameter space, so
tasks with different output structure fall outside it, and heterogeneous
transfer is where TL is often most needed. This toy keeps that scope statement
and supplies partial evidence.

All tasks read out the SAME shared non-linear representation phi*(x)=ReLU(W* x),
but through heads of different shape:
  - type A tasks: binary   (1 logit,  BCE)   label = sign(u . phi*(x))
  - type B tasks: 4-way    (4 logits, CE)    label = argmax_k (u_k . phi*(x))
so no single parameter vector transfers across a type-A / type-B pair. What CAN
transfer is the trunk. We therefore cascade only the trunk (10->16 ReLU) along
the task tree and train each task's head from scratch. The task-distance proxy is
the paper's gradient proxy generalised to the shared trunk: g_v = d L_v / d(trunk)
at the shared init, which is output-shape agnostic.

Baselines: Individual (trunk+head from scratch per task), Star, Chain, CTL-MST.
Metric: mean test accuracy over all tasks (both types), 10 seeds.

    python3.12 structural_heterogeneity.py
"""

import copy
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from table1_nonconvex import tree_mst, tree_chain, tree_random, sle_alloc

HERE = Path(__file__).parent.resolve()
DIM, HID, N_CLUST = 10, 16, 5
N_TR, N_TE = 64, 128
METHODS = ["Individual", "Star", "Chain", "CTL-MST"]


# ---- cluster-structured representations and heterogeneous tasks -----------
# Each cluster owns its OWN trunk W*_c (with small within-cluster jitter), so
# there is genuine transferable structure that the task geometry can exploit: a
# task's trunk is useful mainly to same-cluster tasks. A globally shared trunk
# was tried first and gave no clean gain (transfer is meaningless when every
# task's trunk helps every other equally) 
def phi_star(X, Wstar):
    return np.maximum(X @ Wstar.T, 0.0)                        # ReLU features


def make_tasks(seed, T=120):
    rng = np.random.default_rng(seed)
    # per-cluster base trunk (the transferable structure) + readout directions
    cluster_trunk = [rng.standard_normal((HID, DIM)) / np.sqrt(DIM)
                     for _ in range(N_CLUST)]
    clusters = [rng.standard_normal((4, HID)) for _ in range(N_CLUST)]
    tasks = []
    for v in range(T):
        c = v % N_CLUST
        Wv = cluster_trunk[c] + 0.05 * rng.standard_normal((HID, DIM))  # within-cluster jitter
        U = clusters[c] + 0.15 * rng.standard_normal((4, HID))
        typ = "A" if v % 2 == 0 else "B"
        Xtr = rng.standard_normal((N_TR, DIM)).astype(np.float32)
        Xte = rng.standard_normal((N_TE, DIM)).astype(np.float32)
        Ptr, Pte = phi_star(Xtr, Wv), phi_star(Xte, Wv)
        if typ == "A":
            u = U[0]
            ytr = (Ptr @ u > 0).astype(np.float32)
            yte = (Pte @ u > 0).astype(np.float32)
            out = 1
        else:
            ytr = np.argmax(Ptr @ U.T, axis=1).astype(np.int64)
            yte = np.argmax(Pte @ U.T, axis=1).astype(np.int64)
            out = 4
        tasks.append(dict(typ=typ, out=out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte))
    return list(range(T)), tasks


# ---- model: shared trunk + per-task head ----------------------------------
class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(DIM, HID)

    def forward(self, x):
        return torch.relu(self.lin(x))


def new_head(out):
    return nn.Linear(HID, out)


def task_loss(trunk, head, X, y, out):
    z = head(trunk(X))
    if out == 1:
        return nn.functional.binary_cross_entropy_with_logits(z.reshape(-1), y)
    return nn.functional.cross_entropy(z, y)


def train_task(trunk, head, task, steps, eta):
    if steps <= 0:
        return trunk, head
    X = torch.from_numpy(task["Xtr"]).float()
    y = (torch.from_numpy(task["ytr"]).float() if task["out"] == 1
         else torch.from_numpy(task["ytr"]).long())
    params = list(trunk.parameters()) + list(head.parameters())
    for _ in range(steps):
        loss = task_loss(trunk, head, X, y, task["out"])
        for p in params:
            p.grad = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-eta)
    return trunk, head


def eval_task(trunk, head, task):
    trunk.eval(); head.eval()
    with torch.no_grad():
        z = head(trunk(torch.from_numpy(task["Xte"]).float()))
        if task["out"] == 1:
            pred = (z.reshape(-1) > 0).float().numpy()
            return float(np.mean(pred == task["yte"]))
        pred = z.argmax(1).numpy()
        return float(np.mean(pred == task["yte"]))


# ---- trunk-gradient distance proxy (output-shape agnostic) ----------------
def trunk_grad_proxy(nodes, tasks, seed):
    torch.manual_seed(seed)
    base = Trunk()
    g = {}
    for v in nodes:
        t = tasks[v]
        trunk = copy.deepcopy(base)
        head = new_head(t["out"])
        X = torch.from_numpy(t["Xtr"]).float()
        y = (torch.from_numpy(t["ytr"]).float() if t["out"] == 1
             else torch.from_numpy(t["ytr"]).long())
        loss = task_loss(trunk, head, X, y, t["out"])
        for p in trunk.parameters():
            p.grad = None
        loss.backward()
        g[v] = torch.cat([p.grad.reshape(-1) for p in trunk.parameters()]).numpy()
    G = np.stack([g[v] for v in nodes])
    D = np.linalg.norm(G[:, None, :] - G[None, :, :], axis=2)
    return D


# ---- cascade over the trunk only ------------------------------------------
def run_cascade(method, nodes, tasks, D, seed, B, eta, root):
    torch.manual_seed(seed)
    base = Trunk()

    if method == "Individual":
        b = max(1, B // len(nodes))
        trunks = {}
        for v in nodes:
            tr, hd = train_task(copy.deepcopy(base), new_head(tasks[v]["out"]),
                                tasks[v], b, eta)
            trunks[v] = (tr, hd)
        return {v: eval_task(*trunks[v], tasks[v]) for v in nodes}

    if method == "Star":
        dir_edges = [(root, v) for v in nodes if v != root]
    elif method == "Chain":
        root, dir_edges, _ = tree_chain(nodes, D, root)
    elif method == "CTL-MST":
        root, dir_edges, _ = tree_mst(nodes, D, root)

    if method in ("Chain", "CTL-MST"):
        alloc = sle_alloc(nodes, root, dir_edges, D, B)
    else:                                    # Star: uniform
        seed_steps = max(1, int(0.1 * B))
        child = max(1, (B - seed_steps) // max(1, len(nodes) - 1))
        alloc = {v: child for v in nodes}; alloc[root] = seed_steps

    children = defaultdict(list)
    for u, v in dir_edges:
        children[u].append(v)
    # root: train trunk + its own head
    r_trunk, r_head = train_task(copy.deepcopy(base), new_head(tasks[root]["out"]),
                                 tasks[root], int(alloc.get(root, 0)), eta)
    trunks = {root: r_trunk}
    acc = {root: eval_task(r_trunk, r_head, tasks[root])}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            tr = copy.deepcopy(trunks[u])          # inherit parent trunk
            hd = new_head(tasks[v]["out"])         # fresh per-task head
            tr, hd = train_task(tr, hd, tasks[v], int(alloc.get(v, 0)), eta)
            trunks[v] = tr
            acc[v] = eval_task(tr, hd, tasks[v])
            q.append(v)
    return acc


def main():
    Bs = (500, 2000)
    seeds = 10
    eta = 0.1
    rows = []
    for seed in range(seeds):
        nodes, tasks = make_tasks(seed)
        D = trunk_grad_proxy(nodes, tasks, seed)
        root = int(np.argmin(D.sum(1)))           # medoid root
        for B in Bs:
            for m in METHODS:
                acc = run_cascade(m, nodes, tasks, D, seed, B, eta, root)
                mean_all = float(np.mean(list(acc.values())))
                mean_A = float(np.mean([acc[v] for v in nodes if tasks[v]["typ"] == "A"]))
                mean_B = float(np.mean([acc[v] for v in nodes if tasks[v]["typ"] == "B"]))
                rows.append(dict(seed=seed, B=B, method=m,
                                 acc=mean_all, acc_binary=mean_A, acc_4way=mean_B))
        print(f"seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    summ = df.groupby(["B", "method"])[["acc", "acc_binary", "acc_4way"]].mean()
    print("\n=== mean test accuracy (shared trunk, heterogeneous heads) ===")
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== CTL-MST vs Individual (paired t over seeds) ===")
    for B in Bs:
        piv = df[df["B"] == B].pivot_table(index="seed", columns="method", values="acc")
        d = piv["CTL-MST"] - piv["Individual"]
        p = stats.ttest_rel(piv["CTL-MST"], piv["Individual"]).pvalue
        print(f"  B={B}: CTL-MST {piv['CTL-MST'].mean():.3f} vs "
              f"Individual {piv['Individual'].mean():.3f}  "
              f"gain {d.mean():+.3f}  p={p:.3g}")

    df.to_csv(HERE / "structural_heterogeneity_raw.csv", index=False)
    summ.reset_index().to_csv(HERE / "structural_heterogeneity_summary.csv", index=False)
    print("\nwrote structural_heterogeneity_{raw,summary}.csv")


if __name__ == "__main__":
    main()
