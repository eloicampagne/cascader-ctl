"""CTL avec un modele local de plusieurs millions de parametres.

Le reviewer Lufy objecte que nos experiences ne sont pas a grande echelle. Le
volet "nombre de taches" est traite par `scale_tasks.py`. Ici on traite l'autre
volet, la taille du modele, et on le fait sur des images reelles: un gros modele
sur la famille synthetique a grappes n'aurait aucun sens, la cible y etant
lineaire en dimension 3.

Protocole. Taches binaires CIFAR-10 ou Fashion-MNIST, exactement la composition
du papier (T=200, 20 grappes, 64 exemples d'entrainement par tache). Le modele
local est un ResNet pre-entraine sur ImageNet dont le tronc est gele jusqu'a
layer3 inclus et dont layer4 plus la tete sont entraines. Le tronc gele est
evalue une fois pour toutes et mis en cache, donc le cout par pas de gradient ne
porte que sur la partie entrainee, ce qui est exactement le regime que le
reviewer decrit: adapter un gros modele pre-entraine sous budget contraint.

Parametres entrainables par tache:

    tete lineaire sur PCA-100 (le papier)          ~3.3 k
    ResNet-18, layer4 + tete                        ~8.4 M
    ResNet-50, layer4 + tete                       ~15.0 M

La distance entre taches reste le proxy du papier, X^T y, calcule sur les
representations gelees moyennees spatialement. La racine est le medoide sur les
memes representations. Rien d'autre ne change.

    python3.12 large_model.py --backbone resnet18 --datasets CIFAR --reps 3
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
import torch.nn.functional as F

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (CONFIG, eval_all, cascade_train, train_steps,
                              sle_alloc, uniform_alloc,
                              tree_chain, tree_mst, tree_mstc, CTL_DIR)
from data import load_cifar10, load_fashion_mnist, generate_clustered_binary_tasks
from graph import compute_gradient_distance, select_seed_node

torch.set_num_threads(max(1, torch.get_num_threads()))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SHAPE = {"CIFAR": (3, 32, 32), "FMNIST": (1, 28, 28)}
RES = 64          # les images sont redimensionnees a 64x64 pour le tronc ImageNet
METHODS = ("Star", "CTL-MST")
SLE = {"CTL-MST", "Chain", "CTL-MSTc"}   # Star recoit l'allocation uniforme


# ---------------------------------------------------------------------------
# Tronc gele
# ---------------------------------------------------------------------------

def load_backbone(name):
    from torchvision.models import (resnet18, ResNet18_Weights,
                                    resnet50, ResNet50_Weights)
    if name == "resnet50":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.eval()
    trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                          net.layer1, net.layer2, net.layer3)
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk, net.layer4, net.fc.in_features


class TopBlock(nn.Module):
    """layer4 du ResNet plus une tete a un logit, seule partie entrainee."""

    def __init__(self, layer4, feat_dim):
        super().__init__()
        self.layer4 = copy.deepcopy(layer4)
        for p in self.layer4.parameters():
            p.requires_grad_(True)
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, z):
        h = self.layer4(z)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(h).reshape(-1)


@torch.no_grad()
def encode(trunk, X, bs=256):
    """X en (n, C, H, W) normalise ImageNet, renvoie la sortie de layer3 en fp16."""
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float()
        if xb.shape[1] == 1:
            xb = xb.repeat(1, 3, 1, 1)
        xb = F.interpolate(xb, size=(RES, RES), mode="bilinear",
                           align_corners=False)
        out.append(trunk(xb).half())
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Taches, composition identique au papier
# ---------------------------------------------------------------------------

def image_tasks_imagenet(ds, seed, T=200, n_clusters=20, n_tr=64, n_te=128):
    if ds == "FMNIST":
        Xtr, ytr, Xte, yte = load_fashion_mnist(str(CTL_DIR / "data" / "fashion_mnist"))
    else:
        Xtr, ytr, Xte, yte = load_cifar10(str(CTL_DIR / "data" / "cifar10"))
    c, h, w = SHAPE[ds]

    def prep(X):
        X = np.asarray(X, dtype=np.float32).reshape(len(X), c, h, w)
        if X.max() > 1.5:
            X = X / 255.0
        if c == 3:
            X = (X - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        else:
            X = (X - 0.5) / 0.25
        return X.reshape(len(X), -1)

    rng = np.random.default_rng(seed)
    Xt, yt, Xe, ye, _, _ = generate_clustered_binary_tasks(
        prep(Xtr), ytr, prep(Xte), yte, n_tasks=T, n_clusters=n_clusters,
        n_train_per_task=n_tr, n_test_per_task=n_te, rng=rng)
    nodes = list(range(T))
    s = (c, h, w)
    return (nodes,
            {v: Xt[v].reshape(-1, *s) for v in nodes}, {v: yt[v] for v in nodes},
            {v: Xe[v].reshape(-1, *s) for v in nodes}, {v: ye[v] for v in nodes})


def train_top(model, Z, y, steps, eta, l2):
    if steps <= 0:
        return model
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=eta, weight_decay=l2)
    zt = Z.float()
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        crit(model(zt), yt).backward()
        opt.step()
    return model


@torch.no_grad()
def acc_all(models, Zte, yte):
    a = []
    for v, m in models.items():
        m.eval()
        p = (m(Zte[v].float()) > 0).numpy().astype(int)
        a.append(float((p == np.asarray(yte[v]).reshape(-1)).mean()))
    return float(np.mean(a))


def cascade_top(edges, root, Z, y, alloc, eta, l2, factory):
    from collections import defaultdict, deque
    ch = defaultdict(list)
    for u, v in edges:
        ch[u].append(v)
    models = {root: train_top(factory(), Z[root], y[root],
                              int(max(0, alloc.get(root, 0))), eta, l2)}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in ch[u]:
            models[v] = train_top(copy.deepcopy(models[u]), Z[v], y[v],
                                  int(max(0, alloc.get(v, 0))), eta, l2)
            q.append(v)
    return models


def build(m, nodes, D, root, X, y, lam):
    if m == "Star":
        return root, [(root, v) for v in nodes if v != root]
    if m == "Chain":
        return tree_chain(nodes, D, root)[:2]
    if m == "CTL-MST":
        return tree_mst(nodes, D, root)[:2]
    return tree_mstc(nodes, D, root, X, y, lam)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50"])
    ap.add_argument("--datasets", default="CIFAR")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--budgets", default="500,2000")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    methods = a.methods.split(",")
    Bs = [int(b) for b in a.budgets.split(",")]

    trunk, layer4, feat_dim = load_backbone(a.backbone)
    ntrain = sum(p.numel() for p in layer4.parameters()) + feat_dim + 1
    print(f"{a.backbone}: {ntrain/1e6:.2f} M parametres entrainables par tache",
          flush=True)

    rows = []
    for ds in a.datasets.split(","):
        cfg = CONFIG[ds]
        for rep in range(a.reps):
            t0 = time.perf_counter()
            nodes, Xtr, ytr, Xte, yte = image_tasks_imagenet(ds, rep, T=a.T)
            Ztr = {v: encode(trunk, Xtr[v]) for v in nodes}
            Zte = {v: encode(trunk, Xte[v]) for v in nodes}
            # proxy de distance sur les representations gelees, moyennees
            Ftr = {v: Ztr[v].float().mean(dim=(2, 3)).numpy() for v in nodes}
            D = compute_gradient_distance(nodes, Ftr, ytr)
            root = select_seed_node(Ftr, ytr)
            print(f"  {ds} rep={rep} encode en {time.perf_counter()-t0:.0f}s",
                  flush=True)

            trees = {m: build(m, nodes, D, root, Ftr, ytr, cfg["mstc_lambda"])
                     for m in methods}
            f = lambda: TopBlock(layer4, feat_dim)

            for B in Bs:
                t0 = time.perf_counter()
                b = max(1, B // len(nodes))
                torch.manual_seed(rep)
                models = {v: train_top(f(), Ztr[v], ytr[v], b, a.eta, cfg["reg"])
                          for v in nodes}
                rows.append(dict(dataset=ds, backbone=a.backbone, rep=rep, B=B,
                                 method="Individual", value=acc_all(models, Zte, yte)))
                for m in methods:
                    r, edges = trees[m]
                    alloc = (sle_alloc(nodes, r, edges, D, B) if m in SLE else
                             uniform_alloc(nodes, r, edges, B,
                                           cfg["b_proto_frac"]))
                    torch.manual_seed(rep)
                    models = cascade_top(edges, r, Ztr, ytr, alloc, a.eta,
                                         cfg["reg"], f)
                    rows.append(dict(dataset=ds, backbone=a.backbone, rep=rep,
                                     B=B, method=m,
                                     value=acc_all(models, Zte, yte)))
                print(f"    B={B} en {time.perf_counter()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(
                    HERE / f"large_model_{a.backbone}_raw.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"large_model_{a.backbone}_raw.csv", index=False)
    print("\n=== exactitude moyenne ===", flush=True)
    print(df.pivot_table(index=["dataset", "B"], columns="method",
                         values="value").round(4).to_string(), flush=True)

    from scipy import stats as S
    print("\n=== ecart a Individual, apparie ===", flush=True)
    for (ds, B), g in df.groupby(["dataset", "B"]):
        base = g[g.method == "Individual"].sort_values("rep")["value"].values
        out = []
        for m in methods:
            cur = g[g.method == m].sort_values("rep")["value"].values
            p = S.ttest_rel(cur, base).pvalue if len(cur) > 1 else float("nan")
            out.append(f"{m} {100*(cur-base).mean()/base.mean():+.1f}% (p={p:.3f})")
        print(f"{ds:7s} B={B:5d}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
