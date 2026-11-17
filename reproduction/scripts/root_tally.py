"""Decomptes du paragraphe 'Empirical sensitivity to root selection' : 72 cellules
(Star, RandTree, Chain, Chain-2opt, CTL-kNN, CTL-MST) x (jeu, budget), test de Wilcoxon
apparie a p<0.05, racine medoide contre racine centrale (en sauts) et contre 1-mediane.
Les jeux autres que Syn viennent de _archive_syn10 (inchanges), Syn des relances.

    python3.12 root_tally.py            # Syn = generateur actuel
    python3.12 root_tally.py --old      # Syn-10, pour verifier les chiffres publies
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
HI = {"FMNIST", "CIFAR", "CelebA"}
M = ["Star", "RandTree", "Chain", "Chain-2opt", "CTL-kNN", "CTL-MST"]
A = "_archive_syn10/"
old = "--old" in sys.argv
def load(stem_list):
    arch = pd.concat([pd.read_csv(A + s) for s in stem_list])
    if old:
        return arch
    new = pd.concat([pd.read_csv(s) for s in stem_list])
    return pd.concat([arch[arch.dataset != "Syn-10"], new[new.dataset == "Syn-10"]])
def tally(df, alt):
    out = dict(n=0, imp=0, deg=0)
    for (ds, B, m), g in df[df.method.isin(M)].groupby(["dataset", "B", "method"]):
        p = g.pivot_table(index="rep", columns="root", values="value").dropna()
        x, y = p["medoid"], p[alt]
        out["n"] += 1
        if np.allclose(x, y):
            continue
        pv = wilcoxon(y, x).pvalue
        better = (y.mean() > x.mean()) if ds in HI else (y.mean() < x.mean())
        out["imp"] += pv < 0.05 and better
        out["deg"] += pv < 0.05 and not better
    out["none"] = out["n"] - out["imp"] - out["deg"]
    return out
dc = load(["root_depthcentral_raw.csv", "root_depthcentral_2opt_raw.csv"])
md = load(["root_median_raw.csv"])
print("centre :", tally(dc, "depth-central"))
print("1-mediane :", tally(md, "median"))
d = dc[dc.method.isin(["Chain", "Chain-2opt", "CTL-MST"])].groupby(["dataset", "method", "root"]).depth.mean().unstack()
print((d["depth-central"] / d["medoid"]).round(2))
