"""Profondeur de CTL-MST sur le generateur de Syn sans entrainement, T jusqu'a 10000."""
import sys; sys.path.insert(0, "../../../cascader/src")
import numpy as np, pandas as pd
from cascader.data import load
from cascader.config import get_config
from cascader.methods import build_structure
cfg = get_config("Syn"); rows = []
for T in (200, 500, 1000, 2000, 5000, 10000):
    for s in range(5 if T <= 2000 else 3):
        c = load("Syn", seed=s, T=T)
        t = build_structure("CTL-MST", c, cfg, np.random.default_rng(s))
        rows.append(dict(T=T, seed=s, depth=max(t.depths().values())))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv("mst_depth.csv", index=False)
g = d.groupby("T").depth.mean(); print(g)
print("pente log-log", np.polyfit(np.log(g.index), np.log(g.values), 1)[0])
