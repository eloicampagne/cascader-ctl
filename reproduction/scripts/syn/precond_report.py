"""Analyse de precond_control.py : erreur (RMSE ou exactitude) et cout en equivalents-pas,
par jeu, methode, variante et budget ; pas choisi sur les graines de reglage."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
HI = {"FMNIST", "CIFAR", "CelebA"}
rows = []
for f in sorted(HERE.glob("precond_*.csv")):
    if f.stem in ("precond_diag", "precond_Syn_grid05", "precond_summary"):
        continue
    d = pd.read_csv(f)
    ds = d.dataset.iloc[0]
    for (v, b, m), g in d.groupby(["variant", "b", "method"]):
        t = g[g.split == "tune"].groupby("eta").value.agg(lambda x: x.mean() if x.notna().all() else np.nan).dropna()  # pas divergent exclu
        if t.empty:
            continue
        e = t.idxmax() if ds in HI else t.idxmin()
        x = g[(g.split == "eval") & (g.eta == e)].value.dropna()
        if x.empty:
            continue
        c0 = g.iloc[0]
        steps = b * (c0.per_step if v == "precond" else 1.0) + (c0.setup if v == "precond" else 0.0)
        rows.append(dict(dataset=ds, method=m, variant=v, b=b, eta=e, n=len(x),
                         value=x.mean() * (100 if ds in HI else 1), se=x.std(ddof=1) / np.sqrt(len(x)) * (100 if ds in HI else 1),
                         cost=steps))
r = pd.DataFrame(rows)
r.to_csv(HERE / "precond_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 400):
    for ds, g in r.groupby("dataset"):
        print(f"\n=== {ds} (cout de preparation {g[g.variant=='precond'].cost.min() - g[g.variant=='precond'].b.min():.1f} pas) ===")
        print(g.pivot_table(index=["variant", "b"], columns="method", values="value").round(3).to_string())
        print(g[g.variant == "precond"].groupby("b").cost.first().round(1).to_dict())
