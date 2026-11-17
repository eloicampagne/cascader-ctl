"""Regression de la section 4.5 : log(err / err_Individual) ~ distance au parent + profondeur + log budget.

Syn : syn/mech_raw.csv (sortie de syn/syn_suite.py) (generateur de Syn). WEAVE-UK et CIFAR : mechanism_edge_vs_depth*_raw.csv
et mechanism_individual_raw.csv ; sur CIFAR, l'erreur est le taux de mauvaise classification.
Variables standardisees par cascade (graine), coefficients moyennes sur les graines.

    python3.12 mechanism_logratio.py
"""
import numpy as np
import pandas as pd

syn = pd.read_csv("syn/mech_raw.csv").assign(dataset="Syn")
a = pd.read_csv("mechanism_edge_vs_depth_raw.csv")
b = pd.read_csv("mechanism_edge_vs_depth_2opt_raw.csv")
ind = pd.read_csv("mechanism_individual_raw.csv")
rw = pd.concat([a[(a.dataset != "Syn-10") & (a.method != "CTL-MSTc")],
                b[b.dataset != "Syn-10"]]).merge(ind, on=["dataset", "rep", "B", "node"])
c = rw.dataset == "CIFAR"
rw.loc[c, "err"] = 1 - rw.loc[c, "err"]
rw.loc[c, "err_ind"] = 1 - rw.loc[c, "err_ind"]
z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
out = []
for (ds, m, B), g in pd.concat([rw, syn]).groupby(["dataset", "method", "B"]):
    co = [np.linalg.lstsq(np.column_stack([np.ones(len(h)), z(h.delta), z(h.depth), z(np.log(h.budget))]),
                          z(np.log(h.err / h.err_ind)), rcond=None)[0][1:] for _, h in g.groupby("rep")]
    out.append((ds, m, B, *np.mean(co, 0)))
R = pd.DataFrame(out, columns=["dataset", "method", "B", "distance", "depth", "log_budget"])
print(R.round(3).to_string())
print(f"{len(R)} cellules ; distance > 0 dans {(R.distance > 0).sum()} (moyenne {R.distance.mean():.3f}) ;"
      f" profondeur > 0 dans {(R.depth > 0).sum()} (moyenne {R.depth.mean():.3f})")
