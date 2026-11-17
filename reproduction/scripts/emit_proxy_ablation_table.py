"""Corps LaTeX de tab:all_experiments a partir de proxy_ablation_floor1_*_raw.csv.

Pas retenu par (dataset, b, proxy) sur la moyenne des graines de reglage, score
lu sur les 20 graines d'evaluation. Regression : ecart relatif de RMSE au
gradient, en %. Classification : ecart d'exactitude en points, signe tel que
positif = le gradient gagne. Test t apparie unilateral par cellule (proxy contre
gradient), sans correction, pour le resume imprime.

    python3.12 emit_proxy_ablation_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ORDER = [("Syn-10", "Syn", "$200$"), ("WEAVE-UK", "WEAVE-UK", "$500$"),
         ("FMNIST", "Fashion-MNIST", "$200$"), ("CIFAR", "CIFAR-10", "$200$"),
         ("CelebA", "CelebA", "$400$")]
COLS = ["Feature", "MeanCov", "CKA", "Target", "KL", "Wasserstein", "JS", "Model"]
HI = {"FMNIST", "CIFAR", "CelebA"}


def color(x, cls):
    # caption: dark red > 10% or > 5 pp, red worse, green better, dark green < -5
    if x > (5.0 if cls else 10.0):
        c = "darkred"
    elif x > 0:
        c = "red"
    elif x < -5.0:
        c = "deepgreen"
    else:
        c = "darkgreen"
    return rf"\textcolor{{{c}}}{{${x:+.1f}$}}"


def main():
    rows, summ = [], []
    for key, label, T in ORDER:
        f = HERE / f"proxy_ablation_floor1_{key}_raw.csv"
        if not f.exists():
            print("manque", f); continue
        d = pd.read_csv(f)
        tune, ev = d[d.split == "tune"], d[d.split == "eval"]
        cls = key in HI
        for b, g in ev.groupby("b"):
            vals = {}
            for p in ["Gradient"] + COLS:
                t = tune[(tune.b == b) & (tune.proxy == p)].groupby("eta")["value"].mean().dropna()
                eta = t.idxmax() if cls else t.idxmin()
                v = g[(g.proxy == p) & (g.eta == eta)].sort_values("seed")["value"].to_numpy()
                vals[p] = (eta, v)
            G = vals["Gradient"][1]
            cells = []
            for p in COLS:
                v = vals[p][1]
                if cls:
                    delta = 100 * (G.mean() - v.mean())
                else:
                    delta = 100 * (v.mean() - G.mean()) / G.mean()
                diff = (v - G) if not cls else (G - v)
                pv = stats.ttest_rel(v, G).pvalue if len(v) > 1 and np.std(diff) > 0 else np.nan
                cells.append(color(delta, cls))
                summ.append(dict(dataset=key, b=b, proxy=p, delta=delta, p=pv,
                                 eta=vals[p][0], grad=G.mean(), value=v.mean(), n=len(v)))
            rows.append((key, label, T, b, cells, G.mean(), vals["Gradient"][0]))
    s = pd.DataFrame(summ)
    s.to_csv(HERE / "proxy_ablation_floor1_summary.csv", index=False)
    print(s.pivot_table(index=["dataset", "b"], columns="proxy", values="delta").round(1).to_string())
    print("\nGradient (moyenne, pas):")
    for key, label, T, b, cells, gm, ge in rows:
        print(f"  {key} b={b}: {gm:.4f} eta={ge}")
    print("\nmin delta (meilleure alternative):", s.delta.min().round(2),
          s.loc[s.delta.idxmin(), ["dataset", "b", "proxy"]].to_dict())
    print("cellules ou une alternative bat le gradient:", (s.delta < 0).sum(), "/", len(s))
    print("par proxy:", s[s.delta < 0].groupby("proxy").size().to_dict())
    with open(HERE / "proxy_ablation_floor1_body.tex", "w") as fh:
        for key, label, T, b, cells, gm, ge in rows:
            fh.write(f"{label} & {T} & ${b}$ & " + " & ".join(cells) + r" \\" + "\n")
    print("ecrit proxy_ablation_floor1_body.tex")


if __name__ == "__main__":
    main()
