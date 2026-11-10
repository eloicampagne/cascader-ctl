"""Paired significance of Chain vs CTL-MSTc vs CTL-MST, MLP (non-convex) regime.

Chain often has the lowest mean error among the cascades. This tests whether that
nominal ordering is significant. Every method shares the same per-seed task draw
(run_regression / run_classification generate the tasks once per seed, then run
all methods), so the seed index pairs the samples exactly.

Reads the existing t1nc_*_raw.csv; runs nothing new.

    python3.12 cascade_significance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent.resolve()

# regression -> lower value (RMSE) is better; classification -> higher (accuracy)
LOWER_IS_BETTER = {"Syn-10": True, "WEAVE-UK": True,
                   "FMNIST": False, "CIFAR": False, "CelebA": False}

RAWS = ["t1nc_syn10_raw.csv", "t1nc_weave_raw.csv", "t1nc_fmnist_raw.csv",
        "t1nc_cifar_raw.csv", "t1nc_celeba_raw.csv"]

PAIRS = [("Chain", "CTL-MSTc"), ("Chain", "CTL-MST"), ("CTL-MSTc", "CTL-MST")]


def load():
    frames = []
    for r in RAWS:
        p = HERE / r
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def paired(a, b, lower_better):
    """a, b aligned by seed. Return (mean_diff_favouring_a, t_p, w_p, n).

    diff is defined so that positive = a is better than b.
    """
    d = (b - a) if lower_better else (a - b)   # positive => a better
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return float(np.mean(d)) if n else np.nan, np.nan, np.nan, n
    t_p = stats.ttest_rel(
        (b if lower_better else a), (a if lower_better else b)).pvalue
    try:
        w_p = stats.wilcoxon(d).pvalue
    except ValueError:
        w_p = np.nan
    return float(np.mean(d)), float(t_p), float(w_p), n


def main():
    df = load()
    df = df[df["model"] == "mlp"]
    rows = []
    for (ds, B), g in df.groupby(["dataset", "B"]):
        piv = g.pivot_table(index="seed", columns="method", values="value")
        lb = LOWER_IS_BETTER[ds]
        for x, y in PAIRS:
            if x not in piv or y not in piv:
                continue
            sub = piv[[x, y]].dropna()
            md, tp, wp, n = paired(sub[x].values, sub[y].values, lb)
            better = x if md > 0 else y
            rows.append(dict(dataset=ds, B=B, comparison=f"{x} vs {y}",
                             better=better, mean_gap_for_better=abs(md),
                             t_p=tp, wilcoxon_p=wp, n=n,
                             sig05=(tp is not np.nan and tp < 0.05)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.to_string(index=False,
                        float_format=lambda v: f"{v:.4g}"))
    out.to_csv(HERE / "cascade_significance_summary.csv", index=False)

    # headline: of all Chain-vs-CTL cells, how many are significant either way
    chain = out[out["comparison"].str.startswith("Chain")]
    sig = chain[chain["t_p"] < 0.05]
    print(f"\nChain-vs-CTL cells: {len(chain)} total, "
          f"{len(sig)} significant at p<0.05 (paired t).")
    if len(sig):
        print("Significant ones:")
        print(sig[["dataset", "B", "comparison", "better",
                   "mean_gap_for_better", "t_p"]].to_string(index=False,
                   float_format=lambda v: f"{v:.4g}"))
    print("\nwrote cascade_significance_summary.csv")


if __name__ == "__main__":
    main()
