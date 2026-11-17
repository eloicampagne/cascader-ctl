"""Figure 6 redessinee depuis fig_robust_summary.csv (gain de CTL-MST sur Individual, RMSE, T=1000, b=1)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#1f1f1e", "#52514e", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef3fa", "#9cc0ea", "#3b78c4", "#123f7a"])
CAT = ["#1c5cab", "#b8302f", "#2a8c7e"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "mathtext.fontset": "cm"})

s = pd.read_csv(HERE / "fig_robust_summary.csv")
s = s[s.b == 1] if "b" in s else s
tw, tb = sorted(s.tau_within.unique()), sorted(s.tau_between.unique())
M = s.pivot(index="tau_within", columns="tau_between", values="mean").loc[tw, tb].to_numpy()
E = s.pivot(index="tau_within", columns="tau_between", values="sem").loc[tw, tb].to_numpy()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.35]})
im = a1.imshow(M, origin="lower", cmap=SEQ, vmin=0, vmax=max(80, M.max()), aspect="auto")
for i in range(len(tw)):
    for j in range(len(tb)):
        a1.text(j, i, f"{M[i, j]:.1f}\n$\\pm${E[i, j]:.1f}", ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > 45 else INK)
a1.add_patch(plt.Rectangle((tb.index(10.0) - 0.5, tw.index(3.0) - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
a1.set_xticks(range(len(tb)), [f"{v:g}" for v in tb]); a1.set_yticks(range(len(tw)), [f"{v:g}" for v in tw])
a1.set_xlabel(r"$\tau_{\mathrm{between}}$"); a1.set_ylabel(r"$\tau_{\mathrm{within}}$")
a1.set_title("RMSE gain of CTL-MST over Individual (%)", color=INK, fontsize=8)
cb = fig.colorbar(im, ax=a1, shrink=0.85, pad=0.02); cb.outline.set_edgecolor(GRID)
for k, b in enumerate(tb):
    g = s[s.tau_between == b].sort_values("tau_within")
    a2.errorbar(g.tau_within, g["mean"], yerr=g["sem"], color=CAT[k], marker="o", ms=3.5, lw=1.4, capsize=2,
                label=rf"$\tau_{{\mathrm{{between}}}}={b:g}$")
a2.set_xscale("log"); a2.set_xticks(tw, [f"{v:g}" for v in tw]); a2.minorticks_off()
a2.set_ylim(0, None); a2.grid(axis="y", color=GRID, lw=0.5)
a2.set_xlabel(r"$\tau_{\mathrm{within}}$"); a2.set_ylabel("gain (%)")
a2.set_title("Gain against the within-cluster spread", color=INK, fontsize=8)
a2.legend(frameon=False, fontsize=7, loc="lower right")
for ax in (a1, a2):
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
out = Path(__import__("os").environ.get("FIG_DIR", "figures")) / "robustness_sweep.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out); fig.savefig(HERE / "robustness_preview.png", dpi=160); print(out)
