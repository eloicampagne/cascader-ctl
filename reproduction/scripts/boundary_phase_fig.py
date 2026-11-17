"""Figure de l'annexe D.2 (k = 1, 7, 31) : frontiere CTL / transfert direct, famille A (arc, isotrope).

Couleur : log2(err_CTL / err_direct), divergente bleu (CTL meilleur) / rouge
(CTL moins bon), gris au rapport 1. Traits : frontiere theorique r = 1
(delta_max/d = (1-q)/(1-q^{k+1})), frontiere mesuree (rapport = 1) et
delta_max = d(s,v), limite atteinte par les chemins d'un MST.

    python3.12 boundary_phase_fig.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
FIG = Path(__import__("os").environ.get("FIG_DIR", "figures")) / "boundary_phase.pdf"

INK, MUTED, GRID = "#1f1f1e", "#52514e", "#c3c2b7"
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])

plt.rcParams.update({"font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})


def main():
    A = pd.read_csv(HERE / "boundary_phase_A.csv")
    A = A[A.geom == "arc"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True, constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=1.0)
    for ax, k in zip(axes, (1, 7, 31)):
        g = A[A.k == k]
        P = g.pivot_table(index="xi", columns="q", values="ratio")
        ymap = g.groupby("xi").dmax_over_d.first().reindex(P.index).to_numpy()
        q = P.columns.to_numpy()
        Z = np.log2(P.to_numpy())
        mesh = ax.pcolormesh(q, ymap, np.clip(Z, -2, 1), cmap=CMAP, norm=norm,
                             shading="nearest", rasterized=True)
        ax.contour(q, ymap, Z, levels=[0.0], colors=INK, linewidths=1.2,
                   linestyles="--")
        B = g.pivot_table(index="xi", columns="q", values="bound_ratio").to_numpy()
        ax.contour(q, ymap, B, levels=[1.0], colors=INK, linewidths=1.4)
        qq = np.linspace(q.min(), q.max(), 200)
        ax.plot(qq, (1 - qq) / (1 - qq ** (k + 1)), color=INK, lw=1.0, ls="-.")
        ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
        ax.set_title(f"$k={k}$ intermediate task{'s' if k > 1 else ''}", color=INK, fontsize=8)
        ax.set_xlabel(r"$q=\rho^{b}$")
        ax.set_xlim(q.min(), q.max())
        ax.set_ylim(0.0, 2.0)
    axes[0].set_ylabel(r"$\delta_{\max}/\mathrm{dist}(s,v)$")
    cb = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label(r"$\log_2(\mathrm{err}_{\mathrm{CTL}}/\mathrm{err}_{\mathrm{direct}})$", color=INK)
    cb.outline.set_edgecolor(GRID)
    handles = [plt.Line2D([], [], color=INK, lw=1.4, label="condition of Theorem"),
               plt.Line2D([], [], color=INK, lw=1.0, ls="-.", label=r"particular case in $\delta_{\max}$"),
               plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="measured boundary"),
               plt.Line2D([], [], color=MUTED, lw=1.0, ls=":", label=r"$\delta_{\max}=\mathrm{dist}(s,v)$ (an MST path lies below)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=300)
    fig.savefig(HERE / "boundary_phase_preview.png", dpi=160)
    print(FIG)


if __name__ == "__main__":
    main()
