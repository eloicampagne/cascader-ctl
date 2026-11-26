"""Figure de l'annexe D.2 (k = 1, 7, 31) : frontiere CTL / transfert direct, famille A (arc, isotrope).

Couleur : err_CTL / err_direct en echelle log2, par bandes, bleu quand la cascade
gagne. Traits : condition du theoreme (bound_ratio = 1), son cas particulier en
delta_max, frontiere mesuree (rapport = 1) et delta_max = d(s,v).

Changements visuels :
  - aplats par bandes (contourf) au lieu d'un degrade pixel par pixel ;
  - cote rouge eclairci, pour que l'oeil aille d'abord a la zone ou la cascade gagne ;
  - bande infaisable (delta_max < d/(k+1)) grisee et hachuree au lieu d'un blanc ;
  - courbes distinguees par l'epaisseur et la couleur, pas seulement par le pointille ;
  - polices Computer Modern, comme le texte.

    python3.12 boundary_phase_fig.py [CSV] [OUT]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

INK, MUTED, FRAME = "#1f1f1e", "#5f5e5a", "#bdbcb4"
# la couleur change exactement au rapport 1 : bleu tres pale juste dessous, rose tres pale juste dessus
LEVELS = np.array([-2.0, -1.5, -1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0])
COLORS = ["#1d4f91", "#3f78bf", "#79a6db", "#b7d0ee", "#e4edf8", "#f9e7e4", "#f3c9c4", "#e39a92"]
CMAP = LinearSegmentedColormap.from_list("bands", COLORS, N=len(COLORS))
NORM = BoundaryNorm(LEVELS, CMAP.N)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "font.size": 8.5,
    "axes.edgecolor": FRAME, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.linewidth": 0.6,
})


def main(csv, out):
    A = pd.read_csv(csv)
    A = A[A.geom == "arc"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55), sharey=True, constrained_layout=True)
    for ax, k in zip(axes, (1, 7, 31)):
        g = A[A.k == k]
        P = g.pivot_table(index="xi", columns="q", values="ratio")
        y = g.groupby("xi").dmax_over_d.first().reindex(P.index).to_numpy()
        q = P.columns.to_numpy()
        Z = np.clip(np.log2(P.to_numpy()), LEVELS[0] + 1e-9, LEVELS[-1] - 1e-9)
        cf = ax.contourf(q, y, Z, levels=LEVELS, cmap=CMAP, norm=NORM)
        # bande infaisable : le plus long des k+1 segments vaut au moins d/(k+1)
        floor = 1.0 / (k + 1)
        ax.fill_between([q.min(), q.max()], 0, floor, facecolor="#ecebe6", edgecolor="#c9c8c0",
                        hatch="////", linewidth=0.0, zorder=1)
        if k == 1:
            ax.text(0.5, floor / 2, "infeasible", ha="center", va="center", fontsize=7.5,
                    color=MUTED, zorder=2, bbox=dict(fc="#ecebe6", ec="none", pad=1.0))
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=3)
        B = g.pivot_table(index="xi", columns="q", values="bound_ratio").to_numpy()
        ax.contour(q, y, B, levels=[1.0], colors=INK, linewidths=1.6, zorder=4)
        qq = np.linspace(q.min(), q.max(), 200)
        ax.plot(qq, (1 - qq) / (1 - qq ** (k + 1)), color=INK, lw=0.9, ls=(0, (5, 1.5, 1, 1.5)), zorder=4)
        ax.contour(q, y, np.log2(P.to_numpy()), levels=[0.0], colors="#8a2a22", linewidths=1.3,
                   linestyles=[(0, (4, 2))], zorder=5)
        ax.set_title(f"$k={k}$ intermediate task{'s' if k > 1 else ''}", color=INK, fontsize=9, pad=4)
        ax.set_xlabel(r"$q=\rho^{\,b}$")
        ax.set_xlim(q.min(), q.max())
        ax.set_ylim(0.0, 2.0)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        for s in ax.spines.values():
            s.set_zorder(6)
    axes[0].set_ylabel(r"$\delta_{\max}/\mathrm{dist}(s,v)$")
    cb = fig.colorbar(cf, ax=axes, shrink=0.92, pad=0.015, aspect=22, ticks=[-2, -1, 0, 1])
    cb.ax.set_yticklabels([r"$\frac{1}{4}$", r"$\frac{1}{2}$", r"$1$", r"$2$"])
    cb.set_label(r"$\mathrm{err}_{\mathrm{CTL}}\,/\,\mathrm{err}_{\mathrm{direct}}$ (log scale)", color=INK)
    cb.outline.set_edgecolor(FRAME)
    cb.ax.tick_params(length=2)
    handles = [plt.Line2D([], [], color=INK, lw=1.6, label="sufficient condition (Theorem 2.4)"),
               plt.Line2D([], [], color=INK, lw=0.9, ls=(0, (5, 1.5, 1, 1.5)), label=r"its particular case in $\delta_{\max}$"),
               plt.Line2D([], [], color="#8a2a22", lw=1.3, ls=(0, (4, 2)), label="measured boundary"),
               plt.Line2D([], [], color=MUTED, lw=0.8, ls=":", label=r"$\delta_{\max}=\mathrm{dist}(s,v)$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7.5,
               handlelength=2.6, columnspacing=1.4)
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parent
    csv = sys.argv[1] if len(sys.argv) > 1 else here / "boundary_phase_A.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else Path("figures") / "boundary_phase.pdf"
    main(csv, out)
