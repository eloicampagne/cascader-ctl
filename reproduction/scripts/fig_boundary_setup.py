"""Schema de l'experience de l'annexe D.2 : geometrie de l'arc et cascade isotrope.

k = 3 taches intermediaires (pour la lisibilite ; l'experience utilise k = 1, 7, 31)
aux abscisses j/(k+1) du segment s -> v (dist(s,v) = 1), deplacees de
xi sin(pi j/(k+1)) orthogonalement. Raffinement isotrope de facteur q = 0.5 : chaque
tache part du modele raffine de son parent et parcourt la moitie du chemin vers son
optimum (trait pale = direction du raffinement, point = modele raffine). Le transfert
direct part de theta_s* et s'arrete a mi-chemin de theta_v*.

Deux panneaux : arc bas (xi = 0.5, la cascade gagne) et arc haut (xi = 2.0, la
cascade perd).

    python3.12 fig_boundary_setup.py <paper>/figures/boundary_setup.pdf
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, EDGE = "#1f1f1e", "#6b6a65", "#b9b8b1"
C_CTL, C_DIR = "#1f6fb2", "#e07b12"
K, Q = 3, 0.5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5,
})


def optima(xi):
    return np.array([[j / (K + 1), xi * np.sin(np.pi * j / (K + 1))] for j in range(K + 2)])


def cascade(P):
    t = [P[0].copy()]
    for i in range(1, len(P)):
        t.append(Q * t[-1] + (1 - Q) * P[i])
    return np.array(t)


def arrow(ax, a, b, color, lw=1.5, z=4):
    ax.annotate("", xy=b, xytext=a, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11,
                                shrinkA=4, shrinkB=4))


def panel(ax, xi, title, labels):
    P = optima(xi)
    T = cascade(P)
    s, v = P[0], P[-1]
    direct = Q * s + (1 - Q) * v
    edges = np.linalg.norm(np.diff(P, axis=0), axis=1)
    imax = int(np.argmax(edges))

    # segment s -> v et aretes du chemin entre optima
    ax.plot([s[0], v[0]], [s[1], v[1]], color=EDGE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot(P[:, 0], P[:, 1], color=EDGE, lw=1.4, zorder=1)
    ax.plot(P[imax:imax + 2, 0], P[imax:imax + 2, 1], color=INK, lw=2.2, zorder=2)
    mid = P[imax:imax + 2].mean(axis=0)
    ax.annotate(r"$\delta_{\max}$", mid, xytext=labels["dmax"], textcoords="offset points",
                ha="right", va="center", color=INK)

    # hauteur de l'arc
    j = (K + 1) // 2
    ax.annotate("", xy=(P[j, 0], P[j, 1]), xytext=(P[j, 0], 0.0), zorder=1,
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(P[j, 0] + 0.03, 0.42 * P[j, 1], r"$\xi$", color=MUTED, va="center", fontsize=12)

    # cascade : trait pale vers l'optimum de la tache, fleche jusqu'au modele raffine (mi-chemin)
    for i in range(1, len(T)):
        ax.plot([T[i - 1, 0], P[i, 0]], [T[i - 1, 1], P[i, 1]], color=C_CTL, lw=0.8, alpha=0.3, zorder=2)
        arrow(ax, T[i - 1], T[i], C_CTL)
    ax.scatter(T[1:, 0], T[1:, 1], s=46, color=C_CTL, zorder=5, edgecolors="white", linewidths=0.6)

    # transfert direct : meme regle, depuis theta_s*
    arrow(ax, s, direct, C_DIR)
    ax.scatter([direct[0]], [direct[1]], s=46, color=C_DIR, zorder=5, edgecolors="white", linewidths=0.6)

    # erreurs finales a la cible
    ax.plot([T[-1, 0], v[0]], [T[-1, 1], v[1]], color=C_CTL, lw=4.0, alpha=0.35, solid_capstyle="round", zorder=3)
    ax.plot([direct[0], v[0]], [direct[1], v[1]], color=C_DIR, lw=4.0, alpha=0.35, solid_capstyle="round", zorder=3)

    # optima
    ax.scatter(P[1:-1, 0], P[1:-1, 1], s=58, facecolors="white", edgecolors=MUTED, lw=1.3, zorder=6)
    ax.scatter([s[0], v[0]], [s[1], v[1]], s=70, facecolors="white", edgecolors=INK, lw=1.6, zorder=6)
    ax.annotate(r"$\theta_s^\star$", s, xytext=(-7, -4), textcoords="offset points", ha="right", va="top", fontsize=12)
    ax.annotate(r"$\theta_v^\star$", v, xytext=(7, -4), textcoords="offset points", ha="left", va="top", fontsize=12)
    ax.annotate(r"$\tilde\theta_v^{\,\mathrm{CTL}}$", T[-1], xytext=labels["ctl"], textcoords="offset points",
                color=C_CTL, ha="left", va="center", fontsize=11.5)
    ax.annotate(r"$\tilde\theta_v^{\,\mathrm{direct}}$", direct, xytext=(0, -9), textcoords="offset points",
                color=C_DIR, ha="center", va="top", fontsize=11.5)

    e_ctl = np.linalg.norm(T[-1] - v)
    e_dir = np.linalg.norm(direct - v)
    ax.set_title(title, color=INK, pad=6, fontsize=11)
    ax.text(0.5, -0.02, rf"error: cascade ${e_ctl:.2f}$", color=C_CTL, transform=ax.transAxes,
            ha="right", va="top")
    ax.text(0.5, -0.02, rf",  direct ${e_dir:.2f}$", color=C_DIR, transform=ax.transAxes,
            ha="left", va="top")
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 1.2)
    ax.set_axis_off()
    return e_ctl, e_dir


def main(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), gridspec_kw=dict(width_ratios=[2.15, 1.0]))
    a = panel(axes[0], 0.5, r"(a) low arc, $\xi=0.5$: the cascade wins",
              dict(dmax=(-8, 6), ctl=(7, 9)))
    axes[0].set_ylim(-0.28, 0.62)
    b = panel(axes[1], 2.0, r"(b) high arc, $\xi=2.0$: it loses",
              dict(dmax=(-8, 0), ctl=(8, 0)))
    axes[1].set_ylim(-0.3, 2.12)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=MUTED, mew=1.3, label="task optimum"),
               plt.Line2D([], [], marker="o", ls="", ms=7, color=C_CTL, label="cascade, refined model"),
               plt.Line2D([], [], marker="o", ls="", ms=7, color=C_DIR, label="direct transfer"),
               plt.Line2D([], [], color=EDGE, lw=1.0, ls=(0, (4, 3)), label=r"$\mathrm{dist}(s,v)=1$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.03), handletextpad=0.3, columnspacing=1.3)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.17, wspace=0.08)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(out, "erreurs (a)", np.round(a, 3), "(b)", np.round(b, 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "boundary_setup.pdf")
