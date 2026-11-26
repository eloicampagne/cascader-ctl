"""Schema 3D de l'annexe D : contraction locale, minimiseur unique ou multiple.

Trois panneaux montrent la perte L_v d'une tache cible v sur un espace de
parametres a deux dimensions et le raffinement par descente de gradient a partir
de l'initialisation heritee du parent.

  (a) perte convexe : un seul minimiseur, la descente contracte depuis tout point ;
  (b) perte non convexe, transfert court : l'initialisation tombe dans le bassin
      du minimiseur proche du parent, ou la contraction tient localement ;
  (c) perte non convexe, transfert long : l'initialisation tombe dans un autre
      bassin et la descente rejoint un autre minimiseur.

Les trajectoires sont de vraies descentes de gradient sur les fonctions tracees.

    python3.12 fig_landscapes.py <paper>/figures/contraction_landscapes.pdf
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patheffects as pe

from matplotlib.colors import LightSource, LinearSegmentedColormap

plt.rcParams.update({"font.size": 13, "mathtext.fontset": "cm", "font.family": "serif",
                     "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"]})
C_INIT, C_TRAJ, C_OPT, C_OTHER = "tab:blue", "tab:red", "tab:green", "tab:orange"
LIFT = 0.03  # petit decalage vertical pour que les traces restent au-dessus de la surface
Z0 = -2.6  # plan de base ou l'on projette les lignes de niveau et la geometrie
SURF = LinearSegmentedColormap.from_list("surf", ["#8f9bb3", "#c4cbd8", "#e2e6ed"])
LIGHT = LightSource(azdeg=300, altdeg=40)


CX, CY = 0.3, 0.1  # minimiseur de la perte convexe


def convex(x, y):
    return 0.35 * (x - CX) ** 2 + 0.9 * (y - CY) ** 2


def convex_grad(p):
    x, y = p
    return np.array([0.7 * (x - CX), 1.8 * (y - CY)])


def double_well(x, y):
    # deux puits en x ~ -1 et x ~ +1, le second un peu plus profond
    return 0.55 * (x ** 2 - 1) ** 2 + 0.7 * y ** 2 - 0.12 * x + 0.12


def double_well_grad(p):
    x, y = p
    return np.array([2.2 * x * (x ** 2 - 1) - 0.12, 1.4 * y])


def descend(grad, p0, eta, steps):
    path = [np.array(p0, float)]
    for _ in range(steps):
        path.append(path[-1] - eta * grad(path[-1]))
    return np.array(path)


def minimizer(grad, p0):
    return descend(grad, p0, 0.05, 4000)[-1]


AZIM, ELEV = -62, 24
VIEW = np.array([np.cos(np.radians(AZIM)), np.sin(np.radians(AZIM))])  # direction du spectateur dans le plan


def base(ax, f, center, radii, lim, zmax):
    """Surface sur un domaine elliptique, coupee en une moitie arriere et une moitie avant.

    La moitie avant et son bord sont dessines par-dessus les traits verticaux, et le bord
    avant par-dessus tout, ce qui donne la profondeur que le rendu 3D de matplotlib ne
    calcule pas entre objets differents.
    """
    nphi = 160
    r = np.linspace(0, 1, 70)[:, None]
    # angle du point le plus proche du spectateur, pour centrer la moitie avant
    phi0 = np.arctan2(VIEW[1] * radii[1], VIEW[0] * radii[0])
    phi = phi0 + np.linspace(-np.pi, np.pi, nphi)[None, :]
    X = center[0] + radii[0] * r * np.cos(phi)
    Y = center[1] + radii[1] * r * np.sin(phi)
    Z = f(X, Y)
    rgb = LIGHT.shade(Z, cmap=SURF, blend_mode="soft", vert_exag=0.6, vmin=0.0, vmax=zmax)
    q1, q3 = nphi // 4, 3 * nphi // 4          # moitie avant : phi0 +- pi/2
    back = [np.r_[0:q1 + 1], np.r_[q3:nphi]]
    front = np.r_[q1:q3 + 1]
    for cols in back:
        ax.plot_surface(X[:, cols], Y[:, cols], Z[:, cols], facecolors=rgb[:, cols], rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False, alpha=0.95, zorder=1)
    ax.plot_surface(X[:, front], Y[:, front], Z[:, front], facecolors=rgb[:, front], rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False, alpha=0.88, zorder=3)
    # maillage leger pour le relief, en arriere ou en avant selon la moitie
    for k in range(0, 70, 10):
        for cols, z in ((np.r_[0:q1 + 1], 1.5), (np.r_[q3:nphi], 1.5), (front, 3.5)):
            ax.plot(X[k, cols], Y[k, cols], Z[k, cols], color="white", lw=0.35, alpha=0.7, zorder=z)
    for k in range(0, nphi, 16):
        ax.plot(X[:, k], Y[:, k], Z[:, k], color="white", lw=0.35, alpha=0.7,
                zorder=3.5 if q1 <= k <= q3 else 1.5)
    # bord : arriere sous les traces, avant par-dessus tout
    for cols in back:
        ax.plot(X[-1, cols], Y[-1, cols], Z[-1, cols], color="#7d8698", lw=0.9, zorder=1.6)
    ax.plot(X[-1, front], Y[-1, front], Z[-1, front], color="#6f788b", lw=1.3, zorder=9)
    g = np.linspace(-lim, lim, 200)
    GX, GY = np.meshgrid(center[0] + g, center[1] + g)
    inside = ((GX - center[0]) / radii[0]) ** 2 + ((GY - center[1]) / radii[1]) ** 2 <= 1
    GZ = np.where(inside, f(GX, GY), np.nan)
    ax.contour(GX, GY, GZ, levels=np.linspace(0.02, zmax, 9), zdir="z", offset=Z0,
               colors="0.8", linewidths=0.45, zorder=0.5)
    ax.set_zlim(Z0, zmax)
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.95), zoom=1.22)
    ax.view_init(elev=ELEV, azim=AZIM)


def curve(ax, f, path, color, lw=1.8, base_too=True):
    """Trajectoire posee sur la surface : entre deux iteres, on suit la surface, pas la corde."""
    halo = [pe.Stroke(linewidth=lw + 2.2, foreground="white", alpha=0.85), pe.Normal()]
    xy = np.concatenate([a + np.linspace(0, 1, 30)[:, None] * (b - a) for a, b in zip(path[:-1], path[1:])])
    ax.plot(xy[:, 0], xy[:, 1], f(xy[:, 0], xy[:, 1]) + LIFT, color=color, lw=lw, zorder=5,
            path_effects=halo, solid_capstyle="round")
    ax.scatter(path[1:-1, 0], path[1:-1, 1], f(path[1:-1, 0], path[1:-1, 1]) + LIFT,
               color=color, s=11, zorder=5, depthshade=False, edgecolors="white", linewidths=0.5)
    if base_too:
        ax.plot(path[:, 0], path[:, 1], Z0, color=color, lw=1.0, ls="--", alpha=0.8)


def point(ax, f, p, color, hollow=False, label=None, dx=0.08, dy=0.08, base_too=True):
    kw = dict(s=44, depthshade=False, zorder=6)
    if hollow:
        kw.update(facecolors="white", edgecolors=color, linewidths=1.5)
    else:
        kw.update(color=color, edgecolors="white", linewidths=0.9)
    z = f(p[0], p[1]) + LIFT
    ax.scatter([p[0]], [p[1]], [z], **kw)
    if base_too:
        ax.plot([p[0], p[0]], [p[1], p[1]], [Z0, z], color=color, lw=0.7, ls=(0, (1, 2)), alpha=0.7, zorder=2)
        ax.scatter([p[0]], [p[1]], [Z0], s=20, depthshade=False,
                   **({"facecolors": "white", "edgecolors": color} if hollow else {"color": color}))
    if label:
        ax.text(p[0] + dx, p[1] + dy, Z0, label, color=color, fontsize=21, zorder=7,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])


def ring(ax, center, r, color, ls):
    c = Circle((center[0], center[1]), r, fill=False, ec=color, lw=1.3, ls=ls)
    ax.add_patch(c)
    art3d.pathpatch_2d_to_3d(c, z=Z0, zdir="z")


def main(out):
    fig = plt.figure(figsize=(12, 4.6))

    # (a) perte convexe : contraction globale vers l'unique minimiseur
    ax = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)
    base(ax, convex, (CX, CY), (np.sqrt(1.5 / 0.35), np.sqrt(1.5 / 0.9)), 2.2, 1.7)
    star = np.array([CX, CY])
    init = star + np.array([-1.35, 0.75])
    path = descend(convex_grad, init, 0.55, 4)
    ring(ax, star, np.linalg.norm(init - star), C_INIT, ":")
    ring(ax, star, np.linalg.norm(path[-1] - star), C_TRAJ, "-")
    curve(ax, convex, path, C_TRAJ)
    point(ax, convex, init, C_INIT, label=r"$\tilde\theta_{\mathrm{pa}(v)}$", dx=-1.0, dy=0.45)
    point(ax, convex, path[-1], C_TRAJ, label=r"$\tilde\theta_v$", dx=-0.55, dy=-1.0)
    point(ax, convex, star, C_OPT, hollow=True, label=r"$\theta_v^\star$", dx=0.4, dy=0.35)
    ax.set_title("(a) Convex loss: one minimizer", fontsize=16, y=1.07)

    # (b) perte non convexe, transfert court : on reste dans le bassin de droite
    ax = fig.add_subplot(1, 3, 2, projection="3d", computed_zorder=False)
    base(ax, double_well, (0.0, 0.0), (1.62, 1.15), 1.8, 1.7)
    s_right = minimizer(double_well_grad, [1.0, 0.0])
    s_left = minimizer(double_well_grad, [-1.0, 0.0])
    init = np.array([0.55, 0.75])
    path = descend(double_well_grad, init, 0.12, 6)
    ring(ax, s_right, 0.92, C_OPT, ":")  # voisinage de contraction locale
    curve(ax, double_well, path, C_TRAJ)
    point(ax, double_well, init, C_INIT, label=r"$\tilde\theta_{\mathrm{pa}(v)}$", dx=-0.75, dy=0.15)
    point(ax, double_well, path[-1], C_TRAJ, label=r"$\tilde\theta_v$", dx=-0.5, dy=-0.95)
    point(ax, double_well, s_right, C_OPT, hollow=True, label=r"$\theta_{v,1}^\star$", dx=0.1, dy=0.75)
    point(ax, double_well, s_left, C_OTHER, hollow=True, label=r"$\theta_{v,2}^\star$", dx=-0.85, dy=-0.25)
    ax.set_title("(b) Nonconvex loss, short transfer", fontsize=16, y=1.07)

    # (c) perte non convexe, transfert long : l'initialisation change de bassin
    ax = fig.add_subplot(1, 3, 3, projection="3d", computed_zorder=False)
    base(ax, double_well, (0.0, 0.0), (1.62, 1.15), 1.8, 1.7)
    init = np.array([-0.3, 0.9])
    path = descend(double_well_grad, init, 0.12, 4)
    ring(ax, s_right, 0.92, C_OPT, ":")
    curve(ax, double_well, path, C_TRAJ)
    point(ax, double_well, init, C_INIT, label=r"$\tilde\theta_{s}$", dx=0.2, dy=0.3)
    point(ax, double_well, path[-1], C_TRAJ, label=r"$\tilde\theta_v$", dx=-0.25, dy=0.55)
    point(ax, double_well, s_right, C_OPT, hollow=True, label=r"$\theta_{v,1}^\star$", dx=0.1, dy=0.75)
    point(ax, double_well, s_left, C_OTHER, hollow=True, label=r"$\theta_{v,2}^\star$", dx=-0.35, dy=-0.75)
    ax.set_title("(c) Nonconvex loss, long transfer", fontsize=16, y=1.07)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0, wspace=0.08)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("ecrit", out)
    print("minimiseurs", s_right.round(3), s_left.round(3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "contraction_landscapes.pdf")
