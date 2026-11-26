"""Figure de la section 4.4 : geometrie des taches et ecart arbre / chaine selon le budget.

Haut : optima des T = 200 taches du generateur de Syn (graine 0), projetes sur les deux
premieres composantes principales de l'espace latent de dimension r = 3, colores par
grappe, pour chaque dispersion omega (tau_within = 3, tau_between = 3 omega).
Bas : ecart relatif a la Chaine, (construction - Chain)/Chain en %, de CTL-MST et de
CTL-Forest contre le budget par tache b, moyenne +- erreur type sur 20 graines ; taux
choisi par methode sur cinq graines de reglage disjointes ; marqueur plein si le test t
apparie bilateral donne p < 0.05.

Donnees : tworegime_final_raw.csv (meme protocole que tworegime_final.py).

    python3.12 fig_tworegime.py <paper>/figures/tworegime.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from cascader.data.synthetic import generate_subspace_tasks  # noqa: E402

INK, MUTED, FRAME = "#1f1f1e", "#6b6a65", "#bdbcb4"
C_MST, C_FOR = "#4b3f8f", "#2a8c80"
CLUSTERS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
OMEGAS = (0.5, 2.0, 5.0, 20.0)
B = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8.5, "axes.edgecolor": FRAME, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK,
})


def cells(raw):
    df = pd.read_csv(raw)
    rows = []
    for (w, b), g in df.groupby(["scatter", "b"]):
        ser = {}
        for m in ("Chain", "CTL-MST", "CTL-Forest"):
            tune = g[(g.split == "tune") & (g.method == m)].groupby("eta")["value"].mean().dropna()
            eta = tune.idxmin()
            ser[m] = g[(g.split == "eval") & (g.method == m) & (g.eta == eta)].set_index("rep")["value"]
        p = pd.concat(ser, axis=1).dropna()
        for m in ("CTL-MST", "CTL-Forest"):
            gap = (p[m] - p["Chain"]) / p["Chain"] * 100
            rows.append(dict(scatter=w, b=b, method=m, gap=gap.mean(),
                             se=gap.std(ddof=1) / np.sqrt(len(gap)), p=ttest_rel(p[m], p["Chain"]).pvalue))
    return pd.DataFrame(rows)


def geometry(ax, w):
    _, _, _, _, cid = generate_subspace_tasks(T=200, D=40, r=3, n_train=64, n_test=128, n_clusters=5,
                                              tau_within=3.0, tau_between=3.0 * w, sigma=5.0,
                                              rng=np.random.default_rng(0))
    # memes graines : on retrouve les vecteurs latents en refaisant le tirage
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=3)
    centers = z0 + rng.normal(scale=3.0 * w, size=(5, 3))
    ids = rng.integers(0, 5, size=200)
    Z = centers[ids] + rng.normal(scale=3.0, size=(200, 3))
    assert np.array_equal(ids, cid)
    Zc = Z - Z.mean(axis=0)
    P = Zc @ np.linalg.svd(Zc, full_matrices=False)[2][:2].T
    for k in range(5):
        ax.scatter(P[ids == k, 0], P[ids == k, 1], s=4, color=CLUSTERS[k], alpha=0.8, linewidths=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(rf"$\omega={w:g}$", color=INK, fontsize=9.5, pad=2)


def gaps(ax, c, first):
    ax.axhspan(-100, 0, color="#eef2f8", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    for m, col, ls in (("CTL-MST", C_MST, "-"), ("CTL-Forest", C_FOR, "--")):
        g = c[c.method == m].sort_values("b")
        x, y, e = g.b.to_numpy(), g.gap.to_numpy(), g.se.to_numpy()
        ax.fill_between(x, y - e, y + e, color=col, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, y, color=col, lw=1.2, ls=ls, zorder=3)
        sig = g.p.to_numpy() < 0.05
        ax.scatter(x[sig], y[sig], s=13, color=col, zorder=4, edgecolors="white", linewidths=0.4)
        ax.scatter(x[~sig], y[~sig], s=13, facecolors="white", edgecolors=col, linewidths=0.9, zorder=4)
    lo = min(c.gap.min() - c.se.max(), -1.0)
    hi = max(c.gap.max() + c.se.max(), 1.0)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4, 6])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4", "6"])
    ax.minorticks_off()
    if first:
        ax.set_ylabel("gap to the Chain (%)")
    ax.grid(axis="y", color="#e6e5e0", lw=0.5, zorder=0)


def main(out):
    c = cells(HERE / "tworegime_final_raw.csv")
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 2.45), gridspec_kw=dict(height_ratios=[0.85, 1.35]))
    for j, w in enumerate(OMEGAS):
        geometry(axes[0, j], w)
        gaps(axes[1, j], c[c.scatter == w], j == 0)
    handles = [plt.Line2D([], [], color=C_MST, lw=1.2, marker="o", ms=4, label="CTL-MST"),
               plt.Line2D([], [], color=C_FOR, lw=1.2, ls="--", marker="o", ms=4, label="CTL-Forest"),
               plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=MUTED, label=r"not significant ($p\geq0.05$)"),
               plt.Rectangle((0, 0), 1, 1, fc="#eef2f8", ec="none", label="lower error than the Chain")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.54, -0.03), handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.94, bottom=0.21, wspace=0.32, hspace=0.12)
    fig.text(0.54, 0.1, r"per-task budget $b$", ha="center", va="center", color=INK)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tworegime.pdf")
