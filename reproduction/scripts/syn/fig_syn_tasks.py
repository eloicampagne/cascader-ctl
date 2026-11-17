"""Figure de l'annexe F.1 : geometrie des taches de Syn, projection ACP des parametres,
pour trois dispersions intra-grappe.

    python3.12 fig_syn_tasks.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from cascader.data.synthetic import subspace_collection, generate_subspace_tasks  # noqa: E402
from cascader.config import get_config                                           # noqa: E402
from cascader.methods import build_structure                                     # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#52514e", "#c3c2b7"
COLORS = ["#1c5cab", "#b8302f", "#2a8c7e", "#d08b1c", "#6a4c93"]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})


def main(seed=0):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), constrained_layout=True)
    for ax, tw in zip(axes, (1.0, 3.0, 10.0)):
        thetas, _, _, _, clusters = generate_subspace_tasks(tau_within=tw, rng=np.random.default_rng(seed))
        Z = thetas - thetas.mean(0)
        P = Z @ np.linalg.svd(Z, full_matrices=False)[2][:2].T
        for k in range(clusters.max() + 1):
            m = clusters == k
            ax.scatter(P[m, 0], P[m, 1], s=9, color=COLORS[k], edgecolor="white", linewidth=0.3, zorder=2)
        ax.set_title(rf"$\tau_{{\mathrm{{within}}}}={tw:g}$" + (" (Syn)" if tw == 3 else ""), color=INK, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    out = Path(__import__("os").environ.get("FIG_DIR", "figures")) / "synthetic_tasks.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); fig.savefig(HERE / "synthetic_tasks_preview.png", dpi=160)
    print(out)


if __name__ == "__main__":
    main()
