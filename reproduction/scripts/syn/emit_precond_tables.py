"""Remplit les tables de l'annexe app:precond a partir de precond_summary.csv et precond_diag.csv.

    python3.12 precond_report.py && python3.12 emit_precond_tables.py <section.tex>
"""
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DS = ["Syn", "WEAVE-UK", "FMNIST", "CIFAR", "CelebA"]
M = ["Individual", "Star", "Chain-2opt", "CTL-MST"]
HI = {"FMNIST", "CIFAR", "CelebA"}
COLS = [("plain", 1), ("plain", 4), ("plain", 16), ("precond", 1), ("precond", 4)]
r = pd.read_csv(HERE / "precond_summary.csv")
d = pd.read_csv(HERE / "precond_diag.csv")


def fmt(ds, v):
    return f"{v:.1f}" if ds in HI else (f"{v:,.0f}".replace(",", "{,}") if ds == "WEAVE-UK" else f"{v:.2f}")


diag = []
for ds in DS:
    g = d[d.dataset == ds]
    p = r[(r.dataset == ds) & (r.variant == "precond") & (r.b == 1)].iloc[0]
    setup = p.cost - p.b * (1 + (g.D.iloc[0] + 1) / (2 * g.n.iloc[0]))
    diag.append(f"{ds} & {g.n.iloc[0]} & {g.D.iloc[0]} & {100 * g.rank_def.mean():.0f}\\% & "
                f"{g.kappa.median():,.0f}".replace(",", "{,}") + f" & {g.top_over_mean.median():.1f} & {setup:.1f} \\\\")

perf, se = [], []
for ds in DS:
    g = r[r.dataset == ds]
    for i, m in enumerate(M):
        vals = []
        for v, b in COLS:
            x = g[(g.method == m) & (g.variant == v) & (g.b == b)]
            vals.append(fmt(ds, x.value.iloc[0]))
        head = f"\\multirow{{5}}{{*}}{{{ds}}}" if i == 0 else ""
        perf.append(f"{head} & {m} & " + " & ".join(vals) + " \\\\")
    cst = [g[(g.variant == v) & (g.b == b)].cost.iloc[0] for v, b in COLS]
    perf.append(" & \\emph{cost} & " + " & ".join(f"\\emph{{{c:.0f}}}" if c >= 10 else f"\\emph{{{c:.0f}}}" for c in cst) + " \\\\")
    perf.append("\\midrule")
    sub = g[g.b.isin([1, 4, 16])]
    se.append(f"{ds} {fmt(ds, sub.se.max())}")
perf.pop()

tex = Path(sys.argv[1])
s = tex.read_text()
s = s.replace("%DIAG%", "\n".join(diag)).replace("%PERF%", "\n".join(perf)).replace("%SE%", ", ".join(se))
out = tex.with_name(tex.stem + "_filled.tex")
out.write_text(s)
print(out)
print("\n".join(diag)); print("\n".join(perf)); print(se)
