"""Lignes Syn des tables 2 (lineaire) et 3 (MLP), et recompte de la ligne des recouvrements.

    python3.12 emit_syn_rows.py   # imprime les lignes ; --apply les ecrit dans tmlr.tex
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TEX = Path(__import__("os").environ.get("PAPER_TEX", "tmlr.tex"))   # source LaTeX du papier, mise a jour en place
M7 = ["Individual", "Star", "RandTree", "Chain", "Chain-2opt", "CTL-MST", "CTL-Forest"]
TREES = ["Chain-2opt", "CTL-MST", "CTL-Forest"]


def cells(model):
    df = pd.concat([pd.read_csv(HERE / f"main_{model}_c{c}.csv") for c in range(3)]
                   + [pd.read_csv(HERE / f"main_{model}_ext.csv")])   # grille etendue a 0.7, 1, 2
    out = {}
    for b in (1, 2, 4):
        vals = {}
        for m in M7:
            g = df[(df.b == b) & (df.method == m)]
            t = g[g.split == "tune"].groupby("eta").value.mean().dropna()
            e = g[(g.split == "eval") & (g.eta == t.idxmin())].sort_values("seed").value.to_numpy()
            vals[m] = e
        pv = {m: stats.ttest_rel(vals[m], vals["Chain"], alternative="less").pvalue for m in TREES}
        holm, mx = {}, 0.0
        for i, (m, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
            mx = max(mx, min(1.0, p * (len(pv) - i))); holm[m] = mx
        out[b] = (vals, holm)
    return out


def row(b, vals, holm):
    mu = {m: vals[m].mean() for m in M7}
    se = {m: vals[m].std(ddof=1) / np.sqrt(len(vals[m])) for m in M7}
    order = sorted(M7, key=lambda m: mu[m])
    rank = {m: i + 1 for i, m in enumerate(order)}
    cs = []
    for m in M7:
        s = f"{mu[m]:.2f} \\pm {se[m]:.2f}" + ("^\\dagger" if m in holm and holm[m] < .05 else "")
        cs.append({1: rf"\cellcolor{{rankA}}$\mathbf{{{s}}}$", 2: rf"\cellcolor{{rankB}}$\underline{{{s}}}$",
                   3: rf"\cellcolor{{rankC}}${s}$"}.get(rank[m], f"${s}$"))
    best = order[0]
    ov = {m: mu[m] - se[m] <= mu[best] + se[best] for m in M7}
    return f"Syn & $200$ & ${b}$ & " + " & ".join(cs) + r" \\", ov


def parse_old(line, higher):
    nums = re.findall(r"\$\\?(?:mathbf\{|underline\{)?([0-9.]+) \\pm ([0-9.]+)", line)
    mu = np.array([float(a) for a, _ in nums]); se = np.array([float(b) for _, b in nums])
    best = mu.argmax() if higher else mu.argmin()
    return (mu + se >= mu[best] - se[best]) if higher else (mu - se <= mu[best] + se[best])


def main(apply):
    tex = TEX.read_text()
    for model, label in (("linear", "tab:simplified-opt"), ("mlp", "tab:permethod-mlp")):
        a = tex.index(rf"\label{{{label}}}"); z = tex.index(r"\end{table}", a)
        block = tex[a:z]
        lines = block.split("\n")
        res = cells(model)
        new_rows, counts_new = [], np.zeros(7, int)
        for L in lines:
            if re.match(r"(Syn-\$10\$|Syn) & ", L) or not re.match(r"[A-Za-z-]+ & \$\d+\$ & \$\d\$", L):
                continue
            counts_new += parse_old(L, L.split(" ")[0] in ("FMNIST", "CIFAR", "CelebA")).astype(int)
        for b in (1, 2, 4):
            r, ov = row(b, *res[b]); new_rows.append(r)
            counts_new += np.array([ov[m] for m in M7], int)
        print(f"% {model}"); print("\n".join(new_rows)); print("% recouvrements :", dict(zip(M7, counts_new)))
        it = iter(new_rows)
        lines = [next(it) if re.match(r"(Syn-\$10\$|Syn) & ", L) else L for L in lines]
        order = sorted(range(7), key=lambda i: -counts_new[i])
        cnt = []
        for i, m in enumerate(M7):
            s = f"{counts_new[i]}"
            rk = 1 + len({c for c in counts_new if c > counts_new[i]})   # ex aequo : meme rang
            cnt.append({1: rf"\cellcolor{{rankA}}$\mathbf{{{s}}}$", 2: rf"\cellcolor{{rankB}}$\underline{{{s}}}$"}.get(rk, f"${s}$")
                       if counts_new[i] > 0 else f"${s}$")
        lines = [(r"\multicolumn{3}{c|}{Cells with $\pm1$ SE overlap} & " + " & ".join(cnt) + r" \\")
                 if L.startswith(r"\multicolumn{3}{c|}{Cells with") else L for L in lines]
        tex = tex[:a] + "\n".join(lines) + tex[z:]
    if apply:
        TEX.write_text(tex)


if __name__ == "__main__":
    main("--apply" in sys.argv)
