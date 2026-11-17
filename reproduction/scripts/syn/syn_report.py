"""Analyse des sorties de syn_suite.py (jeu Syn), au protocole des tables du papier.

Pas choisi par (methode, budget) sur la moyenne des graines de reglage, evaluation
sur les graines 0..19. Test : t apparie unilateral contre Chain, Holm sur les trois
autres constructions CTL de la cellule.

    python3.12 syn_report.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
M7 = ["Individual", "Star", "RandTree", "Chain", "Chain-2opt", "CTL-MST", "CTL-Forest"]
TREES = ["Chain-2opt", "CTL-MST", "CTL-Forest"]


def pick(df, keys=("eta",)):
    """Reglage choisi sur tune, valeurs d'evaluation triees par graine."""
    t, e = df[df.split == "tune"], df[df.split == "eval"]
    s = t.groupby(list(keys))["value"].mean().dropna()
    if s.empty:
        return None, None
    k = s.idxmin()
    k = k if isinstance(k, tuple) else (k,)
    m = np.ones(len(e), bool)
    for c, v in zip(keys, k):
        m &= (e[c] == v).to_numpy()
    return k, e[m].sort_values("seed")["value"].to_numpy()


def ms(v):
    return f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}"


def main_tables():
    for model in ("linear", "mlp"):
        df = pd.concat(pd.read_csv(HERE / f"main_{model}_c{c}.csv") for c in range(3))
        print(f"\n=== Table principale, {model} ===")
        overlap = {m: 0 for m in M7}
        for b in (1, 2, 4):
            vals = {}
            for m in M7 + ["kNN-BFS"]:
                k, v = pick(df[(df.b == b) & (df.method == m)])
                if v is not None and np.isfinite(v).all():
                    vals[m] = (k[0], v)
            pv = {m: stats.ttest_rel(vals[m][1], vals["Chain"][1], alternative="less").pvalue for m in TREES}
            holm, mx = {}, 0.0
            for i, (m, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
                mx = max(mx, min(1.0, p * (len(pv) - i)))
                holm[m] = mx
            best = min(M7, key=lambda m: vals[m][1].mean())
            bm, bse = vals[best][1].mean(), vals[best][1].std(ddof=1) / np.sqrt(20)
            for m in M7:
                v = vals[m][1]
                if v.mean() - v.std(ddof=1) / np.sqrt(20) <= bm + bse:
                    overlap[m] += 1
            order = sorted(M7, key=lambda m: vals[m][1].mean())
            print(f" b={b}  " + " | ".join(
                f"{m} {ms(vals[m][1])}{'†' if m in holm and holm[m] < .05 else ''} (eta {vals[m][0]:g})"
                for m in order))
            c2 = vals["Chain"][1]
            print(f"      Chain-2opt vs Chain {100 * (vals['Chain-2opt'][1].mean() / c2.mean() - 1):+.2f}% ;"
                  f" kNN-BFS {ms(vals['kNN-BFS'][1])} ({100 * (vals['kNN-BFS'][1].mean() / vals['CTL-MST'][1].mean() - 1):+.1f}% vs MST) ;"
                  f" meilleure cascade vs Individual {100 * (min(vals[m][1].mean() for m in ['Chain'] + TREES) / vals['Individual'][1].mean() - 1):+.1f}%")
        print("  cellules a ±1 SE du meilleur :", overlap)


def closed():
    df = pd.read_csv(HERE / "closed.csv")
    print("\n=== Formes closes (lineaire) ===")
    for m, g in df.groupby("method"):
        if g.h.notna().any():
            k, v = pick(g, ("h",))
            print(f"  {m:10s} {ms(v)}  (h={k[0]:g})")
        else:
            print(f"  {m:10s} {ms(g[g.split == 'eval'].sort_values('seed').value.to_numpy())}")


def global_model():
    print("\n=== Pre-entrainement poole (g pas sur les taches concatenees, puis b pas locaux) ===")
    for model in ("linear", "mlp"):
        df = pd.read_csv(HERE / f"global_{model}.csv")
        for b in (1, 2, 4):
            g = df[df.b == b]
            k, v = pick(g, ("g", "eta"))
            k0, v0 = pick(g[g.g == 0], ("eta",))
            print(f"  {model} b={b}: meilleur g={k[0]} eta={k[1]:g} -> {ms(v)} ; g=0 -> {ms(v0)}")


def large_T():
    print("\n=== Grand T (MLP, b=4) ===")
    for T in (200, 500, 1000, 2000):
        df = pd.read_csv(HERE / f"largeT_{T}.csv")
        dep = pd.read_csv(HERE / f"largeT_{T}_depth.csv").groupby("method").depth_max.mean()
        r = {m: pick(df[df.method == m])[1] for m in ("Individual", "Chain", "Chain-2opt", "CTL-MST")}
        print(f"  T={T}: " + " | ".join(f"{m} {ms(v)}" for m, v in r.items())
              + f" | MST vs Chain {100 * (r['CTL-MST'].mean() / r['Chain'].mean() - 1):+.1f}%"
              + f" | profondeur MST {dep['CTL-MST']:.0f}, 2opt {dep['Chain-2opt']:.0f}")


def scale_T():
    print("\n=== Passage a l'echelle (MLP, 5 graines) ===")
    for T in (100, 200, 500, 1000, 2000, 5000):
        df = pd.read_csv(HERE / f"scaleT_{T}.csv")
        for b in sorted(df.b.unique()):
            r = {m: pick(df[(df.b == b) & (df.method == m)])[1] for m in ("Individual", "Star", "CTL-MST")}
            print(f"  T={T} b={b}: " + " | ".join(f"{m} {ms(v)}" for m, v in r.items()))


def timing():
    df = pd.read_csv(HERE / "timing.csv")
    print("\n=== Temps de construction / entrainement (CTL-MST lineaire, B=2000) ===")
    g = df.groupby("T")[["graph", "train"]].mean()
    g["part_%"] = 100 * g.graph / g.train
    print(g.round(4).to_string())


def mech():
    df = pd.read_csv(HERE / "mech_raw.csv")
    print("\n=== Erreur par tache ~ distance au parent + profondeur + log budget ===")
    for (m, B), g in df.groupby(["method", "B"]):
        co = []
        for _, h in g.groupby("rep"):
            z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
            X = np.column_stack([np.ones(len(h)), z(h.delta), z(h.depth), z(np.log(h.budget))])
            co.append(np.linalg.lstsq(X, z(h.err), rcond=None)[0][1:])
        co = np.array(co)
        print(f"  {m:10s} B={B}: distance {co[:, 0].mean():+.3f} ({(co[:, 0] > 0).mean():.0%} >0),"
              f" profondeur {co[:, 1].mean():+.3f} ({(co[:, 1] > 0).mean():.0%} >0)")


def tworegime():
    print("\n=== Arbre contre chaine : ecart relatif a Chain (%), pas choisi par methode ===")
    bs = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6)
    for m in ("CTL-MST", "CTL-Forest"):
        for om in (0.5, 2.0, 5.0, 20.0):
            df = pd.read_csv(HERE / f"tworegime_{om}.csv")
            line, ses = [], []
            for b in bs:
                g = df[df.b == b]
                vc, vm = pick(g[g.method == "Chain"])[1], pick(g[g.method == m])[1]
                d = 100 * (vm - vc) / vc
                line.append(f"{d.mean():+6.1f}")
                ses.append(d.std(ddof=1) / np.sqrt(len(d)))
            print(f"  {m:10s} ω={om:4}: " + " ".join(line) + f"   (SE {min(ses):.1f}-{max(ses):.1f})")


def baselines():
    print("\n=== Baselines renforcees (baselines_syn.json) ===")
    for r in json.load(open(HERE / "baselines_syn.json")):
        print(f"  {r['model']} {r['variant']:8s} b={r['b']} {r['method']:11s} {r['mean']:.3f} ± {r['se']:.3f} {r['opts']}")


if __name__ == "__main__":
    main_tables(); closed(); global_model(); large_T(); scale_T(); timing(); mech(); tworegime(); baselines()
