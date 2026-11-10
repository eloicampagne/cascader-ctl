"""CTL on WEAVE-UK with a Transformer local model.

A more expressive local model than the MLP, beyond the
hidden-layer sweep and the CNN. This runs the faithful WEAVE protocol with a small
Transformer-encoder local model: the 54 features become 54 scalar tokens, a shared
embedding lifts each to d_model=16, a single multi-head self-attention encoder
block mixes them, mean-pooling and a linear read-out give the scalar target. Size
(~3.3k params) is comparable to the MLP (3265) and CNN (~2.4k), so the comparison
isolates the architecture, not the parameter count.

Everything else matches table1_nonconvex: gradient distance, medoid root, SLE
allocation, budget in gradient steps, learning rate calibrated on Individual.

    python3.12 weave_transformer.py --exp calibrate
    python3.12 weave_transformer.py --exp run --Bs 4000,8000 --reps 8
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats

from table1_nonconvex import weave_tasks, run_method, CONFIG, eval_all

HERE = Path(__file__).parent.resolve()
METHODS = ["Individual", "Star", "RandTree", "Chain", "CTL-MST", "CTL-MSTc"]


class TransformerHead(nn.Module):
    """Feature-token Transformer encoder for scalar regression."""

    def __init__(self, d, d_model=16, nhead=2, ff=32):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(d, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tok = self.embed(x.unsqueeze(-1)) + self.pos      # (n, d, d_model)
        z = self.enc(tok)                                  # (n, d, d_model)
        return self.head(z.mean(dim=1)).reshape(-1)


def _factory(d):
    return lambda: TransformerHead(d)


def calibrate():
    nodes, Xtr, ytr, Xte, yte, stats_d, D, seed = weave_tasks()
    d = Xtr[nodes[0]].shape[1]
    print("Transformer params:",
          sum(p.numel() for p in TransformerHead(d).parameters()))
    for B in (4000, 8000, 20000):
        for eta in (0.05, 0.1, 0.2, 0.5):
            torch.manual_seed(0)
            models = run_method("Individual", nodes, Xtr, ytr, Xtr, ytr, D, seed,
                                B, eta, CONFIG["WEAVE-UK"], _factory(d),
                                np.random.default_rng(0))
            rmse = eval_all(models, Xte, yte, "regression", stats_d)
            print(f"  B={B:6d} eta={eta}: Individual RMSE={rmse:.0f}", flush=True)


def run(Bs, reps, eta):
    cfg = CONFIG["WEAVE-UK"]
    rows = []
    for rep in range(reps):
        nodes, Xtr, ytr, Xte, yte, stats_d, D, seed = weave_tasks()
        d = Xtr[nodes[0]].shape[1]
        for B in Bs:
            for m in METHODS:
                torch.manual_seed(rep)
                models = run_method(m, nodes, Xtr, ytr, Xtr, ytr, D, seed, B, eta,
                                    cfg, _factory(d), np.random.default_rng(100 + rep))
                rmse = eval_all(models, Xte, yte, "regression", stats_d)
                rows.append(dict(rep=rep, B=B, method=m, rmse=rmse))
        print(f"rep {rep} done: " + " ".join(
            f"{r['method']}@{r['B']}={r['rmse']:.0f}" for r in rows[-len(METHODS):]),
            flush=True)

    df = pd.DataFrame(rows)
    summ = df.groupby(["B", "method"])["rmse"].agg(["mean", "std"]).reset_index()
    print("\n=== WEAVE-UK, Transformer local model, RMSE (lower better) ===")
    print(summ.pivot(index="B", columns="method", values="mean")
          .to_string(float_format=lambda v: f"{v:.0f}"))
    print("\n=== CTL-MST and CTL-MSTc vs Individual / Star (paired t) ===")
    for B in Bs:
        piv = df[df["B"] == B].pivot_table(index="rep", columns="method", values="rmse")
        for m in ("CTL-MST", "CTL-MSTc"):
            for base in ("Individual", "Star"):
                p = stats.ttest_rel(piv[base], piv[m]).pvalue
                print(f"  B={B}: {m} {piv[m].mean():.0f} vs {base} {piv[base].mean():.0f}"
                      f"  gain {piv[base].mean()-piv[m].mean():+.0f}  p={p:.3g}")
    df.to_csv(HERE / "weave_transformer_raw.csv", index=False)
    summ.to_csv(HERE / "weave_transformer_summary.csv", index=False)
    print("\nwrote weave_transformer_{raw,summary}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="calibrate")
    ap.add_argument("--Bs", default="4000,8000")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.1)
    a = ap.parse_args()
    if a.exp == "calibrate":
        calibrate()
    else:
        run([int(x) for x in a.Bs.split(",")], a.reps, a.eta)
