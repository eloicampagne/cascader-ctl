"""Add Chain to the consolidated WEAVE run (same protocol/seeding as weave_all)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).parent.resolve(); sys.path.insert(0, str(HERE))
from table1_nonconvex import CONFIG, MLP_ETA, make_model, eval_all, run_method, weave_tasks, N_REPS_WEAVE

cfg = CONFIG["WEAVE-UK"]; rows = []
for rep in range(N_REPS_WEAVE):
    nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, seed = weave_tasks()
    d = Xn_tr[nodes[0]].shape[1]
    for mc in ("linear", "mlp"):
        eta = cfg["eta"] if mc == "linear" else MLP_ETA["WEAVE-UK"]
        factory = lambda: make_model(mc, d)
        rng = np.random.default_rng(70_000 + rep)
        for B in (500, 1000, 2000):
            torch.manual_seed(rep)
            models = run_method("Chain", nodes, Xn_tr, yn_tr, Xn_tr, yn_tr,
                                D_grad, seed, B, eta, cfg, factory, rng)
            rows.append(dict(model=mc, B=B, seed=rep, method="Chain",
                             value=eval_all(models, Xn_te, y_te, "regression", stats)))
        print(f"rep={rep} {mc}: " + " ".join(f"{r['B']}={r['value']:.0f}" for r in rows[-3:]), flush=True)
pd.DataFrame(rows).to_csv(HERE / "weave_chain_raw.csv", index=False)
df = pd.concat([pd.read_csv(HERE / "weave_all_raw.csv"), pd.DataFrame(rows)])
df.to_csv(HERE / "weave_all_raw.csv", index=False)
print(df.groupby(["model","B","method"])["value"].mean().unstack("method")[["Chain","CTL-MST","CTL-MSTc"]].round(0).to_string(), flush=True)
