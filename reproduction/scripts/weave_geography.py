"""
Geography-based spanning tree on WEAVE-UK (NeurIPS 2026 rebuttal, Lufy W4).

Tests whether the learned task geometry (gradient proxy) beats an informed domain
prior: a spanning tree built from the physical coordinates of the substations
rather than from task distances. Substation coordinates come from the WEAVE
smart-meter parquet (lv_feeder_unique_id -> geometry struct x/y, lon/lat), the
same source as generate_data.ipynb, cached to weave_coords_local.csv by
weave_coords_fetch (run once).

Everything else matches the faithful Table 1 protocol of table1_nonconvex.py.
Geo-MST reuses the exact cascade, SLE allocation, and MST builder, only the
distance matrix and the root change: geographic Euclidean distance in local
meters, rooted at the geographic medoid. We compare it against the gradient-proxy
CTL-MST and CTL-MSTc and the Individual and Star baselines, linear and MLP.

Run: python3.12 weave_geography.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from table1_nonconvex import (
    CONFIG, MLP_ETA, make_model, eval_all, run_method, cascade_train,
    tree_mst, sle_alloc, weave_tasks, N_REPS_WEAVE,
)

# Raw coordinates (id, x=lon, y=lat in degrees) fetched from the WEAVE parquet.
COORDS_RAW = Path("/private/tmp/claude-502/-Users-eloicampagne-Desktop-EDF/"
                  "57d77548-5f54-462d-ba8f-f1bf39f01f47/scratchpad/weave_coords_raw.csv")
COORDS = HERE / "weave_coords.csv"          # cached copy (id_unique, lon, lat, xm, ym)
Bs = (500, 1000, 2000)
N_REPS = N_REPS_WEAVE


def _load_coords():
    """Load lon/lat, convert to local-meter (xm, ym) via an equirectangular
    projection about the mean latitude, and cache. Small area (~10 km), so the
    approximation is well under a metre of relative error."""
    if COORDS.exists():
        return pd.read_csv(COORDS)
    c = pd.read_csv(COORDS_RAW).rename(columns={"id": "id_unique", "x": "lon", "y": "lat"})
    c = c.drop_duplicates("id_unique")
    lat0 = np.deg2rad(c["lat"].mean())
    c["xm"] = np.deg2rad(c["lon"] - c["lon"].mean()) * 6371000.0 * np.cos(lat0)
    c["ym"] = np.deg2rad(c["lat"] - c["lat"].mean()) * 6371000.0
    c.to_csv(COORDS, index=False)
    return c


def geo_distance_matrix(nodes):
    """Euclidean distance in local meters between substations, aligned to nodes.
    Returns (D_geo, geo_root, kept_nodes). Nodes without coordinates are dropped."""
    c = _load_coords().set_index("id_unique")
    kept = [v for v in nodes if v in c.index]
    xy = c.loc[kept, ["xm", "ym"]].to_numpy()
    diff = xy[:, None, :] - xy[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    geo_root = kept[int(np.argmin(D.sum(1)))]     # geographic medoid
    return D, geo_root, kept


def run_geo_mst(nodes, X_tr, y_tr, D_geo, geo_root, B, eta, cfg, factory):
    """CTL-MST cascade but on the geographic tree: MST + SLE from D_geo, rooted
    at the geographic medoid. Same budget accounting as every other method."""
    root, dir_edges, _ = tree_mst(nodes, D_geo, geo_root)
    alloc = sle_alloc(nodes, root, dir_edges, D_geo, B)
    return cascade_train(dir_edges, root, X_tr, y_tr, alloc, eta,
                         cfg["task"], cfg["reg"], factory)


def main():
    if not COORDS.exists() and not COORDS_RAW.exists():
        raise SystemExit(f"missing coordinates ({COORDS} or {COORDS_RAW})")
    cfg = CONFIG["WEAVE-UK"]
    rows = []
    for rep in range(N_REPS):
        nodes, Xn_tr, yn_tr, Xn_te, y_te, stats, D_grad, seed = weave_tasks()
        d = Xn_tr[nodes[0]].shape[1]
        D_geo, geo_root, kept = geo_distance_matrix(nodes)
        # restrict every method to the nodes that have coordinates, for a fair match
        keptset = set(kept)
        sub = lambda dct: {v: dct[v] for v in kept}
        Xk_tr, yk_tr, Xk_te, yk_te, statsk = sub(Xn_tr), sub(yn_tr), sub(Xn_te), sub(y_te), sub(stats)
        idx = [nodes.index(v) for v in kept]
        D_grad_k = D_grad[np.ix_(idx, idx)]
        # keep the exact Table 1 rooting for the gradient methods: the raw-feature
        # medoid returned by weave_tasks. Only the geography tree uses its own root.
        grad_root = seed if seed in keptset else kept[int(np.argmin(D_grad_k.sum(1)))]
        if rep == 0:
            print(f"nodes with coords: {len(kept)}/{len(nodes)}", flush=True)
        for mc in ("linear", "mlp"):
            torch.manual_seed(rep)
            eta = cfg["eta"] if mc == "linear" else MLP_ETA["WEAVE-UK"]
            factory = lambda: make_model(mc, d)
            rng = np.random.default_rng(40_000 + rep)
            for B in Bs:
                def rec(method, models):
                    rows.append(dict(model=mc, B=B, seed=rep, method=method,
                                     value=eval_all(models, Xk_te, yk_te, "regression", statsk)))
                # geography tree
                rec("Geo-MST", run_geo_mst(kept, Xk_tr, yk_tr, D_geo, geo_root, B, eta, cfg, factory))
                # gradient-proxy references, same node subset and rooting rule
                for m in ("Individual", "Star", "CTL-MST", "CTL-MSTc"):
                    rec(m, run_method(m, kept, Xk_tr, yk_tr, Xk_tr, yk_tr,
                                      D_grad_k, grad_root, B, eta, cfg, factory, rng))
            done = [r for r in rows if r["seed"] == rep and r["model"] == mc]
            print(f"  rep={rep} {mc}: " + " ".join(
                f"{r['method']}@{r['B']}={r['value']:.0f}" for r in done), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "weave_geo_raw.csv", index=False)
    summ = (df.groupby(["model", "B", "method"])["value"]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(HERE / "weave_geo_summary.csv", index=False)
    print("\n=== WEAVE geography vs gradient (mean RMSE) ===", flush=True)
    print(summ.pivot_table(index=["model", "B"], columns="method", values="mean").to_string(), flush=True)


if __name__ == "__main__":
    main()
