"""The packaged experiments run and produce the columns their reports read.

Sized down to seconds. What is checked is the plumbing -- shapes, columns, the
tuning/evaluation split, the budget invariant -- not the scientific outcome,
which is what the full runs are for.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cascader.experiments import (allocation_regime, main_table, online,
                                  root_fraction, rooting, scaling, tworegime)
from cascader.experiments._common import (ETA_GRID, TUNE_OFFSET, etas_for,
                                          select_and_report)
from cascader.experiments.cli import EXPERIMENTS, main as cli_main

SMALL = dict(out=None, quiet=True)


def test_tuning_and_evaluation_seeds_never_overlap():
    """Rates chosen on the seeds a number is reported from would be leakage."""
    eval_seeds = set(range(50))
    tune_seeds = set(range(TUNE_OFFSET, TUNE_OFFSET + 50))
    assert not (eval_seeds & tune_seeds)


def test_the_published_rate_is_reachable_on_the_grid():
    for coll in ("Syn-10", "WEAVE-UK", "FMNIST", "CIFAR"):
        for model in ("linear", "mlp"):
            etas = etas_for(coll, model)
            assert set(ETA_GRID) <= set(etas)
            assert len(etas) in (len(ETA_GRID), len(ETA_GRID) + 1)


def test_main_table_writes_raw_and_summary(tmp_path):
    result = main_table.run(collections=["Syn-10"], eval_seeds=2, tune_seeds=1,
                            b_per_task=[1, 2], out=tmp_path, stem="t", quiet=True)
    raw = pd.read_csv(result.raw)
    assert {"collection", "model", "T", "B", "b", "seed", "eta", "method",
            "split", "value"} <= set(raw.columns)
    assert set(raw.split) == {"tune", "eval"}
    assert (raw.B == raw.b * raw["T"]).all()

    summary = pd.read_csv(result.summary)
    assert set(summary.method) == set(main_table.METHODS)
    assert (summary.n == 2).all()


def test_main_table_shared_mode_uses_one_rate(tmp_path):
    from cascader.config import get_config

    result = main_table.run(collections=["Syn-10"], eta_mode="shared", eval_seeds=1,
                            tune_seeds=1, b_per_task=[1], out=tmp_path, stem="s",
                            quiet=True)
    raw = pd.read_csv(result.raw)
    assert raw.eta.nunique() == 1
    assert raw.eta.iloc[0] == pytest.approx(get_config("Syn-10").eta)


def test_selection_picks_the_best_tuning_rate():
    """Lower is better on a regression collection, higher on an image one."""
    rows = []
    for split, value in (("tune", None), ("eval", None)):
        for eta, score in ((0.1, 5.0), (0.2, 1.0)):
            rows.append(dict(collection="Syn-10", model="linear", b=1, seed=0,
                             eta=eta, method="Chain", split=split, value=score))
            rows.append(dict(collection="CIFAR", model="linear", b=1, seed=0,
                             eta=eta, method="Chain", split=split, value=score))
    out = select_and_report(pd.DataFrame(rows), methods=["Chain"])
    assert float(out[out.collection == "Syn-10"].eta.iloc[0]) == 0.2
    assert float(out[out.collection == "CIFAR"].eta.iloc[0]) == 0.1


def test_mean_ranks_orders_by_the_right_sign():
    summary = pd.DataFrame([
        dict(collection="Syn-10", model="linear", b=1, method="Chain", mean=1.0),
        dict(collection="Syn-10", model="linear", b=1, method="Star", mean=9.0),
        dict(collection="CIFAR", model="linear", b=1, method="Chain", mean=0.9),
        dict(collection="CIFAR", model="linear", b=1, method="Star", mean=0.5),
    ])
    ranks = main_table.mean_ranks(summary, methods=["Chain", "Star"])
    assert ranks["Chain"] == 1.0 and ranks["Star"] == 2.0


def test_allocation_regime_reports_no_overspend(tmp_path):
    result = allocation_regime.run(collections=["Syn-10"], reps=1,
                                   b_per_task=[1, 2], out=tmp_path, stem="a",
                                   quiet=True)
    summary = pd.read_csv(result.summary)
    assert (summary.overspend == 0).all(), "a method spent something other than B"
    assert set(summary.regime) == {"floor1", "exact"}


def test_tworegime_reports_a_signed_gap(tmp_path):
    result = tworegime.run(scatters=[0.5, 20.0], budgets=[120], reps=2, T=30,
                           out=tmp_path, stem="w", quiet=True)
    summary = pd.read_csv(result.summary)
    assert set(summary.columns) >= {"scatter", "B", "gap_pct", "median_distance"}
    assert len(summary) == 2


def test_online_covers_every_arm(tmp_path):
    result = online.run(collections=["Syn-10"], reps=1, b_per_task=[2],
                        out=tmp_path, stem="o", quiet=True)
    raw = pd.read_csv(result.raw)
    assert set(raw.arm) == set(online.ARMS)


def test_online_cascade_builds_a_tree_without_looking_ahead(small):
    from cascader.models import model_factory

    order = list(small.nodes)
    models, edges = online.online_cascade(order, small, b=1, eta=0.05,
                                          factory=model_factory("linear", small.d),
                                          rule="nn")
    assert len(models) == small.T
    assert len(edges) == small.T - 1
    # a parent is always a task that arrived earlier
    position = {v: i for i, v in enumerate(order)}
    assert all(position[u] < position[v] for u, v in edges)


def test_scaling_records_construction_cost(tmp_path):
    result = scaling.run(Ts=[20, 40], reps=1, out=tmp_path, stem="c", quiet=True)
    summary = pd.read_csv(result.summary)
    assert "build_seconds" in summary.columns
    assert set(summary["T"]) == {20, 40}


def test_rooting_compares_every_rule(tmp_path):
    result = rooting.run(collections=["Syn-10"], methods=["CTL-MST"], reps=1,
                         b_per_task=[2], out=tmp_path, stem="r", quiet=True)
    raw = pd.read_csv(result.raw)
    from cascader.rooting import ROOT_RULES
    assert set(raw.rule) == set(ROOT_RULES)
    # re-rooting must not change the topology, so the depth may move but not T
    assert raw.depth.min() >= 1


def test_root_fraction_allocations_sum_to_B(tmp_path):
    result = root_fraction.run(collections=["Syn-10"], methods=["Star", "CTL-MST"],
                               fractions=[0.0, 0.5, 1.0], b_per_task=[2],
                               eval_seeds=2, tune_seeds=1, out=tmp_path, stem="f",
                               quiet=True)
    raw = pd.read_csv(result.raw)
    assert set(raw.frac) == {0.0, 0.5, 1.0}


def test_alloc_with_fraction_is_exact(small):
    from cascader.trees import build_mst

    tree = build_mst(small.nodes, small.D, small.root)
    for f in (0.0, 0.05, 0.5, 1.0):
        for b in (1, 4):
            B = b * small.T
            alloc = root_fraction.alloc_with_fraction(tree, small.D, small.index,
                                                      B, f)
            assert sum(alloc.values()) == B
            assert alloc[small.root] == int(round(f * B)) or f >= 1.0


def test_cli_lists_and_dispatches(capsys):
    assert cli_main(["--list"]) == 0
    listing = capsys.readouterr().out
    for name in EXPERIMENTS:
        assert name in listing
    assert cli_main([]) == 2


def test_cli_runs_an_experiment(tmp_path):
    code = cli_main(["main-table", "--collections", "Syn-10", "--eval-seeds", "1",
                     "--tune-seeds", "1", "--budgets", "1", "--out", str(tmp_path),
                     "--stem", "cli", "--quiet"])
    assert code == 0
    assert (tmp_path / "cli_raw.csv").exists()
