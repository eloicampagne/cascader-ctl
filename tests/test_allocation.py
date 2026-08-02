"""The budget contract.

Every allocator hands out exactly ``B`` steps. The paper reports methods at
equal budget, so an allocator that overspends is not a rounding detail: it hands
one method more compute than the number it is credited with, and the comparison
stops being a comparison.

The last test in this file is a regression guard for a real defect. An earlier
implementation clamped each share up to one *after* apportioning, so the total
exceeded ``B`` by the number of tasks whose share had rounded to zero -- 74% of
``B`` for the MST on WEAVE-UK at ``B = T``, and worst precisely for the methods
whose weights concentrate.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascader.allocation import (apportion, individual_alloc, sle_alloc,
                                 sle_weights, uniform_alloc)
from cascader.methods import METHODS, allocate, build_structure
from cascader.trees import build_mst, build_star

REGIMES = ("floor1", "exact")
BUDGETS_PER_TASK = (1, 2, 4, 10)


# --------------------------------------------------------------------------
# Exactness, on every path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("b", BUDGETS_PER_TASK)
def test_every_method_spends_exactly_B(small, cfg, method, regime, b):
    B = b * small.T
    if method == "Individual":
        alloc = individual_alloc(small.nodes, B)
    else:
        tree = build_structure(method, small, cfg, np.random.default_rng(0))
        alloc = allocate(method, tree, small, B, cfg, regime=regime)
    assert sum(alloc.values()) == B
    assert all(v >= 0 for v in alloc.values())
    assert all(isinstance(v, (int, np.integer)) for v in alloc.values())
    assert set(alloc) == set(small.nodes)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("B", [0, 1, 7, 24, 25, 26, 999])
def test_exactness_at_awkward_budgets(small, regime, B):
    """Budgets below, at and above the task count all have to sum to B."""
    tree = build_mst(small.nodes, small.D, small.root)
    for alloc in (sle_alloc(tree, small.D, small.index, B, regime=regime),
                  uniform_alloc(small.nodes, small.root, B, regime=regime),
                  individual_alloc(small.nodes, B)):
        assert sum(alloc.values()) == B


def test_apportionment_is_largest_remainder():
    """Floors, then the largest fractional remainders get the leftover units."""
    w = np.array([1.0, 1.0, 1.0])
    assert sorted(apportion(w, 10)) == [3, 3, 4]
    w = np.array([0.5, 0.3, 0.2])
    got = apportion(w, 10)
    assert got.sum() == 10
    np.testing.assert_array_equal(got, [5, 3, 2])
    # 7 * (0.5, 0.3, 0.2) = (3.5, 2.1, 1.4): floors (3, 2, 1), one unit left,
    # largest remainder is the first.
    np.testing.assert_array_equal(apportion(np.array([0.5, 0.3, 0.2]), 7), [4, 2, 1])


def test_apportionment_of_zero_and_degenerate_weights():
    assert apportion(np.array([1.0, 2.0]), 0).sum() == 0
    assert apportion(np.zeros(4), 9).sum() == 9      # falls back to uniform


# --------------------------------------------------------------------------
# What each regime promises
# --------------------------------------------------------------------------

@pytest.mark.parametrize("b", (1, 2, 4))
def test_floor1_refines_every_task_at_least_once(small, b):
    tree = build_mst(small.nodes, small.D, small.root)
    alloc = sle_alloc(tree, small.D, small.index, b * small.T, regime="floor1")
    assert min(alloc.values()) >= 1


def test_exact_may_starve_a_task_and_that_is_intended(small):
    """Under `exact` a light subtree can round to zero and inherit its parent."""
    tree = build_mst(small.nodes, small.D, small.root)
    alloc = sle_alloc(tree, small.D, small.index, small.T, regime="exact")
    assert sum(alloc.values()) == small.T
    assert min(alloc.values()) == 0


def test_floor1_falls_back_below_the_task_count(small):
    """With B < T not every task can be refined, so floor1 cannot apply."""
    B = small.T - 1
    tree = build_mst(small.nodes, small.D, small.root)
    a = sle_alloc(tree, small.D, small.index, B, regime="floor1")
    b = sle_alloc(tree, small.D, small.index, B, regime="exact")
    assert sum(a.values()) == sum(b.values()) == B
    assert a == b


def test_regime_default_follows_the_environment(small, monkeypatch):
    from cascader.allocation import default_regime

    monkeypatch.setenv("CASCADER_ALLOC", "exact")
    assert default_regime() == "exact"
    monkeypatch.setenv("CASCADER_ALLOC", "floor1")
    assert default_regime() == "floor1"
    monkeypatch.setenv("CASCADER_ALLOC", "nonsense")
    with pytest.raises(ValueError):
        default_regime()
    monkeypatch.delenv("CASCADER_ALLOC")
    assert default_regime() == "floor1"


# --------------------------------------------------------------------------
# The SLE formula, as stated
# --------------------------------------------------------------------------

def test_sle_weights_match_the_published_formula(small):
    r"""w_v = |s_v| log(1 + d(pa(v), v)), and the root carries no weight."""
    tree = build_mst(small.nodes, small.D, small.root)
    w = sle_weights(tree, small.D, small.index)
    sizes = tree.subtree_sizes()
    idx = small.index
    assert small.root not in w
    for v in small.nodes:
        if v == small.root:
            continue
        expected = sizes[v] * np.log1p(small.D[idx[tree.parent[v]], idx[v]])
        assert w[v] == pytest.approx(expected)


def test_subtree_sizes_are_consistent(small):
    tree = build_mst(small.nodes, small.D, small.root)
    sizes = tree.subtree_sizes()
    assert sizes[tree.root] == small.T
    for v in small.nodes:
        children = [c for c in small.nodes if tree.parent.get(c) == v]
        assert sizes[v] == 1 + sum(sizes[c] for c in children)


def test_heavier_subtree_and_longer_edge_get_more_budget(small):
    """The weights are monotone in both factors, so the allocation is too."""
    tree = build_mst(small.nodes, small.D, small.root)
    w = sle_weights(tree, small.D, small.index)
    alloc = sle_alloc(tree, small.D, small.index, 50 * small.T, regime="exact")
    ordered = sorted(w, key=w.get)
    lightest, heaviest = ordered[0], ordered[-1]
    assert alloc[heaviest] >= alloc[lightest]


# --------------------------------------------------------------------------
# Individual and uniform
# --------------------------------------------------------------------------

def test_individual_splits_the_remainder(small):
    """B // T each, and the first B % T tasks get one more. No step is dropped."""
    B = 3 * small.T + 7
    alloc = individual_alloc(small.nodes, B)
    assert sum(alloc.values()) == B
    assert sorted(set(alloc.values())) == [B // small.T, B // small.T + 1]
    assert sum(1 for v in alloc.values() if v == B // small.T + 1) == B % small.T


def test_uniform_is_uniform_off_the_root(small):
    alloc = uniform_alloc(small.nodes, small.root, 10 * small.T, root_frac=0.2,
                          regime="exact")
    others = [alloc[v] for v in small.nodes if v != small.root]
    assert max(others) - min(others) <= 1
    assert sum(alloc.values()) == 10 * small.T


def test_root_fraction_moves_budget_to_the_root(small):
    tree = build_mst(small.nodes, small.D, small.root)
    B = 20 * small.T
    low = sle_alloc(tree, small.D, small.index, B, root_frac=0.0, regime="exact")
    high = sle_alloc(tree, small.D, small.index, B, root_frac=0.5, regime="exact")
    assert high[small.root] > low[small.root]
    assert sum(low.values()) == sum(high.values()) == B


# --------------------------------------------------------------------------
# The regression guard
# --------------------------------------------------------------------------

def test_floor_is_applied_before_apportionment(small):
    """The historical defect: clamping to one *after* apportioning overspends.

    Under `floor1` the floor is part of the apportionment, so the total stays at
    ``B`` no matter how concentrated the weights are. The check below builds the
    most concentrated case available -- a star, whose leaves all hang off the
    root with wildly unequal edge lengths -- at the tightest budget, where the
    old code was worst.
    """
    tree = build_star(small.nodes, small.D, small.root)
    for b in (1, 2, 4):
        B = b * small.T
        alloc = sle_alloc(tree, small.D, small.index, B, regime="floor1")
        assert sum(alloc.values()) == B, "floor1 overspent, the old bug is back"
        assert min(alloc.values()) >= 1

    # And the arithmetic that used to fail: post-hoc clamping adds one step per
    # zero-weight task, which is exactly the overspend that was observed.
    exact = sle_alloc(tree, small.D, small.index, small.T, regime="exact")
    zeros = sum(1 for v in exact.values() if v == 0)
    naive_total = sum(max(v, 1) for v in exact.values())
    assert naive_total == small.T + zeros
    if zeros:
        assert naive_total > small.T


@pytest.mark.parametrize("method", [m for m in METHODS if m != "Individual"])
def test_no_method_is_handed_extra_compute(small, cfg, method):
    """Cross-method check: at a given B, every method spends the same total."""
    B = 2 * small.T
    tree = build_structure(method, small, cfg, np.random.default_rng(0))
    alloc = allocate(method, tree, small, B, cfg, regime="floor1")
    assert sum(alloc.values()) == B
