"""Algorithm 1, checked step by step.

The cascade makes four promises: a task starts from its parent's converged
parameters, the total number of gradient steps equals the budget, a task allotted
zero steps is an exact copy of its parent, and nothing reads the test split
before evaluation. Each is a test here.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cascader.allocation import individual_alloc, sle_alloc
from cascader.cascade import cascade_train, eval_all, independent_train, train_steps
from cascader.methods import METHODS, run_method
from cascader.models import model_factory
from cascader.trees import build_chain, build_mst, build_star


def _params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


# --------------------------------------------------------------------------
# The budget is what it says
# --------------------------------------------------------------------------

def test_total_gradient_steps_equal_the_budget(small, monkeypatch):
    """Count every backward pass and compare with B."""
    import cascader.cascade as C

    calls = {"steps": 0}
    real = C.train_steps

    def counting(model, X, y, steps, *a, **k):
        calls["steps"] += max(0, int(steps))
        return real(model, X, y, steps, *a, **k)

    monkeypatch.setattr(C, "train_steps", counting)

    tree = build_mst(small.nodes, small.D, small.root)
    B = 3 * small.T
    alloc = sle_alloc(tree, small.D, small.index, B)
    cascade_train(tree, small.X_train, small.y_train, alloc, 0.05,
                  model_factory("linear", small.d))
    assert calls["steps"] == B


def test_train_steps_takes_exactly_the_steps_asked(small):
    """One step is one full-batch gradient evaluation, and zero means untouched."""
    X, y = small.X_train[small.nodes[0]], small.y_train[small.nodes[0]]
    torch.manual_seed(0)
    model = model_factory("linear", small.d)()
    before = _params(model).clone()
    train_steps(model, X, y, 0, 0.1)
    assert torch.equal(_params(model), before)
    train_steps(model, X, y, 1, 0.1)
    assert not torch.equal(_params(model), before)


# --------------------------------------------------------------------------
# Inheritance
# --------------------------------------------------------------------------

def test_a_task_with_no_budget_is_a_copy_of_its_parent(small):
    """Legitimate under `exact`, and it must be an exact copy, not a re-init."""
    tree = build_star(small.nodes, small.D, small.root)
    alloc = {v: 0 for v in small.nodes}
    alloc[small.root] = 5
    models = cascade_train(tree, small.X_train, small.y_train, alloc, 0.05,
                           model_factory("linear", small.d))
    root_params = _params(models[small.root])
    for v in small.nodes:
        assert torch.allclose(_params(models[v]), root_params)


def test_a_child_starts_from_its_parents_converged_point(small):
    """Refit one edge by hand from the parent's parameters and compare."""
    tree = build_chain(small.nodes, small.D, small.root)
    alloc = {v: 2 for v in small.nodes}
    torch.manual_seed(0)
    models = cascade_train(tree, small.X_train, small.y_train, alloc, 0.05,
                           model_factory("linear", small.d))
    child = tree.edges[0][1]
    parent = tree.parent[child]

    # rebuild the parent, then take the child's own steps from it
    torch.manual_seed(0)
    again = cascade_train(tree, small.X_train, small.y_train,
                          {**{v: 0 for v in small.nodes},
                           **{small.root: alloc[small.root]}},
                          0.05, model_factory("linear", small.d))
    manual = train_steps(copy.deepcopy(again[parent]), small.X_train[child],
                         small.y_train[child], alloc[child], 0.05)
    assert torch.allclose(_params(models[child]), _params(manual), atol=1e-6)


def test_a_task_depends_only_on_its_ancestors(small):
    """Changing a sibling's budget must not move this task's parameters."""
    tree = build_mst(small.nodes, small.D, small.root)
    children = tree.children[small.root]
    if len(children) < 2:
        pytest.skip("the root has a single child in this draw")
    a, b = children[0], children[1]
    base = {v: 1 for v in small.nodes}

    torch.manual_seed(0)
    m1 = cascade_train(tree, small.X_train, small.y_train, base, 0.05,
                       model_factory("linear", small.d))
    torch.manual_seed(0)
    m2 = cascade_train(tree, small.X_train, small.y_train, {**base, b: 9}, 0.05,
                       model_factory("linear", small.d))
    assert torch.allclose(_params(m1[a]), _params(m2[a]))
    assert not torch.allclose(_params(m1[b]), _params(m2[b]))


def test_individual_shares_nothing(small):
    """Every task from its own initialization: no parameter crosses tasks."""
    alloc = individual_alloc(small.nodes, 0)
    torch.manual_seed(0)
    models = independent_train(small.nodes, small.X_train, small.y_train, alloc,
                               0.05, model_factory("linear", small.d))
    vectors = [_params(models[v]) for v in small.nodes]
    assert not all(torch.allclose(vectors[0], v) for v in vectors[1:])


# --------------------------------------------------------------------------
# No leakage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_fitting_never_touches_the_test_split(small, cfg, method):
    """Corrupt the test split before fitting; the fitted models must not change."""
    poisoned = copy.copy(small)
    poisoned.X_test = {v: np.full_like(small.X_test[v], np.nan)
                       for v in small.nodes}
    poisoned.y_test = {v: np.full_like(small.y_test[v], np.nan)
                       for v in small.nodes}
    factory = model_factory("linear", small.d)
    B = 2 * small.T

    torch.manual_seed(0)
    clean = run_method(method, small, B, 0.05, cfg, factory,
                       rng=np.random.default_rng(0))
    torch.manual_seed(0)
    dirty = run_method(method, poisoned, B, 0.05, cfg, factory,
                       rng=np.random.default_rng(0))
    for v in small.nodes:
        assert torch.allclose(_params(clean[v]), _params(dirty[v]))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def test_eval_all_is_the_unweighted_mean_over_tasks(small):
    from cascader.cascade import eval_task

    factory = model_factory("linear", small.d)
    torch.manual_seed(0)
    models = independent_train(small.nodes, small.X_train, small.y_train,
                               {v: 1 for v in small.nodes}, 0.05, factory)
    per_task = [eval_task(models[v], small.X_test[v], small.y_test[v],
                          small.task, small.stats[v]) for v in small.nodes]
    assert eval_all(models, small.X_test, small.y_test, small.task,
                    small.stats) == pytest.approx(float(np.mean(per_task)))


def test_regression_score_is_reported_in_the_data_units(small):
    """With normalization statistics, predictions are mapped back before scoring."""
    from cascader.cascade import eval_task

    v = small.nodes[0]
    torch.manual_seed(0)
    model = model_factory("linear", small.d)()
    normalized = eval_task(model, small.X_test[v], small.y_test[v], "regression")
    denormalized = eval_task(model, small.X_test[v], small.y_test[v], "regression",
                             small.stats[v])
    assert denormalized != pytest.approx(normalized)
    assert denormalized > 1.0     # targets have unit-scale sigma of 5 in Syn-10


def test_a_non_finite_loss_is_reported_not_swallowed(small):
    """The guard exists so a bad cell shows up as a skipped rate, not as NaN."""
    v = small.nodes[0]
    torch.manual_seed(0)
    poisoned = small.y_train[v].copy()
    poisoned[0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        train_steps(model_factory("linear", small.d)(), small.X_train[v],
                    poisoned, 5, 0.05)


def test_gradient_clipping_bounds_a_single_step(small):
    """Steps are clipped to unit gradient norm, so one step moves at most eta.

    This is why a large rate degrades rather than exploding, and why the rate
    sweep can go to 0.5 without every cell turning into NaN.
    """
    v = small.nodes[0]
    torch.manual_seed(0)
    model = model_factory("linear", small.d)()
    before = _params(model).clone()
    train_steps(model, small.X_train[v], small.y_train[v], 1, 3.0)
    assert float((_params(model) - before).norm()) <= 3.0 + 1e-5
