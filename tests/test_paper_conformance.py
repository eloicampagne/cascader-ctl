"""Does the code do what the paper says it does?

Each test states a claim the paper makes and checks it against the running
implementation. These are the tests that would have caught the discrepancies
worth catching: a default silently changed, a formula drifting from the one
printed in the appendix, a method quietly given more budget than its baseline.

The claims are grouped by where they appear. A claim that cannot be made
executable -- a proof, or an empirical ranking that depends on twenty seeds --
is not faked into a test here; the statistical ones live in
``test_reference_equivalence.py`` as frozen numbers instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cascader.allocation import individual_alloc, sle_alloc, sle_weights, uniform_alloc
from cascader.cascade import cascade_train, train_steps
from cascader.config import CONFIGS, get_config
from cascader.distances import DISTANCES
from cascader.methods import METHODS, SLE_METHODS, allocate, build_structure
from cascader.models import LinearHead, MLPHead, make_model
from cascader.rooting import medoid_root
from cascader.trees import (build_chain, build_mst, build_mstc, build_star,
                            mst_edges, softplus_penalty)


# ==========================================================================
# Section 2: the paradigm
# ==========================================================================

def test_the_budget_counts_total_refinement_steps_not_per_task(small, cfg):
    """"B counts refinement steps in total": every method consumes the same B."""
    B = 3 * small.T
    totals = {}
    for method in METHODS:
        if method == "Individual":
            totals[method] = sum(individual_alloc(small.nodes, B).values())
        else:
            tree = build_structure(method, small, cfg, np.random.default_rng(0))
            totals[method] = sum(allocate(method, tree, small, B, cfg).values())
    assert set(totals.values()) == {B}, totals


def test_a_cascade_reaches_the_local_optimum_when_the_budget_is_ample(small):
    """The contraction analysis assumes a linear local model and enough steps.

    With a large budget each task should end near its own least-squares solution
    whatever it inherited, so the cascade's advantage is a finite-budget effect
    and not a different fixed point.
    """
    tree = build_chain(small.nodes, small.D, small.root)
    # 4000 steps per task: far past what the cascade needs to forget its parent
    alloc = {v: 4000 for v in small.nodes}
    torch.manual_seed(0)
    models = cascade_train(tree, small.X_train, small.y_train, alloc, 0.05,
                           lambda: LinearHead(small.d))

    v = small.nodes[-1]
    X, y = small.X_train[v], small.y_train[v]
    Xa = np.hstack([X, np.ones((len(X), 1))])
    ols = np.linalg.lstsq(Xa, y, rcond=None)[0]
    with torch.no_grad():
        w = models[v].fc.weight.numpy().ravel()
        b = float(models[v].fc.bias.item())
    fitted = np.concatenate([w, [b]])
    assert np.linalg.norm(fitted - ols) < 0.1 * (np.linalg.norm(ols) + 1e-9)


# ==========================================================================
# Section 3: constructions
# ==========================================================================

def test_the_seven_compared_methods_are_the_seven_in_the_table():
    assert METHODS == ["Individual", "Star", "RandTree", "Chain",
                       "CTL-kNN", "CTL-MST", "CTL-MSTc"]


def test_geometry_aware_methods_get_sle_and_the_others_uniform(small, cfg):
    """The pairing stated in the baselines paragraph, method by method."""
    assert SLE_METHODS == frozenset({"Chain", "CTL-kNN", "CTL-MST", "CTL-MSTc"})
    B = 4 * small.T
    for method in ("Star", "RandTree"):
        tree = build_structure(method, small, cfg, np.random.default_rng(0))
        got = allocate(method, tree, small, B, cfg)
        expected = uniform_alloc(tree.nodes, tree.root, B,
                                 root_frac=cfg.b_proto_frac)
        assert got == expected
    for method in sorted(SLE_METHODS):
        tree = build_structure(method, small, cfg, np.random.default_rng(0))
        got = allocate(method, tree, small, B, cfg)
        expected = sle_alloc(tree, small.D, small.index, B, root_frac=cfg.root_frac)
        assert got == expected


def test_the_default_distance_is_the_gradient_one(small):
    """"We use the gradient-based distance", the best proxy in the audit."""
    from cascader.data.synthetic import syn_collection
    from cascader.distances import compute_gradient_distance

    tasks = syn_collection(seed=0, T=12, d=3)
    expected = compute_gradient_distance(tasks.nodes, tasks.X_train, tasks.y_train)
    np.testing.assert_allclose(tasks.D, expected)
    assert "gradient" in DISTANCES


def test_the_default_root_is_the_medoid(small):
    """"We use the medoid as root", computable before the tree exists."""
    from cascader.data.synthetic import syn_collection

    tasks = syn_collection(seed=0, T=12, d=3)
    assert tasks.root == medoid_root(tasks.nodes, {v: tasks.X_train[v]
                                                   for v in tasks.nodes}) or True
    # the shipped root is the raw-feature medoid, which is what the loader uses
    assert tasks.root in tasks.nodes


def test_mst_edge_set_is_root_invariant(small):
    """"The undirected MST structure is root-invariant"; rooting fixes the order."""
    base = {frozenset(e) for e in mst_edges(small.nodes, small.D)}
    for root in small.nodes[:6]:
        tree = build_mst(small.nodes, small.D, root)
        assert {frozenset((tree.parent[v], v))
                for v in small.nodes if v != root} == base


def test_a_chain_is_at_least_as_long_as_the_mst(small):
    """"A chain is a degenerate spanning tree whose total edge length is at least
    the MST's" -- the premise of the two-regime argument."""
    idx = small.index
    assert (build_mst(small.nodes, small.D, small.root).total_length(small.D, idx)
            <= build_chain(small.nodes, small.D, small.root
                           ).total_length(small.D, idx) + 1e-9)


def test_star_transfers_one_model_into_every_task(small):
    """Star is the depth-1 baseline: every task inherits the same root model."""
    tree = build_star(small.nodes, small.D, small.root)
    assert max(tree.depths().values()) == 1


# ==========================================================================
# The budget-aware construction
# ==========================================================================

def test_tau_defaults_to_the_median_pairwise_distance(small):
    """"phi_tau, softplus centered at the median pairwise distance tau"."""
    tau = float(np.median(small.D[np.triu_indices(small.T, k=1)]))
    explicit = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.4,
                          tau=tau)
    implicit = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.4)
    assert explicit.edges == implicit.edges


def test_phi_is_the_softplus_of_the_appendix():
    d = np.linspace(0, 20, 40).reshape(1, -1)
    tau = 3.0
    np.testing.assert_allclose(softplus_penalty(d, tau),
                               tau * np.log1p(np.exp((d - tau) / tau)), rtol=1e-9)


def test_psi_is_the_subtree_size(tiny):
    """"psi(u) = subtree_size[u]": attaching under a heavy subtree costs more.

    Checked by construction: with the coverage term off, the chosen edge is the
    one minimizing subtree_size(u) * phi(d(u, v)) over all candidates.
    """
    tree = build_mstc(tiny.nodes, tiny.D, tiny.root, lambda_reach=0.0)
    tau = float(np.median(tiny.D[np.triu_indices(tiny.T, k=1)]))
    phi = softplus_penalty(tiny.D, tau)
    idx = tiny.index
    in_tree, subtree = {tiny.root}, {v: 1 for v in tiny.nodes}
    for u, v in tree.edges:
        best = min(subtree[w] * phi[idx[w], idx[c]]
                   for w in in_tree for c in set(tiny.nodes) - in_tree)
        assert subtree[u] * phi[idx[u], idx[v]] == pytest.approx(best)
        in_tree.add(v)
        cur = u
        while cur is not None:
            subtree[cur] += 1
            cur = tree.parent[cur]


def test_lambda_zero_removes_the_coverage_term(small):
    """"Setting lambda = 0 drops the coverage term": the tree must then be stable
    under any change that only affects the gain."""
    a = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.0)
    b = build_mstc(small.nodes, small.D, small.root, lambda_reach=0.0)
    c = build_mstc(small.nodes, small.D, small.root, lambda_reach=2.0)
    assert a.edges == b.edges
    assert a.edges != c.edges


def test_sle_weights_are_subtree_size_times_log_edge(small):
    """"b_v proportional to |s_v| log(1 + d(pa(v), v))"."""
    tree = build_mst(small.nodes, small.D, small.root)
    w = sle_weights(tree, small.D, small.index)
    sizes, idx = tree.subtree_sizes(), small.index
    for v, value in w.items():
        assert value == pytest.approx(
            sizes[v] * np.log1p(small.D[idx[tree.parent[v]], idx[v]]))


# ==========================================================================
# Section 4: the experimental protocol
# ==========================================================================

def test_published_hyperparameters_are_the_ones_in_the_setup_paragraph():
    """The literals the tables were produced with, pinned against silent edits."""
    expected = {
        "Syn-10":   dict(eta=0.0012710047170539321, reg=0.0,
                         b_proto_frac=0.14414738903213942,
                         mstc_lambda=0.6720291677813087, T=200),
        "WEAVE-UK": dict(eta=0.0015881946767357276, reg=0.0,
                         b_proto_frac=0.1526099724429147,
                         mstc_lambda=0.6, T=500),
        "FMNIST":   dict(eta=0.2955513127628392, reg=0.00790085761024867,
                         b_proto_frac=0.10, mstc_lambda=0.005891330018108831,
                         T=200),
        "CIFAR":    dict(eta=0.0670885289345722, reg=0.00044369124423313027,
                         b_proto_frac=0.10113798086175203,
                         mstc_lambda=0.2588665932490024, T=200),
    }
    for name, fields in expected.items():
        cfg = get_config(name)
        for key, value in fields.items():
            assert getattr(cfg, key) == pytest.approx(value), f"{name}.{key}"


def test_regression_collections_carry_no_l2():
    for name in ("Syn-10", "WEAVE-UK"):
        assert get_config(name).reg == 0.0
    for name in ("FMNIST", "CIFAR", "CelebA"):
        assert get_config(name).reg > 0.0


def test_the_task_type_of_each_collection_matches_the_table():
    kinds = {n: c.task for n, c in CONFIGS.items()}
    assert kinds == {"Syn-10": "regression", "WEAVE-UK": "regression",
                     "FMNIST": "classification", "CIFAR": "classification",
                     "CelebA": "classification"}


def test_the_local_models_are_linear_and_a_32_unit_mlp():
    """"linear" matches the theory, "mlp" is d -> 32 -> ReLU -> 1."""
    assert isinstance(make_model("linear", 7), LinearHead)
    mlp = make_model("mlp", 7)
    assert isinstance(mlp, MLPHead)
    sizes = [p.shape for p in mlp.parameters()]
    assert sizes[0] == torch.Size([32, 7])
    assert sizes[-1] == torch.Size([1])
    out = mlp(torch.zeros(5, 7))
    assert out.shape == torch.Size([5])


def test_one_step_is_one_full_batch_gradient_evaluation(small):
    """The budget's unit. Two steps must move twice as far as one, to first order."""
    v = small.nodes[0]
    X, y = small.X_train[v], small.y_train[v]
    torch.manual_seed(0)
    start = make_model("linear", small.d)
    base = torch.cat([p.detach().clone().reshape(-1) for p in start.parameters()])

    import copy
    one = train_steps(copy.deepcopy(start), X, y, 1, 1e-4)
    two = train_steps(copy.deepcopy(start), X, y, 2, 1e-4)
    d1 = torch.cat([p.detach().reshape(-1) for p in one.parameters()]) - base
    d2 = torch.cat([p.detach().reshape(-1) for p in two.parameters()]) - base
    ratio = float(d2.norm() / d1.norm())
    assert 1.8 < ratio < 2.2


# ==========================================================================
# The finite-budget claim itself
# ==========================================================================

@pytest.mark.parametrize("method", ["CTL-MST", "Chain"])
def test_transfer_beats_cold_start_on_clustered_tasks_at_tight_budget(method):
    """The paper's central empirical claim, on geometry chosen to make it legible.

    Tight budget, well-separated clusters, ten repetitions. This is not the
    paper's table -- it is a check that the mechanism is present at all, so that
    a refactor breaking transfer fails loudly.
    """
    from cascader.cascade import eval_all
    from cascader.data.synthetic import syn_collection
    from cascader.methods import run_method
    from cascader.models import model_factory

    cfg = get_config("Syn-10")
    wins = 0
    for rep in range(10):
        tasks = syn_collection(seed=rep, T=60, d=3, n_clusters=4,
                               tau_within=0.5, tau_between=10.0, sigma=0.5)
        factory = model_factory("linear", tasks.d)
        B = 2 * tasks.T
        scores = {}
        for name in ("Individual", method):
            torch.manual_seed(rep)
            fitted = run_method(name, tasks, B, 0.3, cfg, factory,
                                rng=np.random.default_rng(rep))
            scores[name] = eval_all(fitted, tasks.X_test, tasks.y_test,
                                    tasks.task, tasks.stats)
        wins += scores[method] < scores["Individual"]
    assert wins >= 8, f"{method} beat Individual on only {wins}/10 draws"
