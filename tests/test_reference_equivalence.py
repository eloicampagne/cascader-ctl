"""The package reproduces the pipeline the paper's numbers came from.

``reference_values.json`` holds scores produced by the experiment scripts that
generated the reported tables, on a fixed synthetic draw, for all seven methods,
three per-task budgets and both local model classes. The package has to match
them to floating-point precision.

This is the test that makes the port trustworthy. Unit tests say each piece
obeys its own definition; this one says the assembled pipeline is the same
pipeline, including the pieces where a plausible-looking change would go
unnoticed -- the order tasks are enumerated in, which task receives an
apportionment remainder, the exact tie-breaking inside a greedy construction.

Regenerating the file is a deliberate act. If a change to the method is intended,
the numbers move and the diff shows by how much; if it is not intended, this
fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cascader.cascade import eval_all
from cascader.config import get_config
from cascader.data.synthetic import syn_collection
from cascader.methods import METHODS, run_method
from cascader.models import model_factory

REFERENCE = Path(__file__).parent / "reference_values.json"

#: Draws the reference was produced on: (tag, task count, model class, rate).
CASES = (("linear-T60", 60, "linear", 0.3),
         ("mlp-T40", 40, "mlp", 0.2))

#: The reference draw. Changing any of these invalidates the stored values.
DRAW = dict(seed=11, d=3, n_clusters=5, tau_within=10.0, tau_between=10.0,
            sigma=5.0, n_train=64, n_test=128)


@pytest.fixture(scope="module")
def reference() -> dict:
    if not REFERENCE.exists():
        pytest.skip("reference_values.json is absent")
    return json.loads(REFERENCE.read_text())


@pytest.mark.parametrize("tag,T,model,eta", CASES)
@pytest.mark.parametrize("b", (1, 2, 4))
@pytest.mark.parametrize("method", METHODS)
def test_matches_the_reference_pipeline(reference, tag, T, model, eta, b, method):
    key = f"{tag}|{b}|{method}"
    if key not in reference:
        pytest.skip(f"{key} not in the reference")

    tasks = syn_collection(T=T, **DRAW)
    cfg = get_config("Syn-10")
    torch.manual_seed(2)
    fitted = run_method(method, tasks, b * T, eta, cfg,
                        model_factory(model, tasks.d),
                        rng=np.random.default_rng(9), regime="floor1")
    got = eval_all(fitted, tasks.X_test, tasks.y_test, tasks.task, tasks.stats)
    assert got == pytest.approx(reference[key], abs=1e-9), (
        f"{key}: reference {reference[key]}, got {got}")


def test_the_reference_draw_is_the_published_synthetic_geometry():
    """Syn-10 as the paper defines it, so the stored numbers are on-protocol."""
    assert DRAW["d"] == 3 and DRAW["n_clusters"] == 5
    assert DRAW["tau_within"] == DRAW["tau_between"] == 10.0
    assert DRAW["sigma"] == 5.0
    assert DRAW["n_train"] == 64 and DRAW["n_test"] == 128


def test_a_rerun_is_bit_identical(small, cfg):
    """Same seed, same numbers: nothing in the pipeline reads global state."""
    factory = model_factory("linear", small.d)
    out = []
    for _ in range(2):
        torch.manual_seed(4)
        fitted = run_method("CTL-MSTc", small, 2 * small.T, 0.1, cfg, factory,
                            rng=np.random.default_rng(4))
        out.append(eval_all(fitted, small.X_test, small.y_test, small.task,
                            small.stats))
    assert out[0] == out[1]
