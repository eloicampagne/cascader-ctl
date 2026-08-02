"""Shared fixtures.

The collections here are deliberately small. Correctness of the allocation, of
the constructions and of the cascade semantics does not depend on scale, and a
suite that runs in seconds gets run.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascader.config import get_config
from cascader.data.synthetic import syn_collection


@pytest.fixture(scope="session")
def small():
    """A 25-task Syn-10 draw, clustered, three-dimensional."""
    return syn_collection(seed=0, T=25, d=3, n_clusters=4)


@pytest.fixture(scope="session")
def tiny():
    """An 8-task draw, small enough to enumerate spanning trees by hand."""
    return syn_collection(seed=1, T=8, d=2, n_clusters=2)


@pytest.fixture(scope="session")
def cfg():
    return get_config("Syn-10")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
