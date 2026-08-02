"""Task collections.

:func:`load` is the single entry point used by every packaged experiment::

    from cascader.data import load
    tasks = load("Syn-10", seed=0)

Availability differs by collection. Syn-10 is generated. Fashion-MNIST and
CIFAR-10 download themselves. WEAVE-UK falls back to a bundled 50-substation
sample when the full CSVs are absent, and CelebA raises
:class:`MissingDataError` with instructions -- see :mod:`cascader.data._paths`.
"""

from __future__ import annotations

from typing import Any

from ..tasks import TaskCollection
from ._paths import MissingDataError, data_dir
from .images import celeba_collection, image_collection
from .synthetic import syn_collection
from .transforms import denormalize_y, normalize_xy, standardize
from .weave import weave_collection

__all__ = [
    "load", "COLLECTIONS", "TaskCollection", "MissingDataError", "data_dir",
    "syn_collection", "weave_collection", "image_collection", "celeba_collection",
    "normalize_xy", "denormalize_y", "standardize",
]

COLLECTIONS = ("Syn-10", "WEAVE-UK", "FMNIST", "CIFAR", "CelebA")


def load(name: str, seed: int = 0, **kwargs: Any) -> TaskCollection:
    """Load a collection by the name it carries in the paper's tables.

    ``seed`` redraws the tasks on the generated and sampled collections. It has
    no effect on WEAVE-UK, whose panel of substations is fixed; repetitions
    there vary only the initialization.
    """
    if name == "Syn-10":
        return syn_collection(seed=seed, **kwargs)
    if name == "WEAVE-UK":
        return weave_collection(**kwargs)
    if name in ("FMNIST", "CIFAR"):
        return image_collection(name, seed=seed, **kwargs)
    if name == "CelebA":
        return celeba_collection(seed=seed, **kwargs)
    raise KeyError(f"unknown collection {name!r}; choose from {COLLECTIONS}")
