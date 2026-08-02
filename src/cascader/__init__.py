"""CASCADER: cascaded transfer learning for many tasks under a global budget.

Given ``T`` related tasks and a total of ``B`` gradient steps to spend across all
of them, CTL arranges the tasks in a rooted tree, trains the root, and lets each
task inherit its parent's parameters before spending its own share of the
budget. The cost is one pass over the collection; the gain is that no task
starts from scratch.

Three components decide what a cascade does, and each is a module here:

``distances``
    how far apart two tasks are, from data alone;
``trees`` and ``rooting``
    which task inherits from which, and where the cascade starts;
``allocation``
    how the global budget is split, always summing to exactly ``B``.

Quick start::

    from cascader import data, run_method, eval_all, model_factory, get_config

    tasks = data.load("Syn-10", seed=0)
    cfg = get_config("Syn-10")
    models = run_method("CTL-MST", tasks, B=2 * tasks.T, eta=cfg.eta, cfg=cfg,
                        factory=model_factory("linear", tasks.d))
    print(eval_all(models, tasks.X_test, tasks.y_test, tasks.task, tasks.stats))

Reproductions of the paper's tables live in :mod:`cascader.experiments` and are
reachable from the command line as ``cascader <experiment>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import data, distances, rooting, trees
from .allocation import (individual_alloc, sle_alloc, sle_weights, uniform_alloc)
from .cascade import cascade_train, eval_all, eval_task, independent_train, train_steps
from .config import CONFIGS, DatasetConfig, get_config
from .distances import distance_matrix
from .methods import METHODS, allocate, build_structure, run_method
from .models import make_model, model_factory
from .rooting import select_root
from .tasks import TaskCollection
from .trees import RootedTree, root_tree

__all__ = [
    "__version__",
    "data", "distances", "rooting", "trees",
    "TaskCollection", "RootedTree", "root_tree",
    "distance_matrix", "select_root",
    "sle_alloc", "sle_weights", "uniform_alloc", "individual_alloc",
    "cascade_train", "independent_train", "train_steps", "eval_task", "eval_all",
    "METHODS", "run_method", "build_structure", "allocate",
    "make_model", "model_factory",
    "DatasetConfig", "CONFIGS", "get_config",
]
