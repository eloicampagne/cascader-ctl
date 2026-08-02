"""Reproductions of the paper's experiments.

Each module is importable as a function and runnable from the command line::

    from cascader.experiments import main_table
    main_table.run(collections=["Syn-10"], eval_seeds=5)

    $ cascader main-table --collections Syn-10 --eval-seeds 5

``main_table``
    the seven methods on the five collections, at three per-task budgets, under
    a shared or a per-method learning rate;
``allocation_regime``
    ``floor1`` against ``exact``, with the realized-spend check;
``root_fraction``
    the share of the budget reserved for the root, swept for every method;
``rooting``
    four rooting rules at fixed topology;
``tworegime``
    scatter against budget, the boundary where the tree overtakes the chain;
``online``
    tasks arriving one at a time, with no tree and no allocation;
``scaling``
    task count against gain, with the construction cost recorded.
"""

from __future__ import annotations

from . import (allocation_regime, main_table, online, rooting, root_fraction,
               scaling, tworegime)

__all__ = ["main_table", "allocation_regime", "root_fraction", "rooting",
           "tworegime", "online", "scaling"]
