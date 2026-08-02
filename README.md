# CASCADER

Given `T` related tasks and a total of `B` gradient steps to spend across all of
them, CTL arranges the tasks in a rooted tree, trains the root, and lets each
task inherit its parent's parameters before spending its own share. Nothing is
trained twice and nothing starts from scratch except the root.

## Install

```bash
pip install cascader-ctl
```

## Quick start

```python
from cascader import data, run_method, eval_all, model_factory, get_config

tasks = data.load("Syn-10", seed=0)          # 200 clustered regression tasks
cfg = get_config("Syn-10")

models = run_method("CTL-MST", tasks, B=2 * tasks.T, eta=cfg.eta, cfg=cfg,
                    factory=model_factory("linear", tasks.d))

print(eval_all(models, tasks.X_test, tasks.y_test, tasks.task, tasks.stats))
```

Two steps per task, one model per task, and the whole run costs `2T` gradient
steps -- the same as training each task alone for two steps, which is the
`Individual` baseline.

## Your own tasks

A collection is a set of tasks over a shared feature space. Build one directly:

```python
import numpy as np
from cascader import TaskCollection, distance_matrix, rooting

nodes = list(X_train)                                    # any hashable ids
D = distance_matrix("gradient", nodes, X_train, y_train)
tasks = TaskCollection(nodes=nodes, X_train=X_train, y_train=y_train,
                       X_test=X_test, y_test=y_test, D=D,
                       root=rooting.medoid_root(nodes, X_train),
                       task="regression")
```

CTL transports a parameter vector, so every task must share one parameter space.
Heterogeneous label spaces or per-task heads fall outside the method as
published.

## The three choices

| module | question | default |
|---|---|---|
| `cascader.distances` | how far apart are two tasks? | `gradient` |
| `cascader.trees`, `cascader.rooting` | who inherits from whom, starting where? | MST, medoid root |
| `cascader.allocation` | how is `B` split? | SLE, `floor1` regime |

Every allocator returns integers summing to **exactly** `B`. That invariant is
the contract behind every reported comparison, and the test suite checks it on
every allocator, regime and budget.

### Methods

| method | structure | allocation |
|---|---|---|
| `Individual` | none | `B // T` each |
| `Star` | depth-1 hub | uniform |
| `RandTree` | random recursive tree | uniform |
| `Chain` | greedy nearest-neighbour path | SLE |
| `CTL-kNN` | BFS tree over the k-NN graph | SLE |
| `CTL-MST` | minimum spanning tree | SLE |
| `CTL-MSTc` | budget-aware greedy tree | SLE |

SLE weights task `v` by `|s_v| log(1 + d(pa(v), v))`: subtree size times log edge
length, so budget goes where many tasks inherit and where the parent is far.

### Allocation regimes

`floor1` (default) refines every task at least once and apportions the surplus.
`exact` apportions `B` over the weights and lets a light subtree round to zero
steps, in which case that task is an exact copy of its parent. Both spend exactly
`B`. Set `CASCADER_ALLOC=exact` or pass `regime=` to compare them; the packaged
`allocation-regime` experiment does exactly that.

## Reproducing the paper

```bash
cascader --list
cascader main-table --collections Syn-10 --eval-seeds 5 --tune-seeds 2
```

| experiment | what it answers |
|---|---|
| `main-table` | the seven methods on five collections at three per-task budgets |
| `allocation-regime` | `floor1` against `exact`, with the realized-spend check |
| `root-fraction` | how much of the budget the root should get, swept per method |
| `rooting` | four rooting rules at fixed topology |
| `tworegime` | scatter against budget: where the tree overtakes the chain |
| `online` | tasks arriving one at a time, no tree and no allocation |
| `scaling` | task count against gain, with construction cost |

Each writes `<stem>_raw.csv` and `<stem>_summary.csv` into `--out`. The raw rows
carry every cell, so a summary can be recomputed without re-running anything.

Defaults are the paper's and they are not small -- five collections, twenty
evaluation seeds, ten learning rates. Start with one collection and three seeds.

Two conventions matter for reading the output. Budgets are per task: `b = B / T`
is comparable across collections and absolute budgets are not. Learning rates are
selected on tuning seeds disjoint from the evaluation seeds, never on the split a
number is reported from.

Experiments outside the packaged set live in [`reproduction/`](reproduction/) as
the scripts that produced them, unmodified and unsupported.

## Data

| collection | availability |
|---|---|
| `Syn-10` | generated, nothing to download |
| `FMNIST`, `CIFAR` | downloaded on first use |
| `WEAVE-UK` | 50-substation sample bundled; the full 500 are not redistributed |
| `CelebA` | not distributed; extract ResNet-18 features yourself |

Point `CASCADER_DATA_DIR` at a directory holding `train_weave_500.csv`,
`test_weave_500.csv`, `celeba_features/`, and the image caches. Loaders raise
`MissingDataError` with the exact path and instructions when something is absent.
WEAVE falls back to the bundled sample automatically, so nothing here blocks a
first run -- but sample results are not the paper's numbers.

## Tests

```bash
pip install "cascader-ctl[dev]"
pytest
```

Beyond the usual unit tests, two suites are worth knowing about.

`tests/test_paper_conformance.py` turns the paper's claims into assertions: the
SLE formula, `phi_tau` as softplus centered at the median pairwise distance, the
MSTc score `-psi(u) phi_tau(d(u,v)) + lambda G(v)` with `psi` the subtree size,
root invariance of the MST edge set, chain length at least the MST's, the
published hyperparameters, and the budget accounting. A default that drifts from
the paper fails here.

`tests/test_reference_equivalence.py` pins the assembled pipeline against scores
produced by the scripts that generated the reported tables, to floating-point
precision, for all seven methods at three budgets and both model classes. Unit
tests say each piece obeys its definition; this one says the whole is the same
whole, including tie-breaking inside greedy constructions and which task receives
an apportionment remainder.

## Citation

```bibtex
@article{campagne2026cascaded,
  author = {Campagne, Eloi and Amara-Ouali, Yvenn and Goude, Yannig and
            Mougeot, Mathilde and Kalogeratos, Argyris},
  title  = {Cascaded Transfer: Learning Many Tasks under Budget Constraints},
  year   = {2026}
}
```

## License

GPL-3.0. See [LICENSE](LICENSE).
