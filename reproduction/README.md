# Reproduction scripts

The scripts under `scripts/` are the ones that produced the paper's tables and
appendices. They are the historical record, not part of the
distributed package: they are not installed, not imported by `cascader`, and not
covered by the test suite.

Use them when you need a number that the packaged experiments do not produce, or
when you want to see exactly what was run. For anything else prefer
`cascader <experiment>`, which runs the same method through the tested code path.

## Running them

They expect the original repository layout: a sibling `CTL/` directory holding
`graph.py`, `data.py` and `data/`, importable from the script's own directory.

```bash
cd scripts
PYTHONPATH=../../CTL CTL_ALLOC=floor1 python3 table1_nonconvex.py --exp all
```

`table1_nonconvex.py` is the common core the others import: configurations, task
loaders, tree builders, the cascade loop and the allocators.

## What corresponds to what

| script | packaged equivalent |
|---|---|
| `table1_nonconvex.py` | `cascader.methods`, `cascader.data` |
| `table2_permethod_lowbudget.py` | `cascader main-table --eta-mode per-method` |
| `star_alloc_test.py`, `star_root_fraction.py`, `cascade_root_fraction.py` | `cascader root-fraction` |
| `depth_sweep.py`, `root_depthcentral.py` | `cascader rooting` |
| `scatter_sweep.py` | `cascader tworegime` |
| `online_arrival.py` | `cascader online` |
| `scale_tasks.py`, `large_t_chain_vs_mst.py` | `cascader scaling` |
| `weave_tune_on_validation.py` | `cascader.data.weave_collection(val_size=...)` |
| `exact_root.py` | `cascader.methods`, root initialized at its least-squares solution |

The rest have no packaged equivalent. They cover the pretrained-feature and
foundation-model studies, the M5 hierarchy, structural heterogeneity, the
parallelism measurements, the alternative distance audits and several
exploratory studies.

## One caveat

These scripts read `CTL_ALLOC` for the allocation regime; the package reads
`CASCADER_ALLOC`. Both default to `floor1`, which is the regime the reported
tables use. Output written before the allocation fix used a version of
`_alloc_from_weights` that clamped shares to one after apportioning and therefore
overspent the budget -- any CSV predating that fix is not comparable with the
current ones.
