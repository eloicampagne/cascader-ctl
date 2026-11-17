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
| `stronger_baselines.py` | `cascader.methods`, Star with an exact root or momentum, Individual with momentum, water-filling allocation |
| `syn/syn_suite.py`, `syn/main_ext.py`, `syn/main_exact.py` | `cascader main-table` on `load("Syn")` |
| `syn/mst_depth.py` | `cascader.trees`, depth of CTL-MST up to $T=10000$ |

The rest have no packaged equivalent. They cover the pretrained-feature and
foundation-model studies, the M5 hierarchy, structural heterogeneity, the
parallelism measurements, the alternative distance audits and several
exploratory studies.

## Scripts added for the current version

Synthetic dataset. The scripts here generate Syn-10 by default; set
`CTL_SYN=subspace` to use Syn, the subspace generator of the paper ($d=40$, $r=3$).
`syn/` holds the Syn suite, which imports the package from `../../src`:
`syn_suite.py` runs the main tables, the closed forms, the pooled model, the
tree--chain grid and the mechanism data, `syn_report.py` summarizes them and
`emit_syn_rows.py` writes the table rows into the paper source given by `PAPER_TEX`.

| script | appendix |
|---|---|
| `boundary_phase.py`, `boundary_phase_fig.py` | measured boundary of the CTL-vs-direct condition |
| `syn/precond_control.py`, `syn/precond_report.py`, `syn/emit_precond_tables.py` | preconditioned refinements and task conditioning |
| `mechanism_individual.py`, `mechanism_edge_vs_depth_2opt.py`, `mechanism_logratio.py` | parent distance against depth |
| `fig_depth_diagnostics.py`, `mlp_depth_syn.py`, `mlp_depth_syn_ind.py`, `mlp_depth_2opt.py` | depth diagnostics |
| `tworegime_configs.py`, `tworegime_final.py`, `tworegime_zahn_local.py`, `zahn_local.py` | tree--chain grid, local and global cut criteria |
| `proxy_ablation_floor1.py`, `emit_proxy_ablation_table.py` | distance proxies |
| `fig_robustness_sweep.py`, `fig_robustness_plot.py` | robustness to the generator |
| `root_tally.py` | sensitivity to root selection |
| `twoopt.py` | 2-opt exchanges imported by `table1_nonconvex.py` |

Figures go to `FIG_DIR` (default `figures/`). `syn/precond_control.py` reads the
image datasets from `CASCADER_DATA_DIR`.

## One caveat

These scripts read `CTL_ALLOC` for the allocation regime; the package reads
`CASCADER_ALLOC`. Both default to `floor1`, which is the regime the reported
tables use. Output written before the allocation fix used a version of
`_alloc_from_weights` that clamped shares to one after apportioning and therefore
overspent the budget -- any CSV predating that fix is not comparable with the
current ones.
