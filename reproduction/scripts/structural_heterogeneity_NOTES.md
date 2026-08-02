# Structural-heterogeneity toy — design log and honest reading

Purpose: partial evidence for n7Am Q1/W1 (does CTL extend to structurally
heterogeneous tasks — different output spaces/heads?). Reformulates CTL as
shared-trunk (10->16 ReLU) + per-task head, cascading only the trunk along the
task tree, head trained from scratch per task. Distance proxy: trunk-gradient at
shared init (the paper's gradient proxy, generalised to be output-shape agnostic).
Tasks: T=120, half binary (1 logit), half 4-way (4 logits), 5 clusters, 10 seeds.

## Two design versions (both kept on record — no result-shopping)

**v1, global trunk (one W* for all tasks).** Transfer barely helped; CTL-MST
underperformed Individual at B=500 (0.497 vs 0.537) and edged it at B=2000
(+0.011). Reason: if every task's trunk helps every other equally, the task
geometry carries no signal and there is nothing for the tree to exploit. This is
a property of the design, not of CTL, so v1 does not test the question.

**v2, cluster-specific trunks (W*_c per cluster + within-cluster jitter).** This
is the correct test: transferable structure aligned with the task geometry. The
design change is principled (transfer needs non-uniform shared structure), not a
tweak toward a positive result. Numbers below are v2.

## Result (v2, 10 seeds, mean test accuracy)

| B | method | all | binary | 4-way |
|---|---|---|---|---|
| 500 | Individual | 0.552 | 0.648 | 0.457 |
| 500 | Chain | 0.537 | 0.687 | 0.386 |
| 500 | CTL-MST | 0.507 | 0.642 | 0.372 |
| 2000 | Individual | 0.655 | 0.738 | 0.572 |
| 2000 | Chain | **0.695** | 0.791 | 0.599 |
| 2000 | CTL-MST | 0.663 | 0.763 | 0.564 |

Paired t vs Individual (all-task accuracy):
- B=500: every cascade LOSES (Chain -0.016 p=9e-3, MST -0.046 p=1e-4). Fresh
  per-task heads are starved at 4 steps/task; the 4-way heads suffer most.
- B=2000: Chain +0.040 (p=4e-5), significant on both binary (+0.054, p=9e-5) and
  4-way (+0.027, p=2e-5); Star +0.019 (p=6e-5); CTL-MST +0.009 (p=0.03).

## Honest reading

The shared-trunk cascade transfers across heterogeneous output structure and
significantly beats independent training AT ADEQUATE BUDGET (the Chain / depth-T
form most clearly; bounded-depth MST barely). At tight budget it hurts, because
each task's head must be learned from scratch and the cascade starves the leaves.

This is toy-scale evidence for a REFORMULATION of CTL (shared trunk + per-head),
not for the method as published, which transfers a full parameter vector and
cannot cross output-shape boundaries. Report as a direction, with the
budget-dependence and the toy scale stated. Do NOT present it as closing the scope
gap. As elsewhere in the rebuttal, the deeper cascade captures more of the gain
than the bounded-depth tree.
