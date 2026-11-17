"""Table principale lineaire de Syn sous l'allocation SLE proportionnelle pure (regime exact),
pour l'annexe sur l'effet de l'allocation. Meme protocole que syn_suite.part_main."""
import pandas as pd
from syn_suite import TUNE, EVAL, out, syn, sweep, METHODS, etas_for
rows = []
for split, seeds in (("tune", TUNE), ("eval", EVAL)):
    sweep("Syn", seeds, split, rows, model="linear", methods=METHODS, b_per_task=(1, 2, 4),
          etas=etas_for("Syn", "linear"), loader=syn, regime="exact",
          on_seed=lambda c, sp, s, t: print(f"exact {sp} {s} {t:.0f}s", flush=True))
    pd.DataFrame(rows).to_csv(out("main_linear_exact"), index=False)
