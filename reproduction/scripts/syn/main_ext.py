"""Extension de la grille de pas de Syn au-dela de 0.5 (0.7, 1, 2), toutes methodes,
meme protocole que syn_suite.part_main. Les baselines choisissaient 0.5, la borne.
    python3.12 main_ext.py linear|mlp"""
import sys
import pandas as pd
from syn_suite import TUNE, EVAL, out, syn, sweep, METHODS
model = sys.argv[1]
rows = []
for split, seeds in (("tune", TUNE), ("eval", EVAL)):
    sweep("Syn", seeds, split, rows, model=model, methods=METHODS + ["kNN-BFS"], b_per_task=(1, 2, 4),
          etas=(0.7, 1.0, 2.0), loader=syn,
          on_seed=lambda c, sp, s, t: print(f"ext {model} {sp} {s} {t:.0f}s", flush=True))
    pd.DataFrame(rows).to_csv(out(f"main_{model}_ext"), index=False)
