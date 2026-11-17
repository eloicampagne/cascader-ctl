"""Balayage en profondeur du modele local (mlp_depth_2opt.py), partie synthetique seule,
sur le generateur de Syn (lancer avec CTL_SYN=subspace). Sortie : mlp_depth_syn_raw.csv."""
import pandas as pd
import mlp_depth_2opt as M
from table1_nonconvex import syn_tasks
rows = M.run_dataset("Syn-10", lambda r: syn_tasks(r))
pd.DataFrame(rows).to_csv(M.HERE / "mlp_depth_syn_raw.csv", index=False)
print("=== Done ===", flush=True)
