"""Pendant Individual de mlp_depth_syn.py (meme calibration du pas par profondeur), sur Syn.
Lancer avec CTL_SYN=subspace. Sortie : mlp_depth_syn_ind_raw.csv."""
import pandas as pd
import mlp_depth as M
from table1_nonconvex import syn_tasks
M.DEPTH_METHODS = ["Individual"]
rows = M.run_dataset("Syn-10", lambda r: syn_tasks(r))
pd.DataFrame(rows).to_csv(M.HERE / "mlp_depth_syn_ind_raw.csv", index=False)
print("=== Done ===", flush=True)
