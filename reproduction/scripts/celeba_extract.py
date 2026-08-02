"""One-time CelebA ResNet-18 feature extraction, cached to .npy.
Images are read locally from data/celeba/celeba/img_align_celeba/; the 40 binary
attributes come from the HuggingFace parquet shards. Run once, then
table1_nonconvex.py reads the cached arrays."""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent.resolve()
CTL_DIR = HERE.parent.parent.parent / "CTL"
sys.path.insert(0, str(CTL_DIR))
from data import load_celeba

OUT = HERE / "celeba_cache"
OUT.mkdir(exist_ok=True)

t0 = time.time()
Xtr, ytr, Xte, yte, names = load_celeba(
    data_dir=str(CTL_DIR / "data" / "celeba"), download=False,
    image_size=64, max_train=12000, max_test=4000,
    backbone="resnet18", device=torch.device("cpu"))
print(f"extracted in {time.time()-t0:.0f}s: train {Xtr.shape} test {Xte.shape}", flush=True)
np.save(OUT / "X_train.npy", Xtr.astype(np.float32))
np.save(OUT / "y_train.npy", ytr.astype(np.int8))
np.save(OUT / "X_test.npy", Xte.astype(np.float32))
np.save(OUT / "y_test.npy", yte.astype(np.int8))
np.save(OUT / "attr_names.npy", np.array(names))
print("cached to", OUT, flush=True)
