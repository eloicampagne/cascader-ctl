"""One-time CelebA raw-image cache, 64x64 uint8, in the exact order used by the
frozen-feature cache.

celeba_extract.py stored ResNet-18 features. To train a convolutional local
model end to end we need the images themselves, in the same order, so that the
attribute labels already cached in celeba_cache/y_*.npy stay valid. This script
replays the same metadata load and the same transform, saves the images as
uint8, and asserts that the labels it recovers match the cached ones exactly.

Run once: python3.12 celeba_raw_extract.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
CTL_DIR = HERE.parent.parent.parent / "CTL"
sys.path.insert(0, str(CTL_DIR))

OUT = HERE / "celeba_cache"
IMG_DIR = CTL_DIR / "data" / "celeba" / "celeba" / "img_align_celeba"
IMAGE_SIZE = 64
MAX_TRAIN, MAX_TEST = 12000, 4000
N_TRAIN, N_VALID = 162_770, 19_867

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]


def load_meta():
    """Same shard read as data.load_celeba, skipping corrupted shards."""
    import io
    import urllib.request
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files, hf_hub_url

    repo_id = "huggan/CelebA-faces-with-attributes"
    cols = ["image_id"] + ATTR_NAMES
    shards = sorted(f for f in list_repo_files(repo_id, repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet"))
    dfs, skipped = [], []
    for i, fname in enumerate(shards):
        print(f"  shard {i+1}/{len(shards)}", end="\r", flush=True)
        try:
            with urllib.request.urlopen(hf_hub_url(repo_id, filename=fname,
                                                   repo_type="dataset")) as r:
                dfs.append(pq.read_table(io.BytesIO(r.read()), columns=cols).to_pandas())
        except Exception:
            skipped.append(fname)
    print(f"\n  {len(skipped)} shards skipped", flush=True)
    return pd.concat(dfs, ignore_index=True).sort_values("image_id").reset_index(drop=True)


def load_split(meta_df):
    import torchvision.transforms as tvt
    from PIL import Image
    tf = tvt.Compose([tvt.Resize(IMAGE_SIZE), tvt.CenterCrop(IMAGE_SIZE)])
    imgs = np.empty((len(meta_df), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, img_id in enumerate(meta_df["image_id"].tolist()):
        if i % 1000 == 0:
            print(f"  {i}/{len(meta_df)}", end="\r", flush=True)
        img = tf(Image.open(os.path.join(IMG_DIR, img_id)).convert("RGB"))
        imgs[i] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    y = meta_df[ATTR_NAMES].values.astype(np.int8)
    if y.min() < 0:
        y = ((y + 1) // 2).astype(np.int8)
    print()
    return imgs, y


if __name__ == "__main__":
    t0 = time.time()
    meta = load_meta()
    print(f"metadata for {len(meta)} images", flush=True)
    tr = meta.iloc[:N_TRAIN].iloc[:MAX_TRAIN]
    te = meta.iloc[N_TRAIN + N_VALID:].reset_index(drop=True).iloc[:MAX_TEST]
    Xtr, ytr = load_split(tr)
    Xte, yte = load_split(te)

    ytr_ref = np.load(OUT / "y_train.npy")
    yte_ref = np.load(OUT / "y_test.npy")
    assert ytr.shape == ytr_ref.shape and np.array_equal(ytr, ytr_ref), "train labels differ"
    assert yte.shape == yte_ref.shape and np.array_equal(yte, yte_ref), "test labels differ"
    print("labels match the frozen-feature cache exactly", flush=True)

    np.save(OUT / "Ximg_train.npy", Xtr)
    np.save(OUT / "Ximg_test.npy", Xte)
    print(f"saved {Xtr.shape} / {Xte.shape} in {time.time()-t0:.0f}s", flush=True)
