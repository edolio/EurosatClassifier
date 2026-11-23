"""Approach 2, step 1: EuroSAT_MS -> one feature row per 16x16 px window (8 px step).

Each window gets the same 73 features as a whole image (prepare.features_batch) and the label of
the image it was cut from. Images are split into train/test exactly like the whole-image model;
train images give PER_IMAGE random windows each, test images all of theirs.

Usage: python prepare_windows.py [EuroSAT_MS root] [frac] [window_features.csv]
Defaults: data/EuroSAT_MS, all images, window_features.csv inside the project folder.
"""
import sys
import time

import numpy as np
import pandas as pd
import tifffile
from sklearn.model_selection import train_test_split

from prepare import BANDS, DATA_ROOT, PROJECT_DIR, features_batch, list_images

SIZE, STEP = 16, 8
PER_IMAGE = 4  # random training windows per image; all 49 overlap heavily
META = ["file", "label", "split"]  # non-feature columns of window_features.csv


def windows(img, size=SIZE, step=STEP):
    """(H, W, 13) -> (rows, cols, size*size, 13): every size x size window, step px apart."""
    v = np.lib.stride_tricks.sliding_window_view(img, (size, size), axis=(0, 1))[::step, ::step]
    return np.moveaxis(v, 2, -1).reshape(*v.shape[:2], size * size, img.shape[-1])


def load_windows(root=DATA_ROOT, frac=1.0, test_size=0.2, progress=None):
    """progress(done, total) is called after each image, if given."""
    paths = list_images(root, float(frac))
    # Same split call as grid_search.split, so test images are the ones the image model never saw.
    train_p, test_p = train_test_split(paths, test_size=test_size,
                                       stratify=[p.parent.name for p in paths], random_state=42)
    rng = np.random.default_rng(42)
    parts, done = [], 0
    for split, split_paths, k in (("train", train_p, PER_IMAGE), ("test", test_p, None)):
        for path in split_paths:
            done += 1
            if progress:
                progress(done, len(paths))
            img = tifffile.imread(path)
            if img.shape != (64, 64, len(BANDS)):
                print("skipped (unexpected shape):", path.name, img.shape)
                continue
            w = windows(img).reshape(-1, SIZE * SIZE, len(BANDS))
            if k:
                w = w[rng.choice(len(w), k, replace=False)]
            df = features_batch(w)
            df.insert(0, "split", split)
            df.insert(0, "label", path.parent.name)
            df.insert(0, "file", path.name)
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def describe(df):
    print(f"\n{len(df)} windows ({SIZE}x{SIZE} px, step {STEP}) from {df.file.nunique()} images, "
          f"{df.shape[1] - len(META)} features")
    print(f"Train: {PER_IMAGE} random windows per image; test: all windows per image\n")
    print(df.groupby(["label", "split"]).size().unstack().to_string())
    print("\nMissing values:", int(df.drop(columns=META).isna().sum().sum()))


if __name__ == "__main__":
    t0 = time.time()
    try:
        df = load_windows(sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT,
                          sys.argv[2] if len(sys.argv) > 2 else 1.0)
    except FileNotFoundError as e:
        sys.exit(str(e))
    out = sys.argv[3] if len(sys.argv) > 3 else PROJECT_DIR / "window_features.csv"
    df.to_csv(out, index=False)
    describe(df)
    print(f"\nSaved {out} ({time.time() - t0:.0f}s)")
