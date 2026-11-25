"""EuroSAT_MS -> one feature row per image, for a Random Forest.

Usage: python prepare.py [EuroSAT_MS root, one subfolder per class] [out.csv]
Defaults: the dataset folder saved as default in the web app (else data/EuroSAT_MS) and
ProcessedData4Train/features.csv in the project folder, from any working directory.
"""
import json
import random
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Band order inside EuroSAT_MS GeoTIFFs (B8A is stored last).
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B09", "B10", "B11", "B12", "B8A"]
B = {name: i for i, name in enumerate(BANDS)}

# Default locations are relative to this file, so the project works from any folder it is cloned to
# and from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
# Personal settings written by the web app ("Save as default" on the training page); not committed.
CONFIG_FILE = PROJECT_DIR / "config.toml"


def data_root():
    """Default dataset folder: the one saved from the web app, else data/EuroSAT_MS."""
    default = PROJECT_DIR / "data" / "EuroSAT_MS"
    try:
        value = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("data_dir")
    except FileNotFoundError:
        return default
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        print(f"Ignoring unreadable {CONFIG_FILE} ({e}); using {default}.")
        return default
    if not isinstance(value, str) or not value.strip():
        return default
    return PROJECT_DIR / Path(value).expanduser()  # relative to the project folder; absolute stays absolute


def save_data_root(folder):
    """Make folder the default dataset folder (stored in config.toml)."""
    # as_posix + JSON quoting gives a valid TOML string for any path, backslashes and quotes included
    CONFIG_FILE.write_text(f"data_dir = {json.dumps(Path(folder).resolve().as_posix())}\n", encoding="utf-8")


DATA_ROOT = data_root()  # read once per script run; the web app calls data_root() on every request
MODELS_DIR = PROJECT_DIR / "models"  # every trained model is saved here
PROCESSED_DIR = PROJECT_DIR / "ProcessedData4Train"  # training data kept in git: features and labels


def model_file(name):
    """models/<name>.joblib for a model name or file name; any folder part is ignored."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR / (Path(name).stem + ".joblib")


def nd(a, b):
    """Normalized difference (a-b)/(a+b), 0 where both are 0."""
    s = a + b
    return np.divide(a - b, s, out=np.zeros_like(s), where=s != 0)


def features_batch(x):
    """x: (n, pixels, 13) -> DataFrame with one feature row per image or window."""
    x = x.astype(np.float32)
    f = {}
    for name, i in B.items():
        b = x[:, :, i]
        p10, p50, p90 = np.percentile(b, [10, 50, 90], axis=1)
        f.update({f"{name}_mean": b.mean(1), f"{name}_std": b.std(1),
                  f"{name}_p10": p10, f"{name}_p50": p50, f"{name}_p90": p90})
    idx = {
        "NDVI": nd(x[:, :, B["B08"]], x[:, :, B["B04"]]),  # vegetation
        "NDWI": nd(x[:, :, B["B03"]], x[:, :, B["B08"]]),  # water
        "NDBI": nd(x[:, :, B["B11"]], x[:, :, B["B08"]]),  # built-up
        "NDMI": nd(x[:, :, B["B08"]], x[:, :, B["B11"]]),  # moisture
    }
    for name, v in idx.items():
        f[f"{name}_mean"], f[f"{name}_std"] = v.mean(1), v.std(1)
    return pd.DataFrame(f)


def features(img):
    """img: (H, W, 13) -> dict of whole-image features."""
    return features_batch(img.reshape(1, -1, img.shape[-1])).iloc[0].to_dict()


def find_root(root=DATA_ROOT):
    """The folder holding <class>/*.tif; also accepts the extra EuroSAT_MS/ level some unzip tools add."""
    root = Path(root)
    for candidate in (root, root / "EuroSAT_MS"):
        if candidate.is_dir() and any(candidate.glob("*/*.tif")):
            return candidate
    raise FileNotFoundError(f"No EuroSAT_MS images (<class>/*.tif) found in {root.resolve()}. "
                            "Download EuroSAT_MS.zip and unzip it there (see README).")


def list_images(root=DATA_ROOT, frac=1.0):
    """frac < 1 keeps that share of images in every class (seeded, for quick tests)."""
    paths, rng = [], random.Random(42)
    for cls in sorted(p for p in find_root(root).iterdir() if p.is_dir()):
        files = sorted(cls.glob("*.tif"))
        paths += files if frac >= 1 else rng.sample(files, round(len(files) * frac))
    return paths


def load(root=DATA_ROOT, frac=1.0, progress=None):
    """progress(done, total) is called after each image, if given."""
    rows, bad = [], []
    paths = list_images(root, frac)
    for n, path in enumerate(paths, 1):
        if progress:
            progress(n, len(paths))
        img = tifffile.imread(path)
        if img.shape != (64, 64, 13) or img.dtype != np.uint16:
            bad.append((path.name, img.shape, img.dtype))
            continue
        rows.append({"file": path.name, "label": path.parent.name, **features(img)})
    for b in bad:
        print("skipped (unexpected shape/dtype):", *b)
    return pd.DataFrame(rows)


def describe(df):
    feats = df.drop(columns=["file", "label"])
    print(f"\n{len(df)} images, {feats.shape[1]} features, {df.label.nunique()} classes")
    print("\nImages per class:\n" + df.label.value_counts().to_string())
    print("\nMissing values:", int(feats.isna().sum().sum()))
    print("\nPer-class mean of band means and indices:")
    cols = [f"{b}_mean" for b in BANDS] + ["NDVI_mean", "NDWI_mean", "NDBI_mean", "NDMI_mean"]
    print(df.groupby("label")[cols].mean().round(2).T.to_string())


if __name__ == "__main__":
    try:
        df = load(sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT)
    except FileNotFoundError as e:
        sys.exit(str(e))
    out = sys.argv[2] if len(sys.argv) > 2 else PROCESSED_DIR / "features.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    describe(df)
    print(f"\nSaved {out}")
