"""Quick end-to-end test of both approaches (prepare -> grid search -> final training) on ~30% of EuroSAT_MS.

Usage: python test_pipeline.py [EuroSAT_MS root]
Writes data and results to test_out/ and the models to models/test_model.joblib and
models/test_window_model.joblib, so the real ones (features.csv, model.joblib, ...) are untouched.
"""
import sys
import time
from pathlib import Path

import grid_search
import prepare
import train_final
import grid_search_windows
import prepare_windows
import train_windows

ROOT = sys.argv[1] if len(sys.argv) > 1 else prepare.DATA_ROOT
FRAC = 0.3
OUT = prepare.PROJECT_DIR / "test_out"

if __name__ == "__main__":  # needed on Windows: grid search starts worker processes
    try:
        prepare.find_root(ROOT)
    except FileNotFoundError as e:
        sys.exit(str(e))
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"[1/6] prepare: {FRAC:.0%} of each class from {ROOT}")
    df = prepare.load(ROOT, frac=FRAC)
    root = prepare.find_root(ROOT)
    n_classes = sum(p.is_dir() for p in root.iterdir())
    n_files = len(list(root.glob("*/*.tif")))
    assert df.label.nunique() == n_classes, f"expected {n_classes} classes, got {df.label.nunique()}"
    assert abs(len(df) - FRAC * n_files) <= n_classes, f"expected ~{FRAC * n_files:.0f} rows, got {len(df)}"
    assert not df.isna().any().any(), "features contain NaN"
    df.to_csv(OUT / "features.csv", index=False)
    prepare.describe(df)
    print(f"      done in {time.time() - t0:.0f}s")

    print("\n[2/6] grid_search: n_estimators only")
    grid_search.run(OUT / "features.csv", {"n_estimators": [50, 100, 200]}, OUT / "cv_results.csv")
    print(f"      done in {time.time() - t0:.0f}s")

    print("\n[3/6] train_final")
    train_final.main(OUT / "features.csv", OUT / "cv_results.csv", OUT / "final_results.png",
                     "test_model")

    print("\n[4/6] prepare_windows")
    wdf = prepare_windows.load_windows(ROOT, FRAC)
    assert wdf.file.nunique() == len(df), "window table must cover the same images"
    assert not wdf.isna().any().any(), "window features contain NaN"
    wdf.to_csv(OUT / "window_features.csv", index=False)
    prepare_windows.describe(wdf)
    print(f"      done in {time.time() - t0:.0f}s")

    print("\n[5/6] grid_search_windows: n_estimators only")
    grid_search_windows.run(OUT / "window_features.csv", {"n_estimators": [50, 100]},
                            OUT / "cv_results_windows.csv")
    print(f"      done in {time.time() - t0:.0f}s")

    print("\n[6/6] train_windows")
    train_windows.main(OUT / "window_features.csv", OUT / "cv_results_windows.csv",
                       "test_window_model")
    print(f"\nAll steps passed in {time.time() - t0:.0f}s. Outputs in {OUT.resolve()}")
