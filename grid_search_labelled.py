"""Approach 3, step 2: grid-search the labelled window Random Forest on window_labels.csv.

Input: window_labels.csv (train_labelled.py label: windows of training images, with final labels and
features) and window_features.csv (its training windows give the saved scaler + k-means centroids).
Features: 73 scaled features + distance to each of the 40 centroids. Human labels weigh HUMAN_WEIGHT
times more. The labelled images are split 80/20 by image; the search uses only the training 80%.
Folds are grouped by image, as for the window model, and scored with macro-F1.

Usage: python grid_search_labelled.py [window_labels.csv] [window_features.csv]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from train_labelled import FEATURES_CSV, OUT_CSV, load_labelled
from prepare import PROJECT_DIR

PARAM_GRID = {  # same grid as the window model
    "n_estimators": [100, 200],
    "max_depth": [None, 40],
    "min_samples_leaf": [1, 2, 4],
}
CV_PER_IMAGE = 8  # windows per image used in the search (all human-labelled ones are kept)


def run(labels_csv=OUT_CSV, features_csv=FEATURES_CSV, param_grid=PARAM_GRID,
        out_csv=PROJECT_DIR / "cv_results_labelled.csv"):
    used, X, y, w, _, _ = load_labelled(labels_csv, features_csv)
    train = (used.split == "train").to_numpy()  # test images stay unseen until train_labelled.py
    used, X, y, w = used[train].reset_index(drop=True), X[train], y[train], w[train]
    # shortcut: CV on CV_PER_IMAGE random windows per image (49 overlapping ones are near-duplicates);
    # raise it if the search runs fast enough. train_labelled.py train uses all windows.
    rng = np.random.default_rng(42)
    rank = used.assign(k=rng.random(len(used))).sort_values("k").groupby("file").cumcount().sort_index()
    pick = ((rank < CV_PER_IMAGE) | (used.source == "human")).to_numpy()
    search = GridSearchCV(
        RandomForestClassifier(max_features="sqrt", random_state=42),
        param_grid,
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),  # windows of one image stay together
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=False,  # train_labelled.py train trains the final model
    )
    search.fit(X[pick], y[pick], groups=used.file[pick], sample_weight=w[pick])

    print(f"\nWindows in the search: {pick.sum()} from {used.file.nunique()} training images "
          f"({(used.source[pick] == 'human').sum()} human-labelled)")
    print("Best params:", search.best_params_)
    print(f"Best CV window macro-F1: {search.best_score_:.4f}")
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print("\nAll candidates:\n" + res[["params", "mean_test_score", "std_test_score", "mean_fit_time"]]
          .to_string(index=False))
    res.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    return search


if __name__ == "__main__":
    try:
        run(*sys.argv[1:3])
    except FileNotFoundError as e:  # input from an earlier step is missing
        sys.exit(f"{e}\nRun the previous step first (see README).")
