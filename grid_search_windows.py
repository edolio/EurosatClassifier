"""Approach 2, step 2: grid-search window Random Forest hyperparameters on window_features.csv.

Usage: python grid_search_windows.py [window_features.csv]
"""
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from prepare import PROJECT_DIR
from prepare_windows import META

PARAM_GRID = {
    "n_estimators": [100, 200],  # the image model's loss curve flattens around 100 trees
    "max_depth": [None, 40],
    "min_samples_leaf": [1, 2, 4],
}


def run(features_csv=PROJECT_DIR / "window_features.csv", param_grid=PARAM_GRID,
        out_csv=PROJECT_DIR / "cv_results_windows.csv"):
    df = pd.read_csv(features_csv)
    train = df[df.split == "train"]  # test windows stay unseen until train_windows.py
    X, y = train.drop(columns=META), train["label"]
    search = GridSearchCV(
        RandomForestClassifier(max_features="sqrt", random_state=42),
        param_grid,
        # Folds split by image: windows of one image overlap, so they must stay in the same fold.
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=False,  # train_windows.py trains the final model
    )
    search.fit(X, y, groups=train["file"])

    print(f"\nTrain windows: {len(train)} from {train.file.nunique()} images")
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
        run(*sys.argv[1:2])
    except FileNotFoundError as e:  # input from an earlier step is missing
        sys.exit(f"{e}\nRun the previous step first (see README).")
