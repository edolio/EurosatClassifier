"""Grid-search Random Forest hyperparameters on features.csv (from prepare.py).

Usage: python grid_search.py [features.csv]
"""
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from prepare import PROJECT_DIR

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [None, 40],
    "min_samples_leaf": [1, 2, 4],
}


def split(df, test_size=0.2):
    """Hold out a test set the grid search never sees. train_final.py reuses this split."""
    X, y = df.drop(columns=["file", "label"]), df["label"]
    if y.nunique() < 2:
        raise ValueError(f"Need at least 2 classes, found: {list(y.unique())}")
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


def run(features_csv=PROJECT_DIR / "features.csv", param_grid=PARAM_GRID, out_csv=PROJECT_DIR / "cv_results.csv"):
    X_train, X_test, y_train, y_test = split(pd.read_csv(features_csv))
    search = GridSearchCV(
        RandomForestClassifier(max_features="sqrt", random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",  # classes are not perfectly balanced
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print("\nBest params:", search.best_params_)
    print(f"Best CV macro-F1: {search.best_score_:.4f}")
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(out_csv, index=False)

    pred = search.best_estimator_.predict(X_test)
    print("\nTest set:\n" + classification_report(y_test, pred, digits=4))
    labels = sorted(y_test.unique())
    print("Confusion matrix (rows = true):")
    print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels).to_string())
    return search


if __name__ == "__main__":
    try:
        run(*sys.argv[1:2])
    except FileNotFoundError as e:  # input from an earlier step is missing
        sys.exit(f"{e}\nRun the previous step first (see README).")
