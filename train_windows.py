"""Approach 2, step 3: train the final window Random Forest with the best grid-search hyperparameters.

Reads window_features.csv (prepare_windows.py) and cv_results_windows.csv (grid_search_windows.py),
evaluates on the test windows and saves models/window_model.joblib (only the file name of the
third argument is used). The whole-image model is untouched.

Usage: python train_windows.py [window_features.csv] [cv_results_windows.csv] [window_model.joblib]
"""
import ast
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from prepare import PROJECT_DIR, model_file
from prepare_windows import META, PER_IMAGE, SIZE, STEP


def main(features_csv=PROJECT_DIR / "window_features.csv", cv_csv=PROJECT_DIR / "cv_results_windows.csv",
         out_model="window_model.joblib"):
    out_model = model_file(out_model)
    t0 = time.time()
    cv = pd.read_csv(cv_csv)
    best = ast.literal_eval(cv.loc[cv.rank_test_score.idxmin(), "params"])  # rank 1
    df = pd.read_csv(features_csv)
    train, test = df[df.split == "train"], df[df.split == "test"]
    X_train, y_train = train.drop(columns=META), train["label"]
    X_test, y_test = test.drop(columns=META), test["label"]

    params = {"max_features": "sqrt", **best}
    rf = RandomForestClassifier(n_jobs=-1, random_state=42, **params).fit(X_train, y_train)
    proba = rf.predict_proba(X_test)
    pred = rf.classes_[proba.argmax(1)]

    # Whole-image prediction from windows: average the window probabilities of each image.
    per_img = pd.DataFrame(proba).groupby(test["file"].to_numpy()).mean()
    img_pred = rf.classes_[per_img.to_numpy().argmax(1)]
    img_true = test.groupby("file")["label"].first().loc[per_img.index]

    metrics = {"cv_macro_f1": cv.mean_test_score.max(),
               "window_accuracy": accuracy_score(y_test, pred),
               "window_macro_f1": f1_score(y_test, pred, average="macro"),
               "image_vote_accuracy": accuracy_score(img_true, img_pred),
               "n_train": len(X_train), "n_test": len(X_test)}
    line = "=" * 56
    print(f"\n{line}\n  WINDOW RANDOM FOREST ({SIZE}x{SIZE} px, step {STEP})\n{line}")
    print("  Hyperparameters : " + ", ".join(f"{k}={v}" for k, v in params.items()))
    print(f"  Train / test    : {len(X_train)} / {len(X_test)} windows "
          f"({train.file.nunique()} / {test.file.nunique()} images)")
    print(f"  CV macro-F1     : {metrics['cv_macro_f1']:.4f}  (windows, folds grouped by image)")
    print(f"  Window accuracy : {metrics['window_accuracy']:.2%}  (window label = its image's label)")
    print(f"  Window macro-F1 : {metrics['window_macro_f1']:.4f}")
    print(f"  Image accuracy  : {metrics['image_vote_accuracy']:.2%}  (average of each image's windows)")
    print(f"{line}\n\nPer-class results (test windows):\n")
    print(classification_report(y_test, pred, digits=3))

    joblib.dump({"model": rf, "name": Path(out_model).stem, "params": params, "metrics": metrics, "size": SIZE, "step": STEP,
                 "per_image": PER_IMAGE, "features": list(X_train.columns),
                 "trained": time.strftime("%Y-%m-%d %H:%M")}, out_model, compress=3)
    print(f"Saved {out_model} ({time.time() - t0:.0f}s)")
    return rf


if __name__ == "__main__":
    try:
        main(*sys.argv[1:4])
    except FileNotFoundError as e:  # input from an earlier step is missing
        sys.exit(f"{e}\nRun the previous step first (see README).")
