"""Train the final Random Forest with the best hyperparameters from grid_search.py.

Usage: python train_final.py [features.csv] [cv_results.csv]
Saves the plots to final_results.png and the model to models/model.joblib (only the file name of
the third argument is used: models always go to the models/ folder).
"""
import ast
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, log_loss)

from grid_search import split
from prepare import PROCESSED_DIR, PROJECT_DIR, model_file


def main(features_csv=PROCESSED_DIR / "features.csv", cv_csv=PROJECT_DIR / "cv_results.csv",
         out_png=PROJECT_DIR / "final_results.png",
         out_model="model.joblib"):
    out_model = model_file(out_model)
    # Winning params from the grid search (rank 1).
    cv = pd.read_csv(cv_csv)
    best = ast.literal_eval(cv.loc[cv.rank_test_score.idxmin(), "params"])

    # Same split as grid_search.py, so the test set is still unseen.
    X_train, X_test, y_train, y_test = split(pd.read_csv(features_csv))

    # A forest has no epochs: "training" = adding trees, so track loss as the forest grows.
    n_total = best.pop("n_estimators")
    rf = RandomForestClassifier(max_features="sqrt", warm_start=True, n_jobs=-1,
                                random_state=42, **best)
    steps = sorted(set(range(10, n_total, max(n_total // 25, 1))) | {n_total})
    train_loss, test_loss = [], []
    for n in steps:
        rf.set_params(n_estimators=n).fit(X_train, y_train)
        train_loss.append(log_loss(y_train, rf.predict_proba(X_train), labels=rf.classes_))
        test_loss.append(log_loss(y_test, rf.predict_proba(X_test), labels=rf.classes_))
        print(f"\r  trees {n:>4}/{n_total}  train loss {train_loss[-1]:.4f}  test loss {test_loss[-1]:.4f}",
              end="", flush=True)
    print()

    pred = rf.predict(X_test)
    line = "=" * 56
    print(f"\n{line}\n  FINAL RANDOM FOREST\n{line}")
    params = {"n_estimators": n_total, "max_features": "'sqrt'", **best}
    print("  Hyperparameters : " + ", ".join(f"{k}={v}" for k, v in params.items()))
    print(f"  Train / test    : {len(X_train)} / {len(X_test)} images, {y_train.nunique()} classes")
    print(f"  CV macro-F1     : {cv.mean_test_score.max():.4f}")
    print(f"  Test accuracy   : {accuracy_score(y_test, pred):.2%}")
    print(f"  Test macro-F1   : {f1_score(y_test, pred, average='macro'):.4f}")
    print(f"  Test log loss   : {test_loss[-1]:.4f}")
    print(f"{line}\n\nPer-class results (test set):\n")
    print(classification_report(y_test, pred, digits=3))

    # Model + what the web app shows about it.
    joblib.dump({
        "model": rf,
        "name": Path(out_model).stem,
        "params": {"n_estimators": n_total, "max_features": "sqrt", **best},
        "metrics": {"test_accuracy": accuracy_score(y_test, pred),
                    "test_macro_f1": f1_score(y_test, pred, average="macro"),
                    "cv_macro_f1": cv.mean_test_score.max(),
                    "n_train": len(X_train), "n_test": len(X_test)},
        "features": list(X_train.columns),
        "trained": time.strftime("%Y-%m-%d %H:%M"),
    }, out_model, compress=3)  # compressed: keeps files well under GitHub's 100 MB limit
    print(f"Saved {out_model}")

    # Colors from the dataviz reference palette: categorical slots 1-2, blue ramp for magnitude.
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})

    for losses, color, name in [(train_loss, BLUE, "Train"), (test_loss, ORANGE, "Test")]:
        ax1.plot(steps, losses, color=color, lw=2, label=name)
        ax1.annotate(f"{name} {losses[-1]:.3f}", (steps[-1], losses[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", color=INK, fontsize=9)
    ax1.set(title="Log loss as trees are added", xlabel="Number of trees", ylabel="Log loss")
    ax1.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax1.legend(frameon=False)

    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    ConfusionMatrixDisplay.from_predictions(y_test, pred, normalize="true", values_format=".0%",
                                            cmap=blues, ax=ax2, colorbar=False)
    ax2.set_title("Confusion matrix (share of each true class)")
    ax2.tick_params(axis="x", labelrotation=45)
    plt.setp(ax2.get_xticklabels(), ha="right", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    plt.show()
    return rf


if __name__ == "__main__":
    try:
        main(*sys.argv[1:3])
    except FileNotFoundError as e:  # input from an earlier step is missing
        sys.exit(f"{e}\nRun the previous step first (see README).")
