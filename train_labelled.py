"""Approach 3: labelled window model, built and trained in two modes (one file).

label: builds the labelled window dataset. A scaler and k-means (40 centroids) are fitted once on
the training windows of window_features.csv (prepare_windows.py), so the same centroids serve every
round and later predictions. Each round takes a fresh batch of training images per class (all 49
windows each) plus every image a human already labelled a window of, then:
1. scales the 73 features and adds each window's distance to every saved centroid;
2. trains the forest on the image labels (human labels override them and weigh HUMAN_WEIGHT times
   more) and scores each window out-of-bag, i.e. only by trees that did not train on it. It
   relabels windows it is confident belong to another class, drops windows it is unsure about,
   and retrains. Relabels are recomputed every round from the image labels, never chained;
3. shows you the most critical windows (one per cluster): the window outlined on its image and
   enlarged. Type a class number in the terminal, m (mixed / unsure: the window is dropped), s (skip)
   or q (stop this round). Each answer is appended to human_labels.csv at once, so nothing is lost on quit.
   Outputs: human_labels.csv (your labels, read again on every run), window_labels.csv (the dataset:
   final label, source and the 73 features of every window seen), window_labels.png (sample windows
   per label, framed by where the label came from) and printed statistics.

train: after grid_search_labelled.py, trains the final forest with the best params, like
train_final.py but every number is per window: loss curve as trees are added, confusion matrix,
accuracy, macro-F1, log loss and per-class results. Train and test windows come from an 80/20 split
by image; test windows keep their cleaned labels and come from images the forest never trains on.
Saves labelled_results.png and models/labelled_window_model.joblib (the web app's window model).

Usage: python train_labelled.py label [images per class, or all] [rounds] [window_features.csv]
       python train_labelled.py train [window_labels.csv] [cv_results_labelled.csv] [window_features.csv]
"""
import ast
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prepare import BANDS, PROJECT_DIR, data_root, features_batch, find_root, list_images, model_file
from prepare_windows import META, SIZE, STEP, windows

ROUNDS, PER_CLASS, N_CLUSTERS = 3, 100, 40
RELABEL_P, DROP_P = 0.8, 0.4  # forest confidence to relabel / below which a window is dropped
HUMAN_WEIGHT, PER_ROUND = 5, 30  # human label weight in training; windows shown per round
MIXED = "Mixed"
HUMAN_CSV, OUT_CSV = PROJECT_DIR / "human_labels.csv", PROJECT_DIR / "window_labels.csv"
FEATURES_CSV = PROJECT_DIR / "window_features.csv"  # from prepare_windows.py: k-means is fitted on its training windows
KEY = ["file", "row", "col"]  # one window
LOSS_SAMPLE = 20000  # windows per side used to draw the loss curve (scoring all at 25 sizes is slow)
MAX_MB = 90  # saved model size limit, below GitHub's 100 MB per file
BYTES_PER_NODE = 16.5  # lzma-compressed size per tree node, measured on a saved model of 3 million nodes
HUMAN_COLS = ["file", "image_label", "row", "col", "label", "round", "time"]


def auto_labels(labels, is_human, pred, conf):
    """Final label and its source for every window (step 2); human labels are never changed."""
    auto = ~is_human
    relabel = auto & (conf >= RELABEL_P) & (pred != labels)
    drop = (auto & (conf < DROP_P)) | (labels == MIXED)
    final = np.where(relabel, pred, labels).astype(object)
    final[drop] = None
    source = np.select([drop, is_human, relabel], ["dropped", "human", "auto"], "image")
    return final, source


# Self-check: confident other class -> relabel; unsure -> drop; human labels (incl. Mixed) win.
_f, _s = auto_labels(np.array(["River", "River", "River", "Forest", MIXED], object),
                     np.array([0, 0, 0, 1, 1], bool), np.array(["Pasture"] * 5), np.array([.9, .3, .6, .9, .9]))
assert _f.tolist() == ["Pasture", None, "River", "Forest", None], _f
assert _s.tolist() == ["auto", "dropped", "image", "human", "dropped"], _s


def load_windows(paths):
    """All 49 windows of each image, with their grid position."""
    parts = []
    for path in paths:
        img = tifffile.imread(path)
        if img.shape != (64, 64, len(BANDS)):
            print("skipped (unexpected shape):", path.name, img.shape)
            continue
        w = windows(img)
        rows, cols = w.shape[:2]
        df = features_batch(w.reshape(rows * cols, SIZE * SIZE, -1))
        df.insert(0, "col", np.tile(np.arange(cols), rows))
        df.insert(0, "row", np.repeat(np.arange(rows), cols))
        df.insert(0, "image_label", path.parent.name)
        df.insert(0, "path", str(path))
        df.insert(0, "file", path.name)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def stretch(img):
    rgb = img[..., [3, 2, 1]].astype(float)  # B04, B03, B02
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))  # per band: removes the blue haze cast
    return np.clip((rgb - lo) / np.maximum(hi - lo, 500), 0, 1)  # min range: flat water stays dark, not noise


def human_ask(win, guess, conf, classes):
    """Show one window on its image; return a class, MIXED, None (skip) or 'quit'."""
    keys = {str(i): c for i, c in enumerate(classes)} | {"m": MIXED, "s": None, "q": "quit"}
    rgb = stretch(tifffile.imread(win.path))
    y, x = win.row * STEP, win.col * STEP
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 6))
    a.imshow(rgb, interpolation="nearest")
    a.add_patch(Rectangle((x - .5, y - .5), SIZE, SIZE, fill=False, ec="#ff3b30", lw=2.5))
    a.set_title(f"{win.file}  (image label: {win.image_label})")
    b.imshow(rgb[y:y + SIZE, x:x + SIZE], interpolation="nearest")
    b.set_title(f"window row {win.row}, col {win.col}  |  forest: {guess} ({conf:.0%})")
    for ax in (a, b):
        ax.set_xticks([]), ax.set_yticks([])
    names = [f"{i} = {c}" for i, c in enumerate(classes)]
    fig.text(.5, .03, "   ".join(names[:5]) + "\n" + "   ".join(names[5:])
             + "\nm = Mixed / unsure     s = skip     q = stop this round", ha="center", fontsize=10)
    plt.show(block=False)
    plt.pause(0.1)  # draw the window, then read the answer from the terminal
    while (k := input("label [0-9 / m / s / q]: ").strip().lower()) not in keys:
        print("  unknown key, type one of: " + " ".join(keys))
    plt.close(fig)
    return keys[k]


def read_human(path):
    try:
        return pd.read_csv(path).drop_duplicates(KEY, keep="last")  # a window labelled twice: latest wins
    except FileNotFoundError:
        return pd.DataFrame(columns=HUMAN_COLS)


def fit_kmeans(features_csv):
    """Scaler + k-means fitted once on the training windows of window_features.csv (never test images)."""
    df = pd.read_csv(features_csv)
    train = df[df.split == "train"]
    scaler = StandardScaler().fit(train.drop(columns=META))  # k-means uses distances: one scale for all features
    km = KMeans(N_CLUSTERS, n_init=4, random_state=42).fit(scaler.transform(train.drop(columns=META)))
    print(f"k-means: {N_CLUSTERS} centroids fitted on {len(train)} training windows from {train.file.nunique()} images")
    return scaler, km, set(train.file)


def window_X(feats, scaler, km):
    """73 scaled features + distance to every saved centroid: the forest's input, in training and prediction."""
    X = scaler.transform(feats[scaler.feature_names_in_])
    return np.hstack([X, km.transform(X)])


def predict_windows(bundle, img):
    """img (64, 64, 13) -> (7, 7) predicted class and (7, 7, n_classes) probabilities, one per window."""
    w = windows(img, bundle["size"], bundle["step"])
    rows, cols = w.shape[:2]
    P = bundle["model"].predict_proba(window_X(features_batch(w.reshape(rows * cols, -1, img.shape[-1])),
                                               bundle["scaler"], bundle["kmeans"]))
    return bundle["model"].classes_[P.argmax(1)].reshape(rows, cols), P.reshape(rows, cols, -1)


def make_forest(oob=False):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", oob_score=oob,
                                  n_jobs=-1, random_state=42)


def forest_proba(X, y, weight, rows):
    """Fit on the windows in rows; class probabilities for every window, out-of-bag for those."""
    rf = make_forest(oob=True)
    with warnings.catch_warnings():  # "some inputs do not have OOB scores": handled below
        warnings.filterwarnings("ignore", "Some inputs do not have OOB scores")
        rf.fit(X[rows], y[rows], sample_weight=weight[rows])
    P = rf.predict_proba(X)  # windows not trained on (dropped, Mixed): ordinary prediction
    oob = rf.oob_decision_function_  # each training window scored only by trees that never saw it
    seen = oob.sum(1) > 0  # sklearn gives all zeros to a window no tree left out: keep predict_proba
    P[np.flatnonzero(rows)[seen]] = oob[seen]
    return rf.classes_, P


def run_round(pool, human_csv, rnd, classes, ask, scaler, km):
    human = read_human(human_csv)

    # 1. Saved k-means (fitted once on all training windows): distance to every centroid is an extra feature.
    X = window_X(pool, scaler, km)
    cluster = km.predict(X[:, :len(scaler.feature_names_in_)])

    # 2. Train on image labels (+ human), relabel / drop, retrain.
    h = pool[KEY].merge(human[KEY + ["label"]], on=KEY, how="left")["label"].to_numpy()  # keeps row order
    is_human = pd.notna(h)
    labels = np.where(is_human, h, pool.image_label).astype(object)
    weight = np.where(is_human, HUMAN_WEIGHT, 1)
    use = labels != MIXED
    classes_, P = forest_proba(X, labels, weight, use)
    final, source = auto_labels(labels, is_human, classes_[P.argmax(1)], P.max(1))
    keep = source != "dropped"
    classes_, P = forest_proba(X, final, weight, keep)
    top2 = np.sort(P, 1)[:, -2:]
    guess, conf, margin = classes_[P.argmax(1)], top2[:, 1], top2[:, 1] - top2[:, 0]

    moved = pd.crosstab(pool.image_label[source == "auto"], final[source == "auto"])
    print(f"\n--- Round {rnd}: {len(pool)} windows from {pool.file.nunique()} images, "
          f"{is_human.sum()} with a human label ---")
    print(f"Relabelled {(source == 'auto').sum()}, dropped {(source == 'dropped').sum()}")
    if len(moved):
        print("Relabels (rows: image label, columns: new label):\n" + moved.to_string())

    # 3. Critical windows: relabelled first, then smallest top-2 margin; one per cluster.
    cand = pd.DataFrame({"cluster": cluster, "score": margin - (source == "auto")})[~is_human]
    picked = cand.sort_values("score").groupby("cluster").head(1).head(PER_ROUND).index
    print(f"Asking you about {len(picked)} windows. Type the answer here and press Enter:\n  "
          + "  ".join(f"{i}={c}" for i, c in enumerate(classes)) + "\n  m=Mixed/unsure  s=skip  q=stop this round")
    agree, answered = 0, 0
    for n, i in enumerate(picked, 1):
        win = pool.loc[i]
        print(f"  {n}/{len(picked)}  {win.file} row {win.row} col {win.col}  ", end="", flush=True)
        label = ask(win, guess[i], conf[i], classes)
        print("->", label)
        if label == "quit":
            break
        if label is None:
            continue
        pd.DataFrame([[win.file, win.image_label, win.row, win.col, label, rnd, time.strftime("%Y-%m-%d %H:%M")]],
                     columns=HUMAN_COLS).to_csv(human_csv, mode="a", header=not human_csv.exists(), index=False)
        if label != MIXED:
            answered += 1
            agree += final[i] == label
        final[i], source[i] = (None, "dropped") if label == MIXED else (label, "human")
    if answered:  # automatic labels checked against you, before your answers train anything
        print(f"Automatic label matched yours on {agree}/{answered} windows ({agree / answered:.0%})")

    out = pool[KEY + ["image_label"]].assign(label=final, source=source, confidence=conf.round(3), round=rnd)
    return pd.concat([out, pool[scaler.feature_names_in_]], axis=1), (agree, answered)  # features: final forest


SOURCE_COLORS = {"human": "#2e9d4f", "auto": "#f08c00", "image": "#9aa5ad"}


def save_proof(out, root, png, per_class=8):
    """Figure: sample training windows per final label; border = where the label came from."""
    used = out[out.source != "dropped"]
    order = {"human": 0, "auto": 1, "image": 2}  # show human and relabelled windows first
    rng = np.random.default_rng(0)
    classes = sorted(used.label.unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.3 + 2, len(classes) * 1.35), squeeze=False)
    for r, c in enumerate(classes):
        g = used[used.label == c]
        n = g.source.value_counts()
        g = g.assign(o=g.source.map(order), k=rng.random(len(g))).sort_values(["o", "k"]).head(per_class)
        axes[r, 0].set_ylabel(f"{c}\n{n.sum()} windows\n{n.get('human', 0)} human, {n.get('auto', 0)} auto",
                              rotation=0, ha="right", va="center", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax, w in zip(axes[r], g.itertuples()):
            y, x = w.row * STEP, w.col * STEP
            ax.imshow(stretch(tifffile.imread(root / w.image_label / w.file))[y:y + SIZE, x:x + SIZE],
                      interpolation="nearest")
            for spine in ax.spines.values():
                spine.set(visible=True, edgecolor=SOURCE_COLORS[w.source], linewidth=3)
            if w.image_label != c:  # label changed: show the image it came from
                ax.set_title(f"from {w.image_label}", fontsize=6)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="none", ec=v, lw=3) for v in SOURCE_COLORS.values()],
               labels=["your label", "relabelled by the forest", "image label"], loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Windows used in training: {len(used)} of {len(out)} "
                 f"({(out.source == 'dropped').sum()} dropped)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(png, dpi=130)
    print(f"Saved {png}")
    plt.show()


def print_stats(out, human, checks):
    """Final statistics: dataset composition, label changes, and how your labels compare."""
    used = out[out.source != "dropped"]
    line = "=" * 72
    print(f"\n{line}\n  LABELLING STATISTICS\n{line}")
    print(f"Windows seen: {len(out)} from {out.file.nunique()} images; used in training: {len(used)} "
          f"({len(used) / len(out):.1%}), dropped: {len(out) - len(used)}")
    src = out.source.value_counts()
    print("Label source: " + ", ".join(f"{k} {v} ({v / len(out):.1%})" for k, v in src.items()))

    t = pd.DataFrame({
        "image windows": out.groupby("image_label").size(),  # windows cut from images of this class
        "kept": used[used.label == used.image_label].groupby("image_label").size(),
        "moved away": used[used.label != used.image_label].groupby("image_label").size(),
        "dropped": out[out.source == "dropped"].groupby("image_label").size(),
        "moved in": used[used.label != used.image_label].groupby("label").size(),
        "final windows": used.groupby("label").size(),
        "human": used[used.source == "human"].groupby("label").size(),
    }).fillna(0).astype(int)
    t["final %"] = (100 * t["final windows"] / t["final windows"].sum()).round(1)
    print("\nPer class (moved = label changed by the forest or by you):\n" + t.to_string())

    changed = used[used.label != used.image_label]
    if len(changed):
        print("\nLabel changes (rows: image class, columns: new label):\n"
              + pd.crosstab(changed.image_label, changed.label).to_string())

    if len(human):
        mixed = (human.label == MIXED).sum()
        named = human[human.label != MIXED]
        same = (named.label == named.image_label).mean() if len(named) else float("nan")
        print(f"\nYour labels: {len(human)} windows, {mixed} mixed / unsure; "
              f"you agreed with the image label on {same:.0%} of the others")
        per = (named.label == named.image_label).groupby(named.image_label).agg(["mean", "size"])
        print("Agreement with the image label, per image class: "
              + ", ".join(f"{k} {m:.0%} of {n}" for k, m, n in per.itertuples()))
    for rnd, (agree, answered) in enumerate(checks, 1):
        if answered:
            print(f"Round {rnd}: automatic label matched yours on {agree}/{answered} ({agree / answered:.0%})")


def label(per_class=PER_CLASS, rounds=ROUNDS, features_csv=FEATURES_CSV, ask=human_ask, human_csv=HUMAN_CSV,
         out_csv=OUT_CSV, root=None):
    per_class = 10**9 if str(per_class) == "all" else int(per_class)  # "all": every training image
    rounds = int(rounds)
    root = find_root(root or data_root())
    scaler, km, train_files = fit_kmeans(features_csv)
    paths = pd.Series([p for p in list_images(root) if p.name in train_files])  # training images only
    by_class = {c: g.tolist() for c, g in paths.groupby(paths.map(lambda p: p.parent.name))}
    classes = sorted(by_class)
    results, checks = [], []
    for rnd in range(1, rounds + 1):
        rng = np.random.default_rng(rnd)
        batch = {p for c in classes for p in rng.choice(by_class[c], min(per_class, len(by_class[c])), replace=False)}
        human = read_human(human_csv)
        batch |= {root / lab / f for f, lab in zip(human.file, human.image_label)}  # always include human-labelled images
        out, check = run_round(load_windows(sorted(batch)), human_csv, rnd, classes, ask, scaler, km)
        results.append(out)
        checks.append(check)
    out = pd.concat(results, ignore_index=True).drop_duplicates(KEY, keep="last")
    out.to_csv(out_csv, index=False, float_format="%.6g")  # with features: the dataset 'train' reads
    print(f"\nSaved {out_csv}: {len(out)} windows\n" + out.source.value_counts().to_string())
    print(f"Human labels: {len(read_human(human_csv))} in {human_csv}")
    print_stats(out, read_human(human_csv), checks)

    save_proof(out, root, Path(out_csv).with_suffix(".png"))
    return out


def load_labelled(labels_csv=OUT_CSV, features_csv=FEATURES_CSV, test_size=0.2):
    """Windows not dropped (with a train/test split column), forest input, labels, weights, scaler + k-means."""
    df = pd.read_csv(labels_csv)
    scaler, km, _ = fit_kmeans(features_csv)  # same seed and data as 'label': same centroids
    if not set(scaler.feature_names_in_) <= set(df.columns):
        raise FileNotFoundError(f"{labels_csv} has no window features: rerun train_labelled.py label")
    used = df[df.source != "dropped"].reset_index(drop=True)
    # 80/20 split by image, stratified by image class, like the whole-image model: test windows keep
    # their cleaned labels and come from images the grid search and final forest never train on.
    files = used.groupby("file").image_label.first()
    _, test_files = train_test_split(files.index, test_size=test_size, stratify=files, random_state=42)
    used["split"] = np.where(used.file.isin(test_files), "test", "train")
    weight = np.where(used.source == "human", HUMAN_WEIGHT, 1)
    return used, window_X(used, scaler, km), used.label.to_numpy(), weight, scaler, km



def train(labels_csv=OUT_CSV, cv_csv=PROJECT_DIR / "cv_results_labelled.csv", features_csv=FEATURES_CSV,
         out_png=PROJECT_DIR / "labelled_results.png", out_model="labelled_window_model"):
    out_model = model_file(out_model)
    cv = pd.read_csv(cv_csv)
    best = ast.literal_eval(cv.loc[cv.rank_test_score.idxmin(), "params"])  # rank 1
    used, X, y, w, scaler, km = load_labelled(labels_csv, features_csv)
    tr, te = (used.split == "train").to_numpy(), (used.split == "test").to_numpy()
    X_train, y_train, X_test, y_test = X[tr], y[tr], X[te], y[te]

    # A forest has no epochs: "training" = adding trees, so track loss as the forest grows.
    rng = np.random.default_rng(0)
    ltr = rng.choice(len(y_train), min(LOSS_SAMPLE, len(y_train)), replace=False)
    lte = rng.choice(len(y_test), min(LOSS_SAMPLE, len(y_test)), replace=False)
    n_total = best.pop("n_estimators")
    steps = sorted(set(range(10, n_total, max(n_total // 25, 1))) | {n_total})
    while True:  # refit with bigger leaves until the saved model fits under MAX_MB (GitHub's limit is 100 MB)
        rf = RandomForestClassifier(max_features="sqrt", warm_start=True, n_jobs=-1, random_state=42, **best)
        train_loss, test_loss = [], []
        for n in steps:
            rf.set_params(n_estimators=n).fit(X_train, y_train, sample_weight=w[tr])
            train_loss.append(log_loss(y_train[ltr], rf.predict_proba(X_train[ltr]), labels=rf.classes_))
            test_loss.append(log_loss(y_test[lte], rf.predict_proba(X_test[lte]), labels=rf.classes_))
            print(f"\r  trees {n:>4}/{n_total}  train loss {train_loss[-1]:.4f}  test loss {test_loss[-1]:.4f}",
                  end="", flush=True)
        print()
        est_mb = sum(t.tree_.node_count for t in rf.estimators_) * BYTES_PER_NODE / 1e6
        if est_mb <= MAX_MB:
            break
        best["min_samples_leaf"] = 2 * rf.min_samples_leaf  # fewer, larger leaves: about half the nodes
        print(f"  Model would be ~{est_mb:.0f} MB (limit {MAX_MB} MB): refitting with "
              f"min_samples_leaf={best['min_samples_leaf']}")

    P = rf.predict_proba(X_test)
    pred = rf.classes_[P.argmax(1)]
    src_tr, src_te = used.source[tr].value_counts(), used.source[te].value_counts()
    metrics = {"cv_macro_f1": cv.mean_test_score.max(),
               "window_accuracy": accuracy_score(y_test, pred),
               "window_macro_f1": f1_score(y_test, pred, average="macro"),
               "window_log_loss": log_loss(y_test, P, labels=rf.classes_),
               "n_train": int(tr.sum()), "n_test": int(te.sum())}
    human_te = (used.source[te] == "human").to_numpy()

    line = "=" * 64
    params = {"n_estimators": n_total, "max_features": "sqrt", **best}
    print(f"\n{line}\n  LABELLED WINDOW RANDOM FOREST ({SIZE}x{SIZE} px, step {STEP}, + {km.n_clusters} centroids)\n{line}")
    print("  Hyperparameters : " + ", ".join(f"{k}={v}" for k, v in params.items()))
    print(f"  Train windows   : {tr.sum()} from {used.file[tr].nunique()} images "
          f"({src_tr.get('human', 0)} yours ×{HUMAN_WEIGHT}, {src_tr.get('auto', 0)} relabelled by the forest)")
    print(f"  Test windows    : {te.sum()} from {used.file[te].nunique()} other images "
          f"({src_te.get('human', 0)} yours, {src_te.get('auto', 0)} relabelled by the forest)")
    print(f"  CV macro-F1     : {metrics['cv_macro_f1']:.4f}  (folds grouped by image)")
    print(f"  Test accuracy   : {metrics['window_accuracy']:.2%}")
    print(f"  Test macro-F1   : {metrics['window_macro_f1']:.4f}")
    print(f"  Test log loss   : {metrics['window_log_loss']:.4f}")
    if human_te.any():
        print(f"  On your labels  : {accuracy_score(y_test[human_te], pred[human_te]):.2%} "
              f"of {human_te.sum()} test windows you labelled")
    print(f"{line}\n\nPer-class results (test windows):\n")
    print(classification_report(y_test, pred, digits=3))

    bundle = {"model": rf, "name": Path(out_model).stem, "scaler": scaler, "kmeans": km,
              "features": list(scaler.feature_names_in_), "size": SIZE, "step": STEP, "per_image": "all",
              "params": {**params, "k-means centroids": km.n_clusters,
                         "human labels (weight)": f"{src_tr.get('human', 0)} (×{HUMAN_WEIGHT})",
                         "relabelled by the forest": int(src_tr.get("auto", 0))},
              "metrics": metrics, "trained": time.strftime("%Y-%m-%d %H:%M")}

    # Self-check: from pixels, the saved bundle predicts a training image exactly as in training.
    for f in used[tr].file.unique()[:50]:
        rows = used.index[used.file == f]
        if len(rows) == 49:  # an image with no dropped window
            img = used.loc[rows[0]]
            grid, _ = predict_windows(bundle, tifffile.imread(find_root(data_root()) / img.image_label / f))
            order = used.loc[rows].sort_values(["row", "col"]).index
            assert (grid.ravel() == rf.predict(X[order])).all(), "prediction path differs from training"
            break

    joblib.dump(bundle, out_model, compress=("lzma", 3))  # ~35% smaller than zlib, loads as fast
    size_mb = out_model.stat().st_size / 1e6
    print(f"Saved {out_model} ({size_mb:.0f} MB)")
    if size_mb > 100:
        print(f"  Warning: over GitHub's 100 MB limit; lower MAX_MB ({MAX_MB}) and train again.")

    # Same look as train_final.py: dataviz palette, loss curve + confusion matrix.
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})
    for losses, color, name in [(train_loss, BLUE, "Train windows"), (test_loss, ORANGE, "Test windows")]:
        ax1.plot(steps, losses, color=color, lw=2, label=name)
        ax1.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", color=INK, fontsize=9)
    ax1.set(title="Log loss as trees are added", xlabel="Number of trees", ylabel="Log loss")
    ax1.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax1.legend(frameon=False)
    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    ConfusionMatrixDisplay.from_predictions(y_test, pred, normalize="true", values_format=".0%",
                                            cmap=blues, ax=ax2, colorbar=False)
    ax2.set_title("Confusion matrix, test windows (share of each true class)")
    ax2.tick_params(axis="x", labelrotation=45)
    plt.setp(ax2.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")
    plt.show()
    return rf



if __name__ == "__main__":
    modes = {"label": label, "train": train}
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        sys.exit(__doc__.split("Usage:")[1].strip())
    try:
        modes[sys.argv[1]](*sys.argv[2:5])
    except FileNotFoundError as e:  # input from an earlier step is missing
        sys.exit(f"{e}\nRun the previous step first (see README).")
