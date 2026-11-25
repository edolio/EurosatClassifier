"""Approach 2: labelled window model, built and trained in two modes (one file).

label: builds the labelled window dataset. A scaler and k-means (40 centroids) are fitted once on
the training windows of window_features.csv (prepare_windows.py), so the same centroids serve every
round and later predictions. Each round takes a fresh batch of training images per class (all 49
windows each) plus every image a human already labelled a window of, then:
1. scales the 73 features and adds each window's distance to every saved centroid;
2. trains the forest on the image labels (human labels override them and weigh HUMAN_WEIGHT times
   more) and scores each window out-of-bag, i.e. only by trees that did not train on it. It
   relabels windows it is confident belong to another class, relabels windows of River/SeaLake
   tiles it sees as clearly land (banks, shores), drops windows it is unsure about, and retrains. Relabels are recomputed every round from the image labels, never chained;
3. shows you the most critical windows (one per cluster): the window outlined on its image and
   enlarged. Type a class number in the terminal, m (mixed / unsure: the window is dropped), s (skip)
   or q (stop this round). Each answer is appended to human_labels.csv at once, so nothing is lost on quit.
   Outputs: human_labels.csv (your labels, read again on every run), window_labels.csv.gz (the dataset:
   final label and source of every window seen, ~1.4 MB), window_labels_features.csv (the 73 features
   of the same windows, large and not in git), window_labels.png (sample windows
   per label, framed by where the label came from) and printed statistics.

train: after grid_search_labelled.py, trains the final forest with the best params. Besides a window's
own features it sees how much water surrounds it (mean NDWI and share of water windows among the 24
windows around it, 5x5), so land on a bank is not pulled into River. For windows it calls water, a
second forest decides River vs SeaLake from the class probabilities of those 24 neighbours (a river
has banks around it). Like train_final.py but every number is per window: loss curve as trees are added, confusion matrix,
accuracy, macro-F1, log loss and per-class results. Train and test windows come from an 80/20 split
by image; test windows keep their cleaned labels and come from images the forest never trains on.
Saves labelled_results.png and models/labelled_window_model.joblib (the web app's window model).

Usage: python train_labelled.py label [images per class, or all] [rounds] [window_features.csv]
       python train_labelled.py train [window_labels.csv.gz] [cv_results_labelled.csv] [window_features.csv]
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
FIT_PER_IMAGE = 8  # windows per image the labelling forests learn from (all windows are scored)
MIXED = "Mixed"
WATER = ["River", "SeaLake"]
BANK_WATER_P = 0.2  # below this water probability, a window in a River/SeaLake tile is relabelled as land
HUMAN_CSV = PROJECT_DIR / "human_labels.csv"
OUT_CSV = PROJECT_DIR / "window_labels.csv.gz"  # labels only, gzipped: ~1.4 MB, small enough for git
PROOF_PNG = PROJECT_DIR / "window_labels.png"
LABEL_COLS = ["file", "row", "col", "image_label", "label", "source", "confidence", "round"]
FEATURES_CSV = PROJECT_DIR / "window_features.csv"  # from prepare_windows.py: k-means is fitted on its training windows
KEY = ["file", "row", "col"]  # one window
LOSS_SAMPLE = 20000  # windows per side used to draw the loss curve (scoring all at 25 sizes is slow)
MAX_MB = 90  # saved model size limit, below GitHub's 100 MB per file
NEIGHBOUR_R = 2  # water context from the 24 windows within 2 steps (5x5): 3x3 often holds only more water
BYTES_PER_NODE = 16.5  # lzma-compressed size per tree node, measured on a saved model of 3 million nodes
HUMAN_COLS = ["file", "image_label", "row", "col", "label", "round", "time"]


def auto_labels(labels, is_human, pred, conf, water_p):
    """Final label and its source for every window (step 2); human labels are never changed.

    water_p: forest probability of River + SeaLake. A window labelled River/SeaLake (its tile's label)
    that the forest sees as clearly land is relabelled to that land class: banks and shores must not
    teach "next to water = water"."""
    auto = ~is_human
    bank = auto & np.isin(labels, WATER) & (water_p < BANK_WATER_P) & ~np.isin(pred, WATER)
    relabel = (auto & (conf >= RELABEL_P) & (pred != labels)) | bank
    drop = (auto & (conf < DROP_P)) | (labels == MIXED)
    final = np.where(relabel, pred, labels).astype(object)
    final[drop] = None
    source = np.select([drop, is_human, relabel], ["dropped", "human", "auto"], "image")
    return final, source


# Self-check: confident other class -> relabel; unsure -> drop; human labels (incl. Mixed) win;
# a River window that is clearly land (water 0.1) -> land even at 0.6 confidence; with water 0.3 it stays.
_f, _s = auto_labels(np.array(["River", "River", "River", "Forest", MIXED, "River", "River"], object),
                     np.array([0, 0, 0, 1, 1, 0, 0], bool), np.array(["Pasture"] * 7),
                     np.array([.9, .3, .6, .9, .9, .6, .6]), np.array([.05, .2, .3, .05, .05, .1, .3]))
assert _f.tolist() == ["Pasture", None, "River", "Forest", None, "Pasture", "River"], _f
assert _s.tolist() == ["auto", "dropped", "image", "human", "dropped", "auto", "image"], _s


def load_windows(paths):
    """All 49 windows of each image, with their grid position."""
    parts = []
    for i, path in enumerate(paths, 1):
        if i % 500 == 0 or i == len(paths):
            print(f"Reading images and computing window features: {i}/{len(paths)}")
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


def neighbour_mean(P, rows=7, cols=7, r=NEIGHBOUR_R):
    """P: (images * rows * cols, classes) in row-major grid order -> mean of each window's neighbours within r
    windows (a (2r+1) x (2r+1) square without the window itself)."""
    G = np.pad(P.reshape(-1, rows, cols, P.shape[-1]), ((0, 0), (r, r), (r, r), (0, 0)), constant_values=np.nan)
    shifts = [G[:, r + dy:r + dy + rows, r + dx:r + dx + cols]
              for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy or dx]
    return np.nanmean(shifts, axis=0).reshape(P.shape)  # edge windows: only the neighbours inside the tile


# Self-check: in a 2x2 grid every window's neighbours are the other three.
assert np.allclose(neighbour_mean(np.array([[0.], [3.], [6.], [9.]]), 2, 2, r=1).ravel(), [6, 5, 4, 3])


def water_context(F, rows=7, cols=7):
    """Per window: mean NDWI and share of water windows (NDWI > 0) among its neighbours within NEIGHBOUR_R.
    A water signal, not class probabilities: "next to water" says nothing about being River."""
    ndwi = F["NDWI_mean"].to_numpy()[:, None]
    return np.hstack([neighbour_mean(ndwi, rows, cols), neighbour_mean((ndwi > 0).astype(float), rows, cols)])


def water_gate(P, Pd, classes, d_classes):
    """Where P's top class is River or SeaLake, split its water probability between the two as the
    decider (Pd) says. Land windows are left exactly as they are."""
    ri, si = list(classes).index("River"), list(classes).index("SeaLake")
    m = np.isin(np.asarray(classes)[P.argmax(1)], WATER)
    water = P[m, ri] + P[m, si]
    P = P.copy()
    P[m, ri], P[m, si] = water * Pd[m, list(d_classes).index("River")], water * Pd[m, list(d_classes).index("SeaLake")]
    return P


# Self-check: a water window follows the decider; a land window (top class Pasture) is untouched.
_P = water_gate(np.array([[.6, .3, .1], [.2, .2, .6]]), np.array([[.1, .9], [.9, .1]]),
                ["River", "SeaLake", "Pasture"], ["River", "SeaLake"])
assert np.allclose(_P, [[.09, .81, .1], [.2, .2, .6]]), _P


def predict_windows(bundle, img):
    """img (64, 64, 13) -> (7, 7) predicted class and (7, 7, n_classes) probabilities, one per window."""
    w = windows(img, bundle["size"], bundle["step"])
    rows, cols = w.shape[:2]
    F = features_batch(w.reshape(rows * cols, -1, img.shape[-1]))
    X = window_X(F, bundle["scaler"], bundle["kmeans"])
    X1 = X
    if bundle.get("water_context"):  # + how much water surrounds each window
        X = np.hstack([X, water_context(F, rows, cols)])
    P = bundle["model"].predict_proba(X)
    if "decider" in bundle:  # River vs SeaLake from the neighbours' class probabilities
        X2 = np.hstack([X1, neighbour_mean(bundle["stage1"].predict_proba(X1), rows, cols)])
        P = water_gate(P, bundle["decider"].predict_proba(X2), bundle["model"].classes_, bundle["decider"].classes_)
    return bundle["model"].classes_[P.argmax(1)].reshape(rows, cols), P.reshape(rows, cols, -1)


def make_forest(oob=False, **params):
    return RandomForestClassifier(**{"n_estimators": 200, "min_samples_leaf": 2, **params}, max_features="sqrt",
                                  oob_score=oob, n_jobs=-1, random_state=42)


def forest_proba(X, y, weight, rows, **params):
    """Fit a forest on the windows in rows; the forest and class probabilities for every window,
    out-of-bag for those."""
    rf = make_forest(oob=True, **params)
    with warnings.catch_warnings():  # "some inputs do not have OOB scores": handled below
        warnings.filterwarnings("ignore", "Some inputs do not have OOB scores")
        rf.fit(X[rows], y[rows], sample_weight=weight[rows])
    P = rf.predict_proba(X)  # windows not trained on (dropped, Mixed): ordinary prediction
    oob = rf.oob_decision_function_  # each training window scored only by trees that never saw it
    seen = oob.sum(1) > 0  # sklearn gives all zeros to a window no tree left out: keep predict_proba
    P[np.flatnonzero(rows)[seen]] = oob[seen]
    return rf, P


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
    # The forests learn from FIT_PER_IMAGE random windows per image (+ all yours) but score every window:
    # neighbouring windows overlap by half, so the rest add little but time.
    rng = np.random.default_rng(rnd)
    rank = pool.assign(k=rng.random(len(pool))).sort_values("k").groupby("file").cumcount().sort_index()
    fit = (rank < FIT_PER_IMAGE).to_numpy() | is_human
    print(f"Fitting forest 1/2 (relabel and drop) on {(use & fit).sum()} windows, scoring all {len(pool)}")
    rf, P = forest_proba(X, labels, weight, use & fit)
    classes_ = rf.classes_
    final, source = auto_labels(labels, is_human, classes_[P.argmax(1)], P.max(1), P[:, np.isin(classes_, WATER)].sum(1))
    keep = source != "dropped"
    print(f"Fitting forest 2/2 (choose the windows to ask about) on {(keep & fit).sum()} windows")
    rf, P = forest_proba(X, final, weight, keep & fit)
    classes_ = rf.classes_
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
    print(f"Asking you about {len(picked)} windows (in the terminal: type the key, then Enter):\n  "
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
    changes = pd.crosstab(changed.image_label, changed.label) if len(changed) else None
    if changes is not None:
        print("\nLabel changes (rows: image class, columns: new label):\n" + changes.to_string())

    def table(df):  # JSON-ready table for the web app (plain ints/floats, not numpy types)
        num = lambda v: int(v) if float(v).is_integer() else float(v)
        return {"columns": [str(c) for c in df.columns],
                "rows": [[str(i), *map(num, r)] for i, r in zip(df.index, df.to_numpy())]}
    stats = {"seen": len(out), "images": int(out.file.nunique()), "used": len(used), "dropped": len(out) - len(used),
             "sources": {k: int(v) for k, v in src.items()}, "per_class": table(t),
             "changes": table(changes) if changes is not None else None, "human": None, "rounds": []}
    if len(human):
        mixed = (human.label == MIXED).sum()
        named = human[human.label != MIXED]
        same = (named.label == named.image_label).mean() if len(named) else float("nan")
        print(f"\nYour labels: {len(human)} windows, {mixed} mixed / unsure; "
              f"you agreed with the image label on {same:.0%} of the others")
        per = (named.label == named.image_label).groupby(named.image_label).agg(["mean", "size"])
        print("Agreement with the image label, per image class: "
              + ", ".join(f"{k} {m:.0%} of {n}" for k, m, n in per.itertuples()))
        stats["human"] = {"total": len(human), "mixed": int(mixed), "agree_image": None if np.isnan(same) else float(same),
                          "per_class": [[k, float(m), int(n)] for k, m, n in per.itertuples()],
                          "labels": {k: int(v) for k, v in human.label.value_counts().items()}}
    for rnd, (agree, answered) in enumerate(checks, 1):
        if answered:
            print(f"Round {rnd}: automatic label matched yours on {agree}/{answered} ({agree / answered:.0%})")
            stats["rounds"].append([rnd, int(agree), int(answered)])
    return stats


def features_path(labels_csv):
    """window_labels.csv.gz -> window_labels_features.csv: the 73 features of the same windows (large, not in git)."""
    return Path(labels_csv).with_name(Path(labels_csv).name.split(".")[0] + "_features.csv")


def label(per_class=PER_CLASS, rounds=ROUNDS, features_csv=FEATURES_CSV, ask=human_ask, human_csv=HUMAN_CSV,
         out_csv=OUT_CSV, root=None, on_stats=None):
    """on_stats(stats), if given, receives the final statistics as soon as they exist (the web app shows them)."""
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
    stats = print_stats(out, read_human(human_csv), checks)
    if on_stats:
        on_stats(stats)
    feats_csv = features_path(out_csv)
    print(f"\nSaving the dataset ({len(out)} windows; the features take about a minute per 100,000 windows)...")
    out[LABEL_COLS].to_csv(out_csv, index=False)  # .gz in the name: pandas compresses
    out[KEY + list(scaler.feature_names_in_)].to_csv(feats_csv, index=False, float_format="%.6g")
    print(f"Saved {out_csv} (labels) and {feats_csv} (features)")
    print(f"Human labels: {len(read_human(human_csv))} in {human_csv}")
    save_proof(out, root, Path(out_csv).with_name(Path(out_csv).name.split(".")[0] + ".png"))
    return out


def load_labelled(labels_csv=OUT_CSV, features_csv=FEATURES_CSV, test_size=0.2, with_dropped=False):
    """Windows (not dropped, or all with with_dropped) with a train/test split column, forest input,
    labels, weights, scaler + k-means. Rows are sorted by image, row, col."""
    df = pd.read_csv(labels_csv)
    scaler, km, _ = fit_kmeans(features_csv)  # same seed and data as 'label': same centroids
    if not set(scaler.feature_names_in_) <= set(df.columns):  # labels and features are saved apart
        df = df.merge(pd.read_csv(features_path(labels_csv)), on=KEY, how="left", validate="one_to_one")
        if df[list(scaler.feature_names_in_)].isna().any().any():
            raise FileNotFoundError(f"{features_path(labels_csv)} does not match {labels_csv}: rerun train_labelled.py label")
    df = df.sort_values(KEY).reset_index(drop=True)
    # 80/20 split by image, stratified by image class, like the whole-image model: test windows keep
    # their cleaned labels and come from images the grid search and final forest never train on.
    files = df[df.source != "dropped"].groupby("file").image_label.first()
    _, test_files = train_test_split(files.index, test_size=test_size, stratify=files, random_state=42)
    used = (df if with_dropped else df[df.source != "dropped"]).reset_index(drop=True)
    used["split"] = np.where(used.file.isin(test_files), "test", "train")
    weight = np.where(used.source == "human", HUMAN_WEIGHT, 1)
    return used, window_X(used, scaler, km), used.label.to_numpy(), weight, scaler, km



def train(labels_csv=OUT_CSV, cv_csv=PROJECT_DIR / "cv_results_labelled.csv", features_csv=FEATURES_CSV,
         out_png=PROJECT_DIR / "labelled_results.png", out_model="labelled_window_model"):
    out_model = model_file(out_model)
    cv = pd.read_csv(cv_csv)
    best = ast.literal_eval(cv.loc[cv.rank_test_score.idxmin(), "params"])  # rank 1
    # All windows (dropped ones too) so every window has its full neighbourhood; only used ones train or score.
    used, X1, y, w, scaler, km = load_labelled(labels_csv, features_csv, with_dropped=True)
    assert (used.groupby("file").size() == 49).all(), "every image needs all 49 windows"
    keep = (used.source != "dropped").to_numpy()
    tr, te = keep & (used.split == "train").to_numpy(), keep & (used.split == "test").to_numpy()
    y_train, y_test = y[tr], y[te]

    # A forest has no epochs: "training" = adding trees, so track loss as the forest grows.
    rng = np.random.default_rng(0)
    ltr = rng.choice(len(y_train), min(LOSS_SAMPLE, len(y_train)), replace=False)
    lte = rng.choice(len(y_test), min(LOSS_SAMPLE, len(y_test)), replace=False)
    n_total = best.pop("n_estimators")
    steps = sorted(set(range(10, n_total, max(n_total // 25, 1))) | {n_total})
    # Own features + water around the window (24 neighbours, 5x5): water in a river tile has banks nearby.
    X = np.hstack([X1, water_context(used)])
    X_train, X_test = X[tr], X[te]
    water_tr = tr & np.isin(y, WATER)
    while True:  # refit with bigger leaves until the saved model fits under MAX_MB (GitHub's limit is 100 MB)
        # River vs SeaLake decider: stage 1 (own features; out-of-bag on training windows, so the decider
        # doesn't learn from over-confident scores), then a forest on water windows that also sees the
        # 24 neighbours' stage-1 probabilities (a river has banks around it, a lake has more lake).
        s1, P1 = forest_proba(X1, y, w, tr, n_estimators=n_total, **best)
        X2 = np.hstack([X1, neighbour_mean(P1)])
        dec = make_forest(n_estimators=n_total, **best).fit(X2[water_tr], y[water_tr], sample_weight=w[water_tr])
        # Main model: own features + water context; decides land vs water and the land class.
        rf = RandomForestClassifier(max_features="sqrt", warm_start=True, n_jobs=-1, random_state=42, **best)
        train_loss, test_loss = [], []
        for n in steps:
            rf.set_params(n_estimators=n).fit(X_train, y_train, sample_weight=w[tr])
            train_loss.append(log_loss(y_train[ltr], rf.predict_proba(X_train[ltr]), labels=rf.classes_))
            test_loss.append(log_loss(y_test[lte], rf.predict_proba(X_test[lte]), labels=rf.classes_))
            print(f"\r  trees {n:>4}/{n_total}  train loss {train_loss[-1]:.4f}  test loss {test_loss[-1]:.4f}",
                  end="", flush=True)
        print()
        est_mb = sum(t.tree_.node_count for f in (rf, s1, dec) for t in f.estimators_) * BYTES_PER_NODE / 1e6
        if est_mb <= MAX_MB:
            break
        best["min_samples_leaf"] = 2 * rf.min_samples_leaf  # fewer, larger leaves: about half the nodes
        print(f"  Model would be ~{est_mb:.0f} MB (limit {MAX_MB} MB): refitting with "
              f"min_samples_leaf={best['min_samples_leaf']}")

    P = water_gate(rf.predict_proba(X_test), dec.predict_proba(X2[te]), rf.classes_, dec.classes_)
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

    bundle = {"model": rf, "water_context": True, "stage1": s1, "decider": dec, "name": Path(out_model).stem, "scaler": scaler, "kmeans": km,
              "features": list(scaler.feature_names_in_), "size": SIZE, "step": STEP, "per_image": "all",
              "params": {**params, "k-means centroids": km.n_clusters,
                         "context": "water share + mean NDWI of the 24 neighbouring windows",
                         "River vs SeaLake": "decided from the 24 neighbours' class probabilities",
                         "human labels (weight)": f"{src_tr.get('human', 0)} (×{HUMAN_WEIGHT})",
                         "relabelled by the forest": int(src_tr.get("auto", 0))},
              "metrics": metrics, "trained": time.strftime("%Y-%m-%d %H:%M")}

    # Self-check: from pixels, the saved bundle predicts a test image exactly as evaluated here.
    f = used.file[te].iloc[0]
    rows = np.flatnonzero((used.file == f).to_numpy())  # its 49 windows, sorted by row and col
    grid, _ = predict_windows(bundle, tifffile.imread(find_root(data_root()) / used.image_label[rows[0]] / f))
    Pg = water_gate(rf.predict_proba(X[rows]), dec.predict_proba(X2[rows]), rf.classes_, dec.classes_)
    assert (grid.ravel() == rf.classes_[Pg.argmax(1)]).all(), "prediction path differs from evaluation"

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
