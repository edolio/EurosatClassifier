"""Background training/validation job for the web app's "Train & validate" page.

One job at a time runs in a thread; the page polls STATE through /train/status.
Features are cached per (approach, data share, test share), so repeated runs skip the slow disk reads.
"""
import threading
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (ParameterGrid, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score)

import prepare
import prepare_windows
from grid_search import split

MODELS_DIR = prepare.MODELS_DIR
MAX_CANDIDATES = 100
STATE = {"running": False}
_cache = {}
_lock = threading.Lock()


def _log(msg):
    STATE["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")


def _stage(name, total):
    STATE.update(stage=name, done=0, total=total)
    _log(name)


def _tick(done, total=None):
    STATE["done"] = done
    if total is not None:
        STATE["total"] = total


def _data(approach, root, frac, test_size):
    """-> X_train, X_test, y_train, y_test, groups (train image of each row, or None), test_files."""
    key = (approach, str(root), frac, test_size)
    if key in _cache:
        _log("Features loaded from cache")
        return _cache[key]
    _stage(f"Reading images and computing features ({frac:.0%} of the data)", 1)
    if approach == "image":
        df = prepare.load(root, frac, progress=_tick)
        X_train, X_test, y_train, y_test = split(df, test_size)
        out = X_train, X_test, y_train, y_test, None, None
    else:
        df = prepare_windows.load_windows(root, frac, test_size, progress=_tick)
        tr, te = df[df.split == "train"], df[df.split == "test"]
        M = prepare_windows.META
        out = tr.drop(columns=M), te.drop(columns=M), tr.label, te.label, tr.file, te.file.to_numpy()
    if out[0].empty or out[1].empty:
        raise ValueError("Not enough images for this data share and test share.")
    _cache.clear()  # keep only the latest feature set in memory
    _cache[key] = out
    unit = "images" if approach == "image" else "windows"
    _log(f"{len(out[0])} train / {len(out[1])} test {unit}, {out[0].shape[1]} features")
    return out


def _validate(X, y, groups, grid, folds):
    candidates = list(ParameterGrid(grid))
    _stage(f"Validating {len(candidates)} candidate(s) with {folds}-fold cross-validation", len(candidates))
    cv = (StratifiedGroupKFold(folds, shuffle=True, random_state=42) if groups is not None
          else StratifiedKFold(folds, shuffle=True, random_state=42))
    rows = []
    for i, params in enumerate(candidates, 1):
        t = time.time()
        s = cross_val_score(RandomForestClassifier(random_state=42, **params), X, y, groups=groups,
                            cv=cv, scoring="f1_macro", n_jobs=-1)
        rows.append({"params": params, "mean": float(s.mean()), "std": float(s.std())})
        _tick(i)
        _log(f"  {i}/{len(candidates)} {params}: macro-F1 {s.mean():.4f} ± {s.std():.4f} ({time.time() - t:.0f}s)")
    rows.sort(key=lambda r: -r["mean"])
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return rows


def _train(approach, data, params, cv_score, name):
    X_train, X_test, y_train, y_test, _, test_files = data
    n_total = params["n_estimators"]
    steps = sorted(set(range(10, n_total, max(n_total // 20, 1))) | {n_total})
    _stage(f"Training final model {params}", n_total)
    rest = {k: v for k, v in params.items() if k != "n_estimators"}
    rf = RandomForestClassifier(warm_start=True, n_jobs=-1, random_state=42, **rest)
    for n in steps:  # add trees in steps so the page can show progress
        rf.set_params(n_estimators=n).fit(X_train, y_train)
        _tick(n)
    proba = rf.predict_proba(X_test)
    pred = rf.classes_[proba.argmax(1)]
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    if path.exists():  # appeared while training: never overwrite
        path = MODELS_DIR / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.joblib"
    common = {"model": rf, "name": name, "params": params, "features": list(X_train.columns),
              "trained": time.strftime("%Y-%m-%d %H:%M")}
    if approach == "image":
        metrics = {"test_accuracy": acc, "test_macro_f1": f1, "cv_macro_f1": cv_score,
                   "n_train": len(X_train), "n_test": len(X_test)}
        joblib.dump({**common, "metrics": metrics}, path, compress=3)
    else:
        per_img = pd.DataFrame(proba).groupby(test_files).mean()
        img_true = pd.Series(y_test.to_numpy()).groupby(test_files).first().loc[per_img.index]
        img_acc = accuracy_score(img_true, rf.classes_[per_img.to_numpy().argmax(1)])
        metrics = {"cv_macro_f1": cv_score, "window_accuracy": acc, "window_macro_f1": f1,
                   "image_vote_accuracy": img_acc, "n_train": len(X_train), "n_test": len(X_test)}
        bundle = {**common, "metrics": metrics, "size": prepare_windows.SIZE,
                  "step": prepare_windows.STEP, "per_image": prepare_windows.PER_IMAGE}
        joblib.dump(bundle, path, compress=3)
    _log(f"Test accuracy {acc:.2%}, macro-F1 {f1:.4f}" +
         (f", image accuracy from windows {metrics['image_vote_accuracy']:.2%}" if approach != "image" else ""))
    _log(f"Saved {path}")
    return {"path": str(path), "name": name, "metrics": metrics, "params": params, "approach": approach}


def _run(cfg):
    try:
        t0 = time.time()
        data = _data(cfg["approach"], cfg["root"], cfg["frac"], cfg["test_size"])
        cv_rows, best, cv_score = None, cfg["grid"], None
        if cfg["mode"] in ("validate", "both"):
            cv_rows = _validate(data[0], data[2], data[4], cfg["grid"], cfg["folds"])
            STATE["cv"] = [{**r, "params": str(r["params"])} for r in cv_rows]
            best, cv_score = cv_rows[0]["params"], cv_rows[0]["mean"]
            _log(f"Best: {best} (macro-F1 {cv_score:.4f})")
        if cfg["mode"] in ("train", "both"):
            params = {k: (v[0] if isinstance(v, list) else v) for k, v in best.items()}
            STATE["model"] = _train(cfg["approach"], data, params, cv_score, cfg["name"])
        STATE.update(stage=f"Finished in {time.time() - t0:.0f}s", done=1, total=1)
    except Exception as e:  # show the error on the page instead of losing it in a thread
        STATE["error"] = str(e)
        _log("ERROR: " + str(e))
        traceback.print_exc()
    finally:
        STATE["running"] = False


def start(cfg):
    """Start a job; returns False if one is already running."""
    with _lock:
        if STATE.get("running"):
            return False
        n = len(ParameterGrid(cfg["grid"]))
        if cfg["mode"] != "train" and n > MAX_CANDIDATES:
            raise ValueError(f"{n} combinations; the limit is {MAX_CANDIDATES}.")
        STATE.clear()
        STATE.update(running=True, cfg={k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                     log=[], stage="Starting", done=0, total=1, cv=None, model=None, error=None)
        threading.Thread(target=_run, args=(cfg,), daemon=True).start()
        return True
