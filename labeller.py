"""Background labelling job for the web app's Label page: train_labelled.label with the browser as labeller.

One job at a time runs in a thread. When the labelling loop asks about a window, the question waits in
STATE["question"] until the page posts an answer to /label/answer.
"""
import base64
import contextlib
import io
import threading
import time
import traceback

import numpy as np
import tifffile
from PIL import Image

import train_labelled as T

STATE = {"running": False}
_answered = threading.Event()
_lock = threading.Lock()


def elapsed():
    """Time since the run started, as H:MM:SS."""
    t = int(time.time() - STATE.get("t0", time.time()))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def _log(line):
    STATE["log"].append(f"{time.strftime('%H:%M:%S')}  +{elapsed()}  {line}")


class _Log(io.TextIOBase):
    """label() prints its progress and statistics; keep the lines for the page, with clock and elapsed time."""
    def write(self, s):
        for line in s.replace("\r", "\n").splitlines():
            if line.strip():
                _log(line.rstrip())
        return len(s)


def _png(rgb, size):
    buf = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.NEAREST).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ask(win, guess, conf, classes):
    """Called by label() for each critical window; blocks until the page answers."""
    rgb = T.stretch(tifffile.imread(win.path))
    y, x = win.row * T.STEP, win.col * T.STEP
    STATE["asked"] += 1
    STATE["answer"] = None
    _answered.clear()
    STATE["question"] = {"id": STATE["asked"], "file": win.file, "image_label": win.image_label,
                         "row": int(win.row), "col": int(win.col), "guess": guess, "conf": float(conf),
                         "tile": _png(rgb, 320), "crop": _png(rgb[y:y + T.SIZE, x:x + T.SIZE], 320),
                         "box": [100 * x / rgb.shape[1], 100 * y / rgb.shape[0], 100 * T.SIZE / rgb.shape[1]]}
    _log(f"Waiting for your answer on window {STATE['asked']}: {win.file} row {win.row + 1}, col {win.col + 1}")
    while not _answered.wait(0.5):
        if STATE.get("stop"):
            STATE["question"] = None
            return "quit"
    STATE["question"] = None
    return STATE["answer"]


def answer(qid, value):
    """value: a class name, 'Mixed', 'skip' or 'quit'. False if qid is not the open question."""
    q = STATE.get("question")
    if not q or q["id"] != qid:
        return False
    STATE["answer"] = None if value == "skip" else value
    _log(f"Your answer on window {qid}: {value}")
    _answered.set()
    return True


def stop():
    STATE["stop"] = True  # no more questions; the run still finishes and saves the dataset


def _run(per_class, rounds, features_csv):
    try:
        with contextlib.redirect_stdout(_Log()):
            T.label(per_class, rounds, features_csv, ask=_ask, on_stats=lambda st: STATE.update(stats=st))
        STATE["finished"] = time.strftime("%H:%M")
        _log("Finished")
    except Exception as e:  # shown on the page; the answers given so far are already saved
        STATE["error"] = f"{type(e).__name__}: {e}"
        _log(f"Failed: {STATE['error']}")
        STATE["log"].append(traceback.format_exc())
    finally:
        STATE.update(running=False, question=None, total=elapsed())


def start(per_class, rounds, features_csv):
    with _lock:
        if STATE.get("running"):
            return False
        STATE.clear()
        STATE.update(running=True, log=[], asked=0, question=None, started=time.strftime("%H:%M"), t0=time.time(),
                     settings={"per_class": per_class, "rounds": rounds, "features_csv": str(features_csv)})
        _log(f"Started: {per_class} images per class, {rounds} round(s), {features_csv}")
        threading.Thread(target=_run, args=(per_class, rounds, features_csv), daemon=True).start()
    return True
