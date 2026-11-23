"""Local web page: upload a EuroSAT_MS image (13-band GeoTIFF), get the predicted land-cover class.

Two approaches on one page: the whole-image model (top) and the window model (below).

A second page (/train) trains and validates new models in the background and shows the progress.

Usage: python app.py [model name] [window model name]
The default dataset folder is chosen on the training page ("Save as default").
Models are always read from and saved to the models/ folder; only their file names are given.
then open http://127.0.0.1:5000
"""
import base64
import io
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tifffile
from flask import Flask, jsonify, redirect, render_template_string, request
from PIL import Image

import trainer
from prepare import B, BANDS, data_root, features, features_batch, find_root, model_file, save_data_root
from prepare_windows import windows


MODELS_DIR = trainer.MODELS_DIR.resolve()  # the only place models are read from and saved to
MODEL_PATH = model_file(sys.argv[1] if len(sys.argv) > 1 else "model").resolve()
WINDOW_PATH = model_file(sys.argv[2] if len(sys.argv) > 2 else "window_model").resolve()
ACTIVE = {"image": None, "window": None}  # path of the model in use, per approach
_meta = {}  # path -> (mtime, info), so each file is unpickled once to read its name and kind


def model_info(path):
    mtime = path.stat().st_mtime
    if _meta.get(path, (None,))[0] != mtime:
        b = joblib.load(path)
        _meta[path] = (mtime, {"path": str(path), "name": b.get("name", path.stem),
                               "kind": "window" if "size" in b else "image", "trained": b.get("trained", "")})
    return _meta[path][1]


def list_models():
    found = []
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            found.append(model_info(p.resolve()))
        except Exception:
            pass  # not a model bundle
    return found


def use_model(kind, path):
    """Load a model file into the classifier."""
    global bundle, model, wbundle
    new = joblib.load(path)
    new.setdefault("name", Path(path).stem)
    if kind == "window":
        wbundle = new
    else:
        bundle, model = new, new["model"]
    ACTIVE[kind] = str(path)


# Both models are optional, so a fresh clone with an empty models/ folder still starts.
bundle = model = wbundle = None
for kind, path in (("image", MODEL_PATH), ("window", WINDOW_PATH)):
    if path.exists():
        use_model(kind, path)
    else:
        print(f"No {kind} model at {path}; train one on the Train & validate page.")
N_FEATURES = len(BANDS) * 5 + 4 * 2  # 5 statistics per band + mean/std of 4 indices

# Conventional land-cover hues (water blue, forest green, ...), labelled everywhere, never color alone.
COLORS = {
    "AnnualCrop": "#eda100", "Forest": "#008300", "HerbaceousVegetation": "#8fbf3f",
    "Highway": "#e34948", "Industrial": "#6250d6", "Pasture": "#c9d96b",
    "PermanentCrop": "#eb6834", "Residential": "#e87ba4", "River": "#1baf7a", "SeaLake": "#2a78d6",
}

CLASSES = {
    "AnnualCrop": "Fields of crops planted and harvested each year.",
    "Forest": "Areas densely covered by trees.",
    "HerbaceousVegetation": "Natural low vegetation such as grass and shrubs.",
    "Highway": "Major roads crossing the image.",
    "Industrial": "Factories, warehouses and large flat roofs.",
    "Pasture": "Grassland used for grazing.",
    "PermanentCrop": "Orchards, vineyards and other long-lived crops.",
    "Residential": "Houses and residential streets.",
    "River": "Rivers and their banks.",
    "SeaLake": "Open water: sea or lakes.",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # EuroSAT tiles are ~110 KB


def read_image(data):
    """Bytes -> (H, W, 13) array, or raise ValueError with a message for the user."""
    try:
        img = tifffile.imread(io.BytesIO(data))
    except Exception:
        raise ValueError("The file is not a readable TIFF. Upload a .tif from EuroSAT_MS.")
    if img.ndim == 3 and img.shape[0] == len(BANDS) and img.shape[-1] != len(BANDS):
        img = np.moveaxis(img, 0, -1)  # bands-first TIFF
    if img.ndim != 3 or img.shape[-1] != len(BANDS):
        raise ValueError(f"Expected a {len(BANDS)}-band Sentinel-2 image, got shape {img.shape}. "
                         "Ordinary RGB photos can't be classified: the model needs all 13 bands.")
    return img


def preview(img):
    """True-color (B04, B03, B02) PNG as a data URI, contrast-stretched and enlarged."""
    rgb = img[..., [B["B04"], B["B03"], B["B02"]]].astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = (np.clip((rgb - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((256, 256), Image.NEAREST).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def window_result(img):
    """Classify every window, then give each step x step block the average of the windows covering it.

    16 px windows every 8 px -> 7x7 windows, but 8x8 blocks of 8 px that tile the whole image.
    """
    size, step = wbundle["size"], wbundle["step"]
    H, W = img.shape[:2]
    if H < size or W < size:
        return {"error": f"Image is smaller than one {size}×{size} px window."}
    w = windows(img, size, step)
    rows, cols = w.shape[:2]
    wm = wbundle["model"]
    P = wm.predict_proba(features_batch(w.reshape(rows * cols, size * size, -1))[wbundle["features"]])
    P = P.reshape(rows, cols, -1)

    k = size // step  # blocks per window side (2): each window covers k x k blocks
    total = np.zeros((rows + k - 1, cols + k - 1, P.shape[-1]))
    n = np.zeros(total.shape[:2])
    for dy in range(k):
        for dx in range(k):
            total[dy:dy + rows, dx:dx + cols] += P
            n[dy:dy + rows, dx:dx + cols] += 1
    B = total / n[..., None]  # (blocks_y, blocks_x, classes)
    labels = wm.classes_[B.argmax(-1)]
    conf = B.max(-1)
    cells = [{"cls": labels[r, c], "p": conf[r, c], "n": int(n[r, c]), "row": r, "col": c,
              "top": r * step / H * 100, "left": c * step / W * 100,
              "h": step / H * 100, "w": step / W * 100}
             for r in range(labels.shape[0]) for c in range(labels.shape[1])]
    mean = P.reshape(-1, P.shape[-1]).mean(0)
    return {"cells": cells, "rows": labels.shape[0], "cols": labels.shape[1], "n_windows": rows * cols,
            "grid": labels.tolist(),
            "counts": list(pd.Series(labels.ravel()).value_counts().items()),
            "vote": wm.classes_[mean.argmax()], "vote_p": mean.max()}


def page(result=None, error=None, wres=None):
    return render_template_string(PAGE, result=result, error=error, wres=wres, b=bundle, w=wbundle,
                                  classes=CLASSES, colors=COLORS, n_features=N_FEATURES,
                                  models=list_models(), active=ACTIVE, models_dir=MODELS_DIR)


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, wres = None, None, None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Choose a file first."
        else:
            try:
                img = read_image(file.read())
                result = {"name": file.filename, "shape": img.shape, "preview": preview(img)}
                if bundle:
                    X = pd.DataFrame([features(img)])[bundle["features"]]
                    probs = sorted(zip(model.classes_, model.predict_proba(X)[0]), key=lambda p: -p[1])
                    result |= {"label": probs[0][0], "conf": probs[0][1], "probs": probs}
                wres = window_result(img) if wbundle else None
            except ValueError as e:
                error = str(e)
    return page(result, error, wres)


@app.errorhandler(413)
def too_large(_):
    return page(error="File too large (max 20 MB)."), 413


# ---------- Train & validate page ----------

HYPER = {  # name: (label, default, what is allowed)
    "n_estimators": ("Number of trees", "100, 200", "integer 1–2000"),
    "max_depth": ("Max tree depth", "None, 40", "integer ≥ 1, or None = unlimited"),
    "max_features": ("Features tried per split", "sqrt", "sqrt, log2, None = all, or a share 0–1"),
    "min_samples_split": ("Min samples to split a node", "2", "integer ≥ 2"),
    "min_samples_leaf": ("Min samples per leaf", "1, 2, 4", "integer ≥ 1"),
}


def parse_value(name, s):
    if s.lower() == "none" and name in ("max_depth", "max_features"):
        return None
    if name == "max_features":
        if s in ("sqrt", "log2"):
            return s
        v = float(s)
        if not 0 < v <= 1:
            raise ValueError
        return v
    v = int(s)
    lo, hi = {"n_estimators": (1, 2000), "min_samples_split": (2, 10**6)}.get(name, (1, 10**6))
    if not lo <= v <= hi:
        raise ValueError
    return v


def number(form, name, lo, hi, label):
    try:
        v = float(form.get(name, ""))
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"{label} must be between {lo:g} and {hi:g}.")
    return v


def parse_form(form):
    """Validate the training form -> job config, or raise ValueError with a message for the user."""
    mode = form.get("mode")
    approach = form.get("approach")
    if mode not in ("train", "validate", "both") or approach not in ("image", "window"):
        raise ValueError("Choose an approach and a mode.")
    try:
        root = find_root(form.get("root", "").strip() or data_root())
    except FileNotFoundError as e:
        raise ValueError(str(e))
    grid = {}
    for name, (label, _, allowed) in HYPER.items():
        parts = [p.strip() for p in form.get(name, "").split(",") if p.strip()]
        try:
            values = list(dict.fromkeys(parse_value(name, p) for p in parts))  # drop duplicates
        except ValueError:
            raise ValueError(f"{label}: use {allowed}.")
        if not values:
            raise ValueError(f"{label}: enter a value.")
        if mode == "train" and len(values) > 1:
            raise ValueError(f"{label}: train only needs exactly one value.")
        grid[name] = values
    name = form.get("model_name", "").strip() or time.strftime(f"{approach}_model_%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9](?:[\w\-. ]{0,78}[\w\-])?", name):
        raise ValueError("Model name: up to 80 letters, digits, spaces, - _ . (start with a letter or digit).")

    if mode != "validate" and (MODELS_DIR / f"{name}.joblib").exists():
        raise ValueError(f"{name}.joblib already exists in models/; choose another name.")
    return {"mode": mode, "approach": approach, "root": root, "grid": grid,
            "name": name,
            "frac": number(form, "data_pct", 1, 100, "Data share") / 100,
            "test_size": number(form, "test_pct", 5, 50, "Test share") / 100,
            "folds": int(number(form, "folds", 2, 10, "Folds"))}


@app.route("/train")
def train_page():
    return render_template_string(TRAIN_PAGE, hyper=HYPER, root=data_root(), out_dir=MODELS_DIR,
                                  max_candidates=trainer.MAX_CANDIDATES)


@app.post("/train/start")
def train_start():
    try:
        cfg = parse_form(request.form)
        if not trainer.start(cfg):
            return jsonify(error="A job is already running."), 409
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True)


@app.post("/settings/data_dir")
def set_data_dir():
    """Training page "Save as default": remember the dataset folder for the app and the scripts."""
    try:
        root = find_root(request.form.get("root", "").strip())
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 400
    save_data_root(root)
    return jsonify(ok=True, root=str(root))


@app.get("/train/status")
def train_status():
    return jsonify({k: v for k, v in trainer.STATE.items() if k != "log"} |
                   {"log": trainer.STATE.get("log", [])[-300:]})


def known_model(path):
    """Info of a listed model file, or None: only files from the scanned folders are ever unpickled."""
    return next((m for m in list_models() if m["path"] == path), None)


@app.post("/train/use")
def train_use():
    """Swap a model trained on this page into the classifier."""
    info = known_model(str(Path(request.form.get("path", "")).resolve()))
    if not info:
        return jsonify(error="Unknown model file."), 400
    use_model(info["kind"], info["path"])
    return jsonify(ok=True)


@app.post("/models/select")
def models_select():
    """Classify page model lists: load the chosen whole-image and window models."""
    for kind in ("image", "window"):
        path = request.form.get(kind)
        if path and path != ACTIVE[kind]:
            info = known_model(path)
            if not info or info["kind"] != kind:
                return page(error="Unknown model file."), 400
            use_model(kind, path)
    return redirect("/")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EuroSAT Classifier</title>
<style>
:root{--bg:#f6f6f4;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e5e4e0;--accent:#2a78d6;--bar:#cde2fb;--err:#e34948}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--accent:#3987e5;--bar:#184f95;--err:#e66767}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px;display:grid;gap:16px;grid-template-columns:1fr 1fr}
header,.wide{grid-column:1/-1}h1{margin:0;font-size:24px}h2{margin:0 0 12px;font-size:17px}
.sub{color:var(--ink2);margin:4px 0 0}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
form{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 16px;font:inherit;cursor:pointer}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.err{color:var(--err);margin:10px 0 0}.res{display:flex;gap:18px;flex-wrap:wrap}
.res img{width:192px;height:192px;border-radius:6px;image-rendering:pixelated}
.big{font-size:26px;font-weight:600;margin:0}.muted{color:var(--ink2)}
.row{display:grid;grid-template-columns:170px 1fr 52px;gap:8px;align-items:center;font-size:13px;margin:3px 0}
.track{background:var(--line);border-radius:4px;height:10px}.fill{background:var(--accent);height:10px;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:14px}td{padding:5px 0;border-bottom:1px solid var(--line)}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:14px}dt{font-weight:600}dd{margin:0;color:var(--ink2)}
.map{position:relative;width:256px;height:256px;flex:none}.map img{width:256px;height:256px}
.cell{position:absolute;opacity:.6;box-shadow:inset 0 0 0 1px rgba(255,255,255,.7)}
.cell:hover,.cell:focus{opacity:.95;z-index:2;outline:2px solid var(--ink)}
.cell:hover::after,.cell:focus::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 6px);left:50%;
transform:translateX(-50%);white-space:pre;background:var(--ink);color:var(--card);font-size:12px;line-height:1.4;
padding:5px 8px;border-radius:6px;pointer-events:none}
#ov:not(:checked)~.res .cell{display:none}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.grid td{padding:2px 4px;font-size:11px;text-align:center;border:1px solid var(--line)}
code{font-size:13px;background:var(--line);padding:1px 5px;border-radius:4px}
select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);max-width:100%}
select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
nav{display:flex;gap:6px;margin-bottom:10px}nav a{color:var(--ink2);text-decoration:none;padding:4px 10px;border-radius:6px;border:1px solid var(--line)}
nav a[aria-current]{color:var(--ink);background:var(--card);font-weight:600}nav a:focus-visible{outline:2px solid var(--accent)}
@media (max-width:760px){main{grid-template-columns:1fr}.row{grid-template-columns:150px 1fr 48px}}
</style></head><body><main>
<header><nav><a href="/" aria-current="page">Classify</a><a href="/train">Train &amp; validate</a></nav>
<h1>EuroSAT land-cover classifier</h1>
<p class="sub">Random Forest on Sentinel-2 multispectral image patches, 10 land-cover classes.</p></header>

<section class="card wide"><h2>Models</h2>
<form method="post" action="/models/select" class="models">
<label for="m_image">Whole-image model</label>
<select id="m_image" name="image" onchange="this.form.submit()">
{% if not active.image %}<option value="" selected>none</option>{% endif %}
{% for m in models if m.kind == "image" %}<option value="{{ m.path }}"{% if m.path == active.image %} selected{% endif %}>{{ m.name }}{% if m.trained %} · {{ m.trained }}{% endif %}</option>{% endfor %}
</select>
<label for="m_window">Window model</label>
<select id="m_window" name="window" onchange="this.form.submit()">
{% if not active.window %}<option value="" selected>none</option>{% endif %}
{% for m in models if m.kind == "window" %}<option value="{{ m.path }}"{% if m.path == active.window %} selected{% endif %}>{{ m.name }}{% if m.trained %} · {{ m.trained }}{% endif %}</option>{% endfor %}
</select>
<noscript><button type="submit">Use</button></noscript>
</form>
<p class="muted" style="font-size:12px;margin:8px 0 0">Lists the models in <code>{{ models_dir }}</code>.
Train new ones on the <a href="/train" style="color:var(--accent)">Train &amp; validate</a> page.</p>
</section>

<section class="card wide"><h2>1 · Whole-image classification{% if b %} <span class="muted" style="font-weight:400;font-size:14px">· model: {{ b.name }}</span>{% endif %}</h2>
{% if not b and not w %}<p class="err">No models in <code>{{ models_dir }}</code> yet. Train one on the
<a href="/train" style="color:var(--accent)">Train &amp; validate</a> page, or with the scripts (see README).</p>{% endif %}
<form method="post" enctype="multipart/form-data">
<label for="image" class="muted">EuroSAT_MS GeoTIFF (13 bands):</label>
<input id="image" type="file" name="image" accept=".tif,.tiff" required>
<button type="submit">Predict</button></form>
{% if error %}<p class="err" role="alert">{{ error }}</p>{% endif %}
{% if result %}<div class="res" style="margin-top:16px">
<figure style="margin:0"><img src="{{ result.preview }}" alt="True-color preview of {{ result.name }}">
<figcaption class="muted" style="font-size:12px">{{ result.name }} · {{ result.shape[0] }}×{{ result.shape[1] }} px</figcaption></figure>
<div style="flex:1;min-width:260px">
{% if not result.probs %}<p class="muted">No whole-image model loaded: choose or train one.</p>{% else %}
<p class="muted" style="margin:0">Predicted class</p>
<p class="big">{{ result.label }}</p>
<p class="muted" style="margin:2px 0 12px">{{ "%.0f"|format(result.conf*100) }}% of trees agree · {{ classes.get(result.label, "") }}</p>
{% for c, p in result.probs %}<div class="row"><span>{{ c }}</span>
<div class="track" role="img" aria-label="{{ c }} {{ '%.1f'|format(p*100) }}%"><div class="fill" style="width:{{ p*100 }}%"></div></div>
<span style="text-align:right">{{ "%.1f"|format(p*100) }}%</span></div>{% endfor %}{% endif %}
</div></div>{% endif %}
</section>

<section class="card"><h2>Whole-image model parameters</h2>
{% if not b %}<p class="muted">No whole-image model loaded.</p>{% else %}<table>
<tr><td>name</td><td>{{ b.name }}</td></tr>
{% for k, v in b.params.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}
<tr><td>features per image</td><td>{{ n_features }}</td></tr>
<tr><td>train / test images</td><td>{{ b.metrics.n_train }} / {{ b.metrics.n_test }}</td></tr>
{% if b.metrics.get("cv_macro_f1") is not none %}<tr><td>cross-validation macro-F1</td><td>{{ "%.3f"|format(b.metrics.cv_macro_f1) }}</td></tr>{% endif %}
<tr><td>test accuracy</td><td>{{ "%.1f"|format(b.metrics.test_accuracy*100) }}%</td></tr>
<tr><td>test macro-F1</td><td>{{ "%.3f"|format(b.metrics.test_macro_f1) }}</td></tr>
<tr><td>trained</td><td>{{ b.trained }}</td></tr></table>{% endif %}</section>

<section class="card"><h2>How it works</h2>
<p style="margin-top:0">Data: <a href="https://zenodo.org/records/7711810" style="color:var(--accent)">EuroSAT</a>,
27,000 Sentinel-2 patches of 64×64 px (10 m/px) with all 13 spectral bands, from 34 European countries.</p>
<p>Each image is summarised into {{ n_features }} numbers: mean, standard deviation and 10th/50th/90th percentile of every
band, plus the mean and spread of four spectral indices: NDVI (vegetation), NDWI (water), NDBI (built-up) and NDMI (moisture).
A Random Forest votes on those numbers; the bars show the share of trees backing each class.</p>
<p class="muted" style="margin-bottom:0">Only 13-band multispectral GeoTIFFs work: an ordinary RGB photo lacks the infrared bands the model relies on.</p></section>

<section class="card wide"><h2>2 · Window-based classification{% if w %} <span class="muted" style="font-weight:400;font-size:14px">· model: {{ w.name }}</span>{% endif %}</h2>
{% if not w %}<p class="err">Window model not loaded. Build it with <code>uv run prepare_windows.py</code>, <code>uv run grid_search_windows.py</code>,
<code>uv run train_windows.py</code> and restart the app.</p>
{% else %}
<p class="muted" style="margin-top:0">The same image cut into {{ w.size }}×{{ w.size }} px windows every {{ w.step }} px
({{ w.size * 10 }} m on the ground). A second Random Forest classifies each window from the same {{ n_features }} features.
The map is split into {{ w.step }}×{{ w.step }} px blocks; each block shows the average of the windows that cover it
(up to {{ (w.size // w.step) ** 2 }}). Hover or tab to a block for its class and confidence.</p>
{% if wres and wres.error %}<p class="err" role="alert">{{ wres.error }}</p>
{% elif wres %}
<input type="checkbox" id="ov" checked> <label for="ov">Show class map</label>
<div class="res" style="margin-top:10px">
<div class="map" role="img" aria-label="Window classification map, {{ wres.rows }} by {{ wres.cols }} blocks; see the table below">
<img src="{{ result.preview }}" alt="">
{% for c in wres.cells %}<div class="cell" style="top:{{ c.top }}%;left:{{ c.left }}%;width:{{ c.w }}%;height:{{ c.h }}%;background:{{ colors.get(c.cls, '#888') }}" tabindex="0"
data-tip="{{ c.cls }}: {{ '%.0f'|format(c.p*100) }}%&#10;block row {{ c.row + 1 }}, col {{ c.col + 1 }} · {{ c.n }} window{{ 's' if c.n != 1 }}"
aria-label="Block row {{ c.row + 1 }}, column {{ c.col + 1 }}: {{ c.cls }}, {{ '%.0f'|format(c.p*100) }}% confidence"></div>{% endfor %}
</div>
<div style="flex:1;min-width:260px">
<p class="muted" style="margin:0">Average over all {{ wres.n_windows }} windows</p>
<p class="big">{{ wres.vote }}</p>
<p class="muted" style="margin:2px 0 12px">{{ "%.0f"|format(wres.vote_p*100) }}% mean probability</p>
<table>{% for c, n in wres.counts %}<tr><td><span class="sw" style="background:{{ colors.get(c, '#888') }}"></span>{{ c }}</td>
<td>{{ n }} block{{ "s" if n != 1 }}</td></tr>{% endfor %}</table>
</div></div>
<details style="margin-top:12px"><summary>Block grid as a table</summary><div style="overflow-x:auto">
<table class="grid" style="width:auto;margin-top:8px">{% for r in wres.grid %}<tr>{% for c in r %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</table></div></details>
{% else %}<p class="muted">Upload an image above to see its windows.</p>{% endif %}
<h2 style="margin-top:18px">Window model parameters</h2><table>
<tr><td>name</td><td>{{ w.name }}</td></tr>
{% for k, v in w.params.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}
<tr><td>window / step</td><td>{{ w.size }} px / {{ w.step }} px</td></tr>
<tr><td>train windows ({{ w.per_image }} random per image) / test windows (all)</td><td>{{ w.metrics.n_train }} / {{ w.metrics.n_test }}</td></tr>
{% if w.metrics.get("cv_macro_f1") is not none %}<tr><td>cross-validation window macro-F1 (folds grouped by image)</td><td>{{ "%.3f"|format(w.metrics.cv_macro_f1) }}</td></tr>{% endif %}
<tr><td>test window accuracy (window label = its image's label)</td><td>{{ "%.1f"|format(w.metrics.window_accuracy*100) }}%</td></tr>
<tr><td>test window macro-F1</td><td>{{ "%.3f"|format(w.metrics.window_macro_f1) }}</td></tr>
<tr><td>test image accuracy (average of each image's windows)</td><td>{{ "%.1f"|format(w.metrics.image_vote_accuracy*100) }}%</td></tr>
<tr><td>trained</td><td>{{ w.trained }}</td></tr></table>
{% endif %}</section>

<section class="card wide"><h2>Classes</h2><dl>
{% for c, d in classes.items() %}<dt>{{ c }}</dt><dd>{{ d }}</dd>{% endfor %}</dl></section>
</main></body></html>"""

CSS = PAGE[PAGE.index("<style>"):PAGE.index("</style>") + len("</style>")]

TRAIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EuroSAT Training</title>""" + CSS + """
<style>
#f{display:block}
.opts{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}.opts label{border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
.opts input{margin-right:6px}.opts label:has(input:checked){border-color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,transparent)}
.field{display:grid;grid-template-columns:220px 1fr;gap:6px 14px;align-items:center;margin:8px 0}
.field input[type=text],.field input[type=number]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);width:100%;max-width:360px}
.hint{color:var(--ink2);font-size:12px;grid-column:2}
.bar{background:var(--line);border-radius:5px;height:12px;overflow:hidden}.bar>div{background:var(--accent);height:12px;width:0;transition:width .3s}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;max-height:260px;overflow:auto;font-size:12px;white-space:pre-wrap;margin:10px 0 0}
th{text-align:left;font-weight:600;padding:5px 0;border-bottom:1px solid var(--line)}
button:disabled{opacity:.5;cursor:default}
@media (max-width:760px){.field{grid-template-columns:1fr}.hint{grid-column:1}}
</style></head><body><main>
<header><nav><a href="/">Classify</a><a href="/train" aria-current="page">Train &amp; validate</a></nav>
<h1>Train &amp; validate</h1>
<p class="sub">Train new Random Forest models in the background. Models are saved to <code>models/</code>; existing models are never overwritten.</p></header>

<form id="f" class="card wide">
<h2>Setup</h2>
<p class="muted" style="margin:0 0 6px">Approach</p>
<div class="opts">
<label><input type="radio" name="approach" value="image" checked>Whole image</label>
<label><input type="radio" name="approach" value="window">Windows (16×16 px, 8 px step)</label></div>
<p class="muted" style="margin:0 0 6px">Mode</p>
<div class="opts">
<label><input type="radio" name="mode" value="train">Train only</label>
<label><input type="radio" name="mode" value="validate">Validate only</label>
<label><input type="radio" name="mode" value="both" checked>Validate, then train the best</label></div>
<div class="field"><label for="root">Dataset folder</label><input type="text" id="root" name="root" value="{{ root }}">
<span class="hint">EuroSAT_MS folder with one subfolder per class.
<button type="button" id="savedef" style="padding:2px 10px;font-size:12px;margin-left:6px">Save as default</button>
<span id="savedefmsg" role="status"></span><br>The default is used here and by the scripts when no folder is given.</span></div>
<div class="field"><label for="data_pct">Data used (%)</label><input type="number" id="data_pct" name="data_pct" value="30" min="1" max="100">
<span class="hint">Share of the images in every class to use at all</span></div>
<div class="field"><label for="test_pct">Test share (%)</label><input type="number" id="test_pct" name="test_pct" value="20" min="5" max="50">
<span class="hint" id="splitinfo"></span></div>
<div class="field trainonly"><label for="model_name">Model name</label><input type="text" id="model_name" name="model_name" maxlength="80" placeholder="e.g. windows_30pct_200trees">
<span class="hint">Shown in the Classify page and used as the file name in models/; empty = approach + date and time</span></div>
<div class="field trainonly"><span>Saved to</span><code>{{ out_dir }}</code></div>
<div class="field" id="foldsrow"><label for="folds">Cross-validation folds</label><input type="number" id="folds" name="folds" value="5" min="2" max="10">
<span class="hint">Training data is split into this many folds; each candidate is trained this many times</span></div>

<h2 style="margin-top:18px">Hyperparameters</h2>
<p class="muted" id="hyperhint" style="margin:0 0 6px"></p>
{% for name, (label, default, allowed) in hyper.items() %}
<div class="field"><label for="{{ name }}">{{ label }} <code>{{ name }}</code></label>
<input type="text" id="{{ name }}" name="{{ name }}" value="{{ default }}" class="hp">
<span class="hint">{{ allowed }}</span></div>{% endfor %}
<p id="cands" class="muted"></p>
<button type="submit" id="go">Start</button>
<p class="err" id="formerr" role="alert"></p>
</form>

<section class="card wide" aria-live="polite"><h2>Progress</h2>
<p id="stage" class="muted" style="margin:0 0 8px">No job has run since the app started.</p>
<div class="bar" role="progressbar" aria-label="Progress of the current stage" aria-valuemin="0" aria-valuemax="100" id="barwrap"><div id="bar"></div></div>
<p id="count" class="muted" style="font-size:12px;margin:4px 0 0"></p>
<pre id="log" hidden></pre></section>

<section class="card wide" id="cvcard" hidden><h2>Validation results</h2>
<p class="muted" style="margin-top:0">Mean macro-F1 over the folds (± standard deviation), best first.</p>
<table><thead><tr><th>Rank</th><th>Hyperparameters</th><th style="text-align:right">Macro-F1</th></tr></thead><tbody id="cvbody"></tbody></table></section>

<section class="card wide" id="modelcard" hidden><h2>Trained model</h2>
<table id="mtable"></table>
<p style="margin:12px 0 0"><button type="button" id="use">Use in classifier</button> <span id="usemsg" class="muted"></span></p></section>

<script>
const $ = id => document.getElementById(id);
const f = $("f");
let timer = null, modelPath = null;

function mode() { return f.mode.value; }
function refresh() {
  const t = Math.min(50, Math.max(5, +$("test_pct").value || 20));
  $("splitinfo").textContent = `Train ${100 - t}% · test ${t}% of the data used`;
  const train = mode() === "train";
  $("foldsrow").hidden = train;
  document.querySelectorAll(".trainonly").forEach(e => e.hidden = mode() === "validate");
  $("hyperhint").textContent = train ? "Train only: one value per hyperparameter."
    : "Validation: comma-separated values; every combination is cross-validated.";
  const n = [...document.querySelectorAll(".hp")].reduce((a, i) => a * Math.max(1, i.value.split(",").filter(s => s.trim()).length), 1);
  $("cands").textContent = train ? "" : `${n} combination(s) × ${$("folds").value} folds = ${n * $("folds").value} fits (limit {{ max_candidates }} combinations)`;
}
f.addEventListener("input", refresh); refresh();

f.addEventListener("submit", async e => {
  e.preventDefault(); $("formerr").textContent = "";
  const r = await fetch("/train/start", {method: "POST", body: new FormData(f)});
  const j = await r.json();
  if (!r.ok) { $("formerr").textContent = j.error; return; }
  poll();
});

function row(cells) { const tr = document.createElement("tr"); cells.forEach(c => { const td = document.createElement("td"); td.textContent = c; tr.append(td); }); return tr; }
const pct = v => (v * 100).toFixed(1) + "%";

async function poll() {
  clearTimeout(timer);
  const s = await (await fetch("/train/status")).json();
  $("go").disabled = !!s.running;
  if (s.stage) {
    $("stage").textContent = (s.error ? "Failed: " + s.error : s.stage);
    const p = s.total ? Math.round(100 * s.done / s.total) : 0;
    $("bar").style.width = p + "%"; $("barwrap").setAttribute("aria-valuenow", p);
    $("count").textContent = s.running ? `${s.done} / ${s.total}` : "";
    $("log").hidden = false; $("log").textContent = (s.log || []).join("\\n"); $("log").scrollTop = 1e9;
  }
  $("cvcard").hidden = !s.cv;
  if (s.cv) $("cvbody").replaceChildren(...s.cv.map(r => row([r.rank, r.params, `${r.mean.toFixed(4)} ± ${r.std.toFixed(4)}`])));
  $("modelcard").hidden = !s.model;
  if (s.model) {
    const m = s.model.metrics, rows = [["Name", s.model.name],
      ["Approach", s.model.approach === "image" ? "Whole image" : "Windows"],
      ...Object.entries(s.model.params).map(([k, v]) => [k, v === null ? "None" : v])];
    if (m.cv_macro_f1 != null) rows.push(["Cross-validation macro-F1", m.cv_macro_f1.toFixed(4)]);
    if ("test_accuracy" in m) rows.push(["Test accuracy", pct(m.test_accuracy)], ["Test macro-F1", m.test_macro_f1.toFixed(4)]);
    else rows.push(["Test window accuracy", pct(m.window_accuracy)], ["Test window macro-F1", m.window_macro_f1.toFixed(4)],
                   ["Test image accuracy (averaged windows)", pct(m.image_vote_accuracy)]);
    rows.push(["Train / test rows", `${m.n_train} / ${m.n_test}`], ["Saved to", s.model.path]);
    $("mtable").replaceChildren(...rows.map(row));
    if (modelPath !== s.model.path) { modelPath = s.model.path; $("usemsg").textContent = ""; }
  }
  if (s.running) timer = setTimeout(poll, 1000);
}

$("savedef").addEventListener("click", async () => {
  const r = await fetch("/settings/data_dir", {method: "POST", body: new URLSearchParams({root: $("root").value})});
  const j = await r.json();
  if (r.ok) $("root").value = j.root;
  $("savedefmsg").textContent = r.ok ? " Saved as default." : " " + j.error;
});

$("use").addEventListener("click", async () => {
  const r = await fetch("/train/use", {method: "POST", body: new URLSearchParams({path: modelPath})});
  $("usemsg").textContent = r.ok ? "Loaded. The Classify page now uses this model." : (await r.json()).error;
});
poll();
</script>
</main></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
