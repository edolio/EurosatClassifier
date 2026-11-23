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
COLORS = {  # map conventions: roads red, built-up pink, industry grey, water blue
    "AnnualCrop": "#e2ac35", "Forest": "#2f9e4f", "HerbaceousVegetation": "#97c45a",
    "Highway": "#e5564d", "Industrial": "#9aa4ae", "Pasture": "#cfdc7a",
    "PermanentCrop": "#ec8a4b", "Residential": "#e98bb5", "River": "#3fb8c8", "SeaLake": "#4d8de6",
}
WATER = {"River", "SeaLake"}  # lettered in italic blue, as hydrography is on maps

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


def geo_info(data, shape):
    """Real collar figures from the GeoTIFF tags (CRS, corner coordinates, pixel size), or None."""
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as t:
            tags = t.pages[0].tags
            tie, scale = tags["ModelTiepointTag"].value, tags["ModelPixelScaleTag"].value
            crs = str(tags["GeoAsciiParamsTag"].value).split("|")[0] if "GeoAsciiParamsTag" in tags else ""
    except Exception:  # not georeferenced: the collar shows nominal Sentinel-2 values only
        return None
    h, w = shape[:2]
    return {"crs": crs.replace("WGS 84 / ", "") or "projected", "w": round(tie[3]), "e": round(tie[3] + w * scale[0]),
            "n": round(tie[4]), "s": round(tie[4] - h * scale[1]), "res": round(scale[0], 2)}


@app.template_filter("sp")
def thin_space_thousands(n):
    return f"{n:,}".replace(",", " ")  # thin space, as on map collars


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
    step = wbundle["step"] if wbundle else 8
    scale = None
    if result:  # four equal round-number segments that fit the tile width, drawn to scale
        metres = result["shape"][1] * (result["geo"]["res"] if result.get("geo") else 10)
        seg = max((s for s in (5, 10, 20, 25, 50, 100, 150, 200, 250, 500) if 4 * s <= metres), default=metres / 4)
        scale = {"seg": seg, "total": 4 * seg, "frac": 4 * seg / metres}
    return render_template_string(PAGE, page="classify", result=result, error=error, wres=wres, b=bundle, w=wbundle,
                                  classes=CLASSES, colors=COLORS, water=WATER, n_features=N_FEATURES,
                                  models=list_models(), active=ACTIVE, models_dir=MODELS_DIR,
                                  geo=result.get("geo") if result else None,
                                  ticks=max(1, result["shape"][1] // step) if result else 8, scale=scale)


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, wres = None, None, None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Choose a file first."
        else:
            try:
                data = file.read()
                img = read_image(data)
                result = {"name": file.filename, "shape": img.shape, "preview": preview(img),
                          "geo": geo_info(data, img.shape)}
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
    return render_template_string(TRAIN_PAGE, page="train", hyper=HYPER, root=data_root(), out_dir=MODELS_DIR,
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


CSS = """<style>
@font-face{font-family:Archivo;font-style:normal;font-weight:100 900;font-display:swap;src:url(/static/fonts/archivo-latin.woff2) format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:Archivo;font-style:normal;font-weight:100 900;font-display:swap;src:url(/static/fonts/archivo-latin-ext.woff2) format("woff2");unicode-range:U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+1E00-1E9F,U+2020,U+20A0-20AB,U+20AD-20C4,U+2113,U+2C60-2C7F,U+A720-A7FF}
@font-face{font-family:Archivo;font-style:italic;font-weight:100 900;font-display:swap;src:url(/static/fonts/archivo-italic-latin.woff2) format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+2000-206F}
@font-face{font-family:"Archivo Narrow";font-style:normal;font-weight:400 700;font-display:swap;src:url(/static/fonts/archivo-narrow-latin.woff2) format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+2000-206F}
@font-face{font-family:"Archivo Narrow";font-style:italic;font-weight:400 700;font-display:swap;src:url(/static/fonts/archivo-narrow-italic-latin.woff2) format("woff2");unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+2000-206F}
:root{color-scheme:dark;--ground:#0e1513;--sheet:#141d1a;--raise:#1a2522;--rule:#2b3a35;--rule2:#22302b;
--ink:#dbe4de;--collar:#93a69d;--purple:#b994f5;--purple-ink:#120d1c;--water:#5cb8ec;--alert:#ff8a7a;
--narrow:"Archivo Narrow",Archivo,sans-serif;--s-small:13px;--s-body:15px}
*{box-sizing:border-box}html{background:var(--ground)}
body{margin:0;background:var(--ground);color:var(--ink);font:400 var(--s-body)/1.5 Archivo,system-ui,sans-serif;
font-variant-numeric:tabular-nums;caret-color:var(--purple);accent-color:var(--purple);scrollbar-color:var(--rule) var(--ground)}
::selection{background:var(--purple);color:var(--purple-ink)}
a{color:var(--ink);text-underline-offset:3px;text-decoration-color:var(--collar)}a:hover{text-decoration-color:var(--purple)}
:focus-visible{outline:2px solid var(--purple);outline-offset:2px}
h1,h2,h3,p{margin:0}
.caps{font:600 13px/1.3 Archivo,sans-serif;letter-spacing:.09em;text-transform:uppercase;color:var(--collar)}
.muted{color:var(--collar)}.small{font-size:var(--s-small)}
.water{color:var(--water);font-style:italic}
code{font:13px/1.4 ui-monospace,Consolas,monospace;color:var(--ink);background:var(--raise);padding:1px 6px;border-radius:3px}
/* top bar */
.top{display:flex;align-items:center;gap:32px;padding:0 40px;min-height:60px;border-bottom:1px solid var(--rule);flex-wrap:wrap}
.mark{font:700 15px/1 var(--narrow);letter-spacing:.06em;text-transform:uppercase;text-decoration:none;color:var(--ink);padding:20px 0}
.mark span{color:var(--purple);margin:0 .35em}
.top nav{display:flex;gap:24px;align-self:stretch}
.top nav a{display:flex;align-items:center;color:var(--collar);text-decoration:none;font-weight:500;box-shadow:inset 0 0 0 transparent;transition:box-shadow .2s,color .2s}
.top nav a:hover{color:var(--ink)}.top nav a[aria-current]{color:var(--ink);box-shadow:inset 0 -2px 0 var(--purple)}
.models{margin-left:auto;display:flex;gap:18px;flex-wrap:wrap;padding:10px 0}
.models label{display:flex;align-items:center;gap:8px}
select,input[type=text],input[type=number]{font:inherit;color:var(--ink);background:var(--sheet);border:1px solid var(--rule);border-radius:4px;padding:7px 10px}
select{padding-right:30px;appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--collar) 50%),linear-gradient(135deg,var(--collar) 50%,transparent 50%);
background-position:calc(100% - 15px) 50%,calc(100% - 10px) 50%;background-size:5px 5px;background-repeat:no-repeat;max-width:260px;text-overflow:ellipsis}
select:hover,input[type=text]:hover,input[type=number]:hover{border-color:var(--collar)}
input::placeholder{color:#74877e}
/* buttons */
.btn{display:inline-flex;align-items:center;gap:8px;font:600 var(--s-body)/1 Archivo,sans-serif;border:1px solid var(--purple);background:var(--purple);
color:var(--purple-ink);padding:12px 20px;border-radius:4px;cursor:pointer;transition:background .15s,transform .15s}
.btn:hover{background:#cbb0f8}.btn:active{transform:translateY(1px)}
.btn.ghost{background:transparent;color:var(--ink);border-color:var(--rule)}.btn.ghost:hover{border-color:var(--collar);background:var(--raise)}
.btn.sm{padding:6px 12px;font-size:var(--s-small)}
.btn:disabled{opacity:.45;cursor:default;transform:none}
.err{color:var(--alert)}
/* tables */
table{border-collapse:collapse;width:100%;font-size:var(--s-body)}
td,th{padding:6px 0;border-bottom:1px solid var(--rule2);text-align:left;vertical-align:baseline}
th{font:600 13px/1.3 Archivo,sans-serif;letter-spacing:.09em;text-transform:uppercase;color:var(--collar)}
td.num,th.num{text-align:right;padding-left:12px;white-space:nowrap}
@media (prefers-reduced-motion:reduce){*,*::after{animation:none!important;transition:none!important}}
</style>"""

TOP = """<header class="top">
<a class="mark" href="/">EuroSAT<span>/</span>land cover</a>
<nav aria-label="Pages"><a href="/"{% if page == 'classify' %} aria-current="page"{% endif %}>Classify</a><a href="/train"{% if page == 'train' %} aria-current="page"{% endif %}>Train &amp; validate</a></nav>
{% if page == 'classify' %}<form method="post" action="/models/select" class="models" aria-label="Models in use">
<label class="caps" for="m_image">Whole image</label>
<select id="m_image" name="image" onchange="this.form.submit()">
{% if not active.image %}<option value="" selected>none</option>{% endif %}
{% for m in models if m.kind == "image" %}<option value="{{ m.path }}"{% if m.path == active.image %} selected{% endif %}>{{ m.name }}{% if m.trained %} · {{ m.trained }}{% endif %}</option>{% endfor %}
</select>
<label class="caps" for="m_window">Windows</label>
<select id="m_window" name="window" onchange="this.form.submit()">
{% if not active.window %}<option value="" selected>none</option>{% endif %}
{% for m in models if m.kind == "window" %}<option value="{{ m.path }}"{% if m.path == active.window %} selected{% endif %}>{{ m.name }}{% if m.trained %} · {{ m.trained }}{% endif %}</option>{% endfor %}
</select>
<noscript><button class="btn sm" type="submit">Use</button></noscript>
</form>{% endif %}
</header>"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EuroSAT land cover · Classify</title>""" + CSS + """<style>
.classify{display:grid;grid-template-columns:minmax(0,690px) minmax(0,1fr);border-bottom:1px solid var(--rule)}
.sheet{padding:36px 40px 40px;border-right:1px solid var(--rule)}
.sheet h1{font:700 26px/1.15 var(--narrow);margin:4px 0 22px;overflow-wrap:anywhere}
.collar{position:relative;padding:22px 128px 22px 22px}
.coord{position:absolute;font-size:var(--s-small);line-height:1;color:var(--collar);white-space:nowrap}
.coord.tl{left:22px;top:0}.coord.tr{left:calc(22px + var(--t));top:0;transform:translateX(-100%)}.coord.rt{left:calc(42px + var(--t));top:16px}.coord.rb{left:calc(42px + var(--t));top:calc(16px + var(--t))}
.collar{container-type:inline-size;--t:min(448px,100cqw)}.tile{position:relative;width:var(--t);aspect-ratio:1;outline:1px solid var(--ink);box-shadow:0 0 0 4px var(--ground),0 0 0 5px var(--collar);background:var(--sheet)}
.tile img{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}
.tick{position:absolute;background:var(--collar)}
.tick.x{top:-8px;width:1px;height:6px}.tick.x.b{top:auto;bottom:-8px}.tick.y{left:-8px;width:6px;height:1px}.tick.y.r{left:auto;right:-8px}
.north{position:absolute;left:calc(46px + var(--t));top:calc(22px + var(--t) / 2 - 20px);width:20px;color:var(--collar)}
.empty{display:grid;place-content:center;text-align:center;gap:6px;padding:24px;color:var(--collar);
background-image:linear-gradient(var(--rule2) 1px,transparent 1px),linear-gradient(90deg,var(--rule2) 1px,transparent 1px);background-size:12.5% 12.5%}
.empty b{font-weight:600;color:var(--ink)}
.scale{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;margin-top:18px;font-size:var(--s-small);color:var(--collar)}
.scale .bars{display:grid;grid-template-columns:repeat(4,1fr);width:calc(var(--t) * var(--f))}
.scale i{display:block;height:6px;border:1px solid var(--ink)}.scale i:nth-child(odd){background:var(--ink)}
.imprint{margin:4px 0 0 22px;font-size:var(--s-small);color:var(--collar);max-width:var(--t)}.imprint b{color:var(--purple);font-weight:600}
.upload{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:30px 0 0}
.upload input[type=file]{position:absolute;width:1px;height:1px;opacity:0}
.upload input[type=file]:focus-visible+label{outline:2px solid var(--purple);outline-offset:2px}
.upload .err{flex-basis:100%;margin-top:4px}
.toggle{display:flex;align-items:center;gap:8px;margin-top:16px;font-size:var(--s-small);color:var(--collar);cursor:pointer;width:max-content}
.cell{position:absolute;opacity:.58;box-shadow:inset 0 0 0 .5px rgba(14,21,19,.75);transition:opacity .2s;animation:survey .9s cubic-bezier(.16,1,.3,1) both;animation-delay:calc(var(--i) * 9ms)}
@keyframes survey{from{opacity:.3}}
.cell:hover,.cell:focus-visible{opacity:.92;z-index:3;outline:2px solid var(--ink);outline-offset:-1px}
.tile.dim .cell{opacity:.14}.tile.dim .cell.hl{opacity:.92}
.cell:hover::after,.cell:focus-visible::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);
white-space:pre;background:var(--ink);color:var(--ground);font-size:var(--s-small);line-height:1.45;padding:6px 9px;border-radius:3px;
box-shadow:0 6px 18px rgba(0,0,0,.45);pointer-events:none}
body:has(#ov:not(:checked)) .cell{display:none}
.answers{padding:36px 48px 40px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px 48px;align-content:start}
.answer h2{margin-bottom:12px}.answer h2 span{color:var(--ink);letter-spacing:.06em}
.verdict{font:700 clamp(40px,4.6vw,60px)/.95 var(--narrow);color:var(--purple);margin:4px 0 10px;overflow-wrap:anywhere}
.verdict.water{color:var(--water);font-style:italic}
.conf{color:var(--collar);margin-bottom:20px;font-size:var(--s-small)}
.bar{width:100%;min-width:56px}.bar b{display:block;height:4px;background:var(--purple);border-radius:1px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:9px;vertical-align:-1px}
tr[data-cls]{cursor:default;transition:background .15s}tr.hl td{background:var(--raise)}
.note{grid-column:1/-1;border-top:1px solid var(--rule);padding-top:16px;color:var(--collar);max-width:75ch}
.note b{color:var(--ink);font-weight:600}
.intro{grid-column:1/-1;max-width:62ch}.intro h2{font:700 26px/1.15 var(--narrow);margin-bottom:12px}
.approach{border-top:1px solid var(--rule);padding-top:16px}.approach h3{font:600 var(--s-body)/1.3 Archivo,sans-serif;margin:0 0 6px}.approach h3 span{color:var(--collar);font-weight:400}
.approach p+p{margin-top:10px}
details{margin-top:18px;font-size:var(--s-small)}summary{cursor:pointer;color:var(--collar);list-style:none;display:flex;align-items:center;gap:8px}summary::-webkit-details-marker{display:none}
summary::before{content:"";width:0;height:0;border:4px solid transparent;border-left:6px solid currentColor;transition:transform .15s}details[open] summary::before{transform:rotate(90deg) translateX(2px)}summary:hover{color:var(--ink)}
.grid td{padding:3px 6px;border:1px solid var(--rule2);font-size:var(--s-small);text-align:center;white-space:nowrap}
.lower{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:40px 48px;padding:40px}
.lower h2{margin-bottom:14px}.lower p{max-width:65ch}.lower p+p{margin-top:12px;color:var(--collar)}
.lower td:first-child{color:var(--collar);padding-right:12px}
.legend{grid-column:1/-1}.legend dl{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:4px 40px;margin:0}
.legend div{padding:9px 0;border-bottom:1px solid var(--rule2)}
.legend dt{font-weight:600}.legend dd{margin:2px 0 0 21px;color:var(--collar);font-size:var(--s-small)}
@media (max-width:1180px){.classify{grid-template-columns:1fr}.sheet{border-right:0;border-bottom:1px solid var(--rule)}.lower{grid-template-columns:1fr 1fr}}
@media (max-width:760px){.top{padding:0 16px;gap:4px 24px}.models{margin-left:0;display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px 12px;width:100%}.models select{max-width:none;width:100%}
.sheet{display:flex;flex-direction:column}.sheet>h1{order:-2}.sheet:not(.has-result) .upload{order:-1;margin:0 0 20px}.sheet,.answers,.lower{padding:28px 16px}
.answers{grid-template-columns:1fr}.lower{grid-template-columns:1fr}.collar{padding-right:22px;padding-top:34px}
.collar{padding-right:22px}.has-result .imprint{order:3;margin-top:18px}.coord.rt,.coord.rb{display:none}.coord.tr{top:16px}.coord.tl{top:0}.north{display:none}}
</style></head><body>
""" + TOP + """
{% macro cname(c) %}{% if c in water %}<span class="water">{{ c }}</span>{% else %}{{ c }}{% endif %}{% endmacro %}
<main class="classify">
<section class="sheet{% if result %} has-result{% endif %}" aria-labelledby="sheet-title">
{% if result %}
<h1 id="sheet-title">{{ result.name }}</h1>
<div class="collar">
{% if geo %}<span class="coord tl">{{ geo.crs }} · E {{ geo.w|sp }}</span><span class="coord tr">E {{ geo.e|sp }}</span>
<span class="coord rt">N {{ geo.n|sp }}</span><span class="coord rb">N {{ geo.s|sp }}</span>{% endif %}
<div class="tile" id="tile">
<img src="{{ result.preview }}" alt="True-colour preview of {{ result.name }}">
{% for i in range(ticks + 1) %}{% set p = i * 100 / ticks %}<i class="tick x" style="left:{{ p }}%"></i><i class="tick x b" style="left:{{ p }}%"></i><i class="tick y" style="top:{{ p }}%"></i><i class="tick y r" style="top:{{ p }}%"></i>{% endfor %}
{% if wres and not wres.error %}{% for c in wres.cells %}<div class="cell" data-cls="{{ c.cls }}" style="--i:{{ loop.index0 }};top:{{ c.top }}%;left:{{ c.left }}%;width:{{ c.w }}%;height:{{ c.h }}%;background:{{ colors.get(c.cls, '#888') }}" tabindex="0"
data-tip="{{ c.cls }} · {{ '%.0f'|format(c.p*100) }}%&#10;block {{ c.row + 1 }}/{{ c.col + 1 }} · {{ c.n }} window{{ 's' if c.n != 1 }}"
aria-label="Block row {{ c.row + 1 }}, column {{ c.col + 1 }}: {{ c.cls }}, {{ '%.0f'|format(c.p*100) }}% confidence"></div>{% endfor %}{% endif %}
</div>
<div class="scale" role="img" aria-label="Scale bar, {{ scale.total|int }} metres" style="--f:{{ scale.frac }}"><span class="bars"><i></i><i></i><i></i><i></i></span><span>0 · {{ scale.seg|int }} · {{ (scale.seg * 2)|int }} · {{ (scale.seg * 3)|int }} · {{ scale.total|int }} m</span></div>
<svg class="north" viewBox="0 0 22 40" aria-hidden="true"><path d="M11 2 18 22 11 18 4 22Z" fill="currentColor"/><text x="11" y="36" text-anchor="middle" font-size="11" font-family="Archivo" fill="currentColor">N</text></svg>
</div>
<p class="imprint">Sentinel-2 · 13 bands · {{ result.shape[0] }} × {{ result.shape[1] }} px at {{ "%g"|format(geo.res if geo else 10) }} m{% if b %}<br><b>Overprinted by {{ b.name }}</b> · trained {{ b.trained }}{% endif %}</p>
{% if wres and not wres.error %}<label class="toggle"><input type="checkbox" id="ov" checked> Show class map</label>{% endif %}
{% else %}
<h1 id="sheet-title">Classify a tile</h1>
<div class="collar"><div class="tile empty"><b>No tile on the sheet</b><span class="small">Your tile appears here with its class map.</span></div></div>
{% endif %}
<form class="upload" method="post" enctype="multipart/form-data">
<input id="image" type="file" name="image" accept=".tif,.tiff" required onchange="this.form.submit()">
<label class="btn" for="image">{{ "Classify another tile" if result else "Choose a tile" }}</label>
<span class="muted small">13-band GeoTIFF, as in EuroSAT_MS</span>
<noscript><button class="btn ghost" type="submit">Classify</button></noscript>
{% if error %}<p class="err small" role="alert">{{ error }}</p>{% endif %}
</form>
</section>

<section class="answers" aria-label="Results">
{% if result %}
<div class="answer">
<h2 class="caps">1 · Whole image{% if b %} · <span>{{ b.name }}</span>{% endif %}</h2>
{% if not result.probs %}<p class="muted">No whole-image model loaded. Choose one above or train one on <a href="/train">Train &amp; validate</a>.</p>
{% else %}
<p class="verdict{% if result.label in water %} water{% endif %}">{{ result.label }}</p>
<p class="conf">{{ "%.0f"|format(result.conf*100) }}% of trees agree{% if result.probs|length > 1 %} · runner-up {{ cname(result.probs[1][0]) }} {{ "%.0f"|format(result.probs[1][1]*100) }}%{% endif %}</p>
<table><thead><tr><th>Class</th><th>Trees</th><th class="num">%</th></tr></thead><tbody>
{% for c, p in result.probs %}<tr><td>{{ cname(c) }}</td><td class="bar" role="img" aria-label="{{ '%.1f'|format(p*100) }}%"><b style="width:{{ p*100 }}%"></b></td><td class="num">{{ "%.1f"|format(p*100) }}</td></tr>{% endfor %}
</tbody></table>{% endif %}
</div>
<div class="answer">
<h2 class="caps">2 · Windows{% if w %} · <span>{{ w.name }}</span>{% endif %}</h2>
{% if not w %}<p class="muted">No window model loaded. Choose one above, or build one with <code>uv run prepare_windows.py</code>,
<code>uv run grid_search_windows.py</code> and <code>uv run train_windows.py</code>.</p>
{% elif wres.error %}<p class="err" role="alert">{{ wres.error }}</p>
{% else %}
<p class="verdict{% if wres.vote in water %} water{% endif %}">{{ wres.vote }}</p>
<p class="conf">{{ "%.0f"|format(wres.vote_p*100) }}% mean probability over {{ wres.n_windows }} windows of {{ w.size }} × {{ w.size }} px</p>
<table id="legend"><thead><tr><th>Class on the map</th><th class="num">Blocks</th></tr></thead><tbody>
{% for c, n in wres.counts %}<tr data-cls="{{ c }}" tabindex="0"><td><span class="sw" style="background:{{ colors.get(c, '#888') }}"></span>{{ cname(c) }}</td><td class="num">{{ n }}</td></tr>{% endfor %}
</tbody></table>
<details><summary>Block grid as a table</summary><div style="overflow-x:auto">
<table class="grid" style="width:auto;margin-top:8px">{% for r in wres.grid %}<tr>{% for c in r %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</table></div></details>
{% endif %}
</div>
{% if result.probs and wres and not wres.error %}<p class="note">{% if result.label == wres.vote %}Both approaches agree: <b>{{ result.label }}</b>.
{% else %}The approaches disagree: the whole tile reads as <b>{{ result.label }}</b>, while the averaged windows read as <b>{{ wres.vote }}</b>. The map shows where each class sits.{% endif %}</p>{% endif %}
{% else %}
<div class="intro"><h2>Two models read every tile</h2>
<p class="muted">One model names the land cover of the whole tile; the other maps classes inside it, window by window.</p></div>
<div class="approach"><h3>1 · Whole image <span>· {{ b.name if b else "no model loaded" }}</span></h3>
{% if b %}<p class="muted small">One class for the whole 64 × 64 px tile (640 m across), from {{ n_features }} band and index statistics.</p>
<p class="small">{{ "%.1f"|format(b.metrics.test_accuracy*100) }}% test accuracy.</p>{% else %}<p class="muted small">Train one on <a href="/train">Train &amp; validate</a>.</p>{% endif %}</div>
<div class="approach"><h3>2 · Windows <span>· {{ w.name if w else "no model loaded" }}</span></h3>
{% if w %}<p class="muted small">{{ w.size }} × {{ w.size }} px windows every {{ w.step }} px, averaged into an {{ w.step }} px class map.</p>
<p class="small">{{ "%.1f"|format(w.metrics.image_vote_accuracy*100) }}% of test tiles right when windows are averaged.</p>{% else %}<p class="muted small">Build one with the window scripts (see README).</p>{% endif %}</div>
{% endif %}
</section>
</main>

<section class="lower">
<div><h2 class="caps">Whole-image model</h2>
{% if not b %}<p class="muted">No whole-image model loaded.</p>{% else %}<table>
<tr><td>name</td><td class="num">{{ b.name }}</td></tr>
{% for k, v in b.params.items() %}<tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>{% endfor %}
<tr><td>features per image</td><td class="num">{{ n_features }}</td></tr>
<tr><td>train / test images</td><td class="num">{{ b.metrics.n_train }} / {{ b.metrics.n_test }}</td></tr>
{% if b.metrics.get("cv_macro_f1") is not none %}<tr><td>cross-validation macro-F1</td><td class="num">{{ "%.3f"|format(b.metrics.cv_macro_f1) }}</td></tr>{% endif %}
<tr><td>test accuracy</td><td class="num">{{ "%.1f"|format(b.metrics.test_accuracy*100) }}%</td></tr>
<tr><td>test macro-F1</td><td class="num">{{ "%.3f"|format(b.metrics.test_macro_f1) }}</td></tr>
<tr><td>trained</td><td class="num">{{ b.trained }}</td></tr></table>{% endif %}</div>
<div><h2 class="caps">Window model</h2>
{% if not w %}<p class="muted">No window model loaded.</p>{% else %}<table>
<tr><td>name</td><td class="num">{{ w.name }}</td></tr>
{% for k, v in w.params.items() %}<tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>{% endfor %}
<tr><td>window / step</td><td class="num">{{ w.size }} px / {{ w.step }} px</td></tr>
<tr><td>train windows ({{ w.per_image }} per image) / test windows</td><td class="num">{{ w.metrics.n_train }} / {{ w.metrics.n_test }}</td></tr>
{% if w.metrics.get("cv_macro_f1") is not none %}<tr><td>cross-validation window macro-F1</td><td class="num">{{ "%.3f"|format(w.metrics.cv_macro_f1) }}</td></tr>{% endif %}
<tr><td>test window accuracy</td><td class="num">{{ "%.1f"|format(w.metrics.window_accuracy*100) }}%</td></tr>
<tr><td>test window macro-F1</td><td class="num">{{ "%.3f"|format(w.metrics.window_macro_f1) }}</td></tr>
<tr><td>test tile accuracy, windows averaged</td><td class="num">{{ "%.1f"|format(w.metrics.image_vote_accuracy*100) }}%</td></tr>
<tr><td>trained</td><td class="num">{{ w.trained }}</td></tr></table>{% endif %}</div>
<div><h2 class="caps">How it works</h2>
<p>Data: <a href="https://zenodo.org/records/7711810">EuroSAT</a>, 27,000 Sentinel-2 tiles of 64 × 64 px at 10 m with all 13 spectral bands, from 34 European countries.</p>
<p>Each tile or window is summarised into {{ n_features }} numbers: mean, standard deviation and 10th, 50th and 90th percentile of every band, plus the mean and spread of NDVI, NDWI, NDBI and NDMI. A Random Forest votes on them.</p>
<p>Window labels come from their tile, so window accuracy is pessimistic. Models are read from <code>models/</code>; train new ones on <a href="/train">Train &amp; validate</a>.</p></div>
<div class="legend"><h2 class="caps">Legend</h2><dl>
{% for c, d in classes.items() %}<div><dt><span class="sw" style="background:{{ colors.get(c, '#888') }}"></span>{{ cname(c) }}</dt><dd>{{ d }}</dd></div>{% endfor %}</dl></div>
</section>
<script>
const tile = document.getElementById("tile"), legend = document.getElementById("legend");
function mark(cls) {
  if (!tile) return;
  tile.classList.toggle("dim", !!cls);
  tile.querySelectorAll(".cell").forEach(c => c.classList.toggle("hl", c.dataset.cls === cls));
  if (legend) legend.querySelectorAll("tr[data-cls]").forEach(r => r.classList.toggle("hl", r.dataset.cls === cls));
}
if (tile && legend) {
  tile.querySelectorAll(".cell").forEach(c => {
    const lightRow = () => legend.querySelectorAll("tr[data-cls]").forEach(r => r.classList.toggle("hl", r.dataset.cls === c.dataset.cls));
    ["mouseenter", "focus"].forEach(e => c.addEventListener(e, lightRow));
    ["mouseleave", "blur"].forEach(e => c.addEventListener(e, () => mark(null)));
  });
  legend.querySelectorAll("tr[data-cls]").forEach(r => {
    ["mouseenter", "focus"].forEach(e => r.addEventListener(e, () => mark(r.dataset.cls)));
    ["mouseleave", "blur"].forEach(e => r.addEventListener(e, () => mark(null)));
  });
}
</script>
</body></html>"""

TRAIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EuroSAT land cover · Train &amp; validate</title>""" + CSS + """<style>
.train{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,520px);align-items:start}
.form{padding:36px 40px 48px;border-right:1px solid var(--rule)}
.form h1{font:700 26px/1.15 var(--narrow);margin:4px 0 8px}.lede{color:var(--collar);max-width:62ch;margin-bottom:28px}
fieldset{border:0;border-top:1px solid var(--rule);margin:0;padding:18px 0 10px}
legend{float:left;width:100%;padding:0;margin-bottom:12px}legend+*{clear:both}
.opts{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.opts label{display:flex;align-items:center;gap:8px;border:1px solid var(--rule);border-radius:4px;padding:9px 14px;cursor:pointer;transition:border-color .15s,background .15s}
.opts label:hover{border-color:var(--collar)}
.opts label:has(input:checked){border-color:var(--purple);background:color-mix(in srgb,var(--purple) 13%,transparent)}
.opts label:has(input:focus-visible){outline:2px solid var(--purple);outline-offset:2px}
.field{display:grid;grid-template-columns:230px minmax(0,1fr);gap:4px 18px;align-items:center;padding:8px 0}
.field>label,.field>span:first-child{font-weight:500}
.field input[type=text],.field input[type=number]{width:100%;max-width:380px}
.field label code{margin-left:6px;font-size:var(--s-small);color:var(--collar)}
.hint{grid-column:2;color:var(--collar);font-size:var(--s-small)}
.savedef{display:flex;align-items:center;gap:10px;margin-top:8px}
.actions{position:sticky;bottom:0;z-index:2;display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:22px;padding:16px 0;border-top:1px solid var(--rule);background:var(--ground);box-shadow:0 -12px 24px rgba(14,21,19,.85)}
#cands{color:var(--collar);font-size:var(--s-small)}
.status{position:sticky;top:0;padding:36px 40px 48px;display:grid;gap:30px;max-height:100vh;overflow:auto}
.status h2{margin-bottom:12px}
.state{display:flex;align-items:center;gap:10px;font:600 13px/1.3 Archivo,sans-serif;letter-spacing:.09em;text-transform:uppercase;margin-bottom:8px}
.state span{padding:3px 0;color:var(--collar)}
[data-state=queued] .state span,[data-state=running] .state span,[data-state=done] .state span,[data-state=failed] .state span{padding:3px 8px;border-radius:2px}
[data-state=queued] .state span{color:var(--purple);outline:1px dashed var(--purple)}
[data-state=running] .state span,[data-state=done] .state span{background:var(--purple);color:var(--purple-ink)}
[data-state=failed] .state span{background:var(--alert);color:var(--ground)}
[data-state=failed] #stage{color:var(--alert)}
#stage{color:var(--collar);font-size:var(--s-small);overflow-wrap:anywhere}
.meter{height:6px;background:var(--raise);margin-top:14px;border-radius:1px;overflow:hidden}
.meter>div{height:100%;width:0;background:var(--purple);transition:width .5s cubic-bezier(.16,1,.3,1)}
#count{font-size:var(--s-small);color:var(--collar);margin-top:6px;min-height:1em}
pre{margin:14px 0 0;background:var(--sheet);border:1px solid var(--rule);border-radius:4px;padding:12px;max-height:260px;overflow:auto;
font:13px/1.55 ui-monospace,Consolas,monospace;color:#b9c8c0;white-space:pre-wrap;scrollbar-color:var(--rule) var(--sheet)}
#mtable td:first-child{color:var(--collar);padding-right:12px}#mtable td:last-child{text-align:right;overflow-wrap:anywhere}
.used{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:14px}
@media (max-width:1100px){.train{grid-template-columns:1fr}.form{border-right:0;border-bottom:1px solid var(--rule)}.status{position:static;max-height:none}}
@media (max-width:760px){.top{padding:0 16px;gap:4px 24px}.form,.status{padding:28px 16px}.field{grid-template-columns:1fr}.hint{grid-column:1}}
</style></head><body>
""" + TOP + """
<main class="train">
<form id="f" class="form" novalidate>
<h1>Train &amp; validate</h1>
<p class="lede">Train or cross-validate Random Forest models in the background. Models are saved to <code>models/</code>; an existing model is never overwritten.</p>
<fieldset><legend class="caps">Approach</legend>
<div class="opts"><label><input type="radio" name="approach" value="image" checked>Whole image</label>
<label><input type="radio" name="approach" value="window">Windows, 16 × 16 px every 8 px</label></div></fieldset>
<fieldset><legend class="caps">Mode</legend>
<div class="opts"><label><input type="radio" name="mode" value="train">Train only</label>
<label><input type="radio" name="mode" value="validate">Validate only</label>
<label><input type="radio" name="mode" value="both" checked>Validate, then train the best</label></div></fieldset>
<fieldset><legend class="caps">Data</legend>
<div class="field"><label for="root">Dataset folder</label><input type="text" id="root" name="root" value="{{ root }}" spellcheck="false">
<span class="hint">EuroSAT_MS folder with one subfolder per class. The default is used here and by the scripts when no folder is given.
<span class="savedef"><button type="button" class="btn ghost sm" id="savedef">Save as default</button><span id="savedefmsg" role="status"></span></span></span></div>
<div class="field"><label for="data_pct">Data used (%)</label><input type="number" id="data_pct" name="data_pct" value="30" min="1" max="100">
<span class="hint">Share of the images in every class to use at all</span></div>
<div class="field"><label for="test_pct">Test share (%)</label><input type="number" id="test_pct" name="test_pct" value="20" min="5" max="50">
<span class="hint" id="splitinfo"></span></div>
<div class="field" id="foldsrow"><label for="folds">Cross-validation folds</label><input type="number" id="folds" name="folds" value="5" min="2" max="10">
<span class="hint">Training data is split into this many folds; each candidate is trained this many times</span></div></fieldset>
<fieldset><legend class="caps">Hyperparameters</legend>
<p class="muted small" id="hyperhint" style="margin-bottom:6px"></p>
{% for name, (label, default, allowed) in hyper.items() %}
<div class="field"><label for="{{ name }}">{{ label }}<code>{{ name }}</code></label>
<input type="text" id="{{ name }}" name="{{ name }}" value="{{ default }}" class="hp" spellcheck="false">
<span class="hint">{{ allowed }}</span></div>{% endfor %}</fieldset>
<fieldset class="trainonly"><legend class="caps">Output</legend>
<div class="field"><label for="model_name">Model name</label><input type="text" id="model_name" name="model_name" maxlength="80" placeholder="e.g. windows_30pct_200trees" spellcheck="false">
<span class="hint">Shown on the Classify page and used as the file name; empty = approach + date and time</span></div>
<div class="field"><span>Saved to</span><code style="justify-self:start;overflow-wrap:anywhere">{{ out_dir }}</code></div></fieldset>
<div class="actions"><button type="submit" class="btn" id="go">Start</button><span id="cands"></span>
<p class="err small" id="formerr" role="alert" style="flex-basis:100%"></p></div>
</form>

<aside class="status" aria-label="Job">
<section id="jobstate" data-state="idle" aria-live="polite"><h2 class="caps">Progress</h2>
<p class="state"><span id="statelabel">Idle</span></p>
<p id="stage">No job has run since the app started.</p>
<div class="meter" role="progressbar" aria-label="Progress of the current stage" aria-valuemin="0" aria-valuemax="100" id="barwrap"><div id="bar"></div></div>
<p id="count"></p>
<pre id="log" hidden></pre></section>
<section id="cvcard" hidden><h2 class="caps">Validation results</h2>
<p class="muted small" style="margin-bottom:8px">Mean macro-F1 over the folds, ± standard deviation, best first.</p>
<table><thead><tr><th>Rank</th><th>Hyperparameters</th><th class="num">Macro-F1</th></tr></thead><tbody id="cvbody"></tbody></table></section>
<section id="modelcard" hidden><h2 class="caps">Trained model</h2>
<table id="mtable"></table>
<p class="used"><button type="button" class="btn" id="use">Use in classifier</button><span id="usemsg" class="muted small" role="status"></span></p></section>
</aside>
</main>
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
  $("cands").textContent = train ? "" : `${n} combination${n === 1 ? "" : "s"} × ${$("folds").value} folds = ${n * $("folds").value} fits · limit {{ max_candidates }} combinations`;
}
f.addEventListener("input", refresh); refresh();

f.addEventListener("submit", async e => {
  e.preventDefault(); $("formerr").textContent = "";
  const r = await fetch("/train/start", {method: "POST", body: new FormData(f)});
  const j = await r.json();
  if (!r.ok) { $("formerr").textContent = j.error; return; }
  poll();
});

function row(cells) { const tr = document.createElement("tr"); cells.forEach((c, i) => { const td = document.createElement("td"); td.textContent = c; if (i && i === cells.length - 1) td.className = "num"; tr.append(td); }); return tr; }
const pct = v => (v * 100).toFixed(1) + "%";
const LABEL = {idle: "Idle", queued: "Queued", running: "Running", done: "Done", failed: "Failed"};

async function poll() {
  clearTimeout(timer);
  const s = await (await fetch("/train/status")).json();
  $("go").disabled = !!s.running;
  const state = s.error ? "failed" : s.running ? (s.stage === "Starting" ? "queued" : "running") : s.stage ? "done" : "idle";
  $("jobstate").dataset.state = state; $("statelabel").textContent = LABEL[state];
  if (s.stage) {
    $("stage").textContent = s.error ? s.error : s.stage;
    const p = s.total ? Math.round(100 * s.done / s.total) : 0;
    $("bar").style.width = (state === "done" ? 100 : p) + "%"; $("barwrap").setAttribute("aria-valuenow", p);
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
                   ["Test tile accuracy, windows averaged", pct(m.image_vote_accuracy)]);
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
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
