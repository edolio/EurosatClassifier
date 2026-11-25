# EuroSAT land-cover classifier (Random Forest)

Classifies Sentinel-2 multispectral image patches from the
[EuroSAT dataset](https://zenodo.org/records/7711810) into 10 land-cover classes
(AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, SeaLake) with a scikit-learn Random Forest, and serves predictions
through a small local web page.

The project has two classification approaches. Both use the same 73 features (mean, standard
deviation and 10th/50th/90th percentiles of every band, plus mean and spread of four spectral
indices: NDVI, NDWI, NDBI, NDMI), computed over different areas:

| Approach | What is classified |
|---|---|
| 1. Whole image | each 64×64 px image: one class per tile (89.1% test accuracy on 30% of the data) |
| 2. Labelled windows | each 16×16 px window, every 8 px (7×7 = 49 windows per image), with window labels cleaned by the forest and by you |

EuroSAT has no per-pixel labels: a window starts with its image's label, so a field window inside
a "Highway" image is labelled Highway. Approach 2 fixes those labels before training (see
*Labelled window model* below).

## Project files

| File | What it does |
|---|---|
| `prepare.py` | Reads every image, computes the features, writes `features.csv` and prints a dataset description |
| `grid_search.py` | 5-fold grid search of Random Forest hyperparameters, writes `cv_results.csv` |
| `train_final.py` | Trains the final model with the best hyperparameters, prints the results, saves `final_results.png` (loss curve + confusion matrix) and `models/model.joblib` |
| `prepare_windows.py` | Approach 2, step 1: cuts images into 16×16 px windows (8 px step), computes their features, writes `window_features.csv` (its training windows fit the k-means centroids) |
| `grid_search_labelled.py` | Approach 2: 5-fold grid search grouped by image on the labelled windows, writes `cv_results_labelled.csv` |
| `train_labelled.py` | Approach 2, two modes. `label`: builds the labelled window dataset (k-means distance features, forest relabelling, your labels for critical windows), writes `human_labels.csv`, `window_labels.csv.gz` (labels) and `window_labels_features.csv` (features). `train`: trains the final labelled window model, prints per-window results, saves `labelled_results.png` and `models/labelled_window_model.joblib` |
| `app.py` | Local web app. **Classify**: upload an image, see the whole-image prediction and the window map. **Train & validate**: train or cross-validate whole-image models from the browser. **Label windows**: label windows for approach 2 |
| `labeller.py` | Background labelling job used by the Label windows page: runs `train_labelled.py label` with the browser as labeller |
| `trainer.py` | Background training/validation job used by the Train & validate page; saves models to `models/` |
| `test_pipeline.py` | Quick end-to-end check of the whole-image steps on 30% of the data, plus the window features for approach 2; data and plots in `test_out/`, model as `models/test_model.joblib` |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python version and exact package versions for uv |
| `models/model.joblib`, `models/labelled_window_model.joblib` | The two trained models included in the repo (see below) |

## Included models

The repo ships two trained models, so the web app works right after setup without any training:

| File | Approach | Trained on | Hyperparameters | Test results |
|---|---|---|---|---|
| `models/model.joblib` | Whole image | 30% of EuroSAT (6,480 train / 1,620 test images) | 200 trees, `max_features=sqrt` (best of a 50/100/200-tree grid search) | 89.1% accuracy, macro-F1 0.885 |
| `models/labelled_window_model.joblib` | 16×16 px windows + distances to 40 k-means centroids (web app default) | `train_labelled.py label`, 3 rounds on the same 30% split: 116,699 windows from 2,546 training images; 81 human labels, 448 forest relabels, 8,055 windows dropped | 200 trees, `max_features=sqrt`, `min_samples_leaf=2`, human labels weighted ×5 | 78.4% of 79,380 test windows, 89.2% of images (averaged windows) |

They were trained with the package versions in `uv.lock`, which `uv sync` installs. A model file
only loads with the scikit-learn version it was trained with.

Running `train_final.py` or `train_labelled.py train` **replaces** these files with your own models,
and `git` will then show them as changed. To keep the shipped ones, give your models other names,
for example on the training page, or restore them with
`git checkout -- models/model.joblib models/labelled_window_model.joblib`.

The dataset itself is **not** in the repo. You need it to train models, and you need at least
some 13-band `.tif` images from it to try the classifier.

## Setup

1. **Install uv** (it also installs the right Python version for you):

   Windows (PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   macOS / Linux:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create the environment** from the project folder. This installs Python 3.12 if needed
   and the exact package versions in `uv.lock` into `.venv/`:
   ```
   uv sync
   ```

3. **Download the data.** Get `EuroSAT_MS.zip` (2 GB, multispectral version) from
   <https://zenodo.org/records/7711810> and unzip it into `data/`, so the folders look like:
   ```
   data/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif
   data/EuroSAT_MS/Forest/...
   ...
   ```
   An extra level (`data/EuroSAT_MS/EuroSAT_MS/AnnualCrop/...`), which some unzip tools create,
   also works. **The data can live anywhere:** type its folder on the web app's training page
   and click **Save as default**. From then on, the app and all scripts use it when no folder is
   given. The choice is stored in `config.toml` in the project folder, which is personal and
   ignored by git. The RGB version (`EuroSAT_RGB.zip`) does **not** work: the model needs all 13 bands.

## Usage

Run the commands from the project folder: `uv run` then picks the project environment
automatically, so there is nothing to activate. All default paths (data, outputs, `models/`) are
relative to the project folder, not to where you run from, and no path is tied to a particular
computer. From another folder, use `uv run --project <project folder> <project folder>/prepare.py`.

If something is missing, the scripts say what to do: no data → where to unzip it; no
`features.csv` etc. → which step to run first. The web app starts with an empty `models/` folder,
which is the case right after cloning, and asks you to train a model.

**Quick check** that everything works (about 8 minutes, 30% of the data): the whole-image steps,
plus `test_out/window_features.csv` for approach 2:
```
uv run test_pipeline.py
```

**Full pipeline:**
```
uv run prepare.py
uv run grid_search.py
uv run train_final.py
```
`prepare.py` takes about 15 minutes for the full dataset (mostly disk reads), `grid_search.py`
roughly 15–30 minutes on 8 cores, `train_final.py` a minute or two.

**Labelled window model (approach 2, human in the loop):**
```
uv run prepare_windows.py
uv run train_labelled.py label all 1
uv run grid_search_labelled.py
uv run train_labelled.py train
```
`prepare_windows.py` cuts the images into windows and computes their features (about 15 minutes
for the full dataset). It splits images into train and test exactly like approach 1 and writes
`window_features.csv`; `train_labelled.py` fits its scaler and k-means centroids on the training
windows of that table. To use a share of the data: `uv run prepare_windows.py data/EuroSAT_MS 0.3`
(the quick check writes the 30% version to `test_out/window_features.csv`). Both prepare scripts
read the default dataset folder (the one saved in the web app, else `data/EuroSAT_MS`); give
another folder as the first argument to override it for one run.

`train_labelled.py label [images per class, or all] [rounds] [window_features.csv]` builds the dataset.
`all 1` labels every training image in one round, the same images as approach 1;
smaller runs such as `100 3` are quicker for labelling sessions.
EuroSAT labels whole images, so a window's label is only its image's label. This script builds a
window dataset in rounds (default 3 rounds of 100 training images per class, all 49 windows each,
plus every image you already labelled a window of). It needs `window_features.csv` from
`prepare_windows.py`: a scaler and k-means (40 centroids) are fitted once on its training windows,
never on test images, and reused in every round and for later predictions.
1. each window's 73 features are scaled and its distance to every saved centroid is added as a feature;
2. the forest trains on the image labels (your labels override them and weigh 5x more), using 8
   random windows per image plus all your labels (neighbouring windows overlap, so the rest add
   little but time). It scores every window with trees that did not train on it, relabels windows
   it is at least 80% sure belong to another class, drops windows below 40%, and retrains.
   Windows of River/SeaLake tiles that the forest sees as clearly land (water probability
   below 20%: banks, shores) are relabelled to that land class.
   Relabels are recomputed from the image labels every round, so mistakes don't build up;
3. it shows you about 30 critical windows, one per cluster: the window outlined on its image and
   enlarged. Type 0–9 in the terminal for the class, `m` for mixed / unsure (the window is dropped), `s` to skip,
   `q` to stop the round. Before your answers are used, it prints how often the automatic label
   matched yours.

Your answers go to `human_labels.csv` straight away and are reused on every run; the final label
and source (`image`, `auto`, `human`, `dropped`) of every window go to `window_labels.csv.gz`, the
labelled dataset (gzipped labels only, about 1.5 MB for 317,520 windows, small enough for git). Their
73 features go to `window_labels_features.csv` (about 160 MB, ignored by git); `grid_search_labelled.py`
and `train` read both. At the end, `window_labels.png`
shows sample training windows for each label, framed by where the label came from (your label,
relabelled by the forest, or the image label). Before that it prints statistics: windows used and dropped,
per-class counts before and after relabelling, a table of label changes, and how your labels
compare with the image labels and with the automatic ones.

- The labelled images are split 80/20 by image (stratified by class, seed 42), like the
  whole-image model. Test windows keep their cleaned labels: yours, the forest's relabels, or the
  image label; dropped windows are left out.
- `grid_search_labelled.py` searches `n_estimators`, `max_depth` and `min_samples_leaf` (12 combinations) with 5-fold
  cross-validation grouped by image and macro-F1, on 8 random windows per training image plus all
  your labelled ones (your labels weigh 5×). Writes `cv_results_labelled.csv`.
- `train_labelled.py train` trains the final model, three forests saved together:
  - the **main forest** sees each window's own features plus a **water context**: the mean NDWI and
    the share of water windows among the 24 windows around it (5×5). It decides land vs water and
    the land class, so land on a riverbank is not pulled into River;
  - for windows it calls water, a **River/SeaLake decider** chooses between the two from the class
    probabilities of the same 24 neighbours (from a third, stage-1 forest): a river has banks
    around it, a lake has more lake.

  It reports everything per window: loss curve and confusion matrix (`labelled_results.png`), test accuracy, macro-F1, log
  loss, per-class results, and accuracy on the test windows you labelled. It saves
  `models/labelled_window_model.joblib` with the scaler and the 40 centroids. It does not predict
  whole tiles.
  The model is kept under 90 MB (GitHub refuses files over 100 MB): it is saved with lzma
  compression, and if its trees would still be too large, it is refitted with `min_samples_leaf`
  doubled until it fits. The printed hyperparameters show the value used.

The web app uses `models/labelled_window_model.joblib` as its default window model. In Python, a new
image is classified window by window without refitting k-means:
```python
import joblib, tifffile
from train_labelled import predict_windows
grid, proba = predict_windows(joblib.load("models/labelled_window_model.joblib"), tifffile.imread("some_tile.tif"))
# grid: 7x7 array with the class of each window
```

**Labelling in the browser:** the web app's **Label windows** page runs the same labelling loop
as `train_labelled.py label`, with you answering in the browser instead of the terminal. Choose
the images per class (`all` or a number), the rounds and the window features file, then press
Start. Each question shows the tile with the window outlined and the window enlarged; answer with
the class buttons or keys `0`–`9`, `m` (mixed / unsure), `s` (skip) or `q` (stop the round). Every
answer goes straight to `human_labels.csv`. The log shows each message with the clock time and the
time since the start. At the end the page shows the statistics as tables (totals, per class, a
heat table of label changes, your labels) and the `window_labels.png` sample figure.

**Web interface:**
```
uv run app.py
```
Then open <http://127.0.0.1:5000> and upload a `.tif` from `data/EuroSAT_MS`. The top of the
page shows the whole-image prediction. Below it is the window map: each coloured square is one
window's class (hover for its confidence; a square split along the diagonal has two likely classes:
top under 60% and runner-up at least 25%, both shown on hover), with window counts per class and the averaged
prediction.

The app loads `models/model.joblib` and, for windows, `models/labelled_window_model.joblib` by
default. If a model is missing, its section shows how to build one. To start with other models,
give their names (files in `models/`), for example the quick-check whole-image model:
```
uv run app.py test_model
```
Stop it with Ctrl+C.

**Train & validate page** (<http://127.0.0.1:5000/train>): trains whole-image models from the browser.
- **Mode:**
  - *Train only:* one value per hyperparameter.
  - *Validate only:* comma-separated values; every combination is cross-validated (up to 100).
  - *Validate, then train the best.*
- **Data used (%):** share of the images in each class to use.
- **Test share (%):** how much of that data is held out for testing (5–50%); the rest is used
  for training and cross-validation.
- **Hyperparameters:** `n_estimators`, `max_depth`, `max_features`, `min_samples_split`,
  `min_samples_leaf`, plus the number of cross-validation folds.

- **Model name** (when training): shown in the Classify page and used as the file name
  `models/<name>.joblib`. It defaults to `image_model` plus date and time. An existing file is
  never overwritten: the page asks for another name.

The page shows each stage's progress, a live log, the validation ranking and the trained model's
test scores. **Use in classifier** loads the new model into the Classify page.

**Choosing models on the Classify page:** the *Models* card at the top has one list for
whole-image models and one for labelled window models. It lists every model in `models/`. Picking one loads it immediately. The name of the model in use is shown in each
section heading and parameter table. The choice lasts until the app restarts. Features
from the last run are kept in memory, so a second run with the same data settings skips
reading the images.

**Default dataset folder:** the training page's *Dataset folder* field starts with the default.
That is `data/EuroSAT_MS` until you type another folder and click **Save as default**. The
folder is checked for images before it is saved, and the scripts use the same default.

## Notes

- **All models are saved in `models/` only**, by the scripts and by the web app. Where a script
  takes a model file name, any folder in it is ignored.
- A model file only loads with the scikit-learn version it was trained with. If you change
  package versions, re-run `train_final.py`.
- Train and test sets are split once (80/20, stratified, seed 42) and shared by
  `grid_search.py` and `train_final.py`, so test images are never used for tuning.
- Ordinary RGB photos cannot be classified; only 13-band EuroSAT_MS GeoTIFFs are accepted.
