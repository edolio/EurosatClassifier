# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Learners and portfolio viewers who find the project on GitHub. They clone it, run the local web app,
upload a EuroSAT Sentinel-2 tile and look at what the models say. Some go on to retrain models to see
how data share and hyperparameters change the results.

## Product Purpose

A local web app around Random Forest land-cover classifiers for EuroSAT_MS (13-band Sentinel-2 tiles,
64×64 px at 10 m, 10 classes). The most important job is **classifying a tile quickly**: upload, get
the land-cover class, understand the answer. Comparing approaches and training models are secondary.

Success: a first-time visitor gets a prediction for a tile within seconds of opening the page and can
tell how confident it is and where in the tile each class sits.

## Positioning

The project shows more than one classification approach on the same tile: a whole-image model and a
16×16 px sliding-window model that maps classes inside the tile. It is built to grow with further
approaches. Everything runs locally on the user's machine, with models trained by the project itself.

## Operating Context

- Started with `uv run app.py` and opened at http://127.0.0.1:5000 on a desktop browser.
- Input: 13-band EuroSAT_MS GeoTIFFs (about 110 KB each); ordinary RGB photos are not accepted.
- Two pages: **Classify** (model choice, upload, whole-image result with class probabilities, window
  class map, model parameters, class descriptions) and **Train & validate** (setup, hyperparameters,
  background job with live progress and log, validation ranking, trained model).
- Models live in `models/`; two trained models ship with the repo.

## Capabilities and Constraints

- Flask app with inline HTML/CSS/JS templates in `app.py`; no frontend build step today.
- Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
  Residential, River, SeaLake.
- Metrics shown are real outputs of training (accuracy, macro-F1, cross-validation scores); the shipped
  models were trained on 30% of the data.
- Undecided: further classification approaches to add later.

## Brand Commitments

- **Dark theme only** (user requirement): no light mode.
- Name in use: "EuroSAT land-cover classifier". No logo or other brand assets exist.

## Evidence on Hand

- Real model metrics in the model files and README (whole-image 89.1% test accuracy; window model 79.5%
  of windows, 91.1% of images, both on 30% of the data).
- Dataset: EuroSAT (Zenodo record 7711810), not stored in the repo.
- No testimonials, users, benchmarks or press: do not invent any.

## Product Principles

1. The prediction comes first: uploading a tile and reading the answer must be the clearest path.
2. Show confidence and evidence, not just a label: probabilities and the window map explain the answer.
3. Every number shown is a real model output; nothing decorative pretends to be data.
4. Approaches sit side by side on the same tile, so new ones can be added without redesigning the page.
