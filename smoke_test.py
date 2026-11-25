"""Fast check run by CI: the app starts, both shipped models load, and a fake 13-band image gets classified.

Needs no dataset. Usage: uv run smoke_test.py
"""
import io

import numpy as np
import tifffile

import app

assert app.bundle is not None, f"image model did not load from {app.MODEL_PATH}"
assert app.wbundle is not None, f"window model did not load from {app.WINDOW_PATH}"

client = app.app.test_client()

r = client.get("/")
assert r.status_code == 200, f"GET / returned {r.status_code}"
assert client.get("/train").status_code == 200, "GET /train failed"

# Random 64x64 tile with the 13 Sentinel-2 bands, in EuroSAT's uint16 reflectance range.
img = np.random.default_rng(0).integers(0, 4000, size=(64, 64, len(app.BANDS)), dtype=np.uint16)
buf = io.BytesIO()
tifffile.imwrite(buf, img)
buf.seek(0)

r = client.post("/", data={"image": (buf, "fake.tif")}, content_type="multipart/form-data")
html = r.get_data(as_text=True)
assert r.status_code == 200, f"POST / returned {r.status_code}"
assert "Expected a" not in html and "not a readable TIFF" not in html, "the app rejected the test image"
assert any(c in html for c in app.CLASSES), "no land-cover class in the prediction page"

# A non-TIFF upload must give the user an error message, not a crash.
r = client.post("/", data={"image": (io.BytesIO(b"hello"), "x.tif")}, content_type="multipart/form-data")
assert r.status_code == 200 and "not a readable TIFF" in r.get_data(as_text=True), "bad file not handled"

print("Smoke test passed.")
