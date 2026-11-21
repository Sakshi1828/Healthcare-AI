"""FastAPI service + demo UI.

Run: ``CXR_BUNDLE=runs/<run>/bundle.pt uvicorn cxr_triage.serve:app``
"""

from __future__ import annotations

import io
import os
from importlib.resources import files

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from . import __version__
from .inference import Predictor

MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs


def create_app(bundle_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="CXR-Triage",
        version=__version__,
        description="Calibrated chest X-ray triage with uncertainty-aware referral.",
    )
    predictor = Predictor(bundle_path or os.environ.get("CXR_BUNDLE", "bundle.pt"))
    index_html = (files("cxr_triage") / "static" / "index.html").read_text()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return index_html

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "labels": predictor.labels}

    @app.get("/samples")
    def samples():
        return [
            {"id": i, "url": f"/samples/{i}.png", "ground_truth": s["labels"]}
            for i, s in enumerate(predictor.samples)
        ]

    @app.get("/samples/{i}.png", include_in_schema=False)
    def sample_image(i: int):
        if not 0 <= i < len(predictor.samples):
            raise HTTPException(404, "No such sample.")
        return Response(
            predictor.sample_png(i),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), explain: bool = False):
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image too large (max 10 MB).")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
            raise HTTPException(400, "Could not decode image (PNG/JPEG expected).") from err
        return predictor.predict(image, explain=explain)

    return app


app = create_app() if os.environ.get("CXR_BUNDLE") else None
