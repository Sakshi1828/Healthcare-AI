"""FastAPI service. Run: ``CXR_BUNDLE=runs/<run>/bundle.pt uvicorn cxr_triage.serve:app``."""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from . import __version__
from .inference import Predictor

MAX_BYTES = 10 * 1024 * 1024


def create_app(bundle_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="CXR-Triage",
        version=__version__,
        description="Calibrated chest X-ray triage with uncertainty-aware referral.",
    )
    predictor = Predictor(bundle_path or os.environ.get("CXR_BUNDLE", "bundle.pt"))

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "labels": predictor.labels}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), explain: bool = False):
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image too large (max 10 MB).")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as err:
            raise HTTPException(400, "Could not decode image (PNG/JPEG expected).") from err
        return predictor.predict(image, explain=explain)

    return app


app = create_app() if os.environ.get("CXR_BUNDLE") else None
