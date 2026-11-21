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
from starlette.concurrency import run_in_threadpool

from . import __version__
from .inference import Predictor
from .report import ReportService, make_generator

MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs


def create_app(bundle_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="CXR-Triage",
        version=__version__,
        description="Calibrated chest X-ray triage with uncertainty-aware referral.",
    )
    predictor = Predictor(bundle_path or os.environ.get("CXR_BUNDLE", "bundle.pt"))
    # Report drafting: CXR_REPORT_BACKEND = auto | gemini | claude | template (auto picks an LLM
    # only if its API key is set). LLM calls are capped per hour; over the cap -> template.
    reports = ReportService(
        make_generator(os.environ.get("CXR_REPORT_BACKEND", "auto")),
        max_llm_calls_per_hour=int(os.environ.get("CXR_LLM_MAX_PER_HOUR", "30")),
    )
    index_html = (files("cxr_triage") / "static" / "index.html").read_text()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return index_html

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "labels": predictor.labels,
            "report_backend": reports.generator.name,
        }

    @app.get("/samples")
    def samples():
        return [
            {"id": i, "url": f"/samples/{i}.png", "ground_truth": s["labels"]}
            for i, s in enumerate(predictor.samples)
        ]

    @app.get("/samples/{i}/predict")
    def sample_predict(i: int, explain: bool = False, report: bool = False):
        """Predict on a bundled sample; uses its pre-generated report when one exists, so the
        demo never needs an API call for samples."""
        if not 0 <= i < len(predictor.samples):
            raise HTTPException(404, "No such sample.")
        out = predictor.predict(Image.fromarray(predictor.samples[i]["image"]), explain=explain)
        if report:
            stored = predictor.sample_reports[i] if i < len(predictor.sample_reports) else None
            out["report"] = {**stored, "pregenerated": True} if stored else reports(out)
        out["ground_truth"] = predictor.samples[i]["labels"]
        return out

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
    async def predict(file: UploadFile = File(...), explain: bool = False, report: bool = False):
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image too large (max 10 MB).")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
            raise HTTPException(400, "Could not decode image (PNG/JPEG expected).") from err
        # Model inference and LLM calls block, so keep them off the event loop.
        out = await run_in_threadpool(predictor.predict, image, explain)
        if report:
            out["report"] = await run_in_threadpool(reports, out)
        return out

    return app


app = create_app() if os.environ.get("CXR_BUNDLE") else None
