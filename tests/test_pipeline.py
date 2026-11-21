import io
import json

import numpy as np
import torch
from PIL import Image

from cxr_triage import LABELS
from cxr_triage.config import load_config
from cxr_triage.inference import Predictor
from cxr_triage.model import build_model


def test_config_overrides():
    cfg = load_config("configs/smoke.yaml", ["train.epochs=7", "model.name=resnet50"])
    assert cfg["train"]["epochs"] == 7 and cfg["model"]["name"] == "resnet50"


def test_model_output_shape():
    for name in ("resnet18", "densenet121", "efficientnet_b0"):
        m = build_model(name, pretrained=False).eval()
        assert m(torch.zeros(2, 3, 64, 64)).shape == (2, len(LABELS))


def test_training_learns_and_writes_artifacts(trained_run):
    run, report = trained_run
    for f in (
        "best.pt",
        "last.pt",
        "metrics.jsonl",
        "report.json",
        "report.md",
        "eval_plots.png",
        "bundle.pt",
    ):
        assert (run / f).exists(), f
    assert report["macro_auroc"] > 0.7  # synthetic task is learnable
    assert json.loads((run / "report.json").read_text())["n_test"] == 256


def test_resume_continues_from_checkpoint(trained_run, tmp_path):
    from cxr_triage.train import train

    run, _ = trained_run
    cfg = load_config(run / "config.yaml", ["train.epochs=4"])
    train(cfg, resume=True)
    epochs = [
        json.loads(line)["epoch"] for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert epochs == [0, 1, 2, 3]


def test_predictor_triage_and_gradcam(trained_run):
    run, _ = trained_run
    pred = Predictor(run / "bundle.pt")
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 280), dtype=np.uint8))
    out = pred.predict(img, explain=True)
    assert out["triage"] in {
        "URGENT",
        "REFER_UNCERTAIN",
        "NON_URGENT_FINDING",
        "NO_FINDING_DETECTED",
    }
    assert set(out["probabilities"]) == set(LABELS)
    assert all(0 <= p <= 1 for p in out["probabilities"].values())
    png = Image.open(io.BytesIO(__import__("base64").b64decode(out["gradcam"]["png_base64"])))
    assert png.size == (64, 64)


def test_api(trained_run):
    from fastapi.testclient import TestClient

    from cxr_triage.serve import create_app

    run, _ = trained_run
    client = TestClient(create_app(str(run / "bundle.pt")))
    assert client.get("/health").json()["status"] == "ok"
    buf = io.BytesIO()
    Image.fromarray(np.zeros((128, 128), dtype=np.uint8)).save(buf, format="PNG")
    r = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and "triage" in r.json()
    bad = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400


def test_demo_ui_and_samples(trained_run):
    from fastapi.testclient import TestClient

    from cxr_triage.serve import create_app

    run, _ = trained_run
    client = TestClient(create_app(str(run / "bundle.pt")))
    page = client.get("/")
    assert page.status_code == 200 and "CXR-Triage" in page.text
    samples = client.get("/samples").json()
    assert len(samples) >= 2
    assert any(s["ground_truth"] == [] for s in samples)
    img = client.get(samples[0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    r = client.post("/predict?explain=true", files={"file": ("s.png", img.content, "image/png")})
    assert r.status_code == 200 and "gradcam" in r.json()
    assert client.get("/samples/999.png").status_code == 404
