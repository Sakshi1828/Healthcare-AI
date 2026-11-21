"""Deployable predictor: model + calibration + operating thresholds + triage policy.

The bundle is produced by ``evaluate.py`` so that every threshold the service uses was chosen on
validation data and audited on the held-out test set.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import to_model_input
from .explain import GradCAM, overlay
from .metrics import sigmoid, uncertainty
from .model import build_model, gradcam_layer

DISCLAIMER = "Research prototype. Not a medical device; not for clinical decision-making."


def make_prediction(
    probs: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray,
    urgent: set[str],
    refer_uncertainty: float,
) -> dict:
    """Triage policy, shared by the API and the offline report evaluation."""
    positive = probs >= thresholds
    unc = float(uncertainty(probs[None])[0])
    findings = [labels[j] for j in np.where(positive)[0]]
    if any(f in urgent for f in findings):
        triage = "URGENT"
    elif unc > refer_uncertainty:
        triage = "REFER_UNCERTAIN"
    elif findings:
        triage = "NON_URGENT_FINDING"
    else:
        triage = "NO_FINDING_DETECTED"
    return {
        "triage": triage,
        "findings": findings,
        "uncertainty": unc,
        "probabilities": {n: float(p) for n, p in zip(labels, probs, strict=True)},
        "thresholds": {n: float(t) for n, t in zip(labels, thresholds, strict=True)},
        "urgent_labels": sorted(urgent),
        "disclaimer": DISCLAIMER,
    }


class Predictor:
    def __init__(self, bundle_path: str | Path, device: str = "cpu"):
        b = torch.load(bundle_path, map_location=device, weights_only=False)
        self.cfg, self.labels = b["config"], b["labels"]
        self.temps = np.asarray(b["temperatures"])
        self.thresholds = np.asarray(b["thresholds"])
        self.urgent = set(b["urgent_labels"])
        self.refer_uncertainty = float(b["refer_uncertainty"])
        self.image_size = self.cfg["data"]["image_size"]
        self.device = torch.device(device)
        m = self.cfg["model"]
        self.model = build_model(m["name"], pretrained=False, dropout=m.get("dropout", 0.0))
        self.model.load_state_dict(b["model"])
        self.model.to(self.device).eval()
        self.cam = GradCAM(self.model, gradcam_layer(self.model, m["name"]))
        self.samples = b.get("samples", [])

    def sample_png(self, i: int) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.samples[i]["image"]).save(buf, format="PNG")
        return buf.getvalue()

    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        gray = np.asarray(
            image.convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        )
        return gray, to_model_input(gray).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image, explain: bool = False) -> dict:
        gray, x = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(x)[0].cpu().numpy()
        probs = sigmoid(logits / self.temps)
        result = make_prediction(
            probs, self.labels, self.thresholds, self.urgent, self.refer_uncertainty
        )
        if explain:
            top = int(np.argmax(probs / self.thresholds))
            cam = self.cam(x, top)
            buf = io.BytesIO()
            Image.fromarray(overlay(gray, cam)).save(buf, format="PNG")
            result["gradcam"] = {
                "label": self.labels[top],
                "png_base64": base64.b64encode(buf.getvalue()).decode(),
            }
        return result
