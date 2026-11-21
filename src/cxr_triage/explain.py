"""Grad-CAM (Selvaraju et al., 2017) for a chosen label."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self._acts = self._grads = None
        layer.register_forward_hook(self._save_acts)

    def _save_acts(self, _module, _inp, out):
        self._acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor, label_idx: int) -> np.ndarray:
        """x: 1x3xHxW. Returns an HxW heatmap in [0, 1]."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits[0, label_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy()


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a uint8 grayscale image with a jet-colored heatmap -> uint8 HxWx3."""
    import matplotlib

    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    return ((1 - alpha) * base + alpha * heat).clip(0, 255).astype(np.uint8)
