"""ImageNet-pretrained backbones with a multi-label head."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from . import LABELS


def build_model(
    name: str, pretrained: bool = True, dropout: float = 0.0, num_labels: int = len(LABELS)
) -> nn.Module:
    w = "DEFAULT" if pretrained else None
    if name == "densenet121":
        m = models.densenet121(weights=w)
        m.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(m.classifier.in_features, num_labels)
        )
    elif name in ("resnet18", "resnet50"):
        m = getattr(models, name)(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_labels))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_labels)
        )
    else:
        raise ValueError(f"Unknown model {name!r}")
    return m


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block, the usual Grad-CAM target."""
    if name == "densenet121":
        return model.features.denseblock4
    if name.startswith("resnet"):
        return model.layer4
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(name)
