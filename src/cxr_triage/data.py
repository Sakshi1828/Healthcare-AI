"""Data loading: NIH ChestX-ray14 via MedMNIST+ (official patient-level splits), or a synthetic
stand-in with learnable structure for CPU tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from . import LABELS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_model_input(img: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 HxW (or HxWxC) -> normalized float 3xHxW tensor, as ImageNet backbones expect."""
    x = torch.from_numpy(np.array(img, dtype=np.float32)).div_(255.0)
    x = x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1)[:1]
    x = x.expand(3, -1, -1)
    return v2.functional.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


class CXRDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False):
        self.images = images
        self.labels = labels.astype(np.float32)
        # No horizontal flips: they swap anatomical laterality (e.g. heart position).
        self.augment = (
            v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05))
            if train
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = to_model_input(self.images[i])
        if self.augment is not None:
            x = self.augment(x)
        return x, torch.from_numpy(self.labels[i])


def make_synthetic(n: int, size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noise images where each label brightens its own grid cell, so a model can learn it."""
    rng = np.random.default_rng(seed)
    k = len(LABELS)
    prevalence = np.linspace(0.05, 0.4, k)
    labels = (rng.random((n, k)) < prevalence).astype(np.int64)
    imgs = rng.normal(100, 25, (n, size, size))
    grid = int(np.ceil(np.sqrt(k)))
    cell = size // grid
    for j in range(k):
        r, c = divmod(j, grid)
        imgs[labels[:, j] == 1, r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] += (
            12  # weak signal
        )
    return np.clip(imgs, 0, 255).astype(np.uint8), labels


def load_split(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    d = cfg["data"]
    if d["source"] == "synthetic":
        sizes = d.get("synthetic_sizes", {"train": 512, "val": 256, "test": 256})
        seed = {"train": 0, "val": 1, "test": 2}[split] + cfg.get("seed", 0) * 10
        return make_synthetic(sizes[split], d["image_size"], seed)
    if d["source"] == "medmnist":
        from medmnist import ChestMNIST

        root = Path(d["root"])
        root.mkdir(parents=True, exist_ok=True)
        ds = ChestMNIST(split=split, download=True, root=str(root), size=d["image_size"])
        return ds.imgs, ds.labels
    raise ValueError(f"Unknown data source {d['source']!r}")


def make_loader(cfg: dict, split: str, shuffle: bool | None = None) -> DataLoader:
    images, labels = load_split(cfg, split)
    train = split == "train"
    return DataLoader(
        CXRDataset(images, labels, train=train),
        batch_size=cfg["train"]["batch_size"],
        shuffle=train if shuffle is None else shuffle,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=cfg["data"].get("num_workers", 0) > 0,
    )
