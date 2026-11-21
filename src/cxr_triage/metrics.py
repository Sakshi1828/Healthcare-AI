"""Evaluation metrics, written for clinical honesty: CIs, calibration, and selective prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean([safe_auroc(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def bootstrap_auroc_ci(
    y: np.ndarray, p: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC of a single label."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            stats.append(roc_auc_score(y[idx], p[idx]))
    if not stats:
        return float("nan"), float("nan")
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE over all (sample, label) pairs, equal-width bins."""
    y, p = y.ravel(), p.ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def threshold_at_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity >= target (chosen on validation, applied to test)."""
    if y.sum() == 0:
        return 1.0
    _, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(min(thr[ok[0]], 1.0))


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def binary_stats(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pred = p >= thr
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    tn = float((~pred & (y == 0)).sum())
    return {
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "ppv": _div(tp, tp + fp),
        "npv": _div(tn, tn + fn),
    }


def uncertainty(p: np.ndarray) -> np.ndarray:
    """Per-image uncertainty: mean binary entropy across labels (bits)."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).mean(axis=1)


def risk_coverage(y: np.ndarray, p: np.ndarray, coverages=None) -> dict[str, list[float]]:
    """Selective prediction: keep the most-confident fraction of images, refer the rest to a
    radiologist, and report macro-AUROC on what the model keeps."""
    coverages = coverages if coverages is not None else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    order = np.argsort(uncertainty(p))
    out = {"coverage": [], "macro_auroc": []}
    for c in coverages:
        keep = order[: max(2, int(round(c * len(order))))]
        out["coverage"].append(float(c))
        out["macro_auroc"].append(macro_auroc(y[keep], p[keep]))
    return out


def summarize(y: np.ndarray, p: np.ndarray, labels: list[str]) -> dict:
    per = {}
    for j, name in enumerate(labels):
        per[name] = {
            "auroc": safe_auroc(y[:, j], p[:, j]),
            "auprc": float(average_precision_score(y[:, j], p[:, j]))
            if y[:, j].any()
            else float("nan"),
            "prevalence": float(y[:, j].mean()),
        }
    return {
        "macro_auroc": macro_auroc(y, p),
        "ece": expected_calibration_error(y, p),
        "per_label": per,
    }
