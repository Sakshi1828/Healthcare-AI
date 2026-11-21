"""Evaluate a trained run and export a deployable bundle.

1. Logits on val + test.
2. Per-label temperature scaling fit on val.
3. Operating thresholds at a target sensitivity, chosen on val, audited on test.
4. Test AUROC with bootstrap 95% CIs, AUPRC, calibration (ECE before/after),
   selective prediction (risk-coverage) and triage-level sensitivity for urgent findings.
5. Writes report.json, report.md, plots, and bundle.pt (consumed by the API).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import apply_temperature, fit_temperature
from .data import make_loader
from .metrics import (
    binary_stats,
    bootstrap_auroc_ci,
    expected_calibration_error,
    risk_coverage,
    sigmoid,
    summarize,
    threshold_at_sensitivity,
    uncertainty,
)
from .model import build_model
from .train import get_device, predict_logits


def _plots(run: Path, y, p_raw, p_cal, rc) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    edges = np.linspace(0, 1, 11)
    for name, p in (("uncalibrated", p_raw), ("temperature-scaled", p_cal)):
        yy, pp = y.ravel(), p.ravel()
        idx = np.minimum(np.digitize(pp, edges) - 1, 9)
        xs = [pp[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [yy[idx == b].mean() for b in range(10) if (idx == b).any()]
        axes[0].plot(xs, ys, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="Reliability diagram (test)", xlabel="predicted", ylabel="observed")
    axes[0].legend()
    axes[1].plot(rc["coverage"], rc["macro_auroc"], marker="o")
    axes[1].invert_xaxis()
    axes[1].set(
        title="Selective prediction (test)",
        xlabel="coverage (fraction auto-read)",
        ylabel="macro-AUROC on kept cases",
    )
    fig.tight_layout()
    fig.savefig(run / "eval_plots.png", dpi=150)
    plt.close(fig)


def evaluate(run_dir: str | Path, bootstrap: int | None = None) -> dict:
    run = Path(run_dir)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg, labels = ck["config"], ck["labels"]
    ec = cfg.get("eval", {})
    n_boot = bootstrap if bootstrap is not None else ec.get("bootstrap", 1000)
    target = ec.get("target_sensitivity", 0.9)
    urgent = [u for u in ec.get("urgent_labels", []) if u in labels]
    refer_fraction = ec.get("refer_fraction", 0.2)

    device = get_device()
    m = cfg["model"]
    model = build_model(m["name"], pretrained=False, dropout=m.get("dropout", 0.0))
    model.load_state_dict(ck["model"])
    model.to(device).to(memory_format=torch.channels_last)
    amp = bool(cfg["train"].get("amp"))
    zv, yv = predict_logits(model, make_loader(cfg, "val", shuffle=False), device, amp)
    zt, yt = predict_logits(model, make_loader(cfg, "test", shuffle=False), device, amp)

    temps = fit_temperature(zv, yv)
    pv = sigmoid(apply_temperature(zv, temps))
    pt_raw = sigmoid(zt)
    pt = sigmoid(apply_temperature(zt, temps))

    thresholds = np.array(
        [threshold_at_sensitivity(yv[:, j], pv[:, j], target) for j in range(len(labels))]
    )
    refer_unc = float(np.quantile(uncertainty(pv), 1 - refer_fraction))

    report = summarize(yt, pt, labels)
    report["ece_uncalibrated"] = expected_calibration_error(yt, pt_raw)
    report["temperatures"] = dict(zip(labels, map(float, temps), strict=True))
    for j, name in enumerate(labels):
        r = report["per_label"][name]
        r["auroc_ci95"] = bootstrap_auroc_ci(yt[:, j], pt[:, j], n=n_boot, seed=j)
        r["threshold"] = float(thresholds[j])
        stats = binary_stats(yt[:, j], pt[:, j], thresholds[j])
        r.update({f"test_{k}": v for k, v in stats.items()})

    if urgent:
        ui = [labels.index(u) for u in urgent]
        y_any = yt[:, ui].max(1)
        flagged = (pt[:, ui] >= thresholds[ui]).any(1)
        report["urgent_triage"] = {
            "labels": urgent,
            "sensitivity": float(flagged[y_any == 1].mean()) if y_any.any() else float("nan"),
            "flag_rate": float(flagged.mean()),
        }
    rc = risk_coverage(yt, pt)
    report["risk_coverage"] = rc
    report["refer_uncertainty_threshold"] = refer_unc
    report["n_test"] = int(len(yt))
    report["target_sensitivity"] = target

    (run / "report.json").write_text(json.dumps(report, indent=2))
    (run / "report.md").write_text(to_markdown(report, labels))
    _plots(run, yt, pt_raw, pt, rc)
    torch.save(
        {
            "model": ck["model"],
            "config": cfg,
            "labels": labels,
            "temperatures": temps.tolist(),
            "thresholds": thresholds.tolist(),
            "urgent_labels": urgent,
            "refer_uncertainty": refer_unc,
        },
        run / "bundle.pt",
    )
    print(to_markdown(report, labels))
    return report


def to_markdown(r: dict, labels: list[str]) -> str:
    lines = [
        f"**Test macro-AUROC: {r['macro_auroc']:.3f}** (n={r['n_test']})  ",
        f"ECE: {r['ece_uncalibrated']:.4f} → {r['ece']:.4f} after temperature scaling  ",
    ]
    if "urgent_triage" in r:
        u = r["urgent_triage"]
        lines.append(
            f"Urgent triage ({', '.join(u['labels'])}): sensitivity "
            f"{u['sensitivity']:.3f} at flag rate {u['flag_rate']:.3f}  "
        )
    rc = r["risk_coverage"]
    lines.append(
        "Selective prediction: "
        + ", ".join(
            f"{c:.0%}→{a:.3f}" for c, a in zip(rc["coverage"], rc["macro_auroc"], strict=True)
        )
        + "\n"
    )
    lines += [
        "| Finding | Prev. | AUROC (95% CI) | AUPRC | Sens@thr | Spec@thr |",
        "|---|---|---|---|---|---|",
    ]
    for name in labels:
        p = r["per_label"][name]
        lo, hi = p["auroc_ci95"]
        lines.append(
            f"| {name} | {p['prevalence']:.3f} | {p['auroc']:.3f} ({lo:.3f}–{hi:.3f}) | "
            f"{p['auprc']:.3f} | {p['test_sensitivity']:.3f} | {p['test_specificity']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args(argv)
    evaluate(args.run_dir, args.bootstrap)


if __name__ == "__main__":
    main()
