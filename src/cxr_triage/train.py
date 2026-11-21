"""Training loop built for scarce, preemptible GPU time.

* ``train.max_minutes`` is a hard wall-clock budget: training checkpoints and exits cleanly.
* ``--resume`` continues from ``last.pt`` (point ``run_dir`` at Google Drive on Colab so a
  disconnect never costs more than one epoch).
* Mixed precision + channels_last for ~2x throughput on T4/L4/A100.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from . import LABELS
from .config import load_config
from .data import make_loader
from .metrics import macro_auroc
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device, amp: bool = False):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            out.append(model(x).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def train(cfg: dict, resume: bool = False) -> Path:
    t0 = time.time()
    seed_everything(cfg["seed"])
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    device = get_device()
    tc = cfg["train"]
    amp = bool(tc.get("amp")) and device.type == "cuda"
    print(f"device={device} amp={amp} run_dir={run}")

    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    m = cfg["model"]
    model = build_model(m["name"], m.get("pretrained", True), m.get("dropout", 0.0))
    model = model.to(device).to(memory_format=torch.channels_last)

    pos_weight = None
    if tc.get("pos_weight_cap"):
        prev = train_loader.dataset.labels.mean(0).clip(1e-4)
        pos_weight = torch.tensor(
            ((1 - prev) / prev).clip(1, tc["pos_weight_cap"]), dtype=torch.float32, device=device
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = len(train_loader)
    total = tc["epochs"] * steps_per_epoch
    warm = tc.get("warmup_epochs", 0) * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best, bad_epochs = 0, -1.0, 0
    if resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad_epochs = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
        print(f"resumed from epoch {ck['epoch']} (best val AUROC {best:.4f})")

    budget_s = tc.get("max_minutes", float("inf")) * 60
    out_of_time = False
    log = open(run / "metrics.jsonl", "a")
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        running, seen, te = 0.0, 0, time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
            if time.time() - t0 > budget_s:
                out_of_time = True
                break

        logits, ys = predict_logits(model, val_loader, device, amp)
        val_auc = macro_auroc(ys, 1 / (1 + np.exp(-logits)))
        rec = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_macro_auroc": val_auc,
            "lr": sched.get_last_lr()[0],
            "epoch_min": (time.time() - te) / 60,
            "partial_epoch": out_of_time,
        }
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()

        if val_auc > best:
            best, bad_epochs = val_auc, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "labels": LABELS,
                    "val_macro_auroc": best,
                    "epoch": epoch,
                },
                run / "best.pt",
            )
        else:
            bad_epochs += 1
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "bad_epochs": bad_epochs,
            },
            run / "last.pt",
        )

        if out_of_time:
            print(f"time budget of {tc['max_minutes']} min reached; checkpoint saved.")
            break
        if bad_epochs >= tc["patience"]:
            print("early stopping.")
            break
    log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val macro-AUROC {best:.4f}")
    return run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=3")
    args = ap.parse_args(argv)
    train(load_config(args.config, args.overrides), resume=args.resume)


if __name__ == "__main__":
    main()
