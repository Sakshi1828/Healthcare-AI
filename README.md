# CXR-Triage: trustworthy chest X-ray triage

[![CI](https://github.com/sakshi1828/healthcare-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sakshi1828/healthcare-ai/actions/workflows/ci.yml)

An end-to-end system that reads a chest X-ray and screens it for 14 findings. It returns
**calibrated** probabilities, flags urgent cases, **refers uncertain cases to a radiologist**
instead of guessing, and shows *where* it looked with Grad-CAM. All of this is served behind a
containerized API.

**Live demo:** _coming soon (deploy with `deploy/cloudrun.sh`)_ · [Model card](docs/MODEL_CARD.md)

> Research prototype on public data. Not a medical device.

## Why this project is different

Most portfolio X-ray classifiers stop at "AUROC = 0.8X". In a real deployment, the questions
that matter are different. This project answers them:

| Clinical question | How the system answers it |
|---|---|
| Can I trust a 70% from the model? | Per-label **temperature scaling** fit on validation data; ECE reported before and after |
| How sure are we about the metric? | **Bootstrap 95% CIs** for every AUROC |
| How many urgent cases does it miss? | Thresholds chosen on val for **90% sensitivity**, then audited on held-out test |
| What happens when it is unsure? | **Selective prediction**: high-uncertainty images go to a human (risk-coverage curve) |
| Why did it say that? | **Grad-CAM** heatmap returned by the API |
| Can it run in production? | FastAPI + Docker + CI; checkpoint/resume; deterministic configs |

## Architecture

```
NIH ChestX-ray14 (MedMNIST+, patient-level splits, 112k images)
        │
        ▼
 DenseNet-121 (ImageNet-pretrained) ── AMP, cosine LR, class-imbalance weighting,
        │                              wall-clock budget, resume-from-checkpoint
        ▼
 Validation set ──► temperature scaling ──► operating thresholds @ 90% sensitivity
        │                                  ──► referral threshold on uncertainty
        ▼
 Test set ──► AUROC + 95% CI · AUPRC · ECE · sens/spec · risk-coverage ──► report.md
        │
        ▼
 bundle.pt ──► FastAPI /predict ──► {triage, findings, probabilities, uncertainty, Grad-CAM}
```

Triage policy: `URGENT` (pneumothorax, effusion, pneumonia, edema, or consolidation above its
threshold) → `REFER_UNCERTAIN` (prediction entropy in the top 20% seen on validation) →
`NON_URGENT_FINDING` → `NO_FINDING_DETECTED`.

## Results

> Fill this in from `runs/<run>/report.md` after your GPU run (see `docs/GPU_BUDGET.md`).
> Published MedMNIST ChestMNIST baselines are around 0.77 macro-AUROC, a useful reference point.

| Model | Res. | Test macro-AUROC (95% CI) | ECE raw → calibrated | Urgent sens. @ flag rate | AUROC @ 80% coverage |
|---|---|---|---|---|---|
| DenseNet-121 | 128 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

<!-- ![eval](docs/results/eval_plots.png) ![gradcam](docs/results/gradcam_gallery.png) -->

## Quickstart

```bash
pip install -e ".[serve,dev]"

# 1. Full pipeline on synthetic data, CPU only, ~30 s
python -m cxr_triage.train --config configs/smoke.yaml
python -m cxr_triage.evaluate runs/smoke
pytest -q

# 2. Real training (GPU). Overrides use dotted keys.
python -m cxr_triage.train --config configs/default.yaml --resume train.max_minutes=100
python -m cxr_triage.evaluate runs/densenet121_128

# 3. Serve the API + demo UI, then open http://localhost:8080
CXR_BUNDLE=runs/densenet121_128/bundle.pt uvicorn cxr_triage.serve:app --port 8080
curl -F "file=@xray.png" "localhost:8080/predict?explain=true"

# 4. Docker
cp runs/densenet121_128/bundle.pt bundle.pt
docker build -t cxr-triage . && docker run -p 8080:8080 cxr-triage
```

## Live demo on Google Cloud Run

The web UI at `/` lets anyone drop in an X-ray or click one of the bundled held-out test images.
It shows the triage decision, calibrated probabilities against their thresholds, and a Grad-CAM
heatmap. API docs are at `/docs`.

```bash
gcloud auth login && gcloud config set project YOUR_PROJECT_ID
deploy/cloudrun.sh runs/densenet121_128/bundle.pt      # prints the public URL
```

The service runs on CPU, scales to zero (no cost while idle, with a cold start of about
10–20 s), and is capped at 3 instances. To test the deploy before your GPU run, deploy the
smoke bundle (`runs/smoke/bundle.pt`). It runs end to end, but it shows synthetic noise
images.

On Colab, use [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb). It does all
non-GPU work on a free CPU runtime first.

## Repo layout

```
src/cxr_triage/
  data.py         MedMNIST loader, CXR-safe augmentation (no flips), synthetic test data
  model.py        DenseNet / ResNet / EfficientNet backbones
  train.py        AMP training, time budget, resume, early stopping
  calibration.py  per-label temperature scaling
  metrics.py      AUROC + bootstrap CI, ECE, thresholds, risk-coverage
  evaluate.py     report + deployable bundle
  explain.py      Grad-CAM
  inference.py    Predictor + triage policy
  serve.py        FastAPI app (API + demo UI)
  static/         demo web page (no external dependencies)
configs/          default.yaml (GPU), smoke.yaml (CPU)
deploy/           cloudrun.sh: one-command Cloud Run deploy
docs/             MODEL_CARD.md, GPU_BUDGET.md, CAREER_PLAYBOOK.md
```

## Limitations

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). In short: NIH labels were text-mined from
reports and are noisy, the data comes from a single institution, low resolution limits
detection of small findings, and Grad-CAM shows correlation, not causation.
