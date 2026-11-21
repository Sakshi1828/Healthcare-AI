# Spending 6 GPU hours well

Rule #1: **never debug on a GPU.** Everything in this repo runs on CPU with
`configs/smoke.yaml` and synthetic data, and CI runs it on every push. By the time you connect
a GPU, the code is already known to work.

Rule #2: **a connected-but-idle runtime still burns budget.** Use "Disconnect and delete
runtime" the moment a run ends. Download data on a CPU runtime (Part A of the notebook).

## Plan (≈ 6 h)

| # | Run | Purpose | Budget |
|---|---|---|---|
| 0 | Smoke test + data download (CPU runtime) | Catch bugs, cache data on Drive | 0 GPU min |
| 1 | DenseNet-121 @128, `train.epochs=2` | Real-data sanity check, measure min/epoch | 20 min |
| 2 | DenseNet-121 @128, 12 epochs, `max_minutes=100` | **Main result** | 100 min |
| 3 | One ablation: **either** @224 (accuracy) **or** EfficientNet-B0 (speed/size) | Shows you can run experiments | 120 min |
| 4 | Evaluate runs 2 and 3, Grad-CAM gallery | Report figures | 20 min |
| – | Buffer for disconnects and mistakes | | 100 min |

Pick the epoch count for run 2 from the per-epoch time `metrics.jsonl` logged in run 1. If
run 1 shows about 6 min/epoch, 12 epochs fits in about 75 min.

## If you have GCP credits instead of Colab

Use a Vertex AI Workbench instance or a Compute Engine VM with one T4/L4. Set an **idle
shutdown** (Workbench → 60 min) and a **budget alert** in Billing. Stop the VM after each run.
The same commands work: `python -m cxr_triage.train ... train.max_minutes=100`. Write
`run_dir` to a GCS-fuse mount or `gsutil cp` the run folder afterwards. A Spot VM is about
60–70% cheaper, and `--resume` makes preemption harmless.
