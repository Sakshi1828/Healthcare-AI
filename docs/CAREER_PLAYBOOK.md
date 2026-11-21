# Turning this project into interviews

## What hiring managers look for (and where it is in this repo)

| Signal | Big Tech (Google, Amazon, Meta, Microsoft, Apple) | Big 4 (Deloitte, PwC, EY, KPMG) AI practices | Where |
|---|---|---|---|
| Engineering quality | Tests, CI, typed modular code | Reproducibility | `tests/`, `.github/workflows/` |
| ML depth | Calibration, CIs, ablations | Sound validation | `metrics.py`, `evaluate.py` |
| Production thinking | API, Docker, resource budgets | Deployability | `serve.py`, `Dockerfile`, `train.py` |
| Responsible AI | Uncertainty, explainability | Governance, model risk, documentation | `MODEL_CARD.md`, referral policy |
| Communication | Design-doc style README | Executive summary, risk framing | `README.md` |

## Roadmap: ship in phases, and commit as you go

A steady commit history over several weeks reads better than one giant commit.

1. **Week 1: core (done in this repo).** Run the GPU plan, put real numbers in the README,
   add the plots and the Grad-CAM gallery.
2. **Week 2: live demo.** Deploy the Docker image to **Google Cloud Run** (CPU inference fits
   the free tier and needs no GPU) and add a small Gradio/Streamlit front end. A clickable link
   on your resume is worth a lot.
3. **Week 3: fairness and robustness audit** (pick one; both cost little or no GPU):
   - Subgroup AUROC by sex/age using the original NIH metadata (`Data_Entry_2017.csv`).
   - External validation: run the trained model on a CheXpert/MIMIC-CXR subset and report the
     drop. This is the strongest "I understand real-world ML" signal.
4. **Week 4: GenAI layer.** Turn the structured output (findings, probabilities, uncertainty)
   into a draft report with an LLM. Use strict guardrails: it may only mention findings above
   threshold and must state uncertainty. Evaluate it for hallucinated findings. This shows you
   can combine classical ML and LLMs safely.
5. **Stretch: foundation models.** Compare against a linear probe on a pretrained medical
   imaging foundation model's embeddings. Training a probe is CPU-cheap, which makes for a
   great "accuracy vs. compute" story.

## Resume bullets (replace the brackets with your real numbers; never inflate)

- Built an end-to-end chest X-ray triage system (PyTorch, FastAPI, Docker) on 112k NIH
  radiographs, reaching [0.XX] test macro-AUROC (95% CI [a–b]) across 14 findings.
- Cut calibration error [X]% with per-label temperature scaling. Added uncertainty-based
  referral that raised AUROC on auto-read cases from [a] to [b] at 80% coverage.
- Designed training for a 6-GPU-hour budget (mixed precision, wall-clock limits,
  preemption-safe resume), with a CPU-only CI pipeline that validated every change before
  using GPU time.
- Wrote a model card covering label noise, dataset shift, and shortcut-learning risks, and set
  operating thresholds at 90% sensitivity for urgent findings.

## Interview talking points (know these cold)

- **Why AUROC isn't enough:** it ignores calibration and the operating point. Clinicians act on
  thresholds, so you report sensitivity and specificity at a chosen threshold.
- **Why fit temperature and thresholds on val, not test:** otherwise you leak test information
  and report optimistic numbers.
- **Why no horizontal flips:** they swap laterality (the heart sits on the left), which creates
  anatomically impossible images.
- **Selective prediction trade-off:** referring 20% of cases costs radiologist time. The
  risk-coverage curve shows what that time buys.
- **Known ChestX-ray14 pitfalls:** noisy NLP labels, chest-tube shortcuts for pneumothorax,
  single-site data. Bringing these up yourself signals maturity.
- **What you'd do with more compute:** higher resolution, external validation, and
  multi-site training with domain generalization.
