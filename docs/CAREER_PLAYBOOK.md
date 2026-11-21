# Big Tech playbook: from this repo to an offer

Target: ML Engineer, Applied Scientist, or SWE (ML) roles at Google, Microsoft, Meta, Amazon,
or Apple, including their health teams (Google Health/DeepMind, Microsoft Health & Life
Sciences/Nuance, Amazon One Medical/AWS HealthAI, Apple Health).

## What Big Tech interviewers actually check

| Signal | What they want to see | Where it lives |
|---|---|---|
| **Shipped, working software** | A link they can click | Cloud Run demo (`deploy/`) |
| **Experiment rigor** | Baselines, ablations, CIs, no test leakage | `evaluate.py`, Results table |
| **Engineering quality** | Tests, CI, clean modules, configs | `tests/`, `.github/workflows/` |
| **Systems and cost awareness** | Budgets, latency, scaling, failure modes | `train.py`, `docs/GPU_BUDGET.md` |
| **Modern ML fluency** | Foundation models and LLMs, used safely | Phase 3 below |
| **Product judgment** | Why each design choice matters to users | README "Why this project is different" |

Responsible AI matters at all of these companies, but there it is a *quality* signal, not the
headline. Lead with results and the demo; bring up calibration and failure modes in the deep dive.

## Roadmap: about 5 weeks, committed in small steps

A steady commit history (several commits a week) is itself a signal. One giant commit looks
copied.

| Phase | Deliverable | GPU cost | Why it matters for Big Tech |
|---|---|---|---|
| **1. Real results** | Main run + one ablation; README results table, plots, Grad-CAM gallery | ~4 h | Proves you can run and report experiments honestly |
| **2. Live demo** (started) | Public Cloud Run URL with sample X-rays, linked at the top of the README and on your resume | 0 | Recruiters spend about 30 seconds per resume, and a link gets clicked |
| **3. GenAI layer** (built) | Run `report_eval` on the free Gemini tier and fill in the README table: first-draft pass rate, hallucination/omission rates, fallback rate. Optional: add a second model for a cost-vs-quality comparison | 0 | The highest-demand skill right now: LLMs plus classic ML plus evals |
| **4. Robustness** | Run the trained model on a second dataset (CheXpert or MIMIC-CXR subset); report the accuracy drop and analyze why | ~0.5 h (inference only) | Distribution shift is a favorite ML-interview topic |
| **5. Efficiency** (pick one) | Linear probe on a medical foundation model's embeddings vs. the fine-tuned DenseNet, **or** int8 quantization / ONNX export with a latency benchmark on Cloud Run | 0–0.5 h | Shows accuracy-vs-cost trade-off thinking, which is core at scale |
| Stretch | Subgroup accuracy by sex and age | 0 | Good for health-team interviews |

Write a short blog post (Medium or a GitHub Pages site) after phases 2 and 3. Share it on
LinkedIn. Hiring managers read these.

## Resume bullets (fill in real numbers; never inflate)

**Project: CXR-Triage, trustworthy chest X-ray triage** · [demo link] · [GitHub link]
- Built and deployed an end-to-end chest X-ray triage service (PyTorch, FastAPI, Docker, Cloud
  Run) on 112k NIH radiographs, reaching [0.XX] test macro-AUROC (95% CI [a–b]) across
  14 findings.
- Designed uncertainty-based referral that raised AUROC on auto-read cases from [a] to [b] at
  80% coverage, and cut calibration error [X]% with per-label temperature scaling.
- Trained within a 6-GPU-hour budget using mixed precision, wall-clock limits, and
  preemption-safe resume; CPU-only CI validated every change before spending GPU time.
- [Phase 3] Added an LLM report-drafting layer where code verifies every draft: measured a
  [a]% first-draft hallucination rate on [n] held-out cases and guaranteed 0% in delivered
  reports through verification, repair, and template fallback, at $0 using the Gemini free tier.

Put the project **above** coursework on your resume, with the demo link in the title line.

## Interview prep

Big Tech loops have three parts. This project covers one of them; you still need the other two.

**1. Coding (biggest filter; start now, in parallel).** LeetCode mediums in Python, about
150 problems covering arrays, hashing, two pointers, trees, graphs, heaps, and DP. Practice
explaining aloud. Aim for 1–2 problems a day.

**2. ML system design (this project is your story).** Practice: *"Design an AI system that
prioritizes a hospital radiology worklist."* Walk through:
- Requirements: latency, the cost of a miss vs. a false alarm, human-in-the-loop
- Data: label noise, patient-level splits, class imbalance
- Model and training: pretrained backbone, imbalance weighting, compute budget
- Evaluation: AUROC with CIs, calibration, operating point at a target sensitivity
- Serving: API, batching, model versioning, rollback
- Monitoring: input drift, calibration drift, referral rate, feedback from radiologist overrides

Rehearse it as a 10-minute and a 30-minute version.

**3. Behavioral / project deep dive.** Use STAR format. Have stories ready for: a hard bug
(e.g. why no horizontal flips), a trade-off you made under constraints (the GPU budget), and
what you'd do differently.

**Questions you must answer cold**
- Why isn't AUROC enough? (It ignores calibration and the operating point; clinicians act on
  thresholds.)
- Why fit temperature and thresholds on validation, not test? (Using test leaks information
  and inflates numbers.)
- Why no horizontal flips? (They swap laterality: the heart sits on the left.)
- What does referring 20% of cases cost, and what does it buy? (Radiologist time vs. the gain
  shown on the risk-coverage curve.)
- Known ChestX-ray14 pitfalls? (Noisy text-mined labels, chest-tube shortcuts for
  pneumothorax, single-site data.)
- How would you scale this to 1M images a day? (Batching, GPU vs. CPU inference cost,
  autoscaling, caching, async queues.)
- With 10x the compute, what would you do? (Higher resolution, multi-site training, external
  validation, foundation-model initialization.)

## Applying

- **Referrals beat cold applications** by a wide margin. Message alumni from your
  university and LinkedIn contacts at target teams with your demo link and a two-line pitch.
- Apply to new-grad/intern ML roles **early in the cycle** (Aug–Oct for the following year).
- Health teams to target by name: Google Health, Google DeepMind (health), Microsoft Health &
  Life Sciences, Nuance (Microsoft), AWS HealthAI / HealthLake, Amazon One Medical, Apple
  Health. Also consider strong health-AI companies with Big-Tech-level engineering bars as
  stepping stones.
