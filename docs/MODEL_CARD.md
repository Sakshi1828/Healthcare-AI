# Model card: CXR-Triage

**Intended use.** A research and educational demonstration of calibrated, uncertainty-aware
triage for chest radiographs. **Not** for diagnosis, clinical decision-making, or any use on
real patients.

**Data.** NIH ChestX-ray14 (Wang et al., 2017), accessed through MedMNIST+ (Yang et al., 2023):
112,120 frontal radiographs from 30,805 patients, 14 labels mined from radiology reports using
NLP. The official MedMNIST splits are patient-disjoint: train 78,468 / val 11,219 / test 22,433.

**Model.** An ImageNet-pretrained DenseNet-121 with a 14-way sigmoid head, trained with
class-weighted BCE. Per-label temperature scaling and decision thresholds are fit on validation
data only.

**Metrics.** Test-set macro-AUROC with per-label bootstrap 95% CIs, AUPRC, ECE, and
sensitivity/specificity at thresholds chosen on validation for 90% sensitivity, plus urgent-case
sensitivity and a risk-coverage curve. Results: see `README.md` → Results.

**Known limitations and risks**
- Label noise: text-mined labels have an estimated error rate of 10–20% for some findings.
  AUROC is measured against these noisy labels, not against radiologist ground truth.
- Single institution (NIH Clinical Center): expect performance to drop on other hospitals,
  scanners, and populations. External validation (e.g. CheXpert, MIMIC-CXR) is not yet done.
- Shortcut learning: models trained on ChestX-ray14 are known to latch onto chest tubes
  (pneumothorax) and text markers. Grad-CAM helps spot this but does not rule it out.
- Subgroup performance: MedMNIST does not ship age/sex metadata, so no fairness audit was
  possible. This is on the roadmap using the original NIH metadata.
- Downsampled images (128–224 px) reduce sensitivity to small nodules.
- Calibration is fit on the same distribution. It will not hold under dataset shift.

**Human oversight.** The design assumes a radiologist reviews every case. The model only
reorders the worklist and marks cases where it is uncertain.
