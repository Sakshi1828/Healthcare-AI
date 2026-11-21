import numpy as np

from cxr_triage.calibration import apply_temperature, fit_temperature
from cxr_triage.metrics import (
    binary_stats,
    bootstrap_auroc_ci,
    expected_calibration_error,
    macro_auroc,
    risk_coverage,
    sigmoid,
    threshold_at_sensitivity,
)


def test_macro_auroc_perfect_and_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (500, 3))
    assert macro_auroc(y, y.astype(float)) == 1.0
    assert abs(macro_auroc(y, rng.random((500, 3))) - 0.5) < 0.08


def test_macro_auroc_skips_single_class_labels():
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    p = np.array([[0.1, 0.5], [0.9, 0.5], [0.2, 0.5], [0.8, 0.5]])
    assert macro_auroc(y, p) == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    lo, hi = bootstrap_auroc_ci(y, p, n=200)
    from sklearn.metrics import roc_auc_score

    assert lo <= roc_auc_score(y, p) <= hi


def test_threshold_meets_target_sensitivity():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    p = np.clip(0.3 * y + rng.random(1000) * 0.7, 0, 1)
    t = threshold_at_sensitivity(y, p, 0.9)
    assert binary_stats(y, p, t)["sensitivity"] >= 0.9


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(3)
    true_logits = rng.normal(0, 1.5, (4000, 2))
    y = (rng.random((4000, 2)) < sigmoid(true_logits)).astype(float)
    overconfident = true_logits * 3.0
    t = fit_temperature(overconfident, y)
    assert np.allclose(t, 3.0, atol=0.4)
    before = expected_calibration_error(y, sigmoid(overconfident))
    after = expected_calibration_error(y, sigmoid(apply_temperature(overconfident, t)))
    assert after < before / 2


def test_risk_coverage_improves_when_uncertainty_is_informative():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, (2000, 2))
    conf = rng.random(2000) < 0.5  # half the cases are "easy"
    p = np.where(conf[:, None], np.where(y == 1, 0.97, 0.03), rng.random((2000, 2)) * 0.2 + 0.4)
    rc = risk_coverage(y, p, coverages=[1.0, 0.5])
    assert rc["macro_auroc"][1] > rc["macro_auroc"][0]
