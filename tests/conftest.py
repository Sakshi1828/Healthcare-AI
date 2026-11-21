import pytest

from cxr_triage.config import load_config
from cxr_triage.evaluate import evaluate
from cxr_triage.train import train


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory):
    """Train + evaluate a tiny model on synthetic data once per test session."""
    run = tmp_path_factory.mktemp("run")
    cfg = load_config(
        "configs/smoke.yaml", [f"run_dir={run}", "train.epochs=3", "eval.bootstrap=20"]
    )
    train(cfg)
    report = evaluate(run)
    return run, report
