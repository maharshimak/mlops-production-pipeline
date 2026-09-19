from mlops_pipeline.artifacts import build_run_manifest, deployment_gate
from mlops_pipeline.core import LinearModel


def test_run_manifest_is_reproducible_and_gateable() -> None:
    model = LinearModel(slope=2.0, intercept=1.0, version="1.2.0")
    first = build_run_manifest(model, train_rows=100, eval_rows=50, mae=0.08, drift_psi=0.05)
    second = build_run_manifest(model, train_rows=100, eval_rows=50, mae=0.08, drift_psi=0.05)
    assert first.model_fingerprint == second.model_fingerprint
    assert deployment_gate(first, max_mae=0.1, max_psi=0.1).allowed is True


def test_deployment_gate_explains_failures() -> None:
    model = LinearModel(slope=2.0, intercept=1.0, version="1.2.0")
    manifest = build_run_manifest(model, train_rows=100, eval_rows=5, mae=0.2, drift_psi=0.3)
    decision = deployment_gate(manifest, max_mae=0.1, max_psi=0.2, min_eval_rows=20)
    assert decision.allowed is False
    assert len(decision.reasons) == 3
