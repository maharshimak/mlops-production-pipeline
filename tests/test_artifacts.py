import math

import pytest

from mlops_pipeline.artifacts import RunManifest, build_run_manifest, deployment_gate
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


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0, True])
def test_deployment_gate_rejects_invalid_quality_thresholds(value) -> None:
    model = LinearModel(slope=2.0, intercept=1.0, version="1.2.0")
    manifest = build_run_manifest(model, train_rows=100, eval_rows=50, mae=0.08, drift_psi=0.05)

    with pytest.raises(ValueError):
        deployment_gate(manifest, max_mae=value, max_psi=0.1)


def test_deployment_gate_validates_external_manifest_before_deciding() -> None:
    malformed = RunManifest(
        model_version="1.2.0",
        model_fingerprint="sha",
        train_rows=100,
        eval_rows=20,
        mae=math.nan,
        drift_psi=0.05,
    )

    with pytest.raises(ValueError, match="mae"):
        deployment_gate(malformed, max_mae=0.1, max_psi=0.1)
