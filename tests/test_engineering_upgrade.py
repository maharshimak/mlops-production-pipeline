import pytest

from mlops_pipeline.artifacts import build_run_manifest, deployment_gate
from mlops_pipeline.core import psi, train


def test_overflow_distribution_is_rejected():
    with pytest.raises(ValueError):
        psi([1e308, 1e308], [1, 1])


def test_integrity_failure_blocks_accurate_model():
    model = train([1, 2, 3], [2, 4, 6])
    manifest = build_run_manifest(model, train_rows=3, eval_rows=20, mae=0, drift_psi=0)
    assert not deployment_gate(manifest, max_mae=1, max_psi=1, integrity_valid=False).allowed
