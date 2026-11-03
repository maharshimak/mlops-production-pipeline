import pytest

from mlops_pipeline.pipeline import run_regression_pipeline
from mlops_pipeline.tabular import run_tabular_regression_pipeline


def test_tabular_dataset_fingerprint_ignores_int_float_spelling() -> None:
    as_int = run_tabular_regression_pipeline(
        [[i, i % 3] for i in range(12)], [2 * i for i in range(12)]
    )
    as_float = run_tabular_regression_pipeline(
        [[float(i), float(i % 3)] for i in range(12)], [2.0 * i for i in range(12)]
    )
    assert as_int.model == as_float.model
    assert as_int.dataset_fingerprint == as_float.dataset_fingerprint
    assert as_int.artifact_fingerprint == as_float.artifact_fingerprint


def test_regression_dataset_fingerprint_ignores_int_float_spelling() -> None:
    as_int = run_regression_pipeline(list(range(1, 13)), [2 * x for x in range(1, 13)],
                                     max_psi=10.0)
    as_float = run_regression_pipeline([float(x) for x in range(1, 13)],
                                       [2.0 * x for x in range(1, 13)], max_psi=10.0)
    assert as_int.dataset_fingerprint == as_float.dataset_fingerprint
    assert as_int.artifact_payload == as_float.artifact_payload


def test_tabular_pipeline_reports_its_own_training_split_minimum() -> None:
    features = [[float(i), float(i % 2)] for i in range(5)]
    targets = [2.0 * i + 1.0 for i in range(5)]
    with pytest.raises(ValueError, match="Training split must contain at least four rows"):
        run_tabular_regression_pipeline(features, targets)
