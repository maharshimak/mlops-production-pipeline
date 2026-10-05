import pytest

from mlops_pipeline.pipeline import run_regression_pipeline
from mlops_pipeline.tracking import SQLiteRunLedger


def test_pipeline_run_is_reproducible_and_emits_verifiable_metadata() -> None:
    xs = [float(value) for value in range(1, 21)]
    ys = [2.0 * value + 1.0 for value in xs]

    first = run_regression_pipeline(xs, ys, seed=7, max_mae=0.01, max_psi=10.0)
    second = run_regression_pipeline(xs, ys, seed=7, max_mae=0.01, max_psi=10.0)

    assert first.model == second.model
    assert first.train_indices == second.train_indices
    assert first.eval_indices == second.eval_indices
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.config_fingerprint == second.config_fingerprint
    assert first.artifact_payload == second.artifact_payload
    assert first.manifest.mae < 1e-9
    assert first.decision.allowed is True


def test_dataset_fingerprint_changes_when_training_data_changes() -> None:
    xs = [float(value) for value in range(1, 13)]
    ys = [3.0 * value for value in xs]

    original = run_regression_pipeline(xs, ys, max_psi=10.0)
    changed = run_regression_pipeline(xs, [*ys[:-1], ys[-1] + 1.0], max_psi=10.0)

    assert original.dataset_fingerprint != changed.dataset_fingerprint


def test_pipeline_rejects_tiny_datasets() -> None:
    with pytest.raises(ValueError, match="six"):
        run_regression_pipeline([1, 2, 3], [2, 4, 6])


def test_default_gate_requires_meaningful_evaluation_sample():
    xs = [float(value) for value in range(1, 13)]
    ys = [2.0 * value for value in xs]

    run = run_regression_pipeline(xs, ys, max_mae=0.01, max_psi=10.0)

    assert run.manifest.eval_rows < 5
    assert run.decision.allowed is False
    assert any("eval_rows" in reason for reason in run.decision.reasons)


def test_eval_gate_threshold_is_part_of_reproducibility_config():
    xs = [float(value) for value in range(1, 25)]
    ys = [2.0 * value for value in xs]

    loose = run_regression_pipeline(xs, ys, max_psi=10.0, min_eval_rows=2)
    strict = run_regression_pipeline(xs, ys, max_psi=10.0, min_eval_rows=5)

    assert loose.config_fingerprint != strict.config_fingerprint


def test_pipeline_manifest_includes_cross_validation_stability() -> None:
    xs = [float(value) for value in range(1, 31)]
    ys = [2.0 * value + 1.0 for value in xs]

    run = run_regression_pipeline(
        xs,
        ys,
        seed=11,
        max_mae=0.01,
        max_psi=10.0,
        min_eval_rows=5,
    )

    assert run.manifest.cv_mean_mae is not None
    assert run.manifest.cv_worst_mae is not None
    assert run.manifest.cv_worst_mae < 0.01
    assert run.decision.allowed


def test_pipeline_blocks_unstable_cross_validation_even_if_holdout_is_permitted() -> None:
    xs = [float(value) for value in range(1, 31)]
    ys = [2.0 * value + 1.0 for value in xs]
    ys[7] += 20.0
    ys[18] -= 15.0

    run = run_regression_pipeline(
        xs,
        ys,
        seed=7,
        max_mae=100.0,
        max_psi=10.0,
        min_eval_rows=5,
        max_cv_worst_mae=1.0,
    )

    assert not run.decision.allowed
    assert any("cv_worst_mae" in reason for reason in run.decision.reasons)


def test_pipeline_tracks_gate_rejection_separately_from_execution_failure(tmp_path) -> None:
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    xs = [float(value) for value in range(1, 13)]
    ys = [2.0 * value for value in xs]

    run = run_regression_pipeline(
        xs,
        ys,
        max_mae=0.01,
        max_psi=10.0,
        ledger=ledger,
    )

    assert not run.decision.allowed
    assert run.tracking_run_id is not None
    assert ledger.get_run(run.tracking_run_id).status == "rejected"


def test_pipeline_records_runtime_failure_after_run_identity_exists(tmp_path) -> None:
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    xs = [1.0] * 10
    ys = [float(value) for value in range(10)]

    with pytest.raises(ValueError, match="variance"):
        run_regression_pipeline(xs, ys, ledger=ledger, run_name="constant-feature")

    runs = ledger.list_runs(name="constant-feature")
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].dataset_fingerprint
    assert runs[0].config_fingerprint
