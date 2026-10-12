from mlops_pipeline.pipeline import run_regression_pipeline
from mlops_pipeline.tracking import SQLiteRunLedger


def test_pipeline_can_record_run_in_ledger(tmp_path):
    ledger = SQLiteRunLedger(str(tmp_path / "tracking.sqlite"))
    xs = [float(value) for value in range(1, 21)]
    ys = [2 * value + 1 for value in xs]

    run = run_regression_pipeline(
        xs,
        ys,
        max_mae=0.01,
        max_psi=10.0,
        ledger=ledger,
        run_name="tracked-regression",
    )

    assert run.tracking_run_id is not None
    tracked = ledger.get_run(run.tracking_run_id)
    assert tracked.name == "tracked-regression"
    assert tracked.metrics["mae"] < 1e-9
    assert tracked.artifacts["model"].startswith("sha256:")


def test_policy_rejection_has_distinct_terminal_status(tmp_path):
    ledger = SQLiteRunLedger(str(tmp_path / "rejected.sqlite"))
    xs = [float(v) for v in range(1, 21)]
    ys = [2 * x + 1 for x in xs]
    run = run_regression_pipeline(
        xs, ys,
        max_mae=0.01,
        max_psi=10.0,
        min_eval_rows=99,
        ledger=ledger,
    )
    assert run.decision.allowed is False
    tracked = ledger.get_run(run.tracking_run_id)
    assert tracked.status == "rejected"


def test_tracking_terminal_state_is_immutable(tmp_path):
    import pytest

    ledger = SQLiteRunLedger(str(tmp_path / "terminal.sqlite"))
    run_id = ledger.start_run(
        "training",
        dataset_fingerprint="data",
        config_fingerprint="config",
    )
    ledger.log_metric(run_id, "mae", 0.5)
    ledger.finish_run(run_id, status="succeeded")

    with pytest.raises(ValueError, match="only running"):
        ledger.log_metric(run_id, "mae", 0.0)
    with pytest.raises(ValueError, match="only running"):
        ledger.log_artifact(run_id, "model", "new-model")
    with pytest.raises(ValueError, match="only running"):
        ledger.finish_run(run_id, status="failed")

    actual = ledger.get_run(run_id)
    assert actual.status == "succeeded"
    assert actual.metrics["mae"] == 0.5
    assert "model" not in actual.artifacts


def test_tracking_state_survives_database_restart(tmp_path):
    db_path = tmp_path / "durable.sqlite"
    writer = SQLiteRunLedger(str(db_path))
    run_id = writer.start_run(
        "training",
        dataset_fingerprint="data",
        config_fingerprint="config",
    )
    writer.log_artifact(run_id, "model", "sha256:abc")
    writer.finish_run(run_id, status="failed")
    writer.close()

    reader = SQLiteRunLedger(str(db_path))
    persisted = reader.get_run(run_id)
    assert persisted.status == "failed"
    assert persisted.artifacts["model"] == "sha256:abc"
    reader.close()
