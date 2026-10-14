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


def test_concurrent_ledger_writers_keep_each_metric(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    db = tmp_path / "concurrent.sqlite"
    original = SQLiteRunLedger(str(db))
    run_id = original.start_run(
        "regression", dataset_fingerprint="d", config_fingerprint="c",
    )
    def append(index):
        # SQLite connections belong to their creating thread. Each writer uses
        # an independent connection, as a real concurrent worker would.
        worker = SQLiteRunLedger(str(db))
        try:
            worker.log_metric(run_id, f"metric_{index}", float(index))
            worker.log_artifact(run_id, f"artifact_{index}", f"sha256:{index}")
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(8)))

    record = original.get_run(run_id)
    assert len(record.metrics) == 8
    assert len(record.artifacts) == 8
    assert {f"metric_{i}" for i in range(8)} == set(record.metrics)
    assert {f"artifact_{i}" for i in range(8)} == set(record.artifacts)
    original.close()


def test_ledger_failed_json_update_rolls_back_transaction(tmp_path):
    import sqlite3

    import pytest

    db = tmp_path / "rollback.sqlite"
    ledger = SQLiteRunLedger(str(db))
    run_id = ledger.start_run(
        "repro", dataset_fingerprint="d", config_fingerprint="c",
    )
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TRIGGER fail_metric BEFORE UPDATE OF metrics_json ON runs "
            "BEGIN SELECT RAISE(ABORT, 'metrics read-only'); END"
        )

    with pytest.raises(sqlite3.DatabaseError, match="metrics read-only"):
        ledger.log_metric(run_id, "accuracy", 0.99)

    assert ledger.get_run(run_id).metrics == {}
    ledger.log_artifact(run_id, "model", "sha256:untouched")
    assert ledger.get_run(run_id).artifacts["model"] == "sha256:untouched"
    ledger.close()


def test_training_failure_is_durably_registered_before_compute(tmp_path, monkeypatch):
    import pytest
    import mlops_pipeline.pipeline as module

    ledger = SQLiteRunLedger(str(tmp_path / "early-failure.sqlite"))
    def fail_training(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(module, "train", fail_training)
    xs = [float(i) for i in range(1, 21)]
    with pytest.raises(RuntimeError, match="simulated training failure"):
        run_regression_pipeline(xs, [2*x+1 for x in xs], ledger=ledger)

    record = ledger.connection.execute("SELECT run_id FROM runs").fetchone()
    assert record is not None
    persisted = ledger.get_run(record["run_id"])
    assert persisted.status == "failed"
    assert persisted.metrics == {}
    ledger.close()


def test_abandoned_run_recovery_is_opt_in_and_idempotent(tmp_path):
    import pytest

    db = tmp_path / "abandoned.sqlite"
    ledger = SQLiteRunLedger(str(db))
    abandoned = ledger.start_run(
        "abandoned", dataset_fingerprint="data", config_fingerprint="config"
    )
    active = ledger.start_run(
        "active", dataset_fingerprint="data", config_fingerprint="config"
    )
    ledger.connection.execute(
        "UPDATE runs SET updated_at = datetime('now', '-2 days') WHERE run_id = ?",
        (abandoned,),
    )
    ledger.connection.commit()
    ledger.close()

    reopened = SQLiteRunLedger(str(db))
    assert reopened.get_run(abandoned).status == "running"
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            reopened.recover_stale_runs(older_than_seconds=invalid)
    assert reopened.recover_stale_runs(older_than_seconds=3600) == 1
    assert reopened.get_run(abandoned).status == "failed"
    assert reopened.get_run(active).status == "running"
    assert reopened.recover_stale_runs(older_than_seconds=3600) == 0
    reopened.close()
