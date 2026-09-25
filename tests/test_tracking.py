from mlops_pipeline.tracking import SQLiteRunLedger


def test_run_ledger_tracks_metrics_artifacts_and_status(tmp_path):
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    run_id = ledger.start_run(
        "fraud-training",
        dataset_fingerprint="data123",
        config_fingerprint="cfg456",
        params={"seed": 42, "model": "linear"},
    )
    ledger.log_metric(run_id, "mae", 0.12)
    ledger.log_artifact(run_id, "model", "file:///models/fraud.json")
    ledger.finish_run(run_id)

    run = ledger.get_run(run_id)
    assert run.status == "succeeded"
    assert run.metrics["mae"] == 0.12
    assert run.artifacts["model"].endswith("fraud.json")
    assert run.params["seed"] == "42"
