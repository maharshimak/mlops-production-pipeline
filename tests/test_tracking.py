import math

import pytest

from mlops_pipeline.tracking import SQLiteRunLedger


def _start(ledger: SQLiteRunLedger, name: str = "fraud-training") -> str:
    return ledger.start_run(
        name,
        dataset_fingerprint="data123",
        config_fingerprint="cfg456",
        params={"seed": 42, "model": "linear"},
    )


def test_run_ledger_tracks_metrics_artifacts_and_status(tmp_path):
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    run_id = _start(ledger)
    ledger.log_metric(run_id, "mae", 0.12)
    ledger.log_artifact(run_id, "model", "file:///models/fraud.json")
    ledger.finish_run(run_id)

    run = ledger.get_run(run_id)
    assert run.status == "succeeded"
    assert run.metrics["mae"] == 0.12
    assert run.artifacts["model"].endswith("fraud.json")
    assert run.params["seed"] == 42


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_run_ledger_rejects_non_finite_or_boolean_metrics(tmp_path, value):
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    run_id = _start(ledger)

    with pytest.raises(ValueError, match="finite numeric"):
        ledger.log_metric(run_id, "mae", value)


def test_terminal_run_cannot_be_mutated_or_reopened(tmp_path):
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    run_id = _start(ledger)
    ledger.finish_run(run_id, status="rejected")

    with pytest.raises(ValueError, match="already terminal"):
        ledger.log_metric(run_id, "mae", 0.2)
    with pytest.raises(ValueError, match="already terminal"):
        ledger.log_artifact(run_id, "model", "sha256:abc")
    with pytest.raises(ValueError, match="already terminal"):
        ledger.finish_run(run_id, status="succeeded")

    assert ledger.get_run(run_id).status == "rejected"


def test_run_ledger_lists_recent_runs_with_optional_name_filter(tmp_path):
    ledger = SQLiteRunLedger(str(tmp_path / "runs.sqlite"))
    first = _start(ledger, "first")
    ledger.finish_run(first)
    second = _start(ledger, "second")
    ledger.finish_run(second, status="failed")

    assert [run.run_id for run in ledger.list_runs(limit=1)] == [second]
    assert [run.run_id for run in ledger.list_runs(name="first")] == [first]
