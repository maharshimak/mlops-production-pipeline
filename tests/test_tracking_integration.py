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
