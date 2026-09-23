from mlops_pipeline.pipeline import run_regression_pipeline


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
    try:
        run_regression_pipeline([1, 2, 3], [2, 4, 6])
    except ValueError as error:
        assert "six" in str(error)
    else:
        raise AssertionError("tiny dataset should be rejected")
