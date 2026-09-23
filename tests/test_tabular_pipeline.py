from mlops_pipeline.tabular import (
    predict_tabular,
    run_tabular_regression_pipeline,
    train_tabular,
)


def test_multifeature_model_learns_linear_relationship() -> None:
    features = [[x, y] for x, y in [
        (1, 2), (2, 1), (3, 0), (0, 3), (4, 2), (2, 4), (5, 1), (1, 5)
    ]]
    targets = [3 * row[0] + 2 * row[1] + 5 for row in features]

    model = train_tabular(features, targets)

    assert abs(predict_tabular(model, [6, 2]) - 27) < 1e-5


def test_multifeature_pipeline_is_reproducible() -> None:
    features = [[float(i), float(i % 3)] for i in range(12)]
    targets = [2 * row[0] - 0.5 * row[1] + 3 for row in features]

    first = run_tabular_regression_pipeline(features, targets, seed=7, max_mae=0.01)
    second = run_tabular_regression_pipeline(features, targets, seed=7, max_mae=0.01)

    assert first.allowed
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.config_fingerprint == second.config_fingerprint
    assert first.artifact_fingerprint == second.artifact_fingerprint
    assert first.train_indices == second.train_indices
    assert first.eval_indices == second.eval_indices
