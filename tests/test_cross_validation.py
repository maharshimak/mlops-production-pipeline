from mlops_pipeline.cross_validation import kfold_regression_cv


def test_kfold_regression_cv_evaluates_every_fold():
    features = [[float(x), float(x % 3)] for x in range(20)]
    targets = [2.0 * x + 0.5 * (x % 3) + 1.0 for x in range(20)]

    result = kfold_regression_cv(features, targets, folds=5, seed=4)

    assert len(result.folds) == 5
    assert sum(fold.eval_rows for fold in result.folds) == 20
    assert result.mean_mae >= 0
    assert result.worst_mae >= result.mean_mae
