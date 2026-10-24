from math import isfinite

import pytest

from mlops_pipeline.tabular import predict_tabular, train_tabular


@pytest.mark.parametrize("features,targets", [
    ([[1e308]] * 4, [1.0, 2.0, 3.0, 4.0]),
    ([[1.0], [2.0], [3.0], [4.0]], [1e308] * 4),
    ([[-1e200], [1e200], [-2e200], [2e200]], [1.0, 2.0, 3.0, 4.0]),
])
def test_training_rejects_nonfinite_derived_parameters(features, targets):
    with pytest.raises(ValueError, match="finite"):
        train_tabular(features, targets)


def test_finite_training_and_prediction_still_work_with_constant_feature():
    features = [[float(index), 10.0] for index in range(8)]
    model = train_tabular(features, [3.0 * index + 5.0 for index in range(8)])
    assert all(isfinite(value) for value in (*model.weights, model.intercept))
    assert predict_tabular(model, [8.0, 10.0]) == pytest.approx(29.0)
