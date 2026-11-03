import pytest

from mlops_pipeline.core import psi
from mlops_pipeline.drift import feature_drift


def test_identical_samples_have_zero_psi() -> None:
    sample = [[float(i), float(i % 4)] for i in range(50)]
    result = feature_drift(["a", "b"], sample, sample)
    assert [item.psi for item in result] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_bins_are_fitted_on_reference_and_hand_computable() -> None:
    reference = [[0.0], [1.0], [2.0], [3.0]]
    current = [[3.0], [3.0], [3.0], [10.0]]
    (item,) = feature_drift(["a"], reference, current, bins=2)
    # reference range [0, 3], width 1.5: bins [0,1.5) and [1.5, 3] (+ out of range clipped)
    assert item.reference_counts == (2.0, 2.0)
    assert item.current_counts == (0.0, 4.0)
    assert item.psi == pytest.approx(psi([2.0, 2.0], [0.0, 4.0]))


def test_constant_reference_feature_still_detects_shift() -> None:
    reference = [[5.0]] * 10
    (same,) = feature_drift(["a"], reference, [[5.0]] * 10)
    (shifted,) = feature_drift(["a"], reference, [[9.0]] * 10)
    assert same.psi == pytest.approx(0.0)
    assert shifted.current_counts == (0.0, 0.0, 10.0)
    assert shifted.psi > 1.0


@pytest.mark.parametrize(
    "names,reference,current,bins",
    [
        (["a"], [], [[1.0]], 5),
        (["a"], [[1.0]], [[float("nan")]], 5),
        (["a", "b"], [[1.0, 2.0]], [[1.0]], 5),
        (["a"], [[1.0]], [[1.0]], 1),
    ],
)
def test_invalid_drift_inputs(names, reference, current, bins) -> None:
    with pytest.raises(ValueError):
        feature_drift(names, reference, current, bins=bins)
