import pytest

from mlops_pipeline.core import psi, quality_gate, train


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_bad_metrics_never_pass(value):
    assert not quality_gate(value, 1)


def test_invalid_training_observations():
    with pytest.raises(ValueError):
        train([1, float("nan")], [1, 2])


def test_psi_normalizes_counts_and_rejects_invalid_bins():
    assert psi([1, 2], [100, 200]) == pytest.approx(0)
    with pytest.raises(ValueError):
        psi([-1, 2], [1, 2])
    with pytest.raises(ValueError):
        psi([], [])
