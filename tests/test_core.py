from mlops_pipeline.core import mae, psi, quality_gate, train


def test_training_and_gate() -> None:
    model = train([1, 2, 3, 4], [3, 5, 7, 9])
    metric = mae(model, [5], [11])
    assert metric < 1e-9
    assert quality_gate(metric, 0.1)


def test_psi_detects_change() -> None:
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0
