from dataclasses import dataclass
from math import isfinite, log


@dataclass(frozen=True)
class LinearModel:
    slope: float
    intercept: float
    version: str


def train(xs: list[float], ys: list[float], version: str = "0.1.0") -> LinearModel:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Need paired observations.")
    if not all(isfinite(v) for v in [*xs, *ys]):
        raise ValueError("Observations must be finite.")
    x_bar, y_bar = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("Feature variance must be non-zero.")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom
    return LinearModel(slope=slope, intercept=y_bar - slope * x_bar, version=version)


def predict(model: LinearModel, x: float) -> float:
    return model.slope * x + model.intercept


def mae(model: LinearModel, xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Need non-empty paired observations.")
    if not all(isfinite(v) for v in [*xs, *ys, model.slope, model.intercept]):
        raise ValueError("Observations and model coefficients must be finite.")
    return sum(abs(predict(model, x) - y) for x, y in zip(xs, ys)) / len(xs)


def quality_gate(metric: float, maximum: float) -> bool:
    return isfinite(metric) and isfinite(maximum) and 0 <= metric <= maximum


def psi(expected: list[float], actual: list[float], epsilon: float = 1e-6) -> float:
    if len(expected) != len(actual) or not expected:
        raise ValueError("Distributions must use the same non-empty bins.")
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if any(not isfinite(v) or v < 0 for v in [*expected, *actual]):
        raise ValueError("Bin counts must be finite and non-negative.")
    if sum(expected) <= 0 or sum(actual) <= 0:
        raise ValueError("Distribution totals must be positive.")
    expected = [v / sum(expected) for v in expected]
    actual = [v / sum(actual) for v in actual]
    total = 0.0
    for expected_value, actual_value in zip(expected, actual):
        e = max(expected_value, epsilon)
        a = max(actual_value, epsilon)
        total += (a - e) * log(a / e)
    return total
