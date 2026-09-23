from __future__ import annotations

import json
import random
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite, sqrt


@dataclass(frozen=True, slots=True)
class TabularLinearModel:
    weights: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    version: str


@dataclass(frozen=True, slots=True)
class TabularPipelineRun:
    model: TabularLinearModel
    eval_mae: float
    allowed: bool
    dataset_fingerprint: str
    config_fingerprint: str
    artifact_fingerprint: str
    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]


def _validate_matrix(
    features: list[list[float]],
    targets: list[float],
    *,
    min_rows: int = 1,
) -> int:
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError("min_rows must be a positive integer.")
    if len(features) != len(targets) or len(features) < min_rows:
        raise ValueError(f"Need at least {min_rows} feature rows paired with targets.")
    if not features or not features[0]:
        raise ValueError("Feature matrix must contain at least one feature.")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise ValueError("All feature rows must have the same width.")
    values = [value for row in features for value in row] + list(targets)
    if not all(type(value) in (int, float) and isfinite(value) for value in values):
        raise ValueError("Features and targets must be finite numbers.")
    return width


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Training matrix is singular; increase ridge regularization.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def train_tabular(
    features: list[list[float]],
    targets: list[float],
    *,
    version: str = "1.0.0",
    ridge: float = 1e-8,
) -> TabularLinearModel:
    width = _validate_matrix(features, targets, min_rows=4)
    if type(ridge) not in (int, float) or not isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative.")

    means = [
        sum(row[column] for row in features) / len(features)
        for column in range(width)
    ]
    scales: list[float] = []
    for column, mean in enumerate(means):
        variance = sum((row[column] - mean) ** 2 for row in features) / len(features)
        scale = sqrt(variance)
        scales.append(scale if scale > 1e-12 else 1.0)

    design = [
        [1.0]
        + [
            (row[column] - means[column]) / scales[column]
            for column in range(width)
        ]
        for row in features
    ]
    parameter_count = width + 1
    gram = [[0.0] * parameter_count for _ in range(parameter_count)]
    rhs = [0.0] * parameter_count
    for row, target in zip(design, targets, strict=True):
        for left in range(parameter_count):
            rhs[left] += row[left] * target
            for right in range(parameter_count):
                gram[left][right] += row[left] * row[right]
    for index in range(1, parameter_count):
        gram[index][index] += ridge

    parameters = _solve(gram, rhs)
    return TabularLinearModel(
        weights=tuple(parameters[1:]),
        intercept=parameters[0],
        feature_means=tuple(means),
        feature_scales=tuple(scales),
        version=version,
    )


def predict_tabular(model: TabularLinearModel, features: list[float]) -> float:
    if len(features) != len(model.weights):
        raise ValueError("Feature count does not match the trained model.")
    if not all(type(value) in (int, float) and isfinite(value) for value in features):
        raise ValueError("Features must be finite numbers.")
    normalized = [
        (value - mean) / scale
        for value, mean, scale in zip(
            features,
            model.feature_means,
            model.feature_scales,
            strict=True,
        )
    ]
    return model.intercept + sum(
        weight * value for weight, value in zip(model.weights, normalized, strict=True)
    )


def tabular_mae(
    model: TabularLinearModel,
    features: list[list[float]],
    targets: list[float],
) -> float:
    _validate_matrix(features, targets, min_rows=1)
    return sum(
        abs(predict_tabular(model, row) - target)
        for row, target in zip(features, targets, strict=True)
    ) / len(targets)


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def run_tabular_regression_pipeline(
    features: list[list[float]],
    targets: list[float],
    *,
    version: str = "1.0.0",
    eval_fraction: float = 0.25,
    seed: int = 42,
    ridge: float = 1e-8,
    max_mae: float = 1.0,
) -> TabularPipelineRun:
    _validate_matrix(features, targets, min_rows=4)
    if not 0.1 <= eval_fraction <= 0.5:
        raise ValueError("eval_fraction must be between 0.1 and 0.5.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if type(max_mae) not in (int, float) or not isfinite(max_mae) or max_mae < 0:
        raise ValueError("max_mae must be finite and non-negative.")

    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    eval_count = max(1, ceil(len(indices) * eval_fraction))
    eval_indices = tuple(sorted(indices[:eval_count]))
    train_indices = tuple(sorted(indices[eval_count:]))
    if len(train_indices) < 3:
        raise ValueError("Training split must contain at least three rows.")

    train_x = [features[index] for index in train_indices]
    train_y = [targets[index] for index in train_indices]
    eval_x = [features[index] for index in eval_indices]
    eval_y = [targets[index] for index in eval_indices]

    model = train_tabular(train_x, train_y, version=version, ridge=ridge)
    eval_mae = tabular_mae(model, eval_x, eval_y)
    dataset_fingerprint = _fingerprint(
        {"features": features, "targets": targets}
    )
    config_fingerprint = _fingerprint(
        {
            "version": version,
            "eval_fraction": eval_fraction,
            "seed": seed,
            "ridge": ridge,
            "max_mae": max_mae,
        }
    )
    artifact_fingerprint = _fingerprint(
        {
            "version": model.version,
            "weights": model.weights,
            "intercept": model.intercept,
            "feature_means": model.feature_means,
            "feature_scales": model.feature_scales,
            "dataset_fingerprint": dataset_fingerprint,
            "config_fingerprint": config_fingerprint,
        }
    )
    return TabularPipelineRun(
        model=model,
        eval_mae=eval_mae,
        allowed=eval_mae <= max_mae,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        train_indices=train_indices,
        eval_indices=eval_indices,
    )
