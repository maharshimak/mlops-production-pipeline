from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from statistics import mean

from mlops_pipeline.tabular import _validate_matrix, tabular_mae, train_tabular


@dataclass(frozen=True, slots=True)
class FoldMetric:
    fold: int
    train_rows: int
    eval_rows: int
    mae: float


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    folds: tuple[FoldMetric, ...]
    mean_mae: float
    std_mae: float
    worst_mae: float


def kfold_regression_cv(
    features: list[list[float]],
    targets: list[float],
    *,
    folds: int = 5,
    seed: int = 42,
    ridge: float = 1e-8,
) -> CrossValidationResult:
    """Deterministic shuffled k-fold evaluation for tabular regression."""
    _validate_matrix(features, targets, min_rows=4)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if folds > len(features):
        raise ValueError("folds cannot exceed the number of rows")
    if len(features) - ((len(features) + folds - 1) // folds) < 3:
        raise ValueError("each training split must contain at least three rows")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    buckets = [indices[offset::folds] for offset in range(folds)]

    metrics: list[FoldMetric] = []
    for fold_index, eval_indices in enumerate(buckets, start=1):
        eval_set = set(eval_indices)
        train_indices = [index for index in indices if index not in eval_set]
        train_x = [features[index] for index in train_indices]
        train_y = [targets[index] for index in train_indices]
        eval_x = [features[index] for index in eval_indices]
        eval_y = [targets[index] for index in eval_indices]

        model = train_tabular(
            train_x,
            train_y,
            version=f"cv-fold-{fold_index}",
            ridge=ridge,
        )
        metrics.append(
            FoldMetric(
                fold=fold_index,
                train_rows=len(train_indices),
                eval_rows=len(eval_indices),
                mae=tabular_mae(model, eval_x, eval_y),
            )
        )

    maes = [metric.mae for metric in metrics]
    average = mean(maes)
    variance = sum((value - average) ** 2 for value in maes) / len(maes)
    return CrossValidationResult(
        folds=tuple(metrics),
        mean_mae=average,
        std_mae=sqrt(variance),
        worst_mae=max(maes),
    )
