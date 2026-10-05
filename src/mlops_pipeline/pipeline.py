from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path

from mlops_pipeline.artifacts import (
    DeploymentDecision,
    RunManifest,
    build_run_manifest,
    deployment_gate,
)
from mlops_pipeline.core import LinearModel, mae, psi, train
from mlops_pipeline.cross_validation import CrossValidationResult, kfold_regression_cv
from mlops_pipeline.integrity import build_integrity_manifest, verify_integrity_manifest
from mlops_pipeline.tracking import SQLiteRunLedger


@dataclass(frozen=True, slots=True)
class PipelineRun:
    model: LinearModel
    manifest: RunManifest
    decision: DeploymentDecision
    dataset_fingerprint: str
    config_fingerprint: str
    artifact_payload: str
    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    cross_validation: CrossValidationResult
    tracking_run_id: str | None = None


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return sha256(payload).hexdigest()


def _histogram(values: list[float], minimum: float, maximum: float, bins: int = 5) -> list[float]:
    if minimum == maximum:
        return [float(len(values)), *([0.0] * (bins - 1))]
    width = (maximum - minimum) / bins
    counts = [0.0] * bins
    for value in values:
        index = int((value - minimum) / width)
        index = max(0, min(bins - 1, index))
        counts[index] += 1.0
    return counts


def run_regression_pipeline(
    xs: list[float],
    ys: list[float],
    *,
    version: str = "1.0.0",
    eval_fraction: float = 0.25,
    seed: int = 42,
    max_mae: float = 1.0,
    max_psi: float = 0.25,
    min_eval_rows: int = 5,
    cv_folds: int = 5,
    max_cv_worst_mae: float | None = None,
    ledger: SQLiteRunLedger | None = None,
    run_name: str = "regression-pipeline",
) -> PipelineRun:
    if len(xs) != len(ys) or len(xs) < 6:
        raise ValueError("Pipeline requires at least six paired observations.")
    if not all(type(value) in (int, float) and isfinite(value) for value in [*xs, *ys]):
        raise ValueError("Pipeline observations must be finite numeric values.")
    if (
        isinstance(eval_fraction, bool)
        or not isinstance(eval_fraction, (int, float))
        or not isfinite(eval_fraction)
        or not 0.1 <= eval_fraction <= 0.5
    ):
        raise ValueError("eval_fraction must be finite and between 0.1 and 0.5.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if isinstance(min_eval_rows, bool) or not isinstance(min_eval_rows, int):
        raise TypeError("min_eval_rows must be an integer.")
    if min_eval_rows <= 0:
        raise ValueError("min_eval_rows must be positive.")
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError("cv_folds must be an integer >= 2.")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version is required.")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name is required.")

    for name, value in (("max_mae", max_mae), ("max_psi", max_psi)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative.")

    effective_cv_threshold = max_mae if max_cv_worst_mae is None else max_cv_worst_mae
    if (
        isinstance(effective_cv_threshold, bool)
        or not isinstance(effective_cv_threshold, (int, float))
        or not isfinite(effective_cv_threshold)
        or effective_cv_threshold < 0
    ):
        raise ValueError("max_cv_worst_mae must be finite and non-negative.")

    dataset_fingerprint = _fingerprint(list(zip(xs, ys, strict=True)))
    config = {
        "version": version,
        "eval_fraction": eval_fraction,
        "seed": seed,
        "max_mae": max_mae,
        "max_psi": max_psi,
        "min_eval_rows": min_eval_rows,
        "cv_folds": cv_folds,
        "max_cv_worst_mae": effective_cv_threshold,
    }
    config_fingerprint = _fingerprint(config)

    tracking_run_id: str | None = None
    if ledger is not None:
        tracking_run_id = ledger.start_run(
            run_name,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            params=config,
        )

    try:
        indices = list(range(len(xs)))
        random.Random(seed).shuffle(indices)
        eval_count = max(2, ceil(len(indices) * eval_fraction))
        eval_indices = tuple(sorted(indices[:eval_count]))
        train_indices = tuple(sorted(indices[eval_count:]))
        if len(train_indices) < 2:
            raise ValueError("Training split must contain at least two observations.")

        train_x = [xs[index] for index in train_indices]
        train_y = [ys[index] for index in train_indices]
        eval_x = [xs[index] for index in eval_indices]
        eval_y = [ys[index] for index in eval_indices]

        model = train(train_x, train_y, version=version)
        eval_mae = mae(model, eval_x, eval_y)
        cross_validation = kfold_regression_cv(
            [[value] for value in xs],
            ys,
            folds=cv_folds,
            seed=seed,
        )

        # Histogram boundaries are fitted only on the training/reference split.
        # Evaluation observations are projected into those frozen bins.
        minimum = min(train_x)
        maximum = max(train_x)
        train_hist = _histogram(train_x, minimum, maximum)
        eval_hist = _histogram(eval_x, minimum, maximum)
        drift_psi = psi(train_hist, eval_hist)

        manifest = build_run_manifest(
            model,
            train_rows=len(train_indices),
            eval_rows=len(eval_indices),
            mae=eval_mae,
            drift_psi=drift_psi,
            cv_mean_mae=cross_validation.mean_mae,
            cv_worst_mae=cross_validation.worst_mae,
        )
        artifact_payload = json.dumps(
            {
                "version": model.version,
                "slope": model.slope,
                "intercept": model.intercept,
                "dataset_fingerprint": dataset_fingerprint,
                "config_fingerprint": config_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        # The deployment decision is based on byte-level verification of the
        # serialized artifact rather than a hard-coded integrity flag.
        with tempfile.TemporaryDirectory(prefix="mlops-pipeline-integrity-") as temp_dir:
            artifact_path = Path(temp_dir) / "model.json"
            artifact_path.write_text(artifact_payload, encoding="utf-8")
            integrity_manifest = build_integrity_manifest(temp_dir, ["model.json"])
            integrity_valid = verify_integrity_manifest(temp_dir, integrity_manifest).valid

        decision = deployment_gate(
            manifest,
            max_mae=max_mae,
            max_psi=max_psi,
            min_eval_rows=min_eval_rows,
            integrity_valid=integrity_valid,
            max_cv_worst_mae=effective_cv_threshold,
        )

        if ledger is not None and tracking_run_id is not None:
            ledger.log_metric(tracking_run_id, "mae", eval_mae)
            ledger.log_metric(tracking_run_id, "psi", drift_psi)
            ledger.log_metric(
                tracking_run_id,
                "cv_mean_mae",
                cross_validation.mean_mae,
            )
            ledger.log_metric(
                tracking_run_id,
                "cv_worst_mae",
                cross_validation.worst_mae,
            )
            artifact_digest = sha256(artifact_payload.encode()).hexdigest()
            ledger.log_artifact(
                tracking_run_id,
                "model",
                f"sha256:{artifact_digest}",
            )
            ledger.finish_run(
                tracking_run_id,
                status="succeeded" if decision.allowed else "rejected",
            )

        return PipelineRun(
            model=model,
            manifest=manifest,
            decision=decision,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            artifact_payload=artifact_payload,
            train_indices=train_indices,
            eval_indices=eval_indices,
            cross_validation=cross_validation,
            tracking_run_id=tracking_run_id,
        )
    except Exception:
        if ledger is not None and tracking_run_id is not None:
            try:
                ledger.finish_run(tracking_run_id, status="failed")
            except Exception:
                # Preserve the original pipeline failure if tracking also fails.
                pass
        raise
