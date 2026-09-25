from __future__ import annotations

import json
import random
from dataclasses import dataclass
from hashlib import sha256
from math import ceil

from mlops_pipeline.artifacts import (
    DeploymentDecision,
    RunManifest,
    build_run_manifest,
    deployment_gate,
)
from mlops_pipeline.core import LinearModel, mae, psi, train
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
    tracking_run_id: str | None = None


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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
    ledger: SQLiteRunLedger | None = None,
    run_name: str = "regression-pipeline",
) -> PipelineRun:
    if len(xs) != len(ys) or len(xs) < 6:
        raise ValueError("Pipeline requires at least six paired observations.")
    if not 0.1 <= eval_fraction <= 0.5:
        raise ValueError("eval_fraction must be between 0.1 and 0.5.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

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

    minimum = min(xs)
    maximum = max(xs)
    train_hist = _histogram(train_x, minimum, maximum)
    eval_hist = _histogram(eval_x, minimum, maximum)
    drift_psi = psi(train_hist, eval_hist)

    manifest = build_run_manifest(
        model,
        train_rows=len(train_indices),
        eval_rows=len(eval_indices),
        mae=eval_mae,
        drift_psi=drift_psi,
    )
    decision = deployment_gate(
        manifest,
        max_mae=max_mae,
        max_psi=max_psi,
        min_eval_rows=2,
        integrity_valid=True,
    )

    dataset_fingerprint = _fingerprint(list(zip(xs, ys, strict=True)))
    config_fingerprint = _fingerprint(
        {
            "version": version,
            "eval_fraction": eval_fraction,
            "seed": seed,
            "max_mae": max_mae,
            "max_psi": max_psi,
        }
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
    )

    tracking_run_id: str | None = None
    if ledger is not None:
        tracking_run_id = ledger.start_run(
            run_name,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            params={
                "version": version,
                "eval_fraction": eval_fraction,
                "seed": seed,
                "max_mae": max_mae,
                "max_psi": max_psi,
            },
        )
        try:
            ledger.log_metric(tracking_run_id, "mae", eval_mae)
            ledger.log_metric(tracking_run_id, "psi", drift_psi)
            artifact_digest = sha256(artifact_payload.encode()).hexdigest()
            ledger.log_artifact(
                tracking_run_id,
                "model",
                f"sha256:{artifact_digest}",
            )
            ledger.finish_run(
                tracking_run_id,
                status="succeeded" if decision.allowed else "failed",
            )
        except Exception:
            ledger.finish_run(tracking_run_id, status="failed")
            raise

    return PipelineRun(
        model=model,
        manifest=manifest,
        decision=decision,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
        artifact_payload=artifact_payload,
        train_indices=train_indices,
        eval_indices=eval_indices,
        tracking_run_id=tracking_run_id,
    )
