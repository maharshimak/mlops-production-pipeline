import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from mlops_pipeline.core import LinearModel


@dataclass(frozen=True, slots=True)
class RunManifest:
    model_version: str
    model_fingerprint: str
    train_rows: int
    eval_rows: int
    mae: float
    drift_psi: float


@dataclass(frozen=True, slots=True)
class DeploymentDecision:
    allowed: bool
    reasons: tuple[str, ...]


def model_fingerprint(model: LinearModel) -> str:
    if not all(isfinite(value) for value in (model.slope, model.intercept)):
        raise ValueError("model coefficients must be finite")
    payload = {
        "version": model.version,
        "slope": model.slope,
        "intercept": model.intercept,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_run_manifest(
    model: LinearModel,
    *,
    train_rows: int,
    eval_rows: int,
    mae: float,
    drift_psi: float,
) -> RunManifest:
    for name, value in (("train_rows", train_rows), ("eval_rows", eval_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isfinite(mae) or mae < 0 or not isfinite(drift_psi) or drift_psi < 0:
        raise ValueError("mae and drift_psi must be finite and non-negative")
    return RunManifest(
        model_version=model.version,
        model_fingerprint=model_fingerprint(model),
        train_rows=train_rows,
        eval_rows=eval_rows,
        mae=mae,
        drift_psi=drift_psi,
    )


def deployment_gate(
    manifest: RunManifest,
    *,
    max_mae: float,
    max_psi: float,
    min_eval_rows: int = 20,
) -> DeploymentDecision:
    if not isfinite(max_mae) or max_mae < 0 or not isfinite(max_psi) or max_psi < 0:
        raise ValueError("quality thresholds must be finite and non-negative")
    if isinstance(min_eval_rows, bool) or not isinstance(min_eval_rows, int) or min_eval_rows <= 0:
        raise ValueError("min_eval_rows must be a positive integer")

    reasons: list[str] = []
    if manifest.eval_rows < min_eval_rows:
        reasons.append(f"eval_rows {manifest.eval_rows} < required {min_eval_rows}")
    if manifest.mae > max_mae:
        reasons.append(f"mae {manifest.mae:.6f} > maximum {max_mae:.6f}")
    if manifest.drift_psi > max_psi:
        reasons.append(f"psi {manifest.drift_psi:.6f} > maximum {max_psi:.6f}")
    return DeploymentDecision(allowed=not reasons, reasons=tuple(reasons))
