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
    cv_mean_mae: float | None = None
    cv_worst_mae: float | None = None


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
    cv_mean_mae: float | None = None,
    cv_worst_mae: float | None = None,
) -> RunManifest:
    for name, value in (("train_rows", train_rows), ("eval_rows", eval_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isfinite(mae) or mae < 0 or not isfinite(drift_psi) or drift_psi < 0:
        raise ValueError("mae and drift_psi must be finite and non-negative")
    for name, value in (("cv_mean_mae", cv_mean_mae), ("cv_worst_mae", cv_worst_mae)):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative when provided")
    return RunManifest(
        model_version=model.version,
        model_fingerprint=model_fingerprint(model),
        train_rows=train_rows,
        eval_rows=eval_rows,
        mae=mae,
        drift_psi=drift_psi,
        cv_mean_mae=cv_mean_mae,
        cv_worst_mae=cv_worst_mae,
    )


def deployment_gate(
    manifest: RunManifest,
    *,
    max_mae: float,
    max_psi: float,
    min_eval_rows: int = 20,
    integrity_valid: bool = True,
    max_cv_worst_mae: float | None = None,
) -> DeploymentDecision:
    if not isfinite(max_mae) or max_mae < 0 or not isfinite(max_psi) or max_psi < 0:
        raise ValueError("quality thresholds must be finite and non-negative")
    if isinstance(min_eval_rows, bool) or not isinstance(min_eval_rows, int) or min_eval_rows <= 0:
        raise ValueError("min_eval_rows must be a positive integer")

    if type(integrity_valid) is not bool:
        raise ValueError("integrity_valid must be boolean")
    if max_cv_worst_mae is not None and (
        not isfinite(max_cv_worst_mae) or max_cv_worst_mae < 0
    ):
        raise ValueError("max_cv_worst_mae must be finite and non-negative")
    reasons: list[str] = []
    if not integrity_valid:
        reasons.append("artifact integrity verification failed")
    if manifest.eval_rows < min_eval_rows:
        reasons.append(f"eval_rows {manifest.eval_rows} < required {min_eval_rows}")
    if manifest.mae > max_mae:
        reasons.append(f"mae {manifest.mae:.6f} > maximum {max_mae:.6f}")
    if manifest.drift_psi > max_psi:
        reasons.append(f"psi {manifest.drift_psi:.6f} > maximum {max_psi:.6f}")
    if max_cv_worst_mae is not None:
        if manifest.cv_worst_mae is None:
            reasons.append("cross-validation result missing")
        elif manifest.cv_worst_mae > max_cv_worst_mae:
            reasons.append(
                f"cv_worst_mae {manifest.cv_worst_mae:.6f} > maximum "
                f"{max_cv_worst_mae:.6f}"
            )
    return DeploymentDecision(allowed=not reasons, reasons=tuple(reasons))
