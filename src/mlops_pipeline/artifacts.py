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


def _require_non_negative_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _validate_manifest(manifest: RunManifest) -> None:
    if not isinstance(manifest.model_version, str) or not manifest.model_version.strip():
        raise ValueError("model_version is required")
    if not isinstance(manifest.model_fingerprint, str) or not manifest.model_fingerprint.strip():
        raise ValueError("model_fingerprint is required")
    for name, value in (("train_rows", manifest.train_rows), ("eval_rows", manifest.eval_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _require_non_negative_number("mae", manifest.mae)
    _require_non_negative_number("drift_psi", manifest.drift_psi)
    for name, value in (
        ("cv_mean_mae", manifest.cv_mean_mae),
        ("cv_worst_mae", manifest.cv_worst_mae),
    ):
        if value is not None:
            _require_non_negative_number(name, value)


def model_fingerprint(model: LinearModel) -> str:
    if not isinstance(model.version, str) or not model.version.strip():
        raise ValueError("model version is required")
    for value in (model.slope, model.intercept):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError("model coefficients must be finite numeric values")
    payload = {
        "version": model.version,
        "slope": model.slope,
        "intercept": model.intercept,
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


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
    manifest = RunManifest(
        model_version=model.version,
        model_fingerprint=model_fingerprint(model),
        train_rows=train_rows,
        eval_rows=eval_rows,
        mae=mae,
        drift_psi=drift_psi,
        cv_mean_mae=cv_mean_mae,
        cv_worst_mae=cv_worst_mae,
    )
    _validate_manifest(manifest)
    return manifest


def deployment_gate(
    manifest: RunManifest,
    *,
    max_mae: float,
    max_psi: float,
    min_eval_rows: int = 20,
    integrity_valid: bool = True,
    max_cv_worst_mae: float | None = None,
) -> DeploymentDecision:
    _validate_manifest(manifest)
    _require_non_negative_number("max_mae", max_mae)
    _require_non_negative_number("max_psi", max_psi)
    if isinstance(min_eval_rows, bool) or not isinstance(min_eval_rows, int) or min_eval_rows <= 0:
        raise ValueError("min_eval_rows must be a positive integer")
    if type(integrity_valid) is not bool:
        raise ValueError("integrity_valid must be boolean")
    if max_cv_worst_mae is not None:
        _require_non_negative_number("max_cv_worst_mae", max_cv_worst_mae)

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
