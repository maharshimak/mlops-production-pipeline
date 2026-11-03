"""MAK'MA ``makma.v1`` execution envelope for this repository's operations.

Pydantic is not a dependency of this package, so the envelope is a frozen
dataclass with explicit validation of the contract invariants. Compatibility is
enforced by ``tests/test_contracts.py``, which validates real
:func:`run_operation` outputs against ``contracts/makma.v1.schema.json``.

All operations are deterministic arithmetic over caller-supplied data, so every
envelope reports ``execution_mode = "deterministic_offline"``.
"""

from __future__ import annotations

import os
import re
import secrets
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from mlops_pipeline.datasets import (
    DatasetSchema,
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)
from mlops_pipeline.drift import feature_drift
from mlops_pipeline.experiment import (
    ExperimentConfig,
    candidate_evaluation_export,
    run_experiment,
    verify_record,
    write_run_outputs,
)
from mlops_pipeline.model_card import model_card_markdown

WORKSPACE_ENV = "MLOPS_PIPELINE_WORKSPACE"
CONTRACT_VERSION = "makma.v1"
SYSTEM = "mlops-production-pipeline"
EXECUTION_MODE = "deterministic_offline"
_STATUSES = {"ok", "error", "blocked", "pending_approval"}
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_OPERATION = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")


def package_version() -> str:
    try:
        return metadata.version("mlops-production-pipeline")
    except metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: str
    message: str
    retryable: bool = False
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ERROR_CODE.match(self.code):
            raise ValueError("error code must be UPPER_SNAKE")
        if len(self.message) > 2000:
            raise ValueError("error message too long")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class Provenance:
    kind: str
    id: str
    sha256: str | None = None
    span: tuple[int, int] | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"document", "chunk", "artifact", "dataset", "model", "query", "record"}:
            raise ValueError("invalid provenance kind")
        if not self.id:
            raise ValueError("provenance id is required")
        if self.sha256 is not None and not _HEX.match(self.sha256):
            raise ValueError("provenance sha256 must be 64 lowercase hex characters")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "id": self.id,
            "sha256": self.sha256,
            "span": list(self.span) if self.span is not None else None,
            "uri": self.uri,
        }


@dataclass(frozen=True, slots=True)
class Envelope:
    operation: str
    run_id: str
    correlation_id: str
    trace_id: str
    span_id: str
    status: str
    data: object
    error: ErrorInfo | None
    provenance: tuple[Provenance, ...]
    policy: dict[str, object] | None
    started_at: str
    duration_ms: float
    execution_mode: str = EXECUTION_MODE
    system: str = SYSTEM
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError("invalid status")
        if (self.status == "ok") != (self.error is None):
            raise ValueError("status 'ok' if and only if error is null")
        if self.status in {"error", "blocked"} and self.data is not None:
            raise ValueError("error/blocked envelopes must have data null")
        if self.status == "pending_approval" and (
            self.policy is None or self.policy.get("decision") != "requires_approval"
        ):
            raise ValueError("pending_approval requires policy.decision requires_approval")
        if not re.fullmatch(r"[0-9a-f]{32}", self.trace_id) or not re.fullmatch(
            r"[0-9a-f]{16}", self.span_id
        ):
            raise ValueError("invalid trace/span id")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "system": self.system,
            "operation": self.operation,
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "data": self.data,
            "error": self.error.to_dict() if self.error is not None else None,
            "provenance": [item.to_dict() for item in self.provenance],
            "policy": self.policy,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
        }


class OperationError(Exception):
    def __init__(self, code: str, message: str, *, details: dict[str, object] | None = None,
                 retryable: bool = False, blocked: bool = False) -> None:
        super().__init__(message)
        self.info = ErrorInfo(code, message, retryable, details or {})
        self.blocked = blocked


def workspace_root() -> Path:
    """Directory that ``train`` may write under: ``$MLOPS_PIPELINE_WORKSPACE`` or the CWD.

    The root itself is resolved (symlinks and ``..`` removed) and must exist.
    """
    configured = os.environ.get(WORKSPACE_ENV)
    root = Path(configured) if configured else Path.cwd()
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise OperationError(
            "WORKSPACE_UNAVAILABLE", f"{WORKSPACE_ENV} does not name an existing directory"
        ) from error
    if not resolved.is_dir():
        raise OperationError(
            "WORKSPACE_UNAVAILABLE", f"{WORKSPACE_ENV} does not name an existing directory"
        )
    return resolved


def _blocked_path() -> OperationError:
    # The resolved path is deliberately not echoed back to the caller.
    return OperationError(
        "PATH_OUTSIDE_WORKSPACE",
        "output_dir resolves outside the configured workspace root",
        blocked=True,
    )


def confine_to_workspace(output_dir: str, root: Path) -> Path:
    """Resolve ``output_dir`` (relative paths are relative to ``root``) and require it
    to be ``root`` or a descendant after symlink and ``..`` resolution."""
    candidate = Path(output_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as error:  # e.g. symlink loops
        raise _blocked_path() from error
    if resolved != root and root not in resolved.parents:
        raise _blocked_path()
    return resolved


def _invalid(message: str, **details: object) -> OperationError:
    return OperationError("INVALID_INPUT", message, details=dict(details))


# --------------------------------------------------------------------------- inputs

_SCHEMA_JSON: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["features", "label"],
    "properties": {
        "features": {"type": "array", "items": {"type": "string", "minLength": 1},
                     "minItems": 1, "maxItems": 256},
        "label": {"type": "string", "minLength": 1},
        "group": {"type": ["string", "null"]},
        "allow_extra_columns": {"type": "boolean"},
    },
}
_RECORDS_JSON: dict[str, object] = {
    "type": "array", "items": {"type": "object"}, "maxItems": 100_000,
}
_CONFIG_JSON: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "model_name": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,63}$"},
        "test_fraction": {"type": "number", "minimum": 0.1, "maximum": 0.5},
        "seed": {"type": "integer", "minimum": 0},
        "ridge": {"type": "number", "minimum": 0},
        "max_test_mae": {"type": "number", "minimum": 0},
        "max_feature_psi": {"type": "number", "minimum": 0},
        "min_test_rows": {"type": "integer", "minimum": 1},
        "psi_bins": {"type": "integer", "minimum": 2, "maximum": 100},
    },
}
_CODE_VERSION_JSON: dict[str, object] = {
    "type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$",
}


def _object(required: list[str], properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _check_keys(payload: Mapping[str, object], required: set[str], optional: set[str]) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise _invalid(f"missing required field(s): {', '.join(missing)}", missing=missing)
    unknown = sorted(set(payload) - required - optional, key=str)
    if unknown:
        raise _invalid("unknown field(s) in payload", unknown=[str(k) for k in unknown])


def _parse_schema(value: object) -> DatasetSchema:
    try:
        return DatasetSchema.from_dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise _invalid(f"invalid schema: {error}") from error


def _parse_config(value: object) -> ExperimentConfig:
    try:
        return ExperimentConfig.from_dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise _invalid(f"invalid config: {error}") from error


def _parse_records(value: object, name: str = "records") -> list[object]:
    if not isinstance(value, list):
        raise _invalid(f"{name} must be an array of objects")
    return value


def _parse_code_version(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", value
    ):
        raise _invalid("code_version must match ^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
    return value


# --------------------------------------------------------------------------- operations

OpResult = tuple[object, list[Provenance]]


def _run_experiment(payload: Mapping[str, object]):
    schema = _parse_schema(payload["schema"])
    records = _parse_records(payload["records"])
    config = _parse_config(payload.get("config"))
    code_version = _parse_code_version(payload.get("code_version"))
    try:
        return run_experiment(records, schema, config, code_version=code_version)
    except DatasetValidationError as error:
        raise OperationError(
            "DATASET_INVALID", str(error), details={"report": error.report.to_dict()}
        ) from error
    except ValueError as error:
        raise OperationError("TRAINING_FAILED", str(error)) from error


def _experiment_provenance(record: Mapping[str, object]) -> list[Provenance]:
    dataset = record["dataset"]
    artifact = record["artifact"]
    return [
        Provenance("dataset", "dataset", dataset["sha256"]),  # type: ignore[index]
        Provenance("artifact", str(record["model_id"]), artifact["sha256"]),  # type: ignore[index]
        Provenance("record", "run_record", str(record["record_sha256"])),
    ]


def _op_validate_dataset(payload: Mapping[str, object]) -> OpResult:
    _check_keys(payload, {"records", "schema"}, {"require_label"})
    schema = _parse_schema(payload["schema"])
    records = _parse_records(payload["records"])
    require_label = payload.get("require_label", True)
    if type(require_label) is not bool:
        raise _invalid("require_label must be boolean")
    report = validate_dataset(records, schema, require_label=require_label)
    provenance = (
        [Provenance("dataset", "dataset", report.dataset_sha256)]
        if report.dataset_sha256
        else []
    )
    return report.to_dict(), provenance


def _op_train(payload: Mapping[str, object]) -> OpResult:
    _check_keys(payload, {"records", "schema", "output_dir"}, {"config", "code_version"})
    output_dir = payload["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip() or "\x00" in output_dir:
        raise _invalid("output_dir must be a non-empty path string")
    root = workspace_root()
    target = confine_to_workspace(output_dir, root)
    result = _run_experiment(payload)
    try:
        target.mkdir(parents=True, exist_ok=True)
        # Re-check after creating directories: a path component may have been
        # swapped for a symlink between the first check and mkdir.
        target = confine_to_workspace(str(target), root)
        written = write_run_outputs(result, target)
    except FileExistsError as error:
        raise OperationError("ARTIFACT_CONFLICT", str(error)) from error
    except OSError as error:
        raise OperationError("WRITE_FAILED", f"could not write run outputs: {error.strerror}",
                             retryable=True) from error
    if not written.verified:
        raise OperationError("INTEGRITY_CHECK_FAILED", "written artifacts failed verification")
    record = result.record
    provenance = _experiment_provenance(record)
    provenance[1] = Provenance(
        "artifact", str(record["model_id"]), result.artifact_sha256,
        uri=written.artifact_path.resolve().as_uri(),
    )
    return {
        "model_id": record["model_id"],
        "artifact_path": str(written.artifact_path),
        "record_path": str(written.record_path),
        "created": written.created,
        "verified": written.verified,
        "run_record": record,
    }, provenance


def _op_evaluate_candidate(payload: Mapping[str, object]) -> OpResult:
    _check_keys(payload, {"records", "schema"}, {"config", "code_version"})
    result = _run_experiment(payload)
    return {
        "candidate": candidate_evaluation_export(result.record),
        "run_record": result.record,
    }, _experiment_provenance(result.record)


def _op_model_card(payload: Mapping[str, object]) -> OpResult:
    _check_keys(payload, {"run_record"}, set())
    try:
        record = verify_record(payload["run_record"])
        markdown = model_card_markdown(record)
    except (KeyError, TypeError, ValueError) as error:
        raise _invalid(f"invalid run_record: {error}") from error
    return {
        "model_id": record.get("model_id"),
        "record_sha256": record["record_sha256"],
        "markdown": markdown,
    }, [Provenance("record", "run_record", str(record["record_sha256"]))]


def _op_evaluate_drift(payload: Mapping[str, object]) -> OpResult:
    _check_keys(
        payload, {"schema", "reference_records", "current_records"}, {"bins", "max_psi"}
    )
    schema = _parse_schema(payload["schema"])
    bins = payload.get("bins", 5)
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
        raise _invalid("bins must be an integer between 2 and 100")
    max_psi = payload.get("max_psi", 0.25)
    if isinstance(max_psi, bool) or not isinstance(max_psi, (int, float)) or not (
        0 <= max_psi < float("inf")
    ):
        raise _invalid("max_psi must be a finite non-negative number")
    samples = {}
    for name in ("reference_records", "current_records"):
        records = _parse_records(payload[name], name)
        try:
            samples[name] = load_dataset(records, schema, min_rows=1, require_label=False)
        except DatasetValidationError as error:
            raise OperationError(
                "DATASET_INVALID", f"{name}: {error}",
                details={"sample": name, "report": error.report.to_dict()},
            ) from error
    reference, current = samples["reference_records"], samples["current_records"]
    drift = feature_drift(schema.features, reference.features, current.features, bins=bins)
    return {
        "reference_sha256": reference.sha256,
        "current_sha256": current.sha256,
        "bins": bins,
        "max_psi_threshold": float(max_psi),
        "features": [
            {
                "feature": item.feature,
                "psi": item.psi,
                "exceeds_threshold": item.psi > max_psi,
                "reference_counts": list(item.reference_counts),
                "current_counts": list(item.current_counts),
            }
            for item in drift
        ],
        "drift_detected": any(item.psi > max_psi for item in drift),
    }, [
        Provenance("dataset", "reference", reference.sha256),
        Provenance("dataset", "current", current.sha256),
    ]


@dataclass(frozen=True, slots=True)
class _Operation:
    handler: Callable[[Mapping[str, object]], OpResult]
    side_effects: bool
    input_schema: dict[str, object]
    timeout_s: int
    description: str


_OPERATIONS: dict[str, _Operation] = {
    "validate_dataset": _Operation(
        _op_validate_dataset, False,
        _object(["records", "schema"], {
            "records": _RECORDS_JSON, "schema": _SCHEMA_JSON,
            "require_label": {"type": "boolean"},
        }),
        30,
        "Schema/type/missing-value/label checks and canonical dataset SHA-256.",
    ),
    "train": _Operation(
        _op_train, True,
        _object(["records", "schema", "output_dir"], {
            "records": _RECORDS_JSON, "schema": _SCHEMA_JSON, "config": _CONFIG_JSON,
            "code_version": _CODE_VERSION_JSON,
            "output_dir": {"type": "string", "minLength": 1},
        }),
        60,
        "Fit the ridge baseline; writes model.json and run_record.json (never overwrites).",
    ),
    "evaluate_candidate": _Operation(
        _op_evaluate_candidate, False,
        _object(["records", "schema"], {
            "records": _RECORDS_JSON, "schema": _SCHEMA_JSON, "config": _CONFIG_JSON,
            "code_version": _CODE_VERSION_JSON,
        }),
        60,
        "Run the experiment in memory and return the control-plane candidate export.",
    ),
    "model_card": _Operation(
        _op_model_card, False,
        _object(["run_record"], {"run_record": {"type": "object"}}),
        30,
        "Render a Markdown model card from a hash-verified run record.",
    ),
    "evaluate_drift": _Operation(
        _op_evaluate_drift, False,
        _object(["schema", "reference_records", "current_records"], {
            "schema": _SCHEMA_JSON, "reference_records": _RECORDS_JSON,
            "current_records": _RECORDS_JSON,
            "bins": {"type": "integer", "minimum": 2, "maximum": 100},
            "max_psi": {"type": "number", "minimum": 0},
        }),
        30,
        "Per-feature PSI with bins fitted on the reference sample (labels optional).",
    ),
}


def capabilities() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "system": SYSTEM,
        "version": package_version(),
        "operations": [
            {
                "name": name,
                "description": op.description,
                "side_effects": op.side_effects,
                "requires_approval": False,
                "input_schema": op.input_schema,
                "timeout_s": op.timeout_s,
            }
            for name, op in _OPERATIONS.items()
        ],
    }


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128


def run_operation(
    operation: str,
    payload: dict,
    *,
    run_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, object]:
    """Execute ``operation`` and return a ``makma.v1`` envelope as a dict."""
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    clock = time.perf_counter()
    trace_id, span_id = secrets.token_hex(16), secrets.token_hex(8)
    error: ErrorInfo | None = None
    blocked = False
    data: object = None
    provenance: list[Provenance] = []

    ids_valid = (run_id is None or _valid_id(run_id)) and (
        correlation_id is None or _valid_id(correlation_id)
    )
    resolved_run_id = run_id if run_id is not None and ids_valid else uuid.uuid4().hex
    resolved_correlation = (
        correlation_id if correlation_id is not None and ids_valid else resolved_run_id
    )
    op_name = operation if isinstance(operation, str) and _OPERATION.match(operation) else "unknown"

    spec = _OPERATIONS.get(operation) if isinstance(operation, str) else None
    if not ids_valid:
        error = ErrorInfo("INVALID_INPUT", "run_id and correlation_id must be 1-128 characters")
    elif spec is None:
        error = ErrorInfo(
            "UNKNOWN_OPERATION",
            "unknown operation",
            details={"available": sorted(_OPERATIONS)},
        )
    elif not isinstance(payload, Mapping):
        error = ErrorInfo("INVALID_INPUT", "payload must be an object")
    else:
        try:
            data, provenance = spec.handler(payload)
        except OperationError as failure:
            error = failure.info
            blocked = failure.blocked
        except Exception as failure:  # noqa: BLE001 - surfaced as a contract error
            error = ErrorInfo(
                "INTERNAL_ERROR",
                "operation failed unexpectedly",
                details={"exception": type(failure).__name__},
            )
    if error is not None:
        data, provenance = None, []

    envelope = Envelope(
        operation=op_name,
        run_id=resolved_run_id,
        correlation_id=resolved_correlation,
        trace_id=trace_id,
        span_id=span_id,
        status="ok" if error is None else ("blocked" if blocked else "error"),
        data=data,
        error=error,
        provenance=tuple(provenance),
        policy={"decision": "deny", "reasons": [error.message]} if blocked and error else None,
        started_at=started_at,
        duration_ms=round((time.perf_counter() - clock) * 1000.0, 3),
    )
    return envelope.to_dict()
