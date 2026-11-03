"""Reproducible experiment runs over validated tabular datasets.

``run_experiment`` validates a dataset, performs a leakage-safe deterministic
split, fits the existing standardized ridge linear regression baseline
(:func:`mlops_pipeline.tabular.train_tabular`), evaluates it on the held-out
rows, measures per-feature PSI drift between train and test, applies a quality
gate and emits:

* the serialized model artifact (canonical JSON bytes) and its SHA-256;
* a run record that binds dataset SHA-256, config SHA-256, code version,
  split, metrics, drift, gate decision and artifact SHA-256. The record carries
  no timestamps, so identical inputs produce byte-identical records.

Everything here is deterministic arithmetic; nothing is model inference by an
external or learned service beyond the closed-form linear fit itself.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, sqrt
from pathlib import Path

from mlops_pipeline.canonical import canonical_json_bytes, canonical_sha256, sha256_hex
from mlops_pipeline.datasets import DatasetSchema, ValidatedDataset, load_dataset
from mlops_pipeline.drift import feature_drift
from mlops_pipeline.integrity import build_integrity_manifest, verify_integrity_manifest
from mlops_pipeline.splits import GroupSplit, group_train_test_split
from mlops_pipeline.tabular import TabularLinearModel, predict_tabular, train_tabular

RUN_RECORD_FORMAT = "mlops-run-record.v1"
MODEL_ARTIFACT_FORMAT = "mlops-model.v1"
CANDIDATE_EXPORT_FORMAT = "mlops-candidate-evaluation.v1"
MODEL_TYPE = "standardized_ridge_linear_regression"
UNKNOWN_CODE_VERSION = "unknown"
ARTIFACT_FILENAME = "model.json"
RECORD_FILENAME = "run_record.json"

_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CODE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_PACKAGE_DIR = Path(__file__).resolve().parent


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model_name: str = "baseline-ridge"
    test_fraction: float = 0.25
    seed: int = 42
    ridge: float = 1e-8
    max_test_mae: float = 1.0
    max_feature_psi: float = 0.25
    min_test_rows: int = 2
    psi_bins: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not _MODEL_NAME.match(self.model_name):
            raise ValueError("model_name must match ^[a-z0-9][a-z0-9._-]{0,63}$")
        fraction = _number(self.test_fraction, "test_fraction")
        if not 0.1 <= fraction <= 0.5:
            raise ValueError("test_fraction must be between 0.1 and 0.5")
        _integer(self.seed, "seed", minimum=0)
        _number(self.ridge, "ridge")
        _number(self.max_test_mae, "max_test_mae")
        _number(self.max_feature_psi, "max_feature_psi")
        _integer(self.min_test_rows, "min_test_rows", minimum=1)
        bins = _integer(self.psi_bins, "psi_bins", minimum=2)
        if bins > 100:
            raise ValueError("psi_bins must be <= 100")

    def to_dict(self) -> dict[str, object]:
        """Normalized config (numbers as floats where fractional) used for hashing."""
        return {
            "model_name": self.model_name,
            "model_type": MODEL_TYPE,
            "test_fraction": float(self.test_fraction),
            "seed": self.seed,
            "ridge": float(self.ridge),
            "max_test_mae": float(self.max_test_mae),
            "max_feature_psi": float(self.max_feature_psi),
            "min_test_rows": self.min_test_rows,
            "psi_bins": self.psi_bins,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object] | None) -> ExperimentConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("config must be an object")
        allowed = {
            "model_name", "test_fraction", "seed", "ridge", "max_test_mae",
            "max_feature_psi", "min_test_rows", "psi_bins",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**value)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def _git_code_version() -> tuple[str, bool | None]:
    """Return (HEAD sha, dirty) when this package is imported from its own checkout.

    The check requires the git top-level directory to contain this exact source
    package (``<toplevel>/src/mlops_pipeline``); an install inside some unrelated
    repository therefore reports ``"unknown"`` instead of a misleading SHA.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.split()
        if len(top) != 2:
            return UNKNOWN_CODE_VERSION, None
        toplevel, sha = Path(top[0]).resolve(), top[1].strip()
        if toplevel / "src" / "mlops_pipeline" != _PACKAGE_DIR or not _GIT_SHA.match(sha):
            return UNKNOWN_CODE_VERSION, None
        status = subprocess.run(
            ["git", "-C", str(toplevel), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_CODE_VERSION, None
    return sha, bool(status.strip())


def resolve_code_version(explicit: str | None = None) -> tuple[str, str, bool | None]:
    """Return ``(code_version, source, tree_dirty)``.

    ``source`` is ``"caller"`` for an explicit value, ``"git"`` when detected from
    this package's own checkout and ``"unknown"`` otherwise.
    """
    if explicit is not None:
        if not isinstance(explicit, str) or not _CODE_VERSION.match(explicit):
            raise ValueError("code_version must match ^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
        return explicit, "caller", None
    sha, dirty = _git_code_version()
    if sha == UNKNOWN_CODE_VERSION:
        return UNKNOWN_CODE_VERSION, "unknown", None
    return sha, "git", dirty


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    model: TabularLinearModel
    dataset: ValidatedDataset
    split: GroupSplit
    artifact_bytes: bytes
    record: dict[str, object]

    @property
    def artifact_sha256(self) -> str:
        return sha256_hex(self.artifact_bytes)

    @property
    def record_bytes(self) -> bytes:
        return canonical_json_bytes(self.record)


def _errors(model: TabularLinearModel, features, labels) -> list[float]:
    return [
        predict_tabular(model, list(row)) - label
        for row, label in zip(features, labels, strict=True)
    ]


def record_sha256(record: Mapping[str, object]) -> str:
    """SHA-256 of the canonical record with the ``record_sha256`` field excluded."""
    return canonical_sha256({k: v for k, v in record.items() if k != "record_sha256"})


def verify_record(record: object) -> dict[str, object]:
    """Check format and self-hash of a run record; return it as a dict.

    This detects accidental or casual modification. It is not an authenticity
    proof: anyone can recompute the hash of an edited record.
    """
    if not isinstance(record, Mapping):
        raise TypeError("run record must be an object")
    if record.get("format") != RUN_RECORD_FORMAT:
        raise ValueError(f"run record format must be {RUN_RECORD_FORMAT}")
    claimed = record.get("record_sha256")
    if not isinstance(claimed, str):
        raise TypeError("run record is missing record_sha256")
    try:
        actual = record_sha256(record)
    except (TypeError, ValueError) as error:
        raise ValueError("run record is not canonical JSON") from error
    if actual != claimed:
        raise ValueError("run record_sha256 does not match its content")
    return dict(record)


def run_experiment(
    records: object,
    schema: DatasetSchema,
    config: ExperimentConfig | None = None,
    *,
    code_version: str | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    dataset = load_dataset(records, schema, min_rows=6)
    if dataset.labels is None:  # pragma: no cover - load_dataset requires labels here
        raise RuntimeError("labelled dataset expected")

    groups = dataset.groups if dataset.groups is not None else tuple(range(dataset.row_count))
    split = group_train_test_split(groups, test_fraction=config.test_fraction, seed=config.seed)
    if len(split.train_indices) < 4:
        raise ValueError("training split must contain at least four rows")

    train_x = [list(dataset.features[i]) for i in split.train_indices]
    train_y = [dataset.labels[i] for i in split.train_indices]
    test_x = [list(dataset.features[i]) for i in split.test_indices]
    test_y = [dataset.labels[i] for i in split.test_indices]

    model = train_tabular(train_x, train_y, version=config.model_name, ridge=config.ridge)
    coefficients = (*model.weights, model.intercept)
    if not all(isfinite(value) for value in coefficients):
        raise ValueError("training produced non-finite model coefficients")

    train_errors = _errors(model, train_x, train_y)
    test_errors = _errors(model, test_x, test_y)
    train_feature_set = {tuple(row) for row in train_x}
    metrics = {
        "train_mae": sum(abs(e) for e in train_errors) / len(train_errors),
        "test_mae": sum(abs(e) for e in test_errors) / len(test_errors),
        "test_rmse": sqrt(sum(e * e for e in test_errors) / len(test_errors)),
        "test_max_abs_error": max(abs(e) for e in test_errors),
        "test_rows_with_train_feature_match": float(
            sum(tuple(row) in train_feature_set for row in test_x)
        ),
    }
    if not all(isfinite(value) for value in metrics.values()):
        raise ValueError("evaluation produced non-finite metrics")

    drift = feature_drift(schema.features, train_x, test_x, bins=config.psi_bins)
    max_psi = max(item.psi for item in drift)
    metrics["max_feature_psi"] = max_psi

    reasons: list[str] = []
    if len(split.test_indices) < config.min_test_rows:
        reasons.append(
            f"test_rows {len(split.test_indices)} < required {config.min_test_rows}"
        )
    if metrics["test_mae"] > config.max_test_mae:
        reasons.append(
            f"test_mae {metrics['test_mae']:.6f} > maximum {config.max_test_mae:.6f}"
        )
    if max_psi > config.max_feature_psi:
        worst = max(drift, key=lambda item: item.psi)
        reasons.append(
            f"psi[{worst.feature}] {max_psi:.6f} > maximum {config.max_feature_psi:.6f}"
        )

    config_sha256 = config.sha256
    artifact_bytes = canonical_json_bytes(
        {
            "format": MODEL_ARTIFACT_FORMAT,
            "model_name": config.model_name,
            "model_type": MODEL_TYPE,
            "features": list(schema.features),
            "label": schema.label,
            "weights": list(model.weights),
            "intercept": model.intercept,
            "feature_means": list(model.feature_means),
            "feature_scales": list(model.feature_scales),
            "dataset_sha256": dataset.sha256,
            "config_sha256": config_sha256,
        }
    )
    artifact_sha256 = sha256_hex(artifact_bytes)
    version, version_source, dirty = resolve_code_version(code_version)

    record: dict[str, object] = {
        "format": RUN_RECORD_FORMAT,
        "model_id": f"{config.model_name}-{artifact_sha256[:16]}",
        "model_name": config.model_name,
        "model_type": MODEL_TYPE,
        "dataset": {
            "sha256": dataset.sha256,
            "row_count": dataset.row_count,
            "features": list(schema.features),
            "label": schema.label,
            "group": schema.group,
        },
        "config": config.to_dict(),
        "config_sha256": config_sha256,
        "code_version": version,
        "code_version_source": version_source,
        "code_tree_dirty": dirty,
        "split": {
            "strategy": "group" if schema.group is not None else "row",
            "seed": config.seed,
            "test_fraction": float(config.test_fraction),
            "train_rows": len(split.train_indices),
            "test_rows": len(split.test_indices),
            "train_groups": split.train_group_count,
            "test_groups": split.test_group_count,
            "train_indices_sha256": canonical_sha256(list(split.train_indices)),
            "test_indices_sha256": canonical_sha256(list(split.test_indices)),
        },
        "metrics": metrics,
        "drift": [
            {"feature": item.feature, "psi": item.psi, "reference": "train", "current": "test"}
            for item in drift
        ],
        "gate": {"allowed": not reasons, "reasons": reasons},
        "artifact": {
            "sha256": artifact_sha256,
            "size_bytes": len(artifact_bytes),
            "media_type": "application/json",
            "filename": ARTIFACT_FILENAME,
        },
    }
    record["record_sha256"] = record_sha256(record)
    return ExperimentResult(
        model=model, dataset=dataset, split=split, artifact_bytes=artifact_bytes, record=record
    )


def candidate_evaluation_export(record: Mapping[str, object]) -> dict[str, object]:
    """Project a verified run record into the control-plane candidate export.

    See ``docs/CANDIDATE_EVALUATION.md`` and
    ``contracts/candidate-evaluation.v1.schema.json``.
    """
    verified = verify_record(record)
    dataset = verified.get("dataset")
    artifact = verified.get("artifact")
    gate = verified.get("gate")
    metrics = verified.get("metrics")
    if not all(isinstance(part, Mapping) for part in (dataset, artifact, gate, metrics)):
        raise ValueError("run record is missing dataset, artifact, gate or metrics sections")
    return {
        "format": CANDIDATE_EXPORT_FORMAT,
        "model_id": verified["model_id"],
        "dataset_sha256": dataset["sha256"],
        "config_sha256": verified["config_sha256"],
        "code_version": verified["code_version"],
        "metrics": dict(metrics),
        "artifact_sha256": artifact["sha256"],
        "gate": {"allowed": gate["allowed"], "reasons": list(gate["reasons"])},
        "run_record_sha256": verified["record_sha256"],
    }


@dataclass(frozen=True, slots=True)
class WrittenRun:
    artifact_path: Path
    record_path: Path
    verified: bool
    created: bool


def write_run_outputs(result: ExperimentResult, output_dir: str | Path) -> WrittenRun:
    """Write ``model.json`` and ``run_record.json`` and verify them by re-reading.

    Existing files are never overwritten: identical bytes are accepted
    idempotently, different bytes raise ``FileExistsError``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        ARTIFACT_FILENAME: result.artifact_bytes,
        RECORD_FILENAME: result.record_bytes,
    }
    for name, payload in outputs.items():
        path = directory / name
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            raise FileExistsError(f"{path} already exists with different content")
    created = False
    for name, payload in outputs.items():
        path = directory / name
        if path.exists():
            continue
        with path.open("xb") as handle:
            handle.write(payload)
        created = True
    manifest = build_integrity_manifest(directory, list(outputs))
    expected = {name: sha256_hex(payload) for name, payload in outputs.items()}
    verified = verify_integrity_manifest(directory, manifest).valid and all(
        entry.sha256 == expected[entry.path] for entry in manifest.artifacts
    )
    return WrittenRun(
        artifact_path=directory / ARTIFACT_FILENAME,
        record_path=directory / RECORD_FILENAME,
        verified=verified,
        created=created,
    )
