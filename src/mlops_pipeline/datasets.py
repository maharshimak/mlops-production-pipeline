"""Tabular dataset validation and content-addressed versioning.

A dataset is a list of JSON-like records (``dict`` per row) described by a
:class:`DatasetSchema`: numeric feature columns, one numeric label column and an
optional group/entity column used for leakage-safe splitting.

Validation is deterministic and rule-based. The dataset version is the SHA-256
of a canonical JSON projection of the schema columns (row order is significant;
extra columns are excluded), so numerically identical inputs such as ``1`` and
``1.0`` produce the same version.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

from mlops_pipeline.canonical import canonical_float, canonical_sha256

DATASET_FORMAT = "mlops-dataset.v1"
MAX_REPORTED_ISSUES = 50
DEFAULT_MAX_ROWS = 100_000
MAX_FEATURES = 256


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    features: tuple[str, ...]
    label: str
    group: str | None = None
    allow_extra_columns: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.features, tuple) or not self.features:
            raise ValueError("schema.features must be a non-empty tuple of column names")
        if len(self.features) > MAX_FEATURES:
            raise ValueError(f"schema.features supports at most {MAX_FEATURES} columns")
        names = [*self.features, self.label]
        if self.group is not None:
            names.append(self.group)
        if not all(isinstance(name, str) and name.strip() for name in names):
            raise ValueError("schema column names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("schema column names must be unique across features, label and group")
        if type(self.allow_extra_columns) is not bool:
            raise ValueError("schema.allow_extra_columns must be boolean")

    @property
    def columns(self) -> tuple[str, ...]:
        extra = (self.group,) if self.group is not None else ()
        return (*self.features, self.label, *extra)

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "label": self.label,
            "group": self.group,
            "allow_extra_columns": self.allow_extra_columns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetSchema:
        if not isinstance(value, Mapping):
            raise TypeError("schema must be an object")
        unknown = set(value) - {"features", "label", "group", "allow_extra_columns"}
        if unknown:
            raise ValueError(f"unknown schema fields: {sorted(unknown)}")
        features = value.get("features")
        if not isinstance(features, list):
            raise TypeError("schema.features must be a list of column names")
        label = value.get("label")
        if not isinstance(label, str):
            raise TypeError("schema.label must be a string")
        group = value.get("group")
        if group is not None and not isinstance(group, str):
            raise ValueError("schema.group must be a string or null")
        return cls(
            features=tuple(features),
            label=label,
            group=group,
            allow_extra_columns=value.get("allow_extra_columns", False),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    column: str | None = None
    row: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "row": self.row,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    valid: bool
    row_count: int
    error_count: int
    warning_count: int
    issues: tuple[ValidationIssue, ...]
    issues_truncated: bool
    missing_values: dict[str, int]
    group_count: int | None
    duplicate_rows: int
    dataset_sha256: str | None
    label_required: bool
    schema: DatasetSchema = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "row_count": self.row_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "issues_truncated": self.issues_truncated,
            "missing_values": dict(self.missing_values),
            "group_count": self.group_count,
            "duplicate_rows": self.duplicate_rows,
            "dataset_sha256": self.dataset_sha256,
            "label_required": self.label_required,
            "schema": self.schema.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    schema: DatasetSchema
    features: tuple[tuple[float, ...], ...]
    labels: tuple[float, ...] | None
    groups: tuple[str | int, ...] | None
    sha256: str

    @property
    def row_count(self) -> int:
        return len(self.features)


class DatasetValidationError(ValueError):
    def __init__(self, report: DatasetValidationReport) -> None:
        super().__init__(f"dataset failed validation with {report.error_count} error(s)")
        self.report = report


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _finite_float(value: float) -> float | None:
    try:
        number = float(value)
    except OverflowError:
        return None
    if not isfinite(number):
        return None
    return canonical_float(number)


def _canonical_rows(
    schema: DatasetSchema,
    features: Sequence[Sequence[float]],
    labels: Sequence[float] | None,
    groups: Sequence[str | int] | None,
) -> list[list[object]]:
    rows: list[list[object]] = []
    for index, row in enumerate(features):
        canonical: list[object] = [*row]
        if labels is not None:
            canonical.append(labels[index])
        if groups is not None:
            canonical.append(groups[index])
        rows.append(canonical)
    return rows


def compute_dataset_sha256(
    schema: DatasetSchema,
    features: Sequence[Sequence[float]],
    labels: Sequence[float] | None,
    groups: Sequence[str | int] | None = None,
) -> str:
    """SHA-256 of the canonical projection of already-validated columns.

    ``labels=None`` hashes an unlabelled (feature-only) sample; the label column
    name is then recorded as ``null`` so labelled and unlabelled versions differ.
    """
    if labels is not None and len(labels) != len(features):
        raise ValueError("labels must align with feature rows")
    if groups is not None and len(groups) != len(features):
        raise ValueError("groups must align with feature rows")
    return canonical_sha256(
        {
            "format": DATASET_FORMAT,
            "columns": {
                "features": list(schema.features),
                "label": schema.label if labels is not None else None,
                "group": schema.group,
            },
            "rows": _canonical_rows(schema, features, labels, groups),
        }
    )


def _validate(
    records: object,
    schema: DatasetSchema,
    *,
    min_rows: int,
    max_rows: int,
    require_label: bool = True,
) -> tuple[DatasetValidationReport, ValidatedDataset | None]:
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError("min_rows must be a positive integer")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < min_rows:
        raise ValueError("max_rows must be an integer >= min_rows")
    if type(require_label) is not bool:
        raise ValueError("require_label must be boolean")

    issues: list[ValidationIssue] = []
    error_count = 0
    warning_count = 0

    def add(severity: str, code: str, message: str, column: str | None = None,
            row: int | None = None) -> None:
        nonlocal error_count, warning_count
        if severity == "error":
            error_count += 1
        else:
            warning_count += 1
        if len(issues) < MAX_REPORTED_ISSUES:
            issues.append(ValidationIssue(severity, code, message, column, row))

    missing: Counter[str] = Counter({column: 0 for column in schema.columns})
    features: list[tuple[float, ...]] = []
    labels: list[float] = []
    groups: list[str | int] = []

    if not isinstance(records, list):
        add("error", "NOT_A_LIST", "records must be a list of objects")
        records = []
    elif len(records) > max_rows:
        add("error", "TOO_MANY_ROWS", f"{len(records)} rows exceeds the limit of {max_rows}")
        records = []
    elif len(records) < min_rows:
        add("error", "TOO_FEW_ROWS", f"{len(records)} rows; at least {min_rows} required")

    expected = set(schema.columns)
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            add("error", "NOT_A_RECORD", "row is not an object", row=row_index)
            continue
        if not schema.allow_extra_columns:
            for column in sorted(set(record) - expected, key=str):
                add("error", "UNEXPECTED_COLUMN", f"column {column!r} is not in the schema",
                    column=str(column), row=row_index)
        row_ok = True
        row_values: list[float] = []
        checked = (*schema.features, schema.label) if require_label else schema.features
        for column in checked:
            value = record.get(column)
            if value is None:
                missing[column] += 1
                code = "MISSING_LABEL" if column == schema.label else "MISSING_VALUE"
                add("error", code, "value is missing", column=column, row=row_index)
                row_ok = False
            elif not _is_number(value):
                add("error", "INVALID_TYPE",
                    f"expected a number, got {type(value).__name__}", column=column,
                    row=row_index)
                row_ok = False
            else:
                number = _finite_float(value)
                if number is None:
                    add("error", "NON_FINITE", "value must be finite", column=column,
                        row=row_index)
                    row_ok = False
                else:
                    row_values.append(number)
        group_value: str | int | None = None
        if schema.group is not None:
            group_value = record.get(schema.group)
            if group_value is None:
                missing[schema.group] += 1
                add("error", "MISSING_GROUP", "group value is missing", column=schema.group,
                    row=row_index)
                row_ok = False
            elif isinstance(group_value, bool) or not isinstance(group_value, (str, int)):
                add("error", "INVALID_TYPE",
                    f"group must be a string or integer, got {type(group_value).__name__}",
                    column=schema.group, row=row_index)
                row_ok = False
            elif isinstance(group_value, str) and not group_value.strip():
                add("error", "MISSING_GROUP", "group value is blank", column=schema.group,
                    row=row_index)
                row_ok = False
        if row_ok:
            features.append(tuple(row_values[: len(schema.features)]))
            if require_label:
                labels.append(row_values[-1])
            if group_value is not None:
                groups.append(group_value)

    duplicate_rows = 0
    group_count: int | None = None
    dataset: ValidatedDataset | None = None
    if error_count == 0:
        for column_index, column in enumerate(schema.features):
            if len({row[column_index] for row in features}) == 1:
                add("warning", "CONSTANT_FEATURE", "feature has a single distinct value",
                    column=column)
        if require_label and len(set(labels)) == 1:
            add("warning", "CONSTANT_LABEL", "label has a single distinct value",
                column=schema.label)
        keys = zip(features, labels, strict=True) if require_label else iter(features)
        counts = Counter(keys)
        duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
        if duplicate_rows:
            add("warning", "DUPLICATE_ROWS",
                f"{duplicate_rows} row(s) duplicate another row's values")
        if schema.group is not None:
            group_count = len(set(groups))
            if group_count < 2:
                add("error", "TOO_FEW_GROUPS", "at least two distinct groups are required",
                    column=schema.group)
    if error_count == 0:
        dataset = ValidatedDataset(
            schema=schema,
            features=tuple(features),
            labels=tuple(labels) if require_label else None,
            groups=tuple(groups) if schema.group is not None else None,
            sha256=compute_dataset_sha256(
                schema,
                features,
                labels if require_label else None,
                groups if schema.group is not None else None,
            ),
        )

    report = DatasetValidationReport(
        valid=error_count == 0,
        row_count=len(records),
        error_count=error_count,
        warning_count=warning_count,
        issues=tuple(issues),
        issues_truncated=error_count + warning_count > len(issues),
        missing_values=dict(missing),
        group_count=group_count,
        duplicate_rows=duplicate_rows,
        dataset_sha256=dataset.sha256 if dataset is not None else None,
        label_required=require_label,
        schema=schema,
    )
    return report, dataset


def validate_dataset(
    records: object,
    schema: DatasetSchema,
    *,
    min_rows: int = 4,
    max_rows: int = DEFAULT_MAX_ROWS,
    require_label: bool = True,
) -> DatasetValidationReport:
    """Validate records against ``schema`` and return a structured report.

    With ``require_label=False`` the label column is not read (it may be absent),
    which is how unlabelled production samples are validated for drift checks.
    """
    report, _ = _validate(
        records, schema, min_rows=min_rows, max_rows=max_rows, require_label=require_label
    )
    return report


def load_dataset(
    records: object,
    schema: DatasetSchema,
    *,
    min_rows: int = 4,
    max_rows: int = DEFAULT_MAX_ROWS,
    require_label: bool = True,
) -> ValidatedDataset:
    """Validate and return typed columns; raise :class:`DatasetValidationError` on errors."""
    report, dataset = _validate(
        records, schema, min_rows=min_rows, max_rows=max_rows, require_label=require_label
    )
    if dataset is None:
        raise DatasetValidationError(report)
    return dataset
