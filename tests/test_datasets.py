import pytest

from mlops_pipeline.canonical import canonical_sha256
from mlops_pipeline.datasets import (
    DatasetSchema,
    DatasetValidationError,
    compute_dataset_sha256,
    load_dataset,
    validate_dataset,
)

SCHEMA = DatasetSchema(features=("a", "b"), label="y", group="g")


def rows(n: int = 6) -> list[dict[str, object]]:
    return [{"a": float(i), "b": float(i * i), "y": float(2 * i), "g": f"g{i // 2}"}
            for i in range(n)]


def codes(report) -> list[str]:
    return [issue.code for issue in report.issues]


def test_valid_dataset_reports_hash_and_counts() -> None:
    report = validate_dataset(rows(), SCHEMA)
    assert report.valid
    assert report.row_count == 6
    assert report.group_count == 3
    assert report.error_count == 0
    assert len(report.dataset_sha256) == 64


def test_dataset_hash_matches_documented_canonical_form() -> None:
    data = rows(4)
    expected = canonical_sha256(
        {
            "format": "mlops-dataset.v1",
            "columns": {"features": ["a", "b"], "label": "y", "group": "g"},
            "rows": [[r["a"], r["b"], r["y"], r["g"]] for r in data],
        }
    )
    assert validate_dataset(data, SCHEMA).dataset_sha256 == expected


def test_int_and_float_values_produce_the_same_version() -> None:
    as_float = rows()
    as_int = [{**r, "a": int(r["a"]), "y": int(r["y"])} for r in as_float]
    assert validate_dataset(as_int, SCHEMA).dataset_sha256 == validate_dataset(
        as_float, SCHEMA
    ).dataset_sha256


def test_negative_zero_is_canonicalised() -> None:
    positive = rows()
    negative = [dict(r) for r in positive]
    negative[0]["a"] = -0.0
    assert validate_dataset(negative, SCHEMA).dataset_sha256 == validate_dataset(
        positive, SCHEMA
    ).dataset_sha256


def test_any_value_change_changes_the_version() -> None:
    base = validate_dataset(rows(), SCHEMA).dataset_sha256
    changed = rows()
    changed[3]["b"] = changed[3]["b"] + 1e-9
    regrouped = rows()
    regrouped[0]["g"] = "other"
    reordered = list(reversed(rows()))
    assert validate_dataset(changed, SCHEMA).dataset_sha256 != base
    assert validate_dataset(regrouped, SCHEMA).dataset_sha256 != base
    assert validate_dataset(reordered, SCHEMA).dataset_sha256 != base


def test_extra_columns_are_rejected_by_default_and_excluded_when_allowed() -> None:
    data = [{**r, "note": "free text"} for r in rows()]
    strict = validate_dataset(data, SCHEMA)
    assert not strict.valid
    assert "UNEXPECTED_COLUMN" in codes(strict)
    permissive = DatasetSchema(features=("a", "b"), label="y", group="g",
                               allow_extra_columns=True)
    allowed = validate_dataset(data, permissive)
    assert allowed.valid
    assert allowed.dataset_sha256 == validate_dataset(rows(), SCHEMA).dataset_sha256


@pytest.mark.parametrize(
    "mutation,code",
    [
        ({"a": None}, "MISSING_VALUE"),
        ({"y": None}, "MISSING_LABEL"),
        ({"a": "1.0"}, "INVALID_TYPE"),
        ({"a": True}, "INVALID_TYPE"),
        ({"a": float("nan")}, "NON_FINITE"),
        ({"y": float("inf")}, "NON_FINITE"),
        ({"a": 10**400}, "NON_FINITE"),
        ({"g": None}, "MISSING_GROUP"),
        ({"g": "  "}, "MISSING_GROUP"),
        ({"g": 1.5}, "INVALID_TYPE"),
    ],
)
def test_invalid_values_are_reported_with_row_and_column(mutation, code) -> None:
    data = rows()
    data[2].update(mutation)
    report = validate_dataset(data, SCHEMA)
    assert not report.valid
    assert report.dataset_sha256 is None
    issue = next(issue for issue in report.issues if issue.code == code)
    assert issue.row == 2
    assert issue.column == next(iter(mutation))


def test_missing_key_counts_as_missing_value() -> None:
    data = rows()
    del data[1]["b"]
    del data[4]["b"]
    report = validate_dataset(data, SCHEMA)
    assert report.missing_values["b"] == 2


def test_structural_errors() -> None:
    assert codes(validate_dataset("not a list", SCHEMA)) == ["NOT_A_LIST"]
    assert "TOO_FEW_ROWS" in codes(validate_dataset(rows(2), SCHEMA))
    assert "NOT_A_RECORD" in codes(validate_dataset([*rows(), 5], SCHEMA))
    assert "TOO_MANY_ROWS" in codes(validate_dataset(rows(8), SCHEMA, max_rows=7))
    single_group = [{**r, "g": "same"} for r in rows()]
    assert "TOO_FEW_GROUPS" in codes(validate_dataset(single_group, SCHEMA))


def test_warnings_do_not_invalidate() -> None:
    data = rows()
    for record in data:
        record["b"] = 1.0
    data[5] = dict(data[4])
    report = validate_dataset(data, SCHEMA)
    assert report.valid
    assert {"CONSTANT_FEATURE", "DUPLICATE_ROWS"} <= set(codes(report))
    assert report.duplicate_rows == 1


def test_issue_list_is_bounded_but_counts_are_exact() -> None:
    data = [{"a": None, "b": None, "y": None, "g": None} for _ in range(40)]
    report = validate_dataset(data, SCHEMA)
    assert report.error_count == 160
    assert len(report.issues) == 50
    assert report.issues_truncated


def test_unlabelled_validation_ignores_label_column() -> None:
    data = [{k: v for k, v in r.items() if k != "y"} for r in rows()]
    assert not validate_dataset(data, SCHEMA).valid
    report = validate_dataset(data, SCHEMA, require_label=False)
    assert report.valid
    dataset = load_dataset(data, SCHEMA, require_label=False)
    assert dataset.labels is None
    assert dataset.sha256 == compute_dataset_sha256(
        SCHEMA, dataset.features, None, dataset.groups
    )
    assert dataset.sha256 != validate_dataset(rows(), SCHEMA).dataset_sha256


def test_load_dataset_raises_with_report() -> None:
    data = rows()
    data[0]["y"] = None
    with pytest.raises(DatasetValidationError) as caught:
        load_dataset(data, SCHEMA)
    assert caught.value.report.error_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"features": (), "label": "y"},
        {"features": ("a", "a"), "label": "y"},
        {"features": ("a",), "label": "a"},
        {"features": ("a",), "label": "y", "group": "a"},
        {"features": ("",), "label": "y"},
        {"features": ["a"], "label": "y"},
    ],
)
def test_schema_rejects_ambiguous_definitions(kwargs) -> None:
    with pytest.raises(ValueError):
        DatasetSchema(**kwargs)


def test_schema_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        DatasetSchema.from_dict({"features": ["a"], "label": "y", "weights": "w"})
    with pytest.raises(TypeError):
        DatasetSchema.from_dict({"features": "a", "label": "y"})
