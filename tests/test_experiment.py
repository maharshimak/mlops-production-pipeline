import json
import random
import subprocess

import pytest
from conftest import make_records

from mlops_pipeline import experiment as experiment_module
from mlops_pipeline.canonical import canonical_json_bytes, sha256_hex
from mlops_pipeline.datasets import DatasetSchema, DatasetValidationError
from mlops_pipeline.experiment import (
    ExperimentConfig,
    candidate_evaluation_export,
    record_sha256,
    resolve_code_version,
    run_experiment,
    verify_record,
    write_run_outputs,
)
from mlops_pipeline.tabular import predict_tabular, train_tabular

SCHEMA = DatasetSchema(features=("x1", "x2"), label="y", group="customer")


def test_ridge_baseline_matches_hand_computed_solution() -> None:
    # Standardised x = [-1, -1, 1, 1] (mean 0, population std 1), y = [0, 0, 2, 2].
    # Normal equations: [[4, 0], [0, 4 + ridge]] @ [b, w] = [4, 4] -> b = 1, w = 4 / (4 + ridge).
    model = train_tabular([[-1.0], [-1.0], [1.0], [1.0]], [0.0, 0.0, 2.0, 2.0], ridge=4.0)
    assert model.intercept == pytest.approx(1.0)
    assert model.weights == (pytest.approx(0.5),)
    assert predict_tabular(model, [1.0]) == pytest.approx(1.5)
    unregularised = train_tabular([[-1.0], [-1.0], [1.0], [1.0]], [0.0, 0.0, 2.0, 2.0], ridge=0.0)
    assert unregularised.weights == (pytest.approx(1.0),)


def test_run_record_binds_hashes_and_is_reproducible() -> None:
    records = make_records()
    first = run_experiment(records, SCHEMA, code_version="test-sha")
    second = run_experiment(records, SCHEMA, code_version="test-sha")

    assert first.record_bytes == second.record_bytes
    assert first.artifact_bytes == second.artifact_bytes
    record = first.record
    assert record["dataset"]["sha256"] == first.dataset.sha256
    assert record["config_sha256"] == ExperimentConfig().sha256
    assert record["artifact"]["sha256"] == sha256_hex(first.artifact_bytes)
    assert record["model_id"] == f"baseline-ridge-{record['artifact']['sha256'][:16]}"
    assert record["code_version"] == "test-sha"
    assert record["code_version_source"] == "caller"
    assert record["record_sha256"] == record_sha256(record)
    artifact = json.loads(first.artifact_bytes)
    assert artifact["dataset_sha256"] == record["dataset"]["sha256"]
    assert artifact["config_sha256"] == record["config_sha256"]


def test_exact_linear_data_is_fit_and_split_is_group_safe() -> None:
    records = make_records(entities=30, rows_per_entity=3)
    result = run_experiment(
        records, SCHEMA, ExperimentConfig(max_feature_psi=100.0), code_version="v1"
    )
    metrics = result.record["metrics"]
    assert metrics["test_mae"] < 1e-6
    assert metrics["train_mae"] < 1e-6
    assert result.record["gate"] == {"allowed": True, "reasons": []}
    train_groups = {records[i]["customer"] for i in result.split.train_indices}
    test_groups = {records[i]["customer"] for i in result.split.test_indices}
    assert train_groups.isdisjoint(test_groups)
    assert result.record["split"]["strategy"] == "group"


def test_config_dataset_or_code_change_changes_identity() -> None:
    records = make_records()
    base = run_experiment(records, SCHEMA, code_version="a").record
    seeded = run_experiment(records, SCHEMA, ExperimentConfig(seed=1), code_version="a").record
    other_code = run_experiment(records, SCHEMA, code_version="b").record
    changed = [dict(r) for r in records]
    changed[0]["y"] = changed[0]["y"] + 0.5
    other_data = run_experiment(changed, SCHEMA, code_version="a").record

    assert seeded["config_sha256"] != base["config_sha256"]
    assert other_data["dataset"]["sha256"] != base["dataset"]["sha256"]
    assert other_code["artifact"]["sha256"] == base["artifact"]["sha256"]
    assert other_code["record_sha256"] != base["record_sha256"]


def test_gate_reports_each_failure() -> None:
    records = make_records()
    rng = random.Random(0)
    for record in records:
        record["y"] = record["y"] + rng.uniform(-5, 5)
    result = run_experiment(
        records,
        SCHEMA,
        ExperimentConfig(max_test_mae=0.01, max_feature_psi=0.0, min_test_rows=50),
        code_version="v",
    )
    reasons = result.record["gate"]["reasons"]
    assert result.record["gate"]["allowed"] is False
    assert any(reason.startswith("test_rows") for reason in reasons)
    assert any(reason.startswith("test_mae") for reason in reasons)
    assert any(reason.startswith("psi[") for reason in reasons)


def test_duplicate_feature_rows_across_split_are_counted() -> None:
    records = make_records(entities=20, rows_per_entity=1)
    duplicated = records + [{**r, "customer": f"dup-{r['customer']}"} for r in records]
    result = run_experiment(duplicated, SCHEMA, code_version="v")
    assert result.record["metrics"]["test_rows_with_train_feature_match"] > 0


def test_ungrouped_schema_uses_row_split() -> None:
    schema = DatasetSchema(features=("x1", "x2"), label="y", allow_extra_columns=True)
    result = run_experiment(make_records(), schema, code_version="v")
    assert result.record["split"]["strategy"] == "row"
    assert result.record["dataset"]["group"] is None


def test_invalid_dataset_raises_validation_error() -> None:
    records = make_records()
    records[3]["x1"] = None
    with pytest.raises(DatasetValidationError):
        run_experiment(records, SCHEMA, code_version="v")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": "Bad Name"},
        {"test_fraction": 0.9},
        {"seed": -1},
        {"seed": True},
        {"ridge": float("nan")},
        {"max_test_mae": -1},
        {"psi_bins": 1},
    ],
)
def test_config_validation(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExperimentConfig(**kwargs)


def test_config_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ExperimentConfig.from_dict({"learning_rate": 0.1})


def test_verify_record_detects_tampering() -> None:
    record = run_experiment(make_records(), SCHEMA, code_version="v").record
    verify_record(record)
    tampered = json.loads(json.dumps(record))
    tampered["metrics"]["test_mae"] = 0.0
    with pytest.raises(ValueError, match="does not match"):
        verify_record(tampered)
    with pytest.raises(ValueError, match="format"):
        verify_record({**record, "format": "other"})


def test_candidate_export_projects_record_fields() -> None:
    record = run_experiment(make_records(), SCHEMA, code_version="abc123").record
    export = candidate_evaluation_export(record)
    assert export == {
        "format": "mlops-candidate-evaluation.v1",
        "model_id": record["model_id"],
        "dataset_sha256": record["dataset"]["sha256"],
        "config_sha256": record["config_sha256"],
        "code_version": "abc123",
        "metrics": record["metrics"],
        "artifact_sha256": record["artifact"]["sha256"],
        "gate": record["gate"],
        "run_record_sha256": record["record_sha256"],
    }


def test_write_run_outputs_verifies_and_never_overwrites(tmp_path) -> None:
    result = run_experiment(make_records(), SCHEMA, code_version="v")
    written = write_run_outputs(result, tmp_path / "run")
    assert written.verified and written.created
    assert written.artifact_path.read_bytes() == result.artifact_bytes
    assert sha256_hex(written.artifact_path.read_bytes()) == result.record["artifact"]["sha256"]
    assert written.record_path.read_bytes() == canonical_json_bytes(result.record)

    again = write_run_outputs(result, tmp_path / "run")
    assert again.verified and not again.created

    other = run_experiment(make_records(), SCHEMA, ExperimentConfig(seed=3), code_version="v")
    with pytest.raises(FileExistsError):
        write_run_outputs(other, tmp_path / "run")
    assert written.artifact_path.read_bytes() == result.artifact_bytes


def test_code_version_detection_reports_unknown_when_git_unavailable(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise FileNotFoundError("git")

    experiment_module._git_code_version.cache_clear()
    monkeypatch.setattr(experiment_module.subprocess, "run", broken)
    try:
        assert resolve_code_version() == ("unknown", "unknown", None)
    finally:
        experiment_module._git_code_version.cache_clear()


def test_code_version_ignores_unrelated_enclosing_repository(monkeypatch, tmp_path) -> None:
    def foreign_repo(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=f"{tmp_path}\n{'a' * 40}\n")

    experiment_module._git_code_version.cache_clear()
    monkeypatch.setattr(experiment_module.subprocess, "run", foreign_repo)
    try:
        assert resolve_code_version()[0] == "unknown"
    finally:
        experiment_module._git_code_version.cache_clear()


def test_code_version_detected_value_is_sha_or_unknown() -> None:
    version, source, _dirty = resolve_code_version()
    assert (source == "git" and len(version) in (40, 64)) or (version, source) == (
        "unknown",
        "unknown",
    )


def test_explicit_code_version_is_validated() -> None:
    with pytest.raises(ValueError):
        resolve_code_version("bad version with spaces")
