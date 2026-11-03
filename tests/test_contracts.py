import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from conftest import SCHEMA_DICT, make_records
from jsonschema import Draft202012Validator

from mlops_pipeline.canonical import sha256_hex
from mlops_pipeline.contracts import SYSTEM, capabilities, run_operation

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = Draft202012Validator(
    json.loads((ROOT / "contracts" / "makma.v1.schema.json").read_text(encoding="utf-8")),
    format_checker=Draft202012Validator.FORMAT_CHECKER,
)
CANDIDATE = Draft202012Validator(
    json.loads(
        (ROOT / "contracts" / "candidate-evaluation.v1.schema.json").read_text(encoding="utf-8")
    )
)


def check(envelope: dict, status: str = "ok") -> dict:
    ENVELOPE.validate(envelope)
    # Round-trips through strict JSON (no NaN/Infinity, no non-JSON types).
    assert json.loads(json.dumps(envelope, allow_nan=False)) == envelope
    assert datetime.fromisoformat(envelope["started_at"]).utcoffset() == timedelta(0)
    assert envelope["system"] == SYSTEM
    assert envelope["execution_mode"] == "deterministic_offline"
    assert envelope["status"] == status
    return envelope


def payload(**extra) -> dict:
    return {"records": make_records(), "schema": dict(SCHEMA_DICT), **extra}


def test_capabilities_declare_real_operations() -> None:
    caps = capabilities()
    assert caps["contract_version"] == "makma.v1"
    assert caps["system"] == SYSTEM
    names = {op["name"] for op in caps["operations"]}
    assert names == {
        "validate_dataset", "train", "evaluate_candidate", "model_card", "evaluate_drift"
    }
    for op in caps["operations"]:
        Draft202012Validator.check_schema(op["input_schema"])
        assert op["side_effects"] is (op["name"] == "train")
        assert op["requires_approval"] is False


def test_operation_payloads_conform_to_declared_input_schemas(tmp_path) -> None:
    schemas = {op["name"]: op["input_schema"] for op in capabilities()["operations"]}
    Draft202012Validator(schemas["validate_dataset"]).validate(payload())
    Draft202012Validator(schemas["evaluate_candidate"]).validate(
        payload(config={"seed": 3}, code_version="abc")
    )
    Draft202012Validator(schemas["train"]).validate(payload(output_dir=str(tmp_path)))


def test_validate_dataset_ok_and_reports_problems() -> None:
    ok = check(run_operation("validate_dataset", payload(), run_id="r1", correlation_id="c1"))
    assert ok["run_id"] == "r1" and ok["correlation_id"] == "c1"
    assert ok["data"]["valid"] is True
    assert ok["provenance"][0]["sha256"] == ok["data"]["dataset_sha256"]

    records = make_records()
    records[0]["y"] = None
    bad = check(run_operation("validate_dataset", {"records": records, "schema": SCHEMA_DICT}))
    assert bad["data"]["valid"] is False
    assert bad["data"]["issues"][0]["code"] == "MISSING_LABEL"
    assert bad["provenance"] == []


def test_evaluate_candidate_returns_control_plane_export() -> None:
    envelope = check(run_operation("evaluate_candidate", payload(code_version="abc123")))
    candidate = envelope["data"]["candidate"]
    CANDIDATE.validate(candidate)
    record = envelope["data"]["run_record"]
    assert candidate["artifact_sha256"] == record["artifact"]["sha256"]
    kinds = {item["kind"]: item["sha256"] for item in envelope["provenance"]}
    assert kinds == {
        "dataset": candidate["dataset_sha256"],
        "artifact": candidate["artifact_sha256"],
        "record": candidate["run_record_sha256"],
    }


def test_train_writes_verified_artifact_and_refuses_conflicts(tmp_path) -> None:
    out = tmp_path / "candidate"
    envelope = check(run_operation("train", payload(output_dir=str(out), code_version="v1")))
    data = envelope["data"]
    artifact = Path(data["artifact_path"])
    assert data["verified"] is True and data["created"] is True
    assert sha256_hex(artifact.read_bytes()) == data["run_record"]["artifact"]["sha256"]
    artifact_ref = next(p for p in envelope["provenance"] if p["kind"] == "artifact")
    assert artifact_ref["uri"].startswith("file:")

    conflict = check(
        run_operation(
            "train", payload(output_dir=str(out), code_version="v1", config={"seed": 9})
        ),
        status="error",
    )
    assert conflict["error"]["code"] == "ARTIFACT_CONFLICT"


def test_model_card_operation_round_trip() -> None:
    record = run_operation("evaluate_candidate", payload(code_version="v"))["data"]["run_record"]
    envelope = check(run_operation("model_card", {"run_record": record}))
    assert envelope["data"]["markdown"].startswith("# Model card: ")

    record["metrics"]["test_mae"] = 0.0
    tampered = check(run_operation("model_card", {"run_record": record}), status="error")
    assert tampered["error"]["code"] == "INVALID_INPUT"


def test_evaluate_drift_on_unlabelled_current_sample() -> None:
    reference = make_records()
    current = [
        {"x1": r["x1"] + 1000.0, "x2": r["x2"], "customer": r["customer"]} for r in reference
    ]
    envelope = check(
        run_operation(
            "evaluate_drift",
            {"schema": SCHEMA_DICT, "reference_records": reference, "current_records": current},
        )
    )
    features = {item["feature"]: item for item in envelope["data"]["features"]}
    assert features["x1"]["exceeds_threshold"] is True
    assert features["x2"]["psi"] == pytest.approx(0.0)
    assert envelope["data"]["drift_detected"] is True


@pytest.mark.parametrize(
    "operation,body,code",
    [
        ("validate_dataset", {"schema": SCHEMA_DICT}, "INVALID_INPUT"),
        ("validate_dataset", {**payload(), "extra": 1}, "INVALID_INPUT"),
        ("validate_dataset", {"records": "x", "schema": SCHEMA_DICT}, "INVALID_INPUT"),
        ("validate_dataset", {"records": [], "schema": {"features": "x1", "label": "y"}},
         "INVALID_INPUT"),
        ("evaluate_candidate", {**payload(), "config": {"lr": 1}}, "INVALID_INPUT"),
        ("evaluate_candidate", {**payload(), "config": {"seed": "1"}}, "INVALID_INPUT"),
        ("evaluate_candidate", {**payload(), "code_version": "has space"}, "INVALID_INPUT"),
        ("train", payload(), "INVALID_INPUT"),
        ("model_card", {"run_record": "nope"}, "INVALID_INPUT"),
        ("evaluate_drift", {"schema": SCHEMA_DICT, "reference_records": [],
                            "current_records": [], "bins": 1}, "INVALID_INPUT"),
        ("evaluate_candidate", {"records": [{"x1": 1.0}], "schema": SCHEMA_DICT},
         "DATASET_INVALID"),
    ],
)
def test_invalid_inputs_return_error_envelopes(operation, body, code) -> None:
    envelope = check(run_operation(operation, body), status="error")
    assert envelope["error"]["code"] == code
    assert envelope["data"] is None


def test_dataset_invalid_error_carries_validation_report() -> None:
    records = make_records()
    records[2]["x2"] = "oops"
    envelope = check(
        run_operation("evaluate_candidate", {"records": records, "schema": SCHEMA_DICT}),
        status="error",
    )
    report = envelope["error"]["details"]["report"]
    assert report["issues"][0] == {
        "severity": "error", "code": "INVALID_TYPE", "message": "expected a number, got str",
        "column": "x2", "row": 2,
    }


def test_unknown_operation_and_bad_payload_types() -> None:
    unknown = check(run_operation("delete_everything", {}), status="error")
    assert unknown["error"]["code"] == "UNKNOWN_OPERATION"
    weird = check(run_operation("Not Valid!", {}), status="error")
    assert weird["operation"] == "unknown"
    not_object = check(run_operation("validate_dataset", ["x"]), status="error")
    assert not_object["error"]["code"] == "INVALID_INPUT"
    bad_ids = check(run_operation("validate_dataset", payload(), run_id=""), status="error")
    assert bad_ids["error"]["code"] == "INVALID_INPUT"


def test_ids_are_fresh_per_call_and_correlation_defaults_to_run_id() -> None:
    first = run_operation("validate_dataset", payload())
    second = run_operation("validate_dataset", payload())
    assert first["run_id"] != second["run_id"]
    assert first["trace_id"] != second["trace_id"]
    assert first["correlation_id"] == first["run_id"]


def test_unexpected_failures_are_contained(monkeypatch) -> None:
    from mlops_pipeline import contracts

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(contracts, "validate_dataset", boom)
    envelope = check(run_operation("validate_dataset", payload()), status="error")
    assert envelope["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in json.dumps(envelope)


def test_singular_training_system_is_reported_as_training_failure() -> None:
    records = [{**r, "x2": r["x1"]} for r in make_records()]
    envelope = check(
        run_operation(
            "evaluate_candidate",
            {"records": records, "schema": SCHEMA_DICT, "config": {"ridge": 0}},
        ),
        status="error",
    )
    assert envelope["error"]["code"] == "TRAINING_FAILED"
