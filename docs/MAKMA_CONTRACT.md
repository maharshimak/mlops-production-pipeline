# `makma.v1` execution contract

`mlops_pipeline.contracts` implements the shared MAK'MA envelope. The JSON Schema in [`contracts/makma.v1.schema.json`](../contracts/makma.v1.schema.json) is vendored byte-for-byte from the shared contract and `tests/test_contracts.py` validates real `run_operation` outputs against it (success, invalid input, dataset errors, unknown operation, contained internal errors).

Pydantic is not a dependency of this package, so the envelope is a frozen dataclass whose constructor enforces the contract invariants (`status == "ok"` if and only if `error` is null; `error`/`blocked` imply `data` is null; trace/span id formats; non-negative duration).

## Entry points

```python
from mlops_pipeline.contracts import capabilities, run_operation

capabilities()  # {"contract_version": "makma.v1", "system": "mlops-production-pipeline", ...}
run_operation("validate_dataset", {"records": [...], "schema": {"features": ["x"], "label": "y"}})
```

`run_id` and `correlation_id` may be supplied (1-128 characters); otherwise a UUID4 hex is generated and `correlation_id` defaults to `run_id`. `trace_id`/`span_id` are random W3C-compatible hex values. There is no HTTP service in this repository, so no `/v1/...` routes exist.

## Operations

All operations are deterministic arithmetic over the payload and report `execution_mode: "deterministic_offline"`. `requires_approval` is false for all of them. `schema` is `{"features": [...], "label": str, "group": str | null, "allow_extra_columns": bool}`; `config` accepts `model_name`, `test_fraction` (0.1-0.5), `seed`, `ridge`, `max_test_mae`, `max_feature_psi`, `min_test_rows`, `psi_bins`. Unknown fields are rejected.

| Operation | Input | `data` | Provenance |
| --- | --- | --- | --- |
| `validate_dataset` | `records`, `schema`, optional `require_label` | validation report (`valid`, counts, `issues` with row/column, `missing_values`, `dataset_sha256`) | `dataset` hash when valid |
| `train` (side effects) | `records`, `schema`, `output_dir`, optional `config`, `code_version` | `model_id`, `artifact_path`, `record_path`, `created`, `verified`, `run_record` | dataset, artifact (with `file:` URI), record |
| `evaluate_candidate` | `records`, `schema`, optional `config`, `code_version` | `candidate` (see [CANDIDATE_EVALUATION.md](CANDIDATE_EVALUATION.md)), `run_record` | dataset, artifact, record |
| `model_card` | `run_record` | `model_id`, `record_sha256`, `markdown` | record |
| `evaluate_drift` | `schema`, `reference_records`, `current_records`, optional `bins`, `max_psi` | per-feature `psi`, bin counts, `exceeds_threshold`, `drift_detected`, both sample hashes | both datasets |

`train` writes `model.json` and `run_record.json` with exclusive-create semantics. Re-running with identical outputs is idempotent (`created: false`); different existing content returns `ARTIFACT_CONFLICT` and leaves the files untouched. After writing, both files are re-read and their SHA-256 compared with the in-memory bytes (`verified`). ### Workspace confinement for `train`

`train` writes only under a workspace root: the directory named by the `MLOPS_PIPELINE_WORKSPACE` environment variable, or the current working directory when it is unset. The root must exist. A relative `output_dir` is interpreted relative to the root. The target is resolved (symlinks, Windows junctions and `..` removed) and must be the root or a descendant of it; otherwise the envelope has `status: "blocked"`, `error.code: "PATH_OUTSIDE_WORKSPACE"`, `policy: {"decision": "deny", ...}`, `data: null`, and nothing is created or written. The check runs before any training work and again after the output directory is created. The resolved path is not echoed back in the error.

Residual risk: a process that can modify the workspace concurrently could still swap a directory for a link between the final check and the file writes; the confinement protects against payload-chosen paths, not against a hostile local process.

`evaluate_drift` validates both samples with `require_label=False`, so current production samples do not need labels.

## Error codes

| Code | When |
| --- | --- |
| `UNKNOWN_OPERATION` | operation name not in `capabilities()` |
| `PATH_OUTSIDE_WORKSPACE` | `train` `output_dir` resolves outside the workspace root; `status` is `blocked` |
| `WORKSPACE_UNAVAILABLE` | `MLOPS_PIPELINE_WORKSPACE` (or the CWD) is not an existing directory |
| `INVALID_INPUT` | payload not an object, missing/unknown fields, bad schema/config/code_version, invalid or tampered `run_record`, invalid ids |
| `DATASET_INVALID` | records fail validation; `error.details.report` holds the validation report |
| `TRAINING_FAILED` | fitting or evaluation fails (for example a singular system with `ridge = 0`) |
| `ARTIFACT_CONFLICT` | `train` output files exist with different content |
| `WRITE_FAILED` | filesystem error while writing (`retryable: true`) |
| `INTEGRITY_CHECK_FAILED` | written files do not match the in-memory bytes |
| `INTERNAL_ERROR` | unexpected exception; only the exception type name is returned |
