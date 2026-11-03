# Run records and the candidate-evaluation export

`mlops_pipeline.experiment.run_experiment` produces two byte-stable outputs: a model artifact and a run record. `candidate_evaluation_export(record)` projects the record into the smaller document an MLOps control plane consumes. All hashes are SHA-256 over canonical JSON: sorted keys, `(",", ":")` separators, UTF-8 (`ensure_ascii=False`), `NaN`/`Infinity` rejected.

## Model artifact (`model.json`, format `mlops-model.v1`)

Canonical JSON with `model_name`, `model_type` (`standardized_ridge_linear_regression`), `features`, `label`, `weights`, `intercept`, `feature_means`, `feature_scales`, `dataset_sha256` and `config_sha256`. `artifact_sha256` is the SHA-256 of these exact bytes. It does not contain the code version, so the same data and config produce the same artifact on any commit that computes the same coefficients.

## Run record (`run_record.json`, format `mlops-run-record.v1`)

| Field | Meaning |
| --- | --- |
| `model_id` | `<model_name>-<first 16 hex of artifact_sha256>` |
| `dataset` | `sha256`, `row_count`, `features`, `label`, `group` |
| `config`, `config_sha256` | normalized experiment config and its canonical hash |
| `code_version`, `code_version_source`, `code_tree_dirty` | caller value (`caller`), git `HEAD` of this package's own checkout (`git`, with dirty flag) or `"unknown"` |
| `split` | strategy (`group`/`row`), seed, fraction, row and group counts, hashes of the train and test index lists |
| `metrics` | `train_mae`, `test_mae`, `test_rmse`, `test_max_abs_error`, `test_rows_with_train_feature_match`, `max_feature_psi` |
| `drift` | per-feature PSI, train as reference and test as current |
| `gate` | `allowed` and human-readable `reasons` |
| `artifact` | `sha256`, `size_bytes`, `media_type`, `filename` |
| `record_sha256` | hash of the canonical record without this field |

`verify_record` checks the format and `record_sha256`. That detects modification but is not an authenticity proof: anyone who edits a record can recompute the hash. Signing is not implemented.

## Candidate-evaluation export (format `mlops-candidate-evaluation.v1`)

JSON Schema: [`contracts/candidate-evaluation.v1.schema.json`](../contracts/candidate-evaluation.v1.schema.json).

| Field | Type | Source in the run record |
| --- | --- | --- |
| `format` | `"mlops-candidate-evaluation.v1"` | constant |
| `model_id` | string | `model_id` |
| `dataset_sha256` | 64 hex | `dataset.sha256` |
| `config_sha256` | 64 hex | `config_sha256` |
| `code_version` | string, 40/64-hex git SHA, caller-supplied value or `"unknown"` | `code_version` |
| `metrics` | object of non-negative numbers | `metrics` |
| `artifact_sha256` | 64 hex | `artifact.sha256` |
| `gate` | `{"allowed": bool, "reasons": [string]}`; `allowed` is true exactly when `reasons` is empty | `gate` |
| `run_record_sha256` | 64 hex | `record_sha256` |

The export is produced only from a record whose `record_sha256` verifies. This repository produces the document; it does not send it anywhere. Consumers should treat `gate.allowed` as this pipeline's quality opinion, not as a deployment approval.

Obtain it in-process with `run_operation("evaluate_candidate", {"records": [...], "schema": {...}})`; the export is in `data.candidate` and the full record in `data.run_record`.
