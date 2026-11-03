# MLOps Production Pipeline

**A MAK'MA Studio Product · MAK'MA Labs**

[Live Product Demo](https://maharshimak.github.io/makma-ai-os/projects/mlops-production-pipeline/) · [MAK'MA Labs](https://maharshimak.github.io/makma-ai-os/projects/)

Dependency-light ML lifecycle building blocks: deterministic single- and multi-feature linear regression, held-out MAE, quality gates, reproducibility fingerprints and PSI drift.


## Product contract — engineering upgrade

**Problem and audience:** A compact reproducible train/evaluate/integrity/deployment-gate workflow for ML engineering education and regression checks.

**Live tool:** https://maharshimak.github.io/makma-ai-os/projects/mlops-production-pipeline/

**Implemented browser workflow:** Editable training/held-out pairs, OLS coefficients, prediction errors/MAE, normalized PSI, exact UTF-8 artifact hashing, tamper controls and a gate combining sample count, error, drift and integrity. Edits invalidate verification until checked again.

**Backend and parity contract:** Python deployment_gate now accepts integrity_valid; false blocks deployment even when quality passes. PSI rejects overflowing totals and training rejects non-finite coefficients. Shared OLS/PSI and exact payload digest fixtures protect browser/Python equivalence. Python model_fingerprint serializes Python floats; byte hashes only match when exact serialization matches.

**Architecture:** `makma-ai-os/demo` is the shared web product source and Pages deployment. This repository owns its Python domain package. The central `tests/e2e` suite exercises all nine products; `tests/fixtures/python-parity.json` plus `scripts/generate_parity.py` guard shared mathematical contracts. Backend revisions used for regeneration are pinned in the central `backend-lock.json`.

**Safety and limitations:** Linear models only (single-feature OLS and multi-feature standardized ridge regression); no automated deployment, experiment tracking server or signed artifact registry. A digest checks bytes against a reference, not the trustworthiness of that reference. Inputs are validated, rendered user values are escaped, and deterministic results are not presented as model inference.

**Verification:** Run `python -m ruff check .` and `python -m pytest -q`. `tests/test_engineering_upgrade.py` protects the new rejection/correctness paths. Central web checks: `npm ci`, `npm test`, `npm run build`, `npx playwright install --with-deps chromium`, `npm run test:e2e`. CI gates publishing on browser interactions and validates all public URLs after deployment.

**Durable experiment tracking:** `SQLiteRunLedger` records run identity, dataset/config fingerprints, parameters, MAE/PSI metrics, artifact references and terminal status. `run_regression_pipeline` can write directly to this ledger so tracking is tied to the actual training run rather than a separate demo.

**Highest-value next work:** Trusted signed manifests and CI deployment integrations.

**Provenance:** Independent MAK’MA Studio engineering implementation; examples are synthetic and no employer code or data is included. Existing MIT license applies.


## Implemented now

- One-feature ordinary least-squares training with finite-data and variance checks.
- Prediction, mean absolute error and fail-closed quality threshold checks.
- Versioned coefficient dataclass and PSI over normalized non-negative bin counts.
- Deterministic seeded train/evaluation splitting with held-out MAE and measured train/eval PSI.
- Dataset and configuration SHA-256 fingerprints, reproducible model artifact payload and deployment-gate result.
- Record-oriented dataset validation (schema, types, missing values, label presence, groups) and content-addressed dataset versions.
- Group-aware deterministic train/test splits, a reproducible run record, model cards generated from that record, per-feature PSI drift and a control-plane candidate-evaluation export.
- `makma.v1` execution envelope with five operations (see below).

## Scope and limitations

This remains a compact reproducible ML pipeline library rather than a production serving/deployment platform. The implemented training model is still univariate OLS, but `run_regression_pipeline` now performs a deterministic seeded split, held-out evaluation, train/eval drift measurement, dataset/config fingerprinting, reproducible artifact serialization and a quality/integrity deployment decision. There is no external dataset service, model registry, artifact store, online serving, monitoring collector or automated infrastructure deployment.

## Multi-feature pipeline

In addition to the original dependency-light single-feature baseline, `run_tabular_regression_pipeline` now supports multiple numeric features with feature normalization, ridge-stabilized linear regression, deterministic train/eval splits, held-out MAE and dataset/config/artifact fingerprints. The simpler baseline remains available for parity and educational inspection.

## Validated experiments

`mlops_pipeline.experiment.run_experiment(records, schema, config)` is the record-oriented workflow:

1. **Validation** (`datasets.validate_dataset`): every row must be an object with finite numeric features and label (`bool`, strings, `NaN`, `inf` and integers too large for a float are rejected), a non-blank string/integer group value when a group column is declared, at least two groups, and no undeclared columns unless `allow_extra_columns` is set. Errors carry row and column; constant features/labels and duplicate rows are warnings. The report lists at most 50 issues but counts all of them.
2. **Dataset version**: SHA-256 of canonical JSON (sorted keys, `(",", ":")` separators, UTF-8, no NaN) over the schema column names and the projected rows. Numbers are canonicalised to float (`1` and `1.0` hash equally, `-0.0` becomes `0.0`); row order is significant; extra columns are excluded.
3. **Leakage-safe split** (`splits.group_train_test_split`): whole groups go to train or test. Groups are ranked by `sha256({"seed", "group": [type, value]})`, so the assignment depends only on the seed and the set of groups, not on row order or Python hash randomisation. Without a group column every row is its own group.
4. **Training and evaluation**: the existing standardized ridge baseline (`train_tabular`), train/test MAE, test RMSE and maximum absolute error, the number of test rows whose feature vector also occurs in training, and per-feature PSI (bins fitted on train only).
5. **Gate and run record**: the gate checks minimum test rows, test MAE and maximum feature PSI. The run record binds dataset SHA-256, config SHA-256, code version, split, metrics, drift, gate and model artifact SHA-256, and carries its own `record_sha256`. It has no timestamps, so identical inputs produce byte-identical records.

`code_version` is the caller's value if given; otherwise the git `HEAD` SHA when the package is imported from its own checkout (with `code_tree_dirty`), and the literal `"unknown"` otherwise.

`model_card.model_card_markdown(record)` renders a Markdown card only from a hash-verified record and states which sections are not recorded. `experiment.candidate_evaluation_export(record)` produces the control-plane export documented in [docs/CANDIDATE_EVALUATION.md](docs/CANDIDATE_EVALUATION.md). The record hash detects modification; it is not a signature.

## `makma.v1` operations

`mlops_pipeline.contracts.run_operation(operation, payload, *, run_id=None, correlation_id=None)` returns the shared MAK'MA envelope (`contracts/makma.v1.schema.json`); `capabilities()` lists the operations with JSON input schemas. All operations are deterministic arithmetic and report `execution_mode: "deterministic_offline"`. Details: [docs/MAKMA_CONTRACT.md](docs/MAKMA_CONTRACT.md).

| Operation | Side effects | Returns in `data` |
| --- | --- | --- |
| `validate_dataset` | none | validation report including `dataset_sha256` |
| `train` | writes `model.json` and `run_record.json` to `output_dir` (never overwrites different content) | paths, `verified`, `created`, run record |
| `evaluate_candidate` | none | control-plane `candidate` export and run record |
| `model_card` | none | Markdown model card for a verified run record |
| `evaluate_drift` | none | per-feature PSI between reference and (optionally unlabelled) current samples |

## Installation and development

Requires Python 3.12 or newer. Run from this project directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python -m pip wheel --no-deps . -w dist
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## Library usage

```python
from mlops_pipeline.core import train, mae, quality_gate, psi
model = train([1, 2, 3], [2, 4, 6], version="demo-1")
error = mae(model, [4, 5], [8, 10])  # separate evaluation observations
print(model, error, quality_gate(error, maximum=0.01))
print(psi([40, 60], [45, 55]))
```

## Configuration

Configuration is supplied through Python function/constructor arguments. No credentials or environment file are needed for the offline example.

## Container

```bash
docker build -t mlops-production-pipeline .
docker run --rm mlops-production-pipeline
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/mlops_pipeline/` | Implementation |
| `tests/` | Offline unit and regression tests |
| `docs/DESIGN.md` | Architecture and trust boundaries |
| `docs/CANDIDATE_EVALUATION.md` | Run record and control-plane export format |
| `docs/MAKMA_CONTRACT.md` | `makma.v1` operations and error codes |
| `contracts/` | JSON Schemas for the `makma.v1` envelope and the candidate export |
| `.github/workflows/ci.yml` | Install, lint, tests, wheel and container build |
| `pyproject.toml` | Dependencies and package configuration |

## Next engineering work

Classification estimators; file-backed dataset loaders; a separate validation split for model selection; CLI pipeline execution; sending candidate exports to a control plane (only the export format exists here); signed manifests; deployment and monitoring adapters. These are planned work, not current capabilities.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). CI runs on every push and pull request through `.github/workflows/ci.yml`.

## License and provenance

[MIT](LICENSE), copyright 2026 Maharshi Patel. This public portfolio implementation is independent of employer systems and contains no confidential employer code or data. Examples and test fixtures are synthetic.
