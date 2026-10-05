# Design and operating boundaries

This repository provides deterministic, dependency-light building blocks for a reproducible
ML training and release workflow. It is a library, not a hosted model-serving platform.

## Pipeline boundary

`run_regression_pipeline` performs a seeded train/evaluation split, fits a univariate OLS
model, measures held-out MAE, computes train/evaluation PSI with training-fitted bins, runs
deterministic k-fold cross-validation, serializes a reproducible model artifact and applies
a deployment gate.

The repository also includes a separate multi-feature regression path with normalized
features and ridge-stabilized fitting. These estimators are intentionally compact and are
not replacements for a general-purpose ML framework.

Configuration and data fingerprints are calculated before model execution. When a
`SQLiteRunLedger` is supplied, the run identity therefore exists before fitting begins,
so runtime training/evaluation failures can be recorded instead of disappearing from the
experiment history.

## Experiment lifecycle

Ledger runs begin in `running` and transition once to one terminal state:

- `succeeded`: execution completed and the deployment gate allowed the artifact;
- `rejected`: execution completed but release evidence failed the deployment gate;
- `failed`: the pipeline raised after the tracked run identity was created;
- `cancelled`: an explicit external cancellation state.

Parameters, metrics and artifact references can only be mutated while a run is active.
Terminal runs cannot be reopened or edited. Metric values must be finite and stored JSON is
serialized with non-standard NaN/Infinity values disabled. The ledger exposes bounded
history reads for audit/debugging.

SQLite provides durable local experiment history, not distributed orchestration or a
multi-writer tracking service.

## Artifact and release evidence

The release gate consumes a validated `RunManifest`. It fails closed when caller-supplied
manifest fields, thresholds or integrity controls are malformed. Current evidence includes:

- model version and deterministic coefficient fingerprint;
- train/evaluation row counts;
- held-out MAE;
- PSI drift;
- cross-validation mean/worst MAE;
- byte-level artifact integrity verification;
- dataset and configuration fingerprints.

The integrity digest proves that checked bytes match the generated reference. It does not
establish who created the reference, sign the artifact or prove that the underlying data
is trustworthy.

## Interfaces and trust

Implementation lives in `src/mlops_pipeline/`. The public interface is a Python library;
there is no network service layer in this repository.

Inputs are caller supplied. There is no external dataset service, feature store, model
registry, artifact registry, online serving system, secrets manager, monitoring collector
or automated infrastructure deployment. A passing deployment decision means configured
deterministic checks passed; it is not independent certification of model quality.

## Validation

CI runs Ruff, pytest, wheel construction and a container build. Regression tests cover
training, held-out gates, PSI, cross-validation, artifact integrity, run-ledger lifecycle,
terminal immutability and failure/rejection tracking.

These tests verify implementation behavior only. They do not certify statistical
appropriateness for arbitrary datasets, fairness, security of downstream infrastructure,
or production SLOs.

## Planned evolution

Highest-value remaining work includes signed artifact manifests, explicit dataset version
adapters, separate train/validation/test policies, model-registry/control-plane handoff,
idempotent deployment integrations, production monitoring ingestion and richer estimators.
