# Changelog

All notable engineering changes to this repository are documented here.

The format follows the intent of [Keep a Changelog](https://keepachangelog.com/) and uses semantic versioning principles where releases are created.

## [Unreleased]

### Added
- Record-oriented dataset validation and canonical SHA-256 dataset versions (`datasets`).
- Group-aware deterministic train/test splits (`splits`) and reference-fitted per-feature PSI (`drift`).
- Reproducible run records, model artifacts and verified local outputs (`experiment`), model cards rendered from verified records (`model_card`) and the `mlops-candidate-evaluation.v1` export.
- `makma.v1` envelope with `validate_dataset`, `train`, `evaluate_candidate`, `model_card` and `evaluate_drift` operations (`contracts`).

### Fixed
- Dataset fingerprints in `run_tabular_regression_pipeline` and `run_regression_pipeline` now canonicalise numbers to float, so integer and float spellings of the same data share one fingerprint. Float-only inputs keep their previous fingerprints.
- `run_tabular_regression_pipeline` reports its own "at least four training rows" error instead of failing inside training with a generic message.

### Engineering
- Added repository ownership and structured contribution workflows.
- Added Dependabot coverage for Python and GitHub Actions dependencies.
- Added issue and pull-request templates focused on reproducibility, validation, risk, and rollback.
- Added consistent editor configuration and engineering standards.

## Release policy

A release should describe user-visible behavior, compatibility notes, validation evidence, and any migration or rollback requirements. Repository history remains the source of truth for individual commits.
