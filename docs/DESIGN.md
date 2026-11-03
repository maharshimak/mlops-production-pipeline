# Design and operating boundaries

Dependency-light ML lifecycle building blocks: deterministic linear regression, held-out MAE, quality gates and PSI drift.

## Scope

Despite the repository name, this is a compact educational library, not a production deployment system. Training is linear regression (single-feature OLS and multi-feature standardized ridge). The record-oriented experiment workflow adds dataset validation, content-addressed dataset versions, group-aware splits, run records, model cards and a candidate-evaluation export; these are in-process functions and local files, not services. There is no model registry, remote artifact storage, online serving, monitoring collector or automated deployment. `core.psi` needs consistent caller-defined bins; `drift.feature_drift` fits bins on the reference sample.

## Interfaces

Implementation lives in `src/mlops_pipeline/`. Public examples in the README use its Python API. This project is a library, without a service layer. `mlops_pipeline.contracts.run_operation` exposes selected operations through the `makma.v1` envelope for in-process callers.

## Validation

Tests include synthetic regression fixtures. Package and container checks verify installation separately from source-tree imports. Tests do not certify general model quality, clinical correctness or multi-tenant isolation.

## Planned evolution

Signed manifests; explicit train/validation/test splitting for model selection; reproducible pipeline command; deployment and monitoring adapters.
