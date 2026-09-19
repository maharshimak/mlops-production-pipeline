# Design and operating boundaries

Dependency-light ML lifecycle building blocks: deterministic linear regression, held-out MAE, quality gates and PSI drift.

## Scope

Despite the repository name, this is a compact educational library, not a production deployment system. Training is univariate regression. Model metadata contains version and coefficients only. There is no dataset splitting service, model registry, artifact storage, online serving, monitoring collector or automated deployment. PSI needs consistent caller-defined bins.

## Interfaces

Implementation lives in `src/mlops_pipeline/`. Public examples in the README use its Python API. This project is a library, without a service layer.

## Validation

Tests include synthetic regression fixtures. Package and container checks verify installation separately from source-tree imports. Tests do not certify general model quality, clinical correctness or multi-tenant isolation.

## Planned evolution

Dataset fingerprints; serialized artifacts and provenance; explicit train/validation/test splitting; reproducible pipeline command; deployment and monitoring adapters.
