# Production engineering

`mlops_pipeline.artifacts` turns model outputs into deterministic run manifests and explicit deployment decisions.

The manifest carries a model fingerprint, training/evaluation row counts, MAE and PSI drift. The deployment gate checks minimum evaluation evidence plus quality and drift thresholds and returns machine-readable failure reasons.

## Operational practice

- Persist manifests with every candidate artifact.
- Require immutable model/data identifiers in a real registry.
- Keep deployment thresholds in version-controlled environment policy.
- Promote only after training, evaluation, package, container and monitoring checks succeed.
