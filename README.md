# MLOps Production Pipeline

**A MAK'MA Studio Product · MAK'MA Labs**

[Live Product Demo](https://maharshimak.github.io/makma-ai-os/projects/mlops-production-pipeline/) · [MAK'MA Labs](https://maharshimak.github.io/makma-ai-os/projects/)

Dependency-light ML lifecycle building blocks: deterministic linear regression, held-out MAE, quality gates and PSI drift.


## Product contract — engineering upgrade

**Problem and audience:** A compact reproducible train/evaluate/integrity/deployment-gate workflow for ML engineering education and regression checks.

**Live tool:** https://maharshimak.github.io/makma-ai-os/projects/mlops-production-pipeline/

**Implemented browser workflow:** Editable training/held-out pairs, OLS coefficients, prediction errors/MAE, normalized PSI, exact UTF-8 artifact hashing, tamper controls and a gate combining sample count, error, drift and integrity. Edits invalidate verification until checked again.

**Backend and parity contract:** Python deployment_gate now accepts integrity_valid; false blocks deployment even when quality passes. PSI rejects overflowing totals and training rejects non-finite coefficients. Shared OLS/PSI and exact payload digest fixtures protect browser/Python equivalence. Python model_fingerprint serializes Python floats; byte hashes only match when exact serialization matches.

**Architecture:** `makma-ai-os/demo` is the shared web product source and Pages deployment. This repository owns its Python domain package. The central `tests/e2e` suite exercises all nine products; `tests/fixtures/python-parity.json` plus `scripts/generate_parity.py` guard shared mathematical contracts. Backend revisions used for regeneration are pinned in the central `backend-lock.json`.

**Safety and limitations:** One-feature OLS only; no automated deployment, experiment tracking server or signed artifact registry. A digest checks bytes against a reference, not the trustworthiness of that reference. Inputs are validated, rendered user values are escaped, and deterministic results are not presented as model inference.

**Verification:** Run `python -m ruff check .` and `python -m pytest -q`. `tests/test_engineering_upgrade.py` protects the new rejection/correctness paths. Central web checks: `npm ci`, `npm test`, `npm run build`, `npx playwright install --with-deps chromium`, `npm run test:e2e`. CI gates publishing on browser interactions and validates all public URLs after deployment.

**Highest-value next work:** Trusted signed manifests, dataset versioning and CI deployment integrations.

**Provenance:** Independent MAK’MA Studio engineering implementation; examples are synthetic and no employer code or data is included. Existing MIT license applies.


## Implemented now

- One-feature ordinary least-squares training with finite-data and variance checks.
- Prediction, mean absolute error and fail-closed quality threshold checks.
- Versioned coefficient dataclass and PSI over normalized non-negative bin counts.
- Deterministic seeded train/evaluation splitting with held-out MAE and measured train/eval PSI.
- Dataset and configuration SHA-256 fingerprints, reproducible model artifact payload and deployment-gate result.

## Scope and limitations

This remains a compact reproducible ML pipeline library rather than a production serving/deployment platform. The implemented training model is still univariate OLS, but `run_regression_pipeline` now performs a deterministic seeded split, held-out evaluation, train/eval drift measurement, dataset/config fingerprinting, reproducible artifact serialization and a quality/integrity deployment decision. There is no external dataset service, model registry, artifact store, online serving, monitoring collector or automated infrastructure deployment.

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
| `.github/workflows/ci.yml` | Install, lint, tests, wheel and container build |
| `pyproject.toml` | Dependencies and package configuration |

## Next engineering work

Multivariate/classification estimators; file-backed dataset loaders; persisted artifact/manifests; explicit train/validation/test modes; CLI pipeline execution; control-plane registration; deployment and monitoring adapters. These are planned work, not current capabilities.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). CI runs on every push and pull request through `.github/workflows/ci.yml`.

## License and provenance

[MIT](LICENSE), copyright 2026 Maharshi Patel. This public portfolio implementation is independent of employer systems and contains no confidential employer code or data. Examples and test fixtures are synthetic.
