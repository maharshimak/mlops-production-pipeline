# MLOps Production Pipeline

Dependency-light ML lifecycle building blocks: deterministic linear regression, held-out MAE, quality gates and PSI drift.

## Implemented now

- One-feature ordinary least-squares training with finite-data and variance checks.
- Prediction, mean absolute error and fail-closed quality threshold checks.
- Versioned coefficient dataclass and PSI over normalized non-negative bin counts.

## Scope and limitations

Despite the repository name, this is a compact educational library, not a production deployment system. Training is univariate regression. Model metadata contains version and coefficients only. There is no dataset splitting service, model registry, artifact storage, online serving, monitoring collector or automated deployment. PSI needs consistent caller-defined bins.

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

Dataset fingerprints; serialized artifacts and provenance; explicit train/validation/test splitting; reproducible pipeline command; deployment and monitoring adapters. These are planned work, not current capabilities.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). The standalone CI workflow runs after migration; while nested in the profile repository, the parent CI validates this project.

## License and provenance

[MIT](LICENSE), copyright 2026 Maharshi Patel. This public portfolio implementation is independent of employer systems and contains no confidential employer code or data. Examples and test fixtures are synthetic.
