# Advanced engineering: artifact integrity

`mlops_pipeline.integrity` creates deterministic SHA-256 manifests for model artifacts, metrics or
other release files and verifies them before deployment.

Each manifest entry records relative path, digest and byte size. Verification reports missing and
modified files separately, while path resolution blocks `../` traversal outside the declared
artifact root.

This provides a small supply-chain integrity primitive that can sit between training, registry and
deployment stages without adding an external dependency.
