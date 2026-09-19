from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class IntegrityManifest:
    artifacts: tuple[ArtifactDigest, ...]


@dataclass(frozen=True, slots=True)
class IntegrityVerification:
    valid: bool
    missing: tuple[str, ...]
    modified: tuple[str, ...]


def _safe_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("artifact paths must be non-empty strings")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"artifact path escapes root: {relative_path}")
    return candidate


def _digest(path: Path, relative_path: str) -> ArtifactDigest:
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return ArtifactDigest(
        path=relative_path,
        sha256=hasher.hexdigest(),
        size_bytes=path.stat().st_size,
    )


def build_integrity_manifest(root: str | Path, paths: Iterable[str]) -> IntegrityManifest:
    root_path = Path(root)
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise ValueError("paths cannot be empty")
    artifacts = tuple(
        _digest(_safe_path(root_path, relative_path), relative_path)
        for relative_path in unique_paths
    )
    return IntegrityManifest(artifacts=artifacts)


def verify_integrity_manifest(
    root: str | Path,
    manifest: IntegrityManifest,
) -> IntegrityVerification:
    root_path = Path(root)
    missing: list[str] = []
    modified: list[str] = []

    for expected in manifest.artifacts:
        candidate = _safe_path(root_path, expected.path)
        if not candidate.is_file():
            missing.append(expected.path)
            continue
        actual = _digest(candidate, expected.path)
        if actual.sha256 != expected.sha256 or actual.size_bytes != expected.size_bytes:
            modified.append(expected.path)

    return IntegrityVerification(
        valid=not missing and not modified,
        missing=tuple(missing),
        modified=tuple(modified),
    )
