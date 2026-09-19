from pathlib import Path

import pytest

from mlops_pipeline.integrity import build_integrity_manifest, verify_integrity_manifest


def test_integrity_manifest_detects_modified_artifact(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    metrics = tmp_path / "metrics.json"
    model.write_bytes(b"model-v1")
    metrics.write_text('{"mae": 0.1}', encoding="utf-8")

    manifest = build_integrity_manifest(tmp_path, ["model.bin", "metrics.json"])
    assert verify_integrity_manifest(tmp_path, manifest).valid

    model.write_bytes(b"tampered")
    result = verify_integrity_manifest(tmp_path, manifest)

    assert not result.valid
    assert result.modified == ("model.bin",)
    assert result.missing == ()


def test_integrity_manifest_detects_missing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    manifest = build_integrity_manifest(tmp_path, ["model.bin"])
    artifact.unlink()

    result = verify_integrity_manifest(tmp_path, manifest)

    assert result.missing == ("model.bin",)


def test_integrity_manifest_blocks_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes root"):
        build_integrity_manifest(tmp_path, ["../secret.txt"])
