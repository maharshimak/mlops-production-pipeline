"""``train`` may only write under the workspace root (MLOPS_PIPELINE_WORKSPACE or CWD)."""

import json
import os
import sys
from pathlib import Path

import pytest
from conftest import SCHEMA_DICT, make_records
from jsonschema import Draft202012Validator

from mlops_pipeline.contracts import WORKSPACE_ENV, run_operation

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = Draft202012Validator(
    json.loads((ROOT / "contracts" / "makma.v1.schema.json").read_text(encoding="utf-8"))
)


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv(WORKSPACE_ENV, str(root))
    return root


def train(output_dir: str) -> dict:
    envelope = run_operation(
        "train",
        {"records": make_records(), "schema": dict(SCHEMA_DICT), "output_dir": output_dir,
         "code_version": "ws-test"},
    )
    ENVELOPE.validate(envelope)
    return envelope


def files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file())


def assert_blocked(envelope: dict) -> None:
    assert envelope["status"] == "blocked"
    assert envelope["error"]["code"] == "PATH_OUTSIDE_WORKSPACE"
    assert envelope["data"] is None
    assert envelope["provenance"] == []
    assert envelope["policy"]["decision"] == "deny"


def test_relative_path_inside_workspace_is_written_there(workspace) -> None:
    envelope = train("runs/candidate-1")
    assert envelope["status"] == "ok"
    artifact = Path(envelope["data"]["artifact_path"])
    assert artifact.parent == (workspace / "runs" / "candidate-1").resolve()
    assert artifact.is_file()


def test_absolute_path_inside_workspace_is_allowed(workspace) -> None:
    assert train(str(workspace / "abs"))["status"] == "ok"


@pytest.mark.parametrize("relative", ["../escape", "runs/../../escape", "..", "a/../../../x"])
def test_relative_traversal_is_blocked_and_nothing_written(workspace, relative) -> None:
    assert_blocked(train(relative))
    assert files_under(workspace.parent) == []


def test_absolute_path_outside_workspace_is_blocked(workspace, tmp_path) -> None:
    outside = tmp_path / "outside"
    assert_blocked(train(str(outside)))
    assert not outside.exists()
    assert files_under(tmp_path) == []


def test_sibling_with_shared_prefix_is_blocked(workspace) -> None:
    # "workspace-evil" starts with the root's name but is not inside it.
    sibling = workspace.parent / f"{workspace.name}-evil"
    assert_blocked(train(str(sibling)))
    assert not sibling.exists()


def test_symlink_escape_is_blocked(workspace, tmp_path) -> None:
    outside = tmp_path / "outside-target"
    outside.mkdir()
    link = workspace / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation not permitted on this platform/user: {error}")
    assert_blocked(train("link/run"))
    assert_blocked(train(str(link)))
    assert files_under(outside) == []


def test_default_workspace_is_current_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(WORKSPACE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert train("inside")["status"] == "ok"
    assert (tmp_path / "inside" / "model.json").is_file()
    assert_blocked(train(str(tmp_path.parent / "elsewhere")))


def test_missing_workspace_root_is_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(WORKSPACE_ENV, str(tmp_path / "does-not-exist"))
    envelope = train("x")
    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "WORKSPACE_UNAVAILABLE"
    assert not (tmp_path / "does-not-exist").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions exist only on Windows")
def test_windows_junction_escape_is_blocked(workspace, tmp_path) -> None:
    # Junctions need no special privilege, so this covers the link-escape case on
    # Windows hosts where the symlink test above is skipped.
    import _winapi

    outside = tmp_path / "junction-target"
    outside.mkdir()
    link = workspace / "junction"
    _winapi.CreateJunction(str(outside), str(link))
    assert_blocked(train("junction/run"))
    assert files_under(outside) == []
