from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TrackedRun:
    run_id: str
    name: str
    status: str
    dataset_fingerprint: str
    config_fingerprint: str
    params: dict[str, str]
    metrics: dict[str, float]
    artifacts: dict[str, str]


class SQLiteRunLedger:
    """Small local experiment tracker for reproducible pipeline runs.

    It records immutable run identity plus parameters, metrics and artifact
    references. This gives the pipeline an auditable lifecycle without forcing a
    heavyweight external tracking server; production deployments can later
    adapt the same concepts to MLflow or another backend.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                params_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_runs_name_created
                ON runs(name, created_at);
            """
        )
        self.connection.commit()

    def start_run(
        self,
        name: str,
        *,
        dataset_fingerprint: str,
        config_fingerprint: str,
        params: dict[str, object] | None = None,
    ) -> str:
        if not name.strip() or not dataset_fingerprint.strip() or not config_fingerprint.strip():
            raise ValueError("name and fingerprints are required.")
        run_id = str(uuid4())
        rendered_params = {
            str(key): json.dumps(value, sort_keys=True, separators=(",", ":"))
            for key, value in (params or {}).items()
        }
        self.connection.execute(
            """
            INSERT INTO runs(
                run_id, name, status, dataset_fingerprint, config_fingerprint,
                params_json, metrics_json, artifacts_json
            )
            VALUES (?, ?, 'running', ?, ?, ?, '{}', '{}')
            """,
            (
                run_id,
                name,
                dataset_fingerprint,
                config_fingerprint,
                json.dumps(rendered_params, sort_keys=True, separators=(",", ":")),
            ),
        )
        self.connection.commit()
        return run_id

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("metric value must be numeric.")
        run = self.get_run(run_id)
        metrics = dict(run.metrics)
        metrics[name] = float(value)
        self._update_json(run_id, "metrics_json", metrics)

    def log_artifact(self, run_id: str, name: str, uri: str) -> None:
        if not name.strip() or not uri.strip():
            raise ValueError("artifact name and URI are required.")
        run = self.get_run(run_id)
        artifacts = dict(run.artifacts)
        artifacts[name] = uri
        self._update_json(run_id, "artifacts_json", artifacts)

    def finish_run(self, run_id: str, *, status: str = "succeeded") -> None:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("invalid run status.")
        cursor = self.connection.execute(
            """
            UPDATE runs SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (status, run_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown run: {run_id}")
        self.connection.commit()

    def _update_json(self, run_id: str, column: str, value: dict[str, object]) -> None:
        if column not in {"metrics_json", "artifacts_json"}:
            raise ValueError("unsupported ledger column.")
        cursor = self.connection.execute(
            f"UPDATE runs SET {column} = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
            (json.dumps(value, sort_keys=True, separators=(",", ":")), run_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown run: {run_id}")
        self.connection.commit()

    def get_run(self, run_id: str) -> TrackedRun:
        row = self.connection.execute(
            """
            SELECT run_id, name, status, dataset_fingerprint, config_fingerprint,
                   params_json, metrics_json, artifacts_json
            FROM runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        return TrackedRun(
            run_id=row["run_id"],
            name=row["name"],
            status=row["status"],
            dataset_fingerprint=row["dataset_fingerprint"],
            config_fingerprint=row["config_fingerprint"],
            params=json.loads(row["params_json"]),
            metrics={key: float(value) for key, value in json.loads(row["metrics_json"]).items()},
            artifacts=json.loads(row["artifacts_json"]),
        )

    def close(self) -> None:
        self.connection.close()
