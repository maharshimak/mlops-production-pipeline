from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from uuid import uuid4

TERMINAL_STATUSES = {"succeeded", "rejected", "failed", "cancelled"}


@dataclass(frozen=True, slots=True)
class TrackedRun:
    run_id: str
    name: str
    status: str
    dataset_fingerprint: str
    config_fingerprint: str
    params: dict[str, object]
    metrics: dict[str, float]
    artifacts: dict[str, str]


def _strict_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class SQLiteRunLedger:
    """Small local experiment tracker for reproducible pipeline runs.

    Run identity is immutable. Parameters, metrics and artifact references may
    only be written while a run is active, and terminal state transitions are
    one-way. This keeps the local ledger useful as audit evidence without
    pretending to be a distributed experiment-tracking service.
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
        values = (name, dataset_fingerprint, config_fingerprint)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("name and fingerprints are required.")
        rendered_params = _strict_json(params or {})
        run_id = str(uuid4())
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
                rendered_params,
            ),
        )
        self.connection.commit()
        return run_id

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric name is required.")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError("metric value must be finite numeric data.")
        run = self.get_run(run_id)
        metrics = dict(run.metrics)
        metrics[name] = float(value)
        self._update_json(run_id, "metrics_json", metrics)

    def log_artifact(self, run_id: str, name: str, uri: str) -> None:
        if (
            not isinstance(name, str)
            or not isinstance(uri, str)
            or not name.strip()
            or not uri.strip()
        ):
            raise ValueError("artifact name and URI are required.")
        run = self.get_run(run_id)
        artifacts = dict(run.artifacts)
        artifacts[name] = uri
        self._update_json(run_id, "artifacts_json", artifacts)

    def finish_run(self, run_id: str, *, status: str = "succeeded") -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("invalid run status.")
        cursor = self.connection.execute(
            """
            UPDATE runs
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ? AND status = 'running'
            """,
            (status, run_id),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            run = self.get_run(run_id)
            raise ValueError(
                f"Run {run_id} is already terminal with status {run.status}."
            )
        self.connection.commit()

    def _update_json(self, run_id: str, column: str, value: dict[str, object]) -> None:
        if column not in {"metrics_json", "artifacts_json"}:
            raise ValueError("unsupported ledger column.")
        payload = _strict_json(value)
        cursor = self.connection.execute(
            f"""
            UPDATE runs
            SET {column} = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ? AND status = 'running'
            """,
            (payload, run_id),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            run = self.get_run(run_id)
            raise ValueError(
                f"Run {run_id} is already terminal with status {run.status}."
            )
        self.connection.commit()

    @staticmethod
    def _render_run(row: sqlite3.Row) -> TrackedRun:
        return TrackedRun(
            run_id=row["run_id"],
            name=row["name"],
            status=row["status"],
            dataset_fingerprint=row["dataset_fingerprint"],
            config_fingerprint=row["config_fingerprint"],
            params=json.loads(row["params_json"]),
            metrics={
                key: float(value)
                for key, value in json.loads(row["metrics_json"]).items()
            },
            artifacts=json.loads(row["artifacts_json"]),
        )

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
        return self._render_run(row)

    def list_runs(
        self,
        *,
        name: str | None = None,
        limit: int = 100,
    ) -> list[TrackedRun]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000.")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be non-empty when provided.")

        if name is None:
            rows = self.connection.execute(
                """
                SELECT run_id, name, status, dataset_fingerprint, config_fingerprint,
                       params_json, metrics_json, artifacts_json
                FROM runs
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT run_id, name, status, dataset_fingerprint, config_fingerprint,
                       params_json, metrics_json, artifacts_json
                FROM runs
                WHERE name = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (name, limit),
            ).fetchall()
        return [self._render_run(row) for row in rows]

    def close(self) -> None:
        self.connection.close()
