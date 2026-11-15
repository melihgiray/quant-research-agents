"""Parameterized SQL and idempotent experiment writes, SQLite or PostgreSQL."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class ExperimentStore:
    def __init__(self, workspace: Path, database_url: str | None = None):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        if database_url and not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must use postgresql:// or postgres://; omit for SQLite")
        self.database_url = database_url
        with self.connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                result_json TEXT NOT NULL
            )""")

    @contextmanager
    def connect(self):
        if self.database_url:
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=5)
        else:
            conn = sqlite3.connect(self.workspace / "experiments.sqlite3", timeout=10)
            conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.database_url else statement

    def save(self, result: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                self._sql("""INSERT INTO experiments
                (experiment_id, created_at, result_json) VALUES (?, ?, ?)
                ON CONFLICT (experiment_id) DO NOTHING"""),
                (
                    result["experiment_id"],
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(result, allow_nan=False),
                ),
            )

    def get(self, experiment_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                self._sql("SELECT result_json FROM experiments WHERE experiment_id = ?"),
                (experiment_id,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def recent(self, limit: int = 5) -> list[dict]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        with self.connect() as conn:
            rows = conn.execute(
                self._sql(
                    "SELECT experiment_id, created_at, result_json FROM experiments "
                    "ORDER BY created_at DESC, experiment_id DESC LIMIT ?"
                ),
                (limit,),
            ).fetchall()
        return [
            {
                "experiment_id": r["experiment_id"],
                "created_at": r["created_at"],
                "manifest": json.loads(r["result_json"])["manifest"],
            }
            for r in rows
        ]
