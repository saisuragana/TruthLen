from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings


def _db_path() -> Path:
    return get_settings().data_dir / "truthlen.sqlite3"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_store() -> None:
    with _connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                external_provider_consent INTEGER NOT NULL DEFAULT 0
            )
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        if "external_provider_consent" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN external_provider_consent INTEGER NOT NULL DEFAULT 0")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at DESC)")


def create_job(
    job_id: str,
    original_name: str,
    media_type: str,
    file_path: str,
    external_provider_consent: bool = False,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        db.execute(
            "INSERT INTO jobs(id, original_name, media_type, file_path, status, created_at, updated_at, external_provider_consent) VALUES(?,?,?,?,?,?,?,?)",
            (job_id, original_name, media_type, file_path, "queued", now, now, int(external_provider_consent)),
        )


def update_job(job_id: str, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        db.execute(
            "UPDATE jobs SET status=?, updated_at=?, result_json=?, error=? WHERE id=?",
            (status, now, json.dumps(result) if result is not None else None, error, job_id),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(limit: int = 12) -> list[dict[str, Any]]:
    with _connect() as db:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["result"] = json.loads(item.pop("result_json")) if item.get("result_json") else None
    return item
