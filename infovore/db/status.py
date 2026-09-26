import sqlite3
from dataclasses import dataclass
from datetime import datetime

from infovore.db.codec import from_db_time


@dataclass(frozen=True)
class StatusReport:
    channels: int
    messages: int
    deleted_messages: int
    exchanges_by_status: dict[str, int]
    claims_by_novelty: dict[str, int]
    retracted_claims: int
    runs_by_outcome: dict[str, int]
    last_extraction_at: datetime | None
    last_probe_at: datetime | None
    live_prompt_version: str | None


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def _counts(conn: sqlite3.Connection, sql: str) -> dict[str, int]:
    return {str(row[0]): int(row[1]) for row in conn.execute(sql)}


def _latest_time(conn: sqlite3.Connection, sql: str) -> datetime | None:
    value = conn.execute(sql).fetchone()[0]
    return from_db_time(value) if value is not None else None


def _live_prompt_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT version FROM prompt_versions WHERE promoted_at IS NOT NULL"
        " ORDER BY promoted_at DESC LIMIT 1"
    ).fetchone()
    return str(row[0]) if row is not None else None


def collect_status(conn: sqlite3.Connection) -> StatusReport:
    return StatusReport(
        channels=_count(conn, "SELECT COUNT(*) FROM channels"),
        messages=_count(conn, "SELECT COUNT(*) FROM messages"),
        deleted_messages=_count(conn, "SELECT COUNT(*) FROM messages WHERE deleted_at IS NOT NULL"),
        exchanges_by_status=_counts(
            conn, "SELECT extraction_status, COUNT(*) FROM exchanges GROUP BY 1 ORDER BY 1"
        ),
        claims_by_novelty=_counts(
            conn,
            "SELECT novelty, COUNT(*) FROM claims WHERE retracted_at IS NULL GROUP BY 1 ORDER BY 1",
        ),
        retracted_claims=_count(conn, "SELECT COUNT(*) FROM claims WHERE retracted_at IS NOT NULL"),
        runs_by_outcome=_counts(
            conn, "SELECT outcome, COUNT(*) FROM extraction_runs GROUP BY 1 ORDER BY 1"
        ),
        last_extraction_at=_latest_time(conn, "SELECT MAX(started_at) FROM extraction_runs"),
        last_probe_at=_latest_time(conn, "SELECT MAX(probed_at) FROM claims"),
        live_prompt_version=_live_prompt_version(conn),
    )
