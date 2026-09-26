import sqlite3
from dataclasses import dataclass
from datetime import datetime

from infovore.config import DEFAULT_TRIAGE_MIN_P_LORE, DEFAULT_TRIAGE_MIN_SCORE
from infovore.db.codec import from_db_time
from infovore.db.labels import label_counts
from infovore.triage.gate import gate_sql
from infovore.triage.score import TRIAGE_VERSION


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
    triaged_exchanges: int
    above_threshold_exchanges: int
    labels_by_source: dict[str, dict[str, int]]
    labels_effective: dict[str, int]
    latest_model_version: int | None
    latest_model_labels_used: int | None
    p_lore_scored: int
    passing_gate: int


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def _count_params(conn: sqlite3.Connection, sql: str, params: tuple[object, ...]) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


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


def _latest_model(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT version, labels_used FROM triage_model ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None, None
    return int(row["version"]), int(row["labels_used"])


def collect_status(
    conn: sqlite3.Connection,
    triage_min_score: float = DEFAULT_TRIAGE_MIN_SCORE,
    triage_min_p_lore: float = DEFAULT_TRIAGE_MIN_P_LORE,
) -> StatusReport:
    counts = label_counts(conn)
    latest_model_version, latest_model_labels_used = _latest_model(conn)
    gate_clause, gate_params = gate_sql(triage_min_score, triage_min_p_lore)
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
        triaged_exchanges=_count_params(
            conn, "SELECT COUNT(*) FROM exchanges WHERE triage_version = ?", (TRIAGE_VERSION,)
        ),
        above_threshold_exchanges=_count_params(
            conn,
            "SELECT COUNT(*) FROM exchanges WHERE triage_version = ? AND triage_score >= ?",
            (TRIAGE_VERSION, triage_min_score),
        ),
        labels_by_source=counts.by_source,
        labels_effective=counts.effective,
        latest_model_version=latest_model_version,
        latest_model_labels_used=latest_model_labels_used,
        p_lore_scored=_count(conn, "SELECT COUNT(*) FROM exchanges WHERE p_lore IS NOT NULL"),
        passing_gate=_count_params(
            conn, f"SELECT COUNT(*) FROM exchanges WHERE {gate_clause}", gate_params
        ),
    )
