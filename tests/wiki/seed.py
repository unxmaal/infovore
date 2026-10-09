import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from infovore.db.claim_checks import CheckRow, record_checks
from infovore.db.claims_v2 import create_run, record_review
from tests.claims.seed import NOW, conversation, db


def wiki_db(tmp_path: Path, runs: int = 1) -> sqlite3.Connection:
    conn = db(tmp_path)
    for _ in range(runs):
        add_run(conn)
    return conn


def add_run(conn: sqlite3.Connection) -> int:
    return create_run(
        conn,
        endpoint="e",
        model_alias="m",
        model_id="m",
        model_id_source="alias",
        prompt_hash="p",
        selection="s",
        recipe={},
        now=NOW,
    )


def add_exchange(conn: sqlite3.Connection, base: int, day: str) -> int:
    exchange_id, _ = conversation(conn, [(1, "x", "hello")], base, slice_name=None)
    started = datetime.fromisoformat(day).replace(tzinfo=UTC).isoformat()
    conn.execute("UPDATE exchanges SET started_at = ? WHERE id = ?", (started, exchange_id))
    return exchange_id


def add_claim(
    conn: sqlite3.Connection,
    exchange_id: int,
    speaker: str,
    statement: str,
    *,
    check: str | None = "supported",
    review: str | None = None,
    run_id: int = 1,
) -> int:
    cursor = conn.execute(
        "INSERT INTO claims_v2 (run_id, exchange_id, speaker, statement) VALUES (?, ?, ?, ?)",
        (run_id, exchange_id, speaker, statement),
    )
    claim_id = int(cursor.lastrowid or 0)
    if check is not None:
        record_checks(conn, [CheckRow(claim_id, check, 1.0, [])], {}, NOW)
    if review is not None:
        record_review(conn, claim_id, review, NOW)
    return claim_id
