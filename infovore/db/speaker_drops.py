import sqlite3
from datetime import datetime
from typing import Final

from infovore.db.codec import to_db_time

DECISIONS: Final = ("drop", "keep")


def record_decision(conn: sqlite3.Connection, author_id: int, decision: str, now: datetime) -> None:
    conn.execute(
        "INSERT INTO speaker_drops (author_id, decision, decided_at) VALUES (?, ?, ?)",
        (author_id, decision, to_db_time(now)),
    )


def decisions(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("SELECT author_id, decision FROM current_speaker_drops")
    return {row[0]: row[1] for row in rows}


def dropped_authors(conn: sqlite3.Connection) -> frozenset[int]:
    return frozenset(a for a, decision in decisions(conn).items() if decision == "drop")
