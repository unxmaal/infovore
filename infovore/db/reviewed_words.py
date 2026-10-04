import sqlite3
from collections.abc import Mapping
from datetime import datetime

from infovore.db.codec import to_db_time

_LATEST = (
    "SELECT r.word, r.tech FROM reviewed_words r"
    " JOIN (SELECT word, MAX(id) AS last FROM reviewed_words GROUP BY word) l ON l.last = r.id"
)


def record_decisions(conn: sqlite3.Connection, decisions: Mapping[str, bool], at: datetime) -> None:
    conn.executemany(
        "INSERT INTO reviewed_words (word, tech, decided_at) VALUES (?, ?, ?)",
        [(word, int(tech), to_db_time(at)) for word, tech in decisions.items()],
    )


def decided_words(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(row["word"] for row in conn.execute(_LATEST))


def approved_words(conn: sqlite3.Connection) -> frozenset[str]:
    return frozenset(row["word"] for row in conn.execute(_LATEST) if row["tech"])


def decision_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    approved = len(approved_words(conn))
    return approved, len(decided_words(conn)) - approved
